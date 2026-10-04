"""Run shell commands on a RunPod pod through its Jupyter server — for sessions that can reach
HTTPS but not SSH (cloud agents behind an egress proxy).

The runpod/pytorch image starts Jupyter Lab on :8888 when the pod env has JUPYTER_PASSWORD; expose
8888/http and the pod is reachable at https://<pod-id>-8888.proxy.runpod.net. The token is read
from the env var named by --token-env (never pass it on the command line).

    export JUPYTER_TOKEN=...            # same value as the pod's JUPYTER_PASSWORD
    python ml/runpod/jupyter_exec.py --pod jr602gal8ql6c0 'nvidia-smi; df -h /workspace'
    python ml/runpod/jupyter_exec.py --pod jr602gal8ql6c0 --timeout 3600 -f job.sh

Long jobs: start them detached (`setsid bash job.sh > /workspace/b2/x.log 2>&1 < /dev/null &`)
and poll the log; the websocket only lives as long as this call. Needs `websocket-client`, `requests`.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import uuid
from urllib.parse import urlparse

import requests
import websocket

RUNNER = r'''
import os, subprocess, sys
# the kernel's env leaks into commands; MPLBACKEND=module://matplotlib_inline breaks mediapipe
env = {k: v for k, v in os.environ.items()
       if k not in ("MPLBACKEND", "JPY_PARENT_PID", "PYDEVD_USE_FRAME_EVAL") and not k.startswith("JPY_")}
p = subprocess.run(["bash", "-lc", CMD], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TIMEOUT)
sys.stdout.write(p.stdout.decode(errors="replace"))
print(f"\n[exit {p.returncode}]")
'''


def _proxy_opts() -> dict:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return {}
    u = urlparse(proxy)
    opts = {"http_proxy_host": u.hostname, "http_proxy_port": u.port or 80, "proxy_type": "http"}
    if u.username:
        opts["http_proxy_auth"] = (u.username, u.password or "")
    return opts


def _sslopt() -> dict:
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    return {"cert_reqs": ssl.CERT_REQUIRED, **({"ca_certs": ca} if ca else {})}


def run(base: str, token: str, cmd: str, timeout: int) -> int:
    hdr = {"Authorization": f"token {token}"}
    r = requests.post(f"{base}/api/kernels", headers=hdr, json={"name": "python3"}, timeout=60)
    r.raise_for_status()
    kid = r.json()["id"]
    try:
        ws_url = base.replace("https://", "wss://") + f"/api/kernels/{kid}/channels"
        ws = websocket.create_connection(ws_url, header=[f"Authorization: token {token}"],
                                         timeout=timeout + 60, sslopt=_sslopt(), **_proxy_opts())
        code = RUNNER.replace("CMD", json.dumps(cmd)).replace("TIMEOUT", str(timeout))
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "agent", "session": uuid.uuid4().hex,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {}, "channel": "shell",
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        }))
        rc, deadline = 1, time.time() + timeout + 60
        while time.time() < deadline:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t, c = m["msg_type"], m["content"]
            if t == "stream":
                sys.stdout.write(c["text"])
                sys.stdout.flush()
                if "[exit " in c["text"]:
                    rc = int(c["text"].rsplit("[exit ", 1)[1].split("]")[0])
            elif t == "error":
                sys.stdout.write("\n".join(c.get("traceback", [])) + "\n")
            elif t == "status" and c.get("execution_state") == "idle":
                break
        ws.close()
        return rc
    finally:
        requests.delete(f"{base}/api/kernels/{kid}", headers=hdr, timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod", required=True)
    ap.add_argument("--token-env", default="JUPYTER_TOKEN")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("-f", "--file", help="run this local bash script instead of CMD")
    ap.add_argument("cmd", nargs="?")
    a = ap.parse_args()
    token = os.environ.get(a.token_env)
    if not token:
        raise SystemExit(f"${a.token_env} is not set")
    cmd = open(a.file).read() if a.file else a.cmd
    if not cmd:
        raise SystemExit("give CMD or -f script")
    sys.exit(run(f"https://{a.pod}-8888.proxy.runpod.net", token, cmd, a.timeout))


if __name__ == "__main__":
    main()
