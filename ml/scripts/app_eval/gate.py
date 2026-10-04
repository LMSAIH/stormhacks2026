"""App-eval regression gate (`./smoke.sh app`): Normal and Instant through the real /app.

Starts its own `pnpm dev` (no sign-in, model served from frontend/public/models, browser phrase
store), plays eval20.y4m in headless Chromium once per mode (e2e_eval.mjs) and fails when a mode's
words-wrong rate is more than MARGIN points above the gate line in .context/app-eval.md. A run
over the line is repeated once and the mean of the two decides (runs vary ~2-3 points).

  uv run --directory ml python scripts/app_eval/gate.py     # ~6 min (~11 with a re-run)

Needs: frontend/public/models/ (the pinned int8 model), data/raw_eval (private HF dataset),
playwright-core (PLAYWRIGHT_CORE=<path>/index.mjs, or found in /opt/node-tools or `pnpm root -g`)
and Chromium (CHROME=<binary>, or /opt/pw-browsers, or playwright's own download).
Exit: 0 pass, 1 a mode is over its line, 2 bad setup.
"""

from __future__ import annotations

import glob
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from score_eval import score

HERE = Path(__file__).resolve().parent
ML = HERE.parents[1]
ROOT = ML.parent
FRONTEND = ROOT / "frontend"
OUT = ML / "artifacts/app_eval"
RECORD = ROOT / ".context/app-eval.md"
MODES = ("normal", "instant")
MARGIN = 5.0  # points over the recorded rate that fail the gate
# The line in app-eval.md this gate reads, e.g. "Gate (`./smoke.sh app`): Normal 25.4%, Instant 27.9%"
GATE_LINE = re.compile(r"^Gate \(`\./smoke\.sh app`\):.*?Normal (\d+(?:\.\d+)?)%.*?Instant (\d+(?:\.\d+)?)%", re.M)


def fail_setup(msg: str) -> None:
    print(f"app gate: setup: {msg}", file=sys.stderr)
    sys.exit(2)


def find_playwright() -> str:
    candidates = [os.environ.get("PLAYWRIGHT_CORE", ""), "/opt/node-tools/node_modules/playwright-core/index.mjs"]
    try:
        root = subprocess.run(["pnpm", "root", "-g"], capture_output=True, text=True, timeout=30).stdout.strip()
        candidates.append(f"{root}/playwright-core/index.mjs")
    except (OSError, subprocess.TimeoutExpired):
        pass
    for c in candidates:
        if c and Path(c).is_file():
            return c
    fail_setup("playwright-core not found: set PLAYWRIGHT_CORE=<path>/playwright-core/index.mjs "
               "(or `pnpm add -g playwright-core && pnpm exec playwright-core install chromium`)")
    return ""


def find_chrome() -> str | None:
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return found[-1] if found else None  # None: playwright's own download


def recorded() -> dict[str, float]:
    m = GATE_LINE.search(RECORD.read_text())
    if not m:
        fail_setup(f"no gate line in {RECORD} (\"Gate (`./smoke.sh app`): Normal N%, Instant N%\")")
    return {"normal": float(m.group(1)), "instant": float(m.group(2))}


def check_inputs() -> None:
    model = FRONTEND / "public/models"
    if not (model / "lipread_ctc.int8.onnx").is_file() or not (model / "tokens.json").is_file():
        fail_setup("no local model in frontend/public/models: hf download eschmechel/auto-avsr-lrs3-vsr-int8-onnx "
                   "lipread_ctc.int8.onnx tokens.json --revision <commit in modelSpec.ts> --local-dir frontend/public/models")
    if not (OUT / "eval20.y4m").is_file() or not (OUT / "eval20_refs.json").is_file():
        if not (ML / "data/raw_eval").is_dir():
            fail_setup("no data/raw_eval: hf download eschmechel/stormhacks-lipread-eval --repo-type dataset "
                       "--local-dir ml/data (private; needs HF_TOKEN)")
        print("── building eval20.y4m (once)", flush=True)
        subprocess.run([sys.executable, str(HERE / "make_eval_video.py")], check=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_dev_server(port: int) -> subprocess.Popen:
    # Env beats .env.local in Vite: pin what the eval depends on whatever the developer has set.
    env = {**os.environ, "VITE_SKIP_AUTH": "1", "VITE_LIPREAD_MODEL_BASE": "/models",
           "VITE_PHRASES_URL": "", "VITE_LIPREAD_URL": "", "VITE_ORT_WEBGPU": ""}  # no live shared bank
    proc = subprocess.Popen(["pnpm", "dev", "--port", str(port), "--strictPort"], cwd=FRONTEND, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://localhost:{port}/models/tokens.json"
    for _ in range(120):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return proc
        except OSError:
            time.sleep(0.5)
        if proc.poll() is not None:
            break
    stop(proc)
    fail_setup(f"pnpm dev did not come up on port {port}")
    return proc


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        os.killpg(proc.pid, signal.SIGKILL)


def run(mode: str, tag: str, base: str, playwright: str, chrome: str | None) -> float:
    env = {**os.environ, "MODE": mode, "TAG": tag, "BASE": base, "PLAYWRIGHT_CORE": playwright}
    if chrome:
        env["CHROME"] = chrome
    subprocess.run(["node", str(HERE / "e2e_eval.mjs")], env=env, check=True)
    s = score(tag)
    if s is None:
        raise RuntimeError(f"no result for {tag}")
    print(f"   {tag}: WER {s['wer']:.1%} ({s['wrong']} wrong, {s['missed']} missed, {s['extra']} extra; "
          f"{s['lines']} lines, {s['locks']} cuts, {s['drops']} drops; tracker {s['tracker']})", flush=True)
    return s["wer"] * 100


def main() -> int:
    lines = recorded()
    check_inputs()
    playwright, chrome = find_playwright(), find_chrome()
    port = free_port()
    proc = start_dev_server(port)
    results: dict[str, list[float]] = {}
    try:
        for mode in MODES:
            limit = lines[mode] + MARGIN
            print(f"── {mode}: recorded {lines[mode]:.1f}%, fails above {limit:.1f}%", flush=True)
            wers = [run(mode, f"gate_{mode}_1", f"http://localhost:{port}", playwright, chrome)]
            if wers[0] > limit:
                print(f"   over the line: running {mode} once more (the mean decides)", flush=True)
                wers.append(run(mode, f"gate_{mode}_2", f"http://localhost:{port}", playwright, chrome))
            results[mode] = wers
    finally:
        stop(proc)
    bad = []
    for mode, wers in results.items():
        mean = sum(wers) / len(wers)
        ok = mean <= lines[mode] + MARGIN
        print(f"app gate: {mode} {mean:.1f}% (recorded {lines[mode]:.1f}%, limit {lines[mode] + MARGIN:.1f}%) "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(mode)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
