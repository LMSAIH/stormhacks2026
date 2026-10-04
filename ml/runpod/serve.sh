#!/usr/bin/env bash
# (Re)start the lip-reader service on the pod, detached. Exposed via the RunPod HTTP proxy:
#   https://<POD_ID>-8000.proxy.runpod.net/health
set -euo pipefail
WS="${WORKSPACE:-/workspace}"
cd "$WS/stormhacks2026/ml"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" PATH="$HOME/.local/bin:$PATH"
export LIPREAD_WARM=1  # load model + detector at startup, not on the first request

# LIPREAD_MODEL=FT_v1_a0.5 serves a B2 fine-tune (stock when unset = the rollback). A checkpoint that
# isn't on this pod yet comes from the private checkpoint repo (needs HF_TOKEN) before the running
# server is touched, so a failed download leaves it serving.
M="${LIPREAD_MODEL:-LRS3_V_WER19.1}"
if [[ ! -s "checkpoints/$M/model.pth" ]]; then
  uv run python -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir="checkpoints", allow_patterns=[sys.argv[2] + "/*"])' \
    "${B2_CKPT_REPO:-eschmechel/stormhacks-b2-checkpoints}" "$M"
  [[ -s "checkpoints/$M/model.pth" ]] || { echo "checkpoints/$M/model.pth still missing" >&2; exit 1; }
fi

pkill -f "uvicorn lipread.serve.app:app" 2>/dev/null || true
# Wait for the old server to exit, or it can still answer the health check below.
for _ in $(seq 1 30); do pgrep -f "uvicorn lipread.serve.app:app" >/dev/null || break; sleep 1; done
# setsid + </dev/null: fully detach so the server outlives the SSH session that started it.
setsid nohup uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
  < /dev/null > "$WS/serve.log" 2>&1 &
echo "started pid $! — log: $WS/serve.log"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT:-8000}/health" 2>/dev/null; then echo; exit 0; fi
  sleep 2
done
echo "service did not come up; tail of log:" >&2
tail -30 "$WS/serve.log" >&2
exit 1
