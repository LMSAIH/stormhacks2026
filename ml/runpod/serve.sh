#!/usr/bin/env bash
# (Re)start the lip-reader service on the pod, detached. Exposed via the RunPod HTTP proxy:
#   https://<POD_ID>-8000.proxy.runpod.net/health
set -euo pipefail
WS="${WORKSPACE:-/workspace}"
cd "$WS/stormhacks2026/ml"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" PATH="$HOME/.local/bin:$PATH"
export LIPREAD_WARM=1  # load model + detector at startup, not on the first request

pkill -f "uvicorn lipread.serve.app:app" 2>/dev/null || true
nohup uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
  > "$WS/serve.log" 2>&1 &
echo "started pid $! — log: $WS/serve.log"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT:-8000}/health" 2>/dev/null; then echo; exit 0; fi
  sleep 2
done
echo "service did not come up; tail of log:" >&2
tail -30 "$WS/serve.log" >&2
exit 1
