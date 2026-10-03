#!/usr/bin/env bash
# Baseline benchmark on the pod (brief §11): 100 LRS3-test clips (pre-made crops), greedy vs beam,
# through the running service (http://127.0.0.1:8000) and in-process. Run detached:
#   nohup bash ml/runpod/bench_baseline.sh > /workspace/bench.log 2>&1 &
# Sequential so the server and the in-process model never share the GPU.
set -uo pipefail
WS="${WORKSPACE:-/workspace}"
cd "$WS/stormhacks2026/ml"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" PATH="$HOME/.local/bin:$PATH"
P=data/lrs3_test/0000.parquet N="${N:-100}"
rm -f "$WS/bench.done"

uv run python scripts/bench.py --lrs3-parquet "$P" --n "$N" --backend http --url http://127.0.0.1:8000 \
  --decode greedy beam --tag pod-http --out artifacts/bench > "$WS/bench_pod_http.log" 2>&1
echo "http exit $?" >> "$WS/bench_pod_http.log"

uv run python scripts/bench.py --lrs3-parquet "$P" --n "$N" --backend local --decode greedy beam \
  --tag pod-local --out artifacts/bench > "$WS/bench_pod_local.log" 2>&1
echo "local exit $?" >> "$WS/bench_pod_local.log"

echo done > "$WS/bench.done"
