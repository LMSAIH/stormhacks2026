#!/usr/bin/env bash
# Self-starting Agentic Condom eval pod, for sessions that can reach neither SSH nor a pod's Jupyter
# password. Use it as the pod's start command (no volume needed, ~100 GB container disk):
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/LMSAIH/stormhacks2026/ml/agentic-condom/ml/runpod/condom_eval_pod.sh | bash; sleep infinity'
# Exposes (http): 8004 = these logs + latency_*.json, 8001-8003 = the candidate LLMs (vLLM OpenAI
# API, NO auth: stop the pod right after the eval), 8000 = the lipread service, /correct → 8001.
set -uo pipefail
export WORKSPACE=/workspace WS=/workspace
mkdir -p "$WS/logs"
(cd "$WS/logs" && setsid nohup python3 -m http.server 8004 > /dev/null 2>&1 < /dev/null &)
exec > "$WS/logs/boot.log" 2>&1
set -x
BRANCH="${BRANCH:-ml/agentic-condom}"
curl -fsSL "https://raw.githubusercontent.com/LMSAIH/stormhacks2026/$BRANCH/ml/runpod/bootstrap.sh" -o "$WS/bootstrap.sh"
BRANCH="$BRANCH" EXTRAS="export dev" bash "$WS/bootstrap.sh"
cd "$WS/stormhacks2026/ml"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python"
# One after another: each vLLM checks free memory when it starts. 7B AWQ ~5.5 GB, 3B bf16 ~6.3 GB.
CANDIDATES="${CANDIDATES:-8001=Qwen/Qwen2.5-7B-Instruct-AWQ=0.30 8002=Qwen/Qwen2.5-3B-Instruct=0.31 8003=unsloth/Llama-3.2-3B-Instruct=0.31}"
for c in $CANDIDATES; do
  IFS== read -r port model util <<< "$c"
  CONDOM_PORT=$port CONDOM_MODEL=$model CONDOM_GPU_UTIL=$util CONDOM_HOST=0.0.0.0 \
    CONDOM_LOG="$WS/logs/vllm_$port.log" bash runpod/condom.sh || echo "FAILED: $model on :$port"
done
nvidia-smi > "$WS/logs/nvidia-smi.txt" 2>&1
# The lipread service without the VSR warm-up (the GPU is the LLMs'): /correct uses the first one.
CORRECTOR_BASE_URL=http://127.0.0.1:8001/v1 CORRECTOR_MODEL=condom LIPREAD_WARM=0 \
  setsid nohup uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port 8000 \
  < /dev/null > "$WS/logs/serve.log" 2>&1 &
for c in $CANDIDATES; do
  IFS== read -r port model util <<< "$c"
  for mode in normal quality; do
    CORRECTOR_BASE_URL="http://127.0.0.1:$port/v1" CORRECTOR_MODEL=condom \
      uv run python scripts/condom_latency.py --mode $mode --out "$WS/logs/latency_${port}_$mode.json" || true
  done
done
echo "EVAL POD READY"
