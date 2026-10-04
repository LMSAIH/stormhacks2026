#!/usr/bin/env bash
# Self-starting Agentic Condom eval pod, for sessions that can reach neither SSH nor a pod's Jupyter
# password. Use it as the pod's start command (no volume needed, ~80 GB container disk), pinned to
# a commit so raw.githubusercontent.com's cache can't serve an older copy:
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/LMSAIH/stormhacks2026/<sha>/ml/runpod/condom_eval_pod.sh | bash; sleep infinity'
# Lean on purpose: only vLLM (its own venv, condom.sh) and the condom's pure-Python modules
# (lipread.agentic_condom needs httpx, which vLLM brings); no lipread env, checkpoints or data.
# Exposes (http): 8004 = these logs + latency_*.json; 8001-8003 = the candidate LLMs (vLLM OpenAI
# API, NO auth: stop the pod right after the eval). /correct for an app eval: run the lipread
# service elsewhere with CORRECTOR_BASE_URL=https://<pod>-8001.proxy.runpod.net/v1.
set -uo pipefail
export WORKSPACE=/workspace WS=/workspace
mkdir -p "$WS/logs"
(cd "$WS/logs" && setsid nohup python3 -m http.server 8004 > /dev/null 2>&1 < /dev/null &)
exec > "$WS/logs/boot.log" 2>&1
echo "== start $(date -u +%T)"
BRANCH="${BRANCH:-ml/agentic-condom}"
git clone -q --depth 1 --branch "$BRANCH" https://github.com/LMSAIH/stormhacks2026.git "$WS/stormhacks2026"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$WS/stormhacks2026/ml"
# One after another: each vLLM checks free memory when it starts. 7B AWQ ~5.5 GB, 3B bf16 ~6.3 GB.
CANDIDATES="${CANDIDATES:-8001=Qwen/Qwen2.5-7B-Instruct-AWQ=0.30 8002=Qwen/Qwen2.5-3B-Instruct=0.31 8003=unsloth/Llama-3.2-3B-Instruct=0.31}"
for c in $CANDIDATES; do
  IFS== read -r port model util <<< "$c"
  echo "== $model on :$port $(date -u +%T)"
  CONDOM_PORT=$port CONDOM_MODEL=$model CONDOM_GPU_UTIL=$util CONDOM_HOST=0.0.0.0 \
    CONDOM_LOG="$WS/logs/vllm_$port.log" bash runpod/condom.sh || echo "FAILED: $model on :$port"
done
nvidia-smi > "$WS/logs/nvidia-smi.txt" 2>&1
for c in $CANDIDATES; do
  IFS== read -r port model util <<< "$c"
  for mode in normal quality; do
    echo "== latency :$port $mode $(date -u +%T)"
    PYTHONPATH=src CORRECTOR_BASE_URL="http://127.0.0.1:$port/v1" CORRECTOR_MODEL=condom \
      "$WS/.venv-vllm/bin/python" scripts/condom_latency.py --mode $mode \
      --out "$WS/logs/latency_${port}_$mode.json" || echo "FAILED: latency :$port $mode"
  done
done
echo "EVAL POD READY $(date -u +%T)"
