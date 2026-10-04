#!/usr/bin/env bash
# Start the Agentic Condom's LLM (vLLM, OpenAI API) on the pod, detached, on 127.0.0.1 only: the
# browser reaches it through the lipread service's POST /correct, never directly.
#   CONDOM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ bash ml/runpod/condom.sh   (usually via CONDOM=1 serve.sh)
# Its own venv on /workspace (vLLM pins its own torch); weights cache on /workspace too, so a pod
# restart only pays the start-up (~1-2 min), not the install (~5 min) or the download.
# Several can run side by side on different CONDOM_PORTs (the eval pod does: condom_eval_pod.sh).
set -euo pipefail
WS="${WORKSPACE:-/workspace}"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$WS/.cache/hf"
VENV="$WS/.venv-vllm"
MODEL="${CONDOM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${CONDOM_PORT:-8001}"
HOST="${CONDOM_HOST:-127.0.0.1}"
# The VSR model + beam search need a few GB of the 24 GB; a 3B bf16 or 7B AWQ model fits in 0.45.
UTIL="${CONDOM_GPU_UTIL:-0.45}"
PIDFILE="$WS/condom-$PORT.pid"
LOG="${CONDOM_LOG:-$WS/condom-$PORT.log}"

[[ -d "$VENV" ]] || uv venv --python 3.11 "$VENV"
# 0.11.0 = torch 2.8 + CUDA 12.8 wheels, the same stack the lipread env already runs on this pod.
# transformers 5 dropped the tokenizer attribute vLLM 0.11 reads at start-up (all_special_tokens_extended).
# Re-checked every start (a no-op when satisfied), so an older venv gets fixed too.
VIRTUAL_ENV="$VENV" uv pip install -q "vllm==0.11.0" "transformers>=4.56,<5"

if curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"condom"'; then
  if [[ "${CONDOM_RESTART:-0}" != 1 ]]; then echo "condom LLM already up on :$PORT"; exit 0; fi
fi
if [[ -s "$PIDFILE" ]]; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
fi
# Short prompts (~600 tokens with the few-shot turns): a small context leaves the memory to the VSR.
setsid nohup "$VENV/bin/vllm" serve "$MODEL" --served-model-name condom --host "$HOST" --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len 2048 --max-num-seqs 8 --enable-prefix-caching \
  < /dev/null > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "vllm pid $! ($MODEL) — log: $LOG"
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "condom LLM up on :$PORT"; exit 0; fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
  sleep 2
done
echo "condom LLM did not come up; tail of log:" >&2
tail -30 "$LOG" >&2
exit 1
