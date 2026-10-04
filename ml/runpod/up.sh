#!/usr/bin/env bash
# Bring the whole tryheard.tech stack up on one RunPod GPU pod. Idempotent: anything already
# healthy is left alone, so it is safe to re-run at any time.
#   1. bootstrap.sh once per container start (repo, uv env, checkpoints on /workspace)
#   2. our ML server (serve.sh, uvicorn :8000)          → https://ml.tryheard.tech
#   3. the backend team's server, unchanged (python main.py: REST :5000, TTS WebSocket :8765)
#                                                       → https://api.tryheard.tech (+ /ws/tts)
#   4. cloudflared, the named tunnel `tryheard`         → routes both hostnames to this pod
#
# Every secret comes from the pod environment, which create_pod.sh fills from RunPod secrets
# ({{ RUNPOD_SECRET_name }}); nothing secret is written to disk or printed.
#
#   bash /workspace/stormhacks2026/ml/runpod/up.sh            # bring everything up, then exit
#   bash /workspace/stormhacks2026/ml/runpod/up.sh watch      # same, then re-check every 30 s
#   RESTART=1 bash .../up.sh                                  # restart all three even if healthy
#
# Logs: /workspace/logs/{up,backend,cloudflared}.log and /workspace/serve.log (ML server).
# Runbook: ml/runpod/README-deploy.md.
set -uo pipefail

WS="${WORKSPACE:-/workspace}"
export BRANCH="${BRANCH:-master}"
DIR="$WS/stormhacks2026"
LOGS="$WS/logs"
mkdir -p "$LOGS" "$WS/bin"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" UV_LINK_MODE=copy
export PATH="$WS/bin:$HOME/.local/bin:$PATH"

ML_PORT="${PORT:-8000}"
API_PORT=5000
TTS_PORT=8765
BE_VENV="$WS/.venv-backend"
CF_METRICS=127.0.0.1:20241
# Secrets only the backend needs; kept out of the ML server's environment.
BACKEND_SECRETS=(ELEVENLABS_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET SESSION_SECRET TIMESCALE_SERVICE_URL)

log() { printf '%s up.sh: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

bootstrap() {
  [[ -f /tmp/tryheard-bootstrapped ]] && return 0  # /tmp is on the container disk: once per start
  log "bootstrap ($BRANCH)"
  if [[ -f "$DIR/ml/runpod/bootstrap.sh" ]]; then
    bash "$DIR/ml/runpod/bootstrap.sh" || return $?
  else
    curl -fsSL "https://raw.githubusercontent.com/LMSAIH/stormhacks2026/$BRANCH/ml/runpod/bootstrap.sh" | bash \
      || return $?
  fi
  touch /tmp/tryheard-bootstrapped
}

ml_ok() {  # healthy AND serving the model we asked for
  curl -fsS -m 5 "http://127.0.0.1:$ML_PORT/health" 2>/dev/null \
    | grep -q "\"model\":\"${LIPREAD_MODEL:-LRS3_V_WER19.1}\""
}

ensure_ml() {
  if [[ "${RESTART:-0}" != 1 ]] && ml_ok; then return 0; fi
  log "starting ML server (serve.sh)"
  # serve.sh owns the ML side (and CONDOM=1's corrector when that lands); it inherits the pod env
  # minus the backend's secrets.
  local unset_args=()
  for v in "${BACKEND_SECRETS[@]}" TUNNEL_TOKEN; do unset_args+=(-u "$v"); done
  env "${unset_args[@]}" bash "$DIR/ml/runpod/serve.sh"
}

backend_install() {
  # Mirrors backend/Dockerfile: python 3.12, requirements.txt, diarization extras only on request.
  # RunPod pods are containers without Docker (no socket, no CAP_SYS_ADMIN), so the same steps
  # run in a venv on /workspace instead of an image.
  [[ -x "$BE_VENV/bin/python" ]] || uv venv -q --python 3.12 "$BE_VENV" || return 1
  uv pip install -q --python "$BE_VENV/bin/python" -r "$DIR/backend/requirements.txt" || return 1
  if [[ "${BACKEND_DIARIZATION:-0}" == 1 ]]; then
    uv pip install -q --python "$BE_VENV/bin/python" -r "$DIR/backend/requirements-diarization.txt" \
      && uv pip install -q --python "$BE_VENV/bin/python" --no-deps resemblyzer==0.1.4 || return 1
  fi
}

backend_ok() {
  curl -fsS -m 5 -o /dev/null "http://127.0.0.1:$API_PORT/openapi.json" 2>/dev/null \
    && (exec 3<>"/dev/tcp/127.0.0.1/$TTS_PORT") 2>/dev/null
}

ensure_backend() {
  if [[ "${RESTART:-0}" != 1 ]] && backend_ok; then return 0; fi
  local missing=()
  for v in "${BACKEND_SECRETS[@]}"; do [[ -n "${!v:-}" ]] || missing+=("$v"); done
  ((${#missing[@]})) && log "WARNING backend secrets not in the pod env: ${missing[*]} (add the RunPod secrets, see README-deploy.md)"
  backend_install || { log "backend install failed"; return 1; }
  pkill -f "$BE_VENV/bin/python main.py" 2>/dev/null || true
  for _ in $(seq 1 15); do pgrep -f "$BE_VENV/bin/python main.py" >/dev/null || break; sleep 1; done
  log "starting backend (REST :$API_PORT, TTS WebSocket :$TTS_PORT)"
  # Production settings (overridable from the pod env). SESSION_HTTPS_ONLY must be the word
  # "true": backend/config.py compares against it, so "1" would leave the cookie non-Secure.
  # CUDA_VISIBLE_DEVICES='' keeps the backend (and diarization's torch) off the GPU.
  # uvicorn trusts X-Forwarded-Proto from 127.0.0.1 (cloudflared), so OAuth callbacks are https.
  (
    cd "$DIR/backend" || exit 1
    export FRONTEND_URL="${FRONTEND_URL:-https://tryheard.tech}"
    export FRONTEND_ORIGINS="${FRONTEND_ORIGINS:-https://tryheard.tech}"
    export SESSION_HTTPS_ONLY="${SESSION_HTTPS_ONLY:-true}"
    export DIARIZATION="${BACKEND_DIARIZATION:-0}"
    export CUDA_VISIBLE_DEVICES='' NUMBA_CACHE_DIR=/tmp XDG_CACHE_HOME=/tmp
    exec env -u TUNNEL_TOKEN -u HF_TOKEN setsid nohup "$BE_VENV/bin/python" main.py \
      < /dev/null >> "$LOGS/backend.log" 2>&1
  ) &
  for _ in $(seq 1 60); do backend_ok && { log "backend up"; return 0; }; sleep 1; done
  log "backend did not come up; tail of $LOGS/backend.log:"; tail -20 "$LOGS/backend.log" >&2
  return 1
}

tunnel_ok() { curl -fsS -m 3 -o /dev/null "http://$CF_METRICS/ready" 2>/dev/null; }

ensure_tunnel() {
  if [[ -z "${TUNNEL_TOKEN:-}" ]]; then
    log "WARNING no TUNNEL_TOKEN in the pod env (RunPod secret cf_tunnel_token): the *.tryheard.tech hostnames won't reach this pod"
    return 1
  fi
  if [[ "${RESTART:-0}" != 1 ]] && tunnel_ok; then return 0; fi
  if [[ ! -x "$WS/bin/cloudflared" ]]; then
    log "installing cloudflared"
    curl -fsSL -o "$WS/bin/cloudflared.part" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      && chmod +x "$WS/bin/cloudflared.part" && mv "$WS/bin/cloudflared.part" "$WS/bin/cloudflared" || return 1
  fi
  pkill -f "cloudflared tunnel" 2>/dev/null || true
  sleep 1
  log "starting cloudflared ($("$WS/bin/cloudflared" --version 2>/dev/null | head -1))"
  # The token stays in the environment (cloudflared reads TUNNEL_TOKEN), never on the command line.
  # http2 over TCP 7844: QUIC (UDP) is not reliable from inside pod containers.
  setsid nohup "$WS/bin/cloudflared" tunnel --no-autoupdate --protocol http2 --metrics "$CF_METRICS" run \
    < /dev/null >> "$LOGS/cloudflared.log" 2>&1 &
  for _ in $(seq 1 30); do tunnel_ok && { log "tunnel connected"; return 0; }; sleep 1; done
  log "tunnel not ready; tail of $LOGS/cloudflared.log:"; tail -20 "$LOGS/cloudflared.log" >&2
  return 1
}

status() {
  local ml be cf
  ml_ok && ml=ok || ml=DOWN
  backend_ok && be=ok || be=DOWN
  tunnel_ok && cf=ok || cf=DOWN
  log "status: ml=$ml backend=$be tunnel=$cf"
  [[ $ml == ok && $be == ok && $cf == ok ]]
}

up_once() {
  ensure_ml || log "ML server failed (see $WS/serve.log)"
  ensure_backend || true
  ensure_tunnel || true
  status
}

{
  bootstrap || { rc=$?; log "bootstrap failed (exit $rc; 3 = bad CUDA host: recreate the pod)"; exit "$rc"; }
  up_once
  rc=$?
  if [[ "${1:-}" == watch ]]; then
    log "watching every 30 s"
    RESTART=0  # from here on, only restart what is unhealthy
    while sleep 30; do
      ml_ok && backend_ok && tunnel_ok && continue
      up_once
    done
  fi
  exit "$rc"
} 2>&1 | tee -a "$LOGS/up.log"
exit "${PIPESTATUS[0]}"
