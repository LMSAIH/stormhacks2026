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
# ({{ RUNPOD_SECRET_name }}); nothing secret is written to disk or printed. A shell that didn't
# inherit the pod env (an SSH session) reads it from the container's PID 1.
#
#   bash /workspace/stormhacks2026/ml/runpod/up.sh            # bring everything up, then exit
#   bash /workspace/stormhacks2026/ml/runpod/up.sh watch      # same, then re-check every 30 s
#   RESTART=1 bash .../up.sh      # pull BRANCH, re-bootstrap, restart all three even if healthy
#   RESTART=backend bash .../up.sh   # same, but restart only the named parts (ml, backend, tunnel)
#   TUNNEL_REQUIRE_HEALTHY=1      # (pod env) join the tunnel only while ML + backend are healthy,
#                                 # and leave it while they're repaired: for a standby replica
#
# Runs one at a time (flock): a manual run waits for the watcher's current pass and vice versa.
# Logs: /workspace/logs/{up,backend,cloudflared}.log and /workspace/serve.log (ML server; the
# previous run's is kept as serve.log.1). Runbook: ml/runpod/README-deploy.md.
set -uo pipefail

# Settings from the pod env that this shell may lack (SSH sessions don't inherit it).
if [[ -r /proc/1/environ ]]; then
  while IFS= read -r -d '' kv; do
    k=${kv%%=*}
    case "$k" in
      TUNNEL_TOKEN | ELEVENLABS_API_KEY | GOOGLE_CLIENT_ID | GOOGLE_CLIENT_SECRET | SESSION_SECRET | \
        TIMESCALE_SERVICE_URL | HF_TOKEN | BRANCH | BACKEND_DIARIZATION | BACKEND_ALLOW_MISSING | CONDOM | \
        TUNNEL_REQUIRE_HEALTHY | \
        FRONTEND_URL | FRONTEND_ORIGINS | SESSION_HTTPS_ONLY | LIPREAD_* | CORRECTOR_*)
        [[ -n "${!k:-}" ]] || export "${kv?}" ;;
    esac
  done < /proc/1/environ
fi

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
MARKER=/tmp/tryheard-bootstrapped  # /tmp is on the container disk: bootstrap once per start
# Secrets only the backend needs; kept out of the ML server's environment.
BACKEND_SECRETS=(ELEVENLABS_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET SESSION_SECRET TIMESCALE_SERVICE_URL)

log() { printf '%s up.sh: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# RESTART=1 or all: every part; otherwise a comma list of ml, backend, tunnel.
restart_wanted() { case ",${RESTART:-0}," in *,1,* | *,all,* | *",$1,"*) return 0 ;; esac; return 1; }

bootstrap() {
  [[ -f "$MARKER" ]] && return 0
  log "bootstrap ($BRANCH)"
  local rc=0
  if [[ -f "$DIR/ml/runpod/bootstrap.sh" ]]; then
    bash "$DIR/ml/runpod/bootstrap.sh" 9>&- || rc=$?
  else
    curl -fsSL "https://raw.githubusercontent.com/LMSAIH/stormhacks2026/$BRANCH/ml/runpod/bootstrap.sh" | bash 9>&- || rc=$?
  fi
  ((rc == 0)) && touch "$MARKER"
  return "$rc"
}

ml_ok() {  # healthy AND serving the model we asked for
  curl -fsS -m 5 "http://127.0.0.1:$ML_PORT/health" 2>/dev/null \
    | grep -q "\"model\": *\"${LIPREAD_MODEL:-LRS3_V_WER19.1}\""
}

ensure_ml() {
  if ! restart_wanted ml && ml_ok; then return 0; fi
  log "starting ML server (serve.sh)"
  # serve.sh truncates its log: keep the previous run's (a crash's traceback) as serve.log.1.
  [[ -s "$WS/serve.log" ]] && mv -f "$WS/serve.log" "$WS/serve.log.1"
  # serve.sh owns the ML side (and CONDOM=1's corrector when that lands); it inherits the pod env
  # minus the backend's secrets and the tunnel token.
  local unset_args=()
  for v in "${BACKEND_SECRETS[@]}" TUNNEL_TOKEN; do unset_args+=(-u "$v"); done
  env "${unset_args[@]}" bash "$DIR/ml/runpod/serve.sh" 9>&-
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
  if ! restart_wanted backend && backend_ok; then return 0; fi
  local missing=()
  for v in "${BACKEND_SECRETS[@]}"; do [[ -n "${!v:-}" ]] || missing+=("$v"); done
  if ((${#missing[@]})) && [[ "${BACKEND_ALLOW_MISSING:-0}" != 1 ]]; then
    # A backend without them would pass backend_ok yet fail sign-in, the DB and TTS, and a new
    # SESSION_SECRET signs everyone out; leave whatever runs alone instead.
    log "NOT (re)starting the backend: missing ${missing[*]} (RunPod secrets, README-deploy.md; BACKEND_ALLOW_MISSING=1 to override)"
    return 1
  fi
  backend_install || { log "backend install failed"; return 1; }
  pkill -f "$BE_VENV/bin/python main.py" 2>/dev/null || true
  for _ in $(seq 1 15); do pgrep -f "$BE_VENV/bin/python main.py" >/dev/null || break; sleep 1; done
  pkill -9 -f "$BE_VENV/bin/python main.py" 2>/dev/null || true
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
  ) 9>&- &
  for _ in $(seq 1 60); do backend_ok && { log "backend up"; return 0; }; sleep 1; done
  log "backend did not come up; tail of $LOGS/backend.log:"; tail -20 "$LOGS/backend.log" >&2
  return 1
}

tunnel_ok() { curl -fsS -m 3 -o /dev/null "http://$CF_METRICS/ready" 2>/dev/null; }

tunnel_stop() {
  # A stopping cloudflared drains open streams for up to 30 s and keeps the metrics port until it
  # exits, so wait for it (then kill it).
  pkill -f "cloudflared tunnel" 2>/dev/null || return 0
  for _ in $(seq 1 35); do pgrep -f "cloudflared tunnel" >/dev/null || return 0; sleep 1; done
  pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
}

# With TUNNEL_REQUIRE_HEALTHY=1 a pod is only in the tunnel while it can serve: Cloudflare doesn't
# health-check what is behind a connector, so a replica with a dead ML server would still get
# requests.
tunnel_gate() {
  [[ "${TUNNEL_REQUIRE_HEALTHY:-0}" == 1 ]] || return 0
  ml_ok && backend_ok && return 0
  if pgrep -f "cloudflared tunnel" >/dev/null; then
    log "leaving the tunnel until the ML server and backend are healthy (TUNNEL_REQUIRE_HEALTHY=1)"
    tunnel_stop
  fi
  return 1
}

ensure_tunnel() {
  if [[ -z "${TUNNEL_TOKEN:-}" ]]; then
    log "WARNING no TUNNEL_TOKEN in the pod env (RunPod secret cf_tunnel_token): the *.tryheard.tech hostnames won't reach this pod"
    return 1
  fi
  tunnel_gate || return 1
  if ! restart_wanted tunnel && tunnel_ok; then return 0; fi
  if [[ ! -x "$WS/bin/cloudflared" ]]; then
    log "installing cloudflared"
    curl -fsSL -o "$WS/bin/cloudflared.part" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      && chmod +x "$WS/bin/cloudflared.part" && mv "$WS/bin/cloudflared.part" "$WS/bin/cloudflared" || return 1
  fi
  tunnel_stop  # frees the metrics port before the new one binds it
  log "starting cloudflared ($("$WS/bin/cloudflared" --version 2>/dev/null | head -1))"
  # The token stays in the environment (cloudflared reads TUNNEL_TOKEN), never on the command line.
  # http2 over TCP 7844: QUIC (UDP) is not reliable from inside pod containers.
  setsid nohup "$WS/bin/cloudflared" tunnel --no-autoupdate --protocol http2 --metrics "$CF_METRICS" run \
    < /dev/null >> "$LOGS/cloudflared.log" 2>&1 9>&- &
  for _ in $(seq 1 30); do tunnel_ok && { log "tunnel connected"; return 0; }; sleep 1; done
  log "tunnel not ready; tail of $LOGS/cloudflared.log:"; tail -20 "$LOGS/cloudflared.log" >&2
  return 1
}

all_ok() { ml_ok && backend_ok && tunnel_ok; }

status() {
  local ml be cf
  ml_ok && ml=ok || ml=DOWN
  backend_ok && be=ok || be=DOWN
  tunnel_ok && cf=ok || cf=DOWN
  log "status: ml=$ml backend=$be tunnel=$cf"
  [[ $ml == ok && $be == ok && $cf == ok ]]
}

# One pass: bootstrap if this container hasn't yet, then make each part healthy. Returns 3 on a bad
# CUDA host (recreate the pod). A failed bootstrap still starts whatever /workspace already has;
# the watcher retries the bootstrap on its next pass.
up_once() {
  local rc=0
  bootstrap || rc=$?
  if ((rc == 3)); then log "bootstrap: bad CUDA host (exit 3): recreate the pod"; return 3; fi
  if ((rc != 0)); then
    if [[ -x "$DIR/ml/.venv/bin/python" && -f "$DIR/backend/main.py" ]]; then
      log "bootstrap failed (exit $rc); starting from the existing checkout on /workspace"
    else
      log "bootstrap failed (exit $rc) and /workspace has nothing to start from"
      return "$rc"
    fi
  fi
  tunnel_gate || true  # a standby leaves the tunnel before anything is repaired
  ensure_ml || log "ML server failed (see $WS/serve.log)"
  ensure_backend || true
  ensure_tunnel || true
  status
}

{
  exec 9> /tmp/tryheard-up.lock
  flock -w 900 9 || log "another up.sh has held the lock for 15 min; going ahead"
  [[ "${RESTART:-0}" != 0 ]] && rm -f "$MARKER"  # a manual restart also pulls BRANCH
  up_once
  rc=$?
  flock -u 9
  if [[ "${1:-}" == watch ]]; then
    ((rc == 3)) && exit 3
    log "watching every 30 s"
    export RESTART=0  # from here on, only restart what is unhealthy
    while sleep 30; do
      [[ -f "$MARKER" ]] && all_ok && continue
      sleep 10  # one slow /health (a long beam read) is not an outage: look again first
      [[ -f "$MARKER" ]] && all_ok && continue
      flock -w 900 9 || log "another up.sh has held the lock for 15 min; going ahead"
      up_once
      flock -u 9
    done
  fi
  exit "$rc"
} 2>&1 | tee -p -a "$LOGS/up.log"  # -p: a closed stdout (a finished jupyter_exec call) must not SIGPIPE the loop
exit "${PIPESTATUS[0]}"
