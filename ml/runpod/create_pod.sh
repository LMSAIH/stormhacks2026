#!/usr/bin/env bash
# Create (or re-point) the tryheard.tech serving pod in one command. Runs on your machine or in a
# cloud session, not on a pod. Needs RUNPOD_API_KEY in the environment (never printed).
#
#   bash ml/runpod/create_pod.sh                 # new secure RTX 4090 pod that runs up.sh on every start
#   bash ml/runpod/create_pod.sh --update POD_ID # re-apply secrets + start command to an existing pod
#                                                # (RunPod restarts it; /workspace survives). Keeps
#                                                # the pod's other settings (BRANCH, PUBLIC_KEY, ...)
#                                                # unless you pass new values
#   DRY_RUN=1 bash ml/runpod/create_pod.sh       # print the request (secret references only)
#
# Secrets are referenced as {{ RUNPOD_SECRET_name }}, so their values never pass through here.
# Only secrets that exist in the RunPod account are wired in; add the rest (README-deploy.md), then
# run with --update. Knobs: NAME, BRANCH (the branch up.sh and bootstrap.sh check out), GPU_TYPES
# (comma list, tried in order), COUNTRIES (comma list, e.g. US,CA; empty = anywhere), VOLUME_GB.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
NAME="${NAME:-tryheard-prod}"
BRANCH_ARG="${BRANCH:-}"  # empty: master for a new pod, the pod's current BRANCH for --update
GPU_TYPES="${GPU_TYPES:-NVIDIA GeForce RTX 4090}"
COUNTRIES="${COUNTRIES-US,CA}"
VOLUME_GB="${VOLUME_GB:-50}"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # same image as every pod so far
REST=https://rest.runpod.io/v1
auth=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")

# env var in the pod  ←  RunPod secret name
declare -A SECRET_FOR=(
  [TUNNEL_TOKEN]=cf_tunnel_token
  [ELEVENLABS_API_KEY]=elevenlabs_api_key
  [GOOGLE_CLIENT_ID]=google_client_id
  [GOOGLE_CLIENT_SECRET]=google_client_secret
  [SESSION_SECRET]=session_secret
  [TIMESCALE_SERVICE_URL]=timescale_service_url
  [HF_TOKEN]=hf_token
  [JUPYTER_PASSWORD]=jupyter_password
)

have=$(curl -fsS "${auth[@]}" https://api.runpod.io/graphql \
  -d '{"query":"{ myself { secrets { name } } }"}' | jq -r '.data.myself.secrets[].name')

pod=""
current_env='{}'
if [[ "${1:-}" == --update ]]; then
  pod="${2:?usage: create_pod.sh --update POD_ID}"
  # The pod's current env, minus the secret-backed keys (re-added below as references, never as
  # values): PATCH replaces the whole env, so anything not carried over would be dropped.
  current_env=$(curl -fsS "${auth[@]}" "$REST/pods/$pod" | jq --argjson drop "$(printf '%s\n' "${!SECRET_FOR[@]}" | jq -R . | jq -s .)" \
    '(.env // {}) | with_entries(select(.key as $k | $drop | index($k) | not))')
fi
BRANCH=$(jq -r --arg arg "$BRANCH_ARG" 'if $arg != "" then $arg else (.BRANCH // "master") end' <<<"$current_env")
env_json=$(jq --arg branch "$BRANCH" '. + {BRANCH: $branch}' <<<"$current_env")
for var in "${!SECRET_FOR[@]}"; do
  s=${SECRET_FOR[$var]}
  if grep -qx "$s" <<<"$have"; then
    env_json=$(jq --arg k "$var" --arg v "{{ RUNPOD_SECRET_$s }}" '. + {($k): $v}' <<<"$env_json")
  else
    echo "note: RunPod secret '$s' does not exist yet → $var not set" >&2
  fi
done
# Optional plain settings passed through from the caller (not secrets).
for var in LIPREAD_MODEL LIPREAD_BEAM_SIZE LIPREAD_LM_WEIGHT LIPREAD_CTC_WEIGHT BACKEND_DIARIZATION CONDOM PUBLIC_KEY; do
  [[ -n "${!var:-}" ]] && env_json=$(jq --arg k "$var" --arg v "${!var}" '. + {($k): $v}' <<<"$env_json")
done

# On every container start: fetch up.sh for $BRANCH (5 tries; else the copy already on /workspace),
# run it in watch mode in the background, then hand over to the image's own start script (Jupyter
# on :8888, SSH). GitHub's raw CDN can serve a just-pushed file up to ~5 min late.
start_cmd='mkdir -p /workspace/logs; ( for i in 1 2 3 4 5; do curl -fsSL "https://raw.githubusercontent.com/LMSAIH/stormhacks2026/$BRANCH/ml/runpod/up.sh" -o /up.sh.part && mv /up.sh.part /up.sh && break; sleep $((i * 5)); done; [ -s /up.sh ] || cp /workspace/stormhacks2026/ml/runpod/up.sh /up.sh; bash /up.sh watch ) >> /workspace/logs/boot.log 2>&1 & exec /start.sh'

body=$(jq -n --argjson env "$env_json" --arg cmd "$start_cmd" '{env: $env, dockerStartCmd: ["bash", "-c", $cmd]}')

if [[ -n "$pod" ]]; then
  [[ "${DRY_RUN:-0}" == 1 ]] && { jq . <<<"$body"; exit 0; }
  curl -fsS "${auth[@]}" -X PATCH "$REST/pods/$pod" -d "$body" | jq '{id, name, desiredStatus, envKeys: (.env|keys)}'
  exit 0
fi

body=$(jq --arg name "$NAME" --arg image "$IMAGE" --arg gpus "$GPU_TYPES" --arg countries "$COUNTRIES" \
  --argjson vol "$VOLUME_GB" '. + {
    name: $name, imageName: $image, computeType: "GPU", cloudType: "SECURE", gpuCount: 1,
    gpuTypeIds: ($gpus | split(",") | map(ltrimstr(" "))), gpuTypePriority: "custom",
    volumeInGb: $vol, containerDiskInGb: 30, volumeMountPath: "/workspace",
    ports: ["8000/http", "8888/http", "22/tcp"]
  } + (if $countries == "" then {} else {countryCodes: ($countries | split(","))} end)' <<<"$body")
[[ "${DRY_RUN:-0}" == 1 ]] && { jq . <<<"$body"; exit 0; }

resp=$(curl -sS "${auth[@]}" -X POST "$REST/pods" -d "$body")
if ! jq -e '.id' <<<"$resp" >/dev/null 2>&1; then
  echo "create failed: $(jq -c . <<<"$resp" 2>/dev/null || echo "$resp")" >&2
  [[ -n "$COUNTRIES" ]] && echo "retry anywhere: COUNTRIES= bash $0" >&2
  exit 1
fi
jq '{id, name, desiredStatus, costPerHr, gpu: .machine.gpuTypeId, location: .machine.location, envKeys: (.env|keys)}' <<<"$resp"
id=$(jq -r .id <<<"$resp")
echo "pod $id: up.sh runs on boot; follow it with"
echo "  python ml/runpod/jupyter_exec.py --pod $id 'tail -f /workspace/logs/up.log'   (JUPYTER_TOKEN = the jupyter_password secret)"
