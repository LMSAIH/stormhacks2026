# tryheard.tech: production runbook

| URL | What | Runs on |
|---|---|---|
| https://tryheard.tech | The app (static Vite build) | Cloudflare Workers static assets (Worker `stormhacks2026`), from `frontend/` |
| https://ml.tryheard.tech | Our ML server (`lipread.serve.app`, Quality reads) | RunPod GPU pod, uvicorn `localhost:8000` |
| https://api.tryheard.tech | The backend team's server: REST, sign-in, voices, chats, `/ws/stt` | Same pod, `localhost:5000` |
| wss://api.tryheard.tech/ws/tts | The backend's TTS WebSocket | Same pod, `localhost:8765` |

The two API hostnames reach the pod through a Cloudflare named tunnel (`tryheard`). `cloudflared`
runs on the pod and connects outward to Cloudflare, so `localhost` in the tunnel's routes means the
pod itself. DNS points at the tunnel, not the pod, so the URLs stay the same when the pod is
replaced: a new pod that runs `cloudflared` with the same token takes over.

The TTS socket is a path on `api.` rather than its own hostname because the sign-in cookie is
host-only on `api.tryheard.tech`; a `ws.` hostname would never receive it and every TTS connection
would close with 4401.

## Recreate the pod (one command)

From a machine with `RUNPOD_API_KEY` set:

```
BRANCH=master bash ml/runpod/create_pod.sh
```

This creates a secure-cloud RTX 4090 (US or Canada first: `COUNTRIES=` for anywhere,
`GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA L40S"` to allow a fallback type). Its start command
fetches `ml/runpod/up.sh` from `BRANCH` and runs it on every container start, so a restarted pod
also comes back by itself. Measured on 2026-10-04: created 15:56:32 UTC, bootstrap done 15:59:05,
ML server and backend healthy 15:59:27, about 3 minutes. Then:

```
scripts/check_live.sh        # from anywhere; exit code = failed checks
```

If the old pod still runs `cloudflared`, both pods are tunnel replicas and Cloudflare sends each
request to one of them. Stop the old one once the new one passes `check_live.sh`.

A stopped pod can fail to start again ("not enough free GPUs", D80). Don't wait on it: create a new
one with the command above.

## Secrets

All secrets are RunPod secrets (RunPod console → Secrets), referenced from the pod env as
`{{ RUNPOD_SECRET_name }}`. `create_pod.sh` wires in whichever exist; after adding one, re-apply
the env to the running pod (RunPod restarts it; `/workspace` survives and `up.sh` reruns). It keeps
the pod's other settings (`BRANCH`, `PUBLIC_KEY`, `LIPREAD_*`, …) unless you pass new values:

```
bash ml/runpod/create_pod.sh --update <POD_ID>
```

| RunPod secret | Pod env var | Used by | Where it comes from |
|---|---|---|---|
| `cf_tunnel_token` | `TUNNEL_TOKEN` | cloudflared | Cloudflare → Zero Trust → Networks → Tunnels → `tryheard` |
| `elevenlabs_api_key` | `ELEVENLABS_API_KEY` | backend | Backend team |
| `google_client_id`, `google_client_secret` | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | backend | Google Cloud Console, the OAuth client |
| `timescale_service_url` | `TIMESCALE_SERVICE_URL` | backend | Backend team (Postgres/Timescale) |
| `session_secret` | `SESSION_SECRET` | backend | Random, made once; changing it signs everyone out |
| `hf_token` | `HF_TOKEN` | ML server | Only to fetch a private fine-tune (`LIPREAD_MODEL=FT_v1_a0.5`) |
| `jupyter_password` | `JUPYTER_PASSWORD` | Jupyter on :8888 | Random; the token for `ml/runpod/jupyter_exec.py` |

`up.sh` keeps the backend's secrets out of the ML server's environment and the tunnel token out of
both servers. Settings that are not secret (`FRONTEND_URL`, `FRONTEND_ORIGINS`,
`SESSION_HTTPS_ONLY=true`) have production defaults in `up.sh`. `SESSION_HTTPS_ONLY` must be the
word `true`: `backend/config.py` compares against it, so `1` would leave the cookie non-Secure.

## What `up.sh` does

Idempotent: a part that is already healthy is left alone. `RESTART=1` pulls `BRANCH`,
re-bootstraps and restarts all three (the way to put a hotfix on the running pod). Runs one at a
time (`flock`), so a manual run and the boot-time watcher don't race. A shell without the pod env
(SSH) reads it from the container's PID 1. If a secret the backend needs is missing, `up.sh` leaves
the backend alone rather than starting it broken (`BACKEND_ALLOW_MISSING=1` overrides).

1. `bootstrap.sh` once per container start: repo at `BRANCH`, the uv env, checkpoints, all on
   `/workspace`. Exit 3 means a bad CUDA host: recreate the pod.
2. ML server via `serve.sh` (model loaded at start-up), unless `/health` already reports
   `LIPREAD_MODEL`. Anything `serve.sh` learns later (`CONDOM=1`, the corrector) works through
   the pod env unchanged.
3. The backend team's server, unchanged: `python main.py` in a Python 3.12 venv on `/workspace`,
   installed exactly as `backend/Dockerfile` does (`requirements.txt`; the diarization extras only
   with `BACKEND_DIARIZATION=1`, which also sets `DIARIZATION=1`). Not in Docker: a RunPod pod is
   itself a container, with no Docker socket and no `CAP_SYS_ADMIN` to run one. Runs with
   `CUDA_VISIBLE_DEVICES=''`, so it never takes GPU memory.
4. `cloudflared tunnel run` with the token from the env (never on the command line), over http2.
5. `watch` (the start command uses it): re-checks every 30 s, looks again 10 s later before acting
   (one slow `/health` during a long read is not an outage), restarts only what is down, and retries
   a failed bootstrap. A failed bootstrap still starts whatever `/workspace` already has.

Logs on the pod: `/workspace/logs/{up,boot,backend,cloudflared}.log`, `/workspace/serve.log`.

## GPU memory

On the 4090 (24 GB) the ML server uses about 1.7 GB loaded and idle (`nvidia-smi`, 2026-10-04);
the backend and cloudflared use none. A vLLM corrector on the same pod (`CONDOM=1`) should cap
itself at `--gpu-memory-utilization 0.75` (18 GB) or lower, which leaves about 4 GB for beam
search peaks.

## Warm standby

See `.context/deploy.md` § Standby for the decision and its cost. A standby is just a second pod
created with `NAME=tryheard-standby bash ml/runpod/create_pod.sh`: it runs the same `up.sh`, so it
joins the tunnel as a second replica. The account allows 2 running pods.

## Frontend (Cloudflare Workers static assets)

Worker `stormhacks2026`, Git-connected (Workers Builds): production branch `master` (it was
`deploy/tryheard` until that PR merged), root directory `frontend`, build command `pnpm build`,
deploy command `npx wrangler deploy` (reads `frontend/wrangler.jsonc`, whose `name` must match the
Worker's). Build variables: `NODE_VERSION=22`, `VITE_API_URL=https://api.tryheard.tech`,
`VITE_TTS_WS_URL=wss://api.tryheard.tech/ws/tts`, `VITE_STT_WS_URL=wss://api.tryheard.tech/ws/stt`,
`VITE_LIPREAD_URL=https://ml.tryheard.tech`; never `VITE_SKIP_AUTH`, and not `VITE_ORT_WEBGPU=1`
(the WebGPU `.wasm` files are over the 25 MiB limit). Custom domains: `tryheard.tech` and
`www.tryheard.tech`, plus a Redirect Rule www → apex (the backend allows only the apex origin).

`public/_headers` sends COOP/COEP (cross-origin isolation for multi-threaded WASM) on every
response; `wrangler.jsonc`'s SPA fallback serves `index.html` for the client routes. There is no
`_redirects`: a redirect rule wins over `_headers` for the same path (so `/app` could lose COOP/COEP),
and `/x /index.html 200` becomes a 308 to `/`. Builds without `VITE_ORT_WEBGPU=1` leave out the
WebGPU-only ORT `.wasm` (27 and 28 MB) and speed mode loads the plain WASM build.

Google sign-in needs `https://api.tryheard.tech/api/auth/google/callback` among the OAuth client's
authorised redirect URIs, and the consent screen published (in "Testing" only listed users can
sign in).
