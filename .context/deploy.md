# Deploy: tryheard.tech (2026-10-04)

Runbook: `ml/runpod/README-deploy.md`. Live check: `scripts/check_live.sh`. Diagram:
`docs/architecture/deployment.md`.

## URLs

| URL | What | Where |
|---|---|---|
| https://tryheard.tech | App (www → apex by a Redirect Rule) | Cloudflare Workers static assets: Worker `stormhacks2026`, Workers Builds from `frontend/` |
| https://ml.tryheard.tech | ML server (`/health`, `/lipread/crops`) | Pod `vh5w7ghb84dpce` (`tryheard-prod`), `localhost:8000` via tunnel `tryheard` |
| https://api.tryheard.tech | Backend REST + `/ws/stt` | Same pod, `localhost:5000` |
| wss://api.tryheard.tech/ws/tts | Backend TTS WebSocket | Same pod, `localhost:8765` |
| https://vh5w7ghb84dpce-8000.proxy.runpod.net | ML server through RunPod's proxy (changes with the pod) | Same pod |
| https://qa5oi7o7g46n4q-8000.proxy.runpod.net | The earlier Quality server (Romania); not behind the tunnel | Pod `qa5oi7o7g46n4q` |

## Pods and spend

| Pod | Use | Location | Created | Rate |
|---|---|---|---|---|
| `vh5w7ghb84dpce` `tryheard-prod` | Production: ML + backend + cloudflared (`up.sh watch`) | US, secure RTX 4090 | 2026-10-04 15:56:32 UTC by the deploy session | US$0.74/h ≈ CA$1.05/h (1.4250 CAD/USD, close of 2 Oct 2026) |
| `qa5oi7o7g46n4q` `stormhacks-serve` | Earlier Quality server, not behind the tunnel. **Stopped 2026-10-04 17:23 UTC** at the user's go-ahead (volume still billed) | Romania | earlier | Same rate while it ran |
| `tryheard-standby` (not created yet) | Warm standby for the judging window (ends 14:30 PT), `TUNNEL_REQUIRE_HEALTHY=1`. Created at 12:00 PT (19:00 UTC; routine `trig_016KSTStQQxSWjSv3CamfxLu`) and stopped at 14:30 PT (21:30 UTC; routine `trig_01Hidu5rPg239Fw7n2WrJToV`), both at the user's request | US/CA first | 19:00 UTC | Same rate while it runs: 2.5 h ≈ US$1.85 ≈ CA$2.64 |

## Decisions

- **D91 Ingress = one Cloudflare named tunnel (`tryheard`) with cloudflared on the pod.** DNS
  points at the tunnel, so a recreated pod keeps every URL. The TTS socket is a path on `api.`
  (`/ws/tts` → `:8765`), not a `ws.` hostname: the session cookie is host-only on
  `api.tryheard.tech` and would never reach `ws.`, so every TTS socket would close with 4401.
- **D92 Backend runs on the GPU pod as a process, not in Docker.** Measured: no Docker socket,
  `CapEff 00000000a80425fb` (no `CAP_SYS_ADMIN`) on `qa5oi7o7g46n4q`; a RunPod pod is itself a
  container. A separate CPU host would need its own tunnel and token (one tunnel routes every
  hostname to every connector) plus an account we don't have, so the backend runs unchanged in a
  Python 3.12 venv built the way `backend/Dockerfile` builds its image, with
  `CUDA_VISIBLE_DEVICES=''`. `DIARIZATION` stays at the backend's default (off);
  `BACKEND_DIARIZATION=1` turns it on.
- **D93 Speed mode loads the plain WASM ORT build unless WebGPU is requested.** Cloudflare (Workers
  static assets and Pages alike) rejects files over 25 MiB; the WebGPU build's `.wasm` is 26,781,914 B (asyncify) and the jsep one
  28,312,028 B. The plain build's is 14,239,897 B. Measured in Chromium (cross-origin isolated,
  2 threads, int8 model sha256 `55143d51…`, the 24-frame synthetic crop fixture): log-probs
  identical (max abs diff 0, same argmax on all 24 frames), session create 4,229 vs 5,207 ms,
  run 774 vs 869 ms (one run each, WASM build vs WebGPU build).
- **D94 Standby**: see below.

## Standby (decision)

Recreating the pod is one command and took about 3 minutes on 2026-10-04 (created 15:56:32 UTC,
bootstrap done 15:59:05, ML + backend healthy 15:59:27); re-applying the env (a restart of the
same pod) took about 1 minute to bootstrap. The risk a standby covers is the one we can't fix in
3 minutes: no free 4090 when we need one ("not enough free GPUs", D80), or a host going down
mid-judging. A second pod running the same `up.sh` joins the tunnel as a replica and takes traffic
the moment the first one's connector drops, with no DNS change.

Decision: **one warm standby only during the judging window**, then stop it. The user set the
window to 12:00–14:30 PT: 2.5 h at US$0.74/h (≈ CA$1.05/h) ≈ US$1.85 ≈ CA$2.64. Outside the window, one pod. The
account's limit is 2 running pods, so the standby means `qa5oi7o7g46n4q` (the Romania pod) has to
be stopped first, which needs the user's go-ahead (it's the laptop's Quality server until the
laptop points at https://ml.tryheard.tech).

Cloudflare doesn't health-check the services behind a connector: if a pod's ML server dies while
its cloudflared stays up, requests reaching that pod fail until `up.sh watch` restarts the server
(re-check every 30 s, plus warm-up).

## Measurements

Warm Quality reads, `POST /lipread/crops` beam on the 24-frame synthetic fixture, round trip from
the cloud session's container through RunPod's proxy, 3 runs each (2026-10-04 ~16:00 UTC):

| Pod | Round trip (s) | Server total (ms) |
|---|---|---|
| `vh5w7ghb84dpce` (US) | 0.588, 0.627, 0.538 | 98.8, 167.6, 35.2 |
| `qa5oi7o7g46n4q` (Romania) | 0.944, 1.290, 1.181 | 43.5, 50.9, 63.5 |

GPU memory on `vh5w7ghb84dpce` with the ML server loaded and idle: 1,705 MiB of 24,564 MiB.

## Verification log

All times 2026-10-04 UTC. "From the pod" = `scripts/check_live.sh` run on `vh5w7ghb84dpce` over the
public internet (real DNS → Cloudflare edge → tunnel); the cloud session's egress proxy cached a
"no such host" for `api.tryheard.tech` from before the record existed and refused it for a while.

- 16:04:57 tunnel `tryheard` registered from the pod: 4 connections, http2, Cloudflare `atl01/06/13/16`.
- ~16:06 https://ml.tryheard.tech: `/health` ok (stock `LRS3_V_WER19.1`, beam 20, LM 0.2), CORS for
  `https://tryheard.tech`, `POST /lipread/crops` beam → 200 (from this session and from the pod).
- 16:13:48 all eight RunPod secrets applied (`create_pod.sh --update`); `up.sh`: ml, backend, tunnel ok.
- ~16:15 https://api.tryheard.tech from the pod: `/openapi.json` 200, `/api/auth/me` 401 signed out,
  CORS preflight allows `https://tryheard.tech` with credentials, `/api/auth/google/login` → 302 to
  Google with `redirect_uri=https://api.tryheard.tech/api/auth/google/callback` (so the forwarded
  proto and host reach uvicorn), `wss://api.tryheard.tech/ws/tts` and `/ws/stt` both upgrade (101).
  `check_live.sh`: 9/16, the other 7 being the Pages checks (project not connected yet).
- ~16:55 the app went live as a Worker (`stormhacks2026`, Workers Builds): the first build failed
  on `master` (`ort-wasm-simd-threaded.asyncify.wasm` 25.5 MiB, over the 25 MiB limit; master lacks
  the size fix), green on `deploy/tryheard`. ~17:00 `check_live.sh` from the pod: **17/17**: `/` and
  `/app` 200 with COOP/COEP/CORP, `.mjs` as `text/javascript`, `.wasm` as `application/wasm`,
  www → apex, plus all ML and API checks (beam read 43.5 ms server total).
- Production build in Chromium (`pnpm preview` of the same `dist`, the app's exact ORT setup: plain
  WASM build, `wasmPaths=/ort/`, proxy worker on): cross-origin isolated, 2 threads, session create
  4,313 ms, 24-frame read 1,481 ms, log-prob sum −1,753,195.0 vs native ONNX Runtime −1,753,116.1
  (relative 4.5e-5). The WebGPU build fails there as designed (its `.wasm` is left out).
- The same check against the live site from the cloud session's browser was inconclusive: its
  egress proxy refused about half of new connections to `tryheard.tech` (8 probes: 4 × 200,
  4 × refused), which broke the ORT worker's fetch.
- ~17:10 a real visitor (a teammate's browser on https://tryheard.tech), from the backend and ML
  server logs: Google callback → 303 to the app, `GET /api/voices` 200 (session cookie through the
  tunnel; ElevenLabs key good), `WebSocket /ws/stt` accepted, and Quality reads: `POST
  /lipread/crops` beam 200 for 483, 211 and 159 frames, each with its CORS preflight and a
  `/lipread/phrases` 200. Earlier attempts: 503 (before the Google secrets) and two 401s (before
  the redirect URI was registered).
- 17:22 merged master into `deploy/tryheard` (master's frontend now calls `DELETE`, and its backend
  allows it); updated the prod pod's backend to match by killing it and running `up.sh`. The pass
  raced the old process (it still answered `/openapi.json` while shutting down) and skipped it; the
  watcher restarted it at 17:22:36. **API down about 10–15 s.** Fix: `RESTART=backend` (tested
  17:24:44–49: re-pull 4 s, backend restart ~1 s, ML server and tunnel PIDs unchanged). CORS now
  allows `GET, POST, PUT, DELETE` for `https://tryheard.tech`.
- 17:30:30 PR #16 merged; the prod pod switched to `master` (`create_pod.sh --update`, which keeps
  the secrets and `PUBLIC_KEY`): container restart, bootstrap from `master` 17:30:34, ML server
  17:30:38, backend 17:30:50, tunnel 17:30:51, about **21 s** down. `check_live.sh` from the pod
  17/17; the pod env now says `BRANCH=master`.
- Not checked from here directly: Postgres (would need the production credentials outside the app);
  the signed-in visitor's requests above ran without errors in the backend log. TTS audio and Normal
  (on-device) reads leave no server trace: confirmed by the user in the browser, or not at all.

## Still to do by hand (user)

- ~~Rotate the HF token in the plaintext env of exited pods `9kjrrcuvzrueu6` and `42dc1t20jc8whf`~~
  done by the user 2026-10-04; the new one is the RunPod secret `hf_token`.
- Run the e2e eval on the laptop (the clips in `ml/data/raw_eval` exist only there). Sign in at
  https://tryheard.tech in a normal browser, copy the `voice_session` cookie of `api.tryheard.tech`
  (DevTools → Application → Cookies), then from `ml/`:
  `BASE=https://tryheard.tech SESSION_COOKIE=<value> MODE=normal TAG=prod_normal PLAYWRIGHT_CORE=<playwright-core/index.mjs> CHROME=<chromium> node scripts/app_eval/e2e_eval.mjs`,
  again with `MODE=quality TAG=prod_quality` (it must report server reads), then
  `uv run python scripts/app_eval/score_eval.py prod_normal prod_quality`. The trace
  (`window.__lipTrace`) exists only in dev builds, so the production run gives lines and WER, not cuts.
