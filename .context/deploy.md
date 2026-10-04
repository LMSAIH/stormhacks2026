# Deploy: tryheard.tech (2026-10-04)

Runbook: `ml/runpod/README-deploy.md`. Live check: `scripts/check_live.sh`. Diagram:
`docs/architecture/deployment.md`.

## URLs

| URL | What | Where |
|---|---|---|
| https://tryheard.tech | App (www → apex) | Cloudflare Pages, Git-connected, root `frontend/` |
| https://ml.tryheard.tech | ML server (`/health`, `/lipread/crops`) | Pod `vh5w7ghb84dpce` (`tryheard-prod`), `localhost:8000` via tunnel `tryheard` |
| https://api.tryheard.tech | Backend REST + `/ws/stt` | Same pod, `localhost:5000` |
| wss://api.tryheard.tech/ws/tts | Backend TTS WebSocket | Same pod, `localhost:8765` |
| https://vh5w7ghb84dpce-8000.proxy.runpod.net | ML server through RunPod's proxy (changes with the pod) | Same pod |
| https://qa5oi7o7g46n4q-8000.proxy.runpod.net | The earlier Quality server (Romania); not behind the tunnel | Pod `qa5oi7o7g46n4q` |

## Pods and spend

| Pod | Use | Location | Created | Rate |
|---|---|---|---|---|
| `vh5w7ghb84dpce` `tryheard-prod` | Production: ML + backend + cloudflared (`up.sh watch`) | US, secure RTX 4090 | 2026-10-04 15:56:32 UTC by the deploy session | US$0.74/h ≈ CA$1.05/h (1.4250 CAD/USD, close of 2 Oct 2026) |
| `qa5oi7o7g46n4q` `stormhacks-serve` | Earlier Quality server, untouched by the deploy | Romania | earlier | Same rate |

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
- **D93 Speed mode loads the plain WASM ORT build unless WebGPU is requested.** Cloudflare Pages
  rejects files over 25 MiB; the WebGPU build's `.wasm` is 26,781,914 B (asyncify) and the jsep one
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

Decision: **one warm standby only during the judging window**, then stop it. At US$0.74/h
(≈ CA$1.05/h) a 6-hour window costs about US$4.44 ≈ CA$6.33. Outside the window, one pod. The
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

(Filled in as each check passes.)

## Still to do by hand (user)

- Rotate the HF token that sits in plaintext in the env of exited pods `9kjrrcuvzrueu6` and
  `42dc1t20jc8whf` (then the new one goes in the RunPod secret `hf_token`).
- Run the e2e eval on the laptop (the clips in `ml/data/raw_eval` exist only there), signed in:
  `BASE=https://tryheard.tech MODE=normal node ml/scripts/app_eval/e2e_eval.mjs` and
  `MODE=quality`; Quality must report server reads.
