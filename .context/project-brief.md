# Project brief — StormHacks 2026 silent-speech assistant

Onboarded 2026-10-03 (~13:00 PT) on branch `ml/model-pipeline`. AGENTS.md is the lean summary;
this file holds the reasoning. Next free numbers: **Q21, D36**.

## 1. Product

A non-vocal user (laryngectomy, ALS, ventilated patients, aphasia) mouths words at a webcam.
The app lip-reads the utterance, an LLM cleans up the noisy text, and ElevenLabs speaks it aloud.
A second path captions *other* people's speech (transcription + diarization) for the user.
Prior art we build on: Chaplin (Auto-AVSR + Ollama corrector, push-to-talk webcam lip reader).

Architecture (from the team diagram):

```
Client ──webcam──► Electron App ◄──────► FastAPI Server ──► Lip reader (Chaplin / Auto-AVSR)
   ▲                  ▲                       └──────────► ElevenLabs API ──► real-time voice ──┐
   │                  └── Transcription model ─► Diarization model (captions for the user)     │
   └───────────────────────────────────── stream ◄────────────────────────────────────────────┘
```

## 2. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Product = visual speech recognition (silent video → text) feeding TTS | Core idea |
| D2 | ML owner scope: lip-reader training pipeline + hosting. Others: Electron, FastAPI, infra | Team split |
| D3 | Feature branches + PRs, light review; ML work on `ml/model-pipeline` | Q2 |
| D4 | Architecture per team diagram (§1) | Provided diagram |
| D5 | Base model Auto-AVSR `LRS3_V_WER19.1`; USR 2.0 stretch. No AV-HuBERT/fairseq/LLM-decoder VSR | Best practical WER, plain PyTorch, Chaplin glue; fairseq archived & painful |
| D6 | Utterance-level, push-to-talk (or mouth-activity trigger). No streaming VSR | Models are utterance-level |
| D7 | Preprocessing reused verbatim from auto_avsr/Chaplin (see §4) | Misaligned crops tank WER |
| D8 | VSR fine-tune via auto_avsr's Lightning recipe (or hand-rolled PEFT LoRA); Unsloth only for the corrector | Unsloth only supports HF `transformers` models |
| D9 | ~~PyTorch default, ONNX nice-to-have~~ → superseded by D22 | |
| D10 | ~~Train on homelab AMD~~ → superseded by D19 | |
| D11 | `smoke.sh` is the verification gate (hackathon mode, no test suite) | Q5 |
| D12 | ~~Two torch builds (CUDA + ROCm)~~ → CUDA only for now; ROCm extra optional later | D19 |
| D13 | `ml/` is a `uv` project on Python 3.11 | System Python 3.14 too new for mediapipe/torchvision stack |
| D14 | Deadline **2026-10-04 12:00 PT** | Q5b / Devpost |
| D15 | ~~`qwen` = training host~~ → deferred, no SSH access | D19 |
| D16 | Cloud demo backend = persistent RunPod pod (community 4090) behind its HTTP proxy | Q14b, Q18: no cold starts |
| D17 | Data: own recordings (Whisper-labelled) + LRS3 trainval slice; eval on `mattymchen/lrs3-test`. Corrector is a **must-have** | Q8: A+B |
| D18 | Top-level `ml/` with `train/`, `serve/`, `scripts/`, `third_party/`, gitignored `data/` | Q16 |
| D19 | Training on RunPod 4090 (CUDA); laptop RTX 4060 for dev/local inference (`supergfxctl -m Hybrid`) | `qwen` unreachable; laptop is Iris Xe until switched |
| D20 | Corrector: Llama-3.2-3B-Instruct LoRA (rank ≤32). Host on the RunPod pod (llama.cpp/vLLM) **or** Cloudflare Workers AI LoRA; local (4060) meanwhile | Q17 "A or C"; Workers AI LoRA only supports non-quantized Llama/Mistral/Gemma, rank ≤32 |
| D21 | Vendor minimal auto_avsr/Chaplin inference + preprocessing into `ml/third_party/`, pinned commit, LICENSE + attribution | Q19 |
| D22 | Inference tiers: (1) ONNX encoder+CTC on `onnxruntime-node` in Electron, (2) `onnxruntime-web` WebGPU in renderer (stretch), (3) RunPod FastAPI w/ Python preprocessing + beam search (backup) | User wants ONNX/WebGPU for latency; research says WebGPU is feasible but risky |
| D23 | Learning mode off. Demo speakers = team **and judges** | Speaker fine-tuning won't help judges → corrector + constrained-phrase mode matter more |
| D24 | Opt into Devpost tracks listed in §8 | Devpost scan |
| D25 | **Corrector is out of the MVP.** Only after the pipeline is verified; host = RunPod. `lipread.corrector` is a passthrough hook (OpenAI-compatible API) until then | User, 2026-10-03 |
| D26 | Demo machine OS = Linux or macOS. Tier choice (local ONNX vs hosted) decided by measured latency + accuracy (`scripts/bench.py`) | User; macOS has no CUDA → ORT CPU/CoreML locally |
| D27 | Frontend uses **pnpm, never npm** | User |
| D28 | Inference uses Chaplin's espnet (`(B,C,T,H,W)` input); fine-tuning uses auto_avsr's espnet. Weights convert between them losslessly (`convert_ckpt.py`, verified max\|Δlogp\| 9.5e-6) → **fine-tune from 19.1**, not auto_avsr's 20.3% zoo ckpt | Same architecture, renamed modules |
| D29 | ONNX = encoder + CTC head only, fp32, 775 MB, dynamic T; parity verified at T=37/73/151 (100% argmax agreement); ORT CPU 0.3–0.5 s/clip on the dev laptop | `export_onnx.py` |
| D30 | Face-coverage gate: reject clips with a face in <50% of frames (`LIPREAD_MIN_FACE_COVERAGE`) | BlazeFace short-range fires on pure noise; upstream interpolates those hits into garbage crops |
| D31 | `/lipread` accepts `precropped=true` (already-aligned 96×96 mouth clip) | JS tier can crop client-side and upload tiny clips; LRS3 HF mirror only ships crops |
| D32 | Pod = **secure-cloud** RTX 4090 (RO, $0.74/hr); `bootstrap.sh` fails fast on a bad host (cuInit preflight) | Two community 4090s (TW, driver 580 / CUDA 13.0) had `cuInit()=999` even with the image's own torch; community hosts with CUDA ≤12.9 unavailable |
| D33 | Baseline WER uses HF `mattymchen/lrs3-test`, which only ships **pre-made 96×96 gray crops** → it measures model + decoding, **not our crop pipeline** | Raw-video LRS3 (`TheNHz/ellipsis-lrs3-raw`) is gated: needs the user to accept terms on HF |
| D34 | Greedy CTC ≈ 7 WER points worse than beam+LM (28.6% vs ~22%) but ~25× faster; local tier = greedy, hosted beam = "accuracy mode" | §11 |
| D35 | Frontend (master `d3134c0`) has an ONNX engine seam built for a LipNet placeholder; wiring = Phase A (§12) | §12 |

## 3. Open questions (defaults apply if unanswered)

| Q | Question | Default if unanswered |
|---|---|---|
| Q11 | How does the FastAPI server reach the lip reader? | Separate model service, contract in §5. Local tiers 1–2 bypass the server for VSR and just POST the text |
| Q15 | Who owns transcription + diarization (maybe ML owner)? | Unowned; if ML owner: faster-whisper + pyannote 3.x on the same RunPod pod |
| Q17 | Corrector host | **Answered: RunPod** (post-MVP, D25) |
| Q20 | `qwen` access (root SSH key rejected, Tailscale SSH not enabled) | Deferred; revisit only if RunPod is a problem |
| — | Which GPU is actually in `qwen`? (written variously as 9600/7900/9700 XTX) | Unknown; gfx1100 vs gfx1201 matters only if we go back to it |

## 4. Preprocessing spec (must match train and serve)

From `auto_avsr/preparation/detectors/mediapipe/` (verified by research):
1. Resample video to **25 fps**.
2. MediaPipe **face detection** (full-range, then short-range fallback) — keep keypoints 0–3:
   right eye, left eye, nose tip, mouth centre. Not the 478-point mesh, not 68 landmarks.
3. Interpolate missing frames; smooth landmarks over a 12-frame window.
4. Reference: `20words_mean_face.npy` reduced to 4 points (means of 68-pt groups 36:42, 42:48,
   31:36, 48:68). `cv2.estimateAffinePartial2D(..., method=LMEDS)` onto a 256×256 reference.
5. Cut 96×96 around the mouth point → grayscale → centre-crop 88×88 → normalise mean 0.421,
   std 0.165. Model input `(1, T, 88, 88)` per clip = `(C, T, H, W)`; the ONNX graph takes
   `video: (1, 1, T, 88, 88)` and returns `log_probs: (T, 5049)` (tokens in `artifacts/tokens.json`).
6. Reject the clip if a face was found in <50% of frames (D30).

JS port (tier 1/2): `@mediapipe/tasks-vision` (1.0.1) **FaceDetector**; check keypoint order
empirically by x-coordinate. Parity risks: LMEDS vs least-squares, int-truncated keypoints,
interpolation mode, gray coefficients, 25 fps resampling. **Dump crops from both pipelines on
the same clip and diff them before trusting the JS path.** If parity fails, ship the raw clip
to Python (tier 3).

## 5. Lip-reader service contract (DRAFT — Q11 still open)

```
GET  /health                → 200 {"status":"ok","model":"LRS3_V_WER19.1","device":"cuda:0"|"cpu"|null,
                                   "loaded":bool,"corrector":bool}    # device null until model loads
POST /lipread               multipart: file=<webm|mp4 clip>, optional fields:
                              decode=greedy|beam (default greedy), correct=true|false (default true),
                              precropped=true|false (default false: raw webcam clip)
                            → 200 {
                                "text": "corrected sentence (== raw_text while corrector is off)",
                                "raw_text": "RAW VSR OUTPUT (uppercase)",
                                "confidence": 0.0-1.0 | null,   # greedy only
                                "frames": n,                     # at 25 fps
                                "latency_ms": {"load", "crop", "vsr", "correct", "total"}
                              }
                            → 422 {"detail": {"error": "no_face_detected" | "clip_too_short" |
                                   "clip_too_long" | "unreadable_video" | "bad_decode"}}
POST /lipread/crops?t=<frames>[&h=96&w=96&decode=beam&correct=false]   # Phase A "accuracy" mode
                            body: raw uint8, exactly t*h*w bytes = t row-major h×w gray frames at
                              25 fps from the client's crop pipeline (phase-a-design §3).
                              Optional `Content-Encoding: gzip` (browser CompressionStream("gzip"));
                              Content-Type is ignored (send application/octet-stream).
                              t: 13–250 (0.5–10 s). h = w = 96 (aligned patch, server centre-crops
                              to 88) or 88 (already centre-cropped; bit-identical model input).
                              decode=greedy|beam (default beam), correct=true|false (default false).
                            → 200 same JSON as /lipread. Lossless (no mp4 round trip, unlike
                              precropped=true); latency_ms.load = gunzip + parse, .crop = normalise.
                            → 422 {"detail": {"error": "bad_shape" | "clip_too_short" |
                                   "clip_too_long" | "body_size_mismatch" | "bad_gzip" | "bad_decode",
                                   "message": "…"}}   # message only on bad_shape / body_size_mismatch
                            → 413 {"detail": {"error": "body_too_large"}}   # > 250·96·96 B + 64 KiB
                            → 415 {"detail": {"error": "unsupported_encoding", "message": "…"}}
                              Missing / non-integer query params → FastAPI's default 422, where
                              `detail` is a list, not an object.
POST /correct               json {"text": "..."} → {"text": "..."}   # used by tiers 1–2
```
Clip expectations: frontal face, ≥0.5 s, ≤10 s, any fps (server resamples to 25).
CORS is `*`; browsers preflight `/lipread/crops` (Content-Encoding, octet-stream) and the server
allows it. Gzip bodies are inflated to at most t*h*w + 1 bytes (zip-bomb guard). Beam decodes are
serialised per process (espnet's CTC prefix scorer keeps per-search state, so concurrent beams crashed);
a queued request's `vsr` includes that wait. Greedy runs concurrently. `/lipread/crops` is live on the
pod only after a redeploy (pull + `ml/runpod/serve.sh`); until then it 404s and the app falls back
to speed mode. Bench it losslessly: `scripts/bench.py --backend http --transport crops`.
Frontend can build against a mock returning canned text until the pod is up.

## 6. Plan (now → 2026-10-04 12:00 PT)

| When (PT) | ML track | Done when |
|---|---|---|
| Sat 13–15 | `ml/` uv scaffold, vendor third_party, download 19.1 ckpt, baseline inference on 4060 + RunPod | One webcam clip → text |
| 15–17 | Record speakers (scripted demo-domain sentences, with audio), Whisper labels, preprocess, manifests; baseline WER on ~100 `lrs3-test` clips | Manifest + baseline WER |
| 17–22 | Fine-tune (decoder + last conformer blocks, low LR, LRS3 slice mixed in) on RunPod | Val WER ≤ baseline |
| 17–22 ∥ | Generate (raw → truth) pairs; Unsloth LoRA on Llama-3.2-3B; export | Corrector lowers WER |
| 22–02 | ONNX export encoder+CTC, diff vs PyTorch; tier-1 `onnxruntime-node` handoff to frontend | ONNX greedy matches PyTorch |
| 02–06 | Deploy pod: FastAPI `/lipread` + corrector; constrained-phrase mode | Contract §5 live |
| 06–09 | WebGPU stretch, JS preprocessing parity; eval table | Numbers for slides |
| 09–12 | Demo video (≤3 min), Devpost, opt into tracks | Submitted |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Judges' faces ≠ training speakers → 30–45% WER | Corrector + constrained-phrase demo mode; keep stock ckpt as default |
| Crop misalignment explodes WER | Verbatim preprocessing; visually inspect crops; JS/Python crop diff |
| ONNX export breaks (rel-pos `extend_pe` traced constant, masks) | Pre-extend PE; espnet_onnx `OnnxRelPosMultiHeadedAttention` as template; bucket T to fixed lengths |
| WebGPU in Electron on Linux | Switches `enable-unsafe-webgpu`, `enable-features=Vulkan`; tier 1 is the real path; no fp16 on Linux NVIDIA |
| Greedy CTC worse than published 19.1% (that's beam + attention) | Tier 3 beam for accuracy; corrector |
| Venue network down during demo (tier 3) | Tier 1 runs offline; record demo video early |
| Fine-tune overfits/forgets | Mix LRS3 slice, early-stop, keep stock ckpt |
| LRS3 mirror legality (`TheNHz/ellipsis-lrs3-raw`, gated) | Own recordings + ungated test set only if in doubt |
| Licences: 19.1 ckpt non-commercial (BBC terms); USR 2.0 weights unstated | Hackathon only; say so in README |

## 8. Devpost tracks (stormhacks2026.devpost.com — opt in per track!)

Submission: project link + ≤3-min video. Criteria: technical complexity, design, pitch, originality.
- **Opt in now:** [MLH] Best Use of ElevenLabs (1 winner), Best MedTech, SSSS Python Track,
  IATSU Best Design, Enactus UNSDG (frame SDG 3 + 10), Surge Choice (3 winners).
- **Cheap adds:** [MLH] Best .Tech Domain (free domain), [MLH] Gemini (fallback corrector),
  **TiDB x AI Open Build** (3 winners, up to $1,000 credits — vector search over the user's
  personal phrase bank as corrector context: real accuracy gain).
- **Optional:** CSSS CS Legacy (needs 50% Discord scavenger hunt; DECtalk/Hawking framing).
- **No AMD / Cloudflare / RunPod prizes** — pitch ROCm/WebGPU as technical complexity only.

## 9. Research wishlist

| Item | Why it matters | Status / default |
|---|---|---|
| Base model choice | Everything downstream | **Done** — Auto-AVSR 19.1 (user-supplied research) |
| ORT Web WebGPU op coverage (Conv3D) | Tier 2 feasibility | **Done** — Conv3D in native WebGPU EP since ORT 1.25 (fp32, naive kernel); ops page stale; per-node wasm fallback |
| Electron WebGPU on Linux | Tier 2 on dev laptop | **Done** — needs switches (§7); Windows/D3D12 likely fine |
| MediaPipe keypoints used by auto_avsr | JS parity | **Done** — BlazeFace 4 keypoints (§4) |
| Cloudflare Workers AI | Q17 | **Done** — LoRA beta: Llama/Mistral/Gemma, rank ≤32, <300 MB, `wrangler ai finetune create`; no custom PyTorch/GPU containers |
| Public Auto-AVSR ONNX export | Saves hours | **Done** — none exists; export ourselves |
| `onnxruntime-node` GPU EPs on demo machine | Tier 1 speed | Open — CPU everywhere, CUDA on Linux x64, DirectML on Windows; measure on demo laptop |
| Demo machine OS (Linux laptop vs a Windows teammate) | Tier 1/2 EP choice | Open — ask team |
| ElevenLabs latency/streaming API + credits | Voice loop latency | Open — infra team; ask MLH table for credits |
| Transcription/diarization models | Q15 | Open — default faster-whisper + pyannote |

## 10. Environment notes

- Dev laptop `saphirex16`: Arch, Intel Iris Xe + RTX 4060 (supergfxctl, currently Integrated),
  nvidia-open 615 + CUDA 13.4 installed, Python 3.14 system, `uv` installed.
- Tailnet hosts: `qwen` (training box, inaccessible), `pve1` (Proxmox), others irrelevant.
- RunPod community 4090 ≈ $0.34 USD/hr; check console prices at deploy time.
- Sources: user-pasted research report (Auto-AVSR/USR/Unsloth/ROCm/ONNX), Devpost scan,
  browser-inference research (ORT PR #27917, electron#40929, auto_avsr mediapipe detector,
  Cloudflare LoRA docs) — URLs in session; key ones inline above.

## 11. Baseline (2026-10-03, stock `LRS3_V_WER19.1`, no fine-tune, no corrector)

**Data:** first 100 clips of HF `mattymchen/lrs3-test` shard 0 (mean 2.53 s). ⚠️ The mirror only
ships pre-made 96×96 grayscale mouth crops (no raw video), so clips are sent `precropped` and our
face-detection/crop path is **not** exercised; treat absolute WER as model+decoding only (D33).
Refs lowercased, punctuation stripped, WER via jiwer. Beam = 40 + RNN-LM (weight 0.3, CTC 0.1).
Latencies in ms, p50 / p95. `vsr` = model + decode; `rtt` = client-measured round trip.

| Where / path | Decode | WER | vsr | server total | rtt | network | upload |
|---|---|---|---|---|---|---|---|
| Pod 4090, in-process PyTorch | greedy | 28.6% | 33 / 34 | 33 / 35 | — | — | — |
| Pod 4090, in-process PyTorch | beam | 22.6% | 878 / 1912 | 879 / 1913 | — | — | — |
| Pod 4090, via service (127.0.0.1) | greedy | 28.9% | 33 / 41 | 42 / 60 | 45 / 63 | 3 / 4 | 17 / 44 KB |
| Pod 4090, via service (127.0.0.1) | beam | 21.7% | 862 / 1933 | 874 / 1960 | 878 / 1963 | 4 / 4 | 17 / 44 KB |
| **Laptop → pod proxy URL** (Vancouver→RO) | greedy | 28.9% | 41 / 44 | 58 / 75 | **310 / 472** | 261 / 416 | 17 / 44 KB |
| **Laptop → pod proxy URL** | beam | 21.7% | 875 / 1853 | 886 / 1879 | **1308 / 2530** | 325 / 1034 | 17 / 44 KB |
| **Laptop local, ONNX (ORT CPU, i9-13900H)** | greedy | 28.6% | 174 / 423 | 242 / 478¹ | — | — | none |
| Laptop local, PyTorch CPU | greedy | 28.6% | 315 / 815 | 316 / 820 | — | — | none |

¹ ONNX "crop" stage (56/111 ms) is mostly ORT's spinning thread pool stealing CPU from numpy prep
(set `allow_spinning=0` later); real crop of precropped input is ~1 ms.
Service-path WER differs from in-process by ±1 point because uploads are mp4-encoded (lossy) —
within noise for n=100. Published 19.1% is beam+LM on all 1,321 test clips.

**Reading it (local vs hosted):**
- Bandwidth is a non-issue: one precropped utterance = 17 KB p50 / 44 KB p95. No streaming needed.
- **Local ONNX greedy (~0.24 s on laptop CPU, offline, private) ties hosted greedy on accuracy and
  beats it on latency** (hosted greedy is dominated by ~260 ms network to Romania).
- Hosted beam buys **~7 WER points** for **~1.3 s p50 / 2.5 s p95** end-to-end. Worth it as an
  "accuracy mode"; beam size is untuned (40) — B1 sweeps it.
- Recommended design (matches the /btw note): detect + crop locally (MediaPipe in the renderer),
  run local ONNX greedy by default, offer hosted beam on the precropped clip as accuracy/fallback.
  ElevenLabs needs the network anyway, so hosted isn't a new dependency — but local keeps video
  on-device (MedTech pitch) and survives a dead pod.

**Pod (left RUNNING):** `jr602gal8ql6c0`, secure RTX 4090, Romania, $0.74/hr (+50 GB volume).
- Service: **https://jr602gal8ql6c0-8000.proxy.runpod.net** (`/health`, `/lipread`, `/correct`)
- SSH: `ssh -p 17418 root@213.173.98.231` (key = your `~/.ssh/id_ed25519`) or via
  `jr602gal8ql6c0-64410d3c@ssh.runpod.io`. Repo/env/checkpoints on `/workspace` (survive restart).
- Restart service: `bash /workspace/stormhacks2026/ml/runpod/serve.sh`; re-run baseline:
  `bash ml/runpod/bench_baseline.sh`. Smoke on pod: `SMOKE_REQUIRE_CUDA=1 ./smoke.sh` → 2/2.
- **Stop it when idle** (RunPod console, or REST `POST /v1/pods/jr602gal8ql6c0/stop`) — ~$17.8/day
  if left on. Spend to stand up + measure: ≈ $0.61 (incl. two failed community pods).

## 12. Next — Phase A (wire ONNX into the frontend), then Phase B (benchmaxx + fine-tune)

Frontend on master (`d3134c0`, Vite web app, `onnxruntime-web@1.30`, `@mediapipe/tasks-vision@1.0.1`)
already has the seam: `createLipReaderEngine()` uses the real engine iff `/models/lipreader.onnx`
exists, else a mock. It was built for a **LipNet placeholder**, so it mismatches our model on:

| | Frontend today | Our model |
|---|---|---|
| Input | `[1, 75, 50, 100, 3]` RGB, fixed T | `video [1, 1, T, 88, 88]` gray, dynamic T |
| Norm | `/255` | `/255`, then `(x − 0.421) / 0.165` |
| Vocab | 28 chars, blank last | 5049 SentencePiece pieces, blank 0, `<eos>` last, `▁` = word start |
| Crop | axis-aligned padded box around mesh lip points | BlazeFace 4 keypoints → similarity warp → 96 → centre 88 (§4) |
| Timing | 75-frame ring buffer, re-infer every 1.5 s | one utterance per push-to-talk / mouth-activity segment |

Smoke against master's frontend: frozen install ✓, build ✓, **lint ✗** — 22/23 errors are the
vendored `public/ort/*.mjs` bundles (add to ESLint ignores); 1 real: `useLipReader.ts:54`
assigns `activeRef.current` during render.

Todo (tracked in the session task list):
- **A1** Plan the wiring with the team (interview) — artifact hosting (775 MB ONNX can't go in git),
  UX for segmenting, local/hosted switch. No code before approval.
- **A2** `AUTO_AVSR` spec + engine (NCTHW tensor, tokens.json decode, blank 0); publish model files.
- **A3** JS crop parity (Tasks `FaceDetector` + similarity warp + smoothing); diff JS crops vs
  `lipread crops` on the same clip.
- **A4** Utterance segmenting instead of the ring buffer.
- **A5** Hosted engine: POST precropped clip to `/lipread` (beam) as accuracy mode / fallback.
- **A6** Frontend lint green.
- **A7** End-to-end with backend (`origin/backend`, `origin/socket-setup` WebSocket) + ElevenLabs;
  put a real-face clip at `ml/data/smoke/face.mp4` so smoke's e2e check runs. **Gate for Phase B.**
- **B1** Benchmaxx: real-webcam eval set (raw video → exercises our crop), beam-size/LM sweep,
  ONNX int8/fp16, ORT threading, WebGPU vs WASM, exported attention decoder for a JS beam.
- **B2** Fine-tune from 19.1 (`convert_ckpt.py to-auto-avsr` → auto_avsr recipe on the pod →
  convert back → bench); then the corrector (D25) and a constrained-phrase demo mode.
- Open: raw-video LRS3 eval needs your OK to accept `TheNHz/ellipsis-lrs3-raw`'s gated terms.
