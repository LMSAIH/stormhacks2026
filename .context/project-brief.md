# Project brief — StormHacks 2026 silent-speech assistant

Onboarded 2026-10-03 (~13:00 PT) on branch `ml/model-pipeline`. AGENTS.md is the lean summary;
this file holds the reasoning. Next free numbers: **Q21, D32**.

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
GET  /health                → 200 {"status":"ok","model":"auto_avsr_lrs3_v19.1","device":"cuda"}
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
POST /correct               json {"text": "..."} → {"text": "..."}   # used by tiers 1–2
```
Clip expectations: frontal face, ≥0.5 s, ≤10 s, any fps (server resamples to 25).
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
