# StormHacks 2026 — silent-speech assistant (lip reading → voice)

## Purpose
Assistive communication for non-vocal users: the user mouths words at a webcam → visual speech
recognition (lip reading) → LLM corrector cleans the text → ElevenLabs speaks it aloud in real time.
Secondary path: transcribe + diarize other people's speech into captions for the user.
Hackathon build. **Submission due 2026-10-04 12:00 PT** (Devpost, ≤3-min demo video).
Demo speakers include judges, so the model must work on faces it was never fine-tuned on.

## Layout & ownership
- `frontend/` — React 19 / Vite 8 / TypeScript 6 (pnpm), onnxruntime-web + MediaPipe tasks-vision;
  Electron planned. Lip reading: `src/hooks/useLipReader.ts` → `src/lib/lipreading/` (`crop/` =
  exact port of the Python crop, `createRecognizers.ts`). The teammate's layout is the truth (D40).
- `backend/` — FastAPI server, ElevenLabs integration. Infra team's — not ours; don't integrate (D42).
- `ml/` — lip-reader fine-tuning, ONNX export, model serving, LLM corrector (planned). ML owner,
  branch `ml/model-pipeline`.

## Stack & key deps
- Lip reader: Auto-AVSR `LRS3_V_WER19.1` (~250M params, PyTorch + vendored ESPnet), code
  vendored from auto_avsr / Chaplin into `ml/third_party/` with licences. USR 2.0 = stretch swap.
- Preprocessing must match training exactly: 25 fps → MediaPipe *face detection* (BlazeFace,
  4 keypoints: eyes, nose, mouth — not the face mesh) → 12-frame smoothing → similarity warp to
  the mean face → 96×96 mouth crop → grayscale → centre-crop 88×88 → normalise 0.421 / 0.165.
- Corrector (post-MVP, only after the pipeline is verified): Llama-3.2-3B LoRA via Unsloth, hosted
  on RunPod; `lipread.corrector` is a passthrough hook until then. Unsloth never touches the VSR model.
- Two modes, user-toggled (D34): **speed** = encoder + CTC head as ONNX in the browser
  (onnxruntime-web WASM; WebGPU opt-in `VITE_ORT_WEBGPU=1`, D37), greedy CTC, on the int8
  `dyn-pw8-rn16` quantization, 203 MB (fp32 is 775 MB; `ml/scripts/quantize_onnx.py`, D39; gated by
  `regress_quantized.py`, D41). **accuracy** = FastAPI `POST /lipread/crops`, beam 40 + LM, on a
  RunPod pod or local uvicorn; falls back to speed. `onnxruntime-node` in Electron = later option.
- Python: `uv` + Python 3.11 (system Python is 3.14 — too new for this stack, don't use it).
- Compute: RunPod secure-cloud RTX 4090 (D32) for training and the demo pod; laptop RTX 4060 for dev
  (`supergfxctl -m Hybrid`, then re-login). Homelab AMD card on `qwen` is deferred (no access).

## Commands
- Frontend (**pnpm only, never npm**): `cd frontend && pnpm install && pnpm dev` · `pnpm build` ·
  `pnpm lint` · `pnpm test` · `pnpm typecheck`; model + `.env.local` setup: `frontend/README.md`.
- ML: `cd ml && uv sync --extra export --extra dev && ./scripts/download_checkpoints.sh`, then
  `uv run lipread transcribe clip.mp4 [--decode beam]` · serve `uv run uvicorn lipread.serve.app:app`
  · `scripts/{bench,export_onnx,convert_ckpt,quantize_onnx,regress_quantized}.py` (see
  `ml/README.md`). Pod: `ml/runpod/*.sh`.
- Smoke: `./smoke.sh` after every product-code change; report `smoke: N/N`
  (`./smoke.sh ml` for one part, `SMOKE_REQUIRE_CUDA=1` on GPU boxes).

## Conventions
- Feature branches + PRs with light review; never push straight to `master`.
- Conventional commits: `feat(scope): …`, `fix(scope): …`.
- Never commit checkpoints, datasets or recordings: `ml/data/`, `ml/checkpoints/` are gitignored;
  keep them on the RunPod volume or a private HF Hub repo.
- Secrets (ElevenLabs, OpenRouter, RunPod, HF) live in `.env` (gitignored), never in the
  frontend bundle; browser → our server → provider.
- Before trusting an ONNX export, diff its logits against PyTorch on the same clip; a re-quantized
  model must pass `regress_quantized.py`, then re-lock with `--update-baseline` (`ml/tests/README.md`).

## Non-goals
- Training VSR from scratch or at LRS3 scale; fairseq / AV-HuBERT family.
- Streaming sub-utterance lip reading — push-to-talk utterances only.
- Languages other than English. Commercial use of the checkpoints (LRS3/BBC terms: hackathon only).

## Pointers
- `.context/project-brief.md` — decision log (D1…), API contract (§5), preprocessing spec (§4),
  risks, Devpost tracks, **baseline numbers + pod status (§11)**, Phase A/B todo (§12).
- `.context/phase-a-design.md` — frontend lip-reading contract (crop spec, model I/O, `/lipread/crops`).
