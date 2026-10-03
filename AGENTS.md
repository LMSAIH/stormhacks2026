# StormHacks 2026 — silent-speech assistant (lip reading → voice)

## Purpose
Assistive communication for non-vocal users: the user mouths words at a webcam → visual speech
recognition (lip reading) → LLM corrector cleans the text → ElevenLabs speaks it aloud in real time.
Secondary path: transcribe + diarize other people's speech into captions for the user.
Hackathon build. **Submission due 2026-10-04 12:00 PT** (Devpost, ≤3-min demo video).
Demo speakers include judges, so the model must work on faces it was never fine-tuned on.

## Layout & ownership
- `frontend/` — React 19 / Vite 8 / TypeScript 6 (pnpm), onnxruntime-web + MediaPipe tasks-vision;
  Electron planned. Lip-reading seam: `src/lib/lipreading/` (engine picked by `createEngine.ts`).
- `backend/` — FastAPI server, ElevenLabs integration (planned). Infra team.
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
- Inference tiers (encoder + CTC head exported to ONNX, fp32, greedy CTC):
  1. `onnxruntime-node` in Electron's main/utility process — the default local path.
  2. `onnxruntime-web` WebGPU in the renderer — stretch; needs Electron Vulkan switches on Linux.
  3. FastAPI on a RunPod pod with Python preprocessing + beam search — backup, best accuracy.
- Python: `uv` + Python 3.11 (system Python is 3.14 — too new for this stack, don't use it).
- Compute: RunPod community RTX 4090 for training and the demo pod; laptop RTX 4060 for dev
  (`supergfxctl -m Hybrid`, then re-login). Homelab AMD card on `qwen` is deferred (no access).

## Commands
- Frontend (**pnpm only, never npm**): `cd frontend && pnpm install && pnpm dev` · `pnpm build` · `pnpm lint`
- ML: `cd ml && uv sync --extra export --extra dev && ./scripts/download_checkpoints.sh`, then
  `uv run lipread transcribe clip.mp4 [--decode beam]` · serve `uv run uvicorn lipread.serve.app:app`
  · `scripts/{bench,export_onnx,convert_ckpt}.py` (see `ml/README.md`). Pod: `ml/runpod/*.sh`.
- Smoke: `./smoke.sh` after every product-code change; report `smoke: N/N`
  (`./smoke.sh ml` for one part, `SMOKE_REQUIRE_CUDA=1` on GPU boxes).

## Conventions
- Feature branches + PRs with light review; never push straight to `master`.
- Conventional commits: `feat(scope): …`, `fix(scope): …`.
- Never commit checkpoints, datasets or recordings: `ml/data/`, `ml/checkpoints/` are gitignored;
  keep them on the RunPod volume or a private HF Hub repo.
- Secrets (ElevenLabs, OpenRouter, RunPod, HF) live in `.env` (gitignored), never in the
  frontend bundle; browser → our server → provider.
- Before trusting an ONNX export, diff its logits against PyTorch on the same clip.
- Pretrained VSR checkpoints are non-commercial (LRS3/BBC terms): hackathon use only.

## Non-goals
- Training VSR from scratch or at LRS3 scale; fairseq / AV-HuBERT family.
- Streaming sub-utterance lip reading — push-to-talk utterances only.
- Languages other than English. Commercial use of the checkpoints.

## Pointers
- `.context/project-brief.md` — decision log (D1…), API contract (§5), preprocessing spec (§4),
  risks, Devpost tracks, **baseline numbers + live pod URL (§11)**, Phase A/B todo (§12).
