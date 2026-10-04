# ml/ — lip reader (Auto-AVSR) for the silent-speech assistant

Silent video of a mouth in, text out. Pretrained Auto-AVSR `LRS3_V_WER19.1` (~250M params) with
the auto_avsr/Chaplin preprocessing, plus fine-tuning glue, a FastAPI service, ONNX export and an
LLM-corrector hook. Decisions and reasoning: `../.context/project-brief.md`.

## Setup (laptop 4060 or RunPod 4090)

```bash
cd ml
uv sync --extra export --extra dev        # Python 3.11 + CUDA 12.8 torch; add --extra train on the training box
./scripts/download_checkpoints.sh          # ~1.3 GB into ml/checkpoints/ (SKIP_LM=1 to skip the beam-search LM)
```

Laptop GPU: `supergfxctl -m Hybrid`, log out/in, check `nvidia-smi`. Everything also runs on CPU (slowly).

## Use

```bash
uv run lipread transcribe clip.mp4                   # greedy CTC (fast, what ONNX/browser will do)
uv run lipread transcribe clip.mp4 --decode beam     # joint CTC/attention beam + LM (best accuracy)
uv run lipread crops clip.mp4 out/                   # dump 96x96 mouth crops — eyeball alignment / diff vs JS
uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port 8000
```

Service contract (`/health`, `/lipread`, `/correct`) is in the project brief §5.
Env: `LIPREAD_CKPT_DIR`, `LIPREAD_DEVICE` (`cuda`/`cpu`, default auto), `LIPREAD_DECODE`
(`greedy`/`beam`), corrector: `CORRECTOR_BASE_URL`, `CORRECTOR_MODEL`, `CORRECTOR_API_KEY`
(any OpenAI-compatible chat endpoint: llama.cpp server, vLLM, OpenRouter, Workers AI).

## Layout

| Path | What |
|---|---|
| `src/lipread/video.py` | Load video with OpenCV, resample to 25 fps by timestamp |
| `src/lipread/preprocess.py` | MediaPipe 4-keypoint detection → aligned 96×96 mouth crop → `(T,1,88,88)` |
| `src/lipread/model.py` | Load checkpoint, greedy CTC + beam decode |
| `src/lipread/corrector.py` | Optional LLM clean-up via OpenAI-compatible API |
| `src/lipread/serve/app.py` | FastAPI service |
| `src/lipread/vendor/` | Chaplin inference code (MIT / Apache-2.0), lightly patched |
| `third_party/espnet/` | Chaplin's espnet copy → installed as `espnet` (inference) |
| `third_party/auto_avsr/` | auto_avsr training recipe with its own espnet (fine-tuning) |
| `scripts/` | checkpoints, dataset prep, ONNX export, checkpoint-compat check |
| `train/` | Fine-tuning how-to + wrapper |
| `corrector/` | Unsloth corrector training (separate env) |
| `runpod/` | Pod bootstrap |

Licences: pretrained weights are non-commercial (LRS3/BBC). Hackathon use only.
