# Lipreader frontend

Vite + React 19 + TypeScript. **pnpm only** (`pnpm install`, `pnpm dev`, `pnpm lint`,
`pnpm typecheck`, `pnpm test`, `pnpm build`).

## Lip reading (push-to-talk)

Hold **Space** (or the talk button), mouth a sentence, release. The utterance is cropped in the
browser (exact port of the Python preprocessing, `src/lib/lipreading/crop/`) and recognized by the
mode picked in the header:

- **Speed** — on-device ONNX (onnxruntime-web, WebGPU else WASM), greedy CTC. Needs the model:
  `../ml/scripts/publish_frontend_model.sh` copies `lipread_ctc.onnx` (775 MB) + `tokens.json`
  into `public/models/` (gitignored).
- **Accuracy** — the hosted service (beam search + LM): set `VITE_LIPREAD_URL` in `.env.local`
  (see `.env.example`). Falls back to Speed if the service fails.

`/lab` runs the whole pipeline deterministically on `public/test/clip.mp4` and publishes every
intermediate result on `window.__lipLab` (used by Playwright checks).

Tests: `pnpm test` (unit + crop parity vs Python). With the model published,
`LIPREAD_ONNX_TEST=1 pnpm test` also runs the real-model golden test (needs the gitignored fixture
from `src/lib/lipreading/__fixtures__/engine/make_fixture.py`).

Note: `pnpm build` copies `public/` (model included, ~875 MB) into `dist/`, so `pnpm preview`
works offline; host the model elsewhere before deploying `dist/` to a static host.

## shadcn/ui

Add components with `pnpm dlx shadcn@latest add button`; they land in `src/components`.
