# Lipreader frontend

Vite + React 19 + TypeScript. **pnpm only** (`pnpm install`, `pnpm dev`, `pnpm lint`,
`pnpm typecheck`, `pnpm test`, `pnpm build`).

## Lip reading (push-to-talk)

Hold **Space** (or the talk button), mouth a sentence, release. The utterance is cropped in the
browser (exact port of the Python preprocessing, `src/lib/lipreading/crop/`) and recognized by the
mode picked in the header:

- **Speed** — on-device ONNX (onnxruntime-web, WASM; WebGPU is opt-in via `VITE_ORT_WEBGPU=1`),
  greedy CTC. Works out of the box: the int8 model (203 MB; the fp32 export is 775 MB) loads from
  [eschmechel/auto-avsr-lrs3-vsr-int8-onnx](https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx),
  pinned to a commit, and the browser keeps it in Cache Storage after the first download. Offline:
  `../ml/scripts/publish_frontend_model.sh` copies `lipread_ctc.int8.onnx` + `tokens.json` into
  `public/models/` (gitignored); then set `VITE_LIPREAD_MODEL_BASE=/models` in `.env.local`.
- **Accuracy** — the hosted service (beam search + LM): set `VITE_LIPREAD_URL` in `.env.local`
  (see `.env.example`). Falls back to Speed if the service fails.

Sign-in needs the backend on `localhost:5000`. To look at the app without it, put `VITE_SKIP_AUTH=1`
in `.env.local`: `pnpm dev` then skips the login gate (ignored in `pnpm build`).

`/lab` runs the whole pipeline deterministically on `public/test/clip.mp4` and publishes every
intermediate result on `window.__lipLab` (used by Playwright checks).

Tests: `pnpm test` (unit + crop parity vs Python). With the model published,
`LIPREAD_ONNX_TEST=1 pnpm test` also runs the real-model golden test: onnxruntime-web on the int8
model must decode the fixture clip exactly like native onnxruntime on the same file (needs the
gitignored fixture from `src/lib/lipreading/__fixtures__/engine/make_fixture.py`; its docstring
says how `expected.json` is produced and how to refresh it after re-quantizing).

Note: `pnpm build` copies `public/` (model included, ~350 MB) into `dist/`, so `pnpm preview`
works offline; host the model elsewhere before deploying `dist/` to a static host.

## Docker

Build and run the production frontend from this directory:

```sh
docker build -t lipreader-frontend .
docker run --rm -p 5173:80 lipreader-frontend
```

Open `http://localhost:5173`. Vite settings are baked into the image at build time; override
`VITE_API_URL`, `VITE_TTS_WS_URL`, `VITE_LIPREAD_URL`, `VITE_LIPREAD_MODEL_BASE`, or
`VITE_ORT_WEBGPU` with `--build-arg` when building for a different deployment.

## shadcn/ui

Add components with `pnpm dlx shadcn@latest add button`; they land in `src/components`.
