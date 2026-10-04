# App-level regression: 20 real faces through `/app`

The model benchmarks (`scripts/bench.py`, `regress_quantized.py`) never touch the app's capture,
sentence cutting or phrase snapping. This plays the 20 `data/raw_eval` clips through the real `/app`
in headless Chromium (fake camera) and scores the whole transcript against the references, so the
app's words-wrong rate can be compared with the model's own on the same clips (25.4% greedy,
29.5% beam). Results so far: `.context/app-eval.md`.

```
cd ml
uv run python scripts/app_eval/make_eval_video.py          # artifacts/app_eval/eval20.y4m (once)
# frontend/.env.local: VITE_SKIP_AUTH=1, VITE_LIPREAD_MODEL_BASE=/models (fixed load time),
#                      VITE_LIPREAD_URL=<pod> for Quality; then `pnpm dev --port 5300`
MODE=normal TAG=mytag PLAYWRIGHT_CORE=.../playwright-core/index.mjs CHROME=.../chrome \
  node scripts/app_eval/e2e_eval.mjs                          # ~2.5 min per run
uv run python scripts/app_eval/score_eval.py mytag ...
```

Serve the model locally (`VITE_LIPREAD_MODEL_BASE=/models`): a fresh profile otherwise downloads
203 MB from Hugging Face each run, and the varying start time drops early clips at random.
Runs differ by ~2 points on the same code (122 words); trust differences well above that.
