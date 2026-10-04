# App-level regression: 20 real faces through `/app`

The model benchmarks (`scripts/bench.py`, `regress_quantized.py`) never touch the app's capture,
sentence cutting or phrase snapping. This plays the 20 `data/raw_eval` clips through the real `/app`
in headless Chromium (fake camera) and scores the whole transcript against the references, so the
app's words-wrong rate can be compared with the model's own on the same clips (25.4% greedy,
29.5% beam). Results so far: `.context/app-eval.md`.

## Gate
`./smoke.sh app` (opt-in, ~5 min; not part of `./smoke.sh`) runs `gate.py`: its own `pnpm dev`
(no sign-in, model from `frontend/public/models`, browser phrase store), one Normal and one Instant
run, and fails when a mode is more than 5 points worse than the `Gate` line in
`.context/app-eval.md` (a run over the line is repeated; the mean of the two decides). Update that
line when an intended change moves the numbers. Needs the model in `frontend/public/models`,
`data/raw_eval`, playwright-core (`PLAYWRIGHT_CORE=…/index.mjs`) and Chromium (`CHROME=…`);
it says what's missing.

## By hand
```
cd ml
uv run python scripts/app_eval/make_eval_video.py          # artifacts/app_eval/eval20.y4m (once)
# frontend/.env.local: VITE_SKIP_AUTH=1, VITE_LIPREAD_MODEL_BASE=/models (fixed load time),
#                      VITE_LIPREAD_URL=<pod> for Quality; then `pnpm dev --port 5300`
MODE=normal TAG=mytag PLAYWRIGHT_CORE=.../playwright-core/index.mjs CHROME=.../chrome \
  node scripts/app_eval/e2e_eval.mjs                          # ~2.5 min per run
uv run python scripts/app_eval/score_eval.py mytag ...
uv run python scripts/app_eval/cuts.py mytag                  # where it cut, what it read, per clip
uv run python scripts/app_eval/faces.py --normal tagA tagB --instant tagC tagD
                                       # words lost per speaker vs face size, light, contrast, motion;
                                       # artifacts/app_eval/faces/sheet.png = their mouth crops
```

Serve the model locally (`VITE_LIPREAD_MODEL_BASE=/models`): a fresh profile otherwise downloads
203 MB from Hugging Face each run, and the varying start time drops early clips at random.
Runs differ by ~2-3 points on the same code (122 words); trust differences well above that, and
run each config at least twice. `e2e_eval.mjs` also saves the app's dev trace
(`trace_<tag>.json`: tracker results, cuts, reads) and the lip tracker's rate and lag (~20/s and
~40 ms here; on software GL with the GPU delegate it was 4/s and 0.7 s): compare runs at similar
tracker rates.

Quality: check `server reads` in the output (also `server` in the JSON). Zero means the app fell
back to on-device reads, e.g. a headless browser behind a TLS-intercepting proxy whose CA it
doesn't trust (add the CA: `certutil -d sql:$HOME/.pki/nssdb -A -t "C,," -n proxy -i <ca.crt>`).
`DUMP=1` also saves each upload (the app's own sentence cuts) to `artifacts/app_eval/crops_<TAG>/`;
`replay_crops.py` decodes them at other beam settings, so settings compare on the same cuts:
```
uv run python scripts/app_eval/replay_crops.py artifacts/app_eval/crops_<TAG> --settings 20,0.1,0.2,0 40,0.1,0.3,0
```
