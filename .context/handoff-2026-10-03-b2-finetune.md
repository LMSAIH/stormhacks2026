# Handoff — B2: fine-tune the lip reader on the team's faces (cloud session)

2026-10-03 ~18:45 PT · from Claude Code (Opus 5.5, local) · branch `ml/phase-b` @ master `19c5c1b` + this file

## Goal

Fine-tune the Auto-AVSR `LRS3_V_WER19.1` visual speech recognition model on short webcam clips of
the team, so the demo reads our faces better — **without** hurting faces it has never seen (judges
will try it). Ship the result only if it passes the gates below; the current model stays the
fallback. Hard stop: a shippable result by **2026-10-04 09:00 PT** (submission is 12:00 PT), else
abandon and report.

## State (all verified, all on master)

- App: hold Space → crop the mouth in the browser (exact TS port of `ml/src/lipread/preprocess.py`)
  → **speed** = int8 ONNX greedy CTC in the browser (onnxruntime-web WASM), model loaded from HF
  `eschmechel/auto-avsr-lrs3-vsr-int8-onnx` at a pinned commit; **accuracy** = our FastAPI
  `POST /lipread/crops` (PyTorch, beam 40 + RNN LM) on a RunPod pod, falls back to speed.
- Baseline (stock 19.1, first 100 clips of HF `mattymchen/lrs3-test`, pre-made crops): greedy 28.6%
  WER, beam+LM 21.7–22.6%. Our 20 raw-face eval clips (`ml/data/raw_eval/`, local only): greedy 26.2%.
- int8 export (`quantize_onnx.py --variant dyn-pw8-rn16`, 203 MB) is locked by
  `ml/scripts/regress_quantized.py` + `ml/tests/quantized_baseline.json` (sha `da02d72e…`).
- `./smoke.sh` 6/6 on master. Weights convert losslessly between Chaplin's inference layout and
  auto_avsr's training layout: `ml/scripts/convert_ckpt.py {to-auto-avsr|to-chaplin} SRC DST --verify`.
- **NOT done: no team recordings exist yet.** The user is collecting them in parallel (format below).

## Decisions that bind this work (full log: `.context/project-brief.md` §2)

- D28: fine-tune **from 19.1** (convert with `convert_ckpt.py to-auto-avsr`), using the vendored
  auto_avsr recipe (`ml/third_party/auto_avsr/train.py`), not auto_avsr's 20.3% zoo checkpoint.
- D36/§4: training crops must come from **our** `lipread.preprocess.MouthCropper` (short-range
  BlazeFace first, ±6 smoothing, similarity warp, 96×96 gray), because that is exactly what the
  browser and server feed the model at inference. Do not use auto_avsr's retinaface detector.
- D32: RunPod **secure-cloud** RTX 4090 only (community 4090s had `cuInit()=999`).
- D25: no LLM corrector. D42: scope is `ml/` (+ the model pin in `frontend/`) — never touch
  `backend/` or the `origin/backend` / `origin/socket-setup` branches.
- D44: a teammate owns the capture-rate fix in `frontend/src/hooks/useLipReader.ts` — don't edit it.
- Never push to `master`; never commit checkpoints, datasets, recordings or secrets.

## Next steps

### Phase 1 — pipeline, no recordings needed (start here)
1. Read `AGENTS.md`, this file, brief §4 (preprocessing), §11 (baseline/pod), `ml/README.md`,
   `ml/third_party/auto_avsr/{README,INSTRUCTION}.md` (custom dataset `cstm` layout + label CSV).
   Branch `ml/b2-finetune` from `ml/phase-b`.
2. Get a GPU: pod `jr602gal8ql6c0` (secure RTX 4090, Romania, $0.74/h, **stopped**; its volume
   `/workspace` holds the repo, uv env and checkpoints). REST API `https://rest.runpod.io/v1`,
   header `Authorization: Bearer $RUNPOD_API_KEY`: `GET /pods/{id}?includeMachine=true`,
   `POST /pods/{id}/start`, `POST /pods/{id}/stop`. The pod only trusts the user's local SSH key:
   generate your own keypair and set it as the pod env `PUBLIC_KEY` (`PATCH /pods/{id}` with
   `{"env": {...keep existing..., "PUBLIC_KEY": "<pub>"}}`, then start). SSH = the pod's public IP +
   mapped port 22 from `GET /pods/{id}`. If that pod can't start (GPU gone), create one new
   secure-cloud RTX 4090 pod and run `BRANCH=ml/b2-finetune bash ml/runpod/bootstrap.sh`
   (its default `BRANCH=ml/model-pipeline` is stale). On the pod: `git fetch && git checkout
   ml/b2-finetune`, `SMOKE_REQUIRE_CUDA=1 ./smoke.sh ml` must pass first.
3. Write `ml/scripts/prepare_finetune_data.py`: input dir of `<speaker>_<nnn>.mp4` +
   `<speaker>_<nnn>.txt` → `MouthCropper` → 96×96 gray mp4 at 25 fps under `ROOT/cstm/…` + the
   auto_avsr label CSV (uppercase text, tokenized with the stock `spm/unigram/unigram5000.model`).
   Skip clips failing the 50% face-coverage gate; log them. Split by **speaker**: hold out one whole
   speaker as `test` (unseen-face check) and ~10% of each other speaker's clips as `val`.
4. Write `ml/scripts/finetune.sh` (or `.py`): convert 19.1 → auto_avsr layout, run auto_avsr
   `train.py --modality video --pretrained-model-path …` with fine-tune settings (start: lr 1e-4,
   1 warmup epoch, ≤15 epochs, keep best by val WER; consider freezing the 3D/ResNet front-end if
   it overfits), average the best checkpoints (`average_checkpoints.py`), convert back
   `to-chaplin --verify`.
5. Prove the loop end to end **before** recordings arrive: run steps 3–4 on 10 clips you make from
   `ml/data/smoke/face.mp4`-style material or a handful of `mattymchen/lrs3-test` clips (eval
   only — never claim test-set gains), 1–2 epochs. Done = trains, checkpoints, converts back,
   `lipread transcribe` runs on the result.
6. Stop the pod while waiting. Push the branch.

### Phase 2 — when the recordings dataset exists
7. Recordings live in a **private** HF dataset (the user will name it; default
   `eschmechel/stormhacks-lipread-recordings`). Format the team was given: frontal face,
   good light, 25–30 fps, 1–8 s, one sentence per clip, `<speaker>_<nnn>.mp4` + `.txt` with the exact
   words; ~50+ clips per speaker, demo phrases included.
8. Prepare → fine-tune → evaluate (greedy and beam+LM) on: held-out speaker, `val`, and LRS3-100
   (`ml/scripts/bench.py`). **Gates** to ship: held-out-speaker greedy WER improves ≥ 3 points vs
   stock 19.1 on the same clips, AND LRS3-100 greedy WER regresses ≤ +2.0 points (unseen faces).
9. If it passes: `export_onnx.py` → `quantize_onnx.py --variant dyn-pw8-rn16` →
   `regress_quantized.py` (all gates) → `--update-baseline` → upload the int8 file + tokens to the HF
   model repo **on a new HF branch** (e.g. `finetuned-v1`, not `main`) → in `frontend/src/lib/lipreading/modelSpec.ts`
   change the pinned commit to the new upload's commit → smoke → open a PR to master (**don't merge**;
   the user merges). Also put the fine-tuned PyTorch weights on the pod volume and document how to
   serve them (`LIPREAD_MODEL` / `LIPREAD_CKPT_DIR` in `ml/src/lipread/`).
10. Write results to `.context/b2-report.md` (numbers table: stock vs fine-tuned, per set, greedy
    and beam; spend; what was tried) and update the brief (next free numbers Q21, D45).

## Verify (prove the starting state before building on it)
- `./smoke.sh ml` locally (CPU, SKIPs needing local data are fine); on the pod
  `SMOKE_REQUIRE_CUDA=1 ./smoke.sh ml` → `smoke: N/N`.
- `uv run --directory ml python scripts/convert_ckpt.py to-auto-avsr <19.1 .pth> /tmp/x.pth --verify`.

## Gotchas
- `ml/.env` (RUNPOD_API_KEY) and `ml/checkpoints/` are local-only and gitignored; in the cloud the
  keys come from environment variables. **Never print, log or commit a key.**
- Chaplin espnet expects `(B,C,T,H,W)`; auto_avsr expects `(B,T,C,H,W)` — that's what `convert_ckpt.py`
  handles; don't hand-edit state dicts.
- auto_avsr's DataModule was patched to skip the removed `babble_noise.wav` for video (see
  `ml/third_party/ATTRIBUTION.md`).
- `uv run --directory ml …`, never `--project ml --directory ml` (resolves `ml/ml`).
- Background processes on the pod: use `setsid … </dev/null &` (see `ml/runpod/serve.sh`) so SSH
  disconnects don't kill training. Log to `/workspace/b2/*.log`.
- Pretrained weights are research / non-commercial (LRS3 terms) — same for the fine-tuned model.
- Few hundred clips + a 250M-param model overfits fast: watch val WER per epoch; the held-out
  speaker and LRS3 gates exist to catch "great on us, worse on judges".

## Budget & guardrails
- **GPU budget: $15 total (edit before pasting)**, max 1 pod. Track `costPerHr` × runtime in the
  report; stop the pod whenever it's idle > 10 min and when done.
- If blocked > 15 min on one thing, write it in `.context/b2-report.md` and move on.

## Environment / dirty state
- Local session: nothing running; worktree clean; pod `jr602gal8ql6c0` stopped (volume billed).

## Key files
- `ml/scripts/convert_ckpt.py` — 19.1 ⇄ auto_avsr layout, `--verify`
- `ml/third_party/auto_avsr/train.py`, `eval.py`, `average_checkpoints.py`, `INSTRUCTION.md` — recipe
- `ml/src/lipread/preprocess.py` (`MouthCropper`), `video.py` — crops identical to inference
- `ml/scripts/bench.py` — WER/latency on LRS3-100 + raw clips; `export_onnx.py`, `quantize_onnx.py`,
  `regress_quantized.py` (+ `ml/tests/README.md`) — ship path
- `ml/runpod/{bootstrap,serve,bench_baseline}.sh` — pod setup / service / baseline
- `frontend/src/lib/lipreading/modelSpec.ts` — the HF pin the app loads
