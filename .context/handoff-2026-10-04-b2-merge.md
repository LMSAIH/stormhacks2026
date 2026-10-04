# Handoff: B2 (fine-tune on our faces), cloud session → local session

2026-10-04 ~02:15 PT · branch `ml/b2-finetune` (pushed, in sync) · submission 12:00 PT, B2 hard stop 09:00 PT

## Goal
Fine-tune Auto-AVSR `LRS3_V_WER19.1` on our own webcam clips so the demo reads the presenter better,
without hurting unseen faces (judges). Ship only if both gates pass (D74); the stock model stays the
fallback. Along the way: model-scored phrase snapping (D78) and the training-pairs → fine-tune link (D77).

## State
**Done and verified**
- Pipeline end to end on a 4090 pod (Phase 1, LRS3 clips): prep → fine-tune (bf16) → pick → convert
  back (max |Δlogp| 2.5e-05) → `lipread` greedy + beam on the result. `SMOKE_REQUIRE_CUDA=1` smoke 2/2.
- Recipe chosen on public data (GRID rehearsal, D73): numbers in `.context/b2-report.md`.
- Prep reads app training pairs (`--pairs`, D77): tested with synthetic 88/96 pairs.
- Model phrase scoring (D78): Python + TS, parity vs sentencepiece (665 texts) and torch ctc_loss; offline
  benchmark vs look-alike.
- ORT-web thread timing (D81).
- `./smoke.sh` 6/6 locally (cloud container, CPU) on `65f0711`; `ml/b2-finetune` merges cleanly into
  `origin/master` and `origin/cloud/ml-live` (`git merge-tree`, 2026-10-04 09:10 UTC).

**Done, not verified**
- Phase 2 on real team clips: never run (no recordings yet).
- One-speaker plan (D76): untested; the held-out "speaker" is the same person.
- `--pairs` on real app pairs (only synthetic ones).
- `frontend/bench/ort-threads` on the demo laptop (6/8 threads).

**In flight:** nothing. Both B2 pods are stopped (`5y2nctzw4mrlj9` 60 GB volume, `jr602gal8ql6c0`
50 GB volume, both still billed for storage). B2 GPU spend ≈ US$1.40. The serving pod `qa5oi7o7g46n4q`
is the local session's (Quality mode), untouched here.

## Decisions
D71–D81 are in `.context/project-brief.md` §2 (right after D44; D45–D70 are in `streaming-plan.md`).
Short form:
- D71 `finetune.py` wrapper, not auto_avsr `train.py` (SLURM/wandb/DDP, single-GPU crash, torch 2.8).
- D72 lossless `.npy` crops from our `MouthCropper`.
- D73 frozen BN + lr 1e-4 × 3 ep + WiSE-FT blend; plain fine-tuning forgot open speech (28.6% → 70.8%).
- D74 gates unchanged; choose α on the held-out set, LRS3-100 only checks.
- D75 recording format/scripts, private dataset separate from public pairs.
- D76 one-speaker fallback: `p1`+`p2` train, separate `p3` 001–040 session = `HOLDOUT=p3`.
- D77 training pairs → train only, typed+picked, 88 crops padded.
- D78 rank phrase snaps by model CTC margin ≥ −0.2, keep `snapAllowed`; wiring pending.
- D79 GRID = rehearsal only; listed speakers excluded.
- D80 pods over Jupyter from cloud; stopped pods may not restart; epoch ckpts auto-deleted.
- D81 ORT threads default = half the cores (max 4); try 6/8 on the laptop.
If master gains D71+ before this branch merges, renumber these upward (they're only referenced in
the brief, this file, and `b2-report.md`).

## Next steps
1. **Merge order:** `cloud/ml-live` → master first (your fixes), then `ml/b2-finetune` (merge, don't rebase).
   Expect no conflicts; re-run `./smoke.sh` after (6/6 expected).
2. **Record** (user): `.context/b2-recording-guide.md` + `.context/b2-scripts/`. One person: `p1`+`p2`
   (160 clips) in one session, `p3` 001–040 in another (other time/light). Upload to a **private** HF
   dataset (default `eschmechel/stormhacks-lipread-recordings`); first ~10 clips early for a prep check:
   `uv run --directory ml python scripts/prepare_finetune_data.py prepare <dir> /tmp/check --holdout-speaker p3`
   (CPU, no GPU; look at skips, frame counts, `prep_report.json`).
3. **Pod:** start `5y2nctzw4mrlj9`; if "not enough free GPUs", create one secure-cloud RTX 4090 (this
   makes 2 running with the serving pod = the D65 limit). On it:
   `BRANCH=ml/b2-finetune EXTRAS="export dev train" bash ml/runpod/bootstrap.sh`
   (a restart wipes the container disk: re-run bootstrap; only `/workspace` survives).
4. **Data on the pod:** `hf download eschmechel/stormhacks-lipread-recordings --repo-type dataset
   --local-dir /workspace/b2/recordings` (needs `HF_TOKEN` in the pod env for a private dataset; never
   print it).
5. **Run:** `PHASE=2 CLIPS=/workspace/b2/recordings HOLDOUT=p3 NAME=FT_v1 setsid bash
   ml/runpod/b2_finetune.sh > /workspace/b2/phase2.log 2>&1 < /dev/null &` (add `PAIRS=<dir>` for app
   pairs). ~30–45 min: prep, fine-tune, blends α 0.25/0.35/0.4/0.5, bench greedy+beam on LRS3-100
   and the held-out clips → `/workspace/b2/bench/*.json`.
6. **Gate:** pick the α with the best held-out greedy WER; ship only if held-out greedy improves ≥ 3 pts
   vs `LRS3_V_WER19.1` AND LRS3-100 greedy ≤ 30.6% (28.6 + 2.0). Report demo phrases (`p3_001–012`)
   and the 28 unseen sentences separately.
7. **If it passes** (do steps a–c where `ml/data/raw_eval` exists, i.e. the laptop; copy the chosen
   `checkpoints/FT_v1_aX/` there):
   a. `uv run python scripts/export_onnx.py --model FT_v1_aX` (diff logits vs PyTorch, as always)
   b. `uv run python scripts/quantize_onnx.py --variant dyn-pw8-rn16`
   c. `uv run python scripts/regress_quantized.py` (all gates) → `--update-baseline`
   d. upload int8 + `tokens.json` to HF `eschmechel/auto-avsr-lrs3-vsr-int8-onnx` on a **new branch**
      (`finetuned-v1`), pin that commit in `frontend/src/lib/lipreading/modelSpec.ts`
   e. Quality mode: put `FT_v1_aX/` in the serving pod's `checkpoints/`, `LIPREAD_MODEL=FT_v1_aX`
   f. `./smoke.sh`, then a PR to master (the user merges). Results → `.context/b2-report.md`.
8. **If it fails:** stock stays; write the numbers in `b2-report.md`.
9. **Phrase scoring wiring (D78)**, owner of `useLipReader.ts`: recipe in `.context/phrase-scoring.md`
   (keep log-probs on the result, `rankByModel` → `modelSnap`, `snapAllowed` as guard, re-run the app eval).

## Gotchas
- `smoke.sh`'s `uv sync --extra export --extra dev` is exact: it **uninstalls `train`** (pytorch-lightning).
  `b2_finetune.sh` re-syncs it; anything else training must `uv sync --extra train` first.
- Commands run through Jupyter inherit `MPLBACKEND=module://matplotlib_inline…`, which breaks mediapipe
  (`jupyter_exec.py` strips it). `pkill -f <pattern>` inside a remote `bash -c` kills itself.
- Fine-tune data: speaker id = text before the last `_`; lowercase letters, digits, hyphens; `pairs` is
  reserved. One flat folder.
- GRID-style beam is worse than greedy on fixed-grammar clips (the LM pulls to open English); expect
  beam to help on natural sentences only.
- `finetune.py` deletes its epoch `.ckpt` files after export (`--keep-ckpts` to keep them); each is 1 GB.
- `regress_quantized.py` needs `ml/data/raw_eval` (local only) → quantize + relock on the laptop.
- Don't touch `backend/`, `origin/backend`, `origin/socket-setup`; `useLipReader.ts` capture loop is D44's
  owner's. Never push to master; never commit checkpoints, recordings or tokens.

## Key files
- `ml/scripts/prepare_finetune_data.py`: clips (+ `--pairs`) → cstm layout, speaker splits
- `ml/scripts/finetune.py`: fine-tune, pick, convert back, stock vs fine-tuned WER
- `ml/scripts/interpolate_ckpt.py`: WiSE-FT blends
- `ml/runpod/b2_finetune.sh`: pod job (PHASE=1|2); `b2_grid.sh`, `b2_sweep.sh`: the GRID rehearsal
- `ml/runpod/jupyter_exec.py`: run commands on a pod over HTTPS (no SSH)
- `ml/scripts/fetch_grid.py`: GRID download (rehearsal only)
- `ml/src/lipread/phrases.py`, `frontend/src/lib/phrases/ctcScore.ts`: model phrase scoring;
  `ml/scripts/bench_phrase_snap.py`, `export_phrase_assets.py`
- `frontend/bench/ort-threads/`: thread timing page
- `.context/b2-report.md` (all numbers), `b2-recording-guide.md`, `b2-scripts/`, `phrase-scoring.md`
