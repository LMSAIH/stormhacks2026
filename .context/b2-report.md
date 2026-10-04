# B2 report: fine-tune on the team's faces

Status 2026-10-04 ~02:30 UTC (2026-10-03 ~19:30 PT) · branch `ml/b2-finetune` (from `ml/phase-b`)

## Where it stands

- **Phase 1 (pipeline) proven on CPU, not yet on GPU.** Prep → fine-tune (2 epochs) → checkpoint
  pick/average → convert back to the served layout (logit check) → `lipread` loads and decodes it
  via `LIPREAD_MODEL`. All ran in the cloud session on 12 LRS3-test clips (idx 600–611, not the
  LRS3-100 gate set) split into 3 fake speakers.
- **Blocked on GPU:** the branch can't be pushed (the Claude GitHub App has no access to
  `LMSAIH/stormhacks2026`), so a pod can't clone it. Copying the code to a pod by other means was
  refused by this session's permissions. The old pod `jr602gal8ql6c0` can't start anyway: its host
  has no free GPU.
- **Phase 2 (real recordings)**: not started, no dataset yet.

## Verify (handoff checklist)

| Check | Result |
|---|---|
| `./smoke.sh ml` (cloud CPU, before changes) | `smoke: 2/2` (checks: 5 pass, 4 skip: no checkpoint/face clip/ONNX locally) |
| `./smoke.sh ml` (after changes, checkpoint downloaded) | `smoke: 2/2` (checks: 6 pass, 3 skip) |
| `convert_ckpt.py to-auto-avsr 19.1 --verify` | max\|Δlogp\| 9.5e-06, argmax agreement 100% ✓ |
| Pod `jr602gal8ql6c0` | matches handoff (stopped, secure 4090, RO, $0.74/h), but **start fails: "not enough free GPUs on the host machine"** |
| `SMOKE_REQUIRE_CUDA=1 ./smoke.sh ml` on a pod | not run (no pod, see above) |

Mismatches found: `ml/README.md` listed a `train/` dir that doesn't exist (removed); vendored
`cosine.WarmupCosineScheduler` passes `verbose`, which torch 2.8 removed (finetune.py uses its own
LambdaLR); stock `train.py` needs SLURM_JOB_ID + wandb + DDP and its `training_step` crashes on one
device (`all_gather` → 0-dim tensor). Hence a small wrapper instead of `train.py`.

## What was built

- `ml/scripts/prepare_finetune_data.py`: `prepare` (clips → `MouthCropper` @25 fps → `(T,96,96)`
  uint8 `.npy` crops in `cstm/`, LRS3-style uppercase labels, unigram5000 ids, label CSVs, one whole
  speaker → `test`, ≥1 and ~10% per other speaker → `val`, coverage-gate skips logged in
  `prep_report.json`); `dump-lrs3` for pipeline tests. Crops are stored lossless rather than as mp4
  (auto_avsr default) so training sees the same pixels the live crop produces, not codec noise.
- `ml/scripts/finetune.py`: auto_avsr `ModelModule` + greedy CTC val WER per epoch, best-k by val
  WER (weights-only ckpts), average vs best (lower val WER wins), `to-chaplin` + verify, stock vs
  fine-tuned WER on val/test → `exp/<name>/summary.json`. Options: `--freeze-frontend`, `--lr`,
  `--epochs`, `--warmup-epochs`, `--max-frames`, `--precision`.
- `[stormhacks patch]` in `auto_avsr/datamodule/av_dataset.py`: load `.npy` crops.
- `LipReader` honours `LIPREAD_MODEL` → the CLI, `bench.py` and the service can all load `FT_*`.
- `ml/runpod/b2_finetune.sh` (pod job, PHASE=1|2, incl. bench stock vs fine-tuned),
  `ml/runpod/jupyter_exec.py` (commands over the pod's Jupyter when SSH egress is blocked — it is,
  in cloud sessions), `bootstrap.sh` `EXTRAS=` (needs `train` for fine-tuning).

## Phase 1 run (CPU, numbers are meaningless: 6 train / 2 val / 4 test clips)

| | val greedy WER | test greedy WER |
|---|---|---|
| stock 19.1 | 0.385 | 0.423 |
| after 2 epochs (best ckpt; avg-of-2 scored 0.615 on val) | 0.462 | 0.269 |

1.1 min training on 4 CPU cores with `--max-frames 120`; a `--max-frames 400` run got OOM-killed
(container memory cap ~8 GB), so GPU memory at `--max-frames 1000` is untested.

## Spend

GPU: **$0.00** (no pod ran). The stopped pod's 50 GB volume is still billed by RunPod.

## Next (in order)

1. Fix GitHub access (install the Claude GitHub App on LMSAIH/stormhacks2026, or reconnect GitHub),
   then push `ml/b2-finetune`.
2. Create one secure-cloud RTX 4090 pod (runpod/pytorch image, ports 8888/http + 22/tcp, env
   `JUPYTER_PASSWORD`), then `BRANCH=ml/b2-finetune EXTRAS="export dev train" bash bootstrap.sh`,
   `SMOKE_REQUIRE_CUDA=1 ./smoke.sh ml`, `PHASE=1 bash ml/runpod/b2_finetune.sh` (~$0.10).
3. Phase 2 when the HF dataset exists: download to `/workspace/b2/recordings`,
   `PHASE=2 CLIPS=… HOLDOUT=<speaker> bash ml/runpod/b2_finetune.sh`, check gates (held-out
   greedy −3 pts, LRS3-100 greedy ≤ +2.0), then the ship path in the handoff.
