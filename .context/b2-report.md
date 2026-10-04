# B2 report: fine-tune on the team's faces

Status 2026-10-04 13:55 UTC (06:55 PT) · branch `ml/b2-phase2` · **Phase 2 shipped `FT_v1_a0.5` to speed
mode** ("Phase 2 results" at the end). Earlier status: 2026-10-04 02:21 UTC, branch `ml/b2-finetune`.

## Where it stands

- **Phase 1 done on GPU.** On pod `5y2nctzw4mrlj9` (new secure RTX 4090, $0.74/h; the old pod
  `jr602gal8ql6c0` can't start because its host has no free GPU): bootstrap → `SMOKE_REQUIRE_CUDA=1
  ./smoke.sh ml` 2/2 → `PHASE=1 bash ml/runpod/b2_finetune.sh` → exit 0. Prep, 2 epochs in bf16,
  checkpoint pick, convert back (max|Δlogp| 2.5e-05), `lipread` greedy + beam+LM on the result.
- **Memory checked for Phase 2:** 120 LRS3 clips (idx 200–319), 4 batches each at `--max-frames`
  1000 and 1600, bf16 → both fine on 24 GB.
- Pod **stopped** 02:20 UTC; its 60 GB volume keeps the env, checkpoints and `/workspace/b2`.
- **Phase 2 (real recordings)**: not started, no dataset yet.

## Verify (handoff checklist)

| Check | Result |
|---|---|
| `./smoke.sh ml` (cloud CPU, before changes) | `smoke: 2/2` (checks: 5 pass, 4 skip: no checkpoint/face clip/ONNX locally) |
| `./smoke.sh ml` (after changes, checkpoint downloaded) | `smoke: 2/2` (checks: 6 pass, 3 skip) |
| `convert_ckpt.py to-auto-avsr 19.1 --verify` | max\|Δlogp\| 9.5e-06, argmax agreement 100% ✓ |
| Pod `jr602gal8ql6c0` | matches handoff (stopped, secure 4090, RO, $0.74/h), but **start fails: "not enough free GPUs on the host machine"** |
| `SMOKE_REQUIRE_CUDA=1 ./smoke.sh ml` on new pod | `smoke: 2/2` (checks: 6 pass, 3 skip) after fixing the Jupyter env leak below |

Mismatches found: `ml/README.md` listed a `train/` dir that doesn't exist (removed); vendored
`cosine.WarmupCosineScheduler` passes `verbose`, which torch 2.8 removed (finetune.py uses its own
LambdaLR); stock `train.py` needs SLURM_JOB_ID + wandb + DDP and its `training_step` crashes on one
device (`all_gather` → 0-dim tensor). Hence a small wrapper instead of `train.py`.
Found on the pod: `smoke.sh`'s `uv sync --extra export --extra dev` is exact and uninstalls the
`train` extra (the job now re-syncs it); commands run via Jupyter inherited
`MPLBACKEND=module://matplotlib_inline…`, which breaks mediapipe (`jupyter_exec.py` now drops it).

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

## Phase 1 run (GPU, pod `5y2nctzw4mrlj9`; again meaningless at 6/2/4 clips)

| | val greedy WER | test greedy WER |
|---|---|---|
| stock 19.1 | 0.385 | 0.423 |
| FT_phase1, 2 epochs bf16 (best = avg2 on val) | 0.385 | 0.385 |

Training 0.3 min. Same train clip, stock vs FT: greedy "AND BUT YOU KNOW THIS IS" vs "AND BUT YOU
KNOW WHAT THIS IS"; beam identical (ref: "AND IF IT'S COMPLETELY OKAY WHAT'S THE JOKE").

## Spend

| Item | Time | Cost |
|---|---|---|
| Pod `5y2nctzw4mrlj9` (secure 4090, $0.74/h) | 02:10–02:20 UTC, ~11 min | ~US$0.13 |
| Pod `5y2nctzw4mrlj9` again | 03:09–04:51 UTC, ~1.7 h | ~US$1.27 |
| **Total B2 GPU** | | **~US$1.40 of $15**; both pods stopped 04:51 UTC |

Volumes of both stopped pods (50 GB old, 60 GB new) are still billed while stopped.

## Next (in order)

1. Phase 2 when the HF dataset exists: download to `/workspace/b2/recordings`,
   `PHASE=2 CLIPS=… HOLDOUT=<speaker> bash ml/runpod/b2_finetune.sh`, check gates (held-out
   greedy −3 pts, LRS3-100 greedy ≤ +2.0), then the ship path in the handoff.

## Public data for a pre-recording rehearsal (2026-10-04 ~02:50 UTC)

Asked to source extra data/models and show a gain without regression before the team clips
arrive. What exists:

| Source | What it is | Usable? |
|---|---|---|
| **GRID** (Zenodo 3625687, CC BY 4.0) | 34 speakers × 1000 frontal 3 s clips, 25 fps, raw face video + word alignments | **Yes**: raw video goes through our MouthCropper (checked: 12/12 clips crop cleanly). Six-slot grammar ("SET WHITE AT I ONE SOON"), so a GRID-tuned model must not ship; it's a rehearsal |
| `bekalemu/grid-corpus-*` (HF) | GRID as 128×128 mouth ROIs | No: crop differs from our similarity-warp 96 crop |
| `wissemkarous/lipreading` (HF) | GRID speaker s1 only (LipNet tutorial) | Subset of the above |
| `TheNHz/ellipsis-lrs3-raw` (HF) | raw LRS3 video | Gated: needs your OK to accept its terms |
| LRS2 / LRS3 full | BBC licence | Gated, sign-up |
| `mattymchen/lrs3-test` idx 100–660 | LRS3 test crops | No for training: same pool as the LRS3-100 gate, would contaminate it |
| `DataoceanAI/...` | commercial corpus sample | No (paid) |
| Images | — | No: the model reads lip motion over time; stills carry no signal |
| Other VSR models (HF search) | LipNet/GRID toys, Thai/Korean models, mpc001 multilingual VSR | None beats 19.1 on English open speech; USR 2.0 stays the stretch option |

Stock 19.1 on 12 GRID s1 clips through our crop: **greedy WER 66.7%** (e.g. "SET RED IN H ZERO NOW"
→ "STAY WHERE DID H ZERO NOW").

Built: `scripts/fetch_grid.py`, `prepare_finetune_data.py --workers` (parallel cropping), comma-
separated `--holdout-speaker`, `.mpg` input, `runpod/b2_grid.sh` (train s1,s2,s3,s4,s7,s11 ×200
clips; hold out s12,s15; two recipes: lr 1e-4 full / lr 5e-5 frozen front-end; bench stock vs both
on 100 held-out GRID clips + LRS3-100, greedy + beam).

Ran on pod `5y2nctzw4mrlj9` once its host freed a GPU (03:09 UTC). Speakers s5, s6, s10, s11, s12,
s13, s15 were dropped: their Zenodo alignment files carry shifted utterance ids (0/1000 match the
videos); `s14.zip` came back corrupt. 20 s8 clips decode to 2–9 frames and were skipped by prep.

## GRID rehearsal results (2026-10-04 03:45–04:42 UTC)

Train s1,s2,s3,s4,s7,s8 (1062 clips, 53 min), val 118, held out s9,s16 (unseen faces). WER on
LRS3-100 (open speech = the regression gate) and 100 held-out GRID clips. n=100 each, so ±2–3 pts noise.

| Model | LRS3-100 greedy | LRS3-100 beam | GRID held-out greedy | GRID held-out beam |
|---|---|---|---|---|
| stock 19.1 | 28.6% | 22.6% | 73.3% | 85.3% |
| FT lr 1e-4, 5 ep | 70.8% | 62.6% | 10.2% | 7.5% |
| FT lr 5e-5, 5 ep, frozen front-end | 52.3% | 44.4% | 17.2% | 10.3% |
| FT lr 1e-4, 5 ep → WiSE α 0.1 / 0.2 / 0.3 / 0.5 | 29.5 / 31.3 / 31.8 / 34.9% | | 67.7 / 56.8 / 45.2 / 23.2% | |
| FT **frozen BN** lr 1e-4, 3 ep | 38.0% | | 9.0% | |
| FT frozen BN lr 3e-5, 3 ep / lr 1e-5, 2 ep | 35.3 / 31.7% | | 16.3 / 46.7% | |
| frozen BN lr 1e-4 → WiSE α 0.15 / 0.25 / 0.35 / 0.5 | 29.0 / 29.7 / 29.5 / 30.9% | | 61.7 / 48.5 / 37.8 / 23.8% | |
| frozen BN lr 1e-4 → WiSE α 0.35 | 29.5% (+0.9) | 23.8% (+1.2) | 37.8% | 46.2% |
| **frozen BN lr 1e-4 → WiSE α 0.4** | **29.7% (+1.1)** | **24.4% (+1.8)** | **32.7% (−40.6)** | **39.2% (−46.1)** |

What it shows:
- The handoff recipe (lr 1e-4, many epochs) **forgets open speech badly** (LRS3-100 greedy +42 pts).
  Phase 2 with it would have failed the gate.
- **BatchNorm running stats are a big part of it.** Even lr 3e-6 for one epoch (unfrozen BN) made
  GRID val *worse* than stock (0.565 → 0.83); keeping 19.1's BN stats (`--freeze-bn`) halves the
  LRS3 damage at the same lr and fits GRID better.
- **WiSE-FT** (blend fine-tuned with stock weights, `scripts/interpolate_ckpt.py`) recovers open
  speech while keeping much of the gain. α 0.35–0.4 passes both ship gates on this data.
- Caveats: GRID's gain is largely its fixed grammar, not faces, so expect a much smaller gain on
  team clips. α was picked by looking at LRS3-100, the gate set itself, so the +1.1 is slightly
  optimistic; for Phase 2 choose α from the blends on the held-out speaker and accept only if
  LRS3-100 stays ≤ +2. GRID beam is worse than greedy because the LM is open-speech English.
- GRID-tuned weights are **not** for the demo (they bias towards GRID words).

Phase 2 defaults updated in `runpod/b2_finetune.sh`: `--freeze-bn`, 3 epochs, `--max-frames 1600`,
then WiSE blends α 0.25/0.35/0.4/0.5, each benched greedy + beam on LRS3-100 and the held-out speaker.

## Phase 2 session (2026-10-04 from 02:57 PT, branch `ml/b2-phase2` off master `1716556`)

**Blocked on recordings.** `eschmechel/stormhacks-lipread-recordings` does not exist (checked 09:59 UTC
with the owner's token; the account's only dataset is `stormhacks-lipread-eval`). No pod started,
spend US$0. A watcher polls the Hub every 2 min and picks up the dataset (or any new dataset) when it
appears. Everything below is prep so Phase 2 → gate → ship fits before 09:00 PT.

| Check | Result |
|---|---|
| `./smoke.sh` on master, cloud CPU | 6/6 (a first run was 5/6: a PyPI download timeout in `ml: sync`; `UV_HTTP_TIMEOUT=300` fixed it) |
| same, with the eval clips, stock checkpoint, fp32 export and int8 in place | 6/6, all 10 ml checks pass (export parity 100%, fast regression lock 7/7) |
| `export_onnx.py` (stock) in the cloud | logits vs PyTorch max\|Δ\| ≤ 1.4e-3, argmax 100% at T 37/73/151 |
| `quantize_onnx.py --variant dyn-pw8-rn16` in the cloud | 203.4 MB, sha `d4dbd28e…`, **not** the locked `da02d72e…`: the quantize output is not byte-identical across machines |
| `regress_quantized.py` on that file | all 11 quality gates pass (int8 vs fp32 +0.00 pts on LRS3-100 and raw-20, agreement 0.994 / 0.992); lock fails (sha, 3/15 texts) |
| same on the HF-pinned stock file (sha matches the lock) | 2/15 locked texts flip: this Xeon (AVX-512 VNNI, AMX) rounds int8 differently from the i9-13900H the lock was made on |
| `prepare_finetune_data.py` on CPU (6 renamed public eval clips, pipeline test only) | 6/6 kept, 0 skips, ~6 s with 4 workers |
| app eval harness in the cloud, stock int8 served from `/models` | Instant 40.2%, Normal 29.5%; Quality unusable on CPU (no beam request finished in the run: Quality needs a GPU server) |

What this means for shipping: quantize + quality gates run fine in the cloud, so the laptop is no
longer needed. A lock re-made here records Xeon texts, and on the laptop a few locked texts may flip
(the fast smoke subset did not flip for stock); re-lock there if they do.

Added: `scripts/b2_gate.py` (D74 decision from the bench JSONs: candidates ranked by held-out greedy,
first one passing both gates ships, demo phrases 001–012/041–052 vs unseen sentences reported apart);
`runpod/b2_finetune.sh` benches into `$B2/bench/$NAME` (the volume's `$B2/bench` still holds GRID JSONs
under the same stock tags), crops with 16 workers, and prints the gate at the end.

## Phase 2 results (2026-10-04 05:12–06:15 PT, pod `42dc1t20jc8whf`) — shipped `FT_v1_a0.5`

**Data.** `eschmechel/stormhacks-lipread-recordings` (private): p1 + p2 = the presenter, one session;
p3 = a teammate (all 80 lines), so `HOLDOUT=p3` measures an **unseen face**, not the presenter in a new
session (D76 assumed the latter). `scripts/check_labels.py` (CTC likelihood of every script line vs every
clip, Hungarian assignment) found two recording-order slips that `rename_clips.py` turned into shifted
labels: p2 line 007 was never recorded (p2_007…079 were lines 008…080) and p3 line 003 was never
recorded, with line 028 taken twice (p3_003…026 were lines 004…027; the first 028 take moved to
`extra/`). 222/239 clips' best line matched that explanation; the rest were near-identical demo wordings
or noise-level margins. Fixed in the dataset (commit `c697505`): 238 clips, 0 prep skips; train 143,
val 16 (presenter), held out 79 (teammate: 23 demo phrases, 56 unseen sentences).

**Run.** `PHASE=2 HOLDOUT=p3 NAME=FT_v1` (D73 recipe: frozen BN, lr 1e-4, 3 epochs, bf16; training
0.6 min on the 4090). The first run died silently right after training: `clip="$(ls … | head -1)"` returns 141
(SIGPIPE) under pipefail with 238 crops on the network volume (fixed in `afc8319`, reproduced
on the pod). Numbers below are from the clean rerun.

| model | LRS3-100 greedy | LRS3-100 beam | held-out greedy | held-out beam | demo greedy / beam | unseen greedy / beam | gain | gates |
|---|---|---|---|---|---|---|---|---|
| stock 19.1 | 28.6% | 22.6% | 57.3% | 48.2% | 41.9% / 25.8% | 62.1% / 55.1% | | |
| FT_v1 (α 1) | 30.9% | 26.3% | 46.7% | 34.7% | 30.6% / 17.7% | 51.6% / 39.9% | +10.7 | LRS3 ✗ |
| α 0.25 | 28.9% | 22.3% | 53.5% | 44.0% | 36.3% / 24.2% | 58.9% / 50.1% | +3.8 | pass |
| α 0.35 | 29.5% | 24.2% | 52.2% | 42.1% | 36.3% / 24.2% | 57.1% / 47.6% | +5.1 | pass |
| α 0.4 | 29.8% | 24.5% | 52.0% | 42.1% | 35.5% / 24.2% | 57.1% / 47.6% | +5.3 | pass |
| **α 0.5** | **30.0%** | 24.4% | **50.5%** | 42.9% | 33.9% / 24.2% | 55.6% / 48.6% | **+6.9** | **ship** |

Gate (D74, `scripts/b2_gate.py`): best held-out greedy among candidates that pass → **α 0.5**: held-out
greedy −6.9 pts on a face it never saw (≥ 3 needed), LRS3-100 greedy 30.0% (≤ 30.6%). Unblended FT_v1
gains more (−10.7) but forgets open speech just past the limit (30.9%). Trade-off: LRS3-100 **beam**
22.6% → 24.4% (Quality mode on strangers); the gate is greedy-only (D74). α 0.5 was the top of the
tested range; higher α was not tried (each extra look at the held-out set makes the gate more optimistic).

Presenter's own val clips (16, same session as training, so optimistic), greedy: stock 42.2%, α 0.25
37.6%, α 0.35 34.9%, α 0.4 33.9%, **α 0.5 29.4%**, FT_v1 26.6%. On the first 12 demo-phrase takes stock
already read 14.9%.

**Ship path** (cloud container, no laptop):
- `export_onnx.py --model FT_v1_a0.5`: logits vs PyTorch max|Δ| 1.1e-05 / 6.0e-05 / 1.6e-04 at T 37/73/151,
  argmax 100%.
- `quantize_onnx.py --variant dyn-pw8-rn16`: 203.4 MB, sha256 `55143d51…65979`.
- `regress_quantized.py`: all 11 quality gates pass (int8 vs fp32: LRS3-100 30.0 → 29.8%, raw-20
  27.9 → 27.0%, agreement 0.993 / 0.999, 250-frame input ok); `--update-baseline` locked it on the cloud
  Xeon (AVX-512 VNNI). On the i9 laptop a few locked texts may flip (see the Phase 2 session notes):
  re-lock there if smoke's fast lock fails.
- HF `eschmechel/auto-avsr-lrs3-vsr-int8-onnx` branch **`finetuned-v1`**, commit `397241eb` (int8,
  tokens.json, quantization.json, model card); `main` (stock, `9359b251`) unchanged. The repo is public,
  so the fine-tuned weights are public; the clips stay private. `modelSpec.ts` pins `397241eb`; the
  browser cache is keyed by URL, so clients fetch the new file once.
- PyTorch checkpoints (FT_v1 and the four blends) in the private HF repo
  `eschmechel/stormhacks-b2-checkpoints` (CLAUDE.md: checkpoints only on the pod volume or a private repo).
- `./smoke.sh` 6/6. Caveat found: smoke's "onnx export parity" compares `artifacts/lipread_ctc.onnx` with
  whatever checkpoint `LIPREAD_MODEL` names (stock by default) on random input, where both models output
  mostly blanks, so it also passed FT ONNX vs stock PyTorch (max|Δ| 6.7). With `LIPREAD_MODEL=FT_v1_a0.5`
  it is like for like: max|Δ| 9.5e-06.
- Quality mode (server beam) still serves stock from `qa5oi7o7g46n4q` (not touched). To switch: copy
  `FT_v1_a0.5/` from the private repo into the serving pod's `checkpoints/` and set `LIPREAD_MODEL=FT_v1_a0.5`
  (weigh the LRS3 beam +1.8 first).

**App eval** (`.context/app-eval.md`, two runs each, 20 strangers' clips): stock → fine-tuned Instant
30.3 → 30.0%, Normal 29.1 → 27.0%, Quality (GPU server) 25.8 → 28.7%. Speed mode: no measurable change on
unseen faces (same-model runs differ by up to 8 points). Quality would lose ~3 points, consistent with
the LRS3 beam result, so it stays stock.

**Spend.** Pod `42dc1t20jc8whf` (secure 4090, US$0.74/h, created because `5y2nctzw4mrlj9`'s host had no
free GPU) ran 10:51–13:51 UTC, 3.0 h ≈ US$2.21 (≈ CA$3.10). B2 GPU total ≈ US$3.61 (≈ CA$5.05).
Stopped; its 60 GB volume (env, data, bench JSONs) and the volumes of `5y2nctzw4mrlj9` (60 GB) and
`jr602gal8ql6c0` (50 GB) are still billed while stopped. Checkpoints are also in the private HF repo,
so the B2 pods can be terminated after the hackathon.

**Security slip.** A helper printed the RunPod stop response, which echoes the pod's env: the HF token
(write scope) and that pod's Jupyter password appeared in the session output. Rotate the HF token
(revoke it at huggingface.co/settings/tokens, update the cloud environment secret and any pod env).
The helper no longer prints response bodies.

## Follow-up: options 1–3 (2026-10-04 07:35–08:20 PT, pod `9kjrrcuvzrueu6`)

Asked: would full precision plus more training beat the quantized model; then implement the three
options unless #13/#14 covered them. They didn't (#14 tuned the stock server's beam to 20 / LM 0.2,
#13 built eval v2), but #13 suggested a third gate for B2, now in `b2_gate.py`: on eval v2
(`raw_eval_v2`, 144 clips of 62 unseen people) a candidate's greedy WER may be at most 2.0 pts worse
than stock, paired. Precision first: the pod already serves full-precision PyTorch; int8 costs
nothing measurable (eval v2 stock int8 41.3% vs fp32 40.7%, #13; α 0.5 int8 vs fp32 on LRS3-100
29.8 vs 30.0%).

`b2_gate.py` on the new run (`afc8319`+ code, beam = #14's deployed 20 / LM 0.2; greedy numbers
reproduce the Phase 2 run exactly):

| model | LRS3-100 greedy | LRS3-100 beam | teammate greedy | teammate beam | eval v2 greedy (Δ) | eval v2 beam (Δ) | gain | gates |
|---|---|---|---|---|---|---|---|---|
| stock 19.1 | 28.6% | 21.9% | 57.3% | 50.1% | 40.7% | 35.0% | | |
| α 0.5 (shipped) | 30.0% | 23.8% | 50.5% | 39.8% | 40.5% (−0.2) | 36.1% (+1.1) | +6.9 | pass |
| α 0.6 | 30.1% | 24.2% | 49.3% | 39.0% | 40.3% (−0.5) | — | +8.0 | pass |
| α 0.7 | 30.5% | 25.4% | 47.8% | 37.9% | 40.3% (−0.5) | — | +9.5 | gate pick |
| FT_all α 0.5 | 30.5% | 24.3% | trained on | | 39.7% (−1.0) | — | n/a | |
| FT_all α 0.7 | 30.1% | — | trained on | | 39.9% (−0.8) | — | n/a | |

Eval v2 intervals (speakers resampled): α 0.5 greedy −0.2 [−1.2, +1.0], beam +1.1 [−1.2, +4.0]
(VidTIMIT +3.8); α 0.7 −0.5 [−1.9, +1.2]; FT_all α 0.5 −1.0 [−2.2, +0.6].

- **Option 2 (higher α): not shipped.** α 0.7 wins the model gates and its int8 passes all 11
  regression quality gates (logits vs PyTorch ≤ 1.8e-4; LRS3-100 30.5 → 30.2%, raw-20 30.3 → 29.5%),
  but it **fails `./smoke.sh app`**: Normal 54.1% / 28.7% → mean 41.4% (limit 30.4%; the first run had
  the tracker at 12.6 Hz and 17/20 lines), Instant 30.3% ok. On those 20 strangers' clips the model alone
  reads 29.5% (int8) vs α 0.5 27.0% and stock 26.2%, so it sits at the gate's edge; not re-run until it
  passed. Speed mode keeps α 0.5, which passes every gate including this one.
- **Option 1 (Quality server): ready, not switched.** α 0.5 on the server: eval v2 beam +1.1 (within
  the +2 rule, wide interval), teammate beam 50.1 → 39.8%. `ml/runpod/serve.sh` now fetches a fine-tuned
  checkpoint from the private repo: on the serving pod,
  `git pull && HF_TOKEN=<token> LIPREAD_MODEL=FT_v1_a0.5 bash ml/runpod/serve.sh` (rollback: the same
  without `LIPREAD_MODEL`). Not done from here: that pod has no Jupyter and cloud sessions have no SSH,
  and a remote restart can lose its GPU (both B2 pods failed to restart today). Whoever demos decides:
  better for the team's faces, about 1 pt worse beam for strangers.
- **Option 3 (retrain on p1 + p2 + p3): not shipped.** `FT_all` (214 train / 24 val; val 40.0 → 22.5%)
  is the best on unseen faces (eval v2 −1.0 at α 0.5) and passes LRS3-100, but with the teammate in
  training nothing measures its gain on an unseen speaker, so it can't pass D74 as written. Checkpoints
  `FT_all`, `FT_all_a0.5`, `FT_all_a0.7` (and `FT_v1_a0.6/0.7`) are in the private checkpoint repo.

Spend: pod `9kjrrcuvzrueu6` 14:38–15:11 UTC ≈ US$0.41 (≈ CA$0.57); `42dc1t20jc8whf` couldn't restart
(no free GPU on its host). B2 GPU total ≈ US$4.02 (≈ CA$5.63). Both pods stopped; volumes bill storage.
