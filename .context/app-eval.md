# App-level quality (2026-10-04)

Share of words wrong (WER), same 20 real-face clips (`ml/data/raw_eval`, 122 words), played through
`/app` with a fake camera (`ml/scripts/app_eval/`). The model alone on the same clips is the floor:
**25.4%** on-device greedy, **29.5%** pod beam + LM (27.9% with the beam re-tuned on `ml/quality-server`).

## Model and server: no drift
Re-run 2026-10-04 on current code, identical to the earlier records: `regress_quantized.py` 13/13
gates, lock 15/15 identical; int8 greedy 28.5% (LRS3 100) / 25.4% (raw 20); pod beam 22.6% / 29.5%.

## App, by version
| version | Instant | Normal | Quality |
|---|---|---|---|
| PR #3 (first streaming) | 90.2% | 72.1% | 75.4% |
| PR #4/#6, 250 ms lead-in, no snap gate | 39.3% | 50.0% | 45.1% |
| 1 s lead-in + snap gate | 41.0% | 36.1% | 36.9% |
| + start bar 0.035 + beam CTC guard | 37.7% | 28.7% | 30.3% |
| **+ keep bar 0.018, Instant cuts at 800 ms (this branch)** | **29.5%** | **28.7%** | **30.3%** |
| + model-scored phrase snapping (`cloud/phrase-scoring`) | — | **~23%** (18.9/24.6/27.0/20.5) | — |
| + server: model snapping, 20 beams / LM 0.2 (`ml/quality-server`)¹ | — | — | **26.2%** (31.2% before, same machine) |
| *model alone, same clips* | *25.4%* | *25.4%* | *29.5%* |

PR #3 numbers used the HF-download harness (see caveat); the next rows are the fixed harness
(model served locally), same code otherwise. Runs vary ~2 points on the same code.
¹ Another machine (a 4 vCPU cloud container, runs vary ~4 points there); details at the end.

## What the gap was
- **Missed words, not misread ones**: whole clips and sentence starts were lost.
- **Phrase snapping overrode correct reads**: "DOGS ARE SITTING BY THE DOOR" (read right) was
  replaced by the earlier saved "Kids are talking by the door". Fix: `wordSpans.snapAllowed`, a
  saved phrase may only change words below 0.9 confidence. It still repairs unsure reads
  ("IS YOUR TALKING BY THE DOOR" → "Kids are talking by the door").
- **Sentence starts clipped**: 250 ms of lead-in before the movement crossed the start bar cut first
  words ("That is exactly what happened" → "What happens"). Now 1 s (`LEAD_MS`).
- **Quiet starters never crossed the start bar**: `ACTIVITY_START` 0.045 → 0.035 caught them
  (Normal missed words 20 → 5).
- **The beam invented lines on still lips** once the lower bar started a few sentences on pauses
  ("I don't know what it is"); `LipReader.beam` now checks the same encoder's CTC reading first and
  returns empty when it has no words (still lips → '', speech unchanged).
- **Slow speakers were split mid-sentence**: movement dipped below the keep bar for 800 ms+ inside
  a sentence ("Maybe tomorrow it will be cold" → "Will behold" + the rest glued to the next line).
  `ACTIVITY_KEEP` 0.025 → 0.018, and Instant now cuts after 800 ms like Normal (600 ms split more):
  Instant missed words 15 → 7.
- Remaining: every mode within ~1-4 points of the model alone on the same clips. Not a startup bug:
  an earlier note here that the first sentence was lost was wrong (it was read, but cut short).

## Harness caveat (fixed)
A fresh browser profile downloads the 203 MB model from HF every run; the varying load time made
the app start listening at a different point of the video, dropping 0-6 early clips at random.
Serve it locally (`VITE_LIPREAD_MODEL_BASE=/models`) when comparing runs.

## Model-scored phrase snapping (wired, `cloud/phrase-scoring`)
On-device reads now carry their log-probs; the hook ranks the user's own saved phrases with
`rankByModel` and snaps with `modelSnap` (margin ≥ −0.2), still behind `snapAllowed`. Normal, 4 runs:
18.9 / 24.6 / 27.0 / 20.5% (≈23%) vs 28.7% look-alike. Caveat: this eval repeats sentences
("kids/dogs by the door" ×3), which is exactly what phrase memory helps; real speech gains less.
Seeds are excluded from model ranking: against a garbled read ("PLEASE BLOW THOSEING SOON") the model
picked the one-word seed "shit". Seeds keep the look-alike rule (needs 0.75 resemblance).
Quality (server reads) uses it too since `ml/quality-server`, below.

## Quality on the server (`ml/quality-server`, 2026-10-04)
Two server-side changes: model-scored phrase snapping for server reads (`POST /lipread/phrases`,
step 3 of `phrase-scoring.md`) and the beam re-tuned to 20 beams, LM weight 0.2 (was 40 and 0.3).

**Harness pitfall, fixed.** In a cloud container the headless browser reaches the pod through a
TLS-intercepting proxy. Until the proxy's CA was in the browser's NSS store
(`certutil -d sql:$HOME/.pki/nssdb -A -t "C,," -n proxy -i <proxy-ca.crt>`), every Quality run fell
back to on-device reads without a sign (the pod logged no app requests). `e2e_eval.mjs` now counts
server reads (`server` in its JSON) and warns when there are none. All runs below had 20-23.

| Quality, cloud container (4 vCPU, 29-30 fps) | runs | mean |
|---|---|---|
| before: master (40 beams, LM 0.3, look-alike snapping) | 28.7 / 33.6 / 36.9 / 27.9 / 28.7% | 31.2% |
| new beam, look-alike snapping | 32.0 / 25.4% | 28.7% |
| **after: new beam + model snapping (this branch)** | **23.8 / 30.3 / 27.0 / 23.8%** | **26.2%** |
| *model alone (beam on the clips, no app), before → after* | | *29.5% → 27.9%* |

Runs where the face tracker failed (8+ "lips-gone" cuts or 3+ no-face drops, 24-38 words missed)
are left out, by the same rule for every row: four "after" runs (39.3-49.2%), none of the
"before" ones. Not the code: the same build then ran four clean "after" runs in a row. Runs differ
by ~4 points on the same code here.

**Same cuts, beam only.** `e2e_eval.mjs` with `DUMP=1` keeps what the app uploads (its own
sentence cuts: 1 s still lead-in, the pause that ended the sentence); `replay_crops.py` decodes
those cuts at other settings, so settings compare without cut-to-cut noise. 66 cuts from the
"before" runs, WER of the reads before snapping (the replay at 40 / 0.1 / 0.3 matches the app's own
reads exactly):

| beams / CTC / LM | reads WER | decode p50 |
|---|---|---|
| 40 / 0.1 / 0.3 (before) | 34.2% | 919 ms |
| **20 / 0.1 / 0.2 (now)** | **32.8%** | **716 ms** |
| 40 / 0.1 / 0.2 | 32.2% | 920 ms |
| 30 / 0.1 / 0.25, 40 / 0.2 / 0.3 | 32.8% | 811, 903 ms |
| 60 / 0.1 / 0.3 | 33.1% | 1095 ms |
| 20 / 0.1 / 0.1, 40 / 0.1 / 0.3 + length bonus 0.5 | 34.2% | |
| 40 / 0.1 / 0.4 | 36.1% | |

40 beams at LM 0.2 reads 2 words better over 366 but decodes 0.2 s slower (round-trip budget below).

**Where the errors are.** Mostly the 6 GRID clips (letters and digits, "bin blue at f two now"):
the LM turns them into English ("PLACEBO OR V TWO NOW", "PLEASE PRONOUNCE IT SOON"). Then sentence
cuts (a clipped start: "IT WILL BE COLD"). The beam reads most RAVDESS lines right on its own;
snapping fixes the rest ("THE JAWS ARE SITTING BY THE DOOR" → "Dogs are sitting by the door").

### Offline sweep (`ml/scripts/sweep_beam.py`, pod 4090)
WER on whole clips. raw-20 split into its 14 natural sentences and 6 GRID clips; held-out = LRS3
test idx 100-399 (300 clips, not the LRS3-100 gate).

| beams / CTC / LM | LRS3-100 | held-out | raw-20 | natural | GRID |
|---|---|---|---|---|---|
| 40 / 0.1 / 0.3 (before) | 22.6% | 27.3% | 29.5% | 8.1% | 80.6% |
| **20 / 0.1 / 0.2 (now)** | **21.9%** | **27.3%** | **27.9%** | **8.1%** | **75.0%** |
| 20 / 0.1 / 0.3 | 22.9% | 27.1% | 32.8% | 12.8% | 80.6% |
| 20 / 0.1 / 0.1 | 23.1% | 28.1% | 25.4% | 8.1% | 66.7% |
| 20 / 0.1 / 0 (no LM) | 23.8% | 28.8% | 24.6% | 9.3% | 61.1% |
| 30 / 0.1 / 0.2, 40 / 0.1 / 0.2 | 22.7, 22.2% | | 27.9% | 8.1% | 75.0% |
| 20 / 0.2 / 0.2, 20 / 0.3 / 0.2 | 24.2, 23.9% | | 27.9, 31.1% | | |
| 5, 10, 60 beams / 0.1 / 0.3 | 26.4, 25.2, 21.9% | | 31.1, 32.8, 29.5% | | |
| 20 / 0.1 / 0.2 + length bonus 0.5, 1.0 | 22.7, 23.6% | | 27.9, 30.3% | | |

- The LM weight trades GRID against natural speech: below 0.2 GRID reads better but held-out LRS3
  gets worse (+0.8 points at 0.1, +1.5 with no LM). 0.2 costs nothing on natural speech, and demo
  speakers say sentences, not letter codes.
- More CTC weight, a length bonus or more than 20 beams: no gain. Fewer than 20: worse.
- Rejected (`scripts/beam_vs_ctc.py`): taking the greedy CTC reading when the beam reading's CTC
  likelihood falls far below it. The beam strays most on LRS3 clips where greedy is wrong too; the
  only cut that helps raw-20 (−0.05 per frame: 27.9 → 23.0%) costs LRS3-100 4 points (21.9 → 26.0%).
- Padding the clips like the app's cuts (`--pad 25 20`: 1 s still lead-in, 0.8 s still tail) costs
  every setting (LRS3-100 22% → 31-37%) and splits the two: the old setting read raw-20's sentences
  better (5.8 vs 11.6%), the new one LRS3-100 (32.9 vs 35.5%). The replay of real app cuts above
  is the deciding test for the app.
- Per-word confidence (`calibrate_conf.py`, temperature 2.0 kept): flagging below 0.6 catches 41%
  of misread words and boxes 4% of right ones.

### Latency
Round trip from the cloud container through the RunPod proxy (`bench.py --transport crops`,
20 / 0.1 / 0.2), clips of 2.5-3.5 s (n = 25): p50 1.14 s = server 0.74 s + network 0.40 s; a
straight-line fit gives 1.29 s at 3 s. Decode alone at 3 s (encoder + beam, pod idle): 0.95 s,
vs 1.19 s for 40 / 0.1 / 0.3, about 0.25 s more per read (a 3 s clip at ~1.4-1.5 s). Phrase scoring
is a second request after the read (one encoder pass on the server; the upload is most of it).
