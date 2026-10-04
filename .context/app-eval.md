# App-level quality (2026-10-04)

Share of words wrong (WER), same 20 real-face clips (`ml/data/raw_eval`, 122 words), played through
`/app` with a fake camera (`ml/scripts/app_eval/`). The model alone on the same clips is the floor:
**25.4%** on-device greedy, **29.5%** pod beam + LM.

Gate (`./smoke.sh app`): Normal 25.4%, Instant 27.9% — fails when a mode reads more than 5 pts worse.

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
| *model alone, same clips* | *25.4%* | *25.4%* | *29.5%* |

PR #3 numbers used the HF-download harness (see caveat); the last two rows are the fixed harness
(model served locally), same code otherwise. Runs vary ~2 points on the same code.

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
Quality (server reads) still uses look-alike; the server-side variant is step 3 of phrase-scoring.md.
