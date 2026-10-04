# App-level quality (2026-10-04)

Share of words wrong (WER), same 20 real-face clips (`ml/data/raw_eval`, 122 words), played through
`/app` with a fake camera (`ml/scripts/app_eval/`). The model alone on the same clips is the floor:
**25.4%** on-device greedy, **29.5%** pod beam + LM.

Gate (`./smoke.sh app`): Normal 25.4%, Instant 27.3% — fails when a mode reads more than 5 pts worse.
(Means of the runs below on `frontend/app-gaps`; update the line when an intended change moves them.)

## Round 2: capture loop (`frontend/app-gaps`, cloud container)
4-core Xeon 2.1 GHz, headless Chromium 1194, model served locally. Here the model alone reads
**26.2%** (32 word errors; `bench.py --backend onnx` on the pinned int8, one word more than 25.4%).

| step (cumulative) | Instant | Normal | cuts / 20 clips | lip tracker |
|---|---|---|---|---|
| master `7727199` (baseline) | 31.1% | 32.0, 33.6% | 21-22 | 4 Hz, 0.6-0.85 s late |
| + a sentence ends where the lips went still, not when the lock fires | 29.5, 45.1% | 23.0, 41.8% | 21-37 | same, stalls to 3 s |
| + full-rate tracking on software GL, activity over a 250 ms span | 27.9, 30.3% | 26.2, 23.8, 26.2% | 21 | 15 Hz, 50 ms late |
| + lips gone / cap on the tracker's clock, lost lips reset activity | 26.2, 29.5% | 27.9*, 26.2% | 20 | 19-23 Hz |
| **+ a saved phrase drops at most 1 word (this branch)** | **26.2%** (gate) | **26.2, 24.6, 25.4%** (gate) | **20** | 23 Hz, 40 ms |
| *model alone, same clips, this container* | *26.2%* | *26.2%* | | |

\* one 4.6 s camera stall (no frames at all) lost clip 1; nothing in that change causes stalls.
Word errors on the final code: Normal 32, 30, 31; Instant 32, 36, 32; the model alone 32: the app
now reads what the model reads. No empty readings, one cut per clip.
Fast tracking alone, before the 250 ms activity span: Normal 37.7 / 34.4%, Instant 30.3 / 33.6% (the
bars were tuned for 4 Hz tracking and missed quiet starts and whole clips at 15 Hz). Forcing the old
slow tracker (GPU delegate on SwiftShader, 5 Hz, 0.5 s late) on the final code: Instant 27.0%, 20 cuts.

Baseline vs the records above: Instant reproduced (31.1 vs 29.5%); Normal did not (32.0 / 33.6% vs
~23%). Here the worker's FaceLandmarker took its GPU delegate on SwiftShader at ~180 ms a frame, so
lips were tracked at ~4 Hz and ~0.7 s behind the camera; locks then fired ~1.8 s after the last
movement and cut into the next clip, and Normal's drafts and phrase memory amplified that.

Quality not measured this round: headless Chromium in this container doesn't trust the egress
proxy's certificate (`ERR_CERT_AUTHORITY_INVALID` for every HTTPS host, the pod included), so every
Quality final fell back to the device. The fixes are in the shared capture loop, so Quality gets
them too; measure it where the browser reaches the pod (the laptop).

### What the gap was (round 2)
Read from the dev trace with `ml/scripts/app_eval/cuts.py` (each cut against the clip timeline).
- **Cuts ran into the next sentence**: the lock fires on the tracker's (late) view of an 800 ms
  pause, and the sentence was cut at that moment, ~1 s later: in one baseline run 10 of 22 cuts
  took 2-14% of the next clip ("…what happened THE" + "AIRPLANE is almost full", "I would like…" → "LIKE NEW A LOT…").
  Now a sentence ends at its last movement + 800 ms; later frames stay buffered as the next lead-in.
- **Lip tracking at 4 Hz on software GL**: MediaPipe's GPU delegate "works" on SwiftShader but is
  6x slower than its CPU delegate (180 vs 30 ms a frame). The worker now skips GPU when WebGL is a
  software rasterizer (headless, VMs, blocklisted GPUs); a real GPU keeps the GPU delegate.
- **Activity depended on the tracker's speed**: deformation was measured between consecutive
  results, so it grows with the gap between them; the bars (0.035 / 0.018) only fit ~4 Hz. Now it is
  the deformation over a fixed 250 ms span with time-based smoothing (0.6 per 250 ms): same bars,
  any tracker rate (4 Hz, 15 Hz here, 30 Hz on a GPU laptop).
- **Lips gone and the cap on the camera's clock**: a tracker 1.5 s behind (it stalls while the
  reader runs on a slow CPU) read as "lips gone" and cut sentences into pieces (one run: 37 cuts,
  16 "lips gone", 6 pieces dropped as faceless or empty); the 6 s cap fired before the pause did. Both now count tracked
  time; at the cap, only a sentence still mid-speech continues as a new one.
- **Empty readings**: the silent pieces of those splits, plus the video looping back to its first
  face, which read as lip motion. Lost lips now reset the activity measure: 0 per run.
- **Phrase memory swallowed full reads**: every line is saved, so a clipped one ("Talking by the
  door", "I'm") later replaced whole readings through model-scored snapping ("KIDS ARE TALKING BY
  THE DOOR" → "Talking by the door" ×3 in one run; "A PIN BROKEN IN HEAD DOWN" → "I'm"). A saved
  phrase may now leave out at most one word of the reading (`MAX_SNAP_DROPS`); one still fixes a
  split word ("THE JAWS AREED UP BY THE DOOR" → "Dogs are sitting by the door"). Instant never snaps.

### Unseen faces (judges)
`ml/scripts/app_eval/faces.py`: per clip, words lost in the app runs above and by the model alone,
next to what the camera sees (eye distance in the app's 640x480 view and in the source, brightness,
mouth-crop contrast, head motion per frame). It also writes `artifacts/app_eval/faces/sheet.png`
(a frame + 8 mouth crops per clip, worst first; crops of the private set, so not committed).

| speakers | clips | words lost: Normal | Instant | model alone | eyes, source px | frame light | mouth contrast |
|---|---|---|---|---|---|---|---|
| GRID s1, s2, s4 (commands) | 6 | 68% | 66% | 56% | 51 | 132 | 25 |
| CREMA-D (8 actors) | 8 | 10% | 11% | 12% | 53 | 90 | 32 |
| RAVDESS (6 actors) | 6 | 5% | 11% | 17% | 157 | 212 | 30 |

- **GRID loses the most, and the model alone does too**: letter/digit commands ("bin blue at f two
  now") are far from the TED-talk sentences the model knows, and the faces are small in a
  compressed 360x288 source (~50 px between the eyes, upscaled for the 640x480 view): the crops are
  soft and grey (contrast 21-29 vs 29-36 for the clips read perfectly) and the mouths barely open.
  s4 (5 of 6 words lost) has the lowest contrast (21-23). The app loses 0.5-1.3 words per GRID clip
  more than the model alone; likely the eval video's 25 → 30 fps duplicates read back at 25 fps
  (only GRID is 25 fps), not something a 30 fps webcam does. Unverified.
- **Mouth-crop contrast is the signal that tracks it** over all 20 clips (Spearman with words lost:
  −0.60 Normal, −0.71 Instant, −0.64 model alone, p ≤ 0.01); frame brightness (|ρ| ≤ 0.17), head
  motion and turn don't. Face size in the view even goes the wrong way (+0.5): GRID is upscaled to
  it, so what counts is the detail the camera really captured, which shows up as contrast.
- **Natural sentences lose little** (0-2 words a clip) and nothing measured explains which: without
  GRID no metric is significant (contrast closest, −0.15 to −0.45). The two darkest faces (CREMA-D
  1013 at frame light 66, 1032 at 78, also the most head movement: 2.1% of eye distance a frame,
  5.5° roll) lose 1-2 words, but so do 1001 and 1007 in normal light. The "too dark" hint (below 60)
  stays silent for all; every face is above the 58 px "move closer" hint (63-93 px in the view).
- For the demo: light the face evenly from the front (contrast in the mouth is what tracked errors),
  keep the "move closer" hint off, and say natural sentences; GRID-like spelled commands read badly
  whoever says them.

### Decisions (round 2)
Numbered after D81; renumber if another branch lands D82+ first.
- D82: A sentence ends at its last movement + the lock pause (800 ms), not at the frame the lock
  fires; frames after it stay buffered for the next sentence's lead-in.
- D83: The face-tracker worker skips MediaPipe's GPU delegate when WebGL is a software rasterizer
  (SwiftShader/llvmpipe): 6x faster there, unchanged on a real GPU.
- D84: Lip activity = deformation over a fixed 250 ms span, smoothing 0.6 per 250 ms, so the
  0.035 / 0.018 bars hold at any tracker rate.
- D85: "Lips gone" (1.5 s; 5 s with no tracker answer) and the length cap count tracked time; at the
  cap, a sentence continues only if still mid-speech (quiet ≤ 300 ms). Lost lips reset activity.
- D86: A saved phrase may drop at most one word of the reading (`snapAllowed`, all modes that snap).
- D87: `./smoke.sh app` (opt-in, ~5 min) fails when Normal or Instant is > 5 pts worse than the Gate
  line above; plain `./smoke.sh` doesn't run it.

## Round 1 (`cloud/ml-live`, `cloud/phrase-scoring`)

### Model and server: no drift
Re-run 2026-10-04 on current code, identical to the earlier records: `regress_quantized.py` 13/13
gates, lock 15/15 identical; int8 greedy 28.5% (LRS3 100) / 25.4% (raw 20); pod beam 22.6% / 29.5%.

### App, by version
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

### What the gap was
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

### Harness caveat (fixed)
A fresh browser profile downloads the 203 MB model from HF every run; the varying load time made
the app start listening at a different point of the video, dropping 0-6 early clips at random.
Serve it locally (`VITE_LIPREAD_MODEL_BASE=/models`) when comparing runs.

### Model-scored phrase snapping (wired, `cloud/phrase-scoring`)
On-device reads now carry their log-probs; the hook ranks the user's own saved phrases with
`rankByModel` and snaps with `modelSnap` (margin ≥ −0.2), still behind `snapAllowed`. Normal, 4 runs:
18.9 / 24.6 / 27.0 / 20.5% (≈23%) vs 28.7% look-alike. Caveat: this eval repeats sentences
("kids/dogs by the door" ×3), which is exactly what phrase memory helps; real speech gains less.
Seeds are excluded from model ranking: against a garbled read ("PLEASE BLOW THOSEING SOON") the model
picked the one-word seed "shit". Seeds keep the look-alike rule (needs 0.75 resemblance).
Quality (server reads) still uses look-alike; the server-side variant is step 3 of phrase-scoring.md.

## B2 fine-tune A/B (2026-10-04, cloud container, Quality served from a 4090 pod)
Stock vs the shipped fine-tune (`FT_v1` WiSE α 0.5, D88), same harness, same session, two runs each;
on-device model served from `/models`, Quality from the pod's GPU (stock or fine-tuned in turn).
The 20 clips are strangers (public corpora), not the team.

| model | Instant | Normal | Quality |
|---|---|---|---|
| stock (2 runs) | 32.0 / 28.7 → **30.3%** | 26.2 / 32.0 → **29.1%** | 23.8 / 27.9 → **25.8%** |
| fine-tuned (2 runs) | 27.9 / 32.0 → **30.0%** | 30.3 / 23.8 → **27.0%** | 32.8 / 24.6 → **28.7%** |

No measurable change for speed mode on strangers: runs of the same model differ by up to 8 points here,
more than the gaps. Quality with a fine-tuned server averages ~3 points worse (model level: LRS3-100 beam
22.6 → 24.4%), so the serving pod stays stock. A local CPU server can't do Quality at all (no beam
request finished within a run); use a GPU server for Quality runs.
