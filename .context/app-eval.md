# App-level quality (2026-10-04)

Share of words wrong (WER), same 20 real-face clips (`ml/data/raw_eval`, 122 words), played through
`/app` with a fake camera (`ml/scripts/app_eval/`). The model alone on the same clips is the floor:
**25.4%** on-device greedy, **29.5%** pod beam + LM (27.9% with the beam re-tuned on `ml/quality-server`).

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
| + server: model snapping, 20 beams / LM 0.2 (`ml/quality-server`)¹ | — | — | **26.2%** (31.2% before, same machine) |
| *model alone, same clips* | *25.4%* | *25.4%* | *29.5%* |

PR #3 numbers used the HF-download harness (see caveat); the next rows are the fixed harness
(model served locally), same code otherwise. Runs vary ~2 points on the same code.
¹ Another machine (a 4 vCPU cloud container, runs vary ~4 points there); details at the end.

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
Quality (server reads) uses it too since `ml/quality-server`, below.

## Quality on the server (`ml/quality-server`, 2026-10-04)
Server-side changes: model-scored phrase snapping for server reads (`POST /lipread/phrases`, step 3
of `phrase-scoring.md`; margins against the likelier under CTC of the beam reading and the greedy
one) and the beam re-tuned to 20 beams, LM weight 0.2 (was 40 and 0.3).

**Harness pitfall, fixed.** In a cloud container the headless browser reaches the pod through a
TLS-intercepting proxy. Until the proxy's CA was in the browser's NSS store
(`certutil -d sql:$HOME/.pki/nssdb -A -t "C,," -n proxy -i <proxy-ca.crt>`), every Quality run fell
back to on-device reads without a sign (the pod logged no app requests). `e2e_eval.mjs` now counts
server reads (`server` in its JSON) and warns when there are none. All runs below had 20-23.

On master's capture loop (round 2 above; cloud container, tracker 14-18 Hz, 20 cuts every run):

| Quality | runs | mean |
|---|---|---|
| before: master (40 beams, LM 0.3, look-alike snapping) | 25.4 / 29.5 / 25.4% | 26.8% |
| new beam + model snapping, margins vs the beam reading | 23.8 / 27.9 / 27.0% | 26.2% |
| **this branch (margins vs the CTC-likelier reading)** | **28.7¹ / 26.2 / 25.4%** | **26.8%** |
| *model alone on the clips (beam, no app), before → after* | | *29.5% → 27.9%* |

¹ Its last read stalled 13 s and missed the 135 s watch window: the last sentence (6 words) counts
as missed.

At app level the three rows are within run-to-run noise (~2-4 points): on this capture the change
is not visible in the app WER. What is measurable: the beam (same cuts, below) and fewer wrong
snaps offline. Each Quality line now also waits for the phrase request: lock → shown line, median,
2.1 s before vs 2.3-3.1 s after from this container (the request itself: 0.24 s round trip, 26 ms
on the server; most of it is re-uploading the crops).

Same cuts, beam only: `e2e_eval.mjs` with `DUMP=1` keeps what the app uploads (its own sentence
cuts), `replay_crops.py` decodes them at other settings; the replay reproduces each run's own reads.
120 cuts from the six runs above, reads before snapping: 40 / 0.1 / 0.3 **28.4%**, 20 / 0.1 / 0.2
**26.6%** (40 / 0.1 / 0.2: 26.9%); decode p50 931 → 744 ms. No empty readings.

Snaps in those runs: look-alike fixed 1 line; model snapping against the beam reading fixed 2
("JOHN IS CRITIQUED BY THE DOOR" → "Dogs are sitting by the door") and broke 2, both GRID lines
snapping to an earlier GRID misread ("PLACEBO OR V TWO NOW" → "People have two now"). Against the
CTC-likelier reading one of the two is blocked, both fixes stay.

Snapping offline (`bench_phrase_snap.py --readings`, the 20 / 0.1 / 0.2 beam readings of LRS3 idx
100-399, 50 saved phrases + 3 one-word-swap decoys each, 27.3% unsnapped):

| rule | in-list fixed | wrong snaps | broke a correct read | WER after |
|---|---|---|---|---|
| look-alike ≥ 0.75 (Quality before) | 14/50 | 0 | 0 | 26.1% |
| model ≥ −0.2, vs the beam reading | 20/50 | 10 | 0 | 25.0% |
| **model ≥ −0.2, vs the CTC-likelier reading (now)** | **18/50** | **2** | **0** | **25.2%** |

**On the capture before round 2** (tracker at ~4 Hz on software GL; runs cut into the next
sentence or lost the lips), same comparison: before 28.7 / 33.6 / 36.9 / 27.9 / 28.7% (31.2%),
after (margins vs the beam reading) 23.8 / 30.3 / 27.0 / 23.8% (26.2%); 66 cuts replayed, beam
only: 34.2% → 32.8%. Four more "after" runs there lost the lips (8+ "lips-gone" cuts, 39-49%) and
are left out; so is every earlier run here, none of which reached the server.

| beams / CTC / LM, the 66 pre-round-2 cuts | reads WER | decode p50 |
|---|---|---|
| 40 / 0.1 / 0.3 (before) | 34.2% | 919 ms |
| **20 / 0.1 / 0.2 (now)** | **32.8%** | **716 ms** |
| 40 / 0.1 / 0.2 | 32.2% | 920 ms |
| 30 / 0.1 / 0.25, 40 / 0.2 / 0.3 | 32.8% | 811, 903 ms |
| 60 / 0.1 / 0.3 | 33.1% | 1095 ms |
| 20 / 0.1 / 0.1, 40 / 0.1 / 0.3 + length bonus 0.5 | 34.2% | |
| 40 / 0.1 / 0.4 | 36.1% | |

**Where the errors are.** Mostly the 6 GRID clips (letters and digits, "bin blue at f two now"):
the LM turns them into English ("PLACEBO OR V TWO NOW", "PLEASE PRONOUNCE IT SOON"). The CREMA-D
sentences mostly read right, and the beam reads most RAVDESS lines right on its own.

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
