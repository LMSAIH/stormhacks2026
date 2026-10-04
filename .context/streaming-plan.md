# Plan: hands-free lip-reading stream with fewer dropped and misread words

Status: approved 2026-10-03 ~19:15 PT. Phases 1–3 in progress on branch `ml/streaming`; phase 5
waits for the teammate's streaming branch.

## Context
The teammate's streamed-on-demand version cuts the video into ~2.5 s pieces, so words at the start
and end of pieces get cut, and it feels choppy. We want to find the longest piece that keeps speed
and accuracy, cut at natural pauses, drop and misread fewer words, and grab the video better. The
B2 fine-tune runs separately in a cloud session.

## Decisions
- D45: The 2.5 s buffer is in the teammate's unpushed streaming version, not master.
- D46: Hands-free. Start when the lips move; when the lips close, send the buffer for reading.
- D47: Build the capture-loop changes on the teammate's branch once pushed; until then, separate
  modules and measurements only.
- D48: Skip the "flight manual AST 86" reference.
- D49: A pause = closed mouth judged per person (resting mouth learned in the first seconds);
  fall back to "closed and still ~300 ms" until learned. Lips close inside words (p, b, m), so an
  instant close never ends a chunk.
- D50: Text grows until a long pause: each short pause re-reads the sentence so far and replaces the
  line; a long pause (0.8 s) locks it. Length is capped.
- D51: Measure delay and word errors at 2/4/6/8/10/15/20 s; the user picks the cap from the table.
- D52: Stream fixes: steady 30 fps 640x480 camera, face-quality warnings, face trackers in a
  background worker, face detection on every 2nd frame with gap filling.
- D53: Fast on-device draft while talking, beam + language model final when a sentence locks, and
  snapping to known phrases from a phrase database.
- D54/D65: Beam finals run on a second RunPod pod used only for serving; the cloud fine-tune keeps
  its own pod. Pod limit is now 2.
- D55: Three modes, delay targets tried and compared: instant (draft only), speed (on-device
  final, under 1 s), accuracy (beam final on the pod, up to ~4 s).
- D56: Test on stitched LRS3 clips now; team long takes (30–60 s with pauses, plus script) later.
- D57: The phrase database is TiDB (PingCAP). It fits the Devpost "TiDB x AI Open Build" track.
- D58: The phrase database is the backend team's; we call their endpoint (user is confirming).
- D59: Learning before the deadline = phrase memory + opt-in training pairs (mouth clip + correct
  text). Live model updates only if everything benchmarked is done.
- D60: Fixing a misread = one-tap pick from the top 3 readings, typing as fallback.
- D61: Training pairs go to the HF dataset the fine-tune reads (public for now, fine for the
  hackathon). Uploads go through our ML server, never from the browser (no tokens in the bundle).
- D62: Matching = TiDB vector search for candidates, then a look-alike score (letters and lip
  shapes) picks the winner.
- D63: Browser stand-in (IndexedDB, same calls) until the backend endpoint is live, then switch
  with one setting.
- D64: Order while waiting on the teammate: length table, server top 3, phrase memory, worker
  face detection.
- D66: Assumptions below confirmed by the user (Q50: A).

## Assumptions (confirmed)
1. A long pause of 0.8 s locks a sentence; tuned on real takes later.
2. The sentence cap stays 10 s until the user picks from the length table.
3. Speed and instant modes have no beam search, so their alternatives come from phrase matches only.
4. The training-clip opt-in is off by default and says clips become public.
5. "Instant" is a 3rd option in our existing mode toggle (`components/app/mode-toggle.tsx`).
6. New work goes on branch `ml/streaming`, rebased onto the teammate's branch once pushed.

## Phases

### 1. Length table (no app changes)
- New `ml/scripts/bench_length.py`: join consecutive `mattymchen/lrs3-test` clips (pre-made crops)
  into 2/4/6/8/10/15/20 s clips with their joined transcripts; for each length record greedy WER
  (int8 ONNX, ORT CPU) and delay (ORT CPU, and onnxruntime-web WASM in Node like
  `frontend/src/lib/lipreading/onnxRecognizer.golden.test.ts`). Beam WER + delay per length once
  the serving pod is up. Reuse `ml/scripts/bench.py` loaders and `lipread.model` decoding.
- Output `.context/streaming-length-table.md`. User picks the cap (D51).
- Check: table has every length; 2–10 s numbers line up with the known 2.8 s ≈ 1.05 s WASM point.

### 2. Top 3 readings from the server
- `ml/src/lipread/model.py`: keep the beam's n-best (top 3 + scores). `ml/src/lipread/serve/app.py`:
  `/lipread/crops` and `/lipread` return `alternatives: [{text, score}]` (backward compatible).
- Frontend: `RecognitionResult.alternatives` in `src/lib/lipreading/types.ts`; read in
  `httpRecognizer.ts`.
- Check: `smoke_checks.py` service check asserts ≤3 alternatives, first equals `text`; vitest case.

### 3. Phrase memory (stand-in now, TiDB later)
- New `frontend/src/lib/phrases/`: `PhraseStore` interface (`add`, `search`, `list`, `remove`),
  `IndexedDbPhraseStore`, `HttpPhraseStore` (backend spec below, picked by `VITE_PHRASES_URL`),
  `lookalike.ts` (word-level edit distance where lip-alike sounds are cheap swaps: p/b/m, f/v,
  t/d/n, k/g, ch/j/sh), `snap.ts` (merge beam alternatives + phrase candidates, rank, snap only
  above a confidence threshold).
- Check: vitest for look-alike ranking ("I WANT A BAT" → "I want a pat" style cases), store
  round-trip, snap threshold.

### 4. Video grabbing modules (wired in phase 5)
- `src/lib/lipreading/detectWorker.ts`: BlazeFace + FaceLandmarker in a Web Worker
  (`ImageBitmap` in, keypoints + lip points out); main-thread fallback if workers fail.
- Detect every 2nd frame; the crop already fills gaps (`interpolateKeypoints`).
- Camera constraints `{width: 640, height: 480, frameRate: {ideal: 30, min: 24}}`.
- `faceQuality.ts`: warn when the face is too small (eye distance < ~60 px), too dark (mean
  luminance), or turned (eye/nose geometry).
- Check: `/lab` fps readout before/after; crop parity tests still pass.

### 5. Streaming on the teammate's branch (blocked until it's pushed)
- Branch `ml/streaming` from his branch. Pause detector from his FaceLandmarker lip points: inner-lip
  gap divided by mouth width, with a per-person resting baseline (first ~2 s) and the 300 ms
  fallback (D49).
- Grow-until-pause: short pause → re-read the whole sentence so far as a draft; long pause (0.8 s) →
  lock → final per mode (D55); cap from phase 1.
- Top-3 picker under each locked line (D60): beam alternatives (accuracy) + phrase matches; a pick or
  a typed fix goes to the phrase store.
- Opt-in toggle (off): locked + confirmed lines send (crops, text) to a new `POST /training-pairs`
  on our ML server, which appends to the HF dataset with its own token (D61).
- Check: stitched LRS3 "talks" with fake pauses → count dropped words, misread words and delay per
  mode, against the teammate's 2.5 s version; then the team's long takes.

### 6. Serving pod
- Start a second secure-cloud RTX 4090 pod for serving only (D65); `ml/runpod/bootstrap.sh` +
  `serve.sh`; stop it outside testing and the demo window. Report spend.

## What we need from the backend team (TiDB phrase bank)
- Table `phrases`: `id`, `user_id`, `text`, `norm_text` (uppercase, no punctuation),
  `embedding VECTOR(d)`, `count`, `source` ("accepted" | "picked" | "typed"), `first_seen`,
  `last_used`. Vector index on `embedding`.
- `POST /phrases {user_id, text, source}` → upsert by `(user_id, norm_text)`, bump `count`,
  return `{id}`.
- `GET /phrases/search?user_id=&q=<draft text>&k=10` → embed `q`, vector search, return
  `[{id, text, count, distance}]`. Target under 150 ms.
- `GET /phrases?user_id=&limit=500` → newest/most used first (we cache it in the browser).
- `DELETE /phrases/{id}`.
- CORS for the frontend origin; `user_id` from their existing Google sign-in (socket-setup branch),
  or a fixed demo user until that's merged. Any text-embedding model; tell us the name.

## Non-goals
- Touching `backend/` code ourselves (D42/D58). Live model updates (D59 C) unless time is left.
- Changing the teammate's capture loop before his branch is pushed (D47).

## Risks / open items
- His branch timing: phases 1–4 don't wait on it; phase 5 does.
- Re-reading a growing sentence costs ~0.37 s per second of video on WASM, so long sentences make
  drafts slow; the length table sets the cap.
- Per-person resting mouth can misfire (people who rest with lips apart); the 300 ms rule is the
  fallback.
- Backend may not ship the TiDB endpoint in time; the browser stand-in keeps phrase snapping working.
- The public HF dataset means opted-in mouth clips are public; the toggle must say so.
