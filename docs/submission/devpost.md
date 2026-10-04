# Devpost text: Lipreader

Paste-ready sections for the Devpost story, in Devpost's order. Every number is from `.context/`
(sources listed at the end) or from the code. "Lipreader" is the name on the app's login page;
change it everywhere if the team picks another.

## Inspiration

Some people can move their lips but can't make sound: after a laryngectomy, on a ventilator, or
with some neuromuscular conditions. Their options are usually typing into a text-to-speech app or
pointing at a letter board, both much slower than talking. Lip reading lets them talk at their own
pace with nothing in their hands. We started from Chaplin, an open-source webcam lip reader built
on the Auto-AVSR model, and set out to make it usable in a conversation: hands-free, in a browser,
on faces the model has never seen, and with a real voice at the end.

## What it does

You mouth a sentence at your laptop's webcam. Lipreader notices when your lips start and stop
moving, reads the sentence, shows it, and speaks it aloud through ElevenLabs in a voice you pick.
There is no button to hold.

- Three modes. Instant reads each sentence once, exactly as the model sees it. Normal (the
  default) shows quick grey drafts while you talk, then rereads the whole sentence on the laptop.
  Quality sends the final read to our GPU server, which runs a beam search with a language model on
  an RTX 4090; if the server is unreachable, the laptop reads it instead.
- Confidence boxes. Words the model was unsure of get a dashed box. Tap one to pick another reading
  or type the right words.
- Phrase memory. Sentences you say or correct are saved in your browser. When a new reading is
  close to one of them and the model was unsure of the words that differ, Lipreader uses your
  saved sentence. It never changes a word the model was sure of.
- Training-clip opt-in, off by default. When you turn it on and correct a sentence, the mouth clip
  (grayscale, mouth only) and the corrected text go to a public dataset we can fine-tune on.
- A listening panel that captions the people you're talking with and, with diarization turned on,
  labels who spoke.

Camera video never leaves the browser. In Instant and Normal the reading runs on the laptop, using
a 203 MB int8 version of the model that downloads once and stays cached. Quality mode sends only
88×88 grayscale mouth crops to the server.

How well it reads: on 20 recorded real-face clips (122 words) from people the model never trained
on, played into the app as its camera, Lipreader got 29.5% of words wrong in Instant, 30.3% in
Quality, and about 23% in Normal, averaged over four runs (18.9% to 27.0%). The model on its own
gets 25.4% wrong on the same clips on the laptop and 29.5% on the server. Our first streaming
version of the app got 72% to 90% wrong on these clips, though that run used an older test setup
that could also drop a few early clips at random. The Normal figure leans on phrase memory, and
these test sentences repeat, so expect less gain on everyday speech.

## Limits

- Research licence only. The pretrained model is trained on LRS3 (BBC/TED material) and its
  weights are for non-commercial research use, so this can only be a research and hackathon project.
- English only.
- Sentence by sentence. The model reads speech in chunks that end at a pause, so text appears after
  each pause and never word by word.
- It still gets 23% to 30.3% of words wrong on new faces in our test, depending on the mode, and
  it needs a frontal face, even light and a webcam that keeps up near 30 frames per second. The
  app warns when the face is too far, too dark or turned.
- Quality mode needs our GPU pod and a network connection. The voice needs the user to sign in.
- Our accuracy numbers come from recorded clips played as a camera. We have no measurements from
  live webcam sessions yet, and the test set is small.

## How we built it

Architecture diagrams, one page each:
[system](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/system.md),
[lip-reading pipeline](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/lipread-pipeline.md),
[modes](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/modes.md),
[training loop](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/ml-training.md),
[deployment](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/deployment.md),
[evaluation](https://github.com/LMSAIH/stormhacks2026/blob/master/docs/architecture/eval.md).

- The model is Auto-AVSR's `LRS3_V_WER19.1` (about 250M parameters), with inference and training
  code vendored from auto_avsr and Chaplin under their licences.
- In the browser (React, Vite, TypeScript), MediaPipe's BlazeFace and FaceLandmarker run in a Web
  Worker. BlazeFace's four keypoints drive the mouth crop; FaceLandmarker's lip points drive the
  lip-movement detector that starts and ends sentences, with a 1 s lead-in and an 800 ms pause cut.
- The mouth crop (25 fps resample, smoothing, a similarity warp onto a mean face, 96×96 then
  88×88) is a TypeScript port of the Python preprocessing that matches it bit for bit in our parity
  test, because the model was trained on crops made exactly this way.
- We exported the encoder and CTC head to ONNX (775 MB) and wrote our own int8 recipe: per-channel
  int8 matrix multiplies (1×1 convolutions rewritten as matrix multiplies), fp16-stored dense
  convolutions, and a position table trimmed to 20 s. The result is 203 MB with 28.5% WER on 100
  LRS3 test clips against 28.6% for fp32. A regression suite locks the file's hash and its exact
  outputs on 15 clips. onnxruntime-web runs it on WASM in its own worker, about 1 s for 2.8 s of
  video on our demo laptop.
- Our FastAPI server on a RunPod RTX 4090 takes gzipped mouth crops and runs beam search (width
  40) with an RNN language model. On 100 LRS3 test clips it gets 22.6% WER against 28.5% for the
  on-device read. It returns up to three readings and a confidence for each word.
- Phrase memory lives in IndexedDB. It ranks saved sentences two ways: a text distance that treats
  lip-alike letters (p/b/m, f/v, t/d/n) as cheap swaps, and the model's own CTC likelihood of the
  sentence given the lip frames. In an offline test with near-duplicate saved sentences, the model
  score put the right one first 49 times out of 50, against 41 for text distance.
- The fine-tuning pipeline (our own crops, frozen BatchNorm statistics, WiSE-FT blending with the
  original weights, ship gates on unseen faces) is built and rehearsed on the public GRID corpus.
- Our teammates built the backend: FastAPI with Google sign-in, ElevenLabs streaming speech over a
  WebSocket, ElevenLabs Scribe live transcription with speaker labels, and Postgres for saved notes.
- An app-level test plays the 20 real-face clips through the real app in headless Chromium as a
  fake camera and scores the whole transcript.

## Challenges we ran into

- Cutting sentences. Our first streaming version got 72% to 90% of words wrong (measured with an
  older test setup that could also drop early clips), mostly words it never read. Sentence starts
  were clipped (fixed with a 1 s lead-in), speakers who started softly never triggered it (lower
  start threshold), slow speakers were split mid-sentence (lower threshold to keep going), a saved
  phrase overrode a correct reading (now it can only change unsure words), and the beam search
  invented fluent sentences from still lips (it now returns nothing when the CTC head hears no
  words).
- Frame rate. Offline, on 30 LRS3 clips, error rose from 38.1% at 25 fps to 48.3% at 15 fps.
  Running both face trackers on the main thread dropped capture from 28 to 16 fps, so we moved them
  into a worker.
- WebGPU. ONNX Runtime's default WebGPU backend rejected the model's 3D convolution padding, and
  the newer one that runs it took 2.8 s for a read that WASM did in about 1.2 s on our laptop, so
  the browser uses WASM.
- Fine-tuning forgot. A plain fine-tune on GRID took our open-speech check from 28.6% to 70.8%
  wrong. Freezing BatchNorm statistics and blending back with the original weights brought it to
  29.7% while keeping much of the gain on unseen GRID speakers.
- Vocabulary. The model learned from TED talks and cannot spell most swear words, so it reads them
  as clipped fragments or clean look-alikes. People who can't speak still swear, so phrase memory
  includes a small set of swear phrases and repairs the fragments.
- GPUs. Two community RTX 4090 hosts failed CUDA initialisation, and a stopped pod could not
  restart once its host's GPU was taken.

## Accomplishments that we're proud of

- The whole reading pipeline runs in a browser tab (camera, face tracking, crop and the int8
  model), and its text matches native ONNX Runtime and PyTorch on our test clip.
- After fixing how sentences are cut, each mode lands within about 4 points of the model alone on
  the same clips, and Normal with phrase memory does better than the model alone.
- A 203 MB on-device model with no measured accuracy loss against the 775 MB original, guarded by
  a regression lock.
- Confidence boxes calibrated on real clips: they catch about half the misread words while boxing
  4–8% of correct ones.

## What we learned

- Most of our errors came from where we cut sentences. Once that was fixed, the app scored close to
  the model on its own.
- A small dataset with a fixed grammar teaches a model fast and makes it forget open speech.
  Freezing BatchNorm statistics and blending weights recovers most of it.
- A language model helps on natural sentences and hurts on made-up ones: on GRID commands like
  "bin blue at f two now" it pulls the reading towards ordinary English.
- Measure the whole app on the same clips as the model, or you can't tell which part lost the words.

## What's next for Lipreader

- Fine-tune on the team's own recordings and ship it only if it passes both gates: at least 3
  points better on a held-out speaker, and no more than 2 points worse on unseen LRS3 faces.
- Move phrase memory to the backend team's TiDB store with vector search, so it follows the user
  across devices.
- Test with live webcams and more speakers.
- Package it as a desktop app with ONNX Runtime for Node.
- Try the LLM corrector (the hook exists but is off) and USR 2.0 as a stronger model.

## Built with

TypeScript, React, Vite, Tailwind CSS, shadcn/ui, onnxruntime-web, WebAssembly, Web Workers,
IndexedDB, MediaPipe Tasks Vision, Python, PyTorch, PyTorch Lightning, ESPnet, Auto-AVSR, ONNX,
ONNX Runtime, SentencePiece, FastAPI, uvicorn, Hugging Face Hub, RunPod, ElevenLabs, Silero VAD,
SpeechBrain, Google OAuth, Postgres, Docker, Playwright, uv, pnpm.

## For the submitter (not part of the Devpost text)

### Tracks

From the prize list on stormhacks2026.devpost.com (opt in to each sponsored and community track):

| Track | Opt in? | Why |
|---|---|---|
| [MLH] Best Use of ElevenLabs | Yes | Every finished line is spoken with ElevenLabs streaming TTS in a voice the user picks, and the listening panel uses ElevenLabs Scribe live transcription |
| Best MedTech | Yes | Assistive communication for people who can't voice; camera video stays on the device |
| SSSS Python Track | Yes | The ML server, fine-tuning, ONNX export, quantization and evaluation are Python, as is the backend |
| Enactus SFU UNSDG Track | Yes | Fits SDG 3 (good health and well-being) and SDG 10 (reduced inequalities) |
| IATSU Best Design Track | Yes | Open to all; our case is the confidence boxes with one-tap fixes, plain mode labels and face hints |
| Surge Choice Award | Yes | No special requirement |
| TiDB x AI Open Build | In progress | Needs a TiDB AI feature. The app's phrase store already talks to a phrase service through `VITE_PHRASES_URL`, but the backend team's TiDB service is not on master or any backend branch (checked 2026-10-04 03:30 PT). Opt in only if it lands and the demo uses it |
| [MLH] Best Use of Tiger Data | Ask the backend team | Their notes and voice preferences are in Postgres set by `TIMESCALE_SERVICE_URL` (Tiger Data is the company formerly called Timescale). Plain tables, no Timescale features. Qualifies only if that database runs on Tiger Cloud |
| [MLH] Best Use of Gemini API | No | Nothing calls Gemini; the corrector hook is unused |
| [MLH] Best .Tech Domain Name | Only with a domain | Needs a registered .tech domain; none in the repo |
| CSSS SFU CS Legacy Track | Only with the hunt | Needs 50% of the scavenger hunt completed |

Best Beginner, Best Highschool Hack, Best Solo and the WiCS Cosmos Track depend on who is on the
team. The rest (game, hardware, sports analytics, Solana, Snowflake, both Huawei challenges, Earth
observation) don't fit this project.

### Before submitting

- The links above point at `master`; check that they open once this branch is merged and that the
  repo is public.
- Submission: project link and a video of 3 minutes or less, due 2026-10-04 12:00 PT.
- If Phase 2 fine-tuning ships before the deadline, update "What it does" and "What's next" with
  the numbers from `.context/b2-report.md`.

### Where the numbers come from

| Claim | Source |
|---|---|
| App 29.5% / ≈23% (18.9–27.0%) / 30.3%; model alone 25.4% / 29.5%; first version 72.1–90.2% with the old harness | `.context/app-eval.md` |
| int8 203 MB vs 775 MB, 28.5% vs 28.6% on LRS3-100 | `.context/project-brief.md` D39 |
| Beam + LM 22.6% vs 28.5% on LRS3-100 | `.context/app-eval.md`, `.context/streaming-length-table.md` |
| About 1 s for 2.8 s of video on the demo laptop (0.93–1.15 s) | `frontend/bench/ort-threads/README.md` |
| 38.1% at 25 fps, 48.3% at 15 fps; capture 28 → 16 fps with both trackers | `.context/project-brief.md` §11 and D40 |
| Phrase snapping 49/50 vs 41/50 | `.context/phrase-scoring.md` |
| Fine-tune 28.6% → 70.8%, frozen BN + WiSE-FT 29.7% | `.context/b2-report.md` |
| Boxes catch about half the misread words, box 4–8% of right ones | `frontend/src/lib/lipreading/wordSpans.ts` |
| WebGPU padding error; newer WebGPU backend 2.8 s vs about 1.2 s on WASM | `.context/project-brief.md` D37 and §12 (B1) |
| Community pods failing `cuInit`, stopped pod unable to restart | `.context/project-brief.md` D32, D80 |
