# Devpost text: HearD

Paste-ready sections for the Devpost story, in Devpost's order. Every number is from `.context/`
(sources listed at the end) or from the code. The app is called HearD. Try it at
https://tryheard.tech.

## Inspiration

Some people can move their lips but can't make sound: after a laryngectomy, on a ventilator, or
with some neuromuscular conditions. Their options are usually typing into a text-to-speech app or
pointing at a letter board, both much slower than talking. Lip reading lets them talk at their own
pace with nothing in their hands. We started from Chaplin, an open-source webcam lip reader built
on the Auto-AVSR model, and set out to make it usable in a conversation: hands-free, in a browser,
on faces the model has never seen, and with a real voice at the end.

## What it does

You mouth a sentence at your laptop's webcam. HearD notices when your lips start and stop moving,
reads the sentence, shows it, and speaks it aloud through ElevenLabs in a voice you pick. There is
no button to hold. It runs live at https://tryheard.tech.

- Three modes. Instant reads each sentence once, exactly as the model sees it. Normal (the
  default) shows quick grey drafts while you talk, then rereads the whole sentence on the laptop.
  Quality sends the final read to our GPU server, which runs a beam search with a language model on
  an RTX 4090; if the server is unreachable, the laptop reads it instead.
- Confidence boxes. Words the model was unsure of get a dashed box. Tap one to pick another reading
  or type the right words.
- Phrase memory. Sentences you say or correct are saved in your browser. When a new reading is
  close to one of them, the model itself scores whether your lip movements fit the saved sentence,
  and HearD uses it only where the model was unsure. It never changes a word the model was sure of.
- Training-clip opt-in, off by default. When you turn it on and correct a sentence, the mouth clip
  (grayscale, mouth only) and the corrected text are sent to our server to fine-tune on. The server
  can publish them to a public Hugging Face dataset, so the toggle says the clips become public.
- A listening panel that captions the people you're talking with and, with diarization turned on,
  labels who spoke.

Camera video never leaves the browser. In Instant and Normal the reading runs on the laptop, using
a 203 MB int8 version of the model, fine-tuned on our team's recordings, that downloads once and
stays cached. Quality mode sends only 88×88 grayscale mouth crops to the server.

How well it reads:
- On a test set we built from 144 clips of 62 people the model never trained on, the original model
  gets 41.3% of words wrong on the laptop and 35.0% with the server's beam search. Short everyday
  sentences filmed straight on get 6.2–13.4% wrong; long sentences with rare words, 46–59%.
- Fine-tuning on one team member cut errors on a second team member it never saw from 57.3% to
  50.5%, and left the 62-person test unchanged (0.2 points better on the laptop read).
- The app now loses nothing on top of the model: on our 20-clip app test, played into the app as
  its camera, Normal mode made 30–32 word errors where the model alone made 32. Our first streaming
  version got 72% to 90% of words wrong on those clips (with an older test setup that could also
  drop clips).

## Limits

- Research licence only. The pretrained model is trained on LRS3 (BBC/TED material) and its
  weights are for non-commercial research use, so this can only be a research and hackathon project.
- English only.
- Sentence by sentence. The model reads speech in chunks that end at a pause, so text appears after
  each pause and never word by word.
- It still gets many words wrong on new faces: 35.0% to 41.3% on our 62-person test. It needs a
  frontal face and a camera at eye level (a camera below the face added 21.6 points on the takes we
  could test), even light and a webcam that keeps up near 30 frames per second. The app warns when
  the face is too far, too dark or turned.
- Quality mode needs our GPU pod and a network connection. The voice needs the user to sign in.
- Our accuracy numbers come from recorded clips played as a camera. We have no measurements from
  live webcam sessions yet, and none in dim light.

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
  lip-movement detector, which measures lip movement over a 250 ms window, starts each sentence
  1 s early and ends it 800 ms after the last movement.
- The mouth crop (25 fps resample, smoothing, a similarity warp onto a mean face, 96×96 then
  88×88) is a TypeScript port of the Python preprocessing that matches it bit for bit in our parity
  test, because the model was trained on crops made exactly this way.
- We exported the encoder and CTC head to ONNX (775 MB) and wrote our own int8 recipe: per-channel
  int8 matrix multiplies (1×1 convolutions rewritten as matrix multiplies), fp16-stored dense
  convolutions, and a position table trimmed to 20 s. The result is 203 MB with 28.5% WER on 100
  LRS3 test clips against 28.6% for fp32. A regression suite locks the file's hash and its exact
  outputs on 15 clips. onnxruntime-web runs it on WASM in its own worker: 1.09 s for 2.8 s of video
  on our demo laptop.
- We fine-tuned on 143 clips of one team member (frozen BatchNorm statistics, 3 epochs, 0.6 minutes
  on an RTX 4090) and blended the result 50/50 with the original weights (WiSE-FT). It shipped only
  after passing two gates: at least 3 points better on a team member it never saw, and no more than
  2 points worse on unseen LRS3 faces (30.0% against 28.6%). Before training, the lip reader itself
  matched every clip to its script line and caught recording slips that had shifted 98 of 239
  labels.
- Our FastAPI server takes gzipped mouth crops and runs beam search with an RNN language model. A
  sweep picked 20 beams and a language-model weight of 0.2: as accurate as our first setting (40,
  0.3) on the 62-person test, and 0.82 s instead of 1.19 s for a 3 s sentence. It returns up to
  three readings and a confidence for each word. Quality keeps the original weights, because the
  fine-tuned ones made its beam search worse on strangers.
- Phrase memory lives in IndexedDB. It ranks saved sentences by a text distance that treats
  lip-alike letters (p/b/m, f/v, t/d/n) as cheap swaps, and by the model's own CTC likelihood of
  the sentence given the lip frames, on the laptop and on the server. In an offline test with
  near-duplicate saved sentences, the model score put the right one first 49 times out of 50,
  against 41 for text distance.
- It runs live at tryheard.tech: the app is static files on Cloudflare Workers, and our ML server
  and the backend share one RunPod RTX 4090 pod behind a Cloudflare tunnel. One command recreates
  the pod in about 3 minutes.
- Our teammates built the backend: FastAPI with Google sign-in, ElevenLabs streaming speech over a
  WebSocket, ElevenLabs Scribe live transcription, an optional speaker diarizer of their own (voice
  activity detection, voiceprints, clustering), and Postgres for saved notes.
- For testing we built a 144-clip set of 62 unseen people from four public datasets, with scripts
  confirmed by Whisper, and an app test that plays clips into the real app in headless Chromium as
  a fake camera and scores the whole transcript.

## Challenges we ran into

- Cutting sentences. Our first streaming version got 72% to 90% of words wrong (measured with an
  older test setup that could also drop early clips), mostly words it never read. Sentence starts
  were clipped (fixed with a 1 s lead-in), speakers who started softly never triggered it (lower
  start threshold), slow speakers were split mid-sentence (lower threshold to keep going), a saved
  phrase overrode a correct reading (now it can only change unsure words), and the beam search
  invented fluent sentences from still lips (it now returns nothing when the CTC head hears no
  words).
- Late lip tracking. On a machine without a real GPU, MediaPipe's GPU mode ran 6 times slower than
  its CPU mode, so lips were tracked about 4 times a second and 0.7 s late, and sentences were cut
  into the next one. The app now picks the CPU mode there, measures movement over a fixed 250 ms
  window, and cuts at the last movement instead of when the pause is noticed.
- Frame rate. Offline, on 30 LRS3 clips, error rose from 38.1% at 25 fps to 48.3% at 15 fps.
  Running both face trackers on the main thread dropped capture from 28 to 16 fps, so we moved them
  into a worker.
- WebGPU. ONNX Runtime's default WebGPU backend rejected the model's 3D convolution padding, and
  the newer one that runs it took 2.8 s for a read that WASM did in about 1.2 s on our laptop, so
  the browser uses WASM.
- Fine-tuning forgot. In a rehearsal on the public GRID corpus, a plain fine-tune took our
  open-speech check from 28.6% to 70.8% wrong. Freezing BatchNorm statistics and blending back with
  the original weights brought it to 29.7%, and that recipe is what we shipped.
- Vocabulary. The model learned from TED talks and cannot spell most swear words, so it reads them
  as clipped fragments or clean look-alikes. People who can't speak still swear, so phrase memory
  includes a small set of swear phrases and repairs the fragments.
- GPUs. Two community RTX 4090 hosts failed CUDA initialisation, and a stopped pod could not
  restart once its host's GPU was taken, so the live stack now recreates its pod in one command.

## Accomplishments that we're proud of

- A fine-tuned lip reader running in a browser tab, live on the web, with camera video that never
  leaves the device.
- The app reads as well as the model alone: 30–32 word errors in Normal mode on our 20-clip app
  test against 32 for the model alone.
- A fine-tune that helps a face it never saw (57.3% to 50.5% wrong) without hurting the 62-person
  test, shipped only after it passed its gates.
- A 203 MB on-device model that stays within 1 point of the 775 MB original on every set we
  checked (28.5% against 28.6% on 100 LRS3 clips), guarded by a regression lock.
- An honest test set: 62 unseen people, with errors broken down by sentence, lip movement, camera
  angle and skin tone.

## What we learned

- The sentence matters more than the face: from 0% wrong on "I wonder what this is about" to 94%
  on "The plaintiff in school desegregation cases".
- People who barely move their lips are read worst, and a camera below the face costs a lot.
- Most of the app's own errors came from where we cut sentences. Once that was fixed, the app
  scored the same as the model on its own.
- A small dataset teaches a model fast and makes it forget open speech. Freezing BatchNorm
  statistics and blending weights recovers most of it.
- A language model helps on natural sentences and hurts on made-up ones: on GRID commands like
  "bin blue at f two now" it pulls the reading towards ordinary English.

## What's next for HearD

- Fine-tune on more people, and check skin-tone differences on a bigger set: ours showed no
  significant gap, but it is too small to settle the question.
- Run speed mode natively in a desktop app (about 0.8 s faster per sentence) and send Quality's
  crops once instead of twice (0.3–1.0 s on venue Wi-Fi).
- Move phrase memory to the backend team's TiDB store with vector search, so it follows the user
  across devices.
- Test with live webcams, in dim rooms, and with more speakers.

## Built with

TypeScript, React, Vite, Tailwind CSS, shadcn/ui, onnxruntime-web, WebAssembly, Web Workers,
IndexedDB, MediaPipe Tasks Vision, Python, PyTorch, PyTorch Lightning, ESPnet, Auto-AVSR, ONNX,
ONNX Runtime, SentencePiece, FastAPI, uvicorn, Hugging Face Hub, RunPod, Cloudflare Workers,
Cloudflare Tunnel, ElevenLabs, Whisper, Silero VAD, SpeechBrain, Google OAuth, Postgres, Docker,
Playwright, uv, pnpm.

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
| [MLH] Best .Tech Domain Name | Yes | The app is live at tryheard.tech |
| TiDB x AI Open Build | No, unless it lands | Needs a TiDB AI feature. The app's phrase store can talk to a phrase service through `VITE_PHRASES_URL`, but no TiDB service exists on master or any branch (checked 2026-10-04 11:20 PT) |
| [MLH] Best Use of Tiger Data | Ask the backend team | Their notes and voice preferences are in Postgres set by `TIMESCALE_SERVICE_URL` (Tiger Data is the company formerly called Timescale). Plain tables, no Timescale features. Qualifies only if that database runs on Tiger Cloud |
| [MLH] Best Use of Gemini API | No | Nothing calls Gemini; the corrector hook is unused |
| CSSS SFU CS Legacy Track | Only with the hunt | Needs 50% of the scavenger hunt completed |

Best Beginner, Best Highschool Hack, Best Solo and the WiCS Cosmos Track depend on who is on the
team. The rest (game, hardware, sports analytics, Solana, Snowflake, both Huawei challenges, Earth
observation) don't fit this project.

### Before submitting

- Submission: project link (https://tryheard.tech or the repo) and a video of 3 minutes or less,
  due 2026-10-04 12:00 PT.
- The app's own screens write the name as a lowercase italic "heard"; this text uses HearD.

### Where the numbers come from

| Claim | Source |
|---|---|
| 144 clips, 62 people; model alone 41.3% (int8 greedy), 35.0% (beam); everyday (CREMA-D, RAVDESS) 6.2–13.4%; VidTIMIT and MEAD 46.0–59.2%; 0% to 94% by sentence; camera below +21.6 points; skin tone: no significant gap | `.context/eval-v2.md` |
| Beam 20 / LM 0.2: 35.0% on eval v2, 0.82 s vs 1.19 s for a 3 s sentence | `.context/eval-v2.md` (round 2) |
| Fine-tune: unseen teammate 57.3% → 50.5%; LRS3-100 greedy 30.0% (limit 30.6%); 143 training clips; 0.6 min; 98 of 239 labels shifted | `.context/b2-report.md`, `.context/project-brief.md` D88, D89 |
| Fine-tune on eval v2: greedy −0.2 points; Quality keeps stock (app test 25.8% → 28.7% with the fine-tune) | `.context/project-brief.md` D90, `.context/b2-report.md` |
| App (Normal) 30–32 word errors vs 32 for the model alone; lip tracking 4 Hz and 0.7 s late, GPU mode 6× slower on software graphics | `.context/app-eval.md` (round 2) |
| First version 72.1–90.2% with the old harness | `.context/app-eval.md` (round 1) |
| int8 203 MB vs 775 MB, 28.5% vs 28.6% on LRS3-100; within 1 point on every set (worst: +0.7 in the frame-rate test) | `.context/project-brief.md` D39 and §11 |
| 1.09 s for 2.8 s of video on the demo laptop, at the app's default thread count | `frontend/bench/ort-threads/README.md` |
| 38.1% at 25 fps, 48.3% at 15 fps; capture 28 → 16 fps with both trackers | `.context/project-brief.md` §11 and D40 |
| Phrase snapping 49/50 vs 41/50 | `.context/phrase-scoring.md` |
| GRID rehearsal: 28.6% → 70.8%, frozen BN + WiSE-FT 29.7% | `.context/b2-report.md` |
| Desktop app ~0.8 s faster; single upload saves 0.3–1.0 s | `.context/eval-v2.md` (round 2, levers) |
| tryheard.tech on Cloudflare Workers, pod behind a Cloudflare tunnel, recreated in about 3 minutes | `ml/runpod/README-deploy.md` |
| WebGPU padding error; newer WebGPU backend 2.8 s vs about 1.2 s on WASM | `.context/project-brief.md` D37 and §12 (B1) |
| Community pods failing `cuInit`, stopped pod unable to restart | `.context/project-brief.md` D32, D80 |
