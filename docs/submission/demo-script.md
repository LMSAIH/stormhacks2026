# Demo video script (3 minutes or less)

A shot-by-shot plan built on what reads reliably: short everyday sentences in Normal mode with
phrase memory seeded with the demo lines, one natural sentence in Quality mode on the GPU pod, and
the one-tap fix for when a word comes out wrong. Aim for 2:45 so the cut stays under 3:00. Set up
with [demo-checklist.md](demo-checklist.md) first.

## Rules for the recording

- Every line on screen is what the app read in that take. No edited text and no dubbed voice: if a
  take misreads, re-take it or show the fix.
- Say that the demo lines are in phrase memory. That is the feature working, and judges will try
  sentences of their own.
- The presenter mouths silently with the recording mic muted, so viewers can hear there is no voice
  until the app speaks.
- Keep the camera panel large enough to see the lips.
- Record the app segments as long takes first, pick the best ones, then record the voiceover.

## Demo lines

From the shared demo phrases (`.context/b2-scripts/p1.md`, lines 001–012), all seeded in phrase
memory:

- Hi, nice to meet you
- Can you help me please
- I would like a glass of water
- Where is the bathroom
- Thank you so much
- Spares if one misreads in rehearsal: "Please call my family", "I need a few more minutes",
  "Can you repeat that more slowly"

For Quality mode, one natural sentence that is not in phrase memory, chosen in rehearsal from
`p1.md` lines 013–040 (for example "Please turn off the lights when you leave").

## Shots

| # | Time | On screen | Audio |
|---|---|---|---|
| 1 | 0:00–0:12 | Presenter mouths "Hi, nice to meet you". The line appears and is spoken. Caption: "No sound. The laptop reads the lips." | App voice only |
| 2 | 0:12–0:30 | Title card "HearD", then the app idle with the lip dots on the face | Voiceover 2 |
| 3 | 0:30–1:05 | Normal mode. Mouth "Can you help me please", "I would like a glass of water", "Where is the bathroom". Grey drafts appear while mouthing; each line locks after the pause and is spoken | App voice, voiceover 3 between lines |
| 4 | 1:05–1:25 | A finished line with a dashed box on an unsure word. Click it, pick the right reading or type it; the fixed line is spoken again | Voiceover 4 |
| 5 | 1:25–1:45 | Mouth the fixed sentence again: it comes out right | App voice, voiceover 5 |
| 6 | 1:45–2:05 | Open the mode menu (Local and Cloud labels), switch to Quality, mouth the natural sentence; the reading comes back from the GPU server | App voice, voiceover 6 |
| 7 | 2:05–2:20 | A teammate speaks off camera; their words appear as captions in the right panel | Teammate's voice, voiceover 7 |
| 8 | 2:20–2:45 | The [system diagram](../architecture/system.md), then a text card with the numbers below | Voiceover 8 |
| 9 | 2:45–2:58 | Text card with the limits, team names, tryheard.tech and the repo link | Voiceover 9 |

Shot 4 needs a line with a box. Boxes appear on words the model was unsure of, so mouth a sentence
that is not in phrase memory if the seeded lines all come out clean. Shot 7 needs the backend's
listening panel running; cut it if it isn't, and give the time to shot 3.

Numbers card for shot 8 (from `.context/eval-v2.md`, `.context/b2-report.md` and `.context/project-brief.md` D39):

- Camera video never leaves the browser; Quality sends mouth crops only
- 203 MB fine-tuned int8 model running in the browser (775 MB original)
- Tested on 62 people the model never saw: 35.0% (server) to 41.3% (laptop) of words wrong, and
  6.2–13.4% on everyday sentences filmed straight on
- Fine-tuned on one teammate, it read another teammate it never saw better: 57.3% → 50.5% wrong
- Live at tryheard.tech

## Voiceover

2. "Some people can move their lips but can't make a sound, after a laryngectomy or on a
   ventilator. Typing every sentence is slow. HearD reads your lips through a webcam and speaks
   for you."
3. "It starts when my lips move and ends the sentence when they stop. The grey text is a quick
   draft; when I pause, the laptop rereads the whole sentence and speaks it. This runs in the
   browser. These lines are in my phrase memory, because I say them every day."
4. "When the model isn't sure of a word, it draws a box. I click it and pick the right word, or
   type it. The fixed sentence is spoken and saved."
5. "Next time, if a reading is close and the model was unsure of the words that differ, the app
   uses my saved sentence. It never changes a word the model was sure of."
6. "Quality mode sends only the mouth crops, not the video, to our GPU server, which runs a slower
   search with a language model. If the server is down, the laptop reads the sentence instead."
7. "The other side of the conversation is captioned for me."
8. "Under the hood: MediaPipe face tracking in a background worker, a mouth crop that matches the
   model's training code exactly, and the Auto-AVSR lip-reading model, fine-tuned on our team and
   shrunk to 203 megabytes for the browser. We tested it on 62 people it has never seen: on
   everyday sentences it gets 86 to 93 percent of the words right; on long sentences with rare
   words, around half."
9. "It's English only, it reads sentence by sentence, and the model's licence is for research only.
   Try it at tryheard.tech. Thanks for watching."

## If live reading misfires

1. While recording: re-take. Phrase memory is seeded, so in Normal mode a close miss on a demo line
   can snap to the saved sentence: that happens when the model rates the saved sentence nearly as
   likely as its own reading (margin ≥ −0.2) and was unsure (below 0.9) of every word that would
   change.
2. A line still comes out wrong: keep the take and use it for shot 4. Fixing a word is part of the
   product.
3. Several misses in a row: check the amber face hint and the light, re-run the checklist's camera
   line (aim for 25–30 fps), and pause a full second between sentences.
4. For natural sentences that aren't in phrase memory, switch to Quality: on 62 unseen people the
   server's beam + LM got 35.0% of words wrong against 41.3% for the laptop read.
5. Server down: if it was already down when the page loaded, the mode menu says "Server offline".
   If it drops later, nothing on screen changes: Quality lines are read on the laptop and only the
   console logs `cloud read failed`. Check `/health` right before shot 6; if the server is down, cut
   the shot or say the laptop read it. Recovery steps are in the checklist.
6. Live judging where nothing reads: play this recorded video.
