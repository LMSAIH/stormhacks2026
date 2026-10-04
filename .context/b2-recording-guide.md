# B2 recording guide: team clips for fine-tuning

What `ml/scripts/prepare_finetune_data.py` expects, and what makes the fine-tune actually help
without hurting unseen faces (lessons from the GRID rehearsal in `b2-report.md`).

## Who and how many

| | Minimum | Good |
|---|---|---|
| Speakers | 3 (2 train + 1 held out) | 4–5 |
| Clips per speaker | 40 | 80–100 |
| Clip length | 2–6 s, one sentence | same |

One whole speaker is held out as the unseen-face test. Pick someone whose face is typical of the
group (not the only one with a beard or glasses); tell Claude who.

## What to say (per speaker)

- **~30% shared demo phrases**: the exact lines from the demo script, each 2–3 times with small
  variations. Every speaker records these.
- **~70% open sentences, different per speaker**: use the Harvard sentences (public domain,
  720 phonetically balanced lines, e.g. "The birch canoe slid on the smooth planks"). Give each
  speaker their own lists (speaker A lists 1–8, B 9–16, …) so the held-out speaker's are new.
- No fixed templates. GRID showed a narrow grammar is learned instantly and breaks open speech.
- Spell out numbers ("twenty five", not "25"); no abbreviations.

## How to record

- **Mouth silently, the way the demo is used.** Silent mouthing looks different from speaking aloud.
  Exaggerate a little, as the user would.
- Same kind of camera and distance as the demo (laptop webcam, ~50–70 cm, face filling ~⅓ of the
  frame), frontal, mouth fully visible, no hands near the face.
- **25–30 fps.** Turn off low-light frame-rate drops (keep the room bright). 15 fps costs ~10 WER points.
- 720p is plenty. Any of mp4, mov, webm, mkv, avi.
- ~0.3 s neutral, closed mouth before and after each sentence.
- Vary a little across the session: two lighting setups, glasses on/off if you wear them, slight
  head turns (≤15°), different backgrounds.
- Redo a clip if the face leaves the frame. Clips with a face in <50% of frames are skipped.

## Scripts (4 people × 80 clips)

`.context/b2-scripts/p1.md` … `p4.md`: one per person, read while recording. Clips 001–012 and
041–052 are the 12 shared demo phrases (two wordings); the other 56 are open sentences unique to
that person. Swap in your real demo lines in all four files if they differ. After recording, run
`python .context/b2-scripts/make_txts.py p1.tsv <folder>` to write every `.txt` automatically.

## Files

```
recordings/                 ← one flat folder, no subfolders
  alice_001.mp4
  alice_001.txt             ← exactly the words mouthed, one line: The birch canoe slid on the smooth planks
  alice_002.mp4
  ...
  bob-k_001.mp4             ← speaker id: lowercase letters, digits, hyphens; no underscores
```

Case and punctuation in the `.txt` don't matter (they're normalised); the words do.

## Upload (private)

```bash
pip install -U huggingface_hub            # provides the `hf` CLI
hf auth login
hf repo create stormhacks-lipread-recordings --repo-type dataset --private
hf upload eschmechel/stormhacks-lipread-recordings ./recordings . --repo-type dataset
```

Upload a first batch of ~10 clips early: Claude can run the prep on them (no GPU) and catch
frame-rate, framing or naming problems before everyone records the rest.

## Consent and licence

Everyone recorded agrees to it, in a private repo, for this hackathon. The fine-tuned model inherits
19.1's research / non-commercial (LRS3) terms.

## Timing

Phase 2 is ~30–45 min of GPU (prep → fine-tune → blends → bench). The hard stop for a shippable
model is 09:00 PT, so upload by ~07:30 PT. Stopped pods can fail to restart (host GPU taken), so
allow for creating a new pod.
