# B2 recording guide: clips for fine-tuning

What `ml/scripts/prepare_finetune_data.py` expects, and what makes the fine-tune help without
hurting unseen faces (lessons from the GRID rehearsal in `b2-report.md`). Upload by ~07:30 PT.

## One person (the plan now, D76)

| Session | Script | Clips | Used for |
|---|---|---|---|
| A | `p1.md` then `p2.md` | 160 | training |
| B, **another time or light** | `p3.md`, clips **001–040 only** | 40 | held-out test (`HOLDOUT=p3`) |

~45–60 min in total. Keep the `p1_`/`p2_`/`p3_` names. Session B's 001–012 are the demo phrases (same
text as in training: "does it read the demo lines in a new session?"); 013–040 are sentences the
model never trains on ("does it generalise?"). Both are reported.

More people instead: each takes their own `pN.md` (all 80 lines); one whole person is held out.

## Before you start
- Same kind of camera and distance as the demo: laptop webcam (ideally the demo laptop), ~50–70 cm,
  face filling ~⅓ of the frame, frontal, mouth fully visible, no hands near the face.
- Bright light on your face, so the webcam holds 30 fps (15 fps costs ~10 WER points).
- OBS: Video → 30 fps, output 1280×720 is plenty; Output → recording format mkv or mp4 (both read at
  the right frame rate); a hotkey each for Start and Stop Recording; the mic track can be off.
- Put the script window right under the camera.

## Each clip
1. Read the line on screen first.
2. Press **Start**, look at the lens, keep your lips closed and still for **~1 s**.
3. **Mouth the line silently**, a little exaggerated, the way you'll use the app.
4. Lips closed and still for **~1 s**, then press **Stop**.
5. Bad take: delete that file at once, then redo the line.

Why the pauses: the app now feeds the model 1 s of still lips before speech (`LEAD_MS`) and locks a
sentence after 0.8 s of stillness, so training clips should look the same.

Vary a little: at clip 041 of each script, switch a lamp, glasses on/off, or the background. Heads
turned ≤ 15°. Clips with a face in < 50% of frames are skipped.

## After each session (per folder)
```bash
python .context/b2-scripts/rename_clips.py ~/Videos/p1 .context/b2-scripts/p1.tsv --first 1          # preview
python .context/b2-scripts/rename_clips.py ~/Videos/p1 .context/b2-scripts/p1.tsv --first 1 --apply
python .context/b2-scripts/make_txts.py .context/b2-scripts/p1.tsv ~/Videos/p1
```
Check the preview lines each file up with the right sentence (a shift = a bad take wasn't deleted).
`make_txts.py` should print 80/80 for p1 and p2; for p3 it prints 40/80 and lists 041–080 as missing
(fine). `rename_clips.py` maps OBS's date-time filenames to ids in recording order.

Alternatives: one long take per 10–20 lines with a desk tap (out of frame) before each line and after
the last, then `split_takes.py take.mp4 pN.tsv --first N --out recordings/` (`--dry-run` first;
needs ffmpeg; quiet room). Pausing in OBS leaves no marker, so don't rely on pauses alone.

## Files and upload
```
recordings/          ← one flat folder, no subfolders
  p1_001.mkv  p1_001.txt   ← the .txt holds exactly the words mouthed, one line
  ...
```
```bash
pip install -U huggingface_hub && hf auth login
hf repo create stormhacks-lipread-recordings --repo-type dataset --private
hf upload eschmechel/stormhacks-lipread-recordings ./recordings . --repo-type dataset
```
Upload the first ~10 clips early: Claude can run the prep on them (no GPU) and catch frame-rate,
framing or naming problems before you record the rest. Private repo, kept apart from the public
training-pairs dataset (D61, D75).

## Content rules (already in the scripts)
- 12 demo phrases × 2 wordings in every script (001–012, 041–052). **If the demo uses other lines,
  swap them into all `pN.tsv` (and `.md`) before recording**: that's where the fine-tune helps most.
- The rest are everyday sentences, different per script. No fixed templates (GRID showed a narrow
  grammar is learned instantly and breaks open speech). Numbers spelled out.
- Speaker ids: lowercase letters, digits, hyphens, no underscores; `pairs` is reserved.

## Consent and licence
Everyone recorded agrees to it, in a private repo, for this hackathon. The fine-tuned model inherits
19.1's research / non-commercial (LRS3) terms.
