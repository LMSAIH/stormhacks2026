# Eval v2: unseen faces (2026-10-04)

**Session 3 (frontend/app-gaps) uses this set.** Clips: private HF dataset
`eschmechel/stormhacks-lipread-eval`, folder `raw_eval_v2/` (`hf download eschmechel/stormhacks-lipread-eval
--repo-type dataset --include "raw_eval_v2/*" --local-dir ml/data`). Builder: `ml/scripts/build_eval_v2.py`;
report: `ml/scripts/eval_v2_report.py`; dataset survey and what needs your clicks: `eval-v2-datasets.md`.

> **Status: v2.0 (first usable cut).** 68 clips, 38 speakers, 385 words from CREMA-D + RAVDESS.
> VidTIMIT (TIMIT sentences, office light) and MEAD (same sentence filmed from 7 angles) are being
> added; this file and the fake-camera video follow within the hour.

## v2.0 numbers (model alone, 68 clips)
| path | WER [95% CI, speakers resampled] |
|---|---|
| int8 greedy, local ORT CPU (speed mode) | 12.5% [8.3, 17.4] |
| pod beam 40 + LM via `/lipread/crops` (accuracy mode) | 9.1% [5.3, 13.1] |

Lower than raw_eval (25.4% / 29.5%): v1 had 6 GRID clips (command grammar) and only 20 clips. These are
studio faces reading short everyday sentences, so treat v2.0 as the easy end; VidTIMIT and MEAD add the hard end.
