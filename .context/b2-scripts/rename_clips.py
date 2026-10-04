"""Rename one-file-per-line recordings (OBS start/stop per line) to script ids, in recording order.

OBS names files by date and time (e.g. "2026-10-04 06-12-33.mkv"), so sorting by name = recording
order. Files already named like a clip id (p1_001, p2_014, …) are left alone, so renaming p2 in a
folder that still holds renamed p1 clips is safe. Delete bad takes first, then:

    python rename_clips.py ~/Videos/p1_session p1.tsv --first 1            # preview
    python rename_clips.py ~/Videos/p1_session p1.tsv --first 1 --apply    # rename

Then run make_txts.py on the same folder. Plain Python 3.
"""

import argparse
import re
from pathlib import Path

VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv"}
CLIP_ID = re.compile(r"^[a-z0-9-]+_\d{3}$")  # already renamed, by this script or for another speaker

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", type=Path)
ap.add_argument("script", type=Path)
ap.add_argument("--first", type=int, default=1, help="script line number of the first file")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

rows = [ln.split("\t", 1) for ln in a.script.read_text().splitlines() if ln.strip()]
files = sorted(p for p in a.folder.iterdir() if p.suffix.lower() in VIDEO and not CLIP_ID.match(p.stem))
stems = [f.stem for f in files]
twice = sorted({s for s in stems if stems.count(s) > 1})
if twice:  # OBS "automatically remux to mp4" keeps the .mkv too: every later id would shift by one
    raise SystemExit(f"same take in two formats: {', '.join(twice[:3])} — delete one format, then re-run")
todo = rows[a.first - 1:a.first - 1 + len(files)]
if len(todo) < len(files):
    raise SystemExit(f"{len(files)} files but only {len(todo)} script lines left from line {a.first}")
for f, (clip_id, text) in zip(files, todo):
    print(f"{f.name:32s} → {clip_id}{f.suffix}   {text.strip()}")
    if a.apply:
        f.rename(f.with_name(clip_id + f.suffix.lower()))
print(("renamed" if a.apply else "preview only (add --apply):"), len(files), "files;",
      "next --first", a.first + len(files))
