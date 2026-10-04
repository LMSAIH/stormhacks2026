"""Rename one-file-per-line recordings (OBS start/stop per line) to script ids, in recording order.

OBS names files by date and time (e.g. "2026-10-04 06-12-33.mkv"), so sorting by name = recording
order. Delete bad takes first, then:

    python rename_clips.py ~/Videos/p1_session p1.tsv --first 1            # preview
    python rename_clips.py ~/Videos/p1_session p1.tsv --first 1 --apply    # rename

Then run make_txts.py on the same folder. Plain Python 3.
"""

import argparse
from pathlib import Path

VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", type=Path)
ap.add_argument("script", type=Path)
ap.add_argument("--first", type=int, default=1, help="script line number of the first file")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

rows = [ln.split("\t", 1) for ln in a.script.read_text().splitlines() if ln.strip()]
ids = {r[0] for r in rows}
files = sorted(p for p in a.folder.iterdir() if p.suffix.lower() in VIDEO and p.stem not in ids)
todo = rows[a.first - 1:a.first - 1 + len(files)]
if len(todo) < len(files):
    raise SystemExit(f"{len(files)} files but only {len(todo)} script lines left from line {a.first}")
for f, (clip_id, text) in zip(files, todo):
    print(f"{f.name:32s} → {clip_id}{f.suffix}   {text.strip()}")
    if a.apply:
        f.rename(f.with_name(clip_id + f.suffix.lower()))
print(("renamed" if a.apply else "preview only (add --apply):"), len(files), "files;",
      "next --first", a.first + len(files))
