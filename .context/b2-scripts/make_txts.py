"""Write <id>.txt next to each recorded <id>.<video> from a speaker script, and list what's missing.

    python make_txts.py p1.tsv ~/recordings        # plain Python 3, no installs

Rename a speaker id first if you like (e.g. p1 → alice): sed -i 's/^p1_/alice_/' p1.tsv
"""

import sys
from pathlib import Path

VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

script, folder = Path(sys.argv[1]), Path(sys.argv[2])
rows = [line.split("\t", 1) for line in script.read_text().splitlines() if line.strip()]
videos = {p.stem: p for p in folder.iterdir() if p.suffix.lower() in VIDEO}
missing = []
for clip_id, text in rows:
    if clip_id in videos:
        (folder / f"{clip_id}.txt").write_text(text.strip() + "\n")
    else:
        missing.append(clip_id)
prefix = rows[0][0].rsplit("_", 1)[0] + "_"
extra = sorted(s for s in videos if s.startswith(prefix) and s not in dict(rows))
print(f"{len(rows) - len(missing)}/{len(rows)} transcripts written to {folder}")
if missing:
    print("missing videos:", " ".join(missing))
if extra:
    print("videos not in the script (check the name):", " ".join(extra))
