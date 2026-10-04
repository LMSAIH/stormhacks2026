"""Join the first 20 raw_eval clips (same order as bench.py) into one fake-camera y4m: 30 s still
lead (model load), each clip, then 1.5 s still pause. Also writes the reference list."""
import json
import subprocess
from pathlib import Path

ML = Path(__file__).resolve().parents[2]
ROOT = ML / "data/raw_eval"
OUT = ML / "artifacts/app_eval"
OUT.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = {".mp4", ".webm", ".mov"}
clips = [p for p in sorted(ROOT.iterdir()) if p.suffix.lower() in VIDEO_EXT and p.with_suffix(".txt").is_file()][:20]

inputs, chains = [], []
for i, p in enumerate(clips):
    inputs += ["-i", str(p)]
    lead = 30 if i == 0 else 0
    chains.append(
        f"[{i}:v]fps=30,scale=640:480:force_original_aspect_ratio=decrease,"
        f"pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"tpad=start_mode=clone:start_duration={lead}:stop_mode=clone:stop_duration=1.5[v{i}]"
    )
concat = "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[out]"
cmd = ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chains + [concat]),
       "-map", "[out]", "-pix_fmt", "yuv420p", str(OUT / "eval20.y4m")]
subprocess.run(cmd, check=True)
refs = [{"id": p.stem, "ref": p.with_suffix(".txt").read_text().strip()} for p in clips]
(OUT / "eval20_refs.json").write_text(json.dumps(refs, indent=1))
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                      str(OUT / "eval20.y4m")], capture_output=True, text=True).stdout.strip()
print(len(clips), "clips,", dur, "s")
