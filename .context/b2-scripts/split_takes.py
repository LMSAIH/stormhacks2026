"""Split one long recording into per-line clips, using desk taps on the audio track as markers.

How to record a take: start recording, TAP the desk (out of frame), mouth line 1 silently, TAP,
mouth line 2, TAP, …, mouth the last line, TAP, stop. So a take with N lines has N+1 taps.
Keep the room quiet; mouthing is silent, so the taps are the only loud sounds.
Do 10–20 lines per take: if a tap is missed, only that take needs redoing.

    python split_takes.py take1.mp4 p1.tsv --first 1  --out recordings/
    python split_takes.py take2.mp4 p1.tsv --first 21 --out recordings/    # continues at p1_021

Needs Python 3 and ffmpeg on PATH; nothing else. Check the printed list: the number of lines found
must match what you mouthed. Then run make_txts.py on the output folder.
"""

from __future__ import annotations

import argparse
import array
import subprocess
import sys
from pathlib import Path

RATE = 16000
HOP = 160  # 10 ms


def audio_energy(video: Path) -> list[float]:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        check=True, capture_output=True).stdout
    s = array.array("h", pcm)
    return [sum(x * x for x in s[i:i + HOP]) / HOP for i in range(0, len(s) - HOP, HOP)]


def find_taps(energy: list[float], min_gap_s: float, k: float) -> list[float]:
    """Tap onsets in seconds: frames far above the noise floor, at least min_gap_s apart."""
    srt = sorted(energy)
    floor = srt[len(srt) // 2]
    loud = srt[int(len(srt) * 0.999)]
    thr = max(floor * k, floor + 0.15 * (loud - floor))
    taps, last = [], -1e9
    for i, e in enumerate(energy):
        t = i * HOP / RATE
        if e > thr and t - last >= min_gap_s:
            taps.append(t)
            last = t
    return taps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("script", type=Path, help="pN.tsv")
    ap.add_argument("--first", type=int, default=1, help="script line number of the take's first line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-gap", type=float, default=1.0, help="seconds between taps, at least")
    ap.add_argument("--sensitivity", type=float, default=20.0, help="lower = picks up quieter taps")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be cut")
    a = ap.parse_args()

    rows = [ln.split("\t", 1) for ln in a.script.read_text().splitlines() if ln.strip()]
    taps = find_taps(audio_energy(a.video), a.min_gap, a.sensitivity)
    n = len(taps) - 1
    if n < 1:
        sys.exit(f"found {len(taps)} tap(s); need one before every line and one after the last")
    todo = rows[a.first - 1:a.first - 1 + n]
    if len(todo) < n:
        sys.exit(f"found {n} lines but the script only has {len(todo)} left from line {a.first}")
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(taps)} taps → {n} lines ({todo[0][0]} … {todo[-1][0]})")
    for (clip_id, text), t0, t1 in zip(todo, taps, taps[1:]):
        start, end = t0 + 0.25, t1 - 0.05  # skip the tap itself
        warn = "  ← check: unusually short/long" if not 1.0 <= end - start <= 10.0 else ""
        print(f"  {clip_id}  {start:7.2f}–{end:7.2f}s ({end - start:4.1f}s)  {text.strip()}{warn}")
        if not a.dry_run:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                            "-i", str(a.video), "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                            "-pix_fmt", "yuv420p", str(a.out / f"{clip_id}.mp4")], check=True)
    print(f"next take starts at --first {a.first + n}")


if __name__ == "__main__":
    main()
