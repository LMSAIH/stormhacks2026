"""Where did the app cut, and what did it read? Maps a run's dev trace onto the eval video's clips.

  uv run python scripts/app_eval/cuts.py mytag [mytag2 ...]

Per sentence the app locked: its span in video seconds, why it ended (pause / cap / lips-gone),
how late the lock fired after the last movement, which clips it covers (share of each clip), and
the reading. A clip split over two sentences, or a sentence reaching into the next clip, is a
cutting bug; a clip read whole but wrong is the model. Needs e2e_eval.mjs output with `origin`.
"""

import json
import statistics
import subprocess
import sys
from functools import cache
from pathlib import Path

ML = Path(__file__).resolve().parents[2]
T = ML / "artifacts/app_eval"
LEAD_S, PAUSE_S = 30.0, 1.5  # make_eval_video.py: still lead before the first clip, still after each


@cache
def clips() -> list[dict]:
    """Start/end of each clip in the eval video, from the source clips' frame counts."""
    out, t = [], LEAD_S
    for r in json.loads((T / "eval20_refs.json").read_text()):
        src = ML / "data/raw_eval" / f"{r['id']}.mp4"
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames,r_frame_rate", "-of", "json", str(src)],
            capture_output=True, text=True, check=True).stdout)["streams"][0]
        num, den = map(int, probe["r_frame_rate"].split("/"))
        dur = int(probe["nb_read_frames"]) * den / num
        out.append({**r, "start": t, "end": t + dur})
        t += dur + PAUSE_S
    return out


def covers(a: float, b: float) -> str:
    parts = []
    for i, c in enumerate(clips()):
        overlap = min(b, c["end"]) - max(a, c["start"])
        if overlap > 0.05:
            parts.append(f"{i}:{overlap / (c['end'] - c['start']):.0%}")
    return " ".join(parts) or "-"


def report(tag: str) -> None:
    run = json.loads((T / f"eval_app_{tag}.json").read_text())
    trace = json.loads((T / f"trace_{tag}.json").read_text())
    origin = run.get("origin")
    if origin is None:
        sys.exit(f"{tag}: no origin (re-run with the current e2e_eval.mjs)")
    sec = lambda ms: (ms - origin) / 1000  # noqa: E731
    reads = [e for e in trace if e["kind"] in ("final", "drop") and "startTms" in e]
    print(f"=== {tag}  (tracker {run.get('tracker')})")
    for lock in (e for e in trace if e["kind"] == "lock"):
        end = lock.get("endTms", lock["tMs"])
        late = (lock["tMs"] - lock["lastActiveTms"]) / 1000
        print(f"{lock['reason']:9s} {sec(lock['startTms']):7.2f}-{sec(end):7.2f}  lock +{late:.2f}s  "
              f"clips {covers(sec(lock['startTms']), sec(end))}")
        for e in reads:  # a read starts at or after its sentence's start (the buffer may begin later)
            if lock["startTms"] - 1 <= e["startTms"] <= end:
                said = f"{e['read']!r} -> {e['shown']!r}" if e["kind"] == "final" else f"DROP {e['reason']}"
                print(f"{'':10s}{said}{'  (snapped)' if e.get('snapped') else ''}")
    lags = [e["at"] - e["tMs"] for e in trace if e["kind"] == "result"]
    if lags:
        print(f"tracker lag median {statistics.median(lags):.0f} ms")


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        report(tag)
