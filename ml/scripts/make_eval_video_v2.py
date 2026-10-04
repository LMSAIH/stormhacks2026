"""Fake-camera videos of raw_eval_v2 for the /app regression, in the format of
app_eval/make_eval_video.py: 640x480 30 fps y4m, 30 s still lead-in (model load), each clip, then a
1.5 s still pause.

~120 clips would be one 9-minute, 8 GB y4m, so the set is cut into parts of about 25 clips. Clips of
the same sentence go to different parts (and never back to back), so the app's phrase memory can't
profit from repeats within one run. Each part folder holds the file names the existing harness uses:

  artifacts/app_eval_v2/partN/eval20.y4m        the video (name kept for e2e_eval.mjs)
  artifacts/app_eval_v2/partN/eval20_refs.json  [{id, ref}] in play order
  artifacts/app_eval_v2/index.json              parts, clip order, durations, WATCH_S per part

    cd ml
    uv run python scripts/make_eval_video_v2.py                    # all parts
    OUT=$PWD/artifacts/app_eval_v2/part1 WATCH_S=<index.json watch_s> MODE=normal TAG=v2p1-normal \\
      node scripts/app_eval/e2e_eval.mjs                            # writes partN/eval_app_<TAG>.json
    uv run python scripts/score_app_v2.py v2p1-normal ...           # per part + per group
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
LEAD_S, PAUSE_S = 30, 1.5


def sentence_key(t: str) -> str:
    return re.sub(r"[^a-z ]+", "", t.lower()).strip()


def plan(clips: list[dict], n_parts: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    by_sent = defaultdict(list)
    for c in clips:
        by_sent[sentence_key(c["transcript"])].append(c)
    parts: list[list[dict]] = [[] for _ in range(n_parts)]
    # biggest sentence groups first, each copy to the currently smallest part without that sentence
    for _, cs in sorted(by_sent.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rng.shuffle(cs)
        for c in cs:
            free = [p for p in parts if all(sentence_key(x["transcript"]) != sentence_key(c["transcript"]) for x in p)]
            target = min(free or parts, key=len)
            target.append(c)
    for p in parts:  # shuffle, then push apart any back-to-back repeats
        rng.shuffle(p)
        for i in range(1, len(p)):
            if sentence_key(p[i]["transcript"]) == sentence_key(p[i - 1]["transcript"]):
                for j in range(len(p)):
                    if all(sentence_key(p[j]["transcript"]) != sentence_key(p[k]["transcript"])
                           for k in (i - 1, i + 1) if 0 <= k < len(p)):
                        p[i], p[j] = p[j], p[i]
                        break
    return parts


def render(part: list[dict], root: Path, out: Path) -> float:
    out.mkdir(parents=True, exist_ok=True)
    inputs, chains = [], []
    for i, c in enumerate(part):
        inputs += ["-i", str(root / c["file"])]
        lead = LEAD_S if i == 0 else 0
        chains.append(
            f"[{i}:v]fps=30,scale=640:480:force_original_aspect_ratio=decrease,"
            f"pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"tpad=start_mode=clone:start_duration={lead}:stop_mode=clone:stop_duration={PAUSE_S}[v{i}]")
    concat = "".join(f"[v{i}]" for i in range(len(part))) + f"concat=n={len(part)}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                    ";".join(chains + [concat]), "-map", "[out]", "-pix_fmt", "yuv420p",
                    str(out / "eval20.y4m")], check=True)
    (out / "eval20_refs.json").write_text(json.dumps(
        [{"id": c["id"], "ref": c["transcript"]} for c in part], indent=1))
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(out / "eval20.y4m")], capture_output=True, text=True).stdout.strip()
    return float(dur)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ML / "data/raw_eval_v2")
    ap.add_argument("--out", type=Path, default=ML / "artifacts/app_eval_v2")
    ap.add_argument("--clips-per-part", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    clips = json.loads((a.root / "manifest.json").read_text())["clips"]
    n_parts = max(1, round(len(clips) / a.clips_per_part))
    parts = plan(clips, n_parts, a.seed)
    index = {"source": str(a.root), "lead_s": LEAD_S, "pause_s": PAUSE_S, "parts": []}
    for k, part in enumerate(parts, 1):
        d = a.out / f"part{k}"
        seconds = sum(c["video"]["seconds"] for c in part) + LEAD_S + PAUSE_S * len(part)
        if not a.plan_only:
            seconds = render(part, a.root, d)
        src = defaultdict(int)
        for c in part:
            src[c["source_name"]] += 1
        index["parts"].append({"dir": d.name, "clips": [c["id"] for c in part], "seconds": round(seconds, 1),
                               "watch_s": int(seconds + 15), "words": sum(c["words"] for c in part),
                               "by_source": dict(src)})
        print(f"part{k}: {len(part)} clips, {sum(c['words'] for c in part)} words, {seconds:.0f} s, {dict(src)}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
