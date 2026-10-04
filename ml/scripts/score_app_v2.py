"""Score /app runs on the raw_eval_v2 fake-camera parts (make_eval_video_v2.py): WER of the app's
whole transcript per part and overall, like app_eval/score_eval.py, plus each error assigned to the
clip whose reference words it falls on (alignment of the joined texts), so the app can be broken down
by the same groups as the model (skin tone, pose, source, ...).

    cd ml
    uv run python scripts/score_app_v2.py normal                 # partN/eval_app_normal.json, every part
    uv run python scripts/score_app_v2.py normal instant --parts 1,2

TAG = the harness's TAG (e2e_eval.mjs writes OUT/eval_app_<TAG>.json; run it with OUT=<part dir>).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_v2_report import GROUPS, boot, fmt, norm  # noqa: E402

ML = HERE.parent


def attribute(refs: list[dict], lines: list[str]) -> tuple[list[dict], dict]:
    """Per-clip errors from one alignment of all references vs all app lines."""
    words, owner = [], []
    for i, r in enumerate(refs):
        w = norm(r["ref"]).split()
        words += w
        owner += [i] * len(w)
    hyp = " ".join(norm(x) for x in lines).split()
    out = [{"id": r["id"], "n": len(norm(r["ref"]).split()), "S": 0, "D": 0, "I": 0} for r in refs]
    m = jiwer.process_words(" ".join(words), " ".join(hyp) or "<empty>")
    for ch in m.alignments[0]:
        if ch.type == "substitute":
            for k in range(ch.ref_start_idx, ch.ref_end_idx):
                out[owner[k]]["S"] += 1
        elif ch.type == "delete":
            for k in range(ch.ref_start_idx, ch.ref_end_idx):
                out[owner[k]]["D"] += 1
        elif ch.type == "insert" and hyp:
            k = min(max(ch.ref_start_idx - 1, 0), len(owner) - 1)
            out[owner[k]]["I"] += ch.hyp_end_idx - ch.hyp_start_idx
    if not hyp:  # nothing read: all deletions (process_words saw one "<empty>" substitution)
        for o in out:
            o.update(S=0, D=o["n"], I=0)
    for o in out:
        o["errs"] = o["S"] + o["D"] + o["I"]
    tot = {k: sum(o[k] for o in out) for k in ("S", "D", "I", "errs", "n")}
    return out, tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--dir", type=Path, default=ML / "artifacts/app_eval_v2")
    ap.add_argument("--manifest", type=Path, default=ML / "data/raw_eval_v2/manifest.json")
    ap.add_argument("--parts", default="", help="comma list of part numbers (default: all with results)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    man = {c["id"]: c for c in json.loads(a.manifest.read_text())["clips"]}
    parts = sorted(a.dir.glob("part*"), key=lambda p: int(p.name[4:]))
    if a.parts:
        keep = {f"part{x}" for x in a.parts.split(",")}
        parts = [p for p in parts if p.name in keep]
    for tag in a.tags:
        clips, rows = [], []
        for p in parts:
            f = p / f"eval_app_{tag}.json"
            if not f.is_file():
                continue
            refs = json.loads((p / "eval20_refs.json").read_text())
            run = json.loads(f.read_text())
            per, tot = attribute(refs, run["lines"])
            rows.append((p.name, tot, len(run["lines"]), run.get("fpsMedian")))
            for o in per:
                m = man[o["id"]]
                g = dict(m["groups"])
                g["race"] = m.get("race", "not labelled")
                g["transcript_check"] = m.get("transcript_check", {}).get("status", "unknown")
                clips.append({**o, "speaker": m["speaker"], "groups": g})
        if not rows:
            print(f"{tag}: no results under {a.dir}/part*/eval_app_{tag}.json")
            continue
        print(f"\n## app {tag}: {len(rows)} parts, {len(clips)} clips")
        print("| part | words | WER | wrong | missed | extra | lines | fps |\n|---|---|---|---|---|---|---|---|")
        for name, t, nl, fps in rows:
            print(f"| {name} | {t['n']} | {t['errs'] / t['n']:.1%} | {t['S']} | {t['D']} | {t['I']} | {nl} | {fps} |")
        print(f"\nWER **{fmt(boot(clips, a.boot, 0))}** (95% CI, speakers resampled)")
        for key in GROUPS:
            vals = defaultdict(list)
            for c in clips:
                vals[c["groups"].get(key, "unknown")].append(c)
            if len(vals) < 2:
                continue
            print(f"\n| {key} | clips | words | WER [95% CI] |\n|---|---|---|---|")
            for v, cs in sorted(vals.items(), key=lambda kv: -len(kv[1])):
                print(f"| {v} | {len(cs)} | {sum(c['n'] for c in cs)} | {fmt(boot(cs, a.boot, 0))} |")


if __name__ == "__main__":
    main()
