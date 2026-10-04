"""WER on raw_eval_v2 overall and per group (speaker, sex, age, skin tone, lighting, pose, crop size,
motion, source), with 95% bootstrap intervals that resample speakers, not clips (clips of one person
are not independent).

    cd ml
    uv run python scripts/bench.py --clips data/raw_eval_v2 --n 1000 --backend onnx \\
        --onnx-path <int8 model> --tag v2-int8           # + once per gated subfolder (--clips data/raw_eval_v2/mead ...)
    uv run python scripts/eval_v2_report.py --run "int8 greedy=artifacts/bench/v2-int8*-greedy.json" \\
        --run "pod beam=artifacts/bench/v2-beam*-beam.json" --out artifacts/eval_v2_report.json

A --run takes a label and one or more bench.py JSONs (glob), merged by clip id. Prints markdown.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import jiwer

ML = Path(__file__).resolve().parents[1]
GROUPS = ("source", "sex", "age_band", "skin_tone", "race", "lighting", "pose", "crop", "motion",
          "transcript_source")


def norm(s: str) -> str:  # same as bench.norm
    import re
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_run(spec: str) -> tuple[str, dict[str, dict]]:
    label, pattern = spec.split("=", 1)
    rows = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"--run {label}: no files match {pattern}")
    for f in files:
        for r in json.loads(Path(f).read_text())["rows"]:
            rows[r["id"]] = r
    return label, rows


def score(ref: str, hyp: str) -> tuple[int, int, dict]:
    ref, hyp = norm(ref), norm(hyp)
    m = jiwer.process_words(ref, hyp or "<empty>")
    errs = m.substitutions + m.deletions + m.insertions
    if not hyp:  # "<empty>" counts as one substitution; an empty read is all deletions
        errs = len(ref.split())
    return errs, len(ref.split()), {"S": m.substitutions, "D": m.deletions, "I": m.insertions}


def boot(clips: list[dict], b: int, seed: int) -> tuple[float, float, float]:
    """Corpus WER and a 95% percentile interval, resampling speakers with replacement."""
    by = defaultdict(lambda: [0, 0])
    for c in clips:
        by[c["speaker"]][0] += c["errs"]
        by[c["speaker"]][1] += c["n"]
    spk = list(by.values())
    e, n = sum(x[0] for x in spk), sum(x[1] for x in spk)
    if len(spk) < 2:
        return e / n, float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(b):
        s = [spk[rng.randrange(len(spk))] for _ in spk]
        stats.append(sum(x[0] for x in s) / max(1, sum(x[1] for x in s)))
    stats.sort()
    return e / n, stats[int(0.025 * b)], stats[int(0.975 * b) - 1]


def fmt(w: tuple[float, float, float]) -> str:
    lo, hi = w[1], w[2]
    ci = "" if lo != lo else f" [{lo:.1%}, {hi:.1%}]"
    return f"{w[0]:.1%}{ci}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ML / "data/raw_eval_v2/manifest.json")
    ap.add_argument("--run", action="append", required=True, help='"label=glob of bench JSONs"')
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worst", type=int, default=12)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    man = {c["id"]: c for c in json.loads(a.manifest.read_text())["clips"]}
    runs = dict(load_run(s) for s in a.run)
    report = {"manifest": str(a.manifest), "runs": {}}
    for label, rows in runs.items():
        clips, missing = [], []
        for cid, m in man.items():
            r = rows.get(cid)
            if r is None:
                missing.append(cid)
                continue
            errs, n, ops = score(m["transcript"], r["hyp"] if not r.get("error") else "")
            g = dict(m["groups"])
            g["race"] = m.get("race", "unknown")
            g["transcript_source"] = m.get("transcript_source", "script")
            clips.append({"id": cid, "speaker": m["speaker"], "errs": errs, "n": n, **ops,
                          "ref": m["transcript"], "hyp": r["hyp"], "error": r.get("error"), "groups": g})
        print(f"\n## {label}: {len(clips)} clips, {len({c['speaker'] for c in clips})} speakers, "
              f"{sum(c['n'] for c in clips)} words" + (f" (missing {len(missing)}: {missing[:5]}…)" if missing else ""))
        overall = boot(clips, a.boot, a.seed)
        S, D, I = (sum(c[k] for c in clips) for k in "SDI")
        print(f"WER **{fmt(overall)}** (95% CI, speakers resampled) · substitutions {S}, deletions {D}, "
              f"insertions {I} · empty reads {sum(1 for c in clips if not norm(c['hyp']))} · errors "
              f"{sum(1 for c in clips if c['error'])}")
        out = {"n_clips": len(clips), "wer": overall, "S": S, "D": D, "I": I, "groups": {}, "missing": missing}
        for key in GROUPS:
            vals = defaultdict(list)
            for c in clips:
                vals[c["groups"].get(key, "unknown")].append(c)
            if len(vals) < 2:
                continue
            print(f"\n| {key} | clips | speakers | words | WER [95% CI] |\n|---|---|---|---|---|")
            out["groups"][key] = {}
            for v, cs in sorted(vals.items(), key=lambda kv: -len(kv[1])):
                w = boot(cs, a.boot, a.seed)
                out["groups"][key][v] = {"clips": len(cs), "speakers": len({c["speaker"] for c in cs}),
                                         "words": sum(c["n"] for c in cs), "wer": w}
                print(f"| {v} | {len(cs)} | {len({c['speaker'] for c in cs})} | {sum(c['n'] for c in cs)} | {fmt(w)} |")
        spk = defaultdict(list)
        for c in clips:
            spk[c["speaker"]].append(c)
        sw = sorted(((sum(c["errs"] for c in cs) / sum(c["n"] for c in cs), s, cs) for s, cs in spk.items()),
                    reverse=True)
        print(f"\nWorst speakers: " + ", ".join(f"{s} {w:.0%}" for w, s, _ in sw[: a.worst]))
        print(f"Best speakers: " + ", ".join(f"{s} {w:.0%}" for w, s, _ in sw[-6:]))
        out["speakers"] = {s: w for w, s, _ in sw}
        worst = sorted(clips, key=lambda c: (-c["errs"] / c["n"], -c["n"]))[: a.worst]
        print("\n| clip | WER | reference | read |\n|---|---|---|---|")
        for c in worst:
            print(f"| {c['id']} | {c['errs'] / c['n']:.0%} | {c['ref']} | {c['hyp'] or '(empty)'} |")
        out["clips"] = clips
        report["runs"][label] = out
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
