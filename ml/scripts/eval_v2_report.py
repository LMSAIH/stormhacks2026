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
          "transcript_check")


def norm(s: str) -> str:  # same as bench.norm
    import re
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_EXPAND = {"it's": "it is", "that's": "that is", "i'm": "i am", "i've": "i have", "we'll": "we will",
           "don't": "do not", "can't": "cannot", "won't": "will not", "you're": "you are",
           "we're": "we are", "they're": "they are", "isn't": "is not", "didn't": "did not",
           "dont": "do not", "cant": "cannot", "isnt": "is not", "didnt": "did not", "thats": "that is",
           "im": "i am"}  # the model sometimes drops the apostrophe


def _num(w: str) -> str:
    if not w.isdigit() or int(w) > 99:
        return w
    n = int(w)
    return _ONES[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else " " + _ONES[n % 10])


def lenient(s: str) -> str:
    """Spelling-only differences don't count: 11 = eleven, to morrow = tomorrow, it's = it is."""
    s = norm(s).replace("to morrow", "tomorrow")
    return " ".join(_EXPAND.get(w, _num(w)) for w in s.split())


LENIENT = False


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
    ref, hyp = (lenient(ref), lenient(hyp)) if LENIENT else (norm(ref), norm(hyp))
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


def latency(rows: dict[str, dict], man: dict) -> dict:
    """Time per stage (bench.py's lat dict): p50 / p95, and the slope in ms per second of speech
    (least squares on clip length), so a budget can be read off for any sentence length."""
    import numpy as np
    stages = {}
    for cid, r in rows.items():
        if cid in man and not r.get("error"):
            for k, v in r.get("lat", {}).items():
                stages.setdefault(k, []).append((man[cid]["video"]["seconds"], float(v)))
    out = {}
    if not stages:
        return out
    print("\n| stage | p50 | p95 | per second of speech | at 3 s |\n|---|---|---|---|---|")
    for k, xs in stages.items():
        sec = np.array([x for x, _ in xs])
        val = np.array([v for _, v in xs])
        slope, icpt = np.polyfit(sec, val, 1) if len(xs) > 2 else (0.0, float(np.median(val)))
        unit = "KB" if k.endswith("_kb") else "ms"
        out[k] = {"p50": float(np.percentile(val, 50)), "p95": float(np.percentile(val, 95)),
                  "per_s": float(slope), "at_3s": float(icpt + 3 * slope)}
        print(f"| {k} | {out[k]['p50']:.0f} {unit} | {out[k]['p95']:.0f} {unit} | "
              f"{slope:+.0f} {unit}/s | {out[k]['at_3s']:.0f} {unit} |")
    return out


COVARIATES = ("crop_scale", "iod_px", "face_p95", "face_luma", "side_light", "motion", "yaw_abs",
              "crop_contrast", "articulation", "mouth_luma", "seconds")


def covariates(clips: list[dict], man: dict) -> dict:
    """Clip WER vs each measured attribute: Spearman rho and corpus WER per tertile, pooled, then
    within each source (pooled numbers mostly say "MEAD/VidTIMIT are harder": big faces, long
    TIMIT sentences), so read the within-source columns for cause."""
    from scipy.stats import spearmanr
    out = {}
    srcs = sorted({man[c["id"]]["source_name"] for c in clips})
    big = [s_ for s_ in srcs if sum(man[c["id"]]["source_name"] == s_ for c in clips) >= 15]
    print("\n| measure | rho (all) | WER low / mid / high third (cuts) | " + " | ".join(f"rho in {s_}" for s_ in big)
          + " |\n|---|---|---|" + "---|" * len(big))
    for k in COVARIATES:
        xs, cs = [], []
        for c in clips:
            m = man[c["id"]]
            v = (abs(m["measured"].get("yaw") or 0) if k == "yaw_abs" else
                 m["video"]["seconds"] if k == "seconds" else m["measured"].get(k))
            if v is not None:
                xs.append(float(v))
                cs.append(c)
        if len(xs) < 10:
            continue
        rho, pv = spearmanr(xs, [c["errs"] / c["n"] for c in cs])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        thirds = [order[: len(order) // 3], order[len(order) // 3: 2 * len(order) // 3], order[2 * len(order) // 3:]]
        w = [sum(cs[i]["errs"] for i in t) / max(1, sum(cs[i]["n"] for i in t)) for t in thirds]
        cut = (xs[order[len(order) // 3]], xs[order[2 * len(order) // 3]])
        within = {}
        for s_ in big:
            idx = [i for i, c in enumerate(cs) if man[c["id"]]["source_name"] == s_]
            if len(idx) >= 15 and len({xs[i] for i in idx}) > 2:
                r_, p_ = spearmanr([xs[i] for i in idx], [cs[i]["errs"] / cs[i]["n"] for i in idx])
                within[s_] = (r_, p_)
        out[k] = {"rho": rho, "p": pv, "tertile_wer": w, "cuts": cut, "within": within}
        cells = " | ".join(f"{within[s_][0]:+.2f}{'*' if within[s_][1] < 0.05 else ''}" if s_ in within else "—"
                           for s_ in big)
        print(f"| {k} | {rho:+.2f}{'*' if pv < 0.05 else ''} | {w[0]:.0%} / {w[1]:.0%} / {w[2]:.0%} "
              f"({cut[0]:.3g} / {cut[1]:.3g}) | {cells} |")
    print("(* p < 0.05)")
    return out


def adjusted(clips: list[dict], b: int, seed: int) -> dict:
    """Observed / expected errors per group, expected = the sentence's pooled error rate x words, over
    sentences read by >= 3 speakers. Separates "this group is harder" from "got harder sentences"."""
    sent = defaultdict(list)
    for c in clips:
        sent[norm(c["ref"])].append(c)
    rate = {k: sum(c["errs"] for c in v) / sum(c["n"] for c in v)
            for k, v in sent.items() if len({c["speaker"] for c in v}) >= 3}
    use = [dict(c, exp=rate[norm(c["ref"])] * c["n"]) for c in clips if norm(c["ref"]) in rate]
    out = {}
    print(f"\nSentence-adjusted observed/expected errors ({len(use)} clips on {len(rate)} shared sentences; "
          "1.00 = as expected for those sentences)")
    for key in ("skin_tone", "race", "sex", "age_band", "source"):
        vals = defaultdict(list)
        for c in use:
            vals[c["groups"].get(key, "unknown")].append(c)
        if len(vals) < 2:
            continue
        cells = []
        for v, cs in sorted(vals.items(), key=lambda kv: -len(kv[1])):
            o, e = sum(c["errs"] for c in cs), sum(c["exp"] for c in cs)
            by = defaultdict(lambda: [0.0, 0.0])
            for c in cs:
                by[c["speaker"]][0] += c["errs"]
                by[c["speaker"]][1] += c["exp"]
            spk = list(by.values())
            rng = random.Random(seed)
            st = sorted(sum(x[0] for x in s_) / max(1e-9, sum(x[1] for x in s_))
                        for s_ in ([spk[rng.randrange(len(spk))] for _ in spk] for _ in range(b)))
            ratio = o / e if e else float("nan")
            out.setdefault(key, {})[v] = {"o": o, "e": e, "ratio": ratio, "ci": (st[int(.025 * b)], st[int(.975 * b) - 1])}
            cells.append(f"{v} {ratio:.2f} [{st[int(.025 * b)]:.2f}, {st[int(.975 * b) - 1]:.2f}] (n={len(cs)})")
        print(f"- {key}: " + " · ".join(cells))
    return out


def paired_views(clips: list[dict], man: dict) -> dict:
    """MEAD: one take filmed from several cameras. WER per view, and the change vs the front camera
    on exactly the same takes."""
    take = defaultdict(dict)
    for c in clips:
        m = man[c["id"]]
        if m.get("view"):
            take[(m["speaker"], m["sentence"])][m["view"]] = c
    if not take:
        return {}
    views = sorted({v for t in take.values() for v in t})
    out = {}
    print("\n| camera | takes | WER | front WER on the same takes |\n|---|---|---|---|")
    for v in views:
        both = [t for t in take.values() if v in t and "front" in t]
        e = sum(t[v]["errs"] for t in both)
        n = sum(t[v]["n"] for t in both)
        ef = sum(t["front"]["errs"] for t in both)
        out[v] = {"takes": len(both), "wer": e / n if n else None, "front_wer": ef / n if n else None}
        if n:
            print(f"| {v} | {len(both)} | {e / n:.1%} | {ef / n:.1%} |")
    cls = {"30° side": ("left_30", "right_30"), "60° side": ("left_60", "right_60"),
           "camera above": ("top",), "camera below": ("down",)}
    print("\n| angle vs front, same takes | takes | WER change, points [95% CI over takes] |\n|---|---|---|")
    for name, vs in cls.items():
        pairs = [(t[v], t["front"]) for t in take.values() if "front" in t for v in vs if v in t]
        if not pairs:
            continue
        d = [(a_["errs"] - f["errs"], f["n"]) for a_, f in pairs]
        mean = sum(x for x, _ in d) / sum(n_ for _, n_ in d)
        rng = random.Random(0)
        st = sorted(sum(x for x, _ in s_) / sum(n_ for _, n_ in s_)
                    for s_ in ([d[rng.randrange(len(d))] for _ in d] for _ in range(2000)))
        out[name] = {"takes": len(pairs), "delta": mean, "ci": (st[50], st[1949])}
        print(f"| {name} | {len(pairs)} | {mean * 100:+.1f} [{st[50] * 100:+.1f}, {st[1949] * 100:+.1f}] |")
    return out


def paired(ca: list[dict], cb: list[dict], la: str, lb: str, b: int) -> dict:
    """WER(B) - WER(A) on the clips both runs read, overall and per source; 95% CI resampling
    people (both runs move together, so this is much tighter than comparing two separate CIs)."""
    xa = {c["id"]: c for c in ca}
    both = [(xa[c["id"]], c) for c in cb if c["id"] in xa]
    print(f"\n## {lb} vs {la}: {len(both)} clips\n| clips | {la} | {lb} | change [95% CI] |\n|---|---|---|---|")
    out = {}
    groups = {"all": both}
    for src in sorted({x["groups"]["source"] for x, _ in both}):
        groups[src] = [(x, y) for x, y in both if x["groups"]["source"] == src]
    for name, pairs in groups.items():
        by = defaultdict(lambda: [0, 0, 0])
        for x, y in pairs:
            by[x["speaker"]][0] += x["errs"]
            by[x["speaker"]][1] += y["errs"]
            by[x["speaker"]][2] += x["n"]
        spk = list(by.values())
        n = sum(v[2] for v in spk)
        wa, wb = sum(v[0] for v in spk) / n, sum(v[1] for v in spk) / n
        rng = random.Random(0)
        st = []
        for _ in range(b):
            smp = [spk[rng.randrange(len(spk))] for _ in spk]
            m = sum(v[2] for v in smp)
            st.append((sum(v[1] for v in smp) - sum(v[0] for v in smp)) / m)
        st.sort()
        out[name] = {"a": wa, "b": wb, "delta": wb - wa, "ci": (st[int(.025 * b)], st[int(.975 * b) - 1])}
        print(f"| {name} | {wa:.1%} | {wb:.1%} | {(wb - wa) * 100:+.1f} pts [{st[int(.025 * b)] * 100:+.1f}, "
              f"{st[int(.975 * b) - 1] * 100:+.1f}] |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ML / "data/raw_eval_v2/manifest.json")
    ap.add_argument("--run", action="append", required=True, help='"label=glob of bench JSONs"')
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worst", type=int, default=12)
    ap.add_argument("--lenient", action="store_true",
                    help="ignore spelling-only differences (11/eleven, to morrow/tomorrow, it's/it is); "
                         "default is bench.py's strict scoring, comparable with earlier numbers")
    ap.add_argument("--paired", nargs=2, action="append", metavar=("A", "B"),
                    help="WER change B - A on the same clips, CI resampling people (repeatable)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    global LENIENT
    LENIENT = a.lenient

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
            g["transcript_check"] = m.get("transcript_check", {}).get("status", "unknown")
            clips.append({"id": cid, "speaker": m["speaker"], "errs": errs, "n": n, **ops,
                          "ref": m["transcript"], "hyp": r["hyp"], "error": r.get("error"), "groups": g})
        print(f"\n## {label}: {len(clips)} clips, {len({c['speaker'] for c in clips})} speakers, "
              f"{sum(c['n'] for c in clips)} words" + (f" (missing {len(missing)}: {missing[:5]}…)" if missing else ""))
        overall = boot(clips, a.boot, a.seed)
        n_sub, n_del, n_ins = (sum(c[k] for c in clips) for k in "SDI")
        print(f"WER **{fmt(overall)}** (95% CI, speakers resampled) · substitutions {n_sub}, deletions {n_del}, "
              f"insertions {n_ins} · empty reads {sum(1 for c in clips if not norm(c['hyp']))} · errors "
              f"{sum(1 for c in clips if c['error'])}")
        out = {"n_clips": len(clips), "wer": overall, "S": n_sub, "D": n_del, "I": n_ins, "groups": {},
               "missing": missing}
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
        print("\nWorst speakers: " + ", ".join(f"{s} {w:.0%}" for w, s, _ in sw[: a.worst]))
        print("Best speakers: " + ", ".join(f"{s} {w:.0%}" for w, s, _ in sw[-6:]))
        out["speakers"] = {s: w for w, s, _ in sw}
        worst = sorted(clips, key=lambda c: (-c["errs"] / c["n"], -c["n"]))[: a.worst]
        print("\n| clip | WER | reference | read |\n|---|---|---|---|")
        for c in worst:
            print(f"| {c['id']} | {c['errs'] / c['n']:.0%} | {c['ref']} | {c['hyp'] or '(empty)'} |")
        sent = defaultdict(list)
        for c in clips:
            sent[c["ref"]].append(c)
        shared = sorted(((sum(c["errs"] for c in cs) / sum(c["n"] for c in cs), r, len(cs))
                         for r, cs in sent.items() if len(cs) >= 3), reverse=True)
        if shared:
            print("\n| sentence (read by 3+ clips) | clips | WER |\n|---|---|---|")
            for w, r, k in shared:
                print(f"| {r} | {k} | {w:.0%} |")
        out["sentences"] = {r: {"clips": k, "wer": w} for w, r, k in shared}
        out["latency"] = latency(rows, man)
        out["covariates"] = covariates(clips, man)
        out["adjusted"] = adjusted(clips, a.boot, a.seed)
        out["views"] = paired_views(clips, man)
        out["clips"] = clips
        report["runs"][label] = out
    for la, lb in a.paired or []:
        report.setdefault("paired", {})[f"{lb} - {la}"] = paired(report["runs"][la]["clips"],
                                                                  report["runs"][lb]["clips"], la, lb, a.boot)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
