"""Check recorded B2 clips against their script lines with the lip reader; find mislabelled ones.

`.context/b2-scripts/rename_clips.py` labels clips by recording order, so one skipped, repeated or
out-of-order take shifts every label after it. This scores every script line of a clip's speaker
against the clip with the model's CTC likelihood, log P(line | video) per frame (`lipread.phrases`),
and solves the clip → line assignment with one line per clip (Hungarian). The likelihood separates
even the near-identical demo wordings ("Hi, nice to meet you" / "Hello, nice to meet you") far better
than comparing greedy text.

    uv run python scripts/check_labels.py <clips dir> ../.context/b2-scripts/p{1,2,3}.tsv \
        [--speakers p3] [--workers 8] [--out labels_check.json]

Prints every clip whose assigned line differs from its label, plus clips the model can't tell apart
(best vs runner-up margin below --unsure), and writes the full table and a rename plan
({old stem: new stem}) to --out. Exit 0 = every label agrees, 1 = suspected mislabels, 2 = setup problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_CROPPER = None


def _init(cov: float) -> None:
    global _CROPPER
    from lipread.preprocess import MouthCropper
    _CROPPER = MouthCropper(min_face_coverage=cov)


def _crop(path: Path) -> tuple[str, np.ndarray | None, str | None]:
    from lipread.video import load_video_25fps
    try:
        return path.stem, _CROPPER.crop(load_video_25fps(path)), None
    except Exception as e:  # noqa: BLE001  report, don't stop: one bad clip shouldn't hide the rest
        return path.stem, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", type=Path)
    ap.add_argument("scripts", type=Path, nargs="+", help="speaker scripts (<id>\\t<text> per line)")
    ap.add_argument("--speakers", help="comma-separated speakers to check (default: all with a script)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-face-coverage", type=float, default=0.5)
    ap.add_argument("--unsure", type=float, default=0.05, help="best − runner-up margin per frame")
    ap.add_argument("--out", type=Path, default=Path("labels_check.json"))
    a = ap.parse_args()

    from scipy.optimize import linear_sum_assignment

    from lipread.model import LipReader
    from lipread.phrases import ctc_log_likelihood
    from lipread.preprocess import to_model_input

    lines: dict[str, list[tuple[str, str]]] = {}
    for s in a.scripts:
        for ln in s.read_text(encoding="utf8").splitlines():
            if ln.strip():
                cid, text = ln.split("\t", 1)
                lines.setdefault(cid.rsplit("_", 1)[0], []).append((cid, text.strip()))
    want = set(a.speakers.split(",")) if a.speakers else set(lines)
    vids = sorted(p for p in a.clips.iterdir() if p.suffix.lower() in VIDEO and p.stem.rsplit("_", 1)[0] in want)
    if not vids:
        print(f"no clips of {sorted(want)} in {a.clips}", file=sys.stderr)
        sys.exit(2)

    with Pool(a.workers, initializer=_init, initargs=(a.min_face_coverage,)) as pool:
        crops = pool.map(_crop, vids, chunksize=2)
    reader = LipReader()
    rows, plan = [], {}
    for spk in sorted({v.stem.rsplit("_", 1)[0] for v in vids}):
        cand = lines[spk]
        mine = [(stem, c, err) for stem, c, err in crops if stem.rsplit("_", 1)[0] == spk]
        scored = []
        for stem, c, err in mine:
            if err:
                rows.append({"clip": stem, "error": err})
                continue
            lp = reader.ctc_log_probs(to_model_input(c)).cpu()
            scored.append((stem, np.array(ctc_log_likelihood(lp, [t for _, t in cand])) / len(lp)))
        if not scored:
            continue
        score = np.stack([s for _, s in scored])          # clips × lines, log-likelihood per frame
        score = np.where(np.isfinite(score), score, -1e3)  # a line longer than the clip can't fit it
        ci, li = linear_sum_assignment(-score)              # one line per clip, best total likelihood
        ids = [cid for cid, _ in cand]
        for k, j in zip(ci, li):
            stem, s = scored[k][0], score[k]
            label = ids.index(stem) if stem in ids else None
            top2 = np.sort(s)[-2:]
            row = {"clip": stem, "assigned": ids[j], "assigned_text": cand[j][1],
                   "label_text": cand[label][1] if label is not None else None,
                   "best": ids[int(s.argmax())], "margin_vs_label": None if label is None else
                   round(float(s[j] - s[label]), 4), "runner_up_margin": round(float(top2[1] - top2[0]), 4)}
            rows.append(row)
            if ids[j] != stem:
                plan[stem] = ids[j]

    mism = [r for r in rows if "assigned" in r and r["assigned"] != r["clip"]]
    unsure = [r for r in rows if "assigned" in r and r["assigned"] == r["clip"] and r["runner_up_margin"] < a.unsure]
    errs = [r for r in rows if "error" in r]
    for title, rs in (("label disagrees with the model", mism), (f"agrees, but runner-up within {a.unsure}", unsure)):
        if rs:
            print(f"\n{title} ({len(rs)}):")
            for r in rs:
                print(f"  {r['clip']} → {r['assigned']}  {r['assigned_text']!r}  (label: {r['label_text']!r}; "
                      f"margin vs label {r['margin_vs_label']}, runner-up {r['runner_up_margin']})")
    for r in errs:
        print(f"  {r['clip']}: {r['error']}")
    n = len([r for r in rows if "assigned" in r])
    print(f"\n{n - len(mism)}/{n} labels agree with the model; {len(mism)} suspected mislabels; "
          f"{len(unsure)} unsure; {len(errs)} unreadable → {a.out}")
    a.out.write_text(json.dumps({"rename_plan": plan, "rows": rows}, indent=1))
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
