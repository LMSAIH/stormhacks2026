"""How far does the beam (attention + LM) reading stray from what the CTC head saw, per clip?

For each clip in a sweep JSON (scripts/sweep_beam.py) and one of its settings: the CTC
log-likelihood per frame of the beam reading minus that of the greedy CTC reading (≤ ~0; very
negative = the LM/decoder overrode the lips), and the WER of both. Then the rule "keep the beam
reading unless its CTC margin is below X, else take the greedy one", per X.

    uv run python scripts/beam_vs_ctc.py artifacts/sweep/sweep-A1.json --setting 0 \
        --lrs3-parquet data/lrs3_test/0000.parquet --clips data/raw_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import clips_from_dir, clips_from_parquet, norm  # noqa: E402

from lipread.model import LipReader  # noqa: E402
from lipread.phrases import ctc_log_likelihood  # noqa: E402
from lipread.preprocess import MouthCropper, precropped_patches, to_model_input  # noqa: E402
from lipread.video import load_video_25fps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--setting", type=int, nargs="+", default=[0], help="index into the sweep's settings")
    ap.add_argument("--lrs3-parquet", type=Path)
    ap.add_argument("--clips", type=Path)
    a = ap.parse_args()

    results = json.loads(a.sweep.read_text())
    src = {}
    if a.lrs3_parquet:
        src.update({c.id: c for c in clips_from_parquet(a.lrs3_parquet, 400)})
    if a.clips:
        src.update({c.id: c for c in clips_from_dir(a.clips, 100)})
    reader = LipReader(use_lm=False)
    cropper = MouthCropper()
    lp_cache: dict[str, tuple[torch.Tensor, str]] = {}
    for k in a.setting:
        res = results[k]
        rows = []
        for r in res["rows"]:
            if r["id"] not in lp_cache:
                c = src[r["id"]]
                x = to_model_input(cropper.crop(load_video_25fps(c.path)) if c.path else precropped_patches(c.crops))
                lp_cache[r["id"]] = (reader.ctc_log_probs(x).float().cpu(), reader.greedy(x).text)
            lp, greedy = lp_cache[r["id"]]
            t = lp.shape[0]
            lls = ctc_log_likelihood(lp, [r["hyp"] or " ", greedy or " "])
            rows.append({**r, "greedy": greedy, "margin": (lls[0] - lls[1]) / t})
        print(f"\n## setting {k}: {res['summary']['settings']}")
        for name in sorted({r["set"] for r in rows}):
            sub = [r for r in rows if r["set"] == name]
            refs = [norm(r["ref"]) for r in sub]
            line = []
            for x in (None, -0.5, -0.3, -0.2, -0.15, -0.1, -0.05):
                hyps = [norm(r["greedy"] if x is not None and r["margin"] < x else r["hyp"]) or "<empty>" for r in sub]
                n = sum(x is not None and r["margin"] < x for r in sub)
                line.append(f"{'beam' if x is None else x}: {jiwer.wer(refs, hyps):.1%}" + (f" ({n})" if x is not None else ""))
            gw = jiwer.wer(refs, [norm(r["greedy"]) or "<empty>" for r in sub])
            print(f"{name}: greedy {gw:.1%} | " + " | ".join(line))
        worst = sorted(rows, key=lambda r: r["margin"])[:8]
        for r in worst:
            print(f"  {r['margin']:+.3f} {r['id']}: beam {r['hyp']!r} / greedy {r['greedy']!r} / ref {r['ref']!r}")


if __name__ == "__main__":
    main()
