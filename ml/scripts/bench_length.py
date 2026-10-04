"""How long can one lip-reading chunk be? Word errors and delay vs clip length for the int8 speed model.

Joins whole consecutive LRS3 test clips (pre-made crops, never cut mid-clip) into ~2/4/6/8/10/15/20 s
clips, then runs greedy CTC on each with native onnxruntime (CPU). Prints a markdown table: per target
length the real mean length, WER, and delay p50/p95. Stitched clips jump between speakers, so this
measures how the model copes with length, not natural pauses. Browser delay is measured in a real
cross-origin-isolated page (Node's onnxruntime-web runs single-threaded and ~3x slower, so it misleads);
see .context/streaming-length-table.md. Beam numbers need the serving pod.

    uv run --directory ml python scripts/bench_length.py [--clips 300] [--per-length 12] [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import regress_quantized as rq  # noqa: E402
from lipread.model import collapse_ctc, ids_to_text  # noqa: E402
from lipread.preprocess import precropped_patches, to_model_input  # noqa: E402

FPS = 25
MAX_FRAMES = 500  # the int8 model's trimmed position table (brief D39)
LENGTHS_S = [2, 4, 6, 8, 10, 15, 20]


def stitch(clips: list[bench.Clip], seconds: float, limit: int) -> list[tuple[np.ndarray, str]]:
    """Whole consecutive clips joined until each piece is within ±15% of `seconds` (frames ≤ MAX_FRAMES)."""
    lo, hi = seconds * FPS * 0.85, min(seconds * FPS * 1.15, MAX_FRAMES)
    out, crops, words = [], [], []
    for c in clips:
        n = sum(len(x) for x in crops)
        if n + len(c.crops) > hi:  # adding this clip overshoots: close or restart the piece
            if n >= lo:
                out.append((np.concatenate(crops), " ".join(words)))
                if len(out) >= limit:
                    return out
            crops, words = [], []
            if len(c.crops) > hi:
                continue
        crops.append(c.crops)
        words.append(c.ref)
        if sum(len(x) for x in crops) >= lo and seconds <= 2.5:  # short targets: close as soon as in range
            out.append((np.concatenate(crops), " ".join(words)))
            crops, words = [], []
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=rq.ART / "lipread_ctc.dyn-pw8-rn16.onnx")
    ap.add_argument("--clips", type=int, default=300, help="LRS3 test clips to draw from")
    ap.add_argument("--per-length", type=int, default=12, help="stitched clips per target length")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads")
    ap.add_argument("--json", type=Path, help="also write the raw results here")
    a = ap.parse_args()

    tokens = json.loads((rq.ART / "tokens.json").read_text())
    clips = rq.load_lrs3(a.clips)
    sess = rq.open_session(a.model, a.threads)
    sess.run(None, {rq.IN_NAME: np.zeros((1, 1, 25, 88, 88), np.float32)})  # warm up

    rows = []
    for seconds in LENGTHS_S:
        pieces = stitch(clips, seconds, a.per_length)
        refs, hyps, delays, lengths = [], [], [], []
        for crops, ref in pieces:
            x = to_model_input(precropped_patches(crops)).unsqueeze(0).numpy()
            t0 = time.perf_counter()
            logp = sess.run(None, {rq.IN_NAME: x})[0]
            delays.append((time.perf_counter() - t0) * 1000)
            hyps.append(bench.norm(ids_to_text(collapse_ctc(logp.argmax(-1).tolist()), tokens)) or "<empty>")
            refs.append(bench.norm(ref))
            lengths.append(len(crops) / FPS)
        if not pieces:
            print(f"{seconds} s: no stitched clips (raise --clips)", file=sys.stderr)
            continue
        q = sorted(delays)
        rows.append({
            "target_s": seconds, "n": len(pieces), "mean_s": statistics.mean(lengths),
            "wer": jiwer.wer(refs, hyps), "words": sum(len(r.split()) for r in refs),
            "delay_p50_ms": statistics.median(q), "delay_p95_ms": q[min(len(q) - 1, int(0.95 * len(q)))],
            "ms_per_s": statistics.median(d / s for d, s in zip(delays, lengths)),
        })
        r = rows[-1]
        print(f"{seconds:>2} s  n={r['n']:>2}  mean {r['mean_s']:.1f} s  WER {r['wer']:.1%}  "
              f"delay p50 {r['delay_p50_ms']:.0f} ms", file=sys.stderr, flush=True)

    print(f"\nint8 `{a.model.name}`, native onnxruntime CPU, {a.threads} threads, greedy CTC, "
          f"stitched LRS3 test clips\n")
    print("| target | clips | mean length | words | WER | delay p50 | delay p95 | ms per second of video |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['target_s']} s | {r['n']} | {r['mean_s']:.1f} s | {r['words']} | {r['wer']:.1%} | "
              f"{r['delay_p50_ms']:.0f} ms | {r['delay_p95_ms']:.0f} ms | {r['ms_per_s']:.0f} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
