"""Calibrate per-word confidence: does a low score actually mark a misread word?

Runs the int8 ONNX greedy (what the browser runs) and PyTorch beam + LM (the Quality server) on the
same clips, labels every hypothesis word right/wrong by word alignment to the reference, and prints,
per threshold, how many wrong words get flagged (caught) and how many right words get flagged too
(noise). Beam confidence is n-best agreement, so it is shown per softmax temperature.

    uv run python scripts/calibrate_conf.py [--raw 20] [--lrs3 60] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from lipread.model import BLANK, LipReader, ctc_words, nbest_words  # noqa: E402
from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches, to_model_input  # noqa: E402
from lipread.video import load_video_25fps  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEMPERATURES = [1.0, 2.0, 4.0, 8.0]


def labelled(words: list[tuple[str, float]], ref: str) -> list[tuple[float, bool]]:
    """(confidence, word is right) for each hypothesis word, by alignment to the reference."""
    hyp = [bench.norm(w) for w, _ in words]
    sm = difflib.SequenceMatcher(a=hyp, b=bench.norm(ref).split(), autojunk=False)
    right = [False] * len(hyp)
    for blk in sm.get_matching_blocks():
        for i in range(blk.a, blk.a + blk.size):
            right[i] = True
    return [(c, r) for (_, c), r in zip(words, right)]


def table(name: str, pairs: list[tuple[float, bool]]) -> None:
    wrong = [c for c, r in pairs if not r]
    right = [c for c, r in pairs if r]
    print(f"\n{name}: {len(pairs)} words, {len(wrong)} wrong "
          f"(median conf right {np.median(right):.2f}, wrong {np.median(wrong) if wrong else float('nan'):.2f})")
    print("| flag below | wrong words caught | right words flagged |")
    print("|---|---|---|")
    for t in THRESHOLDS:
        caught = sum(c < t for c in wrong) / max(1, len(wrong))
        noise = sum(c < t for c in right) / max(1, len(right))
        print(f"| {t:.1f} | {caught:.0%} | {noise:.0%} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=int, default=20)
    ap.add_argument("--lrs3", type=int, default=60)
    ap.add_argument("--device")
    ap.add_argument("--onnx", type=Path, default=ART / "lipread_ctc.dyn-pw8-rn16.onnx")
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    tokens = json.loads((a.onnx.parent / "tokens.json").read_text())
    reader = LipReader(device=a.device)
    cropper = MouthCropper()
    root = Path(__file__).resolve().parents[1]
    clips = bench.clips_from_dir(root / "data/raw_eval", a.raw) + bench.clips_from_parquet(
        root / "data/lrs3_test/0000.parquet", a.lrs3)

    greedy: list[tuple[float, bool]] = []
    readings_all: list[tuple[list[tuple[str, float]], str]] = []
    for i, clip in enumerate(clips):
        try:
            patches = cropper.crop(load_video_25fps(clip.path)) if clip.path else precropped_patches(clip.crops)
        except NoFaceError:
            continue
        x = to_model_input(patches)
        logp = sess.run(None, {"video": x.unsqueeze(0).numpy()})[0]
        best = logp.argmax(-1).tolist()
        probs = np.exp(logp.max(-1)).tolist()
        greedy += labelled(ctc_words(best, probs, tokens), clip.ref)
        res = reader.beam(x, n_best=10, n_conf=10)
        readings_all.append((res.alternatives, clip.ref))
        print(f"{i + 1}/{len(clips)}", end="\r", file=sys.stderr)
    assert BLANK == 0

    table("int8 greedy (CTC frame probs)", greedy)
    for temp in TEMPERATURES:
        pairs: list[tuple[float, bool]] = []
        for readings, ref in readings_all:
            pairs += labelled(nbest_words(readings, temp), ref)
        table(f"beam + LM (n-best agreement, temperature {temp})", pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
