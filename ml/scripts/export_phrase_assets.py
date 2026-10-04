"""Export what the browser needs for model-scored phrase snapping (frontend/src/lib/phrases/ctcScore.ts):

  frontend/public/phrases/pieceScores.json          SentencePiece unigram piece → log-prob score, so the
                                                     browser tokenizes phrases exactly like Python
  frontend/src/lib/phrases/__fixtures__/ctcScore.json parity cases: texts → pieces (vs sentencepiece) and
                                                     phrase log-likelihoods (vs torch ctc_loss)

    uv run python scripts/export_phrase_assets.py --lrs3-parquet data/lrs3_test/0000.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import clips_from_parquet  # noqa: E402

from lipread import phrases as ph  # noqa: E402

ML = Path(__file__).resolve().parents[1]
FRONT = ML.parent / "frontend" / "src" / "lib" / "phrases"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs3-parquet", type=Path, required=True)
    a = ap.parse_args()
    sp, _ = ph._tokenizer()

    scores = {sp.IdToPiece(i): round(sp.GetScore(i), 6) for i in range(sp.GetPieceSize())
              if not (sp.IsUnknown(i) or sp.IsControl(i))}
    (ML.parent / "frontend" / "public" / "phrases" / "pieceScores.json").write_text(json.dumps(scores, ensure_ascii=False, separators=(",", ":")))

    clips = clips_from_parquet(a.lrs3_parquet, 661)
    texts = [c.ref for c in clips] + ["Can you help me please", "It's 25 past nine", "I don't KNOW!", "  "]
    tokenize = [{"text": t, "pieces": sp.EncodeAsPieces(ph.normalize(t))} for t in texts]

    from lipread.model import LipReader
    from lipread.preprocess import precropped_patches, to_model_input

    reader = LipReader(use_lm=False)
    ctc = []
    for k in (100, 101, 102):
        c = clips[k]
        lp = reader.ctc_log_probs(to_model_input(precropped_patches(c.crops))).cpu().float()
        phrases = [c.ref, clips[k + 1].ref, " ".join(ph.normalize(c.ref).split()[:-1] + ["WATER"])]
        lls = ph.ctc_log_likelihood(lp, phrases)
        # CTC only reads the blank column and the target tokens' columns, so a column subset gives
        # the same likelihood as the full (T, 5049) matrix: keep the fixture small.
        cols = sorted({0} | {i for p in phrases for i in ph.token_ids(p)})
        remap = {c_: j for j, c_ in enumerate(cols)}
        ctc.append({"frames": lp.shape[0], "vocab": len(cols),
                    "log_probs": [round(v, 5) for v in lp[:, cols].flatten().tolist()],
                    "cases": [{"ids": [remap[i] for i in ph.token_ids(p)], "log_likelihood": round(ll, 4)}
                              for p, ll in zip(phrases, lls)]})
    (FRONT / "__fixtures__").mkdir(exist_ok=True)
    (FRONT / "__fixtures__" / "ctcScore.json").write_text(json.dumps({"tokenize": tokenize, "ctc": ctc}))
    print(f"{len(scores)} piece scores, {len(tokenize)} tokenize cases, {len(ctc)} ctc clips → {FRONT}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
