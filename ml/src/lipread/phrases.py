"""Model-scored phrase snapping: how well does each saved phrase explain the lip frames?

The app's phrase snapping (`frontend/src/lib/phrases/snap.ts`) compares the *text* of a reading with
saved phrases (letters that look alike on the lips). This scores phrases against the model's own
per-frame CTC log-probs instead: log P(phrase | video), summed over all CTC alignments, compared
with the likelihood of what greedy decoding read. A phrase snaps only when the model finds it nearly
as likely as its own reading.

    margin(p) = (log P(p | x) − log P(greedy text | x)) / T        (per frame, ≤ ~0)

`frontend/src/lib/phrases/ctcScore.ts` is the browser port; keep the two in step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

ML_ROOT = Path(__file__).resolve().parents[2]
SPM_DIR = ML_ROOT / "third_party" / "auto_avsr" / "spm" / "unigram"
# Below this margin a phrase doesn't snap. scripts/bench_phrase_snap.py, LRS3 test idx 100-399 (not
# the LRS3-100 gate), 50 saved phrases + 3 one-word-swap decoys each: −0.2 fixed 29/50 in-list
# readings with 2 wrong snaps in 300 clips; the look-alike rule at the app's 0.75 fixed 17 with 4.
DEFAULT_MARGIN = -0.2


@lru_cache(maxsize=1)
def _tokenizer():
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=str(SPM_DIR / "unigram5000.model"))
    units = (SPM_DIR / "unigram5000_units.txt").read_text(encoding="utf8").splitlines()
    return sp, {u.split()[0]: int(u.split()[-1]) for u in units}


def normalize(text: str) -> str:
    """Same as normalizeText in frontend/src/lib/phrases/lookalike.ts."""
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9' ]+", " ", text.upper())).strip()


def token_ids(text: str) -> list[int]:
    """Model token ids for `text` (stock unigram5000 SentencePiece; blank = 0, so ids start at 1)."""
    sp, piece_id = _tokenizer()
    return [piece_id.get(p, piece_id["<unk>"]) for p in sp.EncodeAsPieces(normalize(text))]


def ctc_log_likelihood(log_probs: torch.Tensor, texts: list[str]) -> list[float]:
    """log P(text | video) for each text, summed over CTC alignments. log_probs: (T, V)."""
    t = log_probs.shape[0]
    targets = [token_ids(s) for s in texts]
    out = [-math.inf] * len(texts)
    ok = [i for i, ids in enumerate(targets) if 0 < len(ids) <= t]
    if not ok:
        return out
    lp = log_probs.float().unsqueeze(1).expand(t, len(ok), -1)
    flat = torch.tensor([i for j in ok for i in targets[j]], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        lp, flat, torch.full((len(ok),), t, dtype=torch.long),
        torch.tensor([len(targets[j]) for j in ok], dtype=torch.long),
        blank=0, reduction="none", zero_infinity=False)
    for j, nll in zip(ok, loss.tolist()):
        out[j] = -nll
    return out


@dataclass
class PhraseScore:
    text: str
    margin: float  # (log P(phrase) − log P(reading)) / T


def rank_phrases(log_probs: torch.Tensor, reading: str, phrases: list[str]) -> list[PhraseScore]:
    """Phrases best first by how well the model thinks they explain the frames."""
    if not phrases:
        return []
    lls = ctc_log_likelihood(log_probs, [reading, *phrases]) if reading.strip() else \
        [math.nan, *ctc_log_likelihood(log_probs, phrases)]
    base = lls[0] if math.isfinite(lls[0]) else max(lls[1:])
    t = max(log_probs.shape[0], 1)
    scored = [PhraseScore(p, (ll - base) / t) for p, ll in zip(phrases, lls[1:])]
    return sorted(scored, key=lambda s: s.margin, reverse=True)


def snap(log_probs: torch.Tensor, reading: str, phrases: list[str],
         margin: float = DEFAULT_MARGIN) -> PhraseScore | None:
    """The phrase to replace `reading` with, or None. Never 'snaps' to the reading itself."""
    ranked = rank_phrases(log_probs, reading, phrases)
    if ranked and ranked[0].margin >= margin and normalize(ranked[0].text) != normalize(reading):
        return ranked[0]
    return None
