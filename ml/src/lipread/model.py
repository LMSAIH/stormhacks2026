"""Auto-AVSR VSR model: checkpoint loading, greedy CTC and joint CTC/attention beam decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lipread.vendor.avsr_model import AVSR

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_DIR = ML_ROOT / "checkpoints"
BLANK = 0


def collapse_ctc(best: list[int]) -> list[int]:
    """Greedy CTC: per-frame argmax ids → merge repeats, drop blanks. Same logic the JS tier needs."""
    ids, prev = [], BLANK
    for t in best:
        if t != prev and t != BLANK:
            ids.append(t)
        prev = t
    return ids


def ids_to_text(ids: list[int], token_list: list[str]) -> str:
    """Token ids → text; skips <blank> (0) and <eos> (last). "▁" marks a word start."""
    pieces = [token_list[i] for i in ids if 0 < i < len(token_list) - 1]
    return "".join(pieces).replace("▁", " ").strip()


def default_device() -> str:
    env = os.environ.get("LIPREAD_DEVICE")
    if env:
        return env
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class Transcript:
    text: str
    confidence: float | None  # mean max-prob over non-blank CTC frames (greedy only)
    # Beam only: up to N distinct readings, best first, as (text, beam score); [0] is `text`.
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class LipReader:
    """Wraps the vendored Chaplin `AVSR` (espnet E2E + beam search)."""

    def __init__(
        self,
        ckpt_dir: str | Path | None = None,
        model_name: str = "LRS3_V_WER19.1",
        device: str | None = None,
        use_lm: bool = True,
        beam_size: int = 40,
        ctc_weight: float = 0.1,
        lm_weight: float = 0.3,
    ):
        ckpt_dir = Path(ckpt_dir or os.environ.get("LIPREAD_CKPT_DIR", DEFAULT_CKPT_DIR))
        model_path = ckpt_dir / model_name / "model.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_path} missing — run ml/scripts/download_checkpoints.sh")
        lm_dir = ckpt_dir / "lm_en_subword"
        have_lm = use_lm and (lm_dir / "model.pth").is_file()
        self.device = device or default_device()
        self.avsr = AVSR(
            "video",
            str(model_path),
            str(ckpt_dir / model_name / "model.json"),
            rnnlm=str(lm_dir / "model.pth") if have_lm else None,
            rnnlm_conf=str(lm_dir / "model.json") if have_lm else None,
            penalty=0.0,
            ctc_weight=ctc_weight,
            lm_weight=lm_weight if have_lm else 0.0,
            beam_size=beam_size,
            device=self.device,
        )
        self.token_list: list[str] = self.avsr.token_list

    @property
    def e2e(self) -> torch.nn.Module:
        return self.avsr.model

    def ctc_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, T, 88, 88) → CTC log-probs (T, vocab). This is the graph ONNX/WebGPU runs."""
        with torch.no_grad():
            enc = self.e2e.encode(x.to(self.device))
            return torch.log_softmax(self.e2e.ctc.ctc_lo(enc), dim=-1)

    def greedy(self, x: torch.Tensor) -> Transcript:
        logp = self.ctc_log_probs(x)
        best = logp.argmax(dim=-1).tolist()
        probs = logp.exp().max(dim=-1).values
        keep = torch.tensor([b != BLANK for b in best])
        conf = float(probs[keep].mean()) if keep.any() else None
        return Transcript(ids_to_text(collapse_ctc(best), self.token_list), conf)

    def beam(self, x: torch.Tensor, n_best: int = 3) -> Transcript:
        """Beam search + LM. Same decode as the vendored `AVSR.infer`, which keeps only the top
        hypothesis; here the ranked ended hypotheses are kept so the UI can offer the next-best ones."""
        with torch.no_grad():
            hyps = self.avsr.beam_search(self.e2e.encode(x.to(self.device)))
        alternatives: list[tuple[str, float]] = []
        for h in hyps:  # already sorted best-first
            text = ids_to_text([int(t) for t in h.yseq[1:]], self.token_list)  # drop <sos>
            if text and all(text != t for t, _ in alternatives):
                alternatives.append((text, float(h.score)))
            if len(alternatives) == n_best:
                break
        return Transcript(alternatives[0][0] if alternatives else "", None, alternatives)

    def transcribe(self, x: torch.Tensor, decode: str = "greedy") -> Transcript:
        if decode == "beam":
            return self.beam(x)
        if decode == "greedy":
            return self.greedy(x)
        raise ValueError(f"decode must be 'greedy' or 'beam', got {decode!r}")
