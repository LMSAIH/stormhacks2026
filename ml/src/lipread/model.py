"""Auto-AVSR VSR model: checkpoint loading, greedy CTC and joint CTC/attention beam decoding."""

from __future__ import annotations

import difflib
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from lipread.vendor.avsr_model import AVSR

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_DIR = ML_ROOT / "checkpoints"
BLANK = 0
# Beam scores are summed log-likelihoods (CTC + attention + LM); dividing by this flattens their
# softmax into usable per-word confidences. Calibrated on raw_eval + LRS3 (scripts/calibrate_conf.py).
BEAM_CONF_TEMPERATURE = 2.0


@dataclass(frozen=True)
class BeamSettings:
    """Joint CTC/attention beam search + RNN-LM. Score per token: (1 − ctc) · attention + ctc · CTC
    + lm · LM + penalty (a bonus per emitted token; > 0 favours longer readings)."""

    beam_size: int = 40
    ctc_weight: float = 0.1
    lm_weight: float = 0.3
    penalty: float = 0.0


# Quality mode's server decode (the service reads LIPREAD_BEAM_SIZE / _CTC_WEIGHT / _LM_WEIGHT /
# _PENALTY over these). The auto_avsr recipe; scripts/sweep_beam.py compares others.
DEFAULT_BEAM = BeamSettings()


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


def ctc_words(best: list[int], probs: list[float], token_list: list[str]) -> list[tuple[str, float]]:
    """Greedy CTC per frame (argmax id, its prob) → [(word, confidence)] for the decoded text.

    A token's confidence is the mean prob over the frames it was emitted on; a word's is the lowest
    of its tokens' (one shaky piece makes the word shaky). Words join exactly like `ids_to_text`."""
    eos = len(token_list) - 1
    pieces: list[tuple[str, list[float]]] = []
    prev = BLANK
    for t, p in zip(best, probs):
        if t != BLANK and t != eos:
            if t != prev:
                pieces.append((token_list[t], [p]))
            else:
                pieces[-1][1].append(p)
        prev = t
    words: list[tuple[str, float]] = []
    for tok, ps in pieces:
        conf = sum(ps) / len(ps)
        if tok.startswith("▁") or not words:
            words.append((tok.replace("▁", ""), conf))
        else:
            w, c = words[-1]
            words[-1] = (w + tok, min(c, conf))
    return [(w, round(c, 3)) for w, c in words if w]


def nbest_words(readings: list[tuple[str, float]], temperature: float) -> list[tuple[str, float]]:
    """Beam n-best [(text, score)], best first → [(word, confidence)] for the best text.

    Each reading gets a posterior softmax(score / temperature); a word's confidence is the posterior
    mass of readings that keep that word in place (word-level alignment to the best text)."""
    if not readings:
        return []
    top = max(sc for _, sc in readings)
    weights = [math.exp((sc - top) / temperature) for _, sc in readings]
    total = sum(weights)
    best = readings[0][0].split()
    agree = [0.0] * len(best)
    for (text, _), w in zip(readings, weights):
        sm = difflib.SequenceMatcher(a=[x.lower() for x in best], b=[x.lower() for x in text.split()],
                                     autojunk=False)
        for blk in sm.get_matching_blocks():
            for i in range(blk.a, blk.a + blk.size):
                agree[i] += w
    return [(word, round(a / total, 3)) for word, a in zip(best, agree)]


def _load_lm(lm_dir: Path, n_tokens: int, device: str) -> torch.nn.Module:
    """The subword RNN-LM, loaded the way the vendored `get_beam_search_decoder` does."""
    from espnet.asr.asr_utils import get_model_conf, torch_load
    from espnet.nets.lm_interface import dynamic_import_lm

    args = get_model_conf(str(lm_dir / "model.pth"), str(lm_dir / "model.json"))
    lm = dynamic_import_lm(getattr(args, "model_module", "default"), args.backend)(n_tokens, args)
    torch_load(str(lm_dir / "model.pth"), lm)
    return lm.to(device).eval()


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
    # Per word of `text`: (word, confidence 0-1). Greedy: CTC frame probs; beam: n-best agreement.
    words: list[tuple[str, float]] = field(default_factory=list)


class LipReader:
    """Wraps the vendored Chaplin `AVSR` (espnet E2E + beam search)."""

    def __init__(
        self,
        ckpt_dir: str | Path | None = None,
        model_name: str | None = None,
        device: str | None = None,
        use_lm: bool = True,
        beam_size: int = DEFAULT_BEAM.beam_size,
        ctc_weight: float = DEFAULT_BEAM.ctc_weight,
        lm_weight: float = DEFAULT_BEAM.lm_weight,
        penalty: float = DEFAULT_BEAM.penalty,
    ):
        ckpt_dir = Path(ckpt_dir or os.environ.get("LIPREAD_CKPT_DIR", DEFAULT_CKPT_DIR))
        # LIPREAD_MODEL picks a fine-tuned model dir (e.g. FT_v1) for the CLI, bench and service alike.
        model_name = model_name or os.environ.get("LIPREAD_MODEL", "LRS3_V_WER19.1")
        model_path = ckpt_dir / model_name / "model.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_path} missing — run ml/scripts/download_checkpoints.sh")
        lm_dir = ckpt_dir / "lm_en_subword"
        have_lm = use_lm and (lm_dir / "model.pth").is_file()
        self.device = device or default_device()
        # The LM is loaded here rather than by AVSR, so configure_beam() can rebuild the search
        # around the same modules (AVSR drops a zero-weight scorer when it builds its own).
        self.avsr = AVSR(
            "video",
            str(model_path),
            str(ckpt_dir / model_name / "model.json"),
            ctc_weight=ctc_weight,
            beam_size=beam_size,
            device=self.device,
        )
        self.token_list: list[str] = self.avsr.token_list
        self._lm = _load_lm(lm_dir, len(self.token_list), self.device) if have_lm else None
        self.beam_settings = DEFAULT_BEAM
        self.configure_beam(BeamSettings(beam_size, ctc_weight, lm_weight, penalty))

    def configure_beam(self, settings: BeamSettings) -> None:
        """Rebuild the beam search with new weights / size (cheap: reuses the loaded modules).
        Not thread-safe: the service only calls it at startup."""
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus

        if self._lm is None:
            settings = replace(settings, lm_weight=0.0)
        scorers = self.e2e.scorers()  # attention decoder + CTC prefix scorer
        scorers["lm"] = self._lm
        scorers["length_bonus"] = LengthBonus(len(self.token_list))
        search = BatchBeamSearch(
            beam_size=settings.beam_size,
            vocab_size=len(self.token_list),
            weights={"decoder": 1.0 - settings.ctc_weight, "ctc": settings.ctc_weight,
                     "lm": settings.lm_weight, "length_bonus": settings.penalty},
            scorers=scorers,
            sos=self.e2e.odim - 1,
            eos=self.e2e.odim - 1,
            token_list=self.token_list,
            pre_beam_score_key=None if settings.ctc_weight == 1.0 else "decoder",
        )
        self.avsr.beam_search = search.to(device=self.device).eval()
        self.beam_settings = settings

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
        words = ctc_words(best, probs.tolist(), self.token_list)
        return Transcript(ids_to_text(collapse_ctc(best), self.token_list), conf, words=words)

    def beam(self, x: torch.Tensor, n_best: int = 3, n_conf: int = 10) -> Transcript:
        """Beam search + LM. Same decode as the vendored `AVSR.infer`, which keeps only the top
        hypothesis; here the ranked ended hypotheses are kept so the UI can offer the next-best ones."""
        with torch.no_grad():
            enc = self.e2e.encode(x.to(self.device))
        return self.beam_encoded(enc, n_best, n_conf)

    def beam_encoded(self, enc: torch.Tensor, n_best: int = 3, n_conf: int = 10) -> Transcript:
        """`beam` from the encoder output (scripts/sweep_beam.py encodes each clip once)."""
        with torch.no_grad():
            # The CTC head hears no speech: return nothing rather than let the LM invent a fluent
            # sentence from still lips (it did: "I don't know what it is" on a pause).
            if not collapse_ctc(self.e2e.ctc.ctc_lo(enc).argmax(dim=-1).reshape(-1).tolist()):
                return Transcript("", None)
            hyps = self.avsr.beam_search(enc)
        readings: list[tuple[str, float]] = []  # distinct texts, best first
        for h in hyps:  # already sorted best-first
            text = ids_to_text([int(t) for t in h.yseq[1:]], self.token_list)  # drop <sos>
            if text and all(text != t for t, _ in readings):
                readings.append((text, float(h.score)))
            if len(readings) == n_conf:
                break
        words = nbest_words(readings, BEAM_CONF_TEMPERATURE)
        return Transcript(readings[0][0] if readings else "", None, readings[:n_best], words)

    def transcribe(self, x: torch.Tensor, decode: str = "greedy") -> Transcript:
        if decode == "beam":
            return self.beam(x)
        if decode == "greedy":
            return self.greedy(x)
        raise ValueError(f"decode must be 'greedy' or 'beam', got {decode!r}")
