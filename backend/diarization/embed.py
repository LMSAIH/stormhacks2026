"""Speaker embeddings ("voiceprints"). An embedder is a callable:
float32 mono 16 kHz waveform -> L2-normalised 1-D numpy vector.
"""

from __future__ import annotations

from diarization.config import DiarizationConfig


class ResemblyzerEmbedder:
    """256-d GE2E voiceprints, CPU friendly. Needs `resemblyzer` (see requirements-diarization.txt)."""

    def __init__(self, _cfg: DiarizationConfig | None = None):
        from resemblyzer import VoiceEncoder  # heavy import, deliberately lazy

        self._encoder = VoiceEncoder(device="cpu", verbose=False)

    def __call__(self, wav):
        import numpy as np

        vec = np.asarray(self._encoder.embed_utterance(wav), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


def make_embedder(cfg: DiarizationConfig):
    if cfg.embedder == "resemblyzer":
        return ResemblyzerEmbedder(cfg)
    raise ValueError(f"unknown DIARIZATION_EMBEDDER {cfg.embedder!r} (resemblyzer)")
