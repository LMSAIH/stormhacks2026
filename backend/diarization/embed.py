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


class EcapaEmbedder:
    """192-d ECAPA-TDNN voiceprints (SpeechBrain, trained on VoxCeleb1+2).

    The industry-standard speaker embedder — far more discriminative and noise-robust than
    Resemblyzer's GE2E. Needs `speechbrain` (see requirements-diarization.txt); the ~80 MB model
    downloads from Hugging Face on first use into ECAPA_DIR (default /tmp). CPU inference.
    """

    def __init__(self, _cfg: DiarizationConfig | None = None):
        import os

        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier

        self._torch = torch
        self._clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.getenv("ECAPA_DIR", "/tmp/spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"},
        )

    def __call__(self, wav):
        import numpy as np

        tensor = self._torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._clf.encode_batch(tensor).reshape(-1).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


def make_embedder(cfg: DiarizationConfig):
    if cfg.embedder == "resemblyzer":
        return ResemblyzerEmbedder(cfg)
    if cfg.embedder == "ecapa":
        return EcapaEmbedder(cfg)
    raise ValueError(
        f"unknown DIARIZATION_EMBEDDER {cfg.embedder!r} (resemblyzer | ecapa)"
    )
