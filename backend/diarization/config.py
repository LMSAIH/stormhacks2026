"""Diarization settings. Everything is overridable through DIARIZATION_* env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

TRUTHY = {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Feature flag: DIARIZATION=1 turns speaker labelling on (default off)."""
    return os.getenv("DIARIZATION", "0").strip().lower() in TRUTHY


@dataclass(frozen=True)
class DiarizationConfig:
    sample_rate: int = 16000
    frame_samples: int = 512  # 32 ms at 16 kHz (Silero's frame size)
    vad: str = "energy"  # energy | silero
    embedder: str = "resemblyzer"
    threshold: float = 0.65  # cosine similarity needed to join an existing speaker (room mic: 0.6-0.7)
    min_speech_s: float = 0.7  # speech needed before the first voiceprint is taken
    recheck_s: float = 0.5  # re-embed this often while a person keeps talking
    window_s: float = 1.5  # length of audio used for each voiceprint
    hangover_s: float = 0.4  # silence that ends an utterance
    short_utterance_s: float = 0.25  # shorter speech bursts are treated as noise
    max_speakers: int = 10
    change_confirmations: int = 2  # consecutive differing rechecks before switching speaker
    centroid_cap: int = 50  # running-average weight cap so a voice can still adapt
    energy_min_rms: float = 0.01  # energy VAD: absolute floor (float PCM, full scale = 1.0)
    energy_ratio: float = 3.0  # energy VAD: speech must exceed noise floor by this factor
    max_segments: int = 500  # segment history kept for word alignment

    @classmethod
    def from_env(cls) -> "DiarizationConfig":
        values = {}
        for f in fields(cls):
            raw = os.getenv(f"DIARIZATION_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            values[f.name] = type(f.default)(raw)
        return cls(**values)
