"""Voice activity detection. Each detector is a callable: float32 frame in [-1, 1] -> bool."""

from __future__ import annotations

from diarization.config import DiarizationConfig


class EnergyVad:
    """Dependency-free VAD: frame RMS versus a slowly adapting noise floor.

    Good enough for a quiet room; use DIARIZATION_VAD=silero when the room is noisy.
    """

    def __init__(self, cfg: DiarizationConfig):
        self.min_rms = cfg.energy_min_rms
        self.ratio = cfg.energy_ratio
        self.floor = cfg.energy_min_rms / self.ratio

    def __call__(self, frame) -> bool:
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
        # Floor drops immediately but rises slowly, so sustained speech never becomes "noise".
        self.floor = rms if rms < self.floor else self.floor + (rms - self.floor) * 0.002
        return rms > max(self.min_rms, self.floor * self.ratio)


class SileroVad:
    """Silero VAD (needs `pip install silero-vad`). 512-sample frames at 16 kHz."""

    def __init__(self, cfg: DiarizationConfig, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._sr = cfg.sample_rate
        self._threshold = threshold

    def __call__(self, frame) -> bool:
        tensor = self._torch.from_numpy(frame)
        return float(self._model(tensor, self._sr).item()) >= self._threshold


def make_vad(cfg: DiarizationConfig):
    if cfg.vad == "silero":
        return SileroVad(cfg)
    if cfg.vad == "energy":
        return EnergyVad(cfg)
    raise ValueError(f"unknown DIARIZATION_VAD {cfg.vad!r} (energy | silero)")
