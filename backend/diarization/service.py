"""Run a whole recording through the tracker and return a plain-dict result."""

from __future__ import annotations

import dataclasses
import time
from functools import lru_cache

from diarization.audio import decode_to_pcm16
from diarization.config import DiarizationConfig

CHUNK_S = 0.1  # feed in the size a live client would stream


@lru_cache(maxsize=4)
def _embedder(name: str):
    """Model load takes ~1 s; share it across requests."""
    from diarization.embed import make_embedder

    return make_embedder(DiarizationConfig(embedder=name))


def diarize_audio(data: bytes, *, suffix: str = "", overrides: dict | None = None) -> dict:
    """Bytes of an audio file -> {"duration", "speakers", "segments": [...], "timing": {...}}."""
    from diarization.tracker import SpeakerTracker
    from diarization.vad import make_vad

    cfg = dataclasses.replace(DiarizationConfig.from_env(), **(overrides or {}))
    t0 = time.perf_counter()
    pcm = decode_to_pcm16(data, cfg.sample_rate, suffix)
    t_decode = time.perf_counter() - t0

    tracker = SpeakerTracker(cfg, vad=make_vad(cfg), embedder=_embedder(cfg.embedder))
    step = int(CHUNK_S * cfg.sample_rate) * 2
    t1 = time.perf_counter()
    for i in range(0, len(pcm), step):
        tracker.feed(pcm[i:i + step])
    t_track = time.perf_counter() - t1

    duration = len(pcm) / 2 / cfg.sample_rate
    return {
        "duration": round(duration, 2),
        "speakers": tracker.speaker_count,
        "segments": [
            {"speaker": s.speaker, "start": round(s.start, 2), "end": round(s.end, 2), "provisional": s.provisional}
            for s in tracker.segments
        ],
        "timing": {
            "decode_s": round(t_decode, 3),
            "diarize_s": round(t_track, 3),
            "realtime_factor": round(duration / t_track, 1) if t_track else None,
        },
    }
