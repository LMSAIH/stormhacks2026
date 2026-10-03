"""Backend-agnostic TTS streaming contract. Backends import ONLY this module."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from streaming.profiles import SegmenterProfile


class TtsError(Exception):
    """Raised for protocol/auth/format errors reported by a backend."""


@dataclass
class TtsConfig:
    api_key: str
    voice_id: str
    model_id: str | None = None  # None -> backend default
    output_format: str = "pcm_24000"  # e.g. pcm_24000, mp3_44100_128
    extra: dict = field(default_factory=dict)  # backend-specific options


@dataclass
class AudioChunk:
    data: bytes  # decoded audio bytes (PCM s16le when output_format is pcm_*)
    sample_rate: int
    codec: str  # "pcm" | "mp3"
    segment_id: int | None  # id of the most recent flushed send_text at receipt time
    t_recv: float  # time.perf_counter() at receipt
    is_final: bool = False  # server said generation is complete (no data)


def parse_output_format(fmt: str) -> tuple[str, int]:
    """'pcm_24000' -> ('pcm', 24000); 'mp3_44100_128' -> ('mp3', 44100)."""
    parts = fmt.split("_")
    return parts[0], int(parts[1])


def audio_seconds(chunk: AudioChunk) -> float:
    """Duration of a PCM chunk (s16le mono). Returns 0.0 for non-PCM."""
    if chunk.codec != "pcm":
        return 0.0
    return len(chunk.data) / 2 / chunk.sample_rate


class TtsBackend(ABC):
    """One persistent streaming TTS session.

    Contract:
      - await open() once; then send_text() repeatedly; end_session() at the end.
      - send_text(text, flush=True) is a segment boundary. Each flushed call gets
        the next segment_id, recorded in self.flush_times[segment_id] (perf_counter
        right before the frame is written to the socket).
      - Audio is read from audio(). Every chunk is tagged with the segment_id of
        the most recently flushed segment at the moment it arrived.
      - Keepalive and reconnect-on-drop are internal; callers never see them.
      - No barge-in support: the client stops playback itself.
    """

    name: str = "base"
    profile: SegmenterProfile = SegmenterProfile()

    def __init__(self, config: TtsConfig):
        self.config = config
        self.codec, self.sample_rate = parse_output_format(config.output_format)
        self.flush_times: dict[int, float] = {}
        self.connect_time: float | None = None  # seconds spent in open()
        self._segment_id = -1

    def _next_segment(self) -> int:
        self._segment_id += 1
        return self._segment_id

    @property
    def current_segment(self) -> int | None:
        return self._segment_id if self._segment_id >= 0 else None

    def _now(self) -> float:
        return time.perf_counter()

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def send_text(self, text: str, *, flush: bool = False) -> int | None:
        """Send text. Returns the new segment_id if flush=True, else None."""

    @abstractmethod
    async def end_turn(self) -> int | None:
        """Flush whatever is buffered (turn boundary). Returns segment_id or None."""

    @abstractmethod
    async def end_session(self) -> None:
        """Close the session; the only call that sends the protocol's closing message."""

    @abstractmethod
    def audio(self) -> AsyncIterator[AudioChunk]:
        """Async iterator of audio chunks until the session ends."""
