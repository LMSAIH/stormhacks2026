"""Async wrapper that runs a SpeakerTracker off the event loop for one live session."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

from diarization.merge import Piece, split_by_speaker

log = logging.getLogger(__name__)

# Diarization gets its own small thread pool. The loop's default executor is also used for DNS
# lookups (every ElevenLabs connect), so sharing it could let diarization delay TTS/STT connects.
_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("DIARIZATION_THREADS", "2")),
                           thread_name_prefix="diarize")


class LiveDiarizer:
    def __init__(self, tracker, on_segments=None):
        self.tracker = tracker
        self.on_segments = on_segments  # optional async callable(changed_segments, audio_clock_s)
        self._q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    def feed(self, pcm: bytes) -> None:
        """Queue audio for the tracker. Never blocks the audio path to the STT service."""
        self._q.put_nowait(pcm)

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pcm = await self._q.get()
            try:
                if pcm is None:
                    return
                changed = await loop.run_in_executor(_POOL, self.tracker.feed, pcm)
                if changed and self.on_segments is not None:
                    await self.on_segments(changed, self.tracker.clock_s)
            except Exception:  # a bad frame must not kill captions; keep the last labels
                log.exception("diarization frame failed")
                continue
            finally:
                self._q.task_done()

    async def drain(self) -> None:
        """Wait until everything fed so far has been processed."""
        await self._q.join()

    def current_speaker(self) -> Optional[str]:
        return self.tracker.current_speaker

    async def wait_for(self, t: float, timeout: float = 0.8) -> None:
        """Wait until the tracker has processed audio up to session time `t` (or give up)."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self.tracker.clock_s < t and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)

    def split(self, words: list[dict]) -> list[Piece]:
        return split_by_speaker(words, self.tracker.speaker_at)

    async def close(self) -> None:
        if self._task is not None:
            self._q.put_nowait(None)
            try:
                await asyncio.wait_for(self._task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None


@lru_cache(maxsize=2)
def _shared_embedder(name: str):
    """The voiceprint model is stateless and slow to load; load once per process, not per socket."""
    from diarization.config import DiarizationConfig
    from diarization.embed import make_embedder

    try:  # keep torch from grabbing every core and starving the event loop / TTS streaming
        import torch

        torch.set_num_threads(int(os.getenv("DIARIZATION_TORCH_THREADS", "2")))
    except ImportError:
        pass
    return make_embedder(DiarizationConfig(embedder=name))


def warm_up() -> None:
    """Load models ahead of the first connection (blocking; call from a thread at startup)."""
    from diarization.config import DiarizationConfig, is_enabled

    if is_enabled():
        _shared_embedder(DiarizationConfig.from_env().embedder)


def create_live_diarizer(on_segments=None) -> Optional[LiveDiarizer]:
    """The one entry point other code uses. Returns None when the feature is off."""
    from diarization.config import DiarizationConfig, is_enabled

    if not is_enabled():
        return None
    from diarization.tracker import SpeakerTracker

    cfg = DiarizationConfig.from_env()
    # VAD is stateful (noise floor / model state) so it is per session; the embedder is shared.
    return LiveDiarizer(SpeakerTracker(cfg, embedder=_shared_embedder(cfg.embedder)), on_segments)
