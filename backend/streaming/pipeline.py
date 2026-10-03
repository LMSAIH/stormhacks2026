"""Words -> Segmenter -> TTS backend -> Recorder glue, plus a real-time script runner."""
from __future__ import annotations

import asyncio
import time

from streaming.metrics import Recorder
from streaming.segmenter import Segment, Segmenter
from tts.base import TtsBackend

SENTENCE_END = (".", "?", "!")
TICK_S = 0.025


class Pipeline:
    def __init__(self, backend: TtsBackend, segmenter: Segmenter, recorder: Recorder,
                 on_audio=None, on_segment=None):
        # on_audio: optional async callable(AudioChunk) invoked for each audio chunk
        self.on_audio = on_audio
        self.on_segment = on_segment  # optional sync callable(segment_id, Segment)
        self.backend = backend
        self.segmenter = segmenter
        self.recorder = recorder
        self.pcm = bytearray()
        self.texts: dict[int, str] = {}
        self.last_chunk_t: float | None = None
        self.last_flush_t: float | None = None
        self._consumer: asyncio.Task | None = None

    def start(self) -> None:
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        async for chunk in self.backend.audio():
            self.recorder.note_chunk(chunk)
            if chunk.data and not chunk.is_final:
                self.last_chunk_t = chunk.t_recv
                if chunk.codec == "pcm":
                    self.pcm += chunk.data
                if self.on_audio is not None:
                    await self.on_audio(chunk)

    async def _send(self, segs: list[Segment]) -> None:
        for seg in segs:
            sid = await self.backend.send_text(seg.text, flush=True)
            if sid is None:
                continue
            self.texts[sid] = seg.text
            self.last_flush_t = self.backend.flush_times[sid]
            self.recorder.note_segment(
                sid, seg.t_words_done, self.backend.flush_times[sid],
                t_first_word=seg.first_word_t, n_words=seg.n_words, text=seg.text,
                reason=seg.reason)
            if self.on_segment is not None:
                self.on_segment(sid, seg)

    async def feed_word(self, word: str) -> None:
        self.start()
        await self._send(self.segmenter.push(word))

    async def tick(self) -> None:
        await self._send(self.segmenter.tick())

    async def end_turn(self) -> None:
        await self._send(self.segmenter.end_turn())

    async def finish(self, timeout: float = 30.0) -> None:
        await self.backend.end_session()
        if self._consumer:
            try:
                await asyncio.wait_for(self._consumer, timeout)
            except asyncio.TimeoutError:
                self._consumer.cancel()

    async def wait_idle(self, idle_s: float = 0.7, timeout: float = 8.0) -> None:
        """Wait until audio for the latest flush arrived and has been quiet for idle_s."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            now = time.perf_counter()
            if self.last_flush_t is None:
                return
            if self.last_chunk_t is not None and self.last_chunk_t > self.last_flush_t \
                    and now - self.last_chunk_t >= idle_s:
                return
            await asyncio.sleep(0.02)


async def _sleep_ticking(pipe: Pipeline, duration: float) -> None:
    end = time.perf_counter() + duration
    while True:
        await pipe.tick()
        rem = end - time.perf_counter()
        if rem <= 0:
            return
        await asyncio.sleep(min(TICK_S, rem))


async def run_script(pipe: Pipeline, words_with_times, wps: float = 2.5,
                     strict: bool = False, sentence_pause: float = 0.3):
    """Pace words in real time. Items are str (paced at wps) or (word, t_offset_s).
    Opens nothing: caller must have awaited backend.open(). Returns (recorder, pcm_bytes)."""
    pipe.start()
    t0 = time.perf_counter()
    period = 1.0 / wps
    for item in words_with_times:
        if isinstance(item, tuple):
            word, t_off = item
            await _sleep_ticking(pipe, max(0.0, t0 + t_off - time.perf_counter()))
        else:
            word = item
            await _sleep_ticking(pipe, period)
        n_before = len(pipe.recorder.segments())
        await pipe.feed_word(word)
        if word.endswith(SENTENCE_END):
            await _sleep_ticking(pipe, sentence_pause)
        if strict and len(pipe.recorder.segments()) > n_before:
            await pipe.wait_idle()
    await pipe.end_turn()
    if strict:
        await pipe.wait_idle()
    await pipe.finish()
    return pipe.recorder, bytes(pipe.pcm)
