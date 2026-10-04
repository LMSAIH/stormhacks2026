"""ElevenLabs Text-to-Dialogue WebSocket backend (eleven_v4_turbo)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator

from websockets.asyncio.client import connect

from streaming.profiles import SegmenterProfile
from tts.base import AudioChunk, TtsBackend, TtsConfig, TtsError

URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
KEEPALIVE_S = 10.0


class V4DialogueWsBackend(TtsBackend):
    name = "v4_ws"
    # Server buffers ~40 chars / 8 words before emitting audio unless flushed.
    profile = SegmenterProfile(min_words=4, max_words=12, max_chars=80, pause_ms=700, punct_min_words=4)

    def __init__(self, config: TtsConfig):
        super().__init__(config)
        self.model_id = config.model_id or "eleven_v4_turbo"
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._keepalive: asyncio.Task | None = None
        self._q: asyncio.Queue = asyncio.Queue()
        self._closing = False
        self._last_send = 0.0

    # -- connection -------------------------------------------------------
    def _url(self) -> str:
        u = f"{URL}?model_id={self.model_id}&output_format={self.config.output_format}"
        if self.config.extra.get("sync_alignment"):
            u += "&sync_alignment=true"
        return u

    async def _connect(self) -> None:
        self._ws = await connect(
            self._url(), additional_headers={"xi-api-key": self.config.api_key}, max_size=None
        )
        await self._send({"voices": [self.config.voice_id]})
        self._reader = asyncio.create_task(self._read_loop(self._ws))

    async def _send(self, obj: dict) -> None:
        await self._ws.send(json.dumps(obj))
        self._last_send = self._now()

    async def open(self) -> None:
        t0 = self._now()
        await self._connect()
        self._keepalive = asyncio.create_task(self._keepalive_loop())
        self.connect_time = self._now() - t0

    async def _keepalive_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(1.0)
                if not self._closing and self._now() - self._last_send >= KEEPALIVE_S:
                    try:
                        await self._send({"keep_alive": True})
                    except Exception:
                        pass  # reader handles reconnect
        except asyncio.CancelledError:
            pass

    async def _read_loop(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("error"):
                    self._q.put_nowait(TtsError(str(msg.get("error")) + " " + str(msg.get("message", ""))))
                    return
                if msg.get("audio"):
                    self._q.put_nowait(
                        AudioChunk(base64.b64decode(msg["audio"]), self.sample_rate, self.codec,
                                   self.current_segment, self._now())
                    )
                if msg.get("is_final"):
                    self._q.put_nowait(
                        AudioChunk(b"", self.sample_rate, self.codec, self.current_segment, self._now(), True)
                    )
                    if self._closing:
                        self._q.put_nowait(None)
                        return
        except Exception:
            pass
        # socket ended
        if self._closing:
            self._q.put_nowait(None)
            return
        try:  # unexpected drop: reconnect and re-register voices
            await self._connect()
        except Exception as e:
            self._q.put_nowait(TtsError(f"reconnect failed: {e!r}"))

    # -- API ----------------------------------------------------------------
    async def _send_safe(self, obj: dict) -> None:
        try:
            await self._send(obj)
        except Exception:
            if self._closing:
                raise
            if self._reader:  # let reader reconnect, then retry once
                await asyncio.wait({self._reader}, timeout=5)
            await self._send(obj)

    async def send_text(self, text: str, *, flush: bool = False) -> int | None:
        msg: dict = {"inputs": [{"text": text + " ", "voice_id": self.config.voice_id, "new_turn": False}]}
        seg = None
        if flush:
            msg["flush"] = True
            seg = self._next_segment()
            self.flush_times[seg] = self._now()
        await self._send_safe(msg)
        return seg

    async def end_turn(self) -> int | None:
        msg: dict = {"flush": True}
        if self.config.extra.get("new_turn_on_end_turn"):
            msg["inputs"] = [{"text": " ", "voice_id": self.config.voice_id, "new_turn": True}]
        seg = self._next_segment()
        self.flush_times[seg] = self._now()
        await self._send_safe(msg)
        return seg

    async def end_session(self) -> None:
        self._closing = True
        if self._keepalive:
            self._keepalive.cancel()
        try:
            await self._send({"close_socket": True})
        except Exception:
            self._q.put_nowait(None)
            return
        if self._reader:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader), timeout=30)
            except Exception:
                self._q.put_nowait(None)
        try:
            await self._ws.close()
        except Exception:
            pass

    async def audio(self) -> AsyncIterator[AudioChunk]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
