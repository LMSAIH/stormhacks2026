"""ElevenLabs Flash v2.5 over the TTS WebSocket (stream-input)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from streaming.profiles import SegmenterProfile
from tts.base import AudioChunk, TtsBackend, TtsConfig, TtsError

KEEPALIVE_S = 10.0
DEFAULT_VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.8, "use_speaker_boost": False}


class FlashWsBackend(TtsBackend):
    name = "flash_ws"
    profile = SegmenterProfile(min_words=3, max_words=8, max_chars=50, pause_ms=700, punct_min_words=3)

    def __init__(self, config: TtsConfig):
        super().__init__(config)
        ex = config.extra
        self.model_id = config.model_id or "eleven_flash_v2_5"
        self.voice_settings = ex.get("voice_settings", DEFAULT_VOICE_SETTINGS)
        params = {"model_id": self.model_id, "output_format": config.output_format}
        if "auto_mode" in ex:
            params["auto_mode"] = "true" if ex["auto_mode"] else "false"
        if "inactivity_timeout" in ex:
            params["inactivity_timeout"] = int(ex["inactivity_timeout"])
        self.url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}/stream-input?"
            + urlencode(params)
        )
        self._ws = None
        self._q: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._keepalive: asyncio.Task | None = None
        self._closing = False
        self._last_send = 0.0

    async def _connect(self) -> None:
        self._ws = await connect(
            self.url, additional_headers={"xi-api-key": self.config.api_key}, max_size=None
        )
        await self._send({"text": " ", "voice_settings": self.voice_settings})

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))
        self._last_send = self._now()

    async def open(self) -> None:
        t0 = self._now()
        try:
            await self._connect()
        except Exception as e:  # handshake rejection (auth, bad format, ...)
            raise TtsError(f"connect failed: {e}") from e
        self.connect_time = self._now() - t0
        self._reader = asyncio.create_task(self._read_loop())
        self._keepalive = asyncio.create_task(self._keepalive_loop())

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    async for raw in self._ws:
                        self._handle(raw)
                except ConnectionClosed:
                    pass
                if self._closing:
                    return
                # unexpected drop: reconnect and re-init
                for attempt in range(3):
                    try:
                        await self._connect()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise TtsError(f"reconnect failed: {e}") from e
                        await asyncio.sleep(0.2 * (attempt + 1))
        except Exception as e:
            self._q.put_nowait(e if isinstance(e, TtsError) else TtsError(str(e)))
        finally:
            self._q.put_nowait(None)

    def _handle(self, raw) -> None:
        msg = json.loads(raw)
        if msg.get("error") or msg.get("message") and "audio" not in msg and not msg.get("isFinal"):
            self._q.put_nowait(TtsError(str(msg.get("error") or msg.get("message"))))
            return
        if msg.get("audio"):
            self._q.put_nowait(AudioChunk(
                base64.b64decode(msg["audio"]), self.sample_rate, self.codec,
                self.current_segment, self._now()))
        if msg.get("isFinal"):
            self._q.put_nowait(AudioChunk(
                b"", self.sample_rate, self.codec, self.current_segment, self._now(), is_final=True))

    async def _keepalive_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(1.0)
            if self._closing or self._now() - self._last_send < KEEPALIVE_S:
                continue
            try:
                await self._send({"text": " "})
            except Exception:
                pass  # reader handles reconnect

    async def send_text(self, text: str, *, flush: bool = False) -> int | None:
        msg = {"text": text + " "}
        if not flush:
            await self._send(msg)
            return None
        msg["flush"] = True
        seg = self._next_segment()
        self.flush_times[seg] = self._now()
        await self._send(msg)
        return seg

    async def end_turn(self) -> int | None:
        seg = self._next_segment()
        self.flush_times[seg] = self._now()
        await self._send({"text": " ", "flush": True})
        return seg

    async def end_session(self) -> None:
        self._closing = True
        if self._keepalive:
            self._keepalive.cancel()
        try:
            await self._send({"text": ""})  # only place the closing message is sent
            if self._reader:
                await asyncio.wait_for(self._reader, timeout=15)
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        finally:
            if self._ws:
                await self._ws.close()
            if self._reader and not self._reader.done():
                self._reader.cancel()

    async def audio(self) -> AsyncIterator[AudioChunk]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
