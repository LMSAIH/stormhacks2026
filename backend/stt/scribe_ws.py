"""ElevenLabs Scribe v2 Realtime speech-to-text over WebSocket (server side, API key auth)."""

from __future__ import annotations

import base64
import json
import os
from typing import AsyncIterator
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

SAMPLE_RATE = 16000
URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class ScribeError(Exception):
    pass


class ScribeRealtime:
    """One streaming transcription session.

    Audio goes in as 16 kHz mono s16le PCM via send_audio(); events come out of events() as the
    raw JSON dicts ElevenLabs sends (partial_transcript, committed_transcript, ...).
    """

    def __init__(self, api_key: str | None = None, *, include_timestamps: bool = False,
                 language_code: str | None = None, silence_threshold_s: float = 0.8):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ScribeError("ELEVENLABS_API_KEY is not set (backend/.env)")
        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": f"pcm_{SAMPLE_RATE}",
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": silence_threshold_s,
            "include_timestamps": "true" if include_timestamps else "false",
            # Without audio the server hangs up after 15 s of no client messages.
            "keepalive_interval_ms": 3000,
        }
        if language_code:
            params["language_code"] = language_code
        self.url = f"{URL}?{urlencode(params)}"
        self._ws = None

    async def open(self) -> None:
        try:
            self._ws = await connect(self.url, additional_headers={"xi-api-key": self.api_key}, max_size=None)
        except Exception as e:
            raise ScribeError(f"connect failed: {e}") from e

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": False,
            "sample_rate": SAMPLE_RATE,
        }))

    async def events(self) -> AsyncIterator[dict]:
        try:
            async for raw in self._ws:
                yield json.loads(raw)
        except ConnectionClosed:
            return

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
