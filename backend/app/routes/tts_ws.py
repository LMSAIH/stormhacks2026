"""`/ws/tts` WebSocket endpoint."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config
from streaming.metrics import Recorder
from streaming.pipeline import TICK_S, Pipeline
from streaming.segmenter import Segmenter
from tts import create_backend

log = logging.getLogger("server")
router = APIRouter()


def _safe(msg: str, key: str | None) -> str:
    return msg.replace(key, "***") if key else msg


@router.websocket("/ws/tts")
async def ws_tts(ws: WebSocket, backend: str | None = None):
    await ws.accept()
    name = backend or config.backend_name()
    cfg = None
    tts = None
    tasks: list[asyncio.Task] = []
    pipe: Pipeline | None = None
    try:
        try:
            cfg = config.load_tts_config()
            tts = create_backend(name, cfg)
            await tts.open()
        except Exception as e:  # noqa: BLE001
            await ws.send_json({"type": "error", "message": _safe(f"backend setup failed: {e}", cfg and cfg.api_key)})
            tts_to_close, tts = tts, None
            if tts_to_close is not None:
                with contextlib.suppress(Exception):
                    await tts_to_close.end_session()
            await ws.close()
            return

        queue: asyncio.Queue = asyncio.Queue()

        async def on_audio(chunk):
            if chunk.data and not chunk.is_final and chunk.codec == "pcm":
                queue.put_nowait(chunk.data)

        pipe = Pipeline(tts, Segmenter(tts.profile), Recorder(), on_audio=on_audio)
        pipe.start()

        await ws.send_json({"type": "ready", "backend": name, "sample_rate": tts.sample_rate,
                            "codec": tts.codec, "encoding": "s16le", "channels": 1})

        async def sender():
            while True:
                await ws.send_bytes(await queue.get())

        async def ticker():
            while True:
                await asyncio.sleep(TICK_S)
                await pipe.tick()

        tasks = [asyncio.create_task(sender()), asyncio.create_task(ticker())]
        recv = asyncio.create_task(_receive(ws, pipe, cfg.api_key))
        tasks.append(recv)
        # any task finishing (disconnect / failure) ends the session
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception() if not t.cancelled() else None
            if exc and not isinstance(exc, WebSocketDisconnect):
                log.warning("session %s ended: %s", name, _safe(repr(exc), cfg.api_key))
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "error", "message": _safe(str(exc), cfg.api_key)})
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(BaseException):
                await t
        if pipe is not None and pipe._consumer is not None:
            pipe._consumer.cancel()
            with contextlib.suppress(BaseException):
                await pipe._consumer
        if tts is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(tts.end_session(), 5)
        with contextlib.suppress(Exception):
            await ws.close()


async def _receive(ws: WebSocket, pipe: Pipeline, key: str) -> None:
    while True:
        raw = await ws.receive_text()
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError("message must be a JSON object")
            kind = msg.get("type")
            if kind == "text":
                if msg.get("final", True):
                    for w in str(msg.get("text", "")).split():
                        await pipe.feed_word(w)
            elif kind == "end_turn":
                await pipe.end_turn()
            else:
                await ws.send_json({"type": "error", "message": f"unknown type {kind!r}"})
        except (json.JSONDecodeError, ValueError) as e:
            await ws.send_json({"type": "error", "message": f"invalid message: {e}"})
        except WebSocketDisconnect:
            raise
        except Exception as e:  # noqa: BLE001
            await ws.send_json({"type": "error", "message": _safe(f"processing failed: {e}", key)})
