"""Diarization-only WebSocket for testing: stream PCM in, get speaker segments out. No ElevenLabs.

    ws://localhost:5000/ws/diarize

Needs the normal session cookie (code 4401 otherwise), like /ws/stt. The socket is meant to stay
open for the whole app session: stream PCM continuously. Client sends: binary frames of 16 kHz mono
signed 16-bit little-endian PCM (~100 ms each, the way the browser would stream). Optional text
commands: "end"/"snapshot" (reply "done" with the timeline so far; socket stays open) and "reset"
(forget all speakers, reply "ready").
Server sends JSON:
    {"type": "ready"}
    {"type": "segment", "id", "speaker", "start", "end", "provisional", "audio_time"}
        a segment appeared, changed speaker, or grew by >= 0.5 s ("audio_time" = how much audio
        the server had processed when it emitted this; compare to what you have sent for latency)
    {"type": "done", "speakers": N, "segments": [...]}     reply to "end"

Only active when DIARIZATION=1 (closes with code 4404 otherwise). Diarization only: no ElevenLabs.
"""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth_session import get_user_from_cookie_header
from diarization.config import is_enabled

router = APIRouter()
EMIT_EVERY_S = 0.5


def _row(seg) -> dict:
    return {"id": seg.id, "speaker": seg.speaker, "start": round(seg.start, 2),
            "end": round(seg.end, 2), "provisional": seg.provisional}


@router.websocket("/ws/diarize")
async def diarize_socket(ws: WebSocket) -> None:
    user = get_user_from_cookie_header(ws.headers.get("cookie", ""))
    await ws.accept()
    if user is None:  # same sign-in rule as /ws/stt
        await ws.close(code=4401, reason="Sign-in required")
        return
    if not is_enabled():
        await ws.close(code=4404, reason="diarization is disabled (set DIARIZATION=1)")
        return

    from diarization.live import create_live_diarizer

    sent: dict[int, tuple[str, float]] = {}

    async def on_segments(changed, clock_s: float) -> None:
        for seg in changed:
            prev = sent.get(seg.id)
            if prev and prev[0] == seg.speaker and seg.end - prev[1] < EMIT_EVERY_S:
                continue
            sent[seg.id] = (seg.speaker, seg.end)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await ws.send_json({"type": "segment", **_row(seg), "audio_time": round(clock_s, 2)})

    live = create_live_diarizer(on_segments)
    try:
        live.start()
        await ws.send_json({"type": "ready"})
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                live.feed(message["bytes"])
            else:
                command = (message.get("text") or "").strip().lower()
                if command in ("end", "snapshot"):
                    # Snapshot of the session so far. The socket stays open; the client closes it.
                    await live.drain()
                    await ws.send_json({
                        "type": "done",
                        "speakers": live.tracker.speaker_count,
                        "segments": [_row(s) for s in live.tracker.segments],
                    })
                elif command == "reset":
                    await live.drain()
                    live.tracker.reset()
                    sent.clear()
                    await ws.send_json({"type": "ready"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await live.close()
