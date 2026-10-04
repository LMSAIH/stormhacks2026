"""Listening-mode websocket: browser mic PCM in, live captions (optionally with speaker labels) out.

Client protocol (binary in, JSON out; see stt/session.py for the event shapes):
  - connect to /ws/stt with the normal session cookie
  - send 16 kHz mono signed 16-bit little-endian PCM, ~100 ms per binary frame, continuously
    (also during silence: speaker timestamps are measured on the audio you send)
"""

import asyncio
import contextlib
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth_session import get_user_from_cookie_header
from diarization import create_live_diarizer, get_tts_gate  # optional feature; see diarization/README.md
from stt.scribe_ws import ScribeError, ScribeRealtime
from stt.session import SttSession

router = APIRouter()


@router.websocket("/ws/stt")
async def stt_socket(ws: WebSocket) -> None:
	user = get_user_from_cookie_header(ws.headers.get("cookie", ""))
	await ws.accept()
	if user is None:
		await ws.close(code=4401, reason="Sign-in required")
		return

	diarizer = create_live_diarizer()  # None unless DIARIZATION=1
	gate = get_tts_gate(user["id"]) if diarizer is not None else None

	async def send(event: dict) -> None:
		with contextlib.suppress(RuntimeError, WebSocketDisconnect):
			await ws.send_json(event)

	try:
		# English only (all we support) — fixing the language improves caption accuracy.
		scribe = ScribeRealtime(
			include_timestamps=diarizer is not None,
			language_code=os.getenv("STT_LANGUAGE", "en"),
		)
		await scribe.open()
	except ScribeError as e:
		await send({"type": "error", "message": str(e)})
		await ws.close(code=1011, reason="Speech-to-text unavailable")
		return

	if diarizer is not None:
		diarizer.start()
	session = SttSession(scribe, send, diarizer=diarizer, gate=gate)

	async def pump_scribe() -> None:
		try:
			async for event in scribe.events():
				await session.on_event(event)
		finally:
			with contextlib.suppress(Exception):
				await ws.close(code=1011, reason="Speech-to-text ended")

	pump = asyncio.create_task(pump_scribe())
	try:
		while True:
			message = await ws.receive()
			if message["type"] == "websocket.disconnect":
				break
			data = message.get("bytes")
			if data:
				await session.on_audio(data)
	except (WebSocketDisconnect, RuntimeError):
		pass
	finally:
		pump.cancel()
		with contextlib.suppress(asyncio.CancelledError, Exception):
			await pump
		await scribe.close()
		if diarizer is not None:
			await diarizer.close()
