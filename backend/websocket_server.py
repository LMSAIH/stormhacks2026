import asyncio
import base64
import json
import os
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from auth_session import get_user_from_cookie_header
from config import HOST, TERMINATOR, WEBSOCKET_PORT
from services.chat_store import ChatStoreUnavailableError, get_user_voice_id
from state import get_default_voice_id
from tts import create_backend
from tts.config import backend_name, load_tts_config

STT_SAMPLE_RATE = 16_000
STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


async def _forward_audio(websocket, backend) -> None:
	try:
		async for chunk in backend.audio():
			if chunk.data and not chunk.is_final:
				await websocket.send(chunk.data)
	except asyncio.CancelledError:
		raise
	except Exception:
		try:
			await websocket.close(code=1011, reason="TTS stream failed")
		except Exception:
			pass


async def handle_connection(websocket) -> None:
	cookie_header = websocket.request.headers.get("Cookie", "")
	user = get_user_from_cookie_header(cookie_header)
	if user is None:
		await websocket.close(code=4401, reason="Sign-in required")
		return

	try:
		voice_id = await get_user_voice_id(user["id"])
	except ChatStoreUnavailableError:
		await websocket.close(code=1011, reason="Voice preferences unavailable")
		return

	backend = create_backend(
		backend_name(),
		load_tts_config(voice_id=voice_id or get_default_voice_id()),
	)
	audio_task = None
	try:
		await backend.open()
		audio_task = asyncio.create_task(_forward_audio(websocket, backend))
		pending_text = ""
		async for message in websocket:
			if not isinstance(message, str):
				continue

			pending_text += message
			while TERMINATOR in pending_text:
				text, pending_text = pending_text.split(TERMINATOR, 1)
				if text.strip():
					await backend.send_text(text.strip(), flush=True)
	finally:
		if audio_task is not None:
			audio_task.cancel()
			try:
				await audio_task
			except BaseException:
				pass
		await backend.end_session()


async def _forward_stt_events(client, upstream) -> None:
	async for event in upstream:
		await client.send(event)


async def _forward_stt_audio(client, upstream) -> None:
	async for message in client:
		if isinstance(message, bytes):
			if not message:
				continue
			if len(message) % 2:
				await client.send(json.dumps({
					"message_type": "error",
					"error": "PCM16 audio chunks must contain an even number of bytes",
				}))
				continue
			payload = {
				"message_type": "input_audio_chunk",
				"audio_base_64": base64.b64encode(message).decode("ascii"),
				"commit": False,
				"sample_rate": STT_SAMPLE_RATE,
			}
			await upstream.send(json.dumps(payload))
			continue

		try:
			control = json.loads(message)
		except json.JSONDecodeError:
			await client.send(json.dumps({"message_type": "error", "error": "Expected binary audio or JSON control"}))
			continue
		if not isinstance(control, dict) or control.get("type") != "commit":
			await client.send(json.dumps({"message_type": "error", "error": "Unknown control message; expected type 'commit'"}))
			continue
		await upstream.send(json.dumps({
			"message_type": "input_audio_chunk",
			"audio_base_64": "",
			"commit": True,
			"sample_rate": STT_SAMPLE_RATE,
		}))


async def handle_stt_connection(websocket) -> None:
	cookie_header = websocket.request.headers.get("Cookie", "")
	if get_user_from_cookie_header(cookie_header) is None:
		await websocket.close(code=4401, reason="Sign-in required")
		return

	api_key = os.getenv("ELEVENLABS_API_KEY")
	if not api_key:
		await websocket.close(code=1011, reason="Speech recognition is unavailable")
		return

	query = urlencode({
		"model_id": "scribe_v2_realtime",
		"audio_format": "pcm_16000",
		"sample_rate": STT_SAMPLE_RATE,
		"commit_strategy": "manual",
	})
	try:
		upstream = await connect(
			f"{STT_URL}?{query}",
			additional_headers={"xi-api-key": api_key},
			max_size=None,
		)
	except Exception:
		await websocket.close(code=1011, reason="Speech recognition connection failed")
		return

	tasks = [
		asyncio.create_task(_forward_stt_events(websocket, upstream)),
		asyncio.create_task(_forward_stt_audio(websocket, upstream)),
	]
	try:
		_, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
		for task in pending:
			task.cancel()
		for task in tasks:
			try:
				await task
			except BaseException:
				pass
	finally:
		await upstream.close()


async def handle_websocket(websocket) -> None:
	if websocket.request.path.split("?", 1)[0] == "/ws/stt":
		await handle_stt_connection(websocket)
		return
	await handle_connection(websocket)


def create_websocket_server():
	return serve(
		handle_websocket,
		HOST,
		WEBSOCKET_PORT,
		compression=None,
		max_size=64 * 1024,
	)