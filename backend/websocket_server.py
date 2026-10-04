import asyncio

from websockets.asyncio.server import serve

from auth_session import get_user_from_cookie_header
from config import HOST, TERMINATOR, WEBSOCKET_PORT
from services.chat_store import ChatStoreUnavailableError, get_user_voice_id
from state import get_default_voice_id
from tts import create_backend
from tts.config import backend_name, load_tts_config


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


def create_websocket_server():
	return serve(
		handle_connection,
		HOST,
		WEBSOCKET_PORT,
		compression=None,
		max_size=64 * 1024,
	)