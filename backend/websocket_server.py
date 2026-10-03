from websockets.asyncio.server import serve

from auth_session import get_user_from_cookie_header
from config import HOST, TERMINATOR, WEBSOCKET_PORT
from services.elevenlabs import generate_speech
from state import get_default_voice_id


async def handle_connection(websocket) -> None:
	cookie_header = websocket.request.headers.get("Cookie", "")
	if get_user_from_cookie_header(cookie_header) is None:
		await websocket.close(code=4401, reason="Sign-in required")
		return

	pending_text = ""
	async for message in websocket:
		if not isinstance(message, str):
			continue

		pending_text += message
		while TERMINATOR in pending_text:
			text, pending_text = pending_text.split(TERMINATOR, 1)
			audio = await generate_speech(text, get_default_voice_id())
			await websocket.send(audio)


def create_websocket_server():
	return serve(
		handle_connection,
		HOST,
		WEBSOCKET_PORT,
		compression=None,
		max_size=64 * 1024,
	)