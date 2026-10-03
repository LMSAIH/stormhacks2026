import asyncio
import io
import math
import wave

from websockets.asyncio.server import serve


HOST = "0.0.0.0"
PORT = 8765
TERMINATOR = "\\x"
SAMPLE_RATE = 8000


def make_dummy_audio() -> bytes:
	samples = bytearray()
	for index in range(SAMPLE_RATE // 8):
		envelope = 1 - index / (SAMPLE_RATE // 8)
		sample = int(7000 * envelope * math.sin(2 * math.pi * 660 * index / SAMPLE_RATE))
		samples.extend(sample.to_bytes(2, byteorder="little", signed=True))

	buffer = io.BytesIO()
	with wave.open(buffer, "wb") as wav:
		wav.setnchannels(1)
		wav.setsampwidth(2)
		wav.setframerate(SAMPLE_RATE)
		wav.writeframes(samples)
	return buffer.getvalue()


DUMMY_AUDIO = make_dummy_audio()


async def callElevenLabs(text: str) -> bytes:
	"""Mock TTS response. Replace this body with the ElevenLabs call."""
	_ = text
	return DUMMY_AUDIO


async def handle_connection(websocket) -> None:
	pending_text = ""
	async for message in websocket:
		if not isinstance(message, str):
			continue

		pending_text += message
		while TERMINATOR in pending_text:
			text, pending_text = pending_text.split(TERMINATOR, 1)
			audio = await callElevenLabs(text)
			await websocket.send(audio)


async def main() -> None:
	async with serve(
		handle_connection,
		HOST,
		PORT,
		compression=None,
		max_size=64 * 1024,
	):
		print(f"WebSocket server listening on ws://{HOST}:{PORT}")
		await asyncio.Future()


if __name__ == "__main__":
	asyncio.run(main())
