import io
import math
import wave

from config import SAMPLE_RATE


def make_dummy_audio() -> bytes:
	samples = bytearray()
	frame_count = SAMPLE_RATE // 8
	for index in range(frame_count):
		envelope = 1 - index / frame_count
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