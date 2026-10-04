import os
import sys

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

TEXT = "The first move is what sets everything in motion."
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George"
MODEL_ID = "eleven_v4"
OUTPUT_FORMAT = "mp3_44100_128"


def tts(text: str) -> bytes:
    """Convert text to audio and return all of the bytes."""
    stream = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    )
    return b"".join(stream)


def whole_sentence() -> bytes:
    return tts(TEXT)


def word_by_word() -> bytes:
    chunks = []
    for word in TEXT.split(" "):
        print(f"  generating: {word!r}")
        chunks.append(tts(word))
    # Concatenating MP3 frames works for playback
    return b"".join(chunks)


def save(path: str, audio: bytes) -> None:
    with open(path, "wb") as f:
        f.write(audio)
    print(f"  saved {path} ({len(audio)} bytes)")


if __name__ == "__main__":
    # usage: python -m tools.rest_reference [sentence|words|both]
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("sentence", "both"):
        print("Whole sentence:")
        audio = whole_sentence()
        save("sentence.mp3", audio)
        play(audio)

    if mode in ("words", "both"):
        print("Word by word (assembled):")
        audio = word_by_word()
        save("words.mp3", audio)
        play(audio)
