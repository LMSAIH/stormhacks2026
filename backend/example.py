from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

text="The first move is what sets everything in motion."
audio_list = text.split(" ")

for audio in audio_list:
    audio = elevenlabs.text_to_speech.convert(
        text=audio,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" - browse voices at elevenlabs.io/app/voice-library
        model_id="eleven_v4",
        output_format="mp3_44100_128",
        )
    play(audio)






