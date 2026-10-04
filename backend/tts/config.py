"""Environment-driven TTS configuration. Backend is TTS_BACKEND=flash|v4."""

import os

from dotenv import load_dotenv

from tts.base import TtsConfig

load_dotenv()

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George"


def backend_name() -> str:
    return os.getenv("TTS_BACKEND", "flash")


def load_tts_config(**overrides) -> TtsConfig:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (backend/.env)")
    cfg = TtsConfig(
        api_key=key,
        voice_id=os.getenv("TTS_VOICE_ID", DEFAULT_VOICE_ID),
        model_id=os.getenv("TTS_MODEL_ID") or None,
        output_format=os.getenv("TTS_OUTPUT_FORMAT", "pcm_24000"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg