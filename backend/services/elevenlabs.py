import os
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from config import ELEVENLABS_VOICES_URL
from services.audio import DUMMY_AUDIO

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
VOICE_PREVIEW_TEXT = "Hello! This is a preview of this voice."


async def list_available_voices() -> list[dict]:
	api_key = os.getenv("ELEVENLABS_API_KEY")
	if not api_key:
		raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured")

	try:
		async with httpx.AsyncClient(timeout=10) as client:
			response = await client.get(
				ELEVENLABS_VOICES_URL,
				headers={"xi-api-key": api_key},
			)
		response.raise_for_status()
		return response.json().get("voices", [])
	except httpx.HTTPStatusError as error:
		raise HTTPException(
			status_code=502,
			detail="ElevenLabs rejected the voices request",
		) from error
	except (httpx.HTTPError, ValueError) as error:
		raise HTTPException(
			status_code=502,
			detail="Unable to retrieve voices from ElevenLabs",
		) from error


async def generate_speech(text: str, voice_id: str) -> bytes:
	"""Mock TTS response. Replace this body with the ElevenLabs TTS call."""
	_ = (text, voice_id)
	return DUMMY_AUDIO


async def generate_voice_preview(voice_id: str, text: str = VOICE_PREVIEW_TEXT) -> bytes:
	api_key = os.getenv("ELEVENLABS_API_KEY")
	if not api_key:
		raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured")

	url = f"{ELEVENLABS_TTS_URL}/{quote(voice_id, safe='')}"
	try:
		async with httpx.AsyncClient(timeout=30) as client:
			response = await client.post(
				url,
				params={
					"output_format": "mp3_44100_128",
					"optimize_streaming_latency": "2",
				},
				headers={"xi-api-key": api_key},
				json={
					"text": text,
					"model_id": os.getenv("TTS_PREVIEW_MODEL_ID", "eleven_flash_v2_5"),
				},
			)
		response.raise_for_status()
		return response.content
	except httpx.HTTPStatusError as error:
		if error.response.status_code in (400, 404, 422):
			raise HTTPException(status_code=422, detail="ElevenLabs could not generate a preview for this voice") from error
		raise HTTPException(status_code=502, detail="ElevenLabs rejected the voice preview request") from error
	except httpx.HTTPError as error:
		raise HTTPException(status_code=502, detail="Unable to retrieve the ElevenLabs voice preview") from error