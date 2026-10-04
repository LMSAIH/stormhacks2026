import os

import httpx
from fastapi import HTTPException

from config import ELEVENLABS_VOICES_URL
from services.audio import DUMMY_AUDIO


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