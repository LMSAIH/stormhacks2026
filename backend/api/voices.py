from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from api.auth import require_authenticated_user
from services.elevenlabs import generate_voice_preview, list_available_voices
from services.chat_store import ChatStoreUnavailableError, get_user_voice_id, set_user_voice_id
from state import get_default_voice_id


router = APIRouter(prefix="/api")


class VoiceSelection(BaseModel):
	model_config = {
		"json_schema_extra": {
			"examples": [{"voice_id": "JBFqnCBsd6RMkjVDRZzb"}],
		}
	}
	voice_id: str = Field(min_length=1, max_length=128, strip_whitespace=True)


class VoicePreviewRequest(VoiceSelection):
	text: str = Field(default="Hello! This is a preview of this voice.", min_length=1, max_length=300)


@router.get(
	"/voices",
	responses={
		200: {
			"description": "Available ElevenLabs voices and this account's selected voice.",
			"content": {
				"application/json": {
					"example": {
						"voices": [
							{
								"voice_id": "JBFqnCBsd6RMkjVDRZzb",
								"name": "George",
								"category": "premade",
								"preview_url": "https://storage.googleapis.com/eleven-public-prod/...",
								"labels": {"accent": "american", "gender": "male"},
							},
						],
						"default_voice_id": "JBFqnCBsd6RMkjVDRZzb",
					},
				},
			},
		},
	},
)
async def list_voices(user: dict = Depends(require_authenticated_user)) -> dict:
	try:
		voice_id = await get_user_voice_id(user["id"])
	except ChatStoreUnavailableError as error:
		raise HTTPException(status_code=503, detail="Voice preferences are unavailable") from error
	return {
		"voices": await list_available_voices(),
		"default_voice_id": voice_id or get_default_voice_id(),
	}


@router.post(
	"/voices/preview",
	dependencies=[Depends(require_authenticated_user)],
	responses={
		200: {
			"description": "MP3 audio bytes for the requested voice.",
			"content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}},
		},
	},
)
async def preview_voice(request: VoicePreviewRequest) -> Response:
	text = request.text.strip()
	if not text:
		raise HTTPException(status_code=422, detail="text must not be blank")
	audio = await generate_voice_preview(request.voice_id.strip(), text)
	return Response(
		content=audio,
		media_type="audio/mpeg",
		headers={"Cache-Control": "no-store"},
	)


@router.put(
	"/voice",
	responses={
		200: {
			"description": "Voice preference saved for the signed-in account.",
			"content": {"application/json": {"example": {"default_voice_id": "JBFqnCBsd6RMkjVDRZzb"}}},
		},
	},
)
async def update_default_voice(
	selection: VoiceSelection,
	user: dict = Depends(require_authenticated_user),
) -> dict:
	voice_id = selection.voice_id.strip()
	if not voice_id:
		raise HTTPException(status_code=422, detail="voice_id must not be blank")
	try:
		await set_user_voice_id(user["id"], voice_id)
	except ChatStoreUnavailableError as error:
		raise HTTPException(status_code=503, detail="Voice preferences are unavailable") from error
	return {"default_voice_id": voice_id}