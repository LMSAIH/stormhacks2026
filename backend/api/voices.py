from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_authenticated_user
from services.elevenlabs import list_available_voices
from state import get_default_voice_id, set_default_voice_id


router = APIRouter(prefix="/api")


class VoiceSelection(BaseModel):
	voice_id: str = Field(min_length=1, max_length=128, strip_whitespace=True)


@router.get("/voices", dependencies=[Depends(require_authenticated_user)])
async def list_voices() -> dict:
	return {
		"voices": await list_available_voices(),
		"default_voice_id": get_default_voice_id(),
	}


@router.put("/voice", dependencies=[Depends(require_authenticated_user)])
async def update_default_voice(selection: VoiceSelection) -> dict:
	voice_id = selection.voice_id.strip()
	if not voice_id:
		raise HTTPException(status_code=422, detail="voice_id must not be blank")
	set_default_voice_id(voice_id)
	return {"default_voice_id": voice_id}