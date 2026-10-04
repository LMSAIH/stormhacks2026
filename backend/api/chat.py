from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.auth import require_authenticated_user
from services.chat_store import (
	ChatStoreUnavailableError,
	ChatTooLargeError,
	get_chat,
	save_chat,
)


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatSpeaker(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str = Field(min_length=1, max_length=128, strip_whitespace=True)
	name: str = Field(min_length=1, max_length=128, strip_whitespace=True)


class ChatMessage(BaseModel):
	model_config = ConfigDict(extra="forbid")

	speaker_id: str = Field(min_length=1, max_length=128, strip_whitespace=True)
	text: str = Field(max_length=100_000)


class ChatPayload(BaseModel):
	model_config = ConfigDict(extra="forbid")

	speakers: list[ChatSpeaker] = Field(max_length=100)
	messages: list[ChatMessage] = Field(max_length=20_000)

	@model_validator(mode="after")
	def validate_speakers(self):
		speaker_ids = [speaker.id for speaker in self.speakers]
		if len(speaker_ids) != len(set(speaker_ids)):
			raise ValueError("speaker IDs must be unique")
		known_speaker_ids = set(speaker_ids)
		if any(message.speaker_id not in known_speaker_ids for message in self.messages):
			raise ValueError("every message must reference a listed speaker")
		return self


@router.put("")
async def save_user_chat(
	chat: ChatPayload,
	user: dict = Depends(require_authenticated_user),
) -> dict[str, bool]:
	try:
		await save_chat(
			user["id"],
			[speaker.model_dump() for speaker in chat.speakers],
			[message.model_dump() for message in chat.messages],
		)
	except ChatTooLargeError as error:
		raise HTTPException(
			status_code=413,
			detail="Chat JSON must be 5 MB or smaller",
		) from error
	except ChatStoreUnavailableError as error:
		raise HTTPException(
			status_code=503,
			detail="Chat storage is unavailable",
		) from error
	return {"saved": True}


@router.get("")
async def retrieve_user_chat(
	user: dict = Depends(require_authenticated_user),
):
	try:
		chat = await get_chat(user["id"])
	except ChatStoreUnavailableError as error:
		raise HTTPException(
			status_code=503,
			detail="Chat storage is unavailable",
		) from error
	if chat is None:
		raise HTTPException(status_code=404, detail="No saved chat found")
	return chat