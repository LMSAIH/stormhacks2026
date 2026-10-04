from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.auth import require_authenticated_user
from services.chat_store import (
	ChatStoreUnavailableError,
	ChatTooLargeError,
	create_user_chat,
	get_chat,
	get_user_chat,
	list_user_chats,
	save_chat,
	save_user_chat,
)


router = APIRouter(prefix="/api/chat", tags=["chat"])
chats_router = APIRouter(prefix="/api/chats", tags=["chats"])


class ChatSpeaker(BaseModel):
	model_config = ConfigDict(extra="forbid")

	id: str = Field(min_length=1, max_length=128, strip_whitespace=True)
	name: str = Field(min_length=1, max_length=128, strip_whitespace=True)


class ChatMessage(BaseModel):
	model_config = ConfigDict(extra="forbid")

	speaker_id: str = Field(min_length=1, max_length=128, strip_whitespace=True)
	text: str = Field(max_length=100_000)


class ChatPayload(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		json_schema_extra={
			"examples": [
				{
					"speakers": [
						{"id": "user", "name": "You"},
						{"id": "assistant", "name": "Assistant"},
					],
					"messages": [
						{"speaker_id": "user", "text": "Help me plan a weekend trip."},
						{"speaker_id": "assistant", "text": "Where would you like to go?"},
					],
				}
			]
		},
	)

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


@router.put(
	"",
	responses={
		200: {"description": "Chat saved.", "content": {"application/json": {"example": {"saved": True}}}},
	},
)
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


@router.get(
	"",
	responses={
		200: {
			"description": "Most recently updated chat for the signed-in user.",
			"content": {
				"application/json": {
					"example": {
						"speakers": [{"id": "user", "name": "You"}, {"id": "assistant", "name": "Assistant"}],
						"messages": [
							{"speaker_id": "user", "text": "Help me plan a weekend trip."},
							{"speaker_id": "assistant", "text": "Where would you like to go?"},
						],
					},
				},
			},
		},
	},
)
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


@chats_router.post(
	"",
	responses={
		200: {
			"description": "Created chat. Save the returned id to retrieve or update this chat.",
			"content": {
				"application/json": {
					"example": {
						"id": "8c3be30f-7d75-4baa-9b28-29622e772844",
						"title": "Help me plan a weekend trip.",
						"created_at": "2026-10-03T14:20:00+00:00",
					},
				},
			},
		},
	},
)
async def create_chat(
	chat: ChatPayload,
	user: dict = Depends(require_authenticated_user),
) -> dict[str, str]:
	try:
		return await create_user_chat(
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
		raise HTTPException(status_code=503, detail="Chat storage is unavailable") from error


@chats_router.get(
	"",
	responses={
		200: {
			"description": "Chats owned by the signed-in user, newest first.",
			"content": {
				"application/json": {
					"example": {
						"chats": [
							{
								"id": "8c3be30f-7d75-4baa-9b28-29622e772844",
								"title": "Help me plan a weekend trip.",
								"created_at": "2026-10-03T14:20:00+00:00",
							},
						],
					},
				},
			},
		},
	},
)
async def retrieve_user_chats(
	user: dict = Depends(require_authenticated_user),
) -> dict[str, list[dict[str, str]]]:
	try:
		chats = await list_user_chats(user["id"])
	except ChatStoreUnavailableError as error:
		raise HTTPException(status_code=503, detail="Chat storage is unavailable") from error
	return {"chats": chats}


@chats_router.get(
	"/{chat_id}",
	responses={
		200: {
			"description": "Chat details belonging to the signed-in user.",
			"content": {
				"application/json": {
					"example": {
						"id": "8c3be30f-7d75-4baa-9b28-29622e772844",
						"title": "Help me plan a weekend trip.",
						"speakers": [{"id": "user", "name": "You"}, {"id": "assistant", "name": "Assistant"}],
						"messages": [
							{"speaker_id": "user", "text": "Help me plan a weekend trip."},
							{"speaker_id": "assistant", "text": "Where would you like to go?"},
						],
						"created_at": "2026-10-03T14:20:00+00:00",
						"updated_at": "2026-10-03T14:22:00+00:00",
					},
				},
			},
		},
	},
)
async def retrieve_chat(
	chat_id: UUID,
	user: dict = Depends(require_authenticated_user),
):
	try:
		chat = await get_user_chat(user["id"], chat_id)
	except ChatStoreUnavailableError as error:
		raise HTTPException(status_code=503, detail="Chat storage is unavailable") from error
	if chat is None:
		raise HTTPException(status_code=404, detail="Chat not found")
	return chat


@chats_router.put(
	"/{chat_id}",
	responses={
		200: {"description": "Chat updated.", "content": {"application/json": {"example": {"saved": True}}}},
	},
)
async def update_chat(
	chat_id: UUID,
	chat: ChatPayload,
	user: dict = Depends(require_authenticated_user),
) -> dict[str, bool]:
	try:
		saved = await save_user_chat(
			user["id"],
			chat_id,
			[speaker.model_dump() for speaker in chat.speakers],
			[message.model_dump() for message in chat.messages],
		)
	except ChatTooLargeError as error:
		raise HTTPException(
			status_code=413,
			detail="Chat JSON must be 5 MB or smaller",
		) from error
	except ChatStoreUnavailableError as error:
		raise HTTPException(status_code=503, detail="Chat storage is unavailable") from error
	if not saved:
		raise HTTPException(status_code=404, detail="Chat not found")
	return {"saved": True}