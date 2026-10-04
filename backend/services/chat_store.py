import asyncio
import json
from typing import Any
from uuid import UUID, uuid4

from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool
from psycopg.types.json import Jsonb


MAX_CHAT_BYTES = 5 * 1024 * 1024
_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


class ChatTooLargeError(ValueError):
	pass


class ChatStoreUnavailableError(RuntimeError):
	pass


async def _get_pool() -> AsyncConnectionPool:
	global _pool
	if _pool is not None:
		return _pool

	async with _pool_lock:
		if _pool is not None:
			return _pool

		from config import DATABASE_URL

		if not DATABASE_URL:
			raise ChatStoreUnavailableError("TIMESCALE_SERVICE_URL is not configured")

		try:
			pool = AsyncConnectionPool(
				conninfo=DATABASE_URL,
				min_size=1,
				max_size=10,
				kwargs={"row_factory": tuple_row},
				open=False,
			)
			await pool.open()
			async with pool.connection() as connection:
				await connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS user_chats (
						chat_id UUID PRIMARY KEY,
						user_id TEXT NOT NULL,
						title TEXT NOT NULL,
						speakers JSONB NOT NULL DEFAULT '[]'::jsonb,
						messages JSONB NOT NULL DEFAULT '[]'::jsonb,
						created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
						updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
					)
					"""
				)
				await connection.execute(
					"CREATE INDEX IF NOT EXISTS user_chats_owner_created_idx "
					"ON user_chats (user_id, created_at DESC)"
				)
				await connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS user_voice_preferences (
						user_id TEXT PRIMARY KEY,
						voice_id TEXT NOT NULL,
						updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
					)
					"""
				)
		except Exception as error:
			if "pool" in locals():
				await pool.close()
			raise ChatStoreUnavailableError("Unable to connect to chat storage") from error

		_pool = pool
		return _pool


async def save_chat(
	user_id: str,
	speakers: list[dict[str, str]],
	messages: list[dict[str, Any]],
) -> None:
	_check_chat_size(speakers, messages)

	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			async with connection.transaction():
				cursor = await connection.execute(
					"""
					UPDATE user_chats
					SET title = %s, speakers = %s, messages = %s,
						updated_at = CURRENT_TIMESTAMP
					WHERE chat_id = (
						SELECT chat_id FROM user_chats
						WHERE user_id = %s
						ORDER BY updated_at DESC, chat_id
						LIMIT 1
					)
					""",
					(_chat_title(messages), Jsonb(speakers), Jsonb(messages), user_id),
				)
				if cursor.rowcount == 0:
					await connection.execute(
						"""
						INSERT INTO user_chats
							(chat_id, user_id, title, speakers, messages)
						VALUES (%s, %s, %s, %s, %s)
						""",
						(uuid4(), user_id, _chat_title(messages), Jsonb(speakers), Jsonb(messages)),
					)
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to save chat") from error


async def get_chat(user_id: str) -> dict[str, Any] | None:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"""
				SELECT speakers, messages
				FROM user_chats
				WHERE user_id = %s
				ORDER BY updated_at DESC, chat_id
				LIMIT 1
				""",
				(user_id,),
			)
			row = await cursor.fetchone()
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to retrieve chat") from error

	if row is None:
		return None
	return {"speakers": row[0], "messages": row[1]}


async def get_user_voice_id(user_id: str) -> str | None:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"SELECT voice_id FROM user_voice_preferences WHERE user_id = %s",
				(user_id,),
			)
			row = await cursor.fetchone()
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to retrieve voice preference") from error
	return row[0] if row else None


async def set_user_voice_id(user_id: str, voice_id: str) -> None:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			await connection.execute(
				"""
				INSERT INTO user_voice_preferences (user_id, voice_id)
				VALUES (%s, %s)
				ON CONFLICT (user_id) DO UPDATE
				SET voice_id = EXCLUDED.voice_id, updated_at = CURRENT_TIMESTAMP
				""",
				(user_id, voice_id),
			)
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to save voice preference") from error


def _chat_title(messages: list[dict[str, Any]]) -> str:
	for message in messages:
		text = " ".join(message["text"].split())
		if text:
			return text[:80]
	return "New chat"


def _check_chat_size(speakers: list[dict[str, str]], messages: list[dict[str, Any]]) -> None:
	serialized_chat = json.dumps(
		{"speakers": speakers, "messages": messages},
		separators=(",", ":"),
		ensure_ascii=False,
	)
	if len(serialized_chat.encode("utf-8")) > MAX_CHAT_BYTES:
		raise ChatTooLargeError


async def create_user_chat(
	user_id: str,
	speakers: list[dict[str, str]],
	messages: list[dict[str, Any]],
) -> dict[str, str]:
	_check_chat_size(speakers, messages)
	chat_id = uuid4()
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"""
				INSERT INTO user_chats (chat_id, user_id, title, speakers, messages)
				VALUES (%s, %s, %s, %s, %s)
				RETURNING created_at
				""",
				(chat_id, user_id, _chat_title(messages), Jsonb(speakers), Jsonb(messages)),
			)
			created_at = (await cursor.fetchone())[0]
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to create chat") from error
	return {"id": str(chat_id), "title": _chat_title(messages), "created_at": created_at.isoformat()}


async def list_user_chats(user_id: str) -> list[dict[str, str]]:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"""
				SELECT chat_id, title, created_at
				FROM user_chats
				WHERE user_id = %s
				ORDER BY created_at DESC, chat_id
				""",
				(user_id,),
			)
			rows = await cursor.fetchall()
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to list chats") from error
	return [
		{"id": str(chat_id), "title": title, "created_at": created_at.isoformat()}
		for chat_id, title, created_at in rows
	]


async def get_user_chat(user_id: str, chat_id: UUID) -> dict[str, Any] | None:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"""
				SELECT chat_id, title, speakers, messages, created_at, updated_at
				FROM user_chats
				WHERE user_id = %s AND chat_id = %s
				""",
				(user_id, chat_id),
			)
			row = await cursor.fetchone()
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to retrieve chat") from error
	if row is None:
		return None
	chat_id, title, speakers, messages, created_at, updated_at = row
	return {
		"id": str(chat_id),
		"title": title,
		"speakers": speakers,
		"messages": messages,
		"created_at": created_at.isoformat(),
		"updated_at": updated_at.isoformat(),
	}


async def save_user_chat(
	user_id: str,
	chat_id: UUID,
	speakers: list[dict[str, str]],
	messages: list[dict[str, Any]],
) -> bool:
	_check_chat_size(speakers, messages)
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"""
				UPDATE user_chats
				SET title = %s, speakers = %s, messages = %s, updated_at = CURRENT_TIMESTAMP
				WHERE user_id = %s AND chat_id = %s
				""",
				(
					_chat_title(messages),
					Jsonb(speakers),
					Jsonb(messages),
					user_id,
					chat_id,
				),
			)
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to save chat") from error
	return cursor.rowcount > 0


async def close_chat_pool() -> None:
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None