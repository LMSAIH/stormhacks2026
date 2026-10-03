import asyncio
import json
from typing import Any

from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool


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
					CREATE TABLE IF NOT EXISTS chat_conversations (
						user_id TEXT PRIMARY KEY,
						updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
					)
					"""
				)
				await connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS chat_speakers (
						user_id TEXT NOT NULL REFERENCES chat_conversations(user_id) ON DELETE CASCADE,
						speaker_id TEXT NOT NULL,
						display_name TEXT NOT NULL,
						position INTEGER NOT NULL CHECK (position >= 0),
						PRIMARY KEY (user_id, speaker_id),
						UNIQUE (user_id, position)
					)
					"""
				)
				await connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS chat_messages (
						user_id TEXT NOT NULL,
						position BIGINT NOT NULL CHECK (position >= 0),
						speaker_id TEXT NOT NULL,
						content TEXT NOT NULL,
						created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
						PRIMARY KEY (user_id, position),
						FOREIGN KEY (user_id, speaker_id)
							REFERENCES chat_speakers(user_id, speaker_id)
							ON DELETE CASCADE
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
	messages: list[dict[str, str]],
) -> None:
	serialized_chat = json.dumps(
		{"speakers": speakers, "messages": messages},
		separators=(",", ":"),
		ensure_ascii=False,
	)
	if len(serialized_chat.encode("utf-8")) > MAX_CHAT_BYTES:
		raise ChatTooLargeError

	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			async with connection.transaction():
				await connection.execute(
					"""
					INSERT INTO chat_conversations (user_id, updated_at)
					VALUES (%s, CURRENT_TIMESTAMP)
					ON CONFLICT (user_id) DO UPDATE
					SET updated_at = CURRENT_TIMESTAMP
					""",
					(user_id,),
				)
				await connection.execute(
					"DELETE FROM chat_speakers WHERE user_id = %s",
					(user_id,),
				)
				if speakers:
					await connection.executemany(
						"""
						INSERT INTO chat_speakers
							(user_id, speaker_id, display_name, position)
						VALUES (%s, %s, %s, %s)
						""",
						[
							(user_id, speaker["id"], speaker["name"], position)
							for position, speaker in enumerate(speakers)
						],
					)
				if messages:
					await connection.executemany(
						"""
						INSERT INTO chat_messages
							(user_id, position, speaker_id, content)
						VALUES (%s, %s, %s, %s)
						""",
						[
							(user_id, position, message["speaker_id"], message["text"])
							for position, message in enumerate(messages)
						],
					)
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to save chat") from error


async def get_chat(user_id: str) -> dict[str, Any] | None:
	pool = await _get_pool()
	try:
		async with pool.connection() as connection:
			cursor = await connection.execute(
				"SELECT 1 FROM chat_conversations WHERE user_id = %s",
				(user_id,),
			)
			if await cursor.fetchone() is None:
				return None

			speaker_cursor = await connection.execute(
				"""
				SELECT speaker_id, display_name
				FROM chat_speakers
				WHERE user_id = %s
				ORDER BY position
				""",
				(user_id,),
			)
			message_cursor = await connection.execute(
				"""
				SELECT speaker_id, content
				FROM chat_messages
				WHERE user_id = %s
				ORDER BY position
				""",
				(user_id,),
			)
			speakers = await speaker_cursor.fetchall()
			messages = await message_cursor.fetchall()
	except Exception as error:
		raise ChatStoreUnavailableError("Unable to retrieve chat") from error

	return {
		"speakers": [{"id": speaker_id, "name": name} for speaker_id, name in speakers],
		"messages": [
			{"speaker_id": speaker_id, "text": content}
			for speaker_id, content in messages
		],
	}


async def close_chat_pool() -> None:
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None