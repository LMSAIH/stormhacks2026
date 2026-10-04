import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.auth import router as auth_router
from api.chat import chats_router, router as chat_router
from api.voices import router as voices_router
from config import (
	API_PORT,
	FRONTEND_ORIGINS,
	HOST,
	SESSION_HTTPS_ONLY,
	SESSION_SECRET,
	WEBSOCKET_PORT,
)
from services.chat_store import close_chat_pool
from websocket_server import create_websocket_server


@asynccontextmanager
async def lifespan(_app):
	try:
		yield
	finally:
		await close_chat_pool()


api = FastAPI(title="Voice API", lifespan=lifespan)
api.add_middleware(
	CORSMiddleware,
	allow_origins=FRONTEND_ORIGINS,
	allow_methods=["GET", "POST", "PUT"],
	allow_headers=["Content-Type"],
	allow_credentials=True,
)
api.add_middleware(
	SessionMiddleware,
	secret_key=SESSION_SECRET,
	session_cookie="voice_session",
	same_site="lax",
	https_only=SESSION_HTTPS_ONLY,
)
api.include_router(auth_router)
api.include_router(chat_router)
api.include_router(chats_router)
api.include_router(voices_router)


async def main() -> None:
	async with create_websocket_server():
		print(f"WebSocket server listening on ws://{HOST}:{WEBSOCKET_PORT}")
		print(f"Voice REST API listening on http://{HOST}:{API_PORT}")
		config = uvicorn.Config(api, host=HOST, port=API_PORT, log_level="info")
		await uvicorn.Server(config).serve()


if __name__ == "__main__":
	asyncio.run(main())