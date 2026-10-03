import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.voices import router as voices_router
from config import API_PORT, FRONTEND_ORIGINS, HOST, WEBSOCKET_PORT
from websocket_server import create_websocket_server


api = FastAPI(title="Voice API")
api.add_middleware(
	CORSMiddleware,
	allow_origins=FRONTEND_ORIGINS,
	allow_methods=["GET", "PUT"],
	allow_headers=["Content-Type"],
)
api.include_router(voices_router)


async def main() -> None:
	async with create_websocket_server():
		print(f"WebSocket server listening on ws://{HOST}:{WEBSOCKET_PORT}")
		print(f"Voice REST API listening on http://{HOST}:{API_PORT}")
		config = uvicorn.Config(api, host=HOST, port=API_PORT, log_level="info")
		await uvicorn.Server(config).serve()


if __name__ == "__main__":
	asyncio.run(main())