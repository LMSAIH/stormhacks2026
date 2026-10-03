"""FastAPI app. Run: uvicorn app.main:app --port 8000"""
from __future__ import annotations

from fastapi import FastAPI

from app.routes.tts_ws import router as tts_router
from tts import available

app = FastAPI()
app.include_router(tts_router)


@app.get("/health")
async def health():
    return {"status": "ok", "backends": available()}
