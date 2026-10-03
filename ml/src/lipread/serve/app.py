"""Lip-reader HTTP service (contract: .context/project-brief.md §5).

    uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lipread.corrector import Corrector
from lipread.model import LipReader
from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches, to_model_input
from lipread.video import MODEL_FPS, load_video_25fps

log = logging.getLogger("lipread.serve")
MIN_SECONDS, MAX_SECONDS = 0.5, 10.0
MODEL_NAME = os.environ.get("LIPREAD_MODEL", "LRS3_V_WER19.1")

app = FastAPI(title="lipread", version="0.1.0")
# Electron renderer / dev server call this directly during the hackathon; tighten later.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def reader() -> LipReader:
    return LipReader(model_name=MODEL_NAME, beam_size=int(os.environ.get("LIPREAD_BEAM_SIZE", "40")))


@lru_cache(maxsize=1)
def cropper() -> MouthCropper:
    return MouthCropper()


@lru_cache(maxsize=1)
def corrector() -> Corrector:
    return Corrector()


@app.on_event("startup")
def _warm() -> None:
    if os.environ.get("LIPREAD_WARM", "0") == "1":
        reader()
        cropper()


@app.get("/health")
def health() -> dict:
    loaded = reader.cache_info().currsize > 0
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": reader().device if loaded else None,
        "loaded": loaded,
        "corrector": corrector().enabled,
    }


@app.post("/lipread")
def lipread(
    file: UploadFile = File(...),
    decode: str = Form(os.environ.get("LIPREAD_DECODE", "greedy")),
    correct: bool = Form(True),
    precropped: bool = Form(False),
) -> dict:
    """`precropped=true`: the clip is already aligned 96x96 mouth crops (skip face detection)."""
    if decode not in ("greedy", "beam"):
        raise HTTPException(422, {"error": "bad_decode"})
    t0 = time.perf_counter()
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            frames = load_video_25fps(tmp.name)
        except ValueError:
            raise HTTPException(422, {"error": "unreadable_video"})
    seconds = len(frames) / MODEL_FPS
    if seconds < MIN_SECONDS:
        raise HTTPException(422, {"error": "clip_too_short"})
    if seconds > MAX_SECONDS:
        raise HTTPException(422, {"error": "clip_too_long"})
    t1 = time.perf_counter()
    try:
        patches = precropped_patches(frames) if precropped else cropper().crop(frames)
    except NoFaceError:
        raise HTTPException(422, {"error": "no_face_detected"})
    x = to_model_input(patches)
    t2 = time.perf_counter()
    result = reader().transcribe(x, decode=decode)
    t3 = time.perf_counter()
    text = result.text
    if correct:
        try:
            text = corrector().correct(result.text)
        except Exception:  # corrector is best-effort; never fail the request over it
            log.exception("corrector failed")
    t4 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    return {
        "text": text,
        "raw_text": result.text,
        "confidence": result.confidence,
        "frames": int(x.shape[1]),
        "latency_ms": {
            "load": ms(t0, t1), "crop": ms(t1, t2), "vsr": ms(t2, t3),
            "correct": ms(t3, t4), "total": ms(t0, t4),
        },
    }


class CorrectIn(BaseModel):
    text: str


@app.post("/correct")
def correct_text(body: CorrectIn) -> dict:
    """For the local ONNX tiers: browser/Electron lip-reads, server only cleans up."""
    return {"text": corrector().correct(body.text)}
