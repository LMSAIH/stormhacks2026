"""Lip-reader HTTP service (contract: .context/project-brief.md §5).

    uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port 8000

POST /lipread        a webcam clip (or an mp4 of aligned crops with precropped=true)
POST /lipread/crops  raw uint8 mouth crops from a client that detects + aligns locally (JS tier)
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lipread.corrector import Corrector
from lipread.model import LipReader
from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches, to_model_input
from lipread.video import MODEL_FPS, load_video_25fps

log = logging.getLogger("lipread.serve")
MIN_SECONDS, MAX_SECONDS = 0.5, 10.0
MODEL_NAME = os.environ.get("LIPREAD_MODEL", "LRS3_V_WER19.1")
# /lipread/crops frame sizes: 96 = the aligned mouth patch, 88 = its centre crop (the model input).
CROP_SIZES = (96, 88)
# Hard cap on a /lipread/crops body: the longest raw clip plus headroom for gzip framing, which can
# make incompressible pixels a few hundred bytes larger than the input.
MAX_CROPS_BODY = int(MAX_SECONDS * MODEL_FPS) * 96 * 96 + 64 * 1024

app = FastAPI(title="lipread", version="0.1.0")
# Electron renderer / dev server call this directly during the hackathon; tighten later.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# espnet's CTC prefix scorer keeps per-search state on the shared scorer (`CTCPrefixScorer.impl`),
# so concurrent beam searches from the endpoint threadpool would corrupt each other. Greedy is
# stateless and runs unlocked.
_beam_lock = threading.Lock()


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


def _check_decode(decode: str) -> None:
    if decode not in ("greedy", "beam"):
        raise HTTPException(422, {"error": "bad_decode"})


def _check_duration(n_frames: int) -> None:
    seconds = n_frames / MODEL_FPS
    if seconds < MIN_SECONDS:
        raise HTTPException(422, {"error": "clip_too_short"})
    if seconds > MAX_SECONDS:
        raise HTTPException(422, {"error": "clip_too_long"})


def _recognize(x: torch.Tensor, decode: str, correct: bool, t0: float, t1: float, t2: float) -> dict:
    """VSR (+ optional corrector) on model input `x` → the response shared by /lipread*.

    The caller times its own stages: load = t0→t1, crop = t1→t2.
    """
    with _beam_lock if decode == "beam" else contextlib.nullcontext():
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
        # beam only (greedy: []): up to 3 distinct readings, best first; [0] matches raw_text
        "alternatives": [{"text": t, "score": round(s, 3)} for t, s in result.alternatives],
        "frames": int(x.shape[1]),
        "latency_ms": {
            "load": ms(t0, t1), "crop": ms(t1, t2), "vsr": ms(t2, t3),
            "correct": ms(t3, t4), "total": ms(t0, t4),
        },
    }


@app.post("/lipread")
def lipread(
    file: UploadFile = File(...),
    decode: str = Form(os.environ.get("LIPREAD_DECODE", "greedy")),
    correct: bool = Form(True),
    precropped: bool = Form(False),
) -> dict:
    """`precropped=true`: the clip is already aligned 96x96 mouth crops (skip face detection)."""
    _check_decode(decode)
    t0 = time.perf_counter()
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            frames = load_video_25fps(tmp.name)
        except ValueError:
            raise HTTPException(422, {"error": "unreadable_video"})
    _check_duration(len(frames))
    t1 = time.perf_counter()
    try:
        patches = precropped_patches(frames) if precropped else cropper().crop(frames)
    except NoFaceError:
        raise HTTPException(422, {"error": "no_face_detected"})
    x = to_model_input(patches)
    t2 = time.perf_counter()
    return _recognize(x, decode, correct, t0, t1, t2)


async def _crops_body(request: Request) -> bytearray:
    """Request body, read with a hard cap. A bytearray so numpy/torch can wrap it without a copy."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_CROPS_BODY:
            raise HTTPException(413, {"error": "body_too_large"})
    return body


def _content_coding(header: str | None) -> str:
    """`Content-Encoding` → "identity" or "gzip"; anything else is a 415."""
    codings = [c.strip().lower() for c in (header or "").split(",")]
    codings = [c for c in codings if c not in ("", "identity")]
    if not codings:
        return "identity"
    if codings in (["gzip"], ["x-gzip"]):
        return "gzip"
    raise HTTPException(415, {"error": "unsupported_encoding",
                              "message": f"Content-Encoding {header!r}: send gzip or no encoding"})


def _gunzip(data: bytes | bytearray, limit: int) -> bytearray:
    """Inflate a (possibly multi-member) gzip body, never past `limit` + 1 bytes (zip-bomb guard)."""
    out = bytearray()
    while data:
        z = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        out += z.decompress(data, limit + 1 - len(out))
        if len(out) > limit:
            break  # already too big: the caller rejects it without inflating the rest
        if not z.eof:
            raise zlib.error("truncated gzip stream")
        data = z.unused_data
    return out


# The body is read by a dependency (raw bytes, any Content-Type), so describe it for /docs here.
_CROPS_BODY_DOC = {"requestBody": {"required": True, "content": {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


@app.post("/lipread/crops", openapi_extra=_CROPS_BODY_DOC)
def lipread_crops(
    t: int = Query(..., description="number of frames at 25 fps (13-250 = 0.5-10 s)"),
    h: int = Query(96, description="frame height: 96 = aligned mouth patch, 88 = its centre crop"),
    w: int = Query(96, description="frame width, must equal h"),
    decode: str = Query("beam", description="greedy | beam"),
    correct: bool = Query(False),
    content_encoding: str | None = Header(None),
    body: bytearray = Depends(_crops_body),
) -> dict:
    """Mouth crops from a client that detects + aligns the face itself (the JS crop pipeline).

    Body: t*h*w raw uint8 bytes (row-major gray frames), optionally `Content-Encoding: gzip`.
    Lossless, unlike `/lipread` with precropped=true, which round-trips the crops through mp4.
    """
    _check_decode(decode)
    if h != w or h not in CROP_SIZES:
        raise HTTPException(422, {"error": "bad_shape",
                                  "message": f"h and w must both be 96 or 88, got {h}x{w}"})
    _check_duration(t)
    coding = _content_coding(content_encoding)
    t0 = time.perf_counter()
    expected = t * h * w
    if coding == "gzip":
        try:
            body = _gunzip(body, expected)
        except zlib.error:
            raise HTTPException(422, {"error": "bad_gzip"})
    if len(body) != expected:
        got = f"more than {expected}" if coding == "gzip" and len(body) > expected else str(len(body))
        raise HTTPException(422, {"error": "body_size_mismatch",
                                  "message": f"body must be t*h*w = {t}*{h}*{w} = {expected} bytes, got {got}"})
    t1 = time.perf_counter()
    # An 88x88 frame is already the centre crop, so to_model_input's CenterCrop(88) leaves it as is:
    # both sizes give bit-identical model input for the same patches.
    x = to_model_input(np.frombuffer(body, dtype=np.uint8).reshape(t, h, w))
    t2 = time.perf_counter()
    return _recognize(x, decode, correct, t0, t1, t2)


class CorrectIn(BaseModel):
    text: str


@app.post("/correct")
def correct_text(body: CorrectIn) -> dict:
    """For the local ONNX tiers: browser/Electron lip-reads, server only cleans up."""
    return {"text": corrector().correct(body.text)}
