"""Test endpoint: POST an audio file, get back who spoke when.

    curl -X POST --data-binary @meeting.mp3 -H "Content-Type: audio/mpeg" \
         "http://localhost:5000/api/diarize?format=mp3"

The body is the raw file (no multipart, so no extra dependency). Only active when DIARIZATION=1,
otherwise it answers 404. Optional query params override any DiarizationConfig field,
e.g. ?threshold=0.7&max_speakers=3. There is deliberately no login on this route: it is a local
testing aid, so keep DIARIZATION off on anything exposed to the internet.
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from diarization.config import DiarizationConfig, is_enabled

router = APIRouter()

MAX_BYTES = 50 * 1024 * 1024
_TUNABLE = {f.name: f.type for f in dataclasses.fields(DiarizationConfig)}


@router.post("/api/diarize")
async def diarize(request: Request) -> dict:
    if not is_enabled():
        raise HTTPException(404, "diarization is disabled (set DIARIZATION=1)")

    overrides = {}
    for key, raw in request.query_params.items():
        if key == "format":
            continue
        if key not in _TUNABLE:
            raise HTTPException(422, f"unknown parameter {key!r}")
        default = getattr(DiarizationConfig, key)
        try:
            overrides[key] = type(default)(raw)
        except ValueError:
            raise HTTPException(422, f"bad value for {key}: {raw!r}")

    data = await request.body()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "audio too large (50 MB max)")

    from diarization.audio import AudioDecodeError
    from diarization.service import diarize_audio

    suffix = request.query_params.get("format", "")
    try:
        return await run_in_threadpool(diarize_audio, data, suffix=suffix, overrides=overrides)
    except AudioDecodeError as e:
        raise HTTPException(422, str(e))
    except ImportError as e:
        raise HTTPException(500, f"diarization dependencies missing: {e}")
