"""Lip-reader HTTP service (contract: .context/project-brief.md §5).

    uv run uvicorn lipread.serve.app:app --host 0.0.0.0 --port 8000

POST /lipread          a webcam clip (or an mp4 of aligned crops with precropped=true)
POST /lipread/crops    raw uint8 mouth crops from a client that detects + aligns locally (JS tier)
POST /lipread/phrases  the same crops + saved phrases → the model's ranking of those phrases
POST /correct          the Agentic Condom: a reading → the line the user most likely meant
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from lipread.agentic_condom import AgenticCondom, CondomRequest
from lipread.model import DEFAULT_BEAM, BeamSettings, LipReader, collapse_ctc, ids_to_text
from lipread.phrases import ctc_log_likelihood, rank_phrases
from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches, to_model_input
from lipread.video import MODEL_FPS, load_video_25fps

log = logging.getLogger("lipread.serve")
# 20 s = quality mode's sentence cap (D67); errors don't rise with length (.context/streaming-length-table.md).
MIN_SECONDS, MAX_SECONDS = 0.5, 20.0
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


def beam_settings() -> BeamSettings:
    """Quality mode's decode: model.DEFAULT_BEAM, each value overridable from the environment."""
    env = lambda name, default: type(default)(os.environ.get(f"LIPREAD_{name}", default))  # noqa: E731
    d = DEFAULT_BEAM
    return BeamSettings(env("BEAM_SIZE", d.beam_size), env("CTC_WEIGHT", d.ctc_weight),
                        env("LM_WEIGHT", d.lm_weight), env("PENALTY", d.penalty))


@lru_cache(maxsize=1)
def reader() -> LipReader:
    b = beam_settings()
    return LipReader(model_name=MODEL_NAME, beam_size=b.beam_size, ctc_weight=b.ctc_weight,
                     lm_weight=b.lm_weight, penalty=b.penalty)


@lru_cache(maxsize=1)
def cropper() -> MouthCropper:
    return MouthCropper()


@lru_cache(maxsize=1)
def condom() -> AgenticCondom:
    return AgenticCondom()


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
        "beam": asdict(reader().beam_settings) if loaded else None,
        "corrector": condom().enabled,
        "condom": {"enabled": condom().enabled, "model": condom().model},
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
    if correct:  # the condom never raises; on any failure it returns the reading
        text = condom().correct(CondomRequest(
            text=result.text, words=list(result.words),
            alternatives=[t for t, _ in result.alternatives[1:]],
            mode="quality" if decode == "beam" else "normal")).text
    t4 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    return {
        "text": text,
        "raw_text": result.text,
        "confidence": result.confidence,
        # beam only (greedy: []): up to 3 distinct readings, best first; [0] matches raw_text
        "alternatives": [{"text": t, "score": round(s, 3)} for t, s in result.alternatives],
        # per word of raw_text: confidence 0-1 (greedy: CTC frame probs; beam: n-best agreement)
        "words": [{"text": w, "confidence": c} for w, c in result.words],
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


def _decode_crops(t: int, h: int, w: int, content_encoding: str | None, body: bytearray) -> np.ndarray:
    """Validate + unpack a crops body (shared by /lipread/crops and /training-pairs) → (t, h, w) uint8."""
    if h != w or h not in CROP_SIZES:
        raise HTTPException(422, {"error": "bad_shape",
                                  "message": f"h and w must both be 96 or 88, got {h}x{w}"})
    _check_duration(t)
    coding = _content_coding(content_encoding)
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
    return np.frombuffer(body, dtype=np.uint8).reshape(t, h, w)


# The body is read by a dependency (raw bytes, any Content-Type), so describe it for /docs here.
_CROPS_BODY_DOC = {"requestBody": {"required": True, "content": {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


@app.post("/lipread/crops", openapi_extra=_CROPS_BODY_DOC)
def lipread_crops(
    t: int = Query(..., description="number of frames at 25 fps (13-500 = 0.5-20 s)"),
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
    t0 = time.perf_counter()
    crops = _decode_crops(t, h, w, content_encoding, body)
    t1 = time.perf_counter()
    # An 88x88 frame is already the centre crop, so to_model_input's CenterCrop(88) leaves it as is:
    # both sizes give bit-identical model input for the same patches.
    x = to_model_input(crops)
    t2 = time.perf_counter()
    return _recognize(x, decode, correct, t0, t1, t2)


# /lipread/phrases: at most this many saved phrases per request (the browser phrase store lists up
# to 500), each at most as long as a /training-pairs text.
MAX_PHRASES, MAX_PHRASE_CHARS = 500, 300
GZIP_TYPES = ("application/gzip", "application/x-gzip")


@app.post("/lipread/phrases")
def lipread_phrases(
    t: int = Query(..., description="number of frames at 25 fps (13-500 = 0.5-20 s)"),
    h: int = Query(96, description="frame height: 96 = aligned mouth patch, 88 = its centre crop"),
    w: int = Query(96, description="frame width, must equal h"),
    crops: UploadFile = File(..., description="the /lipread/crops body (t*h*w raw uint8 bytes); "
                                              "send it as type application/gzip when gzipped"),
    phrases: list[str] = Form(..., description="saved phrases to rank (repeat the field)"),
    reading: str = Form("", description="what the client read from these crops: the margins' baseline "
                                         "unless the greedy CTC reading is likelier"),
) -> dict:
    """Rank saved phrases by how well the model thinks each one explains the mouth crops.

    `margin` = (log P(phrase) − log P(base)) per frame from the encoder's CTC log-probs
    (`lipread.phrases.rank_phrases`; ≤ ~0), best first, where `base` is the likelier under CTC of
    `reading` and the greedy CTC reading (on-device the reading is the greedy one). Quality mode's
    counterpart of the on-device scorer (`frontend/src/lib/phrases/ctcScore.ts`): the client sends
    the crops it sent to /lipread/crops again, with its phrases, once the reading is back. Multipart,
    so browsers send it without a CORS preflight. Phrases the clip is too short for are left out.
    """
    t0 = time.perf_counter()
    if len(phrases) > MAX_PHRASES or any(len(p) > MAX_PHRASE_CHARS for p in phrases):
        raise HTTPException(422, {"error": "too_many_phrases", "message":
                                  f"at most {MAX_PHRASES} phrases of up to {MAX_PHRASE_CHARS} characters"})
    body = bytearray(crops.file.read(MAX_CROPS_BODY + 1))
    if len(body) > MAX_CROPS_BODY:
        raise HTTPException(413, {"error": "body_too_large"})
    gzipped = (crops.content_type or "").lower() in GZIP_TYPES
    frames = _decode_crops(t, h, w, "gzip" if gzipped else None, body)
    t1 = time.perf_counter()
    x = to_model_input(frames)
    t2 = time.perf_counter()
    log_probs = reader().ctc_log_probs(x).float().cpu()
    greedy_ids = collapse_ctc(log_probs.argmax(dim=-1).tolist())
    ranked = []
    # The CTC head hears no speech: no phrase is said either (the same guard as LipReader.beam).
    if greedy_ids:
        # Margins against the likelier, under CTC, of the client's reading and the greedy reading.
        # A beam reading the LM pulled away from the lips is unlikely under CTC, so every phrase
        # gained that slack: 10 wrong snaps instead of 2 on 300 LRS3 clips (.context/app-eval.md).
        # Against a greedy reading (what on-device scoring uses) nothing changes.
        greedy = ids_to_text(greedy_ids, reader().token_list)
        ll_reading, ll_greedy = ctc_log_likelihood(log_probs, [reading or " ", greedy or " "])
        ranked = rank_phrases(log_probs, reading if ll_reading >= ll_greedy else greedy, phrases)
    t3 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    return {
        "phrases": [{"text": s.text, "margin": round(s.margin, 4)} for s in ranked if math.isfinite(s.margin)],
        "frames": int(x.shape[1]),
        "latency_ms": {"load": ms(t0, t1), "crop": ms(t1, t2), "score": ms(t2, t3), "total": ms(t0, t3)},
    }


class CondomWord(BaseModel):
    text: str = Field(max_length=MAX_PHRASE_CHARS)
    confidence: float | None = None


class CondomPhrase(BaseModel):
    text: str = Field(max_length=MAX_PHRASE_CHARS)
    score: float | None = None


class CondomTurn(BaseModel):
    who: str = "other"
    text: str = Field(max_length=2000)


class CorrectIn(BaseModel):
    text: str = Field(max_length=2000)
    words: list[CondomWord] | None = Field(None, max_length=200)
    alternatives: list[str] = Field(default_factory=list, max_length=10)
    phrases: list[CondomPhrase] = Field(default_factory=list, max_length=20)
    context: list[CondomTurn] = Field(default_factory=list, max_length=20)
    mode: str = "normal"


@app.post("/correct", openapi_extra={"requestBody": {"required": True, "content": {
    "application/json": {"schema": CorrectIn.model_json_schema()}}}})
async def correct_text(request: Request) -> dict:
    """The Agentic Condom (`lipread.agentic_condom`): fix the words the reader was unsure of.

    JSON body (`CorrectIn`), read whatever the Content-Type: the browser sends it as text/plain so
    the call is a CORS simple request (no preflight round trip inside the 500 ms budget). Only
    `text` is required (the old `{"text"}` call still works; with no `words`, no word counts as
    unsure, so it comes back as is). Always 200 for a well-formed body; on any LLM failure `text`
    is the input and `status` says why. `edits` index the input's words.
    """
    try:
        body = CorrectIn.model_validate_json(await request.body())
    except ValidationError as e:
        raise HTTPException(422, {"error": "bad_body", "message": str(e.errors(include_url=False))[:500]})
    if body.mode not in ("normal", "quality"):
        raise HTTPException(422, {"error": "bad_mode", "message": "mode must be normal or quality"})
    req = CondomRequest(
        text=body.text,
        words=None if body.words is None else [(w.text, w.confidence) for w in body.words],
        alternatives=body.alternatives,
        phrases=[(p.text, p.score) for p in body.phrases],
        context=[("user" if t.who == "user" else "other", t.text) for t in body.context],
        mode=body.mode,
    )
    return (await run_in_threadpool(condom().correct, req)).as_dict()


# --- Opt-in training pairs (plan D59/D61): mouth crops + the text the user confirmed ---------------
PAIR_SOURCES = ("picked", "typed", "accepted")


def pairs_dir() -> Path:
    return Path(os.environ.get("LIPREAD_PAIRS_DIR", "data/training_pairs"))


def _upload_pair(files: list[Path], repo: str) -> None:
    """Best effort: append one pair to the HF dataset (token from HF_TOKEN / the server's login)."""
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        HfApi().create_commit(
            repo, repo_type="dataset", commit_message=f"training pair {files[0].stem}",
            operations=[CommitOperationAdd(f"pairs/{f.name}", str(f)) for f in files],
        )
    except Exception:  # noqa: BLE001  never lose the local copy over an upload problem
        log.exception("training pair upload to %s failed (kept locally)", repo)


@app.post("/training-pairs", openapi_extra=_CROPS_BODY_DOC)
def training_pair(
    background: BackgroundTasks,
    t: int = Query(..., description="number of frames at 25 fps (13-500 = 0.5-20 s)"),
    h: int = Query(96, description="frame height: 96 = aligned mouth patch, 88 = its centre crop"),
    w: int = Query(96, description="frame width, must equal h"),
    text: str = Query(..., min_length=1, max_length=300, description="the words actually said"),
    source: str = Query("picked", description="picked | typed | accepted"),
    content_encoding: str | None = Header(None),
    body: bytearray = Depends(_crops_body),
) -> dict:
    """Store one (mouth clip, confirmed text) pair for fine-tuning, sent only when the user opted in.

    Same body as /lipread/crops. Saved to $LIPREAD_PAIRS_DIR as <id>.npz (crops, uint8 t×h×w) +
    <id>.txt (uppercase text) + <id>.json; also pushed to the HF dataset $LIPREAD_PAIRS_REPO when set.
    """
    if source not in PAIR_SOURCES:
        raise HTTPException(422, {"error": "bad_source", "message": f"source must be one of {PAIR_SOURCES}"})
    words = " ".join(text.split()).upper()
    if not words:
        raise HTTPException(422, {"error": "empty_text"})
    crops = _decode_crops(t, h, w, content_encoding, body)
    pair_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out = pairs_dir()
    out.mkdir(parents=True, exist_ok=True)
    files = [out / f"{pair_id}.npz", out / f"{pair_id}.txt", out / f"{pair_id}.json"]
    np.savez_compressed(files[0], crops=crops)
    files[1].write_text(words + "\n")
    files[2].write_text(json.dumps({"id": pair_id, "frames": t, "size": h, "fps": MODEL_FPS,
                                    "source": source, "text": words}))
    repo = os.environ.get("LIPREAD_PAIRS_REPO")
    if repo:
        background.add_task(_upload_pair, files, repo)
    return {"id": pair_id, "frames": t, "text": words, "uploading_to": repo}
