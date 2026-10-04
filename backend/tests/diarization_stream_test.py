"""Stream an audio file to the backend the way the browser would and write the speaker labels to a txt.

    # Diarization only (no ElevenLabs needed): server running with DIARIZATION=1
    .venv/bin/python tests/diarization_stream_test.py meeting.mp3

    # Full live path: captions from ElevenLabs Scribe + speaker labels (needs ELEVENLABS_API_KEY
    # and SESSION_SECRET in backend/.env, same as the running server)
    .venv/bin/python tests/diarization_stream_test.py meeting.mp3 --mode stt

Audio is decoded to 16 kHz mono PCM and sent as 100 ms binary frames paced in real time
(--speed 0 sends as fast as possible). Output: tests/diarization_output/<name>-<mode>.txt, written line by line as events arrive
"""

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import InvalidStatus  # noqa: E402

from config import API_PORT  # noqa: E402  (also loads backend/.env)
from diarization.audio import decode_to_pcm16  # noqa: E402

SR = 16000
OUT_DIR = os.path.join(os.path.dirname(__file__), "diarization_output")


def session_cookie() -> str:
    """Sign a session cookie the same way the server does (needs the same SESSION_SECRET)."""
    from itsdangerous import TimestampSigner

    from config import SESSION_SECRET

    payload = base64.b64encode(json.dumps({"user": {"id": "stream-test", "email": "test@example.com"}}).encode())
    return "voice_session=" + TimestampSigner(str(SESSION_SECRET), salt="starlette.sessions").sign(payload).decode()


def lanes(segments: list[dict], duration: float, step: float = 0.5) -> list[str]:
    """One text row per speaker; '#' where that speaker is active (each column = `step` seconds)."""
    cols = max(1, math.ceil(duration / step))
    speakers = sorted({s["speaker"] for s in segments})
    rows = []
    for sp in speakers:
        row = ["."] * cols
        for s in segments:
            if s["speaker"] != sp:
                continue
            for c in range(int(s["start"] / step), min(cols, int(math.ceil(s["end"] / step)))):
                row[c] = "#"
        rows.append(f"{sp:>10} |{''.join(row)}|")
    label_every = max(1, int(round(10 / step)))  # a tick label roughly every 10 s
    marks = "".join(str(int(c * step // 10) % 10) if c % label_every == 0 else " " for c in range(cols))
    rows.append(f"{'seconds':>10} |{marks}|")
    return rows


class Report:
    """Writes every line to the txt file *and* the terminal the moment it happens."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "w", buffering=1)  # line buffered: `tail -f` the file while it runs

    def line(self, text: str = "") -> None:
        self.f.write(text + "\n")
        self.f.flush()
        print(text, flush=True)

    def close(self) -> None:
        self.f.close()


def fmt_event(wall: float, m: dict, mode: str) -> str | None:
    if m["type"] == "segment":
        tag = " (inherited)" if m["provisional"] else ""
        return (f"[wall {wall:6.2f}s | audio {m['audio_time']:6.2f}s] {m['speaker']:<10} "
                f"seg {m['id']:>2}  {m['start']:6.2f} -> {m['end']:6.2f}{tag}")
    if m["type"] == "utterance":
        kind = "FINAL  " if m["final"] else "partial"
        return f"[wall {wall:6.2f}s] {kind} {m['id']:<7} {str(m.get('speaker')):<10} {m['text']}"
    if m["type"] in ("drop", "ready"):
        return f"[wall {wall:6.2f}s] {m['type']} {m.get('id', '')}".rstrip()
    if m["type"] == "error":
        return f"[wall {wall:6.2f}s] !! server error: {m.get('message')}"
    return None


async def run(args, report: Report) -> None:
    with open(args.audio, "rb") as f:
        data = f.read()
    pcm = decode_to_pcm16(data, SR, os.path.splitext(args.audio)[1])
    pcm += b"\x00\x00" * int(SR * args.tail_s)  # trailing silence so the last utterance can close
    duration = len(pcm) / 2 / SR
    frame_bytes = int(SR * args.chunk_ms / 1000) * 2

    path = "/ws/diarize" if args.mode == "diarize" else "/ws/stt"
    url = args.url.rstrip("/") + path
    headers = {"Cookie": session_cookie()}  # both sockets require sign-in

    report.line("Diarization stream test")
    report.line(f"file     : {args.audio}")
    report.line(f"mode     : {args.mode}  ({url})")
    report.line(f"audio    : {duration - args.tail_s:.1f}s (+{args.tail_s:.1f}s trailing silence)")
    report.line(f"streaming: {args.chunk_ms} ms frames, speed x{args.speed or 'max'}")
    report.line()
    if args.mode == "diarize":
        report.line("== Live events (wall = seconds since streaming started; audio = audio the server had processed)")
    else:
        report.line("== Live captions (partial = still changing, FINAL = settled)")

    finals: list[dict] = []
    segments_done: list[dict] = []
    t0 = time.monotonic()
    done = asyncio.Event()

    async def reader(ws):
        try:
            async for raw in ws:
                m = json.loads(raw)
                line = fmt_event(time.monotonic() - t0, m, args.mode)
                if line:
                    report.line(line)
                if m["type"] == "utterance" and m["final"]:
                    finals.append(m)
                if m["type"] == "done":
                    segments_done.extend(m["segments"])
                    done.set()
                    return
        finally:
            done.set()  # connection closed

    try:
        ws_cm = connect(url, additional_headers=headers, max_size=None)
        ws = await ws_cm.__aenter__()
    except InvalidStatus as e:
        code = e.response.status_code
        hint = (" Something else is on that port (macOS AirPlay uses 5000): set API_PORT in .env or pass --url."
                if code == 403 else "")
        raise SystemExit(f"{url}: rejected (HTTP {code}). Is the server running?{hint}")
    except OSError as e:
        raise SystemExit(f"cannot reach {url}: {e}. Is the server running (python main.py)?")

    try:
        read_task = asyncio.create_task(reader(ws))
        t0 = time.monotonic()
        for n, i in enumerate(range(0, len(pcm), frame_bytes)):
            if args.speed > 0:
                delay = t0 + (n * args.chunk_ms / 1000) / args.speed - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(pcm[i:i + frame_bytes])
            if done.is_set():
                break
        if args.mode == "diarize":
            await ws.send("end")
            try:
                await asyncio.wait_for(done.wait(), timeout=60)
            except asyncio.TimeoutError:
                report.line("!! timed out waiting for the server's 'done' message")
        else:
            report.line(f"-- audio finished, waiting {args.settle_s:.0f}s for the last captions --")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=args.settle_s)
        read_task.cancel()
    finally:
        await ws_cm.__aexit__(None, None, None)

    if args.mode == "diarize":
        report.line()
        report.line(f"== Final timeline: {len(set(s['speaker'] for s in segments_done))} speaker(s)")
        report.line(f"{'start':>7} {'end':>7}  speaker")
        for s in segments_done:
            report.line(f"{s['start']:7.2f} {s['end']:7.2f}  {s['speaker']}" + ("  (inherited)" if s["provisional"] else ""))
        report.line()
        step = max(0.5, math.ceil(duration / 120 * 2) / 2)  # keep the chart <= ~120 columns
        report.line(f"== Lanes (each column = {step:g} s)")
        for row in lanes(segments_done, duration, step):
            report.line(row)
    else:
        report.line()
        report.line("== Transcript")
        for m in finals:
            report.line(f"{m.get('speaker') or '?'}: {m['text']}")
        if not finals:
            report.line("(no final captions received; check ELEVENLABS_API_KEY and the server log)")
        if finals and not any("speaker" in m for m in finals):
            report.line("(no speaker labels: start the server with DIARIZATION=1)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="mp3/wav/m4a file")
    ap.add_argument("--mode", choices=["diarize", "stt"], default="diarize")
    ap.add_argument("--url", default=f"ws://localhost:{API_PORT}", help="backend API base (default: API_PORT from .env)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 4 = 4x faster, 0 = as fast as possible")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--tail-s", type=float, default=2.0, help="seconds of silence appended after the file")
    ap.add_argument("--settle-s", type=float, default=4.0, help="stt mode: wait this long after the last frame")
    ap.add_argument("--out", help="output txt (default tests/diarization_output/<name>-<mode>.txt)")
    args = ap.parse_args()

    out = args.out or os.path.join(
        OUT_DIR, f"{os.path.splitext(os.path.basename(args.audio))[0]}-{args.mode}.txt")
    report = Report(out)
    print(f"writing live to {out}  (tail -f it in another window)\n", flush=True)
    try:
        asyncio.run(run(args, report))
    finally:
        report.close()
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
