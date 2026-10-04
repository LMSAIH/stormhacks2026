"""Smoke-test client for the authenticated TTS WebSocket server.

    VOICE_SESSION_COOKIE=... .venv/bin/python tests/ws_client.py [--url ws://localhost:8765] [--wps 2.5]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
import wave

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--cookie", default=os.getenv("VOICE_SESSION_COOKIE"))
    ap.add_argument("--wps", type=float, default=2.5)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--file", default=os.path.join(ROOT, "tools", "text.txt"))
    ap.add_argument("--idle", type=float, default=2.0)
    a = ap.parse_args()
    if not a.cookie:
        ap.error("provide --cookie or set VOICE_SESSION_COOKIE")

    words = open(a.file).read().split()
    pcm = bytearray()
    t_first_send = None
    t_first_audio = None

    async with websockets.connect(
        a.url,
        max_size=None,
        additional_headers={"Cookie": f"voice_session={a.cookie}"},
    ) as ws:
        async def send():
            nonlocal t_first_send
            for w in words:
                await asyncio.sleep(1 / a.wps)
                if t_first_send is None:
                    t_first_send = time.perf_counter()
                await ws.send(w + " ")
            await ws.send("\\x")

        sender = asyncio.create_task(send())
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), a.idle if sender.done() else None)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    pcm += msg
                else:
                    print("server:", msg)
        finally:
            sender.cancel()

    if t_first_audio is None or t_first_send is None:
        raise SystemExit("no audio received")
    print(f"[{a.backend}] first send -> first audio: {t_first_audio - t_first_send:.3f}s")
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    path = os.path.join(ROOT, "output", "ws_client.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(a.sample_rate)
        w.writeframes(bytes(pcm))
    print(f"saved {path} ({len(pcm)} bytes, {len(pcm) / 2 / a.sample_rate:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
