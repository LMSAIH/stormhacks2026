"""Test client for the TTS WebSocket server.

  .venv/bin/python tests/ws_client.py --backend flash [--url ws://localhost:8000/ws/tts] [--wps 2.5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import wave

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="flash")
    ap.add_argument("--url", default="ws://localhost:8000/ws/tts")
    ap.add_argument("--wps", type=float, default=2.5, help="words/second. Default 2.5 SIMULATES a person speaking in real time, so 'time to send' includes speaking time; use 0 for instant (all words sent back-to-back, pure send->audio latency)")
    ap.add_argument("--file", default=os.path.join(ROOT, "tools", "text.txt"))
    ap.add_argument("--idle", type=float, default=2.0)
    a = ap.parse_args()

    words = open(a.file).read().split()
    url = f"{a.url}?backend={a.backend}"
    pcm = bytearray()
    t_first_send = None
    t_first_audio = None

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print("server:", ready)
        if ready.get("type") != "ready":
            raise SystemExit(f"unexpected first message: {ready}")

        async def send():
            nonlocal t_first_send
            for w in words:
                if a.wps > 0:
                    await asyncio.sleep(1 / a.wps)
                if t_first_send is None:
                    t_first_send = time.perf_counter()
                await ws.send(json.dumps({"type": "text", "text": w, "final": True}))
            await ws.send(json.dumps({"type": "end_turn"}))

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
        w.setnchannels(ready["channels"])
        w.setsampwidth(2)
        w.setframerate(ready["sample_rate"])
        w.writeframes(bytes(pcm))
    print(f"saved {path} ({len(pcm)} bytes, {len(pcm) / 2 / ready['sample_rate']:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
