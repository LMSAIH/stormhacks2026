"""Single-run simulation of one backend with per-segment timings. Run: python -m tools.sim"""
from __future__ import annotations

import argparse
import asyncio
import os
import wave

from tts import config
from streaming.metrics import Recorder
from streaming.pipeline import Pipeline, run_script
from streaming.segmenter import Segmenter
from tts import create_backend

DEFAULT_TEXT = ("Hello there, I wanted to let you know that the meeting has moved to three o'clock. "
                "Please bring your laptop, and we will review the plan together.")


def save_wav(path: str, pcm: bytes, rate: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


async def run_once(name: str, words: list[str], wps: float, fmt: str, strict: bool = False,
                   segmenter_factory=None):
    backend = create_backend(name, config.load_tts_config(output_format=fmt))
    await backend.open()
    seg = segmenter_factory(backend) if segmenter_factory else Segmenter(backend.profile)
    pipe = Pipeline(backend, seg, Recorder())
    rec, pcm = await run_script(pipe, words, wps=wps, strict=strict)
    return backend, pipe, rec, pcm


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=config.backend_name())
    ap.add_argument("--wps", type=float, default=2.5)
    ap.add_argument("--format", default="pcm_24000")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    backend, pipe, rec, pcm = await run_once(a.backend, a.text.split(), a.wps, a.format, a.strict)
    s = rec.summarize()
    print(f"backend={a.backend}")
    for d in s["per_segment"]:
        f = lambda v: "-" if v is None else f"{v:.3f}s"
        print(f"seg {d['segment_id']:>2}  flush->audio {f(d['ttfa'])}  end-to-end {f(d['end_to_end'])}"
              f"  text={pipe.texts.get(d['segment_id'], '')!r}")
    if backend.codec == "pcm":
        path = f"output/sim_{a.backend}.wav"
        save_wav(path, pcm, backend.sample_rate)
        print(f"saved {path} ({len(pcm) / 2 / backend.sample_rate:.2f}s audio)")


if __name__ == "__main__":
    asyncio.run(main())
