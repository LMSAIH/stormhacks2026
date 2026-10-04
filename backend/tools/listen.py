"""Edit tools/text.txt, run this, and listen to the result.

  .venv/bin/python -m tools.listen                         # backend from TTS_BACKEND (.env), default flash
  .venv/bin/python -m tools.listen --backend v4
  .venv/bin/python -m tools.listen --backend flash,v4      # plays one after the other
  .venv/bin/python -m tools.listen --mode sentence|segmented|words
  .venv/bin/python -m tools.listen --text "Say this instead"
  .venv/bin/python -m tools.listen --no-play               # just save output/listen_<backend>_<mode>.wav
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess

from app import config
from tools.bench import make_segmenter
from streaming.metrics import fmt_time
from tools.sim import run_once, save_wav


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=config.backend_name(), help="flash, v4, or flash,v4")
    ap.add_argument("--mode", default="segmented", help="segmented | sentence | words")
    ap.add_argument("--file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "text.txt"))
    ap.add_argument("--text", default=None, help="overrides --file")
    ap.add_argument("--wps", type=float, default=2.5, help="words/second. Default 2.5 SIMULATES a person speaking in real time, so 'time to send' includes speaking time; use 0 for instant (all words sent back-to-back, pure send->audio latency)")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args()

    text = a.text if a.text is not None else open(a.file).read()
    words = text.split()
    print(f"{len(words)} words, mode={a.mode}, wps={a.wps}")

    for name in a.backend.split(","):
        backend, pipe, rec, pcm = await run_once(
            name, words, a.wps, "pcm_24000",
            segmenter_factory=lambda b: make_segmenter(a.mode, b))
        s = rec.summarize()
        t = lambda v: "n/a" if v is None else fmt_time(v)
        print(f"\n[{name}] segments={s['n_segments']}  wait for first audio (avg)={t(s['ttfa_mean'])}  "
              f"longest silence between segments={t(s['handoff_gap_max'])}")
        for d in s["per_segment"]:
            print(f"  seg {d['segment_id']:>2} {pipe.texts.get(d['segment_id'], '')!r}")
        path = f"output/listen_{name}_{a.mode}.wav"
        save_wav(path, pcm, backend.sample_rate)
        print(f"  saved {path}")
        if not a.no_play:
            subprocess.run(["afplay", path])  # macOS


if __name__ == "__main__":
    asyncio.run(main())
