"""Benchmark segmented/sentence/words modes. Run: python -m tools.bench [--runs N] [--mode M]"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

from tts import config
from streaming.metrics import aggregate, fmt_time, format_table, write_csv
from streaming.profiles import SegmenterProfile
from streaming.segmenter import Segmenter
from tools.sim import run_once, save_wav

DEFAULT_TEXT = ("Hello there, I wanted to let you know that the meeting moved to three o'clock. "
                "Please bring your laptop, and we will review the plan together.")
BIG = 10**6


def make_segmenter(mode: str, backend) -> Segmenter:
    if mode == "segmented":
        return Segmenter(backend.profile)
    if mode == "words":
        return Segmenter(SegmenterProfile(min_words=1, max_words=1, max_chars=BIG,
                                          pause_ms=BIG, punct_min_words=1))
    if mode == "sentence":  # flushes only at end_turn
        return Segmenter(SegmenterProfile(min_words=BIG, max_words=BIG, max_chars=BIG,
                                          pause_ms=BIG, punct_min_words=BIG))
    raise ValueError(mode)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="flash,v4")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--mode", default="segmented")
    ap.add_argument("--script", default=None)
    ap.add_argument("--format", default="pcm_24000")
    ap.add_argument("--wps", type=float, default=2.5)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    backends = a.backend.split(",")
    modes = a.mode.split(",")
    text = open(a.script).read() if a.script else DEFAULT_TEXT
    words = text.split()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows: list[dict] = []
    for mode in modes:
        runs: dict[str, list[dict]] = {b: [] for b in backends}
        saved: set[str] = set()
        for i in range(a.runs):
            for b in (backends if i % 2 == 0 else backends[::-1]):
                try:
                    be, pipe, rec, pcm = await run_once(
                        b, words, a.wps, a.format, a.strict,
                        segmenter_factory=lambda be, m=mode: make_segmenter(m, be))
                except Exception as e:  # keep the bench going
                    print(f"[{mode}] {b} run {i + 1} FAILED: {e!r}")
                    continue
                s = rec.summarize()
                s.pop("per_segment")
                runs[b].append(s)
                rows.append({"mode": mode, "backend": b, "run": i + 1, **s})
                print(f"[{mode}] {b} run {i + 1}: segs={s['n_segments']} "
                      f"TTS wait avg={fmt_time(s['ttfa_mean'])}  total delay avg={fmt_time(s['end_to_end_mean'])}")
                if be.codec == "pcm" and b not in saved and pcm:
                    save_wav(f"output/{b}_{mode}.wav", pcm, be.sample_rate)
                    saved.add(b)
        agg = {b: aggregate(r) for b, r in runs.items() if r}
        if agg:
            print(f"\n=== mode: {mode} ({a.runs} runs) ===")
            print(format_table(agg))
            print("  TTS wait = we send text -> first audio arrives (ElevenLabs + network).")
            print("  Total delay = listener's wait after the last word of a segment was spoken.")
            print("  Silence between segments = gap a listener hears if audio arrives late.\n")
    os.makedirs("bench_results", exist_ok=True)
    path = f"bench_results/{stamp}.csv"
    if rows:
        write_csv(path, rows)
        print(f"csv: {path}")


if __name__ == "__main__":
    asyncio.run(main())
