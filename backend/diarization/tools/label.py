"""Offline tuning tool: label who speaks when in a WAV file, with no websocket involved.

    .venv/bin/python -m diarization.tools.label meeting.wav
    .venv/bin/python -m diarization.tools.label meeting.wav --threshold 0.7 --vad silero

Prints the speaker timeline plus how long each voiceprint took (the number that decides whether
this keeps up in real time on your machine). WAV, mp3, m4a, ... all work; mixed to mono and resampled.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

from diarization.config import DiarizationConfig
from diarization.tracker import SpeakerTracker

CHUNK_S = 0.1  # the size the browser will stream in


def load_wav_16k_mono(path: str, target_sr: int) -> bytes:
    from diarization.audio import AudioDecodeError, decode_to_pcm16

    with open(path, "rb") as f:
        data = f.read()
    try:
        return decode_to_pcm16(data, target_sr, os.path.splitext(path)[1])
    except AudioDecodeError as e:
        raise SystemExit(f"{path}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    base = DiarizationConfig.from_env()
    for f in dataclasses.fields(DiarizationConfig):
        if f.name in ("sample_rate", "frame_samples", "max_segments"):
            continue
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=getattr(base, f.name))
    args = ap.parse_args()

    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(DiarizationConfig)
              if hasattr(args, f.name)}
    cfg = dataclasses.replace(base, **values)

    pcm = load_wav_16k_mono(args.wav, cfg.sample_rate)
    seconds = len(pcm) / 2 / cfg.sample_rate

    from diarization.embed import make_embedder
    from diarization.vad import make_vad

    inner = make_embedder(cfg)
    timings: list[float] = []

    def timed_embedder(wav):
        t0 = time.perf_counter()
        out = inner(wav)
        timings.append(time.perf_counter() - t0)
        return out

    tracker = SpeakerTracker(cfg, vad=make_vad(cfg), embedder=timed_embedder)
    step = int(CHUNK_S * cfg.sample_rate) * 2
    t0 = time.perf_counter()
    for i in range(0, len(pcm), step):
        tracker.feed(pcm[i:i + step])
    wall = time.perf_counter() - t0

    print(f"{args.wav}: {seconds:.1f}s audio, {tracker.speaker_count} speaker(s)\n")
    print(f"{'start':>7} {'end':>7}  speaker")
    for seg in tracker.segments:
        mark = "  (inherited)" if seg.provisional else ""
        print(f"{seg.start:7.2f} {seg.end:7.2f}  {seg.speaker}{mark}")
    if timings:
        ms = sorted(t * 1000 for t in timings)
        print(f"\nvoiceprints: {len(ms)}, median {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms")
    print(f"processed {seconds:.1f}s of audio in {wall:.2f}s ({seconds / wall:.1f}x real time)")


if __name__ == "__main__":
    main()
