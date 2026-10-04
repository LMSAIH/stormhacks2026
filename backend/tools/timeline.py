"""Timeline of the request loop: when words are spoken, segments sent, audio returns.

  .venv/bin/python -m tools.timeline [--backend flash,v4] [--mode segmented|sentence|words]
                                     [--file tools/text.txt | --text "..."] [--wps 2.5] [--events]

The connection is already open before t=0; t=0 is the moment the first word is fed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

from tts import config
from streaming.metrics import Recorder, audio_seconds, fmt_time, write_csv_rows
from streaming.pipeline import Pipeline, run_script
from tools.bench import make_segmenter
from tts import create_backend

LEGEND = """Legend: 'held' = last word spoken -> segment sent (waiting in our segmenter); 'TTS wait' = sent -> first audio back.
Bar: '.' speaking  '-' held in segmenter  '~' waiting for TTS  '#' audio arriving (blank = idle). 'why sent' = reason the segmenter flushed.
'gap since prev send' = this send minus previous send (small = sent back-to-back, large = delayed by waiting for words/pause/punctuation)."""

HEAD = ["#", "words", "text", "why sent", "first word at", "last word at", "SENT at",
        "held in segmenter", "first audio at", "TTS wait", "audio chunks", "audio length",
        "audio fully received at", "gap since previous send"]


async def run_backend(name: str, words: list[str], a):
    overrides = {"model_id": a.model} if a.model else {}
    backend = create_backend(name, config.load_tts_config(output_format="pcm_24000", **overrides))
    await backend.open()  # before t=0, not reported
    events: list[tuple[float, str]] = []
    state = {"cum": 0.0}

    async def on_audio(chunk):
        state["cum"] += audio_seconds(chunk)
        events.append((chunk.t_recv, f"audio chunk  seg {chunk.segment_id}  {len(chunk.data)} bytes  "
                                     f"total audio {state['cum']:.2f} s"))

    def on_segment(sid, seg):
        events.append((backend.flush_times[sid],
                       f"SEGMENT {sid} SENT ({seg.reason}, {seg.n_words} words) {seg.text!r}"))

    pipe = Pipeline(backend, make_segmenter(a.mode, backend), Recorder(),
                    on_audio=on_audio, on_segment=on_segment)
    orig = pipe.feed_word

    async def feed(word):
        events.append((time.perf_counter(), f"word fed  {word!r}"))
        await orig(word)

    pipe.feed_word = feed
    rec, _ = await run_script(pipe, words, wps=a.wps)
    return rec.segments(), events


def build_rows(segs):
    """Per-segment dicts of raw relative seconds (None when unknown)."""
    t0 = segs[0].t_first_word
    rel = lambda v: None if v is None else v - t0
    rows, prev = [], None
    for s in segs:
        sent, first, last = rel(s.t_flush_sent), rel(s.t_first_audio), rel(s.t_last_audio)
        rows.append({
            "id": s.segment_id, "words": s.n_words, "text": s.text, "reason": s.reason,
            "first_word": rel(s.t_first_word), "last_word": rel(s.t_words_done), "sent": sent,
            "held": sent - rel(s.t_words_done),
            "first_audio": first, "tts_wait": None if first is None else first - sent,
            "chunks": s.n_chunks, "audio_s": s.audio_seconds, "audio_done": last,
            "gap": None if prev is None else sent - prev,
        })
        prev = sent
    return rows


def cells(r):
    f = lambda v: "-" if v is None else fmt_time(v)
    t = r["text"] if len(r["text"]) <= 40 else r["text"][:39] + "…"
    return [str(r["id"]), str(r["words"]), t, r["reason"], f(r["first_word"]), f(r["last_word"]),
            f(r["sent"]), f(r["held"]), f(r["first_audio"]), f(r["tts_wait"]), str(r["chunks"]),
            f(r["audio_s"]) if r["chunks"] else "-", f(r["audio_done"]), f(r["gap"])]


def table(rows) -> str:
    data = [HEAD] + [cells(r) for r in rows]
    w = [max(len(d[i]) for d in data) for i in range(len(HEAD))]
    out = []
    for i, d in enumerate(data):
        out.append("  ".join(c.ljust(w[j]) if j in (2, 3) else c.rjust(w[j]) for j, c in enumerate(d)))
        if i == 0:
            out.append("  ".join("-" * x for x in w))
    return "\n".join(out)


def bars(rows, cols: int = 50) -> str:
    total = max((r["audio_done"] or r["sent"]) for r in rows) or 1.0
    px = lambda v: min(cols - 1, int(v / total * cols))
    out = [f"0{' ' * (cols - 2 - len(fmt_time(total)))}{fmt_time(total)}"]
    for r in rows:
        line = [" "] * cols
        marks = [(r["first_word"], r["last_word"], "."), (r["last_word"], r["sent"], "-"),
                 (r["sent"], r["first_audio"], "~"), (r["first_audio"], r["audio_done"], "#")]
        for a, b, ch in marks:
            if a is None or b is None:
                continue
            for i in range(px(a), max(px(a) + 1, px(b)) if ch != "-" or b > a else px(a)):
                line[i] = ch
        out.append(f"seg {r['id']:>2} |{''.join(line)}|")
    return "\n".join(out)


def summary(name, rows) -> str:
    avg = lambda k: sum(r[k] for r in rows if r[k] is not None) / max(1, sum(r[k] is not None for r in rows))
    tot = [r["first_audio"] - r["last_word"] for r in rows if r["first_audio"] is not None]
    return (f"[{name}] segments={len(rows)}  avg TTS wait={fmt_time(avg('tts_wait'))}  "
            f"avg held in segmenter={fmt_time(avg('held'))}  "
            f"avg total (last word -> first audio)={fmt_time(sum(tot) / len(tot)) if tot else '-'}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="flash,v4")
    ap.add_argument("--mode", default="segmented", help="segmented | sentence | words")
    ap.add_argument("--file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "text.txt"))
    ap.add_argument("--text", default=None, help="overrides --file")
    ap.add_argument("--wps", type=float, default=2.5, help="words/second. Default 2.5 SIMULATES a person speaking in real time, so 'time to send' includes speaking time; use 0 for instant (all words sent back-to-back, pure send->audio latency)")
    ap.add_argument("--model", default=None, help="override TTS_MODEL_ID")
    ap.add_argument("--verbose", action="store_true", help="print tables, bars, legend")
    ap.add_argument("--events", action="store_true", help="print event log (implies --verbose)")
    a = ap.parse_args()
    if a.events:
        a.verbose = True
    words = (a.text if a.text is not None else open(a.file).read()).split()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_rows, summaries = [], []
    for name in a.backend.split(","):
        segs, events = await run_backend(name, words, a)
        if not segs:
            print(f"[{name}] WARNING: no segments")
            continue
        rows = build_rows(segs)
        t0 = segs[0].t_first_word
        if a.verbose:
          print(f"\n=== {name}  mode={a.mode}  {len(words)} words @ {a.wps} wps  (t=0 = first word fed) ===")
          print(table(rows))
          print("\n" + bars(rows))
        if a.events:
            print("\nEvents:")
            for t, msg in sorted(events):
                print(f"  {fmt_time(max(0.0, t - t0)):>9}  {msg}")
        summaries.append(summary(name, rows))
        csv_rows += [[name] + cells(r) for r in rows]
    if a.verbose:
        print("\n" + LEGEND + "\n")
        print("\n".join(summaries))
    if csv_rows:
        os.makedirs("bench_results", exist_ok=True)
        path = f"bench_results/timeline_{stamp}.csv"
        write_csv_rows(path, ["Backend"] + HEAD, csv_rows)
        print(f"saved: {path}")


if __name__ == "__main__":
    asyncio.run(main())
