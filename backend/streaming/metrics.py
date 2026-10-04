"""Latency/streaming metrics for TTS backend comparison."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass

from tts.base import AudioChunk, audio_seconds


@dataclass
class SegmentTiming:
    segment_id: int
    t_first_word: float | None = None
    t_words_done: float | None = None
    t_flush_sent: float | None = None
    t_first_audio: float | None = None
    t_last_audio: float | None = None
    audio_seconds: float = 0.0
    max_chunk_gap: float = 0.0
    n_chunks: int = 0
    n_words: int = 0
    text: str = ""
    reason: str = ""


def _sub(a, b):
    return None if a is None or b is None else a - b


class Recorder:
    def __init__(self) -> None:
        self._segs: dict[int, SegmentTiming] = {}

    def note_segment(self, segment_id: int, t_words_done: float, t_flush_sent: float, *,
                     t_first_word: float | None = None, n_words: int = 0, text: str = "",
                     reason: str = "") -> None:
        s = self._segs.setdefault(segment_id, SegmentTiming(segment_id))
        s.t_first_word = t_first_word
        s.n_words = n_words
        s.text = text
        s.reason = reason
        s.t_words_done = t_words_done
        s.t_flush_sent = t_flush_sent

    def note_chunk(self, chunk: AudioChunk) -> None:
        if chunk.is_final or not chunk.data or chunk.segment_id is None:
            return
        s = self._segs.setdefault(chunk.segment_id, SegmentTiming(chunk.segment_id))
        t = chunk.t_recv
        if s.t_first_audio is None:
            s.t_first_audio = t
        else:
            s.max_chunk_gap = max(s.max_chunk_gap, t - s.t_last_audio)
        s.t_last_audio = t
        s.audio_seconds += audio_seconds(chunk)
        s.n_chunks += 1

    def segments(self) -> list[SegmentTiming]:
        return [self._segs[k] for k in sorted(self._segs)]

    def summarize(self) -> dict:
        segs = self.segments()
        per: list[dict] = []
        play_end: float | None = None
        for s in segs:
            d = {
                "segment_id": s.segment_id,
                "ttfa": _sub(s.t_first_audio, s.t_flush_sent),
                "end_to_end": _sub(s.t_first_audio, s.t_words_done),
                "gen_speed": None,
                "max_chunk_gap": s.max_chunk_gap,
                "handoff_gap": None,
            }
            if s.n_chunks > 1:
                dur = s.t_last_audio - s.t_first_audio
                d["gen_speed"] = s.audio_seconds / dur if dur > 0 else math.inf
            if s.t_first_audio is not None:
                if play_end is None:
                    start = s.t_first_audio
                else:
                    d["handoff_gap"] = max(0.0, s.t_first_audio - play_end)
                    start = max(s.t_first_audio, play_end)
                play_end = start + s.audio_seconds
            per.append(d)

        def col(k):
            return [d[k] for d in per if d[k] is not None]

        def agg(k, fn):
            v = col(k)
            return fn(v) if v else None

        mean = statistics.fmean
        finite_speed = [v for v in col("gen_speed") if math.isfinite(v)]
        return {
            "n_segments": len(segs),
            "ttfa_first": per[0]["ttfa"] if per else None,
            "ttfa_mean": agg("ttfa", mean),
            "ttfa_max": agg("ttfa", max),
            "end_to_end_first": per[0]["end_to_end"] if per else None,
            "end_to_end_mean": agg("end_to_end", mean),
            "end_to_end_max": agg("end_to_end", max),
            "gen_speed_min": (min(col("gen_speed")) if col("gen_speed") else None),
            "gen_speed_mean": mean(finite_speed) if finite_speed else None,
            "max_chunk_gap": max((d["max_chunk_gap"] for d in per), default=None),
            "handoff_gap_max": agg("handoff_gap", max),
            "handoff_gap_total": agg("handoff_gap", sum),
            "per_segment": per,
        }


def _pct(vals: list[float], q: float) -> float:
    if len(vals) == 1:
        return vals[0]
    cuts = statistics.quantiles(vals, n=100, method="inclusive")
    return cuts[min(98, max(0, int(round(q)) - 1))]


def aggregate(runs: list[dict]) -> dict:
    """p50/p95/max per scalar metric across runs. -> {metric: {p50,p95,max}}"""
    keys: list[str] = []
    for r in runs:
        for k, v in r.items():
            if k not in keys and isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.append(k)
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            out[k] = {"p50": None, "p95": None, "max": None}
            continue
        if any(math.isinf(v) for v in vals):
            vs = sorted(vals)
            p50 = statistics.median(vs) if len(vs) % 2 else vs[len(vs) // 2]
            p95 = vs[min(len(vs) - 1, math.ceil(0.95 * len(vs)) - 1)]
            out[k] = {"p50": p50, "p95": p95, "max": vs[-1]}
        else:
            out[k] = {"p50": statistics.median(vals), "p95": _pct(vals, 95), "max": max(vals)}
    return out


# Plain-English names. (label, kind) where kind is "time" (seconds -> ms/s), "x" (speed) or "n".
LABELS = {
    "n_segments": ("Segments sent", "n"),
    "ttfa_first": ("TTS wait, first segment (send -> first audio)", "time"),
    "ttfa_mean": ("TTS wait, average (send -> first audio)", "time"),
    "ttfa_max": ("TTS wait, worst (send -> first audio)", "time"),
    "end_to_end_first": ("Total delay, first segment (last word -> first audio)", "time"),
    "end_to_end_mean": ("Total delay, average (last word -> first audio)", "time"),
    "end_to_end_max": ("Total delay, worst (last word -> first audio)", "time"),
    "gen_speed_min": ("Generation speed, slowest (x realtime)", "x"),
    "gen_speed_mean": ("Generation speed, average (x realtime)", "x"),
    "max_chunk_gap": ("Longest gap between audio chunks", "time"),
    "handoff_gap_max": ("Longest silence between segments", "time"),
    "handoff_gap_total": ("Total silence between segments", "time"),
}


def fmt_time(sec: float) -> str:
    """Round to the nearest 10 ms; show seconds once it reaches 1 s."""
    if sec >= 0.995:
        return f"{sec:.1f} s"
    return f"{int(round(sec * 100)) * 10} ms"


def _fmt(v, kind: str = "time") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        if kind == "time":
            return fmt_time(v)
        if kind == "x":
            return ">100x" if v > 100 else f"{v:.1f}x"
        return f"{v:.0f}"
    return str(v)


def label(key: str) -> str:
    return LABELS.get(key, (key, "n"))[0]


def kind(key: str) -> str:
    return LABELS.get(key, (key, "n"))[1]


def format_table(per_backend: dict[str, dict]) -> str:
    """per_backend: name -> aggregate() result. Rows=metric, cols=backend p50/p95/max."""
    names = list(per_backend)
    metrics: list[str] = []
    for n in names:
        for m in per_backend[n]:
            if m not in metrics:
                metrics.append(m)
    header = ["metric"]
    for n in names:
        header += [f"{n} typical", f"{n} bad-case", f"{n} worst"]
    rows = [header]
    for m in metrics:
        row = [label(m)]
        for n in names:
            st = per_backend[n].get(m) or {}
            k = kind(m)
            row += [_fmt(st.get("p50"), k), _fmt(st.get("p95"), k), _fmt(st.get("max"), k)]
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[0]) if j == 0 else c.rjust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def write_csv(path, rows: list[dict]) -> None:
    """Human-readable CSV: plain-English headers, times as '150 ms' / '1.2 s'."""
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    head = {"mode": "Mode", "backend": "Backend", "run": "Run"}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([head.get(k) or label(k) for k in fields])
        for r in rows:
            w.writerow(["" if r.get(k) is None else (_fmt(r[k], kind(k)) if isinstance(r[k], float) else r[k])
                        for k in fields])


def write_csv_rows(path, header: list[str], rows: list[list[str]]) -> None:
    """Plain header + pre-formatted string rows."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
