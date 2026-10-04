"""Attach speaker labels to Scribe's word-level timestamps. Pure functions, no dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Piece:
    speaker: Optional[str]
    text: str


def split_by_speaker(words: list[dict], speaker_at: Callable[[float], Optional[str]],
                     offset: float = 0.0) -> list[Piece]:
    """Split one committed transcript into per-speaker pieces.

    `words` are Scribe tokens ({"text", "start", "end", "type": "word"|"spacing"|"audio_event"}).
    `speaker_at(t)` maps session seconds to a label; `offset` shifts Scribe's clock onto ours.
    """
    tokens = [(w.get("text") or "", w.get("type", "word"), w) for w in words]
    word_idx = [i for i, (_, typ, _) in enumerate(tokens) if typ != "spacing"]
    if not word_idx:
        return []

    labels: dict[int, Optional[str]] = {}
    for i in word_idx:
        w = tokens[i][2]
        start = float(w.get("start") or 0.0)
        end = float(w.get("end") or start)
        labels[i] = speaker_at((start + end) / 2 + offset)

    # Words nobody claimed take the neighbouring speaker (forward fill, then back fill).
    last = None
    for i in word_idx:
        labels[i] = labels[i] or last
        last = labels[i]
    nxt = None
    for i in reversed(word_idx):
        labels[i] = labels[i] or nxt
        nxt = labels[i]

    # A single word sandwiched between two identical labels is a boundary glitch, not a speaker.
    for a, b, c in zip(word_idx, word_idx[1:], word_idx[2:]):
        if labels[a] == labels[c] and labels[b] != labels[a]:
            labels[b] = labels[a]

    joiner = "" if any(typ == "spacing" for _, typ, _ in tokens) else " "
    pieces: list[Piece] = []
    current: Optional[str] = labels[word_idx[0]]
    for i, (text, typ, _) in enumerate(tokens):
        if typ != "spacing":
            current = labels[i]
        if not pieces or (typ != "spacing" and pieces[-1].speaker != current):
            pieces.append(Piece(current, ""))
        pieces[-1].text += text + (joiner if typ != "spacing" else "")

    out = []
    for p in pieces:
        p.text = p.text.strip()
        if p.text:
            out.append(p)
    return out
