"""Turns a stream of committed ASR words into speakable segments."""
import time
from dataclasses import dataclass
from typing import Callable

from streaming.profiles import SegmenterProfile

PUNCT = (".", "?", "!", ";", ":", ",")


@dataclass
class Segment:
    text: str
    is_turn_end: bool
    t_words_done: float
    n_words: int
    first_word_t: float = 0.0
    reason: str = ""  # "turn_end" | "pause" | "punctuation" | "max_words" | "max_chars"


class Segmenter:
    def __init__(self, profile: SegmenterProfile, clock: Callable[[], float] = time.perf_counter):
        self.p = profile
        self.clock = clock
        self._words: list[str] = []
        self._last_t = 0.0
        self._first_t = 0.0

    def _flush(self, reason: str) -> list[Segment]:
        if not self._words:
            return []
        seg = Segment(" ".join(self._words), reason == "turn_end", self._last_t,
                      len(self._words), self._first_t, reason)
        self._words = []
        return [seg]

    def push(self, word: str, committed: bool = True) -> list[Segment]:
        word = word.strip()
        if not committed or not word:
            return []
        self._last_t = self.clock()
        if not self._words:
            self._first_t = self._last_t
        self._words.append(word)
        n = len(self._words)
        if n < self.p.min_words:
            return []
        text = " ".join(self._words)
        if text.endswith(PUNCT) and n >= self.p.punct_min_words:
            return self._flush("punctuation")
        if n >= self.p.max_words:
            return self._flush("max_words")
        if len(text) >= self.p.max_chars:
            return self._flush("max_chars")
        return []

    def tick(self) -> list[Segment]:
        if self._words and (self.clock() - self._last_t) * 1000 >= self.p.pause_ms:
            return self._flush("pause")
        return []

    def end_turn(self) -> list[Segment]:
        return self._flush("turn_end")
