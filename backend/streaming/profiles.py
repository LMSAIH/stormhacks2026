"""Dependency-free tuning profiles shared by the segmenter and the TTS backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmenterProfile:
    """Thresholds that decide when buffered ASR words get flushed to TTS."""

    min_words: int = 3  # never flush fewer words than this unless the turn ended
    max_words: int = 10  # force a flush at a word boundary beyond this
    max_chars: int = 60  # same, by characters
    # Silence after the last committed word that triggers a flush. Must exceed the normal
    # gap between words (~400ms at 2.5 words/s) or every word becomes its own segment.
    pause_ms: int = 700
    punct_min_words: int = 3  # flush on , . ? ! ; : once this many words are buffered
