"""Recognises the app's own TTS voice coming back through the microphone.

The TTS websocket tells the gate what it spoke and how much audio it streamed out. The STT side
asks whether a transcript is that echo. A transcript counts as echo when its text matches
something we recently said AND either it overlaps our playback window or the match is very
strong. Timing alone is never enough: real people may talk while the app is speaking.

No third-party dependencies, so it is safe to import when diarization is disabled.
"""

from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher

_WORDS = re.compile(r"[a-z0-9']+")


def _norm(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


class TtsGate:
    def __init__(self, *, lead_s: float = 0.2, tail_s: float = 0.4, history_s: float = 30.0,
                 match_ratio: float = 0.7, strong_ratio: float = 0.85, now=time.monotonic):
        self.lead_s = lead_s  # mic/network latency before playback shows up in the mic
        self.tail_s = tail_s  # room reverb + STT lag after playback ends
        self.history_s = history_s
        self.match_ratio = match_ratio
        self.strong_ratio = strong_ratio
        self._now = now
        self._windows: list[list[float]] = []  # [start, end] playback windows (monotonic seconds)
        self._texts: deque[tuple[float, str]] = deque()

    # --- called by the TTS side -------------------------------------------------------------
    def note_text(self, text: str) -> None:
        norm = _norm(text)
        if norm:
            self._texts.append((self._now(), norm))
        self._prune()

    def note_audio(self, duration_s: float) -> None:
        """Audio just streamed to the client; it plays back-to-back from now."""
        if duration_s <= 0:
            return
        now = self._now()
        if self._windows and now <= self._windows[-1][1]:
            self._windows[-1][1] += duration_s  # still playing: queue behind current audio
        else:
            self._windows.append([now, now + duration_s])
        self._prune()

    # --- called by the STT side -------------------------------------------------------------
    def overlaps(self, t0: float, t1: float) -> bool:
        return any(s - self.lead_s < t1 and e + self.tail_s > t0 for s, e in self._windows)

    def best_match(self, text: str) -> float:
        norm = _norm(text)
        if len(norm) < 3:
            return 0.0
        best = 0.0
        recent = [t for _, t in self._texts]
        for candidate in (*recent, " ".join(recent)):
            if not candidate:
                continue
            if norm in candidate:
                return 1.0
            best = max(best, SequenceMatcher(None, norm, candidate).ratio())
        return best

    def is_echo(self, text: str, t0: float | None = None, t1: float | None = None) -> bool:
        """t0/t1 are monotonic wall times the transcript covers (default: just now)."""
        self._prune()
        ratio = self.best_match(text)
        if ratio >= self.strong_ratio:
            return True
        if ratio < self.match_ratio:
            return False
        now = self._now()
        return self.overlaps(now - 1.0 if t0 is None else t0, now if t1 is None else t1)

    def _prune(self) -> None:
        cutoff = self._now() - self.history_s
        while self._texts and self._texts[0][0] < cutoff:
            self._texts.popleft()
        self._windows = [w for w in self._windows if w[1] >= cutoff]


_gates: dict[str, TtsGate] = {}


def get_tts_gate(key: str) -> TtsGate:
    """One gate per signed-in user, shared by that user's TTS and STT websockets."""
    gate = _gates.get(key)
    if gate is None:
        gate = _gates[key] = TtsGate()
    return gate
