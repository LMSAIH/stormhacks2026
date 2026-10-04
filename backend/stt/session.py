"""One listening session: client PCM in -> Scribe Realtime -> caption events out.

Optional collaborators, both duck-typed so this module never imports the diarization package:
  diarizer  feed(pcm), current_speaker(), wait_for(t), split(words) -> [pieces with .speaker/.text]
  gate      is_echo(text, t0, t1) -> bool   (drops the app's own TTS voice picked up by the mic)

Events sent to the client (JSON):
  {"type": "ready"}
  {"type": "utterance", "id", "text", "final", "speaker"?}   speaker is null until identified;
                                                              the key is absent when diarization is off
  {"type": "drop", "id"}                                      a shown partial turned out to be echo
  {"type": "error", "message"}
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

ERROR_TYPES = {
    "error", "auth_error", "quota_exceeded", "transcriber_error", "input_error", "invalid_request",
    "commit_throttled", "unaccepted_terms", "rate_limited", "queue_overflow", "resource_exhausted",
    "session_time_limit_exceeded", "chunk_size_exceeded", "insufficient_audio_activity",
}


class SttSession:
    def __init__(self, scribe, send: Callable[[dict], Awaitable[None]], *, diarizer=None, gate=None,
                 clock=time.monotonic):
        self.scribe = scribe
        self.send = send
        self.diarizer = diarizer
        self.gate = gate
        self._clock = clock
        self._t0: float | None = None  # wall time of the first audio sample
        self._n = 0  # committed utterance counter, used for stable ids

    # --- client -> Scribe ---------------------------------------------------------------------
    async def on_audio(self, pcm: bytes) -> None:
        if self._t0 is None:
            self._t0 = self._clock()
        await self.scribe.send_audio(pcm)
        if self.diarizer is not None:
            self.diarizer.feed(pcm)

    # --- Scribe -> client ---------------------------------------------------------------------
    async def on_event(self, msg: dict) -> None:
        kind = msg.get("message_type")
        if kind == "session_started":
            await self.send({"type": "ready"})
        elif kind == "partial_transcript":
            await self._partial((msg.get("text") or "").strip())
        elif kind == "committed_transcript":
            if self.diarizer is None:  # with a diarizer we wait for the timestamped twin
                await self._final_plain((msg.get("text") or "").strip())
        elif kind == "committed_transcript_with_timestamps":
            if self.diarizer is not None:
                await self._final_words(msg)
        elif kind in ERROR_TYPES or "error" in msg:
            await self.send({"type": "error", "message": str(msg.get("error") or kind)})

    def _uid(self, k: int = 0) -> str:
        return f"u{self._n}-{k}"

    def _wall(self, t: float) -> float:
        return (self._t0 if self._t0 is not None else self._clock()) + t

    def _utterance(self, uid: str, text: str, final: bool, speaker) -> dict:
        event = {"type": "utterance", "id": uid, "text": text, "final": final}
        if self.diarizer is not None:
            event["speaker"] = speaker
        return event

    async def _partial(self, text: str) -> None:
        if not text:  # keepalive
            return
        if self.gate is not None and self.gate.is_echo(text):
            return
        speaker = self.diarizer.current_speaker() if self.diarizer is not None else None
        await self.send(self._utterance(self._uid(), text, False, speaker))

    async def _final_plain(self, text: str) -> None:
        if not text:
            return
        uid = self._uid()
        self._n += 1
        if self.gate is not None and self.gate.is_echo(text):
            await self.send({"type": "drop", "id": uid})
            return
        speaker = self.diarizer.current_speaker() if self.diarizer is not None else None
        await self.send(self._utterance(uid, text, True, speaker))

    async def _final_words(self, msg: dict) -> None:
        words = msg.get("words") or []
        text = (msg.get("text") or "").strip()
        if not words:
            await self._final_plain(text)
            return
        starts = [float(w.get("start") or 0.0) for w in words]
        ends = [float(w.get("end") or 0.0) for w in words]
        await self.diarizer.wait_for(max(ends))  # tracker may still be catching up on the tail

        if self.gate is not None and self.gate.is_echo(text, self._wall(min(starts)), self._wall(max(ends))):
            uid = self._uid()
            self._n += 1
            await self.send({"type": "drop", "id": uid})
            return

        pieces = self.diarizer.split(words) or []
        base = self._n
        self._n += 1
        if not pieces:
            await self.send(self._utterance(f"u{base}-0", text, True, self.diarizer.current_speaker()))
            return
        for k, piece in enumerate(pieces):
            await self.send(self._utterance(f"u{base}-{k}", piece.text, True, piece.speaker))
