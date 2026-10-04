"""Online speaker tracking: VAD -> voiceprint -> match against known speakers.

Feed it the raw session audio (16-bit mono PCM). It keeps its own sample clock, so every
segment time is "seconds since the first sample fed", the same clock Scribe's word timestamps use
as long as the client streams continuously.

The tracker is synchronous and CPU-bound (embedding a window takes tens of milliseconds); call
`feed` from a worker thread when running inside an event loop.
"""

from __future__ import annotations

import numpy as np

from diarization.config import DiarizationConfig
from diarization.types import SpeakerSegment


class _Centroid:
    __slots__ = ("vec", "count")

    def __init__(self, vec: np.ndarray):
        self.vec = vec
        self.count = 1


class SpeakerTracker:
    def __init__(self, config: DiarizationConfig | None = None, vad=None, embedder=None):
        self.cfg = config or DiarizationConfig.from_env()
        if vad is None:
            from diarization.vad import make_vad

            vad = make_vad(self.cfg)
        if embedder is None:
            from diarization.embed import make_embedder

            embedder = make_embedder(self.cfg)
        self._vad = vad
        self._embed = embedder

        self._sr = self.cfg.sample_rate
        self._frame = self.cfg.frame_samples
        self._pending = b""
        self._n = 0  # samples consumed (frame aligned)

        self._centroids: list[_Centroid] = []
        self.segments: list[SpeakerSegment] = []
        self._next_id = 0
        self._last_speaker: str | None = None

        self._in_speech = False
        self._silence_frames = 0
        self._utt: list[np.ndarray] = []
        self._utt_start = 0.0
        self._speech_samples = 0
        self._last_embed_speech = 0
        self._cur: SpeakerSegment | None = None
        self._cur_idx: int | None = None
        self._candidate: tuple[object, int] | None = None  # (speaker key or "new", consecutive hits)

    # --- public API -------------------------------------------------------------------------
    @property
    def clock_s(self) -> float:
        return self._n / self._sr

    @property
    def current_speaker(self) -> str | None:
        """Best guess for who is talking right now (None until anyone has been identified)."""
        if self._cur is not None:
            return self._cur.speaker
        return self._last_speaker

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def feed(self, pcm: bytes) -> list[SpeakerSegment]:
        """Consume 16-bit little-endian mono PCM; return segments created or changed."""
        changed: dict[int, SpeakerSegment] = {}
        data = self._pending + pcm
        step = self._frame * 2
        usable = len(data) - len(data) % step
        self._pending = data[usable:]
        if usable:
            samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
            for frame in samples.reshape(-1, self._frame):
                self._process_frame(frame, changed)
        return list(changed.values())

    def speaker_at(self, t: float, tolerance: float = 0.3) -> str | None:
        """Speaker label for session time `t`, tolerating small gaps between segments."""
        best: tuple[float, str] | None = None
        for seg in reversed(list(self.segments)):  # snapshot: feed() may run in another thread
            if seg.start - tolerance <= t <= seg.end + tolerance:
                gap = 0.0 if seg.start <= t <= seg.end else min(abs(t - seg.start), abs(t - seg.end))
                if best is None or gap < best[0]:
                    best = (gap, seg.speaker)
                    if gap == 0.0:
                        break
        return best[1] if best else None

    def reset(self) -> None:
        self.__init__(self.cfg, self._vad, self._embed)

    # --- frame loop -------------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, changed: dict[int, SpeakerSegment]) -> None:
        t_start = self._n / self._sr
        self._n += self._frame
        t_end = self._n / self._sr
        speech = bool(self._vad(frame))

        if speech:
            if not self._in_speech:
                self._begin_utterance(t_start)
            self._silence_frames = 0
            self._utt.append(frame)
            self._speech_samples += self._frame
            if self._cur is not None:
                self._cur.end = t_end
                changed[self._cur.id] = self._cur
            self._maybe_embed(t_end, changed)
        elif self._in_speech:
            self._silence_frames += 1
            self._utt.append(frame)
            if self._silence_frames * self._frame / self._sr >= self.cfg.hangover_s:
                self._end_utterance(t_end - self._silence_frames * self._frame / self._sr, changed)

    def _begin_utterance(self, t: float) -> None:
        self._in_speech = True
        self._silence_frames = 0
        self._utt = []
        self._utt_start = t
        self._speech_samples = 0
        self._last_embed_speech = 0
        self._cur = None
        self._cur_idx = None
        self._candidate = None

    def _end_utterance(self, t_end: float, changed: dict[int, SpeakerSegment]) -> None:
        if self._cur is not None:
            self._cur.end = max(self._cur.end, t_end)
            changed[self._cur.id] = self._cur
        elif self._speech_samples / self._sr >= self.cfg.short_utterance_s and self._last_speaker:
            # Too short to take a voiceprint ("yeah", "ok"): assume the previous speaker.
            seg = self._new_segment(self._last_speaker, self._utt_start, t_end, provisional=True)
            changed[seg.id] = seg
        self._in_speech = False
        self._utt = []
        self._cur = None
        self._cur_idx = None
        self._candidate = None

    # --- voiceprints ------------------------------------------------------------------------
    def _maybe_embed(self, t_now: float, changed: dict[int, SpeakerSegment]) -> None:
        cfg = self.cfg
        if self._speech_samples / self._sr < cfg.min_speech_s:
            return
        due = self._last_embed_speech == 0 or (
            (self._speech_samples - self._last_embed_speech) / self._sr >= cfg.recheck_s
        )
        if not due:
            return
        self._last_embed_speech = self._speech_samples

        window = int(cfg.window_s * self._sr)
        audio = np.concatenate(self._utt)[-window:]
        emb = np.asarray(self._embed(audio), dtype=np.float32)
        idx, sim = self._best_match(emb)
        # TEMP diagnostic: self-similarity of each voiceprint vs the best existing speaker.
        print(
            f"[diar] embed sim={sim:.3f} best=speaker_{idx} "
            f"centroids={len(self._centroids)} cur={self._cur_idx} thr={cfg.threshold}",
            flush=True,
        )

        if self._cur is None:
            if idx is not None and sim >= cfg.threshold:
                self._adopt(idx, emb)
            else:
                idx = self._create_speaker(emb, fallback=idx)
            seg = self._new_segment(f"speaker_{idx}", self._utt_start, t_now)
            self._cur, self._cur_idx = seg, idx
            self._last_speaker = seg.speaker
            changed[seg.id] = seg
            return

        # Mid-utterance recheck: only switch speaker after several consistent disagreements.
        if idx is not None and sim >= cfg.threshold and idx == self._cur_idx:
            self._candidate = None
            self._adopt(idx, emb)
            return
        key: object = idx if idx is not None and sim >= cfg.threshold else "new"
        hits = self._candidate[1] + 1 if self._candidate and self._candidate[0] == key else 1
        self._candidate = (key, hits)
        if hits < cfg.change_confirmations:
            return

        switch_t = max(self._cur.start, t_now - cfg.recheck_s * cfg.change_confirmations)
        self._cur.end = switch_t
        changed[self._cur.id] = self._cur
        if key == "new":
            new_idx = self._create_speaker(emb, fallback=idx)
        else:
            new_idx = int(key)  # type: ignore[arg-type]
            self._adopt(new_idx, emb)
        seg = self._new_segment(f"speaker_{new_idx}", switch_t, t_now)
        self._cur, self._cur_idx = seg, new_idx
        self._last_speaker = seg.speaker
        self._candidate = None
        changed[seg.id] = seg

    def _best_match(self, emb: np.ndarray) -> tuple[int | None, float]:
        best_idx, best_sim = None, -1.0
        for i, c in enumerate(self._centroids):
            sim = float(np.dot(emb, c.vec))
            if sim > best_sim:
                best_idx, best_sim = i, sim
        return best_idx, best_sim

    def _adopt(self, idx: int, emb: np.ndarray) -> None:
        c = self._centroids[idx]
        weight = min(c.count, self.cfg.centroid_cap)
        merged = c.vec * weight + emb
        norm = float(np.linalg.norm(merged))
        c.vec = merged / norm if norm > 0 else merged
        c.count += 1

    def _create_speaker(self, emb: np.ndarray, fallback: int | None) -> int:
        if len(self._centroids) >= self.cfg.max_speakers and fallback is not None:
            self._adopt(fallback, emb)  # at the cap: fold into the closest known voice
            return fallback
        self._centroids.append(_Centroid(emb))
        return len(self._centroids) - 1

    def _new_segment(self, speaker: str, start: float, end: float, provisional: bool = False) -> SpeakerSegment:
        seg = SpeakerSegment(self._next_id, speaker, start, end, provisional)
        self._next_id += 1
        self.segments.append(seg)
        if len(self.segments) > self.cfg.max_segments:
            del self.segments[: len(self.segments) - self.cfg.max_segments]
        if provisional:
            self._last_speaker = speaker
        return seg
