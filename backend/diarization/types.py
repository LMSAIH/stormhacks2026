from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """A stretch of session audio attributed to one speaker.

    Times are seconds on the session audio clock (samples fed / sample rate).
    `end` keeps growing while the person is still talking.
    """

    id: int
    speaker: str  # "speaker_0", "speaker_1", ...
    start: float
    end: float
    provisional: bool = False  # True when inherited from the previous speaker (too short to embed)
