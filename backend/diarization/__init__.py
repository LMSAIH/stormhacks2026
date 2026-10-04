"""Optional, self-contained speaker diarization (who is talking) for the listening panel.

Public surface (nothing outside this package should import anything else):

    from diarization import create_live_diarizer, get_tts_gate, is_enabled

`create_live_diarizer()` returns None when DIARIZATION is off, so callers need no flag checks.
Text comes from ElevenLabs Scribe Realtime; this package only answers "which speaker?".
Heavy dependencies (numpy, torch, resemblyzer) are imported lazily, so importing this package
when DIARIZATION is off costs nothing. To remove the feature, delete this directory and the
call sites listed in README.md.
"""

from diarization.config import DiarizationConfig, is_enabled
from diarization.tts_gate import TtsGate, get_tts_gate

__all__ = ["DiarizationConfig", "SpeakerTracker", "TtsGate", "create_live_diarizer", "get_tts_gate",
           "is_enabled"]


def create_live_diarizer(on_segments=None):
    from diarization.live import create_live_diarizer as _create

    return _create(on_segments)


def __getattr__(name: str):
    if name == "SpeakerTracker":  # lazy: pulls in numpy
        from diarization.tracker import SpeakerTracker

        return SpeakerTracker
    raise AttributeError(name)
