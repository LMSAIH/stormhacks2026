"""Decode uploaded audio (mp3, wav, m4a, ogg, flac, ...) to 16 kHz mono 16-bit PCM bytes."""

from __future__ import annotations

import io
import os
import tempfile
import wave


class AudioDecodeError(ValueError):
    pass


def _wav_to_pcm(data: bytes, target_sr: int) -> bytes | None:
    """Fast path for plain 16-bit WAV (no extra dependencies beyond numpy)."""
    import numpy as np

    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if width != 2:
        return None
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        n_out = int(len(x) * target_sr / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def decode_to_pcm16(data: bytes, target_sr: int = 16000, suffix: str = "") -> bytes:
    """Any common audio format -> s16le mono PCM at `target_sr`.

    WAV is handled natively; everything else goes through librosa (soundfile / ffmpeg).
    """
    if not data:
        raise AudioDecodeError("empty audio")
    pcm = _wav_to_pcm(data, target_sr)
    if pcm is not None:
        return pcm
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        raise AudioDecodeError("non-WAV audio needs librosa (pip install -r requirements-diarization.txt)") from e
    # Write to a temp file: mp3/m4a decoders want a real path and a matching extension.
    fd, path = tempfile.mkstemp(suffix=suffix if suffix.startswith(".") else f".{suffix}" if suffix else ".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            y, _ = librosa.load(path, sr=target_sr, mono=True)
        except Exception as e:
            raise AudioDecodeError(f"could not decode audio: {e}") from e
    finally:
        os.unlink(path)
    return (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()
