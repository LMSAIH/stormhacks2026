"""Video I/O. OpenCV instead of torchvision.io.read_video (deprecated) so serving doesn't depend on it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_FPS = 25.0


def load_video(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (frames as uint8 RGB array (T, H, W, 3), source fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from: {path}")
    # Browser MediaRecorder webm often reports 0 or 1000 fps; fall back to the model rate.
    if not 1.0 <= fps <= 240.0:
        fps = MODEL_FPS
    return np.stack(frames), float(fps)


def resample_fps(frames: np.ndarray, src_fps: float, dst_fps: float = MODEL_FPS) -> np.ndarray:
    """Nearest-frame resample by timestamp (drop or repeat frames) to dst_fps."""
    if abs(src_fps - dst_fps) < 0.01:
        return frames
    duration = len(frames) / src_fps
    n_out = max(1, round(duration * dst_fps))
    idx = np.minimum((np.arange(n_out) * src_fps / dst_fps).round().astype(int), len(frames) - 1)
    return frames[idx]


def load_video_25fps(path: str | Path) -> np.ndarray:
    frames, fps = load_video(path)
    return resample_fps(frames, fps)


def write_video(path: str | Path, frames: np.ndarray, fps: float = MODEL_FPS) -> None:
    """Write (T, H, W) gray or (T, H, W, 3) RGB uint8 frames as mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(f)
    out.release()
