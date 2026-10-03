"""Mouth-crop pipeline — must match what the model was trained/fine-tuned on.

25 fps RGB frames → MediaPipe face detection (4 keypoints) → interpolate + 12-frame smoothing →
similarity warp to the mean face → 96×96 mouth patch → [model input] centre-crop 88, /255,
normalise (0.421, 0.165) → tensor (1, T, 88, 88).

The JS/WebGPU path has to reproduce exactly this; use `lipread crops` to dump frames and diff.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torchvision

from lipread.vendor.mediapipe.detector import LandmarksDetector
from lipread.vendor.mediapipe.video_process import VideoProcess

MEAN, STD = 0.421, 0.165


class NoFaceError(ValueError):
    """No usable face track in the clip."""


def _coverage(landmarks: list) -> float:
    return sum(lm is not None for lm in landmarks) / max(len(landmarks), 1)


class MouthCropper:
    def __init__(self, convert_gray: bool = True, min_face_coverage: float | None = None):
        self._detector = LandmarksDetector()
        self._process = VideoProcess(convert_gray=convert_gray)
        # The short-range detector fires on a few frames of pure noise; upstream then interpolates
        # those hits across the whole clip and crops garbage. Require a face in most frames.
        self.min_face_coverage = (
            min_face_coverage
            if min_face_coverage is not None
            else float(os.environ.get("LIPREAD_MIN_FACE_COVERAGE", "0.5"))
        )
        # mediapipe graphs are not thread-safe; FastAPI runs sync endpoints in a threadpool.
        self._lock = threading.Lock()

    def landmarks(self, frames: np.ndarray) -> list:
        with self._lock:
            lms = self._detector.detect(frames, self._detector.full_range_detector)
            if _coverage(lms) < self.min_face_coverage:
                short = self._detector.detect(frames, self._detector.short_range_detector)
                if _coverage(short) > _coverage(lms):
                    lms = short
        cov = _coverage(lms)
        if cov < self.min_face_coverage:
            raise NoFaceError(f"face found in {cov:.0%} of frames (need {self.min_face_coverage:.0%})")
        return lms

    def crop(self, frames: np.ndarray) -> np.ndarray:
        """(T, H, W, 3) uint8 RGB @25fps → (T, 96, 96) gray (or (T, 96, 96, 3) RGB) uint8."""
        patches = self._process(frames, self.landmarks(frames))
        if patches is None or len(patches) == 0:
            raise NoFaceError("could not crop a mouth patch")
        return patches


def precropped_patches(frames: np.ndarray) -> np.ndarray:
    """Already-aligned mouth crops (T, H, W[, 3]) → (T, 96, 96) gray uint8.

    For clients that crop locally (JS tier) and for datasets shipped as crops (LRS3 HF mirror).
    """
    import cv2

    if frames.ndim == 4:
        frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    if frames.shape[1:] != (96, 96):
        frames = np.stack([cv2.resize(f, (96, 96), interpolation=cv2.INTER_LINEAR) for f in frames])
    return frames.astype(np.uint8)


_center_crop = torchvision.transforms.CenterCrop(88)


def to_model_input(patches: np.ndarray) -> torch.Tensor:
    """(T, 96, 96) gray uint8 → float tensor (1, T, 88, 88) = (C, T, H, W), normalised like training.

    Chaplin's espnet Conv3dResNet takes (B, C, T, H, W); `E2E.encode` adds B. (auto_avsr's newer
    espnet uses (B, T, C, H, W) — don't mix the two when exporting or porting to JS.)
    """
    x = torch.from_numpy(np.ascontiguousarray(patches)).float().unsqueeze(0) / 255.0
    x = _center_crop(x)
    return (x - MEAN) / STD
