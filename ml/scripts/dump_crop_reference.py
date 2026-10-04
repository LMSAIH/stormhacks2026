"""Dump reference data from the Python mouth-crop pipeline for the TypeScript port's parity tests
(frontend/src/lib/lipreading/crop/parity.test.ts).

  --synthetic              deterministic textured frames + scripted face keypoints (no video needed),
                           run through the real vendored VideoProcess. Default --out is the committed
                           fixture dir frontend/src/lib/lipreading/crop/__fixtures__/synthetic/; also
                           writes ../reference_cases.json (cv2 gray / LMEDS, resample_fps, to_model_input).
  --video PATH --out DIR   real clip: load_video_25fps → MouthCropper.landmarks → VideoProcess
                           (not committed; point CROP_PARITY_DIR at DIR to run the TS parity test on it).

Each output dir gets
  meta.json   T, H, W, raw detector keypoints per frame (null = no face), and per 25 fps frame the
              smoothed keypoints, the 2x3 transform, LMEDS inlier mask, mouth centre in 256-space and
              patch origin — all captured from inside VideoProcess, not recomputed
  frames.bin  T*H*W uint8: the gray frames VideoProcess warps (its cv2 RGB2GRAY output)
  crops.bin   T*96*96 uint8: VideoProcess output

  uv run python scripts/dump_crop_reference.py --synthetic
  uv run python scripts/dump_crop_reference.py --video data/smoke/face.mp4 --out artifacts/crop_ref
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from lipread.vendor.mediapipe.video_process import VideoProcess

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "frontend" / "src" / "lib" / "lipreading" / "crop" / "__fixtures__"


class RecordingVideoProcess(VideoProcess):
    """The vendored VideoProcess, unchanged, plus a record of what each frame went through."""

    def __init__(self) -> None:
        super().__init__(convert_gray=True)
        self.records: list[dict] = []

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        transform = super().estimate_affine_transform(landmarks, stable_points, stable_reference)
        src = np.vstack([landmarks[x] for x in stable_points])
        _, inliers = cv2.estimateAffinePartial2D(src, stable_reference, method=cv2.LMEDS)
        self.records.append({
            "smoothed": np.array(landmarks, dtype=np.float64),
            "transform": transform.copy(),
            "inliers": inliers.ravel().astype(int).tolist(),
        })
        return transform

    def apply_affine_transform(self, frame, landmarks, transform, *args):
        out_frame, out_landmarks = super().apply_affine_transform(frame, landmarks, transform, *args)
        rec = self.records[-1]
        rec["gray"] = frame.copy()  # what warpAffine sees: the cvtColor(RGB2GRAY) output
        cx, cy = out_landmarks[self.start_idx]
        rec["mouth256"] = [float(cx), float(cy)]
        # Same expressions as cut_patch.
        rec["origin"] = [int(round(np.clip(cx - self.crop_width // 2, 0, out_frame.shape[1]))),
                         int(round(np.clip(cy - self.crop_height // 2, 0, out_frame.shape[0])))]
        return out_frame, out_landmarks


def run_pipeline(frames: np.ndarray, landmarks: list, out: Path, source: str) -> None:
    vp = RecordingVideoProcess()
    crops = vp(frames, [None if lm is None else lm.copy() for lm in landmarks])
    T, H, W = frames.shape[:3]
    assert crops.shape == (T, vp.crop_height, vp.crop_width) and crops.dtype == np.uint8, crops.shape
    assert len(vp.records) == T
    gray = np.stack([r["gray"] for r in vp.records])
    meta = {
        "source": source,
        "cv2": cv2.__version__,
        "T": T, "H": H, "W": W, "fps": 25, "patchSize": vp.crop_width,
        "stableReference": vp.get_stable_reference(vp.reference, (256, 256), (256, 256)).tolist(),
        "keypoints": [None if lm is None else np.asarray(lm).astype(int).tolist() for lm in landmarks],
        "smoothed": [r["smoothed"].tolist() for r in vp.records],
        "transforms": [r["transform"].reshape(-1).tolist() for r in vp.records],
        "inliers": [r["inliers"] for r in vp.records],
        "mouth256": [r["mouth256"] for r in vp.records],
        "origins": [r["origin"] for r in vp.records],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "frames.bin").write_bytes(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
    (out / "crops.bin").write_bytes(np.ascontiguousarray(crops).tobytes())
    n_out = sum(1 for r in vp.records if sum(r["inliers"]) < 4)
    print(f"{out}: T={T} {W}x{H}, detected {sum(lm is not None for lm in landmarks)}/{T}, "
          f"LMEDS dropped a point in {n_out} frames, origins x {min(o[0] for o in meta['origins'])}.."
          f"{max(o[0] for o in meta['origins'])} y {min(o[1] for o in meta['origins'])}.."
          f"{max(o[1] for o in meta['origins'])}")


# ---------------------------------------------------------------------------------------- synthetic

MISSING = {0, 1, 7, 12, 13, 14, 23}  # leading gap, single gap, 3-frame gap, trailing gap


def synthetic_clip(T: int = 24, H: int = 120, W: int = 160, seed: int = 20261003, identity=None):
    """Textured RGB frames (bilinear/rounding errors show up) + plausible int keypoints with jitter.

    The face drifts down so the mouth patch runs off the bottom edge in later frames (BORDER_CONSTANT),
    rolls a few degrees, and its proportions differ from the mean face so LMEDS sometimes drops a point.
    """
    rng = np.random.default_rng(seed)
    ref = VideoProcess().get_stable_reference(VideoProcess().reference, (256, 256), (256, 256))
    centre = ref.mean(axis=0)
    if identity is None:  # this "face" vs the mean face, 256-space px: the off nose makes LMEDS drop it in ~1/3 of frames
        identity = np.array([[0.0, 0.0], [0.0, 0.0], [16.0, 10.0], [0.0, 0.0]])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    frames, landmarks = [], []
    for t in range(T):
        r = 128 + 90 * np.sin((xx + 2 * t) / 9.0 + yy / 13.0) + 25 * np.sin(xx * 1.7 + yy * 0.9)
        g = 128 + 80 * np.cos((xx - t) / 7.0 - yy / 11.0) + 40 * (((xx + t) // 3 + yy // 3) % 2)
        b = 40 + 0.9 * xx + 0.6 * yy + 30 * np.cos((xx * yy) / 97.0)
        rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 12, (H, W, 3))
        frames.append(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))

        s = 0.62 + 0.04 * math.sin(t / 5)
        th = 0.12 * math.sin(t / 4 + 0.3)
        rot = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        c = np.array([80 + 6 * math.sin(t / 3), 64 + 0.6 * t])
        pts = (ref - centre + identity) @ rot.T + c + rng.normal(0, 0.7, (4, 2))
        # The detector hands VideoProcess int pixel coords (int(x * iw)), so do the same.
        landmarks.append(None if t in MISSING else np.trunc(pts).astype(np.int64))
    return np.stack(frames), landmarks


def reference_cases(seed: int = 7, n_similarity: int = 160, n_gray: int = 600) -> dict:
    """Direct reference cases for the pieces with non-obvious numerics: cv2 RGB2GRAY, cv2 LMEDS
    similarity, video.resample_fps index choice, and preprocess.to_model_input (torch)."""
    rng = np.random.default_rng(seed)
    ref = VideoProcess().get_stable_reference(VideoProcess().reference, (256, 256), (256, 256))
    centre = ref.mean(axis=0)
    with_outlier, all_inliers = [], []
    while len(with_outlier) < n_similarity // 2 or len(all_inliers) < n_similarity // 2:
        s = rng.uniform(0.3, 2.0)
        th = rng.uniform(-0.6, 0.6)
        rot = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        src = (ref - centre + rng.normal(0, rng.choice([0.5, 2.0, 5.0, 12.0]), (4, 2))) @ rot.T
        src = src + rng.uniform(40, 600, 2)
        # detector-style ints, or 13-frame window means of ints like the smoothed landmarks
        src = np.trunc(src) if rng.random() < 0.5 else np.round(src * 13) / 13
        if min(np.hypot(*(src[i] - src[j])) for i in range(4) for j in range(i + 1, 4)) < 4:
            continue
        m, inl = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        case = {"src": src.tolist(), "M": m.reshape(-1).tolist(), "inliers": inl.ravel().astype(int).tolist()}
        bucket = with_outlier if case["inliers"].count(1) < 4 else all_inliers
        if len(bucket) < n_similarity // 2:
            bucket.append(case)

    rgb = rng.integers(0, 256, (n_gray, 3)).astype(np.uint8)
    # include colours where float round(0.299R+0.587G+0.114B) and cv2's fixed point disagree
    v = rng.integers(0, 1 << 24, 400_000, dtype=np.int64)
    pool = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    pool_gray = cv2.cvtColor(pool[None], cv2.COLOR_RGB2GRAY)[0].astype(np.int64)
    flt = np.floor(pool.astype(np.float64) @ np.array([0.299, 0.587, 0.114]) + 0.5).astype(np.int64)
    tricky = pool[pool_gray != flt][:200]
    edges = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 2, 3]], np.uint8)
    rgb = np.concatenate([edges, tricky, rgb])
    gray = cv2.cvtColor(rgb[None], cv2.COLOR_RGB2GRAY)[0]

    from lipread.video import resample_fps

    resample = []
    exact = {20.0, 31.25, 62.5, 15.625, 12.5}  # frame periods exact in binary → ties are real ties
    for fps in (20.0, 31.25, 62.5, 15.625, 12.5, 30.0, 60.0, 24.0, 25.005):
        for n in (2, 4, 7, 12, 19, 33, 50, 74):
            if fps not in exact and abs((n / fps * 25) % 1 - 0.5) < 1e-6:
                continue  # frame-count tie: nominal vs measured fps may round differently
            idx = resample_fps(np.arange(n), fps)
            resample.append({"fps": fps, "n": n, "idx": [int(i) for i in idx]})

    from lipread.preprocess import to_model_input  # torch; only needed here

    T = 3
    i = np.arange(96 * 96)
    patches = np.stack([(i * 7 + t * 31) % 256 for t in range(T)]).reshape(T, 96, 96)
    lut_patch = np.zeros((96, 96), np.int64)
    centre = lut_patch[4:92, 4:92].copy().reshape(-1)
    centre[:256] = np.arange(256)  # first 256 model-input pixels = 0..255
    lut_patch[4:92, 4:92] = centre.reshape(88, 88)
    x = to_model_input(np.concatenate([patches, lut_patch[None]]).astype(np.uint8)).numpy()  # (1, T+1, 88, 88)
    probes = [[int(t), int(yy), int(xx), float(x[0, t, yy, xx])]
              for t, yy, xx in zip(rng.integers(0, T, 64), rng.integers(0, 88, 64), rng.integers(0, 88, 64))]
    return {
        "cv2": cv2.__version__,
        "stableReference": ref.tolist(),
        "similarity": with_outlier + all_inliers,
        "gray": {"rgb": rgb.reshape(-1).tolist(), "gray": gray.reshape(-1).tolist(), "tricky": int(len(tricky))},
        "resample": resample,
        "modelInput": {
            "patch": "(i * 7 + t * 31) % 256 over the flattened 96x96 patch i, t = 0..2",
            "shape": list(x.shape),
            "probes": probes,
            "lut": [float(v) for v in x[0, T].reshape(-1)[:256]],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="scripted clip → committed fixture")
    mode.add_argument("--video", type=Path, help="real face clip (any fps; resampled to 25)")
    ap.add_argument("--out", type=Path, help="output dir (default for --synthetic: the TS fixture dir)")
    args = ap.parse_args()

    if args.synthetic:
        frames, landmarks = synthetic_clip()
        run_pipeline(frames, landmarks, args.out or FIXTURES / "synthetic", "synthetic")
        if args.out is None:
            cases = reference_cases()
            (FIXTURES / "reference_cases.json").write_text(json.dumps(cases) + "\n")
            print(f"{FIXTURES / 'reference_cases.json'}: {len(cases['similarity'])} similarity, "
                  f"{len(cases['gray']['gray'])} gray, {len(cases['resample'])} resample cases")
        return

    if args.out is None:
        ap.error("--video needs --out")
    from lipread.preprocess import MouthCropper  # pulls in torch + mediapipe
    from lipread.video import load_video_25fps

    frames = load_video_25fps(args.video)
    landmarks = MouthCropper(convert_gray=True).landmarks(frames)
    run_pipeline(frames, landmarks, args.out, args.video.name)


if __name__ == "__main__":
    main()
