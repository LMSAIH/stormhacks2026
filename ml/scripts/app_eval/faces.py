"""Unseen faces: which raw_eval speakers lose the most words, and why (face size, light, movement).

  uv run python scripts/app_eval/faces.py --normal TAG... --instant TAG... \
      [--alone artifacts/bench/<model alone>-greedy.json]

Per clip: words wrong in the given app runs (finished lines mapped to clips by time, as in cuts.py)
and for the model alone (bench.py on the same clips), next to what the camera sees: eye distance
in the app's 640x480 view (the eval video letterboxes every clip to it), frame and mouth brightness,
mouth contrast, head movement and turn. Writes artifacts/app_eval/faces/faces.json and sheet.png
(one row per clip, most words lost first: a frame with the 4 keypoints, then 8 mouth crops as the
model gets them). Crops come from the Python pipeline, which the browser crop matches bit for bit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jiwer
import numpy as np
from PIL import Image, ImageDraw

from cuts import clips
from lipread.preprocess import MouthCropper
from lipread.video import load_video_25fps

ML = Path(__file__).resolve().parents[2]
T = ML / "artifacts/app_eval"
OUT = T / "faces"
VIEW_W, VIEW_H = 640, 480  # the eval video's frame; make_eval_video.py scales each clip to fit
LEAD_S = 1.0  # a sentence starts LEAD_MS before the movement that started it


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s)).strip()


def errors(ref: str, hyp: str) -> int:
    m = jiwer.process_words(norm(ref), norm(hyp) or "<empty>")
    return m.substitutions + m.deletions + m.insertions


def app_reads(tag: str) -> dict[int, str]:
    """Clip index → what the app showed for it in run `tag`."""
    run = json.loads((T / f"eval_app_{tag}.json").read_text())
    trace = json.loads((T / f"trace_{tag}.json").read_text())
    origin = run["origin"]
    out: dict[int, list[str]] = {}
    for f in (e for e in trace if e["kind"] == "final"):
        onset = (f["startTms"] - origin) / 1000 + LEAD_S
        i = min(range(len(clips())), key=lambda k: abs((clips()[k]["start"] + clips()[k]["end"]) / 2 - onset))
        out.setdefault(i, []).append(f["shown"])
    return {i: " ".join(v) for i, v in out.items()}


def measure(path: Path) -> dict:
    frames = load_video_25fps(path)
    h, w = frames.shape[1:3]
    scale = min(VIEW_W / w, VIEW_H / h)
    cropper = MouthCropper()
    lms = cropper.landmarks(frames)
    patches = cropper.crop(frames)
    kp = np.array([lm for lm in lms if lm is not None], dtype=float)  # (n, 4, 2): r eye, l eye, nose, mouth
    eyes = kp[:, 1] - kp[:, 0]
    eye = np.hypot(eyes[:, 0], eyes[:, 1])
    mid = (kp[:, 0] + kp[:, 1]) / 2
    step = np.hypot(*np.diff(mid, axis=0).T) / eye[1:]  # eye-midpoint travel per 25 fps frame
    gray = frames.mean(axis=3)
    return {
        "source": f"{w}x{h}",
        "eye_px": float(np.median(eye) * scale),  # in the app's 640x480 view
        "eye_src_px": float(np.median(eye)),  # what the camera really captured
        "frame_light": float(np.median(gray.mean(axis=(1, 2)))),
        "mouth_light": float(patches.mean()),
        "mouth_contrast": float(np.median([p.std() for p in patches])),
        "motion_pct": float(np.median(step) * 100),
        "motion_p90_pct": float(np.percentile(step, 90) * 100),
        "turn": float(np.median(np.abs(kp[:, 2, 0] - mid[:, 0]) / eye)),
        "roll_deg": float(np.degrees(np.median(np.arctan2(eyes[:, 1], eyes[:, 0])))),
        "frames": frames,
        "lms": lms,
        "patches": patches,
    }


def row_image(m: dict, label: str, n: int = 8) -> Image.Image:
    frames, lms, patches = m["frames"], m["lms"], m["patches"]
    mid = len(frames) // 2
    frame = Image.fromarray(frames[mid])
    if lms[mid] is not None:
        d = ImageDraw.Draw(frame)
        r = max(2, frame.width // 160)
        for x, y in lms[mid]:
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 64, 64), width=max(1, r // 2))
    frame.thumbnail((180, 96))
    picks = np.linspace(0, len(patches) - 1, n).round().astype(int)
    row = Image.new("RGB", (180 + 4 + n * 98, 96 + 16), "white")
    row.paste(frame, (0, 16))
    for k, i in enumerate(picks):
        row.paste(Image.fromarray(patches[i]).convert("RGB"), (184 + k * 98, 16))
    ImageDraw.Draw(row).text((2, 2), label, fill="black")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", nargs="*", default=[])
    ap.add_argument("--instant", nargs="*", default=[])
    ap.add_argument("--alone", type=Path, default=ML / "artifacts/bench/model_alone_int8-greedy.json")
    args = ap.parse_args()
    alone = {}
    if args.alone.exists():
        alone = {r["id"]: r["hyp"] for r in json.loads(args.alone.read_text())["rows"]}
    runs = {"normal": [app_reads(t) for t in args.normal], "instant": [app_reads(t) for t in args.instant]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(clips()):
        m = measure(ML / "data/raw_eval" / f"{c['id']}.mp4")
        words = len(norm(c["ref"]).split())
        lost = {mode: (float(np.mean([errors(c["ref"], r.get(i, "")) for r in rs])) if rs else None)
                for mode, rs in runs.items()}
        rows.append({"id": c["id"], "ref": c["ref"], "words": words, **lost,
                     "alone": errors(c["ref"], alone[c["id"]]) if c["id"] in alone else None,
                     "reads": {mode: [r.get(i, "") for r in rs] for mode, rs in runs.items()},
                     **{k: v for k, v in m.items() if k not in ("frames", "lms", "patches")}, "_m": m})
    key = lambda r: -((r["normal"] or 0) + (r["instant"] or 0) + (r["alone"] or 0)) / r["words"]  # noqa: E731
    rows.sort(key=key)
    images = []
    for r in rows:
        label = (f"{r['id']}  lost/{r['words']}: normal {r['normal']:.1f}  instant {r['instant']:.1f}  "
                 f"alone {r['alone']}  | eyes {r['eye_px']:.0f}px  light {r['frame_light']:.0f}  "
                 f"contrast {r['mouth_contrast']:.0f}  motion {r['motion_pct']:.1f}%  turn {r['turn']:.2f}")
        images.append(row_image(r.pop("_m"), label))
    sheet = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), "white")
    y = 0
    for im in images:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(OUT / "sheet.png")
    (OUT / "faces.json").write_text(json.dumps(rows, indent=1))
    print("| clip | words | lost: normal | instant | alone | eyes px (view / source) | light | mouth contrast "
          "| motion %/frame | turn |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['words']} | {r['normal']:.1f} | {r['instant']:.1f} | {r['alone']} | "
              f"{r['eye_px']:.0f} / {r['eye_src_px']:.0f} | {r['frame_light']:.0f} | {r['mouth_contrast']:.0f} | "
              f"{r['motion_pct']:.1f} | {r['turn']:.2f} |")
    print(f"wrote {OUT / 'sheet.png'}")


if __name__ == "__main__":
    main()
