"""`lipread` CLI: transcribe a clip, or dump mouth crops for alignment / JS-parity checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def cmd_transcribe(args: argparse.Namespace) -> None:
    from lipread.model import LipReader
    from lipread.preprocess import MouthCropper, to_model_input
    from lipread.video import load_video_25fps

    t0 = time.perf_counter()
    frames = load_video_25fps(args.clip)
    x = to_model_input(MouthCropper().crop(frames))
    t1 = time.perf_counter()
    reader = LipReader(device=args.device, use_lm=args.decode == "beam",
                       **({"beam_size": args.beam_size} if args.beam_size else {}))
    t2 = time.perf_counter()
    result = reader.transcribe(x, decode=args.decode)
    t3 = time.perf_counter()
    print(result.text)
    print(f"# frames={x.shape[1]} conf={result.confidence} preprocess={t1 - t0:.2f}s "
          f"load={t2 - t1:.2f}s decode={t3 - t2:.2f}s device={reader.device}")


def cmd_crops(args: argparse.Namespace) -> None:
    import cv2

    from lipread.preprocess import MouthCropper
    from lipread.video import load_video_25fps, write_video

    patches = MouthCropper().crop(load_video_25fps(args.clip))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches):
        cv2.imwrite(str(out / f"{i:04d}.png"), p)
    write_video(out / "crops.mp4", patches)
    print(f"{len(patches)} crops ({patches.shape[1]}x{patches.shape[2]}) → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="lipread")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="lip-read a video clip")
    t.add_argument("clip")
    t.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    t.add_argument("--beam-size", type=int, default=None, help="default: model.DEFAULT_BEAM")
    t.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    t.set_defaults(func=cmd_transcribe)

    c = sub.add_parser("crops", help="dump 96x96 mouth crops (png + mp4)")
    c.add_argument("clip")
    c.add_argument("out_dir")
    c.set_defaults(func=cmd_crops)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
