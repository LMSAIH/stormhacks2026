"""Send an audio file (mp3, wav, m4a, ...) to the running backend and print who spoke when.

    .venv/bin/python -m diarization.tools.post_file meeting.mp3
    .venv/bin/python -m diarization.tools.post_file meeting.mp3 --threshold 0.7 --url http://localhost:5000

Needs the server running with DIARIZATION=1. No server handy? Use `diarization.tools.label`
to run the same code in-process.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
        ".flac": "audio/flac", ".webm": "audio/webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--url", default="http://localhost:5000", help="backend base URL")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--param", "-p", action="append", default=[], metavar="NAME=VALUE",
                    help="DiarizationConfig override, e.g. -p threshold=0.7 -p max_speakers=3")
    args = ap.parse_args()

    ext = os.path.splitext(args.file)[1].lower()
    params = dict(p.split("=", 1) for p in args.param)
    if ext:
        params["format"] = ext.lstrip(".")
    with open(args.file, "rb") as f:
        data = f.read()
    try:
        r = httpx.post(f"{args.url.rstrip('/')}/api/diarize", params=params, content=data,
                       headers={"Content-Type": MIME.get(ext, "application/octet-stream")},
                       timeout=args.timeout)
    except httpx.HTTPError as e:
        sys.exit(f"could not reach {args.url}: {e}")
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text}")

    out = r.json()
    print(f"{args.file}: {out['duration']}s audio, {out['speakers']} speaker(s)\n")
    print(f"{'start':>7} {'end':>7}  speaker")
    for s in out["segments"]:
        print(f"{s['start']:7.2f} {s['end']:7.2f}  {s['speaker']}{'  (inherited)' if s['provisional'] else ''}")
    t = out["timing"]
    print(f"\ndiarized in {t['diarize_s']}s ({t['realtime_factor']}x real time), decode {t['decode_s']}s")


if __name__ == "__main__":
    main()
