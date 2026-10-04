"""Sweep Quality mode's beam decode: beam size, CTC weight, LM weight (and the length bonus) against
WER and decode time, on LRS3-100 (pre-made crops) and raw_eval-20 (our own crop of real faces).

Loads the model once and encodes each clip once; a setting only re-runs the beam search
(`LipReader.configure_beam`). Time per clip = encoder + beam search on this machine, i.e. the
service's `vsr` stage; a round trip adds upload, crop and network (~0.3-0.45 s laptop → pod, brief §11).
`s@3s` = least-squares fit of time vs clip length, read at 3 s (the round-trip budget is for a 3 s clip).

    uv run python scripts/sweep_beam.py --lrs3-parquet data/lrs3_test/0000.parquet --clips data/raw_eval \
        --beam 20 --ctc 0.1 0.2 0.3 --lm 0.1 0.3 0.5 --tag weights

Every combination of the listed values runs (plus `--settings B,CTC,LM,PEN ...`). raw_eval-20 holds 6
GRID clips ("bin blue at f two now"): fixed-grammar letters and digits, where the LM hurts; they are
also reported apart from the 14 natural sentences. Writes <out>/sweep-<tag>.json (per-clip readings).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import clips_from_dir, clips_from_parquet, norm, pct  # noqa: E402

from lipread.model import DEFAULT_BEAM, BeamSettings, LipReader  # noqa: E402
from lipread.preprocess import MouthCropper, precropped_patches, to_model_input  # noqa: E402
from lipread.video import load_video_25fps  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wer(rows: list[dict]) -> float:
    return jiwer.wer([norm(r["ref"]) for r in rows], [norm(r["hyp"]) or "<empty>" for r in rows]) if rows else float("nan")


def at_3s(rows: list[dict]) -> float:
    """Decode time (ms) a 3 s clip takes, from a straight-line fit over all clips."""
    secs = np.array([r["frames"] / 25 for r in rows])
    ms = np.array([r["ms"] for r in rows])
    slope, icept = np.polyfit(secs, ms, 1)
    return float(slope * 3 + icept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs3-parquet", type=Path)
    ap.add_argument("--n-lrs3", type=int, default=100)
    ap.add_argument("--lrs3-start", type=int, default=0, help="first LRS3 index (100+ = outside the LRS3-100 gate)")
    ap.add_argument("--clips", type=Path, help="raw videos + .txt (raw_eval)")
    ap.add_argument("--n-clips", type=int, default=20)
    ap.add_argument("--beam", type=int, nargs="+", default=[DEFAULT_BEAM.beam_size])
    ap.add_argument("--ctc", type=float, nargs="+", default=[DEFAULT_BEAM.ctc_weight])
    ap.add_argument("--lm", type=float, nargs="+", default=[DEFAULT_BEAM.lm_weight])
    ap.add_argument("--penalty", type=float, nargs="+", default=[DEFAULT_BEAM.penalty])
    ap.add_argument("--settings", nargs="*", default=[], help="extra settings as BEAM,CTC,LM,PENALTY")
    ap.add_argument("--device")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out", type=Path, default=Path("artifacts/sweep"))
    a = ap.parse_args()

    sets: list[tuple[str, list]] = []
    if a.lrs3_parquet:
        lrs3 = clips_from_parquet(a.lrs3_parquet, a.lrs3_start + a.n_lrs3)[a.lrs3_start:]
        sets.append((f"lrs3[{a.lrs3_start}:{a.lrs3_start + len(lrs3)}]", lrs3))
    if a.clips:
        sets.append(("raw_eval", clips_from_dir(a.clips, a.n_clips)))
    if not sets:
        raise SystemExit("give --lrs3-parquet and/or --clips")

    reader = LipReader(device=a.device)
    cropper = MouthCropper()
    clips = []  # (set name, clip id, ref, frames, encoder output, encoder ms)
    for name, items in sets:
        for c in items:
            x = to_model_input(cropper.crop(load_video_25fps(c.path)) if c.path else precropped_patches(c.crops))
            for _ in range(2 if not clips else 1):  # first clip: warm up cuDNN before timing
                sync()
                t = time.perf_counter()
                with torch.no_grad():
                    enc = reader.e2e.encode(x.to(reader.device))
                sync()
            clips.append((name, c.id, c.ref, int(x.shape[1]), enc, (time.perf_counter() - t) * 1000))
    print(f"# {len(clips)} clips encoded on {reader.device}; encoder p50 "
          f"{statistics.median(c[5] for c in clips):.0f} ms", flush=True)

    grid = [BeamSettings(*s) for s in itertools.product(a.beam, a.ctc, a.lm, a.penalty)]
    for s in a.settings:
        b, c, lm, pen = s.split(",")
        grid.append(BeamSettings(int(b), float(c), float(lm), float(pen)))

    results = []
    for k, s in enumerate(grid, 1):
        reader.configure_beam(s)
        reader.beam_encoded(clips[0][4])  # warm-up: allocator + kernels for this beam size
        rows = []
        for name, cid, ref, frames, enc, enc_ms in clips:
            sync()
            t = time.perf_counter()
            hyp = reader.beam_encoded(enc).text
            sync()
            rows.append({"set": name, "id": cid, "ref": ref, "hyp": hyp, "frames": frames,
                         "ms": enc_ms + (time.perf_counter() - t) * 1000})
        summary = {"settings": s.__dict__}
        for name, _ in sets:
            sub = [r for r in rows if r["set"] == name]
            summary[name] = {"wer": wer(sub), "ms_p50": pct([r["ms"] for r in sub], 0.5),
                             "ms_p95": pct([r["ms"] for r in sub], 0.95)}
            if name == "raw_eval":
                summary["raw_grid"] = wer([r for r in sub if r["id"].startswith("grid_")])
                summary["raw_natural"] = wer([r for r in sub if not r["id"].startswith("grid_")])
        summary["ms_at_3s"] = at_3s(rows)
        results.append({"summary": summary, "rows": rows})
        cells = " | ".join(f"{summary[n]['wer']:.1%}" for n, _ in sets)
        extra = f" | {summary['raw_natural']:.1%} | {summary['raw_grid']:.1%}" if "raw_grid" in summary else ""
        print(f"[{k}/{len(grid)}] beam {s.beam_size} ctc {s.ctc_weight} lm {s.lm_weight} pen {s.penalty}: "
              f"{cells}{extra} | {summary['ms_at_3s']:.0f} ms @3s", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"sweep-{a.tag}.json").write_text(json.dumps(results, indent=1))
    names = [n for n, _ in sets]
    has_raw = "raw_eval" in names
    head = " | ".join(f"{n} WER" for n in names) + (" | raw natural | raw GRID" if has_raw else "")
    print(f"\n| beam | ctc | lm | penalty | {head} | ms @3s | ms p50/p95 ({names[0]}) |")
    print("|---" * (6 + len(names) + 2 * has_raw) + "|")
    for r in results:
        s, m = r["summary"]["settings"], r["summary"]
        cells = " | ".join(f"{m[n]['wer']:.1%}" for n in names)
        if has_raw:
            cells += f" | {m['raw_natural']:.1%} | {m['raw_grid']:.1%}"
        print(f"| {s['beam_size']} | {s['ctc_weight']} | {s['lm_weight']} | {s['penalty']} | {cells} | "
              f"{m['ms_at_3s']:.0f} | {m[names[0]]['ms_p50']:.0f} / {m[names[0]]['ms_p95']:.0f} |")


if __name__ == "__main__":
    main()
