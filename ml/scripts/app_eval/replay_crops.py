"""Replay the app's own Quality uploads (`e2e_eval.mjs` with DUMP=1) through the beam at several settings.

Each saved /lipread/crops body is one sentence as the app cut it: about 1 s of still lips before
the words and the pause that ended it after. Every setting decodes the same cuts, so differences
are the decoder's alone (two app runs also differ in where the sentences got cut). Per setting and
run: WER of the joined readings vs the joined references (no phrase snapping; score_eval.py scores
the lines shown), empty readings (the app drops those), decode time.

    uv run python scripts/app_eval/replay_crops.py artifacts/app_eval/crops_q_rec1 \\
        artifacts/app_eval/crops_q_rec2 --settings 40,0.1,0.3,0 20,0.1,0.2,0
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from lipread.model import BeamSettings, LipReader
from lipread.preprocess import to_model_input

REFS = Path(__file__).resolve().parents[2] / "artifacts/app_eval/eval20_refs.json"


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(d: Path) -> list[np.ndarray]:
    """The uploads of one run, in the order the app sent them → (t, h, w) uint8 each."""
    out = []
    for m in json.loads((d / "meta.json").read_text()):
        body = (d / f"{m['i']:03d}.bin").read_bytes()
        out.append(np.frombuffer(gzip.decompress(body) if m["gzip"] else body, np.uint8).reshape(m["t"], m["h"], m["w"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--settings", nargs="+", required=True, help="BEAM,CTC,LM,PENALTY")
    ap.add_argument("--refs", type=Path, default=REFS)
    ap.add_argument("--out", type=Path, help="write every reading here (JSON)")
    a = ap.parse_args()

    refs = " ".join(norm(r["ref"]) for r in json.loads(a.refs.read_text()))
    reader = LipReader()
    runs = [(d.name, [to_model_input(c) for c in load(d)]) for d in a.dirs]
    print(f"# {', '.join(f'{n}: {len(xs)} uploads' for n, xs in runs)} on {reader.device}")
    print("| beam | ctc | lm | penalty | " + " | ".join(n for n, _ in runs) + " | mean | empty | decode p50 ms |")
    print("|---" * (6 + len(runs)) + "|")
    dump = []
    for spec in a.settings:
        b, c, lm, pen = spec.split(",")
        s = BeamSettings(int(b), float(c), float(lm), float(pen))
        reader.configure_beam(s)
        wers, empty, ms = [], 0, []
        for name, xs in runs:
            texts = []
            for x in xs:
                t = time.perf_counter()
                texts.append(reader.beam(x).text)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000)
            empty += sum(not t.strip() for t in texts)
            hyp = " ".join(norm(t) for t in texts if t.strip())
            wers.append(jiwer.wer(refs, hyp or "<empty>"))
            dump.append({"settings": s.__dict__, "run": name, "readings": texts})
        cells = " | ".join(f"{w:.1%}" for w in wers)
        print(f"| {s.beam_size} | {s.ctc_weight} | {s.lm_weight} | {s.penalty} | {cells} | "
              f"{statistics.mean(wers):.1%} | {empty} | {statistics.median(ms):.0f} |", flush=True)
    if a.out:
        a.out.write_text(json.dumps(dump, indent=1))


if __name__ == "__main__":
    main()
