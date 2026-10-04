"""WER of the app's transcript (all finished lines joined) vs the 20 references joined, per mode.
Joining makes it independent of where the app cut sentences; missed and extra words still count."""
import json
import re
import sys
from pathlib import Path

import jiwer

T = Path(__file__).resolve().parents[2] / "artifacts/app_eval"


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


refs = " ".join(norm(r["ref"]) for r in json.loads((T / "eval20_refs.json").read_text()))
for mode in sys.argv[1:]:
    f = T / f"eval_app_{mode}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    hyp = " ".join(norm(x) for x in d["lines"])
    m = jiwer.process_words(refs, hyp or "<empty>")
    n = len(refs.split())
    print(f"{mode}: WER {m.wer:.1%}  ({n} ref words: {m.substitutions} wrong, {m.deletions} missed, "
          f"{m.insertions} extra)  lines {len(d['lines'])}  fps {d['fpsMedian']}")
