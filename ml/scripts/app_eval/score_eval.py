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


def score(tag: str) -> dict | None:
    """WER and counts for eval_app_<tag>.json, or None when that run doesn't exist."""
    f = T / f"eval_app_{tag}.json"
    if not f.exists():
        return None
    refs = " ".join(norm(r["ref"]) for r in json.loads((T / "eval20_refs.json").read_text()))
    d = json.loads(f.read_text())
    hyp = " ".join(norm(x) for x in d["lines"])
    m = jiwer.process_words(refs, hyp or "<empty>")
    return {"wer": m.wer, "words": len(refs.split()), "wrong": m.substitutions, "missed": m.deletions,
            "extra": m.insertions, "lines": len(d["lines"]), "locks": len(d.get("locks", [])),
            "drops": len(d.get("drops", [])), "tracker": d.get("tracker"), "fps": d.get("fpsMedian")}


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        s = score(tag)
        if s is None:
            continue
        health = f"tracker {s['tracker']}" if s["tracker"] else f"fps {s['fps']}"
        print(f"{tag}: WER {s['wer']:.1%}  ({s['words']} ref words: {s['wrong']} wrong, {s['missed']} missed, "
              f"{s['extra']} extra)  lines {s['lines']}  cuts {s['locks']}  drops {s['drops']}  {health}")
