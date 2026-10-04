"""Agentic Condom latency on an LLM endpoint: the condom's real prompts on built-in sentences.

    CORRECTOR_BASE_URL=http://127.0.0.1:8001/v1 CORRECTOR_MODEL=condom \
        uv run python scripts/condom_latency.py --out latency_8001.json

Each sentence goes in with two words marked unsure (and one other reading), the way a lip-read
line does; reports the LLM call's p50/p95 (ms) and how many answers passed the gate. No private
data: the sentences are below. Run it on the pod for the server-side number (no network).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from lipread.agentic_condom import AgenticCondom, CondomRequest

# (reading as a lip reader might give it, the sentence meant)
LINES = [
    ("I WANT TO GO HOMB NOW", "I WANT TO GO HOME NOW"),
    ("CAN YOU PASS THE PALT PLEASE", "CAN YOU PASS THE SALT PLEASE"),
    ("I THINK YOU ARE WIGHT", "I THINK YOU ARE RIGHT"),
    ("WHERE DID YOU PUT MY KEYS", "WHERE DID YOU PUT MY KEYS"),
    ("I AM NOT FEELING VERY WELL TODAY", "I AM NOT FEELING VERY WELL TODAY"),
    ("COULD YOU OPEN THE WIMDOW", "COULD YOU OPEN THE WINDOW"),
    ("THAT IS EXACTLY WHAT HAPPENED", "THAT IS EXACTLY WHAT HAPPENED"),
    ("I NEED SOME BATER", "I NEED SOME WATER"),
    ("THANK YOU SO MUSH FOR COMING", "THANK YOU SO MUCH FOR COMING"),
    ("WHAT TIME DOES THE MEETING STARD", "WHAT TIME DOES THE MEETING START"),
    ("PLEASE CALL MY MOTHER", "PLEASE CALL MY MOTHER"),
    ("I WOULD LIKE A CUP OF TEE", "I WOULD LIKE A CUP OF TEA"),
    ("THE DOGS ARE SITTING BY THE DOOR", "THE DOGS ARE SITTING BY THE DOOR"),
    ("HOW WAS YOUR WEEKEMD", "HOW WAS YOUR WEEKEND"),
    ("I AM SO PROUD OF YOU", "I AM SO PROUD OF YOU"),
    ("TURN THE MUSIC DOWM A LITTLE", "TURN THE MUSIC DOWN A LITTLE"),
    ("WE SHOULD LEAVE BEFORE IT GETS DARG", "WE SHOULD LEAVE BEFORE IT GETS DARK"),
    ("DO YOU WANT TO WATCH A MOVIE TONIGHT", "DO YOU WANT TO WATCH A MOVIE TONIGHT"),
    ("I CAN NOT FIND MY PHOME", "I CAN NOT FIND MY PHONE"),
    ("WHAT THE FUCK IS GOING ON", "WHAT THE FUCK IS GOING ON"),
]


def request(reading: str, meant: str, mode: str) -> CondomRequest:
    words = reading.split()
    # The misread word(s) and the word before the last are unsure; the rest are sure.
    unsure = {i for i, (a, b) in enumerate(zip(words, meant.split())) if a != b} | {len(words) - 2}
    conf = [(w, 0.45 if i in unsure else 0.97) for i, w in enumerate(words)]
    return CondomRequest(text=reading, words=conf, context=[("other", "How are you doing?")], mode=mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="normal", choices=["normal", "quality"])
    ap.add_argument("--rounds", type=int, default=3, help="passes over the sentences (the 1st warms up)")
    ap.add_argument("--budget-ms", type=float, default=5000, help="generous: measure, don't cut off")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    condom = AgenticCondom(budget_ms={"normal": a.budget_ms, "quality": a.budget_ms})
    lat, statuses, right = [], {}, 0
    for r in range(a.rounds):
        for reading, meant in LINES:
            t = time.perf_counter()
            res = condom.correct(request(reading, meant, a.mode))
            ms = (time.perf_counter() - t) * 1000
            if r == 0:
                continue  # warm-up pass (prefix cache, CUDA graphs)
            lat.append(ms)
            statuses[res.status] = statuses.get(res.status, 0) + 1
            right += res.text == meant
    lat.sort()
    out = {"model": condom.model, "base_url": condom.corrector.base_url, "calls": len(lat),
           "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
           "max_ms": round(lat[-1], 1), "statuses": statuses, "right": f"{right}/{len(lat)}"}
    print(json.dumps(out))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
