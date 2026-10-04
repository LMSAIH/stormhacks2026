"""B2 ship gate (brief D74) from the bench JSONs `runpod/b2_finetune.sh` writes for PHASE=2.

    uv run python scripts/b2_gate.py /workspace/b2/bench --name FT_v1 [--holdout p3]

Reads <bench>/{lrs3,heldout}_<model>-{greedy,beam}.json for the stock model, <name> and every
<name>_a<alpha> WiSE blend. Candidates are ranked by held-out greedy WER (α is picked on the held-out
speaker; LRS3-100 is only the check). The best-ranked candidate that passes both gates ships:

  held-out greedy WER <= stock - 3.0 pts   AND   LRS3-100 greedy WER <= 30.6% (stock 28.6 + 2.0)
  AND, when <bench>/v2{top,mead}_<model>-greedy.json exist for stock: eval v2 (raw_eval_v2, 144 clips
  of 62 unseen people, `.context/eval-v2.md`) greedy WER <= stock + 2.0 pts on the clips both read
  (paired); a candidate without eval v2 results then fails it. Eval v2 beam is reported, not gated.

Held-out WER is also split into the demo phrases (ids 001-012 and 041-052 in every script: same text
as in training, new session) and the unseen sentences (the rest), reported apart. Prints a markdown table, writes
<bench>/gate.json. Exit code: 0 = a candidate ships, 1 = none does (stock stays), 2 = missing files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bench  # noqa: E402  same text normalisation as every other WER in the repo

STOCK = "LRS3_V_WER19.1"


def wer(rows: list[dict]) -> float | None:
    ok = [r for r in rows if r.get("error") is None]
    if not ok:
        return None
    return jiwer.wer([bench.norm(r["ref"]) for r in ok], [bench.norm(r["hyp"]) or "<empty>" for r in ok])


def clip_no(clip_id: str) -> int | None:
    m = re.search(r"_(\d+)$", clip_id)
    return int(m.group(1)) if m else None


def load(bench_dir: Path, tag: str) -> list[dict] | None:
    f = bench_dir / f"{tag}.json"
    return json.loads(f.read_text())["rows"] if f.is_file() else None


def parse_ids(spec: str) -> set[int]:
    """'1-12,41-52' → {1..12, 41..52}"""
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def scores(bench_dir: Path, model: str, demo_ids: set[int]) -> dict:
    out: dict = {"model": model}
    for decode in ("greedy", "beam"):
        lrs3 = load(bench_dir, f"lrs3_{model}-{decode}")
        held = load(bench_dir, f"heldout_{model}-{decode}")
        out[f"lrs3_{decode}"] = wer(lrs3) if lrs3 is not None else None
        out[f"heldout_{decode}"] = wer(held) if held is not None else None
        if held is not None:
            demo = [r for r in held if clip_no(r["id"]) in demo_ids]
            unseen = [r for r in held if clip_no(r["id"]) not in demo_ids]
            out[f"demo_{decode}"], out[f"unseen_{decode}"] = wer(demo), wer(unseen)
            out["n_demo"], out["n_unseen"] = len(demo), len(unseen)
        v2 = [r for part in ("v2top", "v2mead") for r in (load(bench_dir, f"{part}_{model}-{decode}") or [])]
        out[f"_v2_{decode}"] = {r["id"]: r for r in v2 if r.get("error") is None} or None
    return out


def v2_delta(stock: dict, cand: dict, decode: str) -> float | None:
    """Candidate minus stock eval v2 WER (points) on the clips both read."""
    s, c = stock.get(f"_v2_{decode}"), cand.get(f"_v2_{decode}")
    if not s or not c:
        return None
    ids = sorted(s.keys() & c.keys())
    return 100 * (wer([c[i] for i in ids]) - wer([s[i] for i in ids]))


def decide(stock: dict, cands: list[dict], min_gain: float, lrs3_max: float, v2_max: float = 2.0) -> dict:
    """Rank by held-out greedy WER (ties: lower LRS3 greedy); ship the first that passes every gate."""
    ranked = sorted((c for c in cands if c["heldout_greedy"] is not None and c["lrs3_greedy"] is not None),
                    key=lambda c: (c["heldout_greedy"], c["lrs3_greedy"]))
    for c in ranked:
        c["gain_pts"] = 100 * (stock["heldout_greedy"] - c["heldout_greedy"])
        c["pass_gain"] = c["gain_pts"] >= min_gain - 1e-9
        c["pass_lrs3"] = 100 * c["lrs3_greedy"] <= lrs3_max + 1e-9
        c["v2_delta_greedy"], c["v2_delta_beam"] = v2_delta(stock, c, "greedy"), v2_delta(stock, c, "beam")
        c["pass_v2"] = (not stock.get("_v2_greedy")) or (c["v2_delta_greedy"] is not None
                                                         and c["v2_delta_greedy"] <= v2_max + 1e-9)
    ship = next((c for c in ranked if c["pass_gain"] and c["pass_lrs3"] and c["pass_v2"]), None)
    return {"ranked": [c["model"] for c in ranked], "top": ranked[0]["model"] if ranked else None,
            "ship": ship["model"] if ship else None}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir", type=Path)
    ap.add_argument("--name", default="FT_v1", help="fine-tune name; blends are <name>_a<alpha>")
    ap.add_argument("--demo", default="1-12,41-52", help="clip numbers that are demo phrases (.context/b2-scripts)")
    ap.add_argument("--min-gain", type=float, default=3.0, help="held-out greedy gain needed, WER points")
    ap.add_argument("--lrs3-max", type=float, default=30.6, help="LRS3-100 greedy ceiling, percent")
    ap.add_argument("--v2-max", type=float, default=2.0, help="eval v2 greedy: most points worse than stock")
    a = ap.parse_args()

    blends = {p.name.removesuffix("-greedy.json").removeprefix("heldout_")
              for p in a.bench_dir.glob(f"heldout_{a.name}_a*-greedy.json")}
    models = [STOCK, a.name] + sorted(blends, key=lambda m: float(m.rsplit("_a", 1)[1]))
    rows = [scores(a.bench_dir, m, parse_ids(a.demo)) for m in models]
    stock, cands = rows[0], [r for r in rows[1:] if r["heldout_greedy"] is not None]
    if stock["heldout_greedy"] is None or stock["lrs3_greedy"] is None or not cands:
        print(f"missing bench files in {a.bench_dir} (need stock + at least one candidate)", file=sys.stderr)
        sys.exit(2)
    d = decide(stock, cands, a.min_gain, a.lrs3_max, a.v2_max)
    v2_on = bool(stock.get("_v2_greedy"))
    for r in rows:
        for k in ("greedy", "beam"):
            v = r.pop(f"_v2_{k}")
            r[f"v2_{k}"] = wer(list(v.values())) if v else None
            r[f"v2_n_{k}"] = len(v) if v else 0

    print(f"held-out: demo phrases n={stock.get('n_demo')}, unseen sentences n={stock.get('n_unseen')}\n")
    print("| model | LRS3-100 greedy | LRS3-100 beam | held-out greedy | held-out beam | demo greedy / beam | "
          "unseen greedy / beam | eval v2 greedy (Δ) | eval v2 beam (Δ) | gain | gates |")
    print("|---" * 11 + "|")
    for r in rows:
        gates = "" if r is stock else ("ship" if r["model"] == d["ship"] else
                                       ("pass" if r.get("pass_gain") and r.get("pass_lrs3") and r.get("pass_v2", True) else
                                        ", ".join(g for g, ok in (("gain", r.get("pass_gain")),
                                                                  ("LRS3", r.get("pass_lrs3")),
                                                                  ("v2", r.get("pass_v2", True))) if not ok) + " ✗"))
        dv = lambda k: "" if r.get(f"v2_delta_{k}") is None else f" ({r[f'v2_delta_{k}']:+.1f})"  # noqa: E731
        gain = "" if r is stock or "gain_pts" not in r else f"{r['gain_pts']:+.1f}"
        print(f"| {r['model']} | {pct(r['lrs3_greedy'])} | {pct(r['lrs3_beam'])} | {pct(r['heldout_greedy'])} | "
              f"{pct(r['heldout_beam'])} | {pct(r.get('demo_greedy'))} / {pct(r.get('demo_beam'))} | "
              f"{pct(r.get('unseen_greedy'))} / {pct(r.get('unseen_beam'))} | {pct(r['v2_greedy'])}{dv('greedy')} | "
              f"{pct(r['v2_beam'])}{dv('beam')} | {gain} | {gates} |")
    verdict = (f"SHIP {d['ship']}" + ("" if d["ship"] == d["top"] else f" (best held-out {d['top']} fails a gate)")
               if d["ship"] else f"NO SHIP: stock {STOCK} stays (best held-out {d['top']})")
    v2_note = (f", eval v2 greedy <= stock + {a.v2_max} pts (n={stock['v2_n_greedy']})" if v2_on
               else ", eval v2 not run")
    print(f"\n{verdict}  [gates: held-out greedy gain >= {a.min_gain} pts, LRS3-100 greedy <= {a.lrs3_max}%{v2_note}]")
    (a.bench_dir / "gate.json").write_text(json.dumps({"decision": d, "rows": rows, "min_gain": a.min_gain,
                                                        "lrs3_max": a.lrs3_max, "v2_max": a.v2_max if v2_on else None},
                                                       indent=1))
    sys.exit(0 if d["ship"] else 1)


if __name__ == "__main__":
    main()
