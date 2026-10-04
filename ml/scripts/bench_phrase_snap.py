"""Does scoring saved phrases with the model beat the app's look-alike text matching?

Setup: N clips' references form the "phrase memory"; those N clips are in-list, the rest out of list.
For every clip, greedy-read it, then try to snap the reading to a saved phrase two ways:
  lookalike  port of frontend/src/lib/phrases/{lookalike,snap}.ts: text similarity ≥ threshold
  model      lipread.phrases: per-frame CTC log-likelihood margin vs the reading ≥ threshold
and sweep the threshold. Reports, per threshold: in-list clips fixed (snapped to their own phrase),
wrong snaps (to another phrase, in- or out-of-list), resulting WER over all clips, and the
picker's top-3 recall for in-list clips.

    uv run python scripts/bench_phrase_snap.py --lrs3-parquet data/lrs3_test/0000.parquet \
        --start 100 --n 300 --phrases 50

Uses LRS3 test idx ≥ 100 by default (idx 0-99 is the LRS3-100 regression gate).
`--readings SWEEP.json --setting K` snaps the beam readings of one setting of a scripts/sweep_beam.py
run instead of greedy ones (Quality mode), and adds `model_max`: margins against the likelier
(under CTC) of the beam and the greedy reading.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import jiwer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import clips_from_parquet, norm  # noqa: E402

from lipread import phrases as ph  # noqa: E402
from lipread.model import LipReader  # noqa: E402
from lipread.preprocess import precropped_patches, to_model_input  # noqa: E402

# --- port of frontend/src/lib/phrases/lookalike.ts -------------------------------------------
LIP_GROUPS = ["pbm", "fv", "tdnl", "kgcq", "szx", "jy", "aeiu", "ow", "hr"]
GROUP_OF = {ch: i for i, g in enumerate(LIP_GROUPS) for ch in g}


def _letter_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ga = GROUP_OF.get(a.lower())
    return 0.3 if ga is not None and ga == GROUP_OF.get(b.lower()) else 1.0


def _edit(x, y, cost) -> float:
    prev = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        diag, prev[0] = prev[0], i
        for j in range(1, len(y) + 1):
            up = prev[j]
            prev[j] = min(prev[j] + 1, prev[j - 1] + 1, diag + cost(x[i - 1], y[j - 1]))
            diag = up
    return prev[len(y)]


def word_distance(a: str, b: str) -> float:
    return 0.0 if a == b else _edit(a, b, _letter_cost) / max(len(a), len(b), 1)


def lookalike(a: str, b: str) -> float:
    x, y = ph.normalize(a).split(), ph.normalize(b).split()
    if not x and not y:
        return 1.0
    return 1 - _edit(x, y, word_distance) / max(len(x), len(y))
# ----------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs3-parquet", type=Path, required=True)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--phrases", type=int, default=50, help="size of the phrase memory")
    ap.add_argument("--decoys", type=int, default=0,
                    help="add K near-copies of each saved phrase (one word swapped) to the memory: "
                         "the hard case of short, similar phrases")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readings", type=Path, help="scripts/sweep_beam.py JSON: snap its beam readings")
    ap.add_argument("--setting", type=int, default=0, help="which setting of --readings")
    ap.add_argument("--out", type=Path, default=Path("artifacts/bench/phrase_snap.json"))
    a = ap.parse_args()

    clips = clips_from_parquet(a.lrs3_parquet, a.start + a.n)[a.start:]
    rng = random.Random(a.seed)
    in_list = set(rng.sample(range(len(clips)), a.phrases))
    memory = [clips[i].ref for i in sorted(in_list)]
    if a.decoys:
        pool = sorted({w for c in clips for w in ph.normalize(c.ref).split() if len(w) > 2})
        decoys = []
        for ref in memory:
            words = ph.normalize(ref).split()
            for _ in range(a.decoys):
                w = list(words)
                i = rng.randrange(len(w))
                w[i] = rng.choice([x for x in pool if x != w[i] and abs(len(x) - len(w[i])) <= 2] or pool)
                decoys.append(" ".join(w))
        memory += decoys
    reader = LipReader(use_lm=False)
    beam = None
    if a.readings:
        beam = {r["id"]: r["hyp"] for r in json.loads(a.readings.read_text())[a.setting]["rows"]}

    # sanity: our tokenizer ids map to the model's own token list
    probe = "THE FIRST LESSON IS ABOUT HUMILITY"
    pieces = "".join(reader.token_list[i] for i in ph.token_ids(probe)).replace("▁", " ").strip()
    assert pieces == probe, (pieces, probe)

    rows = []
    for k, c in enumerate(clips):
        x = to_model_input(precropped_patches(c.crops))
        lp = reader.ctc_log_probs(x).cpu()
        greedy = reader.greedy(x).text
        reading = beam[c.id] if beam is not None else greedy
        ranked = ph.rank_phrases(lp, reading, memory)
        look = sorted(((lookalike(reading, p), p) for p in memory), reverse=True)
        row = {"ref": c.ref, "reading": reading, "in_list": k in in_list,
               "model": [(s.text, s.margin) for s in ranked[:3]],
               "look": [(p, s) for s, p in look[:3]]}
        if beam is not None:  # margins against whichever reading the CTC head finds likelier
            ll_beam, ll_greedy = ph.ctc_log_likelihood(lp, [reading or " ", greedy or " "])
            alt = ph.rank_phrases(lp, reading if ll_beam >= ll_greedy else greedy, memory)
            row["model_max"] = [(s.text, s.margin) for s in alt[:3]]
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(clips)}", flush=True)

    def evaluate(method: str, thr: float) -> dict:
        hyps, fixed, wrong, kept_right_broken = [], 0, 0, 0
        for r in rows:
            best_text, best_score = r[method][0]
            snapped = best_score >= thr and ph.normalize(best_text) != ph.normalize(r["reading"])
            hyp = best_text if snapped else r["reading"]
            if snapped and ph.normalize(best_text) == ph.normalize(r["ref"]):
                fixed += 1
            elif snapped:
                wrong += 1
                if norm(r["reading"]) == norm(r["ref"]):
                    kept_right_broken += 1
            hyps.append(norm(hyp))
        return {"method": method, "threshold": thr, "fixed": fixed, "wrong_snaps": wrong,
                "broke_correct_readings": kept_right_broken,
                "wer": round(jiwer.wer([norm(r["ref"]) for r in rows], hyps), 4)}

    n_in = sum(r["in_list"] for r in rows)
    base_wer = jiwer.wer([norm(r["ref"]) for r in rows], [norm(r["reading"]) for r in rows])
    top3 = {m: sum(any(ph.normalize(t) == ph.normalize(r["ref"]) for t, _ in r[m]) for r in rows if r["in_list"])
            for m in ("look", "model")}
    top1 = {m: sum(ph.normalize(r[m][0][0]) == ph.normalize(r["ref"]) for r in rows if r["in_list"])
            for m in ("look", "model")}
    results = [evaluate("look", t) for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)]
    margins = (-0.5, -0.3, -0.2, -0.15, -0.1, -0.075, -0.05, -0.03, -0.02, -0.01, 0.0)
    results += [evaluate("model", t) for t in margins]
    if beam is not None:
        results += [evaluate("model_max", t) for t in margins]

    what = f"beam readings ({a.readings.name} setting {a.setting})" if beam is not None else "greedy"
    print(f"\n{len(rows)} clips (LRS3 test idx {a.start}–{a.start + len(rows) - 1}), phrase memory {len(memory)} ({a.decoys} decoys each) "
          f"(in-list {n_in}, out-of-list {len(rows) - n_in}); {what} WER with no snapping {base_wer:.1%}")
    print(f"in-list: right phrase ranked 1st — lookalike {top1['look']}/{n_in}, model {top1['model']}/{n_in}; "
          f"in top 3 — lookalike {top3['look']}/{n_in}, model {top3['model']}/{n_in}\n")
    print("| method | threshold | in-list fixed | wrong snaps | broke a correct reading | WER after |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['method']} | {r['threshold']} | {r['fixed']}/{n_in} | {r['wrong_snaps']} | "
              f"{r['broke_correct_readings']} | {r['wer']:.1%} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"base_wer": base_wer, "top1": top1, "top3": top3, "results": results,
                                 "rows": rows}, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(4)
    np.random.seed(0)
    main()
