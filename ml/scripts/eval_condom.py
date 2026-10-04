"""Agentic Condom eval: share of words wrong with the condom on vs off, on the same readings.

1. readings: the VSR's reading, per-word confidences and beam alternatives per clip, from a lipread
   server's POST /lipread/crops (what the browser sends; raw videos are cropped here first):
     uv run python scripts/eval_condom.py readings --clips data/raw_eval --url $POD --decode greedy beam
     uv run python scripts/eval_condom.py readings --lrs3-parquet data/lrs3_test/0000.parquet --n 100 --url $POD ...
   → artifacts/condom/readings_<set>_<decode>.json (greedy = Normal's kind of reading, beam = Quality's)
2. score: the condom on each reading, in-process (CORRECTOR_BASE_URL / CORRECTOR_MODEL /
   CORRECTOR_API_KEY, e.g. vLLM on the pod) or through a server's POST /correct (--correct-url):
     uv run python scripts/eval_condom.py score artifacts/condom/readings_*.json --tag qwen7b [--budget-ms 500]
   Reports WER off/on, lines changed, lines where it changed a word the reader had right (the ship
   gate: < 2% of lines), statuses and LLM latency. No phrases or conversation here: each clip is
   a lone sentence, so this is the cold-start case; the app eval covers phrase memory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from lipread.agentic_condom import AgenticCondom, CondomRequest, gate  # noqa: E402

OUT = Path("artifacts/condom")


def readings(a: argparse.Namespace) -> None:
    import httpx

    from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches
    from lipread.video import load_video_25fps
    if a.clips:
        clips = bench.clips_from_dir(a.clips, a.start + a.n)[a.start:]
    else:
        clips = bench.clips_from_parquet(a.lrs3_parquet, a.start + a.n)[a.start:]
    name = a.name or (a.clips.name if a.clips else f"lrs3_{a.n}" + (f"_from{a.start}" if a.start else ""))
    client = httpx.Client(timeout=120)
    cropper = MouthCropper() if a.clips else None
    crops = {}
    for c in clips:
        try:
            crops[c.id] = cropper.crop(load_video_25fps(c.path)) if c.path else precropped_patches(c.crops)
        except NoFaceError:
            print(f"{c.id}: no face (client-side crop), left out", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)
    for decode in a.decode:
        rows = []
        for c in clips:
            if c.id not in crops:
                continue
            x = np.ascontiguousarray(crops[c.id], dtype=np.uint8)
            n, h, w = x.shape
            r = client.post(f"{a.url.rstrip('/')}/lipread/crops", content=gzip.compress(x.tobytes()),
                            headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip"},
                            params={"t": n, "h": h, "w": w, "decode": decode, "correct": "false"})
            r.raise_for_status()
            j = r.json()
            rows.append({"id": c.id, "ref": c.ref, "text": j["raw_text"], "words": j["words"],
                         "alternatives": [x["text"] for x in j["alternatives"]], "frames": j["frames"]})
            print(f"{decode} {c.id}: {j['raw_text']}", file=sys.stderr)
        path = a.out / f"readings_{name}_{decode}.json"
        path.write_text(json.dumps({"set": name, "decode": decode, "url": a.url, "rows": rows}, indent=1))
        wer = jiwer.wer([bench.norm(r["ref"]) for r in rows], [bench.norm(r["text"]) or "<empty>" for r in rows])
        print(f"{path}: {len(rows)} clips, WER {wer:.1%}")


def line_errors(ref: str, hyp: str) -> int:
    ref, hyp = bench.norm(ref), bench.norm(hyp)
    if not ref:
        return len(hyp.split())
    o = jiwer.process_words(ref, hyp or "<empty>")
    return o.substitutions + o.deletions + o.insertions


def run_condom(a: argparse.Namespace, row: dict, mode: str) -> dict:
    req = CondomRequest(text=row["text"], words=[(w["text"], w["confidence"]) for w in row["words"]],
                        alternatives=row["alternatives"][1:], mode=mode)
    if a.correct_url:
        import httpx
        body = {"text": req.text, "words": row["words"], "alternatives": req.alternatives, "mode": mode}
        t = time.perf_counter()
        j = httpx.post(f"{a.correct_url.rstrip('/')}/correct", content=json.dumps(body),
                       headers={"Content-Type": "text/plain"}, timeout=30).json()
        j["latency_ms"]["rtt"] = (time.perf_counter() - t) * 1000
        return j
    return CONDOM.correct(req).as_dict() | {"detail": None}


def score_file(a: argparse.Namespace, path: Path) -> dict:
    data = json.loads(path.read_text())
    mode = "quality" if data["decode"] == "beam" else "normal"
    rows = data["rows"]
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda r: run_condom(a, r, mode), rows))
    lines = []
    for row, res in zip(rows, results):
        toks = gate.split_words(row["text"])
        ref_toks = bench.norm(row["ref"]).split()
        _, right = gate.align_words(toks, ref_toks)  # right[i]: the reader had word i right
        touched_right = any(right[i] for e in res["edits"] for i in range(e["start"], e["end"]))
        lines.append({"id": row["id"], "ref": row["ref"], "raw": row["text"], "on": res["text"],
                      "status": res["status"], "edits": res["edits"], "touched_right": touched_right,
                      "errors_off": line_errors(row["ref"], row["text"]),
                      "errors_on": line_errors(row["ref"], res["text"]), "latency_ms": res["latency_ms"]})
    refs = [bench.norm(r["ref"]) for r in rows]
    n_words = sum(len(r.split()) for r in refs)
    changed = [ln for ln in lines if ln["edits"]]
    llm = [ln["latency_ms"]["llm"] for ln in lines if ln["latency_ms"].get("llm")]
    out = {
        "file": str(path), "set": data["set"], "decode": data["decode"], "mode": mode, "lines": len(lines),
        "words": n_words,
        "wer_off": sum(ln["errors_off"] for ln in lines) / n_words,
        "wer_on": sum(ln["errors_on"] for ln in lines) / n_words,
        "lines_changed": len(changed),
        "lines_better": sum(ln["errors_on"] < ln["errors_off"] for ln in lines),
        "lines_worse": sum(ln["errors_on"] > ln["errors_off"] for ln in lines),
        "lines_touched_right_word": sum(ln["touched_right"] for ln in lines),
        "status": {s: sum(ln["status"] == s for ln in lines) for s in sorted({ln["status"] for ln in lines})},
        "llm_ms_p50": bench.pct(llm, 0.5) if llm else None, "llm_ms_p95": bench.pct(llm, 0.95) if llm else None,
        "examples": [{k: ln[k] for k in ("id", "ref", "raw", "on", "errors_off", "errors_on", "touched_right")}
                     for ln in changed],
    }
    return out


def score(a: argparse.Namespace) -> None:
    global CONDOM
    CONDOM = AgenticCondom(budget_ms={"normal": a.budget_ms, "quality": a.budget_ms} if a.budget_ms else None,
                           flag_below=a.flag_below)
    if not a.correct_url and not CONDOM.enabled:
        raise SystemExit("set CORRECTOR_BASE_URL + CORRECTOR_MODEL (or --correct-url)")
    reports = [score_file(a, p) for p in a.readings]
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"score_{a.tag}.json"
    out.write_text(json.dumps({"tag": a.tag, "model": a.correct_url or CONDOM.model, "reports": reports}, indent=1))
    print(f"\n{a.tag} ({a.correct_url or CONDOM.model}) → {out}\n")
    print("| set | decode | lines | WER off | WER on | changed | better | worse | changed a right word | "
          "statuses | LLM ms p50/p95 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        st = ", ".join(f"{k} {v}" for k, v in r["status"].items())
        lat = f"{r['llm_ms_p50']:.0f} / {r['llm_ms_p95']:.0f}" if r["llm_ms_p50"] is not None else "-"
        print(f"| {r['set']} | {r['decode']} | {r['lines']} | {r['wer_off']:.1%} | {r['wer_on']:.1%} | "
              f"{r['lines_changed']} | {r['lines_better']} | {r['lines_worse']} | "
              f"{r['lines_touched_right_word']} ({r['lines_touched_right_word'] / r['lines']:.1%}) | {st} | {lat} |")
    total = sum(r["lines"] for r in reports)
    touched = sum(r["lines_touched_right_word"] for r in reports)
    print(f"\nall: {touched}/{total} lines ({touched / total:.1%}) had a right word changed")
    if a.examples:
        for r in reports:
            for e in r["examples"]:
                print(f"  [{r['set']} {r['decode']}] {e['raw']!r} → {e['on']!r} (ref {bench.norm(e['ref'])!r}; "
                      f"errors {e['errors_off']}→{e['errors_on']}{'; CHANGED A RIGHT WORD' if e['touched_right'] else ''})")


CONDOM: AgenticCondom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("readings")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", type=Path)
    src.add_argument("--lrs3-parquet", type=Path)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--start", type=int, default=0, help="skip this many clips (LRS3 100+ = dev set, not the gate)")
    r.add_argument("--name")
    r.add_argument("--url", required=True, help="lipread server (POST /lipread/crops)")
    r.add_argument("--decode", nargs="+", choices=["greedy", "beam"], default=["greedy", "beam"])
    r.add_argument("--out", type=Path, default=OUT)
    s = sub.add_parser("score")
    s.add_argument("readings", nargs="+", type=Path)
    s.add_argument("--tag", required=True)
    s.add_argument("--correct-url", help="score through this server's POST /correct instead of in-process")
    s.add_argument("--budget-ms", type=float, help="in-process LLM budget for both modes (default: 500 / 1000)")
    s.add_argument("--flag-below", type=float, help="bracket threshold (default: the condom's)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--examples", action="store_true", help="print every changed line")
    s.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    {"readings": readings, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
