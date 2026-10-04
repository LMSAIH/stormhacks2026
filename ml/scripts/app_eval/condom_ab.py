"""Agentic Condom A/B through the real /app: each mode with the condom off and on, N runs each.

  uv run python scripts/app_eval/condom_ab.py --lipread-url https://<serving-pod>-8000.proxy.runpod.net \
      --condom-url https://<pod with the new /correct>-8000.proxy.runpod.net [--modes normal quality] [--runs 2]

Same harness as the gate (`gate.py`: its own `pnpm dev`, e2e_eval.mjs, score_eval.py); runs alternate
off/on so drift hits both. Quality needs a browser that reaches the pod (see README: proxy CA).
Prints WER per run and the condom's answers by status (from the page's POST /correct responses).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import gate
from score_eval import T, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lipread-url", default="", help="VITE_LIPREAD_URL (Quality reads)")
    ap.add_argument("--condom-url", required=True, help="VITE_CONDOM_URL (POST /correct)")
    ap.add_argument("--budget-ms", help="VITE_CONDOM_BUDGET_MS: override both budgets (slow links only)")
    ap.add_argument("--modes", nargs="+", default=["normal", "quality"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--watch-s", default="140")
    ap.add_argument("--prefix", default="condom")
    a = ap.parse_args()
    gate.check_inputs()
    os.environ.update({"VITE_LIPREAD_URL": a.lipread_url, "VITE_CONDOM_URL": a.condom_url,
                       "VITE_CONDOM_BUDGET_MS": a.budget_ms or "", "WATCH_S": a.watch_s})
    playwright, chrome = gate.find_playwright(), gate.find_chrome()
    port = gate.free_port()
    proc = gate.start_dev_server(port)
    rows = []
    try:
        for mode in a.modes:
            for run in range(1, a.runs + 1):
                for condom in ("off", "on"):
                    tag = f"{a.prefix}_{mode}_{condom}_{run}"
                    os.environ["CONDOM"] = condom
                    print(f"── {tag}", flush=True)
                    wer = gate.run(mode, tag, f"http://localhost:{port}", playwright, chrome)
                    d = json.loads((T / f"eval_app_{tag}.json").read_text())
                    rows.append({"mode": mode, "condom": condom, "run": run, "wer": wer,
                                 "server": d.get("server", {}), **(score(tag) or {})})
    finally:
        gate.stop(proc)
    print("\n| mode | condom | runs (WER) | mean | server reads | condom answers |")
    print("|---|---|---|---|---|---|")
    for mode in a.modes:
        for condom in ("off", "on"):
            rs = [r for r in rows if r["mode"] == mode and r["condom"] == condom]
            if not rs:
                continue
            answers: dict[str, int] = {}
            for r in rs:
                for k, v in r["server"].get("condom", {}).items():
                    answers[k] = answers.get(k, 0) + v
            wers = ", ".join("%.1f%%" % r["wer"] for r in rs)
            mean = statistics.mean(r["wer"] for r in rs)
            reads = sum(r["server"].get("reads", 0) for r in rs)
            got = ", ".join("%s %d" % kv for kv in sorted(answers.items())) or "-"
            print(f"| {mode} | {condom} | {wers} | {mean:.1f}% | {reads} | {got} |")
    (T / f"{a.prefix}_ab.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
