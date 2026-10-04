"""The Agentic Condom: what the lip reader read → the line the user most likely meant.

One LLM call per line, only when the reader was unsure of at least one word, bounded by the mode's
budget; the answer passes `gate.plan_edits` (rule 1) or is dropped. Every failure returns the line
as read: speech never waits longer than the budget and never gets worse than the reading.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from lipread.agentic_condom import gate, prompt
from lipread.corrector import Corrector

log = logging.getLogger("lipread.agentic_condom")

# Wait at most this long for the LLM, per mode (the browser gives the whole round trip the same).
BUDGET_MS = {"normal": 500.0, "quality": 1000.0}
# Default bracket threshold (≤ gate.SURE_ABOVE), chosen on the dev set (LRS3 test 100-299): at 0.9
# the LLM broke a right word in 8.5% of lines, at 0.6 in 4.0% (before the look-alike rule), .context/agentic-condom.md.
FLAG_BELOW = 0.6


@dataclass
class CondomRequest:
    text: str
    # Per word of `text` (any reading of the line; aligned like wordSpans.confidenceFor): None = unknown.
    words: list[tuple[str, float | None]] | None = None
    alternatives: list[str] = field(default_factory=list)
    # Phrase-memory candidates, best first: (text, model margin or None).
    phrases: list[tuple[str, float | None]] = field(default_factory=list)
    # Conversation before this line, oldest first: ("user" | "other", text).
    context: list[tuple[str, str]] = field(default_factory=list)
    mode: str = "normal"


@dataclass
class CondomResult:
    text: str
    raw: str
    edits: list[gate.Edit]
    # corrected | unchanged | skipped | rejected | timeout | error | off
    status: str
    model: str | None = None
    latency_ms: dict = field(default_factory=dict)
    detail: str = ""

    def as_dict(self) -> dict:
        return {"text": self.text, "raw": self.raw, "edits": [e.as_dict() for e in self.edits],
                "status": self.status, "model": self.model, "latency_ms": self.latency_ms}


class AgenticCondom:
    def __init__(self, corrector: Corrector | None = None, budget_ms: dict | None = None,
                 flag_below: float | None = None):
        self.corrector = corrector or Corrector()
        self.budget_ms = {**BUDGET_MS, **(budget_ms or {})}
        # Words under this confidence go in [brackets] and may change: the gate (0.9) or stricter.
        self.flag_below = min(gate.SURE_ABOVE, flag_below if flag_below is not None
                              else float(os.environ.get("CONDOM_FLAG_BELOW", FLAG_BELOW)))
        for mode in self.budget_ms:  # e.g. CONDOM_BUDGET_MS_NORMAL=450
            if env := os.environ.get(f"CONDOM_BUDGET_MS_{mode.upper()}"):
                self.budget_ms[mode] = float(env)

    @property
    def enabled(self) -> bool:
        return self.corrector.enabled

    @property
    def model(self) -> str | None:
        return self.corrector.model or None

    def correct(self, req: CondomRequest) -> CondomResult:
        """Never raises: any failure is the line as read, with the reason in `status`."""
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        raw = req.text
        tokens = gate.split_words(raw)

        def done(status: str, edits: list[gate.Edit] | None = None, llm_ms: float = 0.0, detail: str = ""):
            text = gate.apply_edits(tokens, edits) if edits else raw
            return CondomResult(text, raw, edits or [], status, self.model,
                                {"llm": llm_ms, "total": ms()}, detail)

        if len(tokens) < gate.MIN_WORDS:
            return done("skipped", detail="too short")
        conf = gate.confidence_for(tokens, req.words)
        flagged = [gate.unsure(c, self.flag_below) for c in conf]
        if not any(flagged):
            return done("skipped", detail="no unsure word")
        if not self.enabled:
            return done("off")
        messages = prompt.build_messages(tokens, flagged, req.alternatives,
                                         [p for p, _ in req.phrases], req.context)
        budget_s = self.budget_ms.get(req.mode, BUDGET_MS["normal"]) / 1000
        t1 = time.perf_counter()
        try:
            answer = self.corrector.complete(messages, max_tokens=prompt.max_tokens(len(tokens)),
                                             timeout=max(0.05, budget_s - (t1 - t0)), stop=["\n"])
        except httpx.TimeoutException:
            return done("timeout", llm_ms=round((time.perf_counter() - t1) * 1000, 1))
        except Exception as e:  # noqa: BLE001  the condom is best-effort
            log.warning("agentic condom: LLM call failed: %s", e)
            return done("error", llm_ms=round((time.perf_counter() - t1) * 1000, 1), detail=type(e).__name__)
        llm_ms = round((time.perf_counter() - t1) * 1000, 1)
        out = prompt.parse_answer(answer, upper=raw == raw.upper())
        plan = gate.plan_edits(tokens, out, conf, self.flag_below)
        if isinstance(plan, str):
            log.info("agentic condom: rejected (%s)", plan)
            return done("rejected", llm_ms=llm_ms, detail=plan)
        return done("corrected" if plan else "unchanged", plan, llm_ms)
