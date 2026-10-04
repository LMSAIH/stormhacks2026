"""Rule 1 of the Agentic Condom, enforced in code after the LLM answers (never trust the prompt).

A word may change only when the reader was unsure of it: confidence below SURE_ABOVE, the same gate
as the app's phrase snapping (`frontend/src/lib/lipreading/wordSpans.ts` `snapAllowed`). Unlike
`snapAllowed`, a word with no confidence counts as sure here: such words came from a saved phrase or
an expanded swear seed, i.e. were picked on purpose. The one exception is an obviously clipped word
at the edge of the line (the cut took its start or end): the first word may become a longer word that
ends with it, the last word a longer word that starts with it. At most MAX_DROPS words may go, and
the line may grow by at most MAX_GROWTH words. Anything else rejects the whole answer.

`align_words` / `confidence_for` are exact ports of the TS ones, so server and client agree on which
words an answer changed (`frontend/src/lib/agenticCondom/gate.ts` re-checks every answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# wordSpans.ts SURE_ABOVE / MAX_SNAP_DROPS: a word at or above this confidence never changes.
SURE_ABOVE = 0.9
MAX_DROPS = 1
MAX_GROWTH = 2
# Never call the LLM on a line shorter than this: it would invent content.
MIN_WORDS = 2

_NON_WORD = re.compile(r"[^a-z0-9']")


def norm(word: str) -> str:
    return _NON_WORD.sub("", word.lower())


def split_words(text: str) -> list[str]:
    return text.split()


def align_words(a: list[str], b: list[str]) -> tuple[list[int], list[bool]]:
    """Word alignment of `a` to `b` (edit distance on normalised words), as wordSpans.alignWords.

    Returns (cut, same): a[i]'s counterpart is b[cut[i]:cut[i + 1]] (extra b-words attach to the
    preceding a-word, leading ones to the first); same[i] = a[i] matched a word of b.
    """
    n, m = len(a), len(b)
    na, nb = [norm(w) for w in a], [norm(w) for w in b]
    d = [[i if j == 0 else (j if i == 0 else 0) for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (na[i - 1] != nb[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    cut, same = [0] * (n + 1), [False] * n
    cut[n] = m
    i, j = n, m
    while i > 0:
        if j > 0 and d[i][j] == d[i - 1][j - 1] + (na[i - 1] != nb[j - 1]):
            same[i - 1] = na[i - 1] == nb[j - 1]
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            j -= 1  # extra b-word: stays inside the current a-word's range
            continue
        i -= 1
        cut[i] = j
    cut[0] = 0
    return cut, same


def confidence_for(tokens: list[str], words: list[tuple[str, float | None]] | None) -> list[float | None]:
    """Map per-word confidences (any reading of the line) onto `tokens`; unmatched words get None."""
    if not words:
        return [None] * len(tokens)
    cut, same = align_words(tokens, [w for w, _ in words])
    return [words[cut[i]][1] if same[i] and cut[i] < len(words) else None for i in range(len(tokens))]


def unsure(confidence: float | None) -> bool:
    return confidence is not None and confidence < SURE_ABOVE


@dataclass(frozen=True)
class Edit:
    """Words [start, end) of the raw line became `to` ("" = deleted)."""

    start: int
    end: int
    from_: str
    to: str
    reason: str  # "unsure" | "clipped"

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "from": self.from_, "to": self.to, "reason": self.reason}


def _clipped(i: int, n: int, token: str, replacement: list[str]) -> bool:
    """An edge word the cut clipped: first word → longer word ending in it, last → starting with it."""
    if len(replacement) != 1 or n < 2:
        return False
    old, new = norm(token), norm(replacement[0])
    if not old or len(new) <= len(old):
        return False
    return (i == 0 and new.endswith(old)) or (i == n - 1 and new.startswith(old))


def plan_edits(tokens: list[str], out: list[str], conf: list[float | None]) -> list[Edit] | str:
    """The edits that turn `tokens` into `out`, or why the answer breaks rule 1 (a str)."""
    n = len(tokens)
    if not out:
        return "empty answer"
    if len(out) > n + MAX_GROWTH:
        return f"answer grew from {n} to {len(out)} words"
    cut, same = align_words(tokens, out)
    dropped = sum(cut[i + 1] == cut[i] for i in range(n))
    if dropped > MAX_DROPS:
        return f"dropped {dropped} words"
    changed = [not same[i] or cut[i + 1] - cut[i] != 1 for i in range(n)]
    reasons: list[str | None] = []
    for i in range(n):
        if not changed[i]:
            reasons.append(None)
        elif unsure(conf[i]):
            reasons.append("unsure")
        elif _clipped(i, n, tokens[i], out[cut[i]:cut[i + 1]]):
            reasons.append("clipped")
        else:
            return f"changed sure word {i} {tokens[i]!r} (confidence {conf[i]})"
    edits: list[Edit] = []
    i = 0
    while i < n:
        if reasons[i] is None:
            i += 1
            continue
        s = i
        while i < n and reasons[i] is not None:
            i += 1
        reason = "clipped" if all(r == "clipped" for r in reasons[s:i]) else "unsure"
        edits.append(Edit(s, i, " ".join(tokens[s:i]), " ".join(out[cut[s]:cut[i]]), reason))
    return edits


def apply_edits(tokens: list[str], edits: list[Edit]) -> str:
    """The line with each edit's words replaced; untouched words stay byte-for-byte."""
    out: list[str] = []
    i = 0
    for e in sorted(edits, key=lambda e: e.start):
        out += tokens[i:e.start] + split_words(e.to)
        i = e.end
    return " ".join(out + tokens[i:])
