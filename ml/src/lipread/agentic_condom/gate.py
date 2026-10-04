"""Rule 1 of the Agentic Condom, enforced in code after the LLM answers (never trust the prompt).

A word may change only when the reader was unsure of it: confidence below SURE_ABOVE, the same gate
as the app's phrase snapping (`frontend/src/lib/lipreading/wordSpans.ts` `snapAllowed`); the condom
brackets a stricter set (`condom.FLAG_BELOW`). Unlike `snapAllowed`, a word with no confidence
counts as sure: such words came from a saved phrase or an expanded swear seed, i.e. were picked on
purpose. And it may only become ONE word that looks like it on the lips (`lip_distance` ≤
MAX_LIP_DISTANCE, the look-alike measure of `frontend/src/lib/phrases/lookalike.ts`) or that
completes or trims it ("LU" → "LUTHER", "TERRORISMISM" → "TERRORISM"): the line keeps its word
count. On the dev set, inserted glue words and lip-unlike swaps ("DOG" → "CAR") were most of the
harm (`.context/agentic-condom.md`). The one exception for sure words is an obviously clipped edge
word: the first word may become a longer word that ends with it, the last a longer word that starts
with it. Anything else rejects the whole answer.

`align_words` / `confidence_for` are exact ports of the TS ones, so server and client agree on which
words an answer changed (`frontend/src/lib/agenticCondom/gate.ts` re-checks every answer; parity
fixture: `frontend/src/lib/agenticCondom/__fixtures__/gate.json`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# wordSpans.ts SURE_ABOVE: a word at or above this confidence never changes.
SURE_ABOVE = 0.9
# A replacement must look this alike on the lips (0 = same letters, 1 = nothing alike).
MAX_LIP_DISTANCE = 0.4
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


# lookalike.ts LIP_GROUPS: letters that look the same on the lips; a swap inside a group costs 0.3.
_LIP_GROUP = {ch: i for i, group in enumerate(["pbm", "fv", "tdnl", "kgcq", "szx", "jy", "aeiu", "ow", "hr"])
              for ch in group}


def lip_distance(a: str, b: str) -> float:
    """lookalike.ts `wordDistance` on normalised words: 0 identical .. 1 nothing alike."""
    a, b = norm(a), norm(b)
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        diag, prev[0] = prev[0], i
        for j in range(1, len(b) + 1):
            up = prev[j]
            ga = _LIP_GROUP.get(a[i - 1])
            cost = 0 if a[i - 1] == b[j - 1] else (0.3 if ga is not None and ga == _LIP_GROUP.get(b[j - 1]) else 1)
            prev[j] = min(prev[j] + 1, prev[j - 1] + 1, diag + cost)
            diag = up
    return prev[len(b)] / max(len(a), len(b), 1)


def looks_alike(old: str, new: str) -> bool:
    """Could the lips have shown `new` where the reader read `old`? Alike, completed or trimmed."""
    a, b = norm(old), norm(new)
    if not a or not b:
        return False
    if b.startswith(a) or b.endswith(a) or a.startswith(b) or a.endswith(b):
        return True
    return lip_distance(a, b) <= MAX_LIP_DISTANCE


def unsure(confidence: float | None, below: float = SURE_ABOVE) -> bool:
    """May this word change? `below` can be stricter than SURE_ABOVE, never looser."""
    return confidence is not None and confidence < min(below, SURE_ABOVE)


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


def _clipped(i: int, n: int, token: str, new: str) -> bool:
    """An edge word the cut clipped: first word → longer word ending in it, last → starting with it."""
    if n < 2:
        return False
    old, new = norm(token), norm(new)
    if not old or len(new) <= len(old):
        return False
    return (i == 0 and new.endswith(old)) or (i == n - 1 and new.startswith(old))


def plan_edits(tokens: list[str], out: list[str], conf: list[float | None],
               below: float = SURE_ABOVE) -> list[Edit] | str:
    """The edits that turn `tokens` into `out`, or why the answer breaks rule 1 (a str)."""
    n = len(tokens)
    if not out:
        return "empty answer"
    if len(out) != n:
        return f"answer has {len(out)} words, the line {n}"
    cut, same = align_words(tokens, out)
    reasons: list[str | None] = []
    for i in range(n):
        if cut[i + 1] - cut[i] != 1:
            return f"word {i} {tokens[i]!r} not replaced one for one"
        new = out[cut[i]]
        if same[i]:
            reasons.append(None)
        elif _clipped(i, n, tokens[i], new):
            reasons.append("clipped")
        elif not unsure(conf[i], below):
            return f"changed sure word {i} {tokens[i]!r} (confidence {conf[i]})"
        elif not looks_alike(tokens[i], new):
            return f"{tokens[i]!r} → {new!r} doesn't look alike on the lips"
        else:
            reasons.append("unsure")
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
