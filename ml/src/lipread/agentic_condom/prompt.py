"""The Agentic Condom's prompt: one short system prompt + few-shot turns (cached by vLLM's prefix
cache), then one user turn per line. The line goes in lowercase (fewer tokens than the reader's
capitals, and closer to the text LLMs read) with the unsure words in [brackets]."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You fix one line read by a lip-reading model. The person cannot speak aloud; the line will be \
spoken in their voice.
Words in [brackets] are ones the lip reader was unsure of. Every other word is certain: copy it \
exactly.
For each bracketed word: keep it, or replace it with the word the person most likely mouthed, one \
that looks alike on the lips (p/b/m, f/v, t/d/n/l, s/z, k/g, ch/j/sh, w/r, and most vowels look \
the same) and makes the line make sense in this conversation. The other readings and the \
person's saved phrases are hints. If a bracketed word already fits, keep it.
You may complete a word cut off at the very start or end of the line.
Never add or drop other words, never rephrase, never censor or soften swearing.
Answer with the corrected line only, in lowercase, without brackets or punctuation."""

# (user turn, assistant turn): fix two unsure words, keep a fitting one, keep the swearing.
FEW_SHOT: list[tuple[str, str]] = [
    ("Line: i [thing] you are [wight]", "i think you are right"),
    ("Conversation:\nOther: Do you want something to drink?\nLine: can i have some [mater] [plead]",
     "can i have some water please"),
    ("Line: that is [exactly] what happened", "that is exactly what happened"),
    ("Line: what the [fuck] is [going] on", "what the fuck is going on"),
]

MAX_CONTEXT = 6
MAX_ALTERNATIVES = 2
MAX_PHRASES = 3
MAX_CHARS = 200


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"


def user_turn(tokens: list[str], flagged: list[bool], alternatives: list[str], phrases: list[str],
              context: list[tuple[str, str]]) -> str:
    """The user turn for one line; `flagged[i]` = tokens[i] goes in brackets."""
    parts: list[str] = []
    if context:
        lines = [f"{'Me' if who == 'user' else 'Other'}: {_clip(text)}" for who, text in context[-MAX_CONTEXT:]]
        parts.append("Conversation:\n" + "\n".join(lines))
    line = " ".join(f"[{t.lower()}]" if f else t.lower() for t, f in zip(tokens, flagged))
    raw = " ".join(tokens).lower()
    alts = [a.lower() for a in dict.fromkeys(_clip(a) for a in alternatives) if a.lower() != raw]
    if alts:
        parts.append("Other readings: " + " | ".join(alts[:MAX_ALTERNATIVES]))
    if phrases:
        parts.append("Saved phrases: " + " | ".join(_clip(p).lower() for p in phrases[:MAX_PHRASES]))
    parts.append(f"Line: {line}")
    return "\n".join(parts)


def build_messages(tokens: list[str], flagged: list[bool], alternatives: list[str], phrases: list[str],
                   context: list[tuple[str, str]]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEW_SHOT:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": user_turn(tokens, flagged, alternatives, phrases, context)})
    return messages


def max_tokens(n_words: int) -> int:
    """Room for the corrected line (lowercase words are ~1.3 tokens) and nothing much more."""
    return 2 * n_words + 8


_LABEL = re.compile(r"^\s*(corrected line|corrected|line|answer|output)\s*:\s*", re.IGNORECASE)
_NOT_WORD = re.compile(r"[^A-Za-z0-9' ]+")


def parse_answer(answer: str, upper: bool) -> list[str]:
    """The answer's words: first non-empty line, labels/quotes/brackets/punctuation dropped."""
    line = next((ln for ln in answer.splitlines() if ln.strip()), "")
    line = _LABEL.sub("", line).replace("’", "'")
    words = [w.strip("'") for w in _NOT_WORD.sub(" ", line).split()]
    words = [w for w in words if w]
    return [w.upper() for w in words] if upper else words
