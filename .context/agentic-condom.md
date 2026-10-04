# The Agentic Condom: corrections between lip reading and speech (2026-10-04)

Branch `ml/agentic-condom`. It takes what the lip reader read and returns the sentence the user most
likely meant, before ElevenLabs speaks it. Normal and Quality only; Instant stays raw.

## Where it sits
`useLipReader.ts` `finish()`: reading → `expandClipped` → phrase memory (snap) → **Agentic Condom**
→ transcript line (spoken). One call per finished line, from `frontend/src/lib/agenticCondom/`
(`correctLine`), to our server's `POST /correct` (`ml/src/lipread/serve/app.py`), which runs
`lipread.agentic_condom.AgenticCondom` → `lipread.corrector.Corrector` (the thin OpenAI-compatible
client) → vLLM on the pod (`CONDOM=1 ml/runpod/serve.sh`, `ml/runpod/condom.sh`) or OpenRouter
(`CORRECTOR_API_KEY` alone, server side only). The browser never talks to the LLM or holds a key.

## What it gets, what it returns
Request (JSON, sent as `text/plain` so the browser skips the CORS preflight round trip):
`{text, words: [{text, confidence|null}], alternatives: [..], phrases: [{text, score|null}],
context: [{who: "user"|"other", text}], mode: "normal"|"quality"}`
- `text`: the line after phrase snapping; `words`: per-word confidence aligned to it (CTC frame
  probabilities on-device, n-best agreement for beam); `alternatives`: the beam's other readings
  (Quality); `phrases`: phrase-memory candidates, best first, with the model's CTC margin when it
  scored them; `context`: the last 6 lines of the conversation, the user's own lines and the other
  people's captions (STT), oldest first.

Response: `{text, raw, edits: [{start, end, from, to, reason}], status, model, latency_ms}`; edits
index `raw`'s words. `status`: corrected, unchanged, skipped (too short / nothing unsure), rejected
(failed the gate), timeout, error, off (no LLM configured). The old `{text}` call still answers.

## Hard rules, and where the code enforces them
| Rule | Enforced in |
|---|---|
| Only words below the confidence gate change (0.9, `SURE_ABOVE`, the `snapAllowed` gate); a word with no confidence (from a saved phrase or an expanded swear seed) counts as sure | `agentic_condom/gate.py` `plan_edits` on the server, re-checked by `agenticCondom/gate.ts` `checkCorrection` in the browser |
| Exception: an obviously clipped edge word: the first word may become a longer word ending in it ("APPENED" → "HAPPENED"), the last a longer word starting with it ("DOO" → "DOOR") | same |
| At most 1 word dropped, at most 2 words longer; anything else rejects the whole answer | same |
| Untouched words come back byte-for-byte (the line is rebuilt from the reading + edits, never taken from the LLM) | `gate.apply_edits` |
| Budget 500 ms Normal, 1 s Quality; timeout, error, server down, bad answer → the line as read | server: `condom.py` (`BUDGET_MS`, the LLM call's timeout); browser: `client.ts` aborts at the same budget and never throws |
| Instant bypasses it | `useLipReader.ts` (`plain`) |
| Never invent on empty or very short readings | fewer than 2 words, or no unsure word → no LLM call (`MIN_WORDS`) |
| Swearing is allowed, never softened | prompt + few-shot; the gate keeps sure words, and an expanded swear seed has no confidence, so it counts as sure |

The UI: changed words show in the amber box (`lineSegments`), with the original reading's words as
the first option (the pre-condom line is `choices[1]`), so one click undoes it. Toggle: "Agentic
Condom" in the lip-mode menu (`settings.ts`, localStorage `lipread.condom`).

## Prompt (`ml/src/lipread/agentic_condom/prompt.py`)
System prompt + 4 few-shot turns (fixed prefix, cached by vLLM's prefix cache), then one user turn:
the conversation, the other readings, the saved phrases, and the line in lowercase with the unsure
words in [brackets]. Lowercase: fewer tokens than the reader's capitals. The answer's first line
is parsed (labels, quotes and punctuation dropped), uppercased like the reading, then gated.

## Numbers
(filled in below by the eval)

## Deploy (serving pod)
On the serving pod, after this branch is merged (`git -C /workspace/stormhacks2026 pull`):
`CONDOM=1 bash /workspace/stormhacks2026/ml/runpod/serve.sh` installs vLLM 0.11.0 into
`/workspace/.venv-vllm` (once, ~5 min), downloads the model into `/workspace/.cache/hf` (once),
starts it on 127.0.0.1:8001 at `CONDOM_GPU_UTIL` (0.45 of the 4090) and restarts uvicorn with
`CORRECTOR_BASE_URL`/`CORRECTOR_MODEL` pointing at it. `/health` then shows
`"condom": {"enabled": true, "model": "condom"}`. If vLLM fails, the server starts without it and
`/correct` answers with the reading (`status: "off"`). Nothing changes in the frontend: the app
calls `/correct` on `VITE_LIPREAD_URL` (or `VITE_CONDOM_URL`). OpenRouter instead of vLLM:
put `CORRECTOR_API_KEY` in as a RunPod secret (`{{ RUNPOD_SECRET_… }}`) and set
`CORRECTOR_MODEL` (the URL defaults to OpenRouter when only the key is set); not measured here.
