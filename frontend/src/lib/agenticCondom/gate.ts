/**
 * Rule 1 of the Agentic Condom, checked here on every answer (never trust the server alone).
 *
 * A word may change only when the reader was unsure of it: a confidence below SURE_ABOVE, the same
 * bar as phrase snapping (`wordSpans.snapAllowed`). Unlike snapping, a word with no confidence counts
 * as sure: it came from a saved phrase or an expanded swear seed, i.e. was picked on purpose. And it
 * may only become ONE word that looks like it on the lips (`wordDistance` ≤ MAX_LIP_DISTANCE) or that
 * completes or trims it ("LU" → "LUTHER"): the line keeps its word count. The one exception for sure
 * words is an edge word the cut clipped: the first word may become one longer word that ends with it
 * ("APPENED" → "HAPPENED"), the last word one that starts with it ("DOO" → "DOOR"). Anything else
 * rejects the whole answer. Twin of the server's `ml/src/lipread/agentic_condom/gate.py`
 * (parity fixture: `__fixtures__/gate.json`).
 */
import { alignWords, splitWords, SURE_ABOVE } from "@/lib/lipreading/wordSpans"
import { wordDistance } from "@/lib/phrases/lookalike"

/** A replacement must look this alike on the lips (`wordDistance`: 0 same letters .. 1 nothing alike). */
export const MAX_LIP_DISTANCE = 0.4
/**
 * The condom only asks about words under this confidence (the server brackets the same set,
 * `condom.FLAG_BELOW`): at 0.9 the LLM broke right words in 8.5% of lines on the dev set.
 */
export const CONDOM_FLAG_BELOW = 0.6

export type CondomEditReason = "unsure" | "clipped"

/** Words [start, end) of the line became `to` ("" = deleted). */
export interface CondomEdit {
  readonly start: number
  readonly end: number
  readonly from: string
  readonly to: string
  readonly reason: CondomEditReason
}

export type CondomVerdict =
  | { readonly ok: true; readonly edits: CondomEdit[] }
  | { readonly ok: false; readonly reason: string }

const norm = (w: string) => w.toLowerCase().replace(/[^a-z0-9']/g, "")

/** The condom may change a word only below SURE_ABOVE; no confidence (null) counts as sure. */
export const unsure = (confidence: number | null | undefined): boolean =>
  typeof confidence === "number" && confidence < SURE_ABOVE

/** A word the condom asks the LLM about (below CONDOM_FLAG_BELOW). */
export const bracketed = (confidence: number | null | undefined): boolean =>
  typeof confidence === "number" && confidence < CONDOM_FLAG_BELOW

/** Could the lips have shown `next` where the reader read `old`? Alike, completed or trimmed. */
export function looksAlike(old: string, next: string): boolean {
  const a = norm(old)
  const b = norm(next)
  if (!a || !b) return false
  if (b.startsWith(a) || b.endsWith(a) || a.startsWith(b) || a.endsWith(b)) return true
  return wordDistance(a, b) <= MAX_LIP_DISTANCE
}

/** An edge word the cut clipped: the first word → a longer word ending in it, the last → starting with it. */
function clipped(i: number, n: number, token: string, next: string): boolean {
  if (n < 2) return false
  const old = norm(token)
  const longer = norm(next)
  if (!old || longer.length <= old.length) return false
  return (i === 0 && longer.endsWith(old)) || (i === n - 1 && longer.startsWith(old))
}

/**
 * May `corrected` replace `raw`? `confidence` is per word of `raw`, as sent. Case and punctuation
 * alone don't count as changes. The edits are `raw`'s changed runs, for `applyEdits`.
 */
export function checkCorrection(
  raw: string,
  corrected: string,
  confidence: readonly (number | null)[]
): CondomVerdict {
  const tokens = splitWords(raw)
  const out = splitWords(corrected)
  const n = tokens.length
  if (n === 0 || out.length === 0) return { ok: false, reason: "empty line or answer" }
  if (out.length !== n) return { ok: false, reason: `answer has ${out.length} words, the line ${n}` }
  const { cut, same } = alignWords(tokens, out)

  const reasons: (CondomEditReason | null)[] = []
  for (let i = 0; i < n; i++) {
    if (cut[i + 1] - cut[i] !== 1) return { ok: false, reason: `word ${i} "${tokens[i]}" not replaced one for one` }
    const next = out[cut[i]]
    if (same[i]) reasons.push(null)
    else if (clipped(i, n, tokens[i], next)) reasons.push("clipped")
    else if (!unsure(confidence[i]))
      return { ok: false, reason: `changed sure word ${i} "${tokens[i]}" (${confidence[i] ?? "no confidence"})` }
    else if (!looksAlike(tokens[i], next))
      return { ok: false, reason: `"${tokens[i]}" → "${next}" doesn't look alike on the lips` }
    else reasons.push("unsure")
  }

  const edits: CondomEdit[] = []
  for (let i = 0; i < n; ) {
    if (reasons[i] === null) {
      i++
      continue
    }
    const start = i
    while (i < n && reasons[i] !== null) i++
    edits.push({
      start,
      end: i,
      from: tokens.slice(start, i).join(" "),
      to: out.slice(cut[start], cut[i]).join(" "),
      reason: reasons.slice(start, i).every((r) => r === "clipped") ? "clipped" : "unsure",
    })
  }
  return { ok: true, edits }
}

const byStart = (edits: readonly CondomEdit[]) => [...edits].sort((a, b) => a.start - b.start)

/** The line with each edit applied; every other word stays exactly as it was. */
export function applyEdits(raw: string, edits: readonly CondomEdit[]): string {
  const tokens = splitWords(raw)
  const out: string[] = []
  let i = 0
  for (const e of byStart(edits)) {
    out.push(...tokens.slice(i, e.start), ...splitWords(e.to))
    i = e.end
  }
  return [...out, ...tokens.slice(i)].join(" ")
}

/** Word indices of the corrected line (`applyEdits`) that the edits put there. */
export function editedWords(edits: readonly CondomEdit[]): number[] {
  const words: number[] = []
  let shift = 0
  for (const e of byStart(edits)) {
    const added = splitWords(e.to).length
    for (let k = 0; k < added; k++) words.push(e.start + shift + k)
    shift += added - (e.end - e.start)
  }
  return words
}
