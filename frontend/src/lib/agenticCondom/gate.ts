/**
 * Rule 1 of the Agentic Condom, checked here on every answer (never trust the server alone).
 *
 * A word may change only when the reader was unsure of it: a confidence below SURE_ABOVE, the same
 * bar as phrase snapping (`wordSpans.snapAllowed`). Unlike snapping, a word with no confidence counts
 * as sure: it came from a saved phrase or an expanded swear seed, i.e. was picked on purpose. The one
 * exception is an edge word the cut clipped: the first word may become one longer word that ends
 * with it ("APPENED" → "HAPPENED"), the last word one that starts with it ("DOO" → "DOOR"). At most
 * MAX_SNAP_DROPS words may go and the line may grow by at most MAX_GROWTH words. Anything else
 * rejects the whole answer. Twin of the server's `ml/src/lipread/agentic_condom/gate.py`.
 */
import { alignWords, MAX_SNAP_DROPS, splitWords, SURE_ABOVE } from "@/lib/lipreading/wordSpans"

/** An answer may be at most this many words longer than the line. */
export const MAX_GROWTH = 2

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

/** An edge word the cut clipped: the first word → one longer word ending in it, the last → starting with it. */
function clipped(i: number, n: number, token: string, replacement: readonly string[]): boolean {
  if (replacement.length !== 1 || n < 2) return false
  const old = norm(token)
  const next = norm(replacement[0])
  if (!old || next.length <= old.length) return false
  return (i === 0 && next.endsWith(old)) || (i === n - 1 && next.startsWith(old))
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
  if (out.length > n + MAX_GROWTH) return { ok: false, reason: `grew from ${n} to ${out.length} words` }
  const { cut, same } = alignWords(tokens, out)
  const dropped = tokens.filter((_, i) => cut[i + 1] === cut[i]).length
  if (dropped > MAX_SNAP_DROPS) return { ok: false, reason: `dropped ${dropped} words` }

  const reasons: (CondomEditReason | null)[] = []
  for (let i = 0; i < n; i++) {
    const changed = !same[i] || cut[i + 1] - cut[i] !== 1
    if (!changed) reasons.push(null)
    else if (unsure(confidence[i])) reasons.push("unsure")
    else if (clipped(i, n, tokens[i], out.slice(cut[i], cut[i + 1]))) reasons.push("clipped")
    else return { ok: false, reason: `changed sure word ${i} "${tokens[i]}" (${confidence[i] ?? "no confidence"})` }
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
