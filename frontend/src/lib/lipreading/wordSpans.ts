import type { WordConfidence } from "./types"

/**
 * Which words of a finished line get a box, and what each box offers.
 *
 * A word is flagged when the reader was unsure of it (per-word confidence below FLAG_BELOW,
 * calibrated in ml/scripts/calibrate_conf.py: catches ~half the misread words while boxing ~4-8%
 * of right ones), or when it has no confidence (e.g. it came from a saved phrase) and another
 * reading disagrees there. Flagged words grow into spans covering where the other readings differ,
 * so a box holds one word or a short phrase, never the confident rest of the sentence.
 */

/** Per-word confidence below this gets a box. */
export const FLAG_BELOW = 0.6
/** A box only grows over a neighbouring word the reader was less sure of than this. */
export const SURE_ABOVE = 0.9

export type { WordConfidence } from "./types"

export interface LineSegment {
  /** Word range [start, end) of the line. */
  readonly start: number
  readonly end: number
  readonly text: string
  readonly flagged: boolean
  /** Flagged spans: the current words first, then what other readings say here. */
  readonly options: readonly string[]
}

export const splitWords = (text: string): string[] => text.split(/\s+/).filter(Boolean)
const norm = (w: string) => w.toLowerCase().replace(/[^a-z0-9']/g, "")

/**
 * Word alignment of `a` to `b` (edit distance on normalised words). Returns `cut`, where
 * cut[i] is the index in `b` where a[i]'s counterpart starts (cut[a.length] = b.length); extra
 * words in `b` attach to the preceding word of `a`. `same[i]` = a[i] matched a word of b.
 */
export function alignWords(a: readonly string[], b: readonly string[]) {
  const n = a.length
  const m = b.length
  const na = a.map(norm)
  const nb = b.map(norm)
  const d: number[][] = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  )
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      d[i][j] = Math.min(
        d[i - 1][j - 1] + (na[i - 1] === nb[j - 1] ? 0 : 1),
        d[i - 1][j] + 1,
        d[i][j - 1] + 1
      )
  // Walk back: record for each a-word the b-index it starts at.
  const cut = new Array<number>(n + 1).fill(0)
  const same = new Array<boolean>(n).fill(false)
  cut[n] = m
  let i = n
  let j = m
  while (i > 0) {
    if (j > 0 && d[i][j] === d[i - 1][j - 1] + (na[i - 1] === nb[j - 1] ? 0 : 1)) {
      same[i - 1] = na[i - 1] === nb[j - 1]
      j--
    } else if (j > 0 && d[i][j] === d[i][j - 1] + 1) {
      j-- // extra b-word: stays inside the current a-word's range
      continue
    }
    i--
    cut[i] = j
  }
  cut[0] = 0 // leading extra b-words attach to the first a-word
  return { cut, same }
}

/** Map a reading's per-word confidences onto `text`'s words; unmatched words get null. */
export function confidenceFor(
  text: string,
  words: readonly WordConfidence[] | undefined
): (number | null)[] {
  const tokens = splitWords(text)
  if (!words?.length) return tokens.map(() => null)
  const { cut, same } = alignWords(tokens, words.map((w) => w.text))
  return tokens.map((_, i) => (same[i] ? (words[cut[i]]?.confidence ?? null) : null))
}

/** Split a line into plain and flagged segments (see the module comment). */
export function lineSegments(
  text: string,
  confidence: readonly (number | null)[] | undefined,
  alternatives: readonly string[]
): LineSegment[] {
  const tokens = splitWords(text)
  const n = tokens.length
  if (n === 0) return []
  const conf = tokens.map((_, i) => confidence?.[i] ?? null)
  const alts = alternatives
    .map(splitWords)
    .filter((w) => w.join(" ").toLowerCase() !== text.toLowerCase())
    .map((words) => ({ words, ...alignWords(tokens, words) }))
    // Another reading of the same words keeps most of them; a mostly different text (e.g. an
    // unrelated saved phrase) says nothing about which words are shaky.
    .filter((a) => a.same.filter(Boolean).length * 2 >= n)

  // A word differs when some reading puts something else (or extra words) in its slot.
  const differs = tokens.map((_, i) =>
    alts.some((a) => !a.same[i] || a.cut[i + 1] - a.cut[i] !== 1)
  )
  const flag = tokens.map((_, i) => {
    const c = conf[i]
    return c === null ? differs[i] : c < FLAG_BELOW
  })

  // Grow each flagged word over neighbouring words the other readings disagree on, but never
  // over a word the reader was sure of.
  const growable = (k: number) => differs[k] && !flag[k] && (conf[k] ?? 0) < SURE_ABOVE
  for (let i = 0; i < n; i++) {
    if (!flag[i]) continue
    for (let k = i - 1; k >= 0 && growable(k); k--) flag[k] = true
    for (let k = i + 1; k < n && growable(k); k++) flag[k] = true
  }

  const out: LineSegment[] = []
  let s = 0
  while (s < n) {
    let e = s + 1
    while (e < n && flag[e] === flag[s]) e++
    const words = tokens.slice(s, e).join(" ")
    const options = [words]
    if (flag[s]) {
      for (const a of alts) {
        const alt = a.words.slice(a.cut[s], a.cut[e]).join(" ")
        if (alt && !options.some((o) => o.toLowerCase() === alt.toLowerCase())) options.push(alt)
      }
    }
    out.push({ start: s, end: e, text: words, flagged: flag[s], options })
    s = e
  }
  return out
}

/** The line with words [start, end) replaced (or removed when `replacement` is ""). */
export function replaceSpan(text: string, start: number, end: number, replacement: string): string {
  const tokens = splitWords(text)
  return [...tokens.slice(0, start), ...splitWords(replacement), ...tokens.slice(end)].join(" ")
}

/** Confidences for the line after `replaceSpan`: the edited words are now certain (1). */
export function replaceSpanConfidence(
  confidence: readonly (number | null)[],
  start: number,
  end: number,
  replacement: string
): (number | null)[] {
  return [
    ...confidence.slice(0, start),
    ...splitWords(replacement).map(() => 1),
    ...confidence.slice(end),
  ]
}

/**
 * May a saved phrase replace this reading? Only where the reader was unsure: every word the phrase
 * would change must be below SURE_ABOVE (or have no confidence). Without this, a saved line that
 * merely looks alike overrode correct readings ("DOGS ARE SITTING BY THE DOOR" → "Kids are
 * talking by the door").
 */
export function snapAllowed(
  reading: string,
  phrase: string,
  words: readonly WordConfidence[] | undefined
): boolean {
  const tokens = splitWords(reading)
  const conf = confidenceFor(reading, words)
  const { cut, same } = alignWords(tokens, splitWords(phrase))
  return tokens.every((_, i) => {
    const changed = !same[i] || cut[i + 1] - cut[i] !== 1
    const c = conf[i]
    return !changed || c === null || c < SURE_ABOVE
  })
}
