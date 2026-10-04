/**
 * How alike two readings look on the lips. Lip reading confuses sounds that share a mouth shape
 * (p/b/m, f/v, t/d/n, k/g, s/z, ...), so a misread "PAT" for "BAT" should count as close while
 * "CAT" for "BAT" should not. Text only (no phonemes): letters stand in for sounds, which is rough
 * but good enough to rank a short phrase list.
 */

/** Letters that look the same on the lips; a swap inside a group is cheap. */
const LIP_GROUPS = ["pbm", "fv", "tdnl", "kgcq", "szx", "jy", "aeiu", "ow", "hr"]
const GROUP_OF = new Map<string, number>()
LIP_GROUPS.forEach((group, i) => {
  for (const ch of group) GROUP_OF.set(ch, i)
})
const SAME_SHAPE_COST = 0.3

/** Uppercase words, punctuation dropped: what the model emits and what we store. */
export function normalizeText(text: string): string {
  return text
    .toUpperCase()
    .replace(/[^A-Z0-9' ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function letterCost(a: string, b: string): number {
  if (a === b) return 0
  const ga = GROUP_OF.get(a.toLowerCase())
  return ga !== undefined && ga === GROUP_OF.get(b.toLowerCase()) ? SAME_SHAPE_COST : 1
}

/** Edit distance between two words with cheap same-mouth-shape swaps, 0..1 (0 = identical). */
export function wordDistance(a: string, b: string): number {
  if (a === b) return 0
  const prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0]
    prev[0] = i
    for (let j = 1; j <= b.length; j++) {
      const up = prev[j]
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + letterCost(a[i - 1], b[j - 1]))
      diag = up
    }
  }
  return prev[b.length] / Math.max(a.length, b.length, 1)
}

/**
 * Word-level edit distance between two readings, each word swap priced by `wordDistance`, scaled to
 * 0..1 by the longer reading's word count. 0 = same words.
 */
export function readingDistance(a: string, b: string): number {
  const x = normalizeText(a).split(" ").filter(Boolean)
  const y = normalizeText(b).split(" ").filter(Boolean)
  if (x.length === 0 && y.length === 0) return 0
  const prev = Array.from({ length: y.length + 1 }, (_, j) => j)
  for (let i = 1; i <= x.length; i++) {
    let diag = prev[0]
    prev[0] = i
    for (let j = 1; j <= y.length; j++) {
      const up = prev[j]
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + wordDistance(x[i - 1], y[j - 1]))
      diag = up
    }
  }
  return prev[y.length] / Math.max(x.length, y.length)
}

/** 0..1 similarity, 1 = same words. */
export function lookalike(a: string, b: string): number {
  return 1 - readingDistance(a, b)
}
