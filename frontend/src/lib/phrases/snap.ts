import { lookalike, normalizeText } from "./lookalike"
import type { PhraseHit } from "./store"

/** One option offered under a locked line. */
export interface Choice {
  readonly text: string
  /** 0..1 how alike it looks to what was read. */
  readonly similarity: number
  readonly from: "reading" | "phrase"
}

export interface Ranked {
  /** Up to 3 distinct options, best first, for the one-tap picker. */
  readonly choices: readonly Choice[]
  /** A saved phrase close enough to replace the reading outright, if any. */
  readonly snap?: Choice
}

/** A saved phrase replaces the reading only when it looks at least this alike. */
export const SNAP_THRESHOLD = 0.75

/**
 * Rank what to offer for one reading: the recognizer's own readings (beam top 3, or just the text
 * in speed mode) plus saved phrases, by how alike they look on the lips. Phrases used more often
 * get a small boost, so a frequent phrase wins a near-tie.
 */
export function rankChoices(
  readings: readonly string[],
  phrases: readonly PhraseHit[],
  limit = 3
): Ranked {
  const best = readings[0] ?? ""
  const scored: (Choice & { rank: number })[] = []
  readings.forEach((text, i) => {
    // Beam order already reflects the language model; keep it ahead of a tie.
    scored.push({ text, similarity: lookalike(best, text), from: "reading", rank: 1 - i * 0.01 })
  })
  for (const p of phrases) {
    const similarity = Math.max(...readings.map((r) => lookalike(r, p.text)), 0)
    scored.push({
      text: p.text,
      similarity,
      from: "phrase",
      rank: similarity + Math.min(0.05, 0.01 * Math.log2(1 + p.count)),
    })
  }
  scored.sort((a, b) => b.rank - a.rank)

  const seen = new Set<string>()
  const choices: Choice[] = []
  for (const s of scored) {
    const key = normalizeText(s.text)
    if (!key || seen.has(key)) continue
    seen.add(key)
    choices.push({ text: s.text, similarity: s.similarity, from: s.from })
    if (choices.length === limit) break
  }

  const phraseHit = choices.find((c) => c.from === "phrase")
  const snap =
    phraseHit && phraseHit.similarity >= SNAP_THRESHOLD && normalizeText(phraseHit.text) !== normalizeText(best)
      ? phraseHit
      : undefined
  return { choices: snap ? [snap, ...choices.filter((c) => c !== snap)] : choices, snap }
}
