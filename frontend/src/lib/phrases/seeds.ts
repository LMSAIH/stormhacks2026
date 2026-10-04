/**
 * Built-in phrase memory: swearing. The lip-reading model learned from TED-talk transcripts, so
 * its vocabulary has no "fuck", "shit", "hell", "bitch" or "ass" (only "damn"), and it reads them
 * as the nearest clean words or a clipped fragment ("fuck you" → "FU"). These seeds put them back:
 * - phrases join every phrase search, so a whole utterance that looks like one snaps to it;
 * - `expandClipped` turns fragments that are not English words into the word they cut off;
 * - `swearLookalikes` offers a swear word in a word's fix popup when it looks alike on the lips.
 * Normal and Quality only: Instant keeps the reading exactly as read.
 */
import { lookalike, normalizeText, wordDistance } from "./lookalike"
import type { PhraseHit } from "./store"

export const SEED_PHRASES: readonly string[] = [
  "fuck",
  "fuck you",
  "fuck off",
  "fuck this",
  "fuck that",
  "fuck yeah",
  "what the fuck",
  "shut the fuck up",
  "are you fucking kidding me",
  "shit",
  "oh shit",
  "holy shit",
  "piece of shit",
  "bullshit",
  "damn it",
  "god damn it",
  "go to hell",
  "what the hell",
  "son of a bitch",
  "asshole",
]

/** Seed phrases as search hits (count 1: no frequency boost over the user's own phrases). */
export const SEED_HITS: readonly PhraseHit[] = SEED_PHRASES.map((text) => ({
  id: `seed:${normalizeText(text)}`,
  text,
  count: 1,
  source: "accepted",
  lastUsed: 0,
}))

/** The user's hits plus the seeds, without duplicating a phrase the user already saved. */
export function withSeeds(hits: readonly PhraseHit[]): PhraseHit[] {
  const have = new Set(hits.map((h) => normalizeText(h.text)))
  return [...hits, ...SEED_HITS.filter((s) => !have.has(normalizeText(s.text)))]
}

/** Fragments the model emits for a swear word it can't spell; none is an English word. */
const CLIPPED: Readonly<Record<string, string>> = {
  fu: "fuck",
  fuk: "fuck",
  fuc: "fuck",
  fuckin: "fucking",
}

/** Replace clipped fragments with the word they cut off, keeping the rest of the text as is. */
export function expandClipped(text: string): string {
  return text.replace(/[A-Za-z']+/g, (word) => {
    const full = CLIPPED[word.toLowerCase()]
    if (!full) return word
    return word === word.toUpperCase() ? full.toUpperCase() : full
  })
}

// "damn" is left out: the model already has it.
const SWEAR_WORDS = ["fuck", "fucking", "shit", "hell", "bitch", "ass", "asshole", "bullshit"]
/** A swear word is offered for a read word at most this far apart on the lips (0..1). */
const OFFER_WITHIN = 0.5

/** Swear words that look like `word` on the lips, closest first (for the word's fix popup). */
export function swearLookalikes(word: string): string[] {
  const w = normalizeText(word)
  if (!w || w.includes(" ")) return []
  return SWEAR_WORDS.filter((s) => s !== w)
    .map((s) => ({ s, d: wordDistance(w, s) }))
    .filter(({ d }) => d <= OFFER_WITHIN)
    .sort((a, b) => a.d - b.d)
    .map(({ s }) => s)
}

/** Exposed for tests: how alike a reading is to a seed phrase. */
export const seedSimilarity = (reading: string, seed: string) => lookalike(reading, seed)
