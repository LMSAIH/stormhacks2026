/**
 * The shared phrase bank: corrections other opted-in users typed or picked (never "accepted",
 * which is mostly the model's own output), counted across everyone by the lip-read server
 * (`GET /phrases/shared`). Merged into each reading's phrase candidates next to the user's own
 * phrases, so one person's fix helps everyone snap. Off when `VITE_LIPREAD_URL` is not set.
 */
import { MAX_PHRASES } from "@/lib/lipreading/httpRecognizer"
import { pairsBaseUrl } from "@/lib/lipreading/trainingPairs"

import { normalizeText } from "./lookalike"
import { SEED_HITS } from "./seeds"
import type { PhraseHit } from "./store"

/** Most shared phrases we use: the top ones by count. */
export const MAX_SHARED = 200
/** Longer shared texts are dropped: rare as phrases, slow to score. */
const MAX_SHARED_WORDS = 15
const REFRESH_MS = 5 * 60_000
const RETRY_MS = 60_000
const TIMEOUT_MS = 3_000

let cache: PhraseHit[] = []
let nextRefreshAt = 0
/** A fetch has finished (ok or not): later calls return the cache and refresh in the background. */
let settled = false
let inflight: Promise<PhraseHit[]> | null = null

export interface SharedPhraseOptions {
  /** Default `VITE_LIPREAD_URL`; "" turns the feature off. */
  readonly baseUrl?: string
  readonly fetchImpl?: typeof fetch
  readonly timeoutMs?: number
}

/**
 * The shared phrases as search hits, most used first. Never throws: a failed or slow (3 s) fetch
 * keeps the last good list (initially []). Refreshes at most every 5 minutes (1 minute after a
 * failure); only the very first fetch is waited on, later ones run in the background.
 */
export function getSharedPhrases(
  options: SharedPhraseOptions = {}
): Promise<PhraseHit[]> {
  const baseUrl = (options.baseUrl ?? pairsBaseUrl()).replace(/\/+$/, "")
  if (!baseUrl) return Promise.resolve([])
  if (!inflight && Date.now() >= nextRefreshAt)
    inflight = refresh(
      baseUrl,
      options.fetchImpl ?? fetch,
      options.timeoutMs ?? TIMEOUT_MS
    ).finally(() => {
      inflight = null
    })
  return !settled && inflight ? inflight : Promise.resolve(cache)
}

async function refresh(
  baseUrl: string,
  fetchImpl: typeof fetch,
  timeoutMs: number
): Promise<PhraseHit[]> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let ok = false
  try {
    const res = await fetchImpl(
      `${baseUrl}/phrases/shared?limit=${MAX_SHARED}`,
      { signal: controller.signal }
    )
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    cache = parseShared(await res.json())
    ok = true
  } catch (err) {
    console.warn("[phrases] shared phrase bank unavailable:", err)
  } finally {
    clearTimeout(timer)
    settled = true
    nextRefreshAt = Date.now() + (ok ? REFRESH_MS : RETRY_MS)
  }
  return cache
}

/** `{"phrases": [{"text", "count"}]}` → hits, deduplicated, top `MAX_SHARED` by count. */
function parseShared(json: unknown): PhraseHit[] {
  const list = (json as { phrases?: unknown } | null)?.phrases
  if (!Array.isArray(list)) throw new Error("bad /phrases/shared response")
  const seen = new Set<string>()
  const hits: PhraseHit[] = []
  for (const item of list as unknown[]) {
    const { text, count } = (item ?? {}) as { text?: unknown; count?: unknown }
    if (typeof text !== "string") continue
    const norm = normalizeText(text)
    // Long texts are rare as phrases and slow to score on the main thread.
    if (!norm || seen.has(norm) || norm.split(" ").length > MAX_SHARED_WORDS)
      continue
    seen.add(norm)
    hits.push({
      id: `shared:${norm}`,
      text: text.trim(),
      count:
        typeof count === "number" && count > 0 && Number.isFinite(count)
          ? count
          : 1,
      source: "picked", // confirmed by a person (typed or picked); the id prefix marks it shared
      lastUsed: 0,
    })
  }
  return hits.sort((a, b) => b.count - a.count).slice(0, MAX_SHARED)
}

/**
 * The user's own hits, then shared ones they don't already have (their own wins), adding shared
 * phrases only while the total stays within `max` (the server's `/lipread/phrases` limit). The
 * user's own hits are never dropped.
 */
export function mergeShared(
  own: readonly PhraseHit[],
  shared: readonly PhraseHit[],
  max = MAX_PHRASES
): PhraseHit[] {
  // Seeds keep their look-alike-only rule (useLipReader): a shared copy of a seed would be
  // model-scored, the way the model once snapped a garbled reading to the seed "shit".
  const have = new Set([...own, ...SEED_HITS].map((h) => normalizeText(h.text)))
  const extra = shared.filter((s) => !have.has(normalizeText(s.text)))
  return [...own, ...extra.slice(0, Math.max(0, max - own.length))]
}

/** Tests only: forget the cached list and the refresh schedule. */
export function resetSharedPhrases(): void {
  cache = []
  nextRefreshAt = 0
  settled = false
  inflight = null
}
