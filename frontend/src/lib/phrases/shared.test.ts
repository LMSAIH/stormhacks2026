import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  getSharedPhrases,
  MAX_SHARED,
  mergeShared,
  resetSharedPhrases,
} from "./shared"
import type { PhraseHit } from "./store"

const BASE = "https://pod.example/" // trailing slash must be stripped

const hit = (text: string, count = 1, id = text): PhraseHit => ({
  id,
  text,
  count,
  source: "typed",
  lastUsed: 0,
})

const shared = (text: string, count = 1) =>
  hit(text, count, `shared:${text.toUpperCase()}`)

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  })
}

/** A fetch that never answers, but rejects like the real one when its signal aborts. */
function hangingFetch(
  _url: string | URL | Request,
  init?: RequestInit
): Promise<Response> {
  return new Promise((_, reject) => {
    init?.signal?.addEventListener("abort", () =>
      reject(new DOMException("aborted", "AbortError"))
    )
  })
}

describe("mergeShared", () => {
  it("never adds a shared copy of a seed (seeds keep their look-alike-only rule)", () => {
    const merged = mergeShared(
      [hit("call my mom")],
      [
        hit("FUCK YOU", 9, "shared:FUCK YOU"),
        hit("I NEED WATER", 2, "shared:I NEED WATER"),
      ]
    )
    expect(merged.map((h) => h.text)).toEqual(["call my mom", "I NEED WATER"])
  })

  it("keeps the user's own phrase when the shared bank has the same words", () => {
    const merged = mergeShared(
      [hit("I'd like a coffee", 2)],
      [shared("I'D LIKE A COFFEE", 9), shared("THANK YOU", 3)]
    )
    expect(merged.map((h) => h.text)).toEqual([
      "I'd like a coffee",
      "THANK YOU",
    ])
    expect(merged[0].id).toBe("I'd like a coffee")
  })

  it("adds shared phrases only up to the cap, never dropping the user's own", () => {
    const own = [hit("a"), hit("b")]
    const bank = [shared("c"), shared("d"), shared("e")]
    expect(mergeShared(own, bank, 3).map((h) => h.text)).toEqual([
      "a",
      "b",
      "c",
    ])
    expect(mergeShared(own, bank, 1).map((h) => h.text)).toEqual(["a", "b"])
    expect(mergeShared([], bank).map((h) => h.text)).toEqual(["c", "d", "e"])
  })
})

describe("getSharedPhrases", () => {
  beforeEach(() => {
    resetSharedPhrases()
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("is off without a server URL", async () => {
    const fetchImpl = vi.fn<typeof fetch>()
    await expect(getSharedPhrases({ baseUrl: "", fetchImpl })).resolves.toEqual(
      []
    )
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("drops shared texts over 15 words", async () => {
    const long = Array.from({ length: 16 }, (_, i) => `word${i}`).join(" ")
    const fetchImpl = vi.fn<typeof fetch>(async () =>
      json({
        phrases: [
          { text: long, count: 9 },
          { text: "I NEED WATER", count: 1 },
        ],
        updated: 1,
      })
    )
    const hits = await getSharedPhrases({ baseUrl: BASE, fetchImpl })
    expect(hits.map((h) => h.text)).toEqual(["I NEED WATER"])
  })

  it("maps the server's list to hits, top by count, and caches it", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () =>
      json({
        phrases: [
          { text: "THANK YOU", count: 2 },
          { text: "WHERE IS THE BATHROOM", count: 5 },
          { text: "thank you!", count: 1 }, // same words: dropped
          { text: "", count: 4 },
          { count: 3 },
        ],
        updated: 1,
      })
    )
    const hits = await getSharedPhrases({ baseUrl: BASE, fetchImpl })
    expect(String(fetchImpl.mock.calls[0][0])).toBe(
      `https://pod.example/phrases/shared?limit=${MAX_SHARED}`
    )
    expect(hits).toEqual([
      {
        id: "shared:WHERE IS THE BATHROOM",
        text: "WHERE IS THE BATHROOM",
        count: 5,
        source: "picked",
        lastUsed: 0,
      },
      {
        id: "shared:THANK YOU",
        text: "THANK YOU",
        count: 2,
        source: "picked",
        lastUsed: 0,
      },
    ])
    await getSharedPhrases({ baseUrl: BASE, fetchImpl })
    expect(fetchImpl).toHaveBeenCalledTimes(1) // within 5 minutes: cached
  })

  it("caps the list at the top MAX_SHARED by count", async () => {
    const phrases = Array.from({ length: MAX_SHARED + 50 }, (_, i) => ({
      text: `PHRASE ${i}`,
      count: i + 1,
    }))
    const hits = await getSharedPhrases({
      baseUrl: BASE,
      fetchImpl: async () => json({ phrases }),
    })
    expect(hits).toHaveLength(MAX_SHARED)
    expect(hits[0].count).toBe(MAX_SHARED + 50)
  })

  it("keeps the last good list when a later fetch fails", async () => {
    vi.useFakeTimers()
    const ok = await getSharedPhrases({
      baseUrl: BASE,
      fetchImpl: async () =>
        json({ phrases: [{ text: "HELLO THERE", count: 2 }] }),
    })
    expect(ok.map((h) => h.text)).toEqual(["HELLO THERE"])

    vi.advanceTimersByTime(5 * 60_000)
    const failing = vi.fn<typeof fetch>(async () =>
      json({ detail: "boom" }, 500)
    )
    // Stale: refreshes in the background and answers with the cached list meanwhile.
    await expect(
      getSharedPhrases({ baseUrl: BASE, fetchImpl: failing })
    ).resolves.toEqual(ok)
    await vi.runAllTimersAsync()
    expect(failing).toHaveBeenCalledTimes(1)
    await expect(
      getSharedPhrases({ baseUrl: BASE, fetchImpl: failing })
    ).resolves.toEqual(ok)
  })

  it("gives up after the timeout and never throws", async () => {
    vi.useFakeTimers()
    const pending = getSharedPhrases({ baseUrl: BASE, fetchImpl: hangingFetch })
    await vi.advanceTimersByTimeAsync(3_000)
    await expect(pending).resolves.toEqual([])

    vi.advanceTimersByTime(60_000) // a failure retries sooner than the 5-minute refresh
    const rejecting = vi.fn<typeof fetch>(async () => {
      throw new TypeError("network down")
    })
    await expect(
      getSharedPhrases({ baseUrl: BASE, fetchImpl: rejecting })
    ).resolves.toEqual([])
    expect(rejecting).toHaveBeenCalledTimes(1)
  })
})
