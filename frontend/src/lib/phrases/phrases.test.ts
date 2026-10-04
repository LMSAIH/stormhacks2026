import { describe, expect, it, vi } from "vitest"

import { lookalike, normalizeText, wordDistance } from "./lookalike"
import { rankChoices, SNAP_THRESHOLD } from "./snap"
import { HttpPhraseStore, type PhraseHit } from "./store"

const hit = (text: string, count = 1): PhraseHit => ({
  id: text,
  text,
  count,
  source: "accepted",
  lastUsed: 0,
})

describe("lookalike", () => {
  it("normalizes case and punctuation", () => {
    expect(normalizeText("  I'd like a coffee, please! ")).toBe("I'D LIKE A COFFEE PLEASE")
  })

  it("treats same-mouth-shape letters as close", () => {
    expect(wordDistance("BAT", "PAT")).toBeLessThan(wordDistance("BAT", "CAT"))
    expect(wordDistance("FAN", "VAN")).toBeCloseTo(0.1)
  })

  it("scores identical readings 1 and unrelated ones low", () => {
    expect(lookalike("CAN YOU REPEAT THAT", "can you repeat that?")).toBe(1)
    expect(lookalike("CAN YOU REPEAT THAT", "I LIKE TRAINS")).toBeLessThan(0.4)
  })

  it("forgives a lip-alike misread more than a different word", () => {
    expect(lookalike("I WANT A BAT", "I WANT A PAT")).toBeGreaterThan(
      lookalike("I WANT A BAT", "I WANT A CAT")
    )
  })
})

describe("rankChoices", () => {
  it("snaps to a saved phrase that looks close enough", () => {
    const r = rankChoices(["I WOULD LIKE A COPY"], [hit("I would like a coffee"), hit("Where is the bathroom")])
    expect(r.snap?.text).toBe("I would like a coffee")
    expect(r.choices[0].text).toBe("I would like a coffee")
    expect(r.choices.map((c) => c.text)).toContain("I WOULD LIKE A COPY")
  })

  it("does not snap below the threshold", () => {
    const r = rankChoices(["THE WEATHER IS NICE TODAY"], [hit("I would like a coffee")])
    expect(r.snap).toBeUndefined()
    expect(r.choices[0]).toMatchObject({ text: "THE WEATHER IS NICE TODAY", from: "reading" })
    expect(SNAP_THRESHOLD).toBeGreaterThan(0.5)
  })

  it("keeps beam order, drops duplicates, caps at 3", () => {
    const r = rankChoices(
      ["HELLO THERE", "HELLO THEY'RE", "HELLO THEIR"],
      [hit("hello there"), hit("HELL OF A DAY")]
    )
    expect(r.choices).toHaveLength(3)
    expect(new Set(r.choices.map((c) => normalizeText(c.text))).size).toBe(3)
    expect(r.snap).toBeUndefined() // the best phrase is the reading itself
  })

  it("lets a frequent phrase win a near-tie", () => {
    const r = rankChoices(["SEE YOU LATER"], [hit("see you later alligator", 1), hit("see you later mate", 50)])
    const phrases = r.choices.filter((c) => c.from === "phrase").map((c) => c.text)
    expect(phrases[0]).toBe("see you later mate")
  })
})

describe("HttpPhraseStore", () => {
  it("sends the backend spec's requests", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("[]", { status: 200 }))
    const store = new HttpPhraseStore("https://phrases.example/", "u1", fetchImpl)
    await store.search("HELLO THERE", 5)
    await store.add("Hello there", "picked")
    const [searchUrl] = fetchImpl.mock.calls[0]
    const [addUrl, addInit] = fetchImpl.mock.calls[1]
    expect(searchUrl).toBe("https://phrases.example/phrases/search?user_id=u1&q=HELLO+THERE&k=5")
    expect(addUrl).toBe("https://phrases.example/phrases")
    expect(JSON.parse(addInit?.body as string)).toEqual({ user_id: "u1", text: "Hello there", source: "picked" })
  })
})
