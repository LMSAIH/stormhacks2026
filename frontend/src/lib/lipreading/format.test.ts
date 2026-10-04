import { describe, expect, it } from "vitest"

import { formatLatency, toSentenceCase } from "./format"

describe("formatLatency", () => {
  it("uses ms under a second and seconds above", () => {
    expect(formatLatency(0)).toBe("0 ms")
    expect(formatLatency(849.6)).toBe("850 ms")
    expect(formatLatency(999)).toBe("999 ms")
    expect(formatLatency(1000)).toBe("1.00 s")
    expect(formatLatency(2345)).toBe("2.35 s")
  })
})

describe("toSentenceCase", () => {
  it("reads uppercase model output as a sentence", () => {
    expect(toSentenceCase("WHAT ARE YOU DOING TODAY")).toBe(
      "What are you doing today"
    )
  })

  it("keeps the pronoun I (and contractions) capitalised", () => {
    expect(toSentenceCase("I THINK I'M READY AND I'LL GO")).toBe(
      "I think I'm ready and I'll go"
    )
  })

  it("leaves text that already has lowercase letters alone", () => {
    expect(toSentenceCase("Hello there")).toBe("Hello there")
    expect(toSentenceCase("hello")).toBe("hello")
  })

  it("handles empty and non-letter text", () => {
    expect(toSentenceCase("")).toBe("")
    expect(toSentenceCase("123")).toBe("123")
  })
})
