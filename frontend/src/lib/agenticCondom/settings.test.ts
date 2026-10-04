import { afterEach, describe, expect, it, vi } from "vitest"

import { conversationContext, MAX_CONTEXT_CHARS, setConversationContext } from "./context"
import { CONDOM_DEFAULT_ON, condomEnabled, setCondomEnabled } from "./settings"

function memoryStorage() {
  const items = new Map<string, string>()
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
  }
}

afterEach(() => vi.unstubAllGlobals())

describe("the condom switch", () => {
  it("starts at the default and remembers the choice as lipread.condom", () => {
    const storage = memoryStorage()
    vi.stubGlobal("localStorage", storage)
    expect(condomEnabled()).toBe(CONDOM_DEFAULT_ON)
    setCondomEnabled(false)
    expect(storage.getItem("lipread.condom")).toBe("0")
    expect(condomEnabled()).toBe(false)
    setCondomEnabled(true)
    expect(storage.getItem("lipread.condom")).toBe("1")
    expect(condomEnabled()).toBe(true)
  })

  it("still switches for this session when storage is blocked", () => {
    const blocked = () => {
      throw new DOMException("blocked", "SecurityError")
    }
    vi.stubGlobal("localStorage", { getItem: blocked, setItem: blocked })
    setCondomEnabled(false)
    expect(condomEnabled()).toBe(false)
    setCondomEnabled(true)
    expect(condomEnabled()).toBe(true)
  })
})

describe("conversation context", () => {
  it("keeps the last 6 lines in time order, without their times", () => {
    setConversationContext([
      ...[0, 10, 20, 30, 40].map((at) => ({ who: "user" as const, text: `line ${at}`, at })),
      { who: "other", text: "Where are you going?", at: 15 },
      { who: "other", text: "  ", at: 16 },
      { who: "other", text: "Now?", at: 100 },
    ])
    expect(conversationContext()).toEqual([
      { who: "user", text: "line 10" },
      { who: "other", text: "Where are you going?" },
      { who: "user", text: "line 20" },
      { who: "user", text: "line 30" },
      { who: "user", text: "line 40" },
      { who: "other", text: "Now?" },
    ])
  })

  it("keeps only the end of a long caption, from a word start", () => {
    const long = Array.from({ length: 100 }, (_, i) => `word${i}`).join(" ")
    setConversationContext([{ who: "other", text: long, at: 0 }])
    const [line] = conversationContext()
    expect(line.text.length).toBeLessThanOrEqual(MAX_CONTEXT_CHARS)
    expect(line.text).toMatch(/^word\d+ .* word99$/)
  })
})
