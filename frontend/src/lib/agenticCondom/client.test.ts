import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  BREAKER_PAUSE_MS,
  condomBaseUrl,
  condomBudgetMs,
  condomChangedWords,
  condomPhrases,
  CONDOM_BUDGET_MS,
  correctLine,
  resetCondomClient,
  warmCondom,
  WARM_EVERY_MS,
  type CondomOptions,
  type CondomRequest,
} from "./client"
import { setConversationContext } from "./context"

const BASE = "https://pod.example/" // trailing slash must be stripped
const LINE = "I WANT TO GO HOMB NOW"
const FIXED = "I WANT TO GO HOME NOW"
const REQ: CondomRequest = { text: LINE, confidence: [0.98, 0.97, 0.99, 0.95, 0.4, 0.97] }
const AS_READ = { text: LINE, raw: LINE, edits: [], changed: false }

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  })
}

/** The server's answer for LINE. */
const reply = (text: string, status = "corrected", raw = LINE) =>
  json({ text, raw, edits: [], status, model: "test", latency_ms: { llm: 120, total: 130 } })

/** A fetch that never answers, but rejects like the real one when its signal aborts. */
function hangingFetch(_url: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return new Promise((_, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
  })
}

const opts = (fetchImpl: typeof fetch, extra: CondomOptions = {}): CondomOptions => ({
  baseUrl: BASE,
  fetch: fetchImpl,
  enabled: true,
  budgetMs: 50,
  ...extra,
})

beforeEach(() => {
  resetCondomClient()
  setConversationContext([])
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
})

describe("settings from the build env", () => {
  it("runs on VITE_CONDOM_URL when set, else on the lip-read server", async () => {
    vi.stubEnv("VITE_LIPREAD_URL", " https://pod.example/ ")
    vi.stubEnv("VITE_CONDOM_URL", "")
    expect(condomBaseUrl()).toBe("https://pod.example")
    vi.stubEnv("VITE_CONDOM_URL", "https://llm.example//")
    expect(condomBaseUrl()).toBe("https://llm.example")

    const fetchImpl = vi.fn<typeof fetch>(async () => reply(FIXED))
    await correctLine(REQ, { fetch: fetchImpl, enabled: true })
    expect(fetchImpl.mock.calls[0][0]).toBe("https://llm.example/correct")

    vi.stubEnv("VITE_CONDOM_URL", "")
    vi.stubEnv("VITE_LIPREAD_URL", "")
    expect(condomBaseUrl()).toBe("")
  })

  it("takes one budget for both modes from VITE_CONDOM_BUDGET_MS", async () => {
    vi.stubEnv("VITE_CONDOM_BUDGET_MS", "") // whatever frontend/.env.local says
    expect(condomBudgetMs("normal")).toBe(CONDOM_BUDGET_MS.normal)
    vi.stubEnv("VITE_CONDOM_BUDGET_MS", "nonsense")
    expect(condomBudgetMs("quality")).toBe(CONDOM_BUDGET_MS.quality)
    vi.stubEnv("VITE_CONDOM_BUDGET_MS", "20")
    expect([condomBudgetMs("normal"), condomBudgetMs("quality")]).toEqual([20, 20])

    const started = performance.now()
    const out = await correctLine(REQ, { baseUrl: BASE, fetch: hangingFetch, enabled: true, mode: "quality" })
    expect(out.status).toBe("timeout")
    expect(performance.now() - started).toBeLessThan(400) // not Quality's 1 s
  })
})

describe("correctLine", () => {
  it("posts the line as a CORS simple request and returns the fix", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => reply(FIXED))
    setConversationContext([
      { who: "user", text: "I am tired", at: 20 },
      { who: "other", text: "Where are you going?", at: 10 },
    ])
    const out = await correctLine(
      {
        ...REQ,
        alternatives: [FIXED, LINE, "", FIXED],
        phrases: [{ text: "I want to go home", score: -0.12 }],
      },
      opts(fetchImpl, { mode: "quality" })
    )

    expect(out).toEqual({
      text: FIXED,
      raw: LINE,
      edits: [{ start: 4, end: 5, from: "HOMB", to: "HOME", reason: "unsure" }],
      status: "corrected",
      changed: true,
    })
    const [url, init] = fetchImpl.mock.calls[0]
    expect(url).toBe("https://pod.example/correct")
    expect(init?.method).toBe("POST")
    // text/plain: no preflight round trip inside the budget (the server parses JSON anyway)
    expect(init?.headers).toEqual({ "Content-Type": "text/plain;charset=UTF-8" })
    expect(JSON.parse(init?.body as string)).toEqual({
      text: LINE,
      words: [
        { text: "I", confidence: 0.98 },
        { text: "WANT", confidence: 0.97 },
        { text: "TO", confidence: 0.99 },
        { text: "GO", confidence: 0.95 },
        { text: "HOMB", confidence: 0.4 },
        { text: "NOW", confidence: 0.97 },
      ],
      alternatives: [FIXED], // the line itself, blanks and repeats left out
      phrases: [{ text: "I want to go home", score: -0.12 }],
      context: [
        { who: "other", text: "Where are you going?" },
        { who: "user", text: "I am tired" },
      ],
      mode: "quality",
    })
  })

  it("sends unknown confidence as null and keeps those words", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () =>
      json({ text: "WHAT THE DUCK IS GOING ON", raw: "WHAT THE FUCK IS GOING ONE", status: "corrected" })
    )
    const out = await correctLine(
      { text: "WHAT THE FUCK IS GOING ONE", confidence: [0.99, 0.98, null, 0.97, 0.5] },
      opts(fetchImpl)
    )
    const sent = JSON.parse(fetchImpl.mock.calls[0][1]?.body as string)
    expect(sent.words[2]).toEqual({ text: "FUCK", confidence: null })
    expect(sent.words[5]).toEqual({ text: "ONE", confidence: null }) // missing → null
    expect(out).toEqual({ ...AS_READ, text: "WHAT THE FUCK IS GOING ONE", raw: "WHAT THE FUCK IS GOING ONE", status: "rejected" })
  })

  it.each([
    ["a timeout", hangingFetch, "timeout"],
    ["a network error", () => Promise.reject(new TypeError("Failed to fetch")), "error"],
    ["a fetch that throws", () => { throw new Error("boom") }, "error"],
    ["malformed JSON", () => Promise.resolve(new Response("<html>oops</html>")), "error"],
    ["HTTP 502", () => Promise.resolve(json({ detail: "bad gateway" }, 502)), "error"],
    ["HTTP 422", () => Promise.resolve(json({ detail: { error: "bad_body" } }, 422)), "error"],
    ["an answer for another line", () => Promise.resolve(reply(FIXED, "corrected", "I WANT TO GO")), "error"],
    ["an answer without raw (old server)", () => Promise.resolve(json({ text: FIXED })), "error"],
    ["an answer that changes a sure word", () => Promise.resolve(reply("I WANT TO GO HOME SOON")), "rejected"],
    ["an answer that drops two words", () => Promise.resolve(reply("I WANT HOME NOW")), "rejected"],
    ["the server's own timeout", () => Promise.resolve(reply(LINE, "timeout")), "timeout"],
    ["a server that says it didn't correct", () => Promise.resolve(reply(FIXED, "rejected")), "rejected"],
    ["a server with no LLM", () => Promise.resolve(reply(LINE, "off")), "off"],
  ] as const)("gives the line back as read after %s", async (_, impl, status) => {
    const out = await correctLine(REQ, opts(vi.fn<typeof fetch>(impl)))
    expect(out).toEqual({ ...AS_READ, status })
  })

  it("never waits past the budget, even when fetch ignores the abort", async () => {
    const started = performance.now()
    const out = await correctLine(REQ, opts(() => new Promise<Response>(() => undefined), { budgetMs: 30 }))
    expect(out.status).toBe("timeout")
    expect(performance.now() - started).toBeLessThan(500)
  })

  it("reports an unchanged line, case and punctuation aside", async () => {
    const out = await correctLine(REQ, opts(vi.fn<typeof fetch>(async () => reply("i want to go homb now."))))
    expect(out).toEqual({ ...AS_READ, status: "unchanged" })
  })

  it("puts the fix in the line's case", async () => {
    const lower = await correctLine(REQ, opts(vi.fn<typeof fetch>(async () => reply("i want to go home now"))))
    expect(lower.text).toBe(FIXED)

    const phrase = "Dogs are sitting by the dor" // a saved phrase, not capitals
    const fetchImpl = vi.fn<typeof fetch>(async () => reply("DOGS ARE SITTING BY THE DOOR", "corrected", phrase))
    const out = await correctLine({ text: phrase, confidence: [null, 0.99, 0.95, 0.99, 0.99, 0.3] }, opts(fetchImpl))
    expect(out.text).toBe("Dogs are sitting by the door")
  })

  it.each([
    ["a one-word line", { ...REQ, text: "HOMB", confidence: [0.4] }, {}, "skipped"],
    ["a line it is sure of", { ...REQ, confidence: [0.98, 0.97, 0.99, 0.95, 0.9, null] }, {}, "skipped"],
    ["the switch off", REQ, { enabled: false }, "off"],
    ["no server", REQ, { baseUrl: "  " }, "off"],
    ["Instant mode", REQ, { mode: "instant" }, "off"],
  ] as const)("doesn't call the server for %s", async (_, req, extra, status) => {
    const fetchImpl = vi.fn<typeof fetch>()
    const out = await correctLine(req, opts(fetchImpl, extra))
    expect(out).toEqual({ ...AS_READ, text: req.text, raw: req.text, status })
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("reads the stored switch", async () => {
    vi.stubGlobal("localStorage", { getItem: () => "0", setItem: () => undefined })
    const fetchImpl = vi.fn<typeof fetch>()
    const out = await correctLine(REQ, { baseUrl: BASE, fetch: fetchImpl })
    expect(out.status).toBe("off")
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("rests 30 s after two failures in a row, then tries again", async () => {
    let t = 1_000
    const fetchImpl = vi.fn<typeof fetch>(() => Promise.reject(new TypeError("Failed to fetch")))
    const o = opts(fetchImpl, { now: () => t })
    expect((await correctLine(REQ, o)).status).toBe("error")
    expect((await correctLine(REQ, o)).status).toBe("error")
    expect((await correctLine(REQ, o)).status).toBe("offline")
    expect(fetchImpl).toHaveBeenCalledTimes(2)

    t += BREAKER_PAUSE_MS
    fetchImpl.mockImplementation(async () => reply(FIXED))
    expect((await correctLine(REQ, o)).status).toBe("corrected")
    expect(fetchImpl).toHaveBeenCalledTimes(3)
  })

  it("counts timeouts and 5xx, not answers or aborts", async () => {
    let t = 0
    const fetchImpl = vi.fn<typeof fetch>(hangingFetch)
    const o = opts(fetchImpl, { now: () => t, budgetMs: 10 })
    expect((await correctLine(REQ, o)).status).toBe("timeout")
    fetchImpl.mockImplementationOnce(async () => json({ detail: "bad" }, 422)) // answered: resets
    expect((await correctLine(REQ, o)).status).toBe("error")
    expect((await correctLine(REQ, o)).status).toBe("timeout")
    const ctrl = new AbortController()
    const pending = correctLine(REQ, { ...o, signal: ctrl.signal, budgetMs: 5_000 })
    setTimeout(() => ctrl.abort(), 5)
    expect((await pending).status).toBe("aborted") // not a failure
    fetchImpl.mockImplementationOnce(async () => json({ detail: "down" }, 503))
    expect((await correctLine(REQ, o)).status).toBe("error")
    expect((await correctLine(REQ, o)).status).toBe("offline")
    expect(fetchImpl).toHaveBeenCalledTimes(5)
    t += BREAKER_PAUSE_MS
    expect((await correctLine(REQ, o)).status).toBe("timeout")
    expect((await correctLine(REQ, o)).status).toBe("offline") // still failing: rests again at once
  })

  it("gives up when the caller aborts", async () => {
    const ctrl = new AbortController()
    ctrl.abort()
    const fetchImpl = vi.fn<typeof fetch>()
    expect((await correctLine(REQ, opts(fetchImpl, { signal: ctrl.signal }))).status).toBe("aborted")
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("remembers which words it changed, for the line's hover hint", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => reply("I WANT DO GO HOME NOW"))
    await correctLine({ ...REQ, confidence: [0.98, 0.97, 0.5, 0.95, 0.4, 0.97] }, opts(fetchImpl, { lineId: "lip-1" }))
    expect(condomChangedWords("lip-1", "I want do go home now")).toEqual([2, 4])
    expect(condomChangedWords("lip-1", "I want to go homb now")).toEqual([]) // edited since
    expect(condomChangedWords("lip-2", "I want do go home now")).toEqual([])
  })
})

describe("condomPhrases", () => {
  it("uses the model's margins, best first", () => {
    expect(
      condomPhrases(
        LINE,
        [
          { text: "I want to go home soon", margin: -0.5 },
          { text: "I want to go home", margin: -0.1 },
          { text: "x".repeat(301), margin: 0 }, // over the server's limit
          { text: "Go away", margin: Number.NEGATIVE_INFINITY },
        ],
        ["unused"]
      )
    ).toEqual([
      { text: "I want to go home", score: -0.1 },
      { text: "I want to go home soon", score: -0.5 },
    ])
  })

  it("else ranks the saved phrases by look-alike, without a score", () => {
    const own = ["See you tomorrow", "I want to go home", "I want to go home now", ...Array(9).fill("I want to go")]
    const out = condomPhrases(LINE, null, own)
    expect(out.slice(0, 2)).toEqual([
      { text: "I want to go home now", score: null },
      { text: "I want to go home", score: null },
    ])
    expect(out).toHaveLength(5)
    expect(out.map((p) => p.text)).not.toContain("See you tomorrow")
  })
})

describe("warmCondom", () => {
  it("GETs /health at most once per 45 s, only when it can run", () => {
    let t = 0
    const fetchImpl = vi.fn<typeof fetch>(async () => json({ status: "ok" }))
    const o = { baseUrl: BASE, fetch: fetchImpl, now: () => t, enabled: true }
    warmCondom(o)
    warmCondom(o)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    const [url, init] = fetchImpl.mock.calls[0]
    expect(url).toBe("https://pod.example/health")
    expect(init?.method).toBeUndefined() // a plain GET: no preflight
    expect(init?.headers).toBeUndefined()

    t += WARM_EVERY_MS
    warmCondom({ ...o, enabled: false })
    warmCondom({ ...o, baseUrl: "" })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    warmCondom(o)
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it("never throws", () => {
    expect(() =>
      warmCondom({ baseUrl: BASE, enabled: true, fetch: () => { throw new Error("boom") } })
    ).not.toThrow()
  })
})
