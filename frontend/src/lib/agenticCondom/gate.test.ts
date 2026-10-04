import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { describe, expect, it } from "vitest"

import { toSentenceCase } from "@/lib/lipreading/format"
import { confidenceFor, lineSegments } from "@/lib/lipreading/wordSpans"

import { applyEdits, checkCorrection, editedWords, type CondomEdit } from "./gate"

const LINE = "I WANT TO GO HOMB NOW"
const CONF = [0.98, 0.97, 0.99, 0.95, 0.4, 0.97]

/** The server's gate (ml/src/lipread/agentic_condom/gate.py plan_edits + apply_edits) on 404 answers. */
const parity = JSON.parse(readFileSync(resolve(import.meta.dirname, "__fixtures__/gate.json"), "utf8")) as {
  raw: string
  answer: string
  confidence: (number | null)[]
  ok: boolean
  edits: CondomEdit[] | null
  text: string | null
}[]

describe("checkCorrection parity with the server's gate", () => {
  it("accepts, rejects, edits and rebuilds every fixture case exactly as gate.py", () => {
    const plain = (edits: readonly CondomEdit[]) =>
      JSON.stringify(edits.map(({ start, end, from, to, reason }) => ({ start, end, from, to, reason })))
    const mismatches = parity.filter((c) => {
      const v = checkCorrection(c.raw, c.answer, c.confidence)
      if (v.ok !== c.ok) return true
      return v.ok && (plain(v.edits) !== plain(c.edits ?? []) || applyEdits(c.raw, v.edits) !== c.text)
    })
    expect(mismatches).toEqual([])
    expect(parity.length).toBeGreaterThanOrEqual(400)
    expect(parity.filter((c) => c.ok).length).toBeGreaterThan(100) // both outcomes covered
    expect(parity.filter((c) => !c.ok).length).toBeGreaterThan(100)
  })
})

describe("checkCorrection", () => {
  it("accepts a swap of an unsure word", () => {
    const v = checkCorrection(LINE, "I WANT TO GO HOME NOW", CONF)
    expect(v).toEqual({
      ok: true,
      edits: [{ start: 4, end: 5, from: "HOMB", to: "HOME", reason: "unsure" }],
    })
  })

  it("accepts a fill of a clipped first or last word, even a sure one", () => {
    expect(checkCorrection("APPENED TO THE DOOR", "HAPPENED TO THE DOOR", [0.95, 0.99, 0.99, 0.99])).toEqual({
      ok: true,
      edits: [{ start: 0, end: 1, from: "APPENED", to: "HAPPENED", reason: "clipped" }],
    })
    expect(checkCorrection("SHUT THE DOO", "SHUT THE DOOR", [0.99, 0.99, null])).toMatchObject({
      ok: true,
      edits: [{ start: 2, end: 3, to: "DOOR", reason: "clipped" }],
    })
  })

  it("only fills clipped words at the edges, with one longer word", () => {
    const sure = [0.99, 0.95, 0.99, 0.99]
    expect(checkCorrection("THE DOO IS OPEN", "THE DOOR IS OPEN", sure).ok).toBe(false) // middle
    expect(checkCorrection("SHUT THE DOOR", "SHUT THE DO", [0.99, 0.99, 0.99]).ok).toBe(false) // shorter
    expect(checkCorrection("APPENED TO US", "IT HAPPENED TO US", [0.99, 0.99, 0.99]).ok).toBe(false) // two words
  })

  it("rejects a change to a sure word", () => {
    expect(checkCorrection(LINE, "I WANT TO GO HOME SOON", CONF)).toMatchObject({ ok: false })
    // an unsure word fixed alongside doesn't excuse it
    expect(checkCorrection(LINE, "I WANT TO SEE HOME NOW", CONF).ok).toBe(false)
  })

  it("treats a word without confidence as sure (saved phrase, swear seed)", () => {
    expect(checkCorrection("WHAT THE FUCK", "WHAT THE DUCK", [0.99, 0.99, null]).ok).toBe(false)
    expect(checkCorrection("FUCK YOU MAN", "FUCK YOU NOW", [null, 0.99, 0.5]).ok).toBe(true)
  })

  it("drops at most one word", () => {
    const shaky = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    expect(checkCorrection(LINE, "I WANT TO GO HOME", shaky)).toMatchObject({
      ok: true,
      edits: [{ start: 4, end: 6, from: "HOMB NOW", to: "HOME", reason: "unsure" }],
    })
    expect(checkCorrection(LINE, "I WANT HOME NOW", shaky)).toMatchObject({ ok: false, reason: "dropped 2 words" })
  })

  it("rejects long rewrites and words added after a sure word", () => {
    const shaky = [0.5, 0.5, 0.5]
    expect(checkCorrection("I AM TIRED", "I AM SO VERY TIRED TODAY", shaky)).toMatchObject({ ok: false })
    expect(checkCorrection("I AM TIRED", "I AM SO TIRED", [0.99, 0.99, 0.5]).ok).toBe(false)
    expect(checkCorrection("I AM TIRED", "I AM SO TIRED", [0.99, 0.6, 0.99]).ok).toBe(true)
  })

  it("ignores case and punctuation, and rejects an empty answer", () => {
    expect(checkCorrection(LINE, "i want to go homb now.", CONF)).toEqual({ ok: true, edits: [] })
    expect(checkCorrection(LINE, "  ", CONF).ok).toBe(false)
  })
})

describe("applyEdits / editedWords", () => {
  it("replaces only the edited words and maps where they land", () => {
    const edits = [
      { start: 1, end: 2, from: "HAVE", to: "HAD A", reason: "unsure" as const },
      { start: 3, end: 4, from: "DOCTOR", to: "", reason: "unsure" as const },
      { start: 4, end: 5, from: "APPOINT", to: "APPOINTMENT", reason: "clipped" as const },
    ]
    expect(applyEdits("I HAVE  NICE DOCTOR APPOINT", edits)).toBe("I HAD A NICE APPOINTMENT")
    expect(editedWords(edits)).toEqual([1, 2, 4])
  })
})

describe("a corrected line in the transcript", () => {
  it("boxes the changed words with the original as an option", () => {
    // What useLipReader does with a correction: the original reading becomes choices[1], and the
    // changed words get no confidence from the reading.
    const words = LINE.split(" ").map((text, i) => ({ text, confidence: CONF[i] }))
    const v = checkCorrection(LINE, "I WANT TO GO HOME NOW", CONF)
    if (!v.ok) throw new Error(v.reason)
    const shown = toSentenceCase(applyEdits(LINE, v.edits))
    const segs = lineSegments(shown, confidenceFor(shown, words), [toSentenceCase(LINE)])
    expect(segs.filter((s) => s.flagged)).toEqual([
      { start: 4, end: 5, text: "home", flagged: true, options: ["home", "homb"] },
    ])
  })
})
