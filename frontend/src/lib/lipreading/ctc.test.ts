import { describe, expect, it } from "vitest"

import casesJson from "./__fixtures__/engine/ctc_cases.json?raw"
import { greedyCtcDecode } from "./ctc"

/** Written by __fixtures__/engine/make_fixture.py from `LipReader.greedy` itself. */
interface PythonCase {
  name: string
  timesteps: number
  log_probs: number[]
  ids: number[]
  text: string
  confidence: number | null
}

const fixture = JSON.parse(casesJson) as {
  tokens: string[]
  cases: PythonCase[]
}

/** Log-probs whose per-frame argmax is `ids` (winner gets ~all the mass). */
function peaked(ids: readonly number[], vocab: number): Float32Array {
  const out = new Float32Array(ids.length * vocab).fill(Math.log(1e-4))
  ids.forEach((id, t) => (out[t * vocab + id] = Math.log(0.9)))
  return out
}

describe("greedyCtcDecode — parity with ml/src/lipread/model.py", () => {
  it.each(fixture.cases)("$name", (c) => {
    const out = greedyCtcDecode(
      Float32Array.from(c.log_probs),
      c.timesteps,
      fixture.tokens
    )
    expect(out.text).toBe(c.text)
    if (c.confidence === null) expect(out.confidence).toBeUndefined()
    else expect(out.confidence).toBeCloseTo(c.confidence, 5)
  })
})

describe("greedyCtcDecode", () => {
  // Same layout as the real tokens.json: <blank>, <unk>, pieces…, <eos>.
  const tokens = [
    "<blank>",
    "<unk>",
    "▁THE",
    "▁FIRST",
    "▁LESS",
    "ON",
    "S",
    "<eos>",
  ]

  it("merges repeats, drops blanks and <eos>, turns ▁ into spaces", () => {
    const ids = [0, 2, 2, 0, 3, 3, 0, 4, 5, 5, 0, 0, 7, 7]
    const out = greedyCtcDecode(peaked(ids, tokens.length), ids.length, tokens)
    expect(out.text).toBe("THE FIRST LESSON")
    expect(out.confidence).toBeCloseTo(0.9, 6)
  })

  it("keeps a repeated piece that a blank separates", () => {
    const ids = [6, 0, 6]
    expect(
      greedyCtcDecode(peaked(ids, tokens.length), ids.length, tokens).text
    ).toBe("SS")
  })

  it("returns empty text and no confidence for silence", () => {
    const ids = [0, 0, 0, 0]
    expect(
      greedyCtcDecode(peaked(ids, tokens.length), ids.length, tokens)
    ).toEqual({ text: "" })
    expect(greedyCtcDecode(new Float32Array(0), 0, tokens)).toEqual({
      text: "",
    })
  })

  it("rejects log-probs whose vocab doesn't match tokens.json", () => {
    expect(() => greedyCtcDecode(new Float32Array(3 * 5), 3, tokens)).toThrow(
      /tokens\.json/
    )
  })
})
