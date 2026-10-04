import { describe, expect, it } from "vitest"

import { expandClipped, SEED_HITS, withSeeds } from "./seeds"
import { rankChoices } from "./snap"

describe("swear seeds", () => {
  it("expands the model's clipped fragment, keeping its case", () => {
    expect(expandClipped("FU")).toBe("FUCK")
    expect(expandClipped("Oh fu that")).toBe("Oh fuck that")
    expect(expandClipped("FUN FUTURE")).toBe("FUN FUTURE")
  })

  it("snaps a whole utterance that looks like a seeded phrase", () => {
    expect(rankChoices(["WHAT THE FAX"], SEED_HITS).snap?.text).toBe("what the fuck")
    expect(rankChoices(["FUCK"], SEED_HITS).snap).toBeUndefined() // already the seed
  })

  it("leaves ordinary sentences alone", () => {
    for (const reading of ["I THINK I HAVE A DOCTOR'S APPOINTMENT", "SHUT THE DOOR", "SHE SAID HELLO"])
      expect(rankChoices([reading], SEED_HITS).snap).toBeUndefined()
  })


  it("does not duplicate a phrase the user already saved", () => {
    const own = { id: "1", text: "Fuck you", count: 4, source: "typed" as const, lastUsed: 1 }
    expect(withSeeds([own]).filter((h) => h.text.toLowerCase() === "fuck you")).toHaveLength(1)
  })
})
