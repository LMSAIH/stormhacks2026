import { describe, expect, it } from "vitest"

import {
  alignWords,
  confidenceFor,
  lineSegments,
  replaceSpan,
  replaceSpanConfidence,
  snapAllowed,
} from "./wordSpans"

const flagged = (segs: ReturnType<typeof lineSegments>) =>
  segs.filter((s) => s.flagged).map((s) => ({ text: s.text, options: s.options }))

describe("alignWords", () => {
  it("maps substitutions, extra words and missing words", () => {
    const { cut, same } = alignWords(["i", "have", "a", "doctor"], ["i", "had", "a", "nice", "doctor"])
    expect(same).toEqual([true, false, true, true])
    // "nice" sits between "a" and "doctor": attached to "a"
    expect(cut).toEqual([0, 1, 2, 4, 5])
  })
})

describe("lineSegments", () => {
  it("boxes only the unsure word, with what the other readings say there", () => {
    const segs = lineSegments(
      "I think I have a doctor's appointment",
      [0.99, 0.98, 0.99, 0.41, 0.97, 0.99, 0.99],
      ["I think I had a doctor's appointment", "I think I have a little disappointment"]
    )
    expect(flagged(segs)).toEqual([{ text: "have", options: ["have", "had"] }])
    expect(segs.map((s) => s.text).join(" ")).toBe("I think I have a doctor's appointment")
  })

  it("grows a box over neighbouring words the readings disagree on", () => {
    const segs = lineSegments(
      "place blue at v two now",
      [0.7, 0.5, 0.7, 0.75, 0.8, 0.99],
      ["plate blew it v two now"]
    )
    expect(flagged(segs)[0].text).toBe("place blue at")
  })

  it("ignores a mostly different reading and never grows over a sure word", () => {
    const segs = lineSegments(
      "place beautiful v two now",
      [0.97, 0.42, 0.95, 0.96, 0.99],
      ["to now", "place blue v two now"]
    )
    expect(flagged(segs)).toEqual([{ text: "beautiful", options: ["beautiful", "blue"] }])
  })

  it("boxes nothing when every word is confident", () => {
    const segs = lineSegments("kids are talking", [0.99, 0.97, 0.95], ["kids are walking"])
    expect(flagged(segs)).toEqual([])
  })

  it("without confidence, boxes where the readings differ", () => {
    const segs = lineSegments("we stop in a minute", undefined, ["we'll stop in a minute"])
    expect(flagged(segs)).toEqual([{ text: "we", options: ["we", "we'll"] }])
  })
})

describe("confidenceFor / replaceSpan", () => {
  it("carries confidences across case changes, null for new words", () => {
    expect(
      confidenceFor("I think so", [
        { text: "I", confidence: 0.9 },
        { text: "THINK", confidence: 0.4 },
      ])
    ).toEqual([0.9, 0.4, null])
  })

  it("replaces or deletes a span and marks the edit certain", () => {
    expect(replaceSpan("I have a cat", 1, 2, "had")).toBe("I had a cat")
    expect(replaceSpan("I have a cat", 1, 3, "")).toBe("I cat")
    expect(replaceSpanConfidence([0.9, 0.4, 0.5, 0.9], 1, 3, "")).toEqual([0.9, 0.9])
  })
})

describe("snapAllowed", () => {
  const w = (text: string, confs: number[]) =>
    text.split(" ").map((t, i) => ({ text: t, confidence: confs[i] }))

  it("never lets a saved phrase override words the reader was sure of", () => {
    const words = w("DOGS ARE SITTING BY THE DOOR", [0.97, 0.99, 0.95, 0.99, 0.99, 0.99])
    expect(snapAllowed("DOGS ARE SITTING BY THE DOOR", "kids are talking by the door", words)).toBe(false)
  })

  it("lets it replace unsure words", () => {
    const words = w("WHAT THE FAX", [0.98, 0.97, 0.31])
    expect(snapAllowed("WHAT THE FAX", "what the fuck", words)).toBe(true)
  })

  it("allows the snap when there is no confidence to go on", () => {
    expect(snapAllowed("WHAT THE FAX", "what the fuck", undefined)).toBe(true)
  })

  it("never lets a shorter saved line swallow the reading", () => {
    // Clipped lines get saved too; they must not eat later full readings.
    expect(snapAllowed("KIDS ARE TALKING BY THE DOOR", "Talking by the door", undefined)).toBe(false)
    expect(snapAllowed("A PIN BROKEN IN HEAD DOWN", "I'm", undefined)).toBe(false)
  })

  it("still drops one word, e.g. a word the reader split in two", () => {
    expect(snapAllowed("JOHN IS ARE TAKEN BY THE DOOR", "Dogs are sitting by the door", undefined)).toBe(true)
  })
})
