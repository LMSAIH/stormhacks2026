import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { describe, expect, it } from "vitest"

import {
  ctcLogLikelihood,
  modelSnap,
  MODEL_SNAP_MARGIN,
  pieceScores,
  rankByModel,
  tokenizePieces,
} from "./ctcScore"

const here = import.meta.dirname
const sp = pieceScores(
  JSON.parse(readFileSync(resolve(here, "../../../public/phrases/pieceScores.json"), "utf8")) as Record<string, number>
)
const fixture = JSON.parse(readFileSync(resolve(here, "__fixtures__/ctcScore.json"), "utf8")) as {
  tokenize: { text: string; pieces: string[] }[]
  ctc: { frames: number; vocab: number; log_probs: number[]; cases: { ids: number[]; log_likelihood: number }[] }[]
}

describe("tokenizePieces", () => {
  it("matches sentencepiece on every LRS3 test reference and a few app-style phrases", () => {
    const mismatches = fixture.tokenize.filter((c) => tokenizePieces(c.text, sp).join(" ") !== c.pieces.join(" "))
    expect(mismatches.map((m) => m.text)).toEqual([])
    expect(fixture.tokenize.length).toBeGreaterThan(600)
  })
})

describe("ctcLogLikelihood", () => {
  it("matches torch ctc_loss on real model outputs", () => {
    for (const clip of fixture.ctc) {
      const lp = Float32Array.from(clip.log_probs)
      for (const c of clip.cases) {
        expect(ctcLogLikelihood(lp, clip.frames, clip.vocab, c.ids)).toBeCloseTo(c.log_likelihood, 1)
      }
    }
  })

  it("is -Infinity when the ids can't fit the frames", () => {
    const lp = new Float32Array([Math.log(0.5), Math.log(0.5)]) // 1 frame, vocab 2
    expect(ctcLogLikelihood(lp, 1, 2, [1, 1])).toBe(-Infinity)
    expect(ctcLogLikelihood(lp, 1, 2, [])).toBe(-Infinity)
  })

  it("sums both alignments of a single token over two frames", () => {
    // P(blank)=P(a)=0.5 per frame; paths for "a": a-a, a-blank, blank-a → 3 × 0.25
    const lp = new Float32Array([Math.log(0.5), Math.log(0.5), Math.log(0.5), Math.log(0.5)])
    expect(Math.exp(ctcLogLikelihood(lp, 2, 2, [1]))).toBeCloseTo(0.75, 6)
  })
})

describe("rankByModel + modelSnap", () => {
  // tiny vocab: blank, ▁A, ▁B; the frames clearly say "A"
  const tokens = ["<blank>", "▁A", "▁B", "<eos>"]
  const tiny = pieceScores({ "▁A": -1, "▁B": -1 })
  const frame = (pA: number) => [Math.log(1 - pA - 0.01), Math.log(pA), Math.log(0.005), Math.log(0.005)]
  const lp = Float32Array.from([...frame(0.9), ...frame(0.9), ...frame(0.05)])

  it("ranks the phrase the frames support first", () => {
    const ranked = rankByModel(lp, 3, tokens, tiny, "B", ["B", "A"])
    expect(ranked[0].text).toBe("A")
    expect(ranked[0].margin).toBeGreaterThan(0)
  })

  it("snaps only above the margin and never to the reading itself", () => {
    const ranked = rankByModel(lp, 3, tokens, tiny, "B", ["A"])
    expect(modelSnap(ranked, "B")?.text).toBe("A")
    expect(modelSnap(ranked, "A")).toBeUndefined()
    expect(modelSnap([{ text: "A", margin: MODEL_SNAP_MARGIN - 0.01 }], "B")).toBeUndefined()
  })
})
