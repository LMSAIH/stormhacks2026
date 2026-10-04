import { describe, expect, it } from "vitest"

import { MockRecognizer } from "./mockRecognizer"
import type { CropResult } from "./types"

const withFace: CropResult = {
  patches: [new Uint8Array(96 * 96)],
  faceCoverage: 1,
  keypoints: [],
}
const empty: CropResult = { patches: [], faceCoverage: 0, keypoints: [] }

describe("MockRecognizer", () => {
  it("is always available, never real, and cycles canned phrases", async () => {
    const rec = new MockRecognizer("speed", 0)
    await rec.init()
    expect(rec).toMatchObject({ available: true, isReal: false, mode: "speed" })
    const a = await rec.recognize(withFace)
    const b = await rec.recognize(withFace)
    expect(a.text).toMatch(/^[A-Z ]+$/)
    expect(b.text).not.toBe(a.text)
    expect(a).toMatchObject({
      mode: "speed",
      engine: rec.name,
      confidence: 0.5,
    })
  })

  it("returns nothing when there were no mouth crops", async () => {
    const result = await new MockRecognizer("accuracy", 0).recognize(empty)
    expect(result).toMatchObject({ text: "", mode: "accuracy" })
    expect(result.confidence).toBeUndefined()
  })

  it("can be aborted", async () => {
    const ctrl = new AbortController()
    const pending = new MockRecognizer("speed", 1_000).recognize(
      withFace,
      ctrl.signal
    )
    ctrl.abort()
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  })
})
