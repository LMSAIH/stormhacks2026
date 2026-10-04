import { describe, expect, it } from "vitest"

import { faceIssues, meanBrightness } from "./faceQuality"
import type { Keypoints } from "./types"

// A frontal face at 640 px wide: eyes 100 px apart, nose centred below them.
const frontal: Keypoints = [
  [270, 200],
  [370, 200],
  [320, 250],
  [320, 290],
]

describe("faceIssues", () => {
  it("passes a near, frontal, lit face", () => {
    expect(faceIssues(frontal, 640, 120)).toEqual([])
  })

  it("flags no face, and darkness alongside it", () => {
    expect(faceIssues(null, 640)).toEqual(["no_face"])
    expect(faceIssues(null, 640, 20)).toEqual(["no_face", "too_dark"])
  })

  it("flags a far-away face", () => {
    const far: Keypoints = [
      [310, 200],
      [340, 200],
      [325, 215],
      [325, 228],
    ]
    expect(faceIssues(far, 640, 120)).toContain("too_small")
  })

  it("flags a turned head", () => {
    const turned: Keypoints = [frontal[0], frontal[1], [365, 250], frontal[3]]
    expect(faceIssues(turned, 640, 120)).toEqual(["turned"])
  })

  it("puts too_small first", () => {
    const farTurned: Keypoints = [
      [310, 200],
      [340, 200],
      [338, 215],
      [325, 228],
    ]
    expect(faceIssues(farTurned, 640, 20)[0]).toBe("too_small")
  })
})

describe("meanBrightness", () => {
  it("averages sampled pixels", () => {
    expect(meanBrightness(new Uint8Array(64).fill(200), 4)).toBe(200)
    expect(meanBrightness(new Uint8Array(0))).toBe(0)
  })
})
