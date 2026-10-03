/**
 * TS crop vs the real Python pipeline (vendored VideoProcess + cv2 4.11).
 * Fixture: `ml/scripts/dump_crop_reference.py --synthetic` (committed).
 * Real clip: dump with `--video clip.mp4 --out DIR`, then run with
 * CROP_PARITY_DIR=DIR pnpm exec vitest run src/lib/lipreading/crop
 */
import { beforeAll, describe, expect, it } from "vitest"

import {
  fixtureFrames,
  loadCropFixture,
  type CropFixture,
} from "./__fixtures__/fixture"
import { cropFrames, cropUtterance, type CropDetail } from "./index"

const env =
  (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env ?? {}

const sources: [string, URL | string][] = [
  ["synthetic", new URL("./__fixtures__/synthetic/", import.meta.url)],
]
if (env.CROP_PARITY_DIR) sources.push(["real clip", env.CROP_PARITY_DIR])

function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  expect(a.length).toBe(b.length)
  return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)
}

describe.each(sources)("crop parity with Python — %s", (name, dir) => {
  let fx: CropFixture
  let detail: CropDetail

  beforeAll(async () => {
    fx = await loadCropFixture(dir)
    detail = cropFrames(fixtureFrames(fx))
  })

  it("smooths keypoints exactly like VideoProcess.crop_patch", () => {
    const err = detail.keypoints.map((k, i) =>
      maxAbsDiff(k.flat(), fx.meta.smoothed[i].flat())
    )
    expect(Math.max(...err)).toBeLessThanOrEqual(1e-9)
  })

  it("estimates the same similarity transforms (cv2 LMEDS) within 1e-3", () => {
    const err = detail.transforms.map((m, i) =>
      maxAbsDiff(m, fx.meta.transforms[i])
    )
    const worst = Math.max(...err)
    console.info(`[${name}] max |ΔM| = ${worst.toExponential(2)}`)
    expect(worst).toBeLessThanOrEqual(1e-3)
  })

  it("cuts the patch at the same canvas origin", () => {
    expect(detail.origins.map((o) => [...o])).toEqual(fx.meta.origins)
  })

  it("produces the same 96x96 crops through cropUtterance (mean |Δ| <= 1.5, max <= 8)", () => {
    const { patches, faceCoverage } = cropUtterance({
      frames: fixtureFrames(fx),
      startedAt: 0,
      endedAt: fx.meta.T * 40,
    })
    expect(patches.length).toBe(fx.meta.T)
    const detected = fx.meta.keypoints.filter((k) => k !== null).length
    expect(faceCoverage).toBeCloseTo(detected / fx.meta.T, 12)

    const px = 96 * 96
    let sum = 0
    let max = 0
    let differing = 0
    let notLikeCropFrames = 0
    patches.forEach((patch, t) => {
      for (let i = 0; i < px; i++) {
        if (patch[i] !== detail.patches[t][i]) notLikeCropFrames++
        const d = Math.abs(patch[i] - fx.crops[t * px + i])
        sum += d
        if (d > max) max = d
        if (d) differing++
      }
    })
    expect(notLikeCropFrames).toBe(0) // 40 ms spacing → no resampling
    const mean = sum / (patches.length * px)
    console.info(
      `[${name}] crops: mean |Δ| = ${mean.toFixed(4)}, max |Δ| = ${max}, ` +
        `${differing}/${patches.length * px} px differ`
    )
    expect(mean).toBeLessThanOrEqual(1.5)
    expect(max).toBeLessThanOrEqual(8)
  })
})

describe("synthetic fixture coverage", () => {
  it("exercises gaps, LMEDS outlier rejection and the frame border", async () => {
    const fx = await loadCropFixture(
      new URL("./__fixtures__/synthetic/", import.meta.url)
    )
    const { keypoints, inliers } = fx.meta
    expect(keypoints[0]).toBeNull() // leading gap
    expect(keypoints[keypoints.length - 1]).toBeNull() // trailing gap
    expect(inliers.some((m) => m.reduce((a, b) => a + b) < 4)).toBe(true)
    expect(inliers.some((m) => m.reduce((a, b) => a + b) === 4)).toBe(true)
    // later frames' patches run off the bottom of the source frame → zeros
    const last = fx.crops.subarray((fx.meta.T - 1) * 96 * 96)
    expect(last.filter((v) => v === 0).length).toBeGreaterThan(500)
  })
})
