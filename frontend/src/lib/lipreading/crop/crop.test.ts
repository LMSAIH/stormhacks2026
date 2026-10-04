import { describe, expect, it } from "vitest"

import { NoFaceError, type Keypoints, type Point } from "../types"
import casesJson from "./__fixtures__/reference_cases.json?raw"
import {
  LMEDS_PAIRS,
  STABLE_REFERENCE,
  applyAffine,
  cropUtterance,
  estimateSimilarity,
  interpolateKeypoints,
  patchOrigin,
  resampleIndices,
  rgbaToGray,
  roundHalfEven,
  smoothKeypoints,
  toModelInput,
  warpPatch,
  type Affine,
} from "./index"

/** Python / cv2 outputs written by `dump_crop_reference.py --synthetic`. */
const cases = JSON.parse(casesJson) as {
  stableReference: number[][]
  similarity: { src: Keypoints; M: number[]; inliers: number[] }[]
  gray: { rgb: number[]; gray: number[]; tricky: number }
  resample: { fps: number; n: number; idx: number[] }[]
  modelInput: { shape: number[]; probes: number[][]; lut: number[] }
}

const kp = (...xy: number[]): Keypoints => [
  [xy[0], xy[1]],
  [xy[2], xy[3]],
  [xy[4], xy[5]],
  [xy[6], xy[7]],
]
const mapKp = (k: Keypoints, f: (p: Point) => Point): Keypoints => [
  f(k[0]),
  f(k[1]),
  f(k[2]),
  f(k[3]),
]
const shift = (k: Keypoints, dx: number, dy: number) =>
  mapKp(k, ([x, y]) => [x + dx, y + dy])
const maxAbs = (a: readonly number[], b: readonly number[]) =>
  a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)

describe("roundHalfEven", () => {
  it("sends ties to even like numpy and cvRound", () => {
    const xs = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, 2.6]
    expect(xs.map((x) => roundHalfEven(x))).toEqual([0, 2, 2, -0, -2, -2, 2, 3])
  })
})

describe("rgbaToGray", () => {
  it("matches cv2 RGB2GRAY bit for bit, incl. colours where float rounding differs", () => {
    const { rgb, gray, tricky } = cases.gray
    const n = gray.length
    const rgba = new Uint8ClampedArray(n * 4)
    for (let i = 0; i < n; i++) {
      rgba.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], 4 * i)
    }
    expect(tricky).toBeGreaterThan(50)
    expect(Array.from(rgbaToGray(rgba, n, 1))).toEqual(gray)
  })

  it("rejects a short buffer", () => {
    expect(() => rgbaToGray(new Uint8ClampedArray(8), 2, 2)).toThrow(RangeError)
  })
})

describe("resampleIndices", () => {
  const stamps = (n: number, fps: number, t0 = 0) =>
    Array.from({ length: n }, (_, i) => t0 + (i * 1000) / fps)

  it("picks the same frames as video.resample_fps", () => {
    for (const { fps, n, idx } of cases.resample) {
      expect(
        resampleIndices(stamps(n, fps)),
        `${fps} fps, ${n} frames`
      ).toEqual(idx)
    }
  })

  it("is the identity at 25 fps, even with capture jitter", () => {
    const ids = [...Array(40).keys()]
    expect(resampleIndices(stamps(40, 25, 1234))).toEqual(ids)
    const jittery = stamps(40, 25, 1234).map((t, i) =>
      i % 39 ? t + ((i * 7) % 5) - 2 : t
    )
    expect(resampleIndices(jittery)).toEqual(ids)
  })

  it("follows the real capture times when frames arrive unevenly", () => {
    // a 30 fps camera that dropped frames 10..14
    const t = stamps(60, 30, 500).filter((_, i) => i < 10 || i > 14)
    const idx = resampleIndices(t)
    expect(idx.length).toBe(50)
    idx.forEach((j, k) => {
      const target = t[0] + k * 40
      const nearest = Math.min(...t.map((s) => Math.abs(s - target)))
      expect(Math.abs(t[j] - target)).toBe(nearest)
    })
  })

  it("handles empty, single-frame and frozen-clock input", () => {
    expect(resampleIndices([])).toEqual([])
    expect(resampleIndices([5])).toEqual([0])
    expect(resampleIndices([7, 7, 7])).toEqual([0, 1, 2])
  })
})

describe("interpolateKeypoints", () => {
  const A = kp(10, 20, 30, 20, 20, 30, 20, 40)
  const B = shift(A, 30, -6)

  it("lerps inner gaps and copies the first / last detection into the edges", () => {
    expect(interpolateKeypoints([null, A, null, null, B, null])).toEqual([
      A,
      A,
      shift(A, 10, -2),
      shift(A, 20, -4),
      B,
      B,
    ])
  })

  it("returns null when nothing was detected", () => {
    expect(interpolateKeypoints([null, null])).toBeNull()
  })
})

describe("smoothKeypoints", () => {
  it("averages ±m frames (m = min(6, i, n-1-i)) then re-centres on the frame", () => {
    const k0 = kp(0, 0, 10, 0, 5, 5, 5, 10)
    const k1 = shift(k0, 10, 0)
    // last frame: shifted 20 px, and its point 0 another +6 px
    const k2: Keypoints = [
      [26, 0],
      [30, 0],
      [25, 5],
      [25, 10],
    ]
    const [s0, s1, s2] = smoothKeypoints([k0, k1, k2])
    expect(s0).toEqual(k0) // m = 0 at both ends
    expect(s2).toEqual(k2)
    // window mean = k1 with point 0 at +2; its centroid is +0.5 off k1's
    expect(s1).toEqual([
      [11.5, 0],
      [19.5, 0],
      [14.5, 5],
      [14.5, 10],
    ])
  })

  it("uses a 13-frame window away from the ends", () => {
    const base = kp(0, 0, 10, 0, 5, 5, 5, 10)
    const seq = Array.from({ length: 41 }, (_, i): Keypoints =>
      i === 20 ? [[13, 0], base[1], base[2], base[3]] : base
    )
    // the blip at frame 20 reaches frames 14..26 only
    const moved = smoothKeypoints(seq).map((k) => k[0][0] !== 0)
    expect(moved.indexOf(true)).toBe(14)
    expect(moved.lastIndexOf(true)).toBe(26)
  })
})

describe("estimateSimilarity", () => {
  it("tries the same 13 point pairs as cv::RNG((uint64)-1) in LMeDS", () => {
    let state = 0xffffffffffffffffn
    const next = () => {
      state = (state & 0xffffffffn) * 4164903690n + (state >> 32n)
      return Number((state & 0xffffffffn) % 4n)
    }
    const pairs = Array.from({ length: 13 }, () => {
      const a = next()
      let b = next()
      while (b === a) b = next()
      return [a, b]
    })
    expect(pairs).toEqual(LMEDS_PAIRS.map((p) => [...p]))
    // RANSACUpdateNumIters(0.99, 0.45, 2, 2000)
    expect(Math.round(Math.log(0.01) / Math.log(1 - 0.55 ** 2))).toBe(13)
  })

  it("recovers a known similarity", () => {
    // ×2, rotate 90°, shift: every value exact in float32
    const M: Affine = [0, -2, 300, 2, 0, -40]
    const src = kp(100, 90, 150, 92, 127, 120, 128, 150)
    const dst = mapKp(src, (p) => applyAffine(M, p))
    expect(maxAbs(estimateSimilarity(src, dst), M)).toBeLessThan(1e-9)

    const [c, s] = [1.3 * Math.cos(0.3), 1.3 * Math.sin(0.3)]
    const R: Affine = [c, -s, 12.5, s, c, 7.25]
    const dst2 = mapKp(STABLE_REFERENCE, (p) => applyAffine(R, p))
    expect(maxAbs(estimateSimilarity(STABLE_REFERENCE, dst2), R)).toBeLessThan(
      1e-4
    )
  })

  it("matches cv2.estimateAffinePartial2D(LMEDS), dropped outliers included", () => {
    expect(maxAbs(cases.stableReference.flat(), STABLE_REFERENCE.flat())).toBe(
      0
    )
    let withOutlier = 0
    let worst = 0
    for (const c of cases.similarity) {
      worst = Math.max(worst, maxAbs(estimateSimilarity(c.src), c.M))
      if (c.inliers.includes(0)) withOutlier++
    }
    expect(withOutlier).toBeGreaterThan(40)
    expect(worst).toBeLessThan(1e-6)
  })

  it("ignores a gross outlier, unlike plain least squares", () => {
    const G: Affine = [0.5, -0.1, 40, 0.1, 0.5, 30] // canvas → source
    const good = mapKp(STABLE_REFERENCE, (p) => applyAffine(G, p))
    const bad: Keypoints = [
      good[0],
      good[1],
      [good[2][0] + 25, good[2][1] - 18],
      good[3],
    ]
    const fit = estimateSimilarity(bad)
    for (const k of [0, 1, 3]) {
      expect(
        maxAbs(applyAffine(fit, bad[k]), STABLE_REFERENCE[k])
      ).toBeLessThan(1e-3)
    }
  })
})

describe("warpPatch / patchOrigin", () => {
  const W = 40
  const H = 30
  const gray = Uint8Array.from({ length: W * H }, (_, i) => (i * 37) % 251)

  it("copies pixels for an integer shift and zero-fills outside the frame", () => {
    // canvas = source + (60, 70), so the window at (60, 70) starts at source (0, 0)
    const patch = warpPatch(gray, W, H, [1, 0, 60, 0, 1, 70], 60, 70, 48)
    const want = Uint8Array.from({ length: 48 * 48 }, (_, i) => {
      const [u, v] = [i % 48, Math.floor(i / 48)]
      return u < W && v < H ? gray[v * W + u] : 0
    })
    expect(patch).toEqual(want)
  })

  it("blends a half-pixel offset with OpenCV's round-half-up", () => {
    const g = Uint8Array.from([10, 13, 0, 0])
    expect(warpPatch(g, 2, 2, [1, 0, -0.5, 0, 1, 0], 0, 0, 1)[0]).toBe(12)
  })

  it("places the 96 px window like cut_patch, shifted inward at the edges", () => {
    expect(patchOrigin(129.5, 156)).toEqual([82, 108]) // 81.5 → 82 (even)
    expect(patchOrigin(128.5, 157)).toEqual([80, 109]) // 80.5 → 80 (even)
    expect(patchOrigin(10, 250)).toEqual([0, 160])
  })
})

describe("toModelInput", () => {
  const T = 3
  const patches = Array.from({ length: T }, (_, t) =>
    Uint8Array.from({ length: 96 * 96 }, (_, i) => (i * 7 + t * 31) % 256)
  )

  it("equals preprocess.to_model_input: centre 88, /255, (x-.421)/.165, [1,1,T,88,88]", () => {
    const { data, dims } = toModelInput({
      patches,
      faceCoverage: 1,
      keypoints: [],
    })
    expect(dims).toEqual([1, 1, T, 88, 88])
    expect(cases.modelInput.shape).toEqual([1, T + 1, 88, 88])
    expect(data.length).toBe(T * 88 * 88)
    for (const [t, y, x, value] of cases.modelInput.probes) {
      expect(data[t * 88 * 88 + y * 88 + x]).toBe(value)
    }
  })

  it("normalises every uint8 level to torch's float32 value exactly", () => {
    const lutPatch = new Uint8Array(96 * 96)
    for (let v = 0; v < 256; v++) {
      lutPatch[(4 + Math.floor(v / 88)) * 96 + 4 + (v % 88)] = v
    }
    const { data } = toModelInput({
      patches: [lutPatch],
      faceCoverage: 1,
      keypoints: [],
    })
    expect(Array.from(data.subarray(0, 256))).toEqual(cases.modelInput.lut)
  })

  it("rejects patches that are not 96x96", () => {
    expect(() =>
      toModelInput({
        patches: [new Uint8Array(88 * 88)],
        faceCoverage: 1,
        keypoints: [],
      })
    ).toThrow(RangeError)
  })
})

describe("cropUtterance", () => {
  const W = 64
  const H = 48
  const gray = new Uint8Array(W * H).fill(128)
  const face = kp(20, 14, 42, 14, 31, 25, 31, 34)
  const utterance = (
    n: number,
    fps: number,
    hasFace: (i: number) => boolean
  ) => ({
    frames: Array.from({ length: n }, (_, i) => ({
      tMs: (i * 1000) / fps,
      width: W,
      height: H,
      gray,
      keypoints: hasFace(i) ? face : null,
    })),
    startedAt: 0,
    endedAt: (n * 1000) / fps,
  })

  it("turns a 30 fps capture into 25 fps 96x96 patches", () => {
    const res = cropUtterance(utterance(36, 30, () => true))
    expect(res.patches.length).toBe(30)
    expect(res.keypoints.length).toBe(30)
    expect(res.faceCoverage).toBe(1)
    expect(res.patches.every((p) => p.length === 96 * 96)).toBe(true)
  })

  it("doesn't count frames detection skipped (tracked: false) against coverage", () => {
    // Background tracker looked at every 3rd frame only, and found the face each time.
    const u = utterance(30, 25, (i) => i % 3 === 0)
    const sparse = {
      ...u,
      frames: u.frames.map((f, i) => (i % 3 === 0 ? f : { ...f, tracked: false })),
    }
    expect(() => cropUtterance(u)).toThrow(NoFaceError) // as plain misses: 33% coverage
    const res = cropUtterance(sparse)
    expect(res.faceCoverage).toBe(1)
    expect(res.patches.length).toBe(30)
    // Looked at but no face is still a miss.
    const looked = { ...sparse, frames: sparse.frames.map((f) => ({ ...f, keypoints: null })) }
    expect(() => cropUtterance(looked)).toThrow(NoFaceError)
  })

  it("throws NoFaceError below 50% face coverage and accepts exactly 50%", () => {
    expect(() => cropUtterance(utterance(20, 25, (i) => i < 9))).toThrow(
      NoFaceError
    )
    expect(cropUtterance(utterance(20, 25, (i) => i < 10)).faceCoverage).toBe(
      0.5
    )
    expect(() => cropUtterance(utterance(10, 25, () => false))).toThrow(
      NoFaceError
    )
    expect(() =>
      cropUtterance({ frames: [], startedAt: 0, endedAt: 0 })
    ).toThrow(NoFaceError)
  })
})
