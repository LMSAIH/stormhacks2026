import type { Keypoints, Point } from "../types"

/** Row-major 2×3 affine [a, b, c, d, e, f]: x' = a·x + b·y + c, y' = d·x + e·y + f. */
export type Affine = readonly [number, number, number, number, number, number]

/**
 * Right eye, left eye, nose tip and mouth centre of the 20-word mean face in
 * the 256×256 aligned space (`VideoProcess.get_stable_reference`, full
 * float64 precision).
 */
export const STABLE_REFERENCE: Keypoints = [
  [102.0739430570659, 94.27230352389712],
  [156.3613054162131, 93.57815605186966],
  [129.00373787023733, 135.90343028603127],
  [129.31337322587925, 157.82299634918854],
]

export function applyAffine(m: Affine, p: Point): Point {
  return [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]]
}

const f32 = Math.fround

/**
 * The point pairs `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)` tries
 * for 4 points: LMeDS reseeds `cv::RNG((uint64)-1)` on every call and runs
 * RANSACUpdateNumIters(0.99, 0.45, 2, 2000) = 13 draws, so the sequence never
 * changes (left eye + mouth is never tried). Derivation re-checked in the tests.
 */
// prettier-ignore
export const LMEDS_PAIRS: readonly (readonly [number, number])[] = [
  [1, 0], [0, 1], [3, 0], [1, 0], [1, 2], [1, 2], [2, 1],
  [2, 3], [2, 0], [2, 0], [0, 3], [3, 0], [1, 0],
]

/** Exact similarity through 2 point pairs (cv AffinePartial2D runKernel). */
function fitTwoPoints(
  p1: Point,
  p2: Point,
  q1: Point,
  q2: Point
): Affine | null {
  const [x1, y1] = p1
  const [x2, y2] = p2
  const [X1, Y1] = q1
  const [X2, Y2] = q2
  const d = 1 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
  if (!Number.isFinite(d)) return null
  const S0 = d * ((X1 - X2) * (x1 - x2) + (Y1 - Y2) * (y1 - y2))
  const S1 = d * ((Y1 - Y2) * (x1 - x2) - (X1 - X2) * (y1 - y2))
  const S2 =
    d *
    ((Y1 - Y2) * (x1 * y2 - x2 * y1) -
      (X1 * y2 - X2 * y1) * (y1 - y2) -
      (X1 * x2 - X2 * x1) * (x1 - x2))
  const S3 =
    d *
    (-(X1 - X2) * (x1 * y2 - x2 * y1) -
      (Y1 * x2 - Y2 * x1) * (x1 - x2) -
      (Y1 * y2 - Y2 * y1) * (y1 - y2))
  return [S0, -S1, S2, S1, S0, S3]
}

/** Squared reprojection errors in float32, as OpenCV's computeError does. */
function squaredErrors(
  m: Affine,
  src: readonly Point[],
  dst: readonly Point[]
): number[] {
  const [F0, F1, F2, F3, F4, F5] = m.map(f32)
  return src.map(([x, y], i) => {
    const a = f32(f32(f32(f32(F0 * x) + f32(F1 * y)) + F2) - dst[i][0])
    const b = f32(f32(f32(f32(F3 * x) + f32(F4 * y)) + F5) - dst[i][1])
    return f32(f32(a * a) + f32(b * b))
  })
}

/** Closed-form least-squares similarity (rotation + uniform scale + shift). */
function fitLeastSquares(src: readonly Point[], dst: readonly Point[]): Affine {
  const n = src.length
  let px = 0
  let py = 0
  let qx = 0
  let qy = 0
  for (let i = 0; i < n; i++) {
    px += src[i][0]
    py += src[i][1]
    qx += dst[i][0]
    qy += dst[i][1]
  }
  px /= n
  py /= n
  qx /= n
  qy /= n
  let sa = 0
  let sb = 0
  let ss = 0
  for (let i = 0; i < n; i++) {
    const ux = src[i][0] - px
    const uy = src[i][1] - py
    const vx = dst[i][0] - qx
    const vy = dst[i][1] - qy
    sa += ux * vx + uy * vy
    sb += ux * vy - uy * vx
    ss += ux * ux + uy * uy
  }
  const a = sa / ss
  const b = sb / ss
  return [a, -b, qx - (a * px - b * py), b, a, qy - (b * px + a * py)]
}

/**
 * Similarity transform taking the 4 `src` keypoints onto `dst`, reproducing
 * `cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)` (what the vendored
 * `VideoProcess` calls): points rounded to float32, the 2-point fit with the
 * least median error over `LMEDS_PAIRS`, inliers within
 * 2.5·1.4826·(1 + 5/2)·√median, then the least-squares fit over the inliers
 * (OpenCV's Levenberg–Marquardt refine converges to exactly that). With all 4
 * points inliers it is the plain least-squares (Umeyama, no reflection) fit;
 * on real faces LMeDS drops one point now and then, and so must we.
 * Matches OpenCV 4.11 to ~1e-9.
 */
export function estimateSimilarity(
  src: Keypoints,
  dst: Keypoints = STABLE_REFERENCE
): Affine {
  const p: Point[] = src.map(([x, y]) => [f32(x), f32(y)])
  const q: Point[] = dst.map(([x, y]) => [f32(x), f32(y)])

  let best: Affine | null = null
  let minMedian = Infinity
  for (const [i, j] of LMEDS_PAIRS) {
    const m = fitTwoPoints(p[i], p[j], q[i], q[j])
    if (!m) continue
    const median = squaredErrors(m, p, q).sort((a, b) => a - b)[2]
    if (median < minMedian) {
      minMedian = median
      best = m
    }
  }
  if (!best) return fitLeastSquares(p, q)

  const sigma = Math.max(
    2.5 * 1.4826 * (1 + 5 / 2) * Math.sqrt(minMedian),
    0.001
  )
  const threshold = f32(sigma * sigma)
  const errors = squaredErrors(best, p, q)
  const inliers = [0, 1, 2, 3].filter((k) => errors[k] <= threshold)
  if (inliers.length < 2) return best
  return fitLeastSquares(
    inliers.map((k) => p[k]),
    inliers.map((k) => q[k])
  )
}
