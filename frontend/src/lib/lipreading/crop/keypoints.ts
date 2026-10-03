import type { Keypoints, Point } from "../types"

function mapPoints(k: Keypoints, f: (p: Point, i: number) => Point): Keypoints {
  return [f(k[0], 0), f(k[1], 1), f(k[2], 2), f(k[3], 3)]
}

/** Per-axis mean of the 4 points, summed in index order like numpy. */
function centroid(k: Keypoints): Point {
  let x = 0
  let y = 0
  for (const p of k) {
    x += p[0]
    y += p[1]
  }
  return [x / 4, y / 4]
}

/**
 * Fill frames without a detection, as the vendored
 * `VideoProcess.interpolate_landmarks` does: linear interpolation between
 * neighbouring detections, the first / last detection copied over leading /
 * trailing gaps. Returns null when no frame has keypoints.
 */
export function interpolateKeypoints(
  kps: readonly (Keypoints | null)[]
): Keypoints[] | null {
  const found: { i: number; k: Keypoints }[] = []
  kps.forEach((k, i) => {
    if (k) found.push({ i, k })
  })
  if (found.length === 0) return null

  const out = kps.slice()
  for (let v = 1; v < found.length; v++) {
    const { i: a, k: start } = found[v - 1]
    const { i: b, k: stop } = found[v]
    const gap = b - a
    for (let i = 1; i < gap; i++) {
      // start + i/gap * (stop - start), same operation order as Python
      const f = i / gap
      out[a + i] = mapPoints(start, (p, q) => [
        p[0] + f * (stop[q][0] - p[0]),
        p[1] + f * (stop[q][1] - p[1]),
      ])
    }
  }

  const first = found[0]
  const last = found[found.length - 1]
  return out.map((k, i) => k ?? (i < first.i ? first.k : last.k))
}

/**
 * Temporal smoothing from the vendored `VideoProcess.crop_patch`: frame i
 * becomes the mean of frames i-m..i+m with m = min(halfWindow, i, n-1-i),
 * then is shifted so its centroid equals frame i's own centroid. Sums run in
 * numpy's order, so the result matches Python bit for bit.
 */
export function smoothKeypoints(
  kps: readonly Keypoints[],
  halfWindow = 6
): Keypoints[] {
  const n = kps.length
  const out: Keypoints[] = []
  for (let i = 0; i < n; i++) {
    const m = Math.min(halfWindow, i, n - 1 - i)
    const sum = [0, 0, 0, 0, 0, 0, 0, 0]
    for (let j = i - m; j <= i + m; j++) {
      const k = kps[j]
      for (let p = 0; p < 4; p++) {
        sum[2 * p] += k[p][0]
        sum[2 * p + 1] += k[p][1]
      }
    }
    const count = 2 * m + 1
    const mean = mapPoints(kps[i], (_, p) => [
      sum[2 * p] / count,
      sum[2 * p + 1] / count,
    ])
    const own = centroid(kps[i])
    const smoothed = centroid(mean)
    const dx = own[0] - smoothed[0]
    const dy = own[1] - smoothed[1]
    out.push(mapPoints(mean, (p) => [p[0] + dx, p[1] + dy]))
  }
  return out
}
