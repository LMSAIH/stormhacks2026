import { roundHalfEven } from "./math"

/**
 * For each output frame at `fps`, the index of the captured frame to use —
 * nearest by timestamp. Mirrors `ml/src/lipread/video.py::resample_fps`:
 * n = max(1, round(duration · fps)) with duration = frame count / source fps,
 * frames past the end clamp to the last one, exact ties go to the even index
 * (numpy rounding), and a source already within 0.01 fps of `fps` is kept as
 * is. With evenly spaced timestamps this is exactly
 * `min(round(k · srcFps / fps), n - 1)`; with camera jitter it follows the
 * real capture times.
 */
export function resampleIndices(
  timestampsMs: readonly number[],
  fps = 25
): number[] {
  const n = timestampsMs.length
  if (n <= 1) return n === 0 ? [] : [0]
  const t0 = timestampsMs[0]
  const spanMs = timestampsMs[n - 1] - t0
  const srcFps = ((n - 1) * 1000) / spanMs
  // A clock that did not advance can't be resampled; keep every frame.
  if (!(spanMs > 0) || Math.abs(srcFps - fps) < 0.01) {
    return Array.from({ length: n }, (_, i) => i)
  }

  const count = Math.max(1, roundHalfEven((n / srcFps) * fps))
  const stepMs = 1000 / fps
  const out = new Array<number>(count)
  let j = 0
  for (let k = 0; k < count; k++) {
    const t = t0 + k * stepMs
    while (j < n - 1 && timestampsMs[j + 1] <= t) j++
    if (j === n - 1) {
      out[k] = j
      continue
    }
    const before = t - timestampsMs[j]
    const after = timestampsMs[j + 1] - t
    if (before < after) out[k] = j
    else if (before > after) out[k] = j + 1
    else out[k] = j % 2 === 0 ? j : j + 1
  }
  return out
}
