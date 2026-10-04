import { clamp, roundHalfEven } from "./math"
import type { Affine } from "./similarity"

/** Side of the aligned face canvas the vendored pipeline warps into. */
export const CANVAS_SIZE = 256
/** Side of the mouth patch cut from that canvas. */
export const PATCH_SIZE = 96

/**
 * Top-left of the mouth patch on the 256 canvas given the warped mouth centre,
 * as `cut_patch` computes it: round(clip(c - 48, 0, 256)), half to even.
 * Python raises when the mouth lands > 53 px off-centre (never on a sane
 * face); here the window is shifted inward instead so it stays 96 wide.
 */
export function patchOrigin(
  cx: number,
  cy: number,
  size = PATCH_SIZE
): [number, number] {
  const half = size / 2
  const max = CANVAS_SIZE - size
  return [
    clamp(roundHalfEven(clamp(cx - half, 0, CANVAS_SIZE)), 0, max),
    clamp(roundHalfEven(clamp(cy - half, 0, CANVAS_SIZE)), 0, max),
  ]
}

/** The inverse map `cv2.warpAffine` builds internally (same operation order). */
function invertAffine(m: Affine): Affine {
  const D0 = m[0] * m[4] - m[1] * m[3]
  const D = D0 !== 0 ? 1 / D0 : 0
  const a = m[4] * D
  const b = m[1] * -D
  const d = m[3] * -D
  const e = m[0] * D
  return [a, b, -a * m[2] - b * m[5], d, e, -d * m[2] - e * m[5]]
}

/**
 * A `size`×`size` window at (x0, y0) of
 * `cv2.warpAffine(gray, m, (256, 256), flags=INTER_LINEAR,
 * borderMode=BORDER_CONSTANT, borderValue=0)`, reproduced bit-exactly:
 * OpenCV maps each output pixel through the inverse transform in 1/1024 px
 * fixed point, snaps it to a 1/32 px grid and blends the 4 neighbours with
 * integer weights; neighbours outside the frame count as 0. (Float bilinear
 * would be off by up to ~5 levels on sharp texture.)
 */
export function warpPatch(
  gray: Uint8Array,
  width: number,
  height: number,
  m: Affine,
  x0: number,
  y0: number,
  size = PATCH_SIZE
): Uint8Array {
  const [a, b, c, d, e, f] = invertAffine(m)
  const out = new Uint8Array(size * size)
  // Per-column fixed-point offsets, from absolute canvas coordinates.
  const adelta = new Int32Array(size)
  const bdelta = new Int32Array(size)
  for (let u = 0; u < size; u++) {
    adelta[u] = roundHalfEven(a * (x0 + u) * 1024)
    bdelta[u] = roundHalfEven(d * (x0 + u) * 1024)
  }
  const at = (x: number, y: number) =>
    x >= 0 && x < width && y >= 0 && y < height ? gray[y * width + x] : 0

  for (let v = 0, o = 0; v < size; v++) {
    const y = y0 + v
    const X0 = roundHalfEven((b * y + c) * 1024) + 16
    const Y0 = roundHalfEven((e * y + f) * 1024) + 16
    for (let u = 0; u < size; u++, o++) {
      const X = (X0 + adelta[u]) >> 5
      const Y = (Y0 + bdelta[u]) >> 5
      const sx = X >> 5
      const sy = Y >> 5
      const fx = X & 31
      const fy = Y & 31
      let p00: number, p01: number, p10: number, p11: number
      if (sx >= 0 && sy >= 0 && sx < width - 1 && sy < height - 1) {
        const i = sy * width + sx
        p00 = gray[i]
        p01 = gray[i + 1]
        p10 = gray[i + width]
        p11 = gray[i + width + 1]
      } else {
        p00 = at(sx, sy)
        p01 = at(sx + 1, sy)
        p10 = at(sx, sy + 1)
        p11 = at(sx + 1, sy + 1)
      }
      // Weights sum to 1024; cv2's 15-bit table is these × 32, same rounding.
      out[o] =
        (p00 * (32 - fy) * (32 - fx) +
          p01 * (32 - fy) * fx +
          p10 * fy * (32 - fx) +
          p11 * fy * fx +
          512) >>
        10
    }
  }
  return out
}
