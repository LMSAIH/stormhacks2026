/**
 * RGBA (canvas `ImageData.data`) → 8-bit grayscale, bit-exact with
 * `cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`: Rec.601 weights in 15-bit fixed
 * point, rounded half up. Checked against OpenCV 4.11 on all 2^24 colours;
 * the float form round(0.299R + 0.587G + 0.114B) is off by one on ~0.13%.
 */
export function rgbaToGray(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): Uint8Array {
  const n = width * height
  if (rgba.length < n * 4) {
    throw new RangeError(
      `rgbaToGray: need ${n * 4} bytes for ${width}x${height}, got ${rgba.length}`
    )
  }
  const gray = new Uint8Array(n)
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    gray[i] =
      (rgba[j] * 9798 + rgba[j + 1] * 19235 + rgba[j + 2] * 3735 + 16384) >> 15
  }
  return gray
}
