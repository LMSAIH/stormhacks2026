/**
 * Round half to even: numpy / Python 3 `round` and OpenCV's `cvRound`
 * (`lrint`). `Math.round` sends halves up, which would move crop windows and
 * fixed-point sample positions by one step on exact ties.
 */
export function roundHalfEven(x: number): number {
  const r = Math.round(x)
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x
}
