import type { LipModelSpec } from "./types"

/**
 * Greedy CTC decode: argmax per timestep, collapse runs, drop blanks.
 *
 * `logits` is a flat [T * vocab] array (row-major, timestep-major). This is the
 * standard decode for LipNet-style outputs; swap for beam search later if the
 * team wants a language-model-weighted decode.
 */
export function greedyCtcDecode(
  logits: Float32Array,
  timesteps: number,
  spec: LipModelSpec
): string {
  const vocab = spec.charset.length
  const chars: string[] = []
  let prevIndex = -1

  for (let t = 0; t < timesteps; t++) {
    const offset = t * vocab
    let bestIndex = 0
    let bestValue = logits[offset]
    for (let c = 1; c < vocab; c++) {
      const v = logits[offset + c]
      if (v > bestValue) {
        bestValue = v
        bestIndex = c
      }
    }

    // Collapse repeats and skip the blank token.
    if (bestIndex !== prevIndex && bestIndex !== spec.blankIndex) {
      chars.push(spec.charset[bestIndex] ?? "")
    }
    prevIndex = bestIndex
  }

  return chars.join("").trim()
}
