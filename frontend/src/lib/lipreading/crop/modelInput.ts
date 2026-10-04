import type { CropResult } from "../types"
import { PATCH_SIZE } from "./warp"

/** Training normalisation (`ml/src/lipread/preprocess.py` MEAN, STD). */
export const MODEL_MEAN = 0.421
export const MODEL_STD = 0.165
/** Centre crop the model sees. */
export const INPUT_SIZE = 88

const OFFSET = (PATCH_SIZE - INPUT_SIZE) / 2 // torchvision CenterCrop → 4

/** uint8 → normalised value, op by op in float32 exactly like torch. */
const NORMALISE = (() => {
  const f32 = Math.fround
  const lut = new Float32Array(256)
  for (let v = 0; v < 256; v++) {
    lut[v] = f32(f32(f32(v / 255) - f32(MODEL_MEAN)) / f32(MODEL_STD))
  }
  return lut
})()

/**
 * 96×96 patches → the local model's `video` input: centre 88×88, /255,
 * (x - 0.421) / 0.165, laid out [1, 1, T, 88, 88] (N, C, T, H, W). Same values
 * as `preprocess.to_model_input` (+ batch dim) to the last float32 bit.
 */
export function toModelInput(crops: CropResult): {
  data: Float32Array
  dims: [1, 1, number, 88, 88]
} {
  const T = crops.patches.length
  const plane = INPUT_SIZE * INPUT_SIZE
  const data = new Float32Array(T * plane)
  crops.patches.forEach((patch, t) => {
    if (patch.length !== PATCH_SIZE * PATCH_SIZE) {
      throw new RangeError(
        `toModelInput: patch ${t} has ${patch.length} px, expected ${PATCH_SIZE}x${PATCH_SIZE}`
      )
    }
    let o = t * plane
    for (let y = 0; y < INPUT_SIZE; y++) {
      const row = (y + OFFSET) * PATCH_SIZE + OFFSET
      for (let x = 0; x < INPUT_SIZE; x++) data[o++] = NORMALISE[patch[row + x]]
    }
  })
  return { data, dims: [1, 1, T, INPUT_SIZE, INPUT_SIZE] }
}
