/**
 * Mouth-crop pipeline: a pure (DOM-free) port of `ml/src/lipread/preprocess.py`
 * + the vendored `VideoProcess` that the model was trained on —
 *
 *   25 fps resample → face coverage gate → interpolate missing keypoints →
 *   ±6-frame smoothing → similarity onto the mean face (cv2 LMEDS) →
 *   96×96 mouth patch of the 256×256 warp (cv2 fixed-point bilinear).
 *
 * Bit-exact with Python on the parity fixture (`parity.test.ts`); regenerate
 * it with `ml/scripts/dump_crop_reference.py --synthetic`.
 */
import {
  NoFaceError,
  type CapturedFrame,
  type CropResult,
  type LipModelSpec,
  type Utterance,
} from "../types"
import { interpolateKeypoints, smoothKeypoints } from "./keypoints"
import { resampleIndices } from "./resample"
import {
  STABLE_REFERENCE,
  applyAffine,
  estimateSimilarity,
  type Affine,
} from "./similarity"
import { PATCH_SIZE, patchOrigin, warpPatch } from "./warp"

export { rgbaToGray } from "./gray"
export { roundHalfEven } from "./math"
export { INPUT_SIZE, MODEL_MEAN, MODEL_STD, toModelInput } from "./modelInput"
export { interpolateKeypoints, smoothKeypoints } from "./keypoints"
export { resampleIndices } from "./resample"
export {
  LMEDS_PAIRS,
  STABLE_REFERENCE,
  applyAffine,
  estimateSimilarity,
  type Affine,
} from "./similarity"
export { CANVAS_SIZE, PATCH_SIZE, patchOrigin, warpPatch } from "./warp"

/** Python's values (`MODEL_FPS`, `LIPREAD_MIN_FACE_COVERAGE`), used without a spec. */
const DEFAULT_FPS = 25
const DEFAULT_MIN_FACE_COVERAGE = 0.5

/** What the crop needs from a frame (a `CapturedFrame` minus its timestamp). */
export type CropFrame = Pick<
  CapturedFrame,
  "width" | "height" | "gray" | "keypoints" | "tracked"
>

/** `CropResult` plus per-frame geometry, for debugging and parity tests. */
export interface CropDetail extends CropResult {
  /** 2×3 source → 256-canvas transform per output frame. */
  readonly transforms: readonly Affine[]
  /** Top-left of each 96×96 patch on the 256 canvas. */
  readonly origins: readonly (readonly [number, number])[]
}

/**
 * Crop frames that are already at the model's 25 fps (`VideoProcess` on a
 * 25 fps clip). Throws `NoFaceError` when fewer than `minFaceCoverage` of the
 * frames have keypoints.
 */
export function cropFrames(
  frames: readonly CropFrame[],
  minFaceCoverage = DEFAULT_MIN_FACE_COVERAGE
): CropDetail {
  if (frames.length === 0) throw new NoFaceError("no frames to crop")
  const raw = frames.map((f) => f.keypoints)
  // Coverage over frames detection actually looked at; skipped ones (`tracked: false`) are only filled.
  const checked = frames.filter((f) => f.tracked !== false)
  const faceCoverage = checked.length
    ? checked.filter((f) => f.keypoints !== null).length / checked.length
    : 0
  const filled = interpolateKeypoints(raw)
  if (!filled || faceCoverage < minFaceCoverage) {
    const pct = (x: number) => `${Math.round(x * 100)}%`
    throw new NoFaceError(
      `face found in ${pct(faceCoverage)} of frames (need ${pct(minFaceCoverage)})`
    )
  }

  const keypoints = smoothKeypoints(filled)
  const transforms: Affine[] = []
  const origins: [number, number][] = []
  const patches = frames.map((frame, i) => {
    const { width, height, gray } = frame
    if (gray.length !== width * height) {
      throw new RangeError(
        `frame ${i}: gray has ${gray.length} px, expected ${width}x${height}`
      )
    }
    const m = estimateSimilarity(keypoints[i], STABLE_REFERENCE)
    if (!m.every(Number.isFinite)) {
      throw new NoFaceError(`frame ${i}: degenerate face keypoints`)
    }
    const [cx, cy] = applyAffine(m, keypoints[i][3])
    const [x0, y0] = patchOrigin(cx, cy)
    transforms.push(m)
    origins.push([x0, y0])
    return warpPatch(gray, width, height, m, x0, y0, PATCH_SIZE)
  })
  return { patches, faceCoverage, keypoints, transforms, origins }
}

/**
 * A push-to-talk utterance → 25 fps aligned 96×96 mouth patches. Captured
 * frames are resampled to the model rate by timestamp first, so any camera
 * frame rate works. Throws `NoFaceError` when the face is missing from more
 * than half the frames (`spec.minFaceCoverage`).
 */
export function cropUtterance(u: Utterance, spec?: LipModelSpec): CropResult {
  const idx = resampleIndices(
    u.frames.map((f) => f.tMs),
    spec?.fps ?? DEFAULT_FPS
  )
  const { patches, faceCoverage, keypoints } = cropFrames(
    idx.map((i) => u.frames[i]),
    spec?.minFaceCoverage ?? DEFAULT_MIN_FACE_COVERAGE
  )
  return { patches, faceCoverage, keypoints }
}
