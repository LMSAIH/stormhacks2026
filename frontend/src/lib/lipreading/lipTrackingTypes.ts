/**
 * Crop shapes used by `faceLandmarker.ts` (the lip-tracking overlay, kept as the teammate wrote
 * it). `LipLandmarker` still builds a small mouth crop next to the lip points, but the model no
 * longer reads it: the model's crop comes from the BlazeFace keypoints via `crop/`. These shapes
 * live here, apart from `LipModelSpec` in `types.ts`, because that one now describes the model.
 */

/** A preprocessed mouth-crop frame. Produced by `LipLandmarker`; nothing consumes it any more. */
export interface LipTrackingFrame {
  readonly data: Float32Array
  readonly width: number
  readonly height: number
  /** Channels per pixel (1 = grayscale, 3 = RGB). */
  readonly channels: number
}

/** The part of the old model spec `LipLandmarker` reads to size and normalize its crop. */
export interface LipTrackingSpec {
  readonly cropWidth: number
  readonly cropHeight: number
  readonly channels: number
  readonly mean: readonly number[]
  readonly std: readonly number[]
}

/** The crop he built it for (LipNet/GRID: 100×50 RGB, pixels scaled to [0, 1]). */
export const LIP_TRACKING_SPEC: LipTrackingSpec = {
  cropWidth: 100,
  cropHeight: 50,
  channels: 3,
  mean: [0, 0, 0],
  std: [1, 1, 1],
}
