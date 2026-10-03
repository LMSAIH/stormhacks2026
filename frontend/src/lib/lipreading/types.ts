/**
 * Core contracts for the lip-reading pipeline.
 *
 * The UI and capture loop depend ONLY on these interfaces, never on a concrete
 * model. Teammates building the real model implement `LipReaderEngine` and swap
 * it in via `createLipReaderEngine` — nothing else changes.
 */

/** A single preprocessed mouth-crop frame, ready to be stacked into a tensor. */
export interface LipFrame {
  /** Pixel data, normalized to the model's expected range. */
  readonly data: Float32Array
  readonly width: number
  readonly height: number
  /** Channels per pixel (1 = grayscale, 3 = RGB). */
  readonly channels: number
}

/** Everything model-specific lives here so swapping models touches one object. */
export interface LipModelSpec {
  /** Human label, e.g. "LipNet (GRID)". */
  readonly name: string
  /** Number of frames in one inference window (temporal length T). */
  readonly windowFrames: number
  /** Target capture rate the window assumes (fps). */
  readonly targetFps: number
  /** Mouth crop width fed to the model. */
  readonly cropWidth: number
  /** Mouth crop height fed to the model. */
  readonly cropHeight: number
  /** 1 = grayscale, 3 = RGB. */
  readonly channels: number
  /** Per-channel normalization applied after scaling pixels to [0, 1]. */
  readonly mean: readonly number[]
  readonly std: readonly number[]
  /**
   * Output label alphabet, index-aligned with the model's logits.
   * The CTC blank token must be included (see `blankIndex`).
   */
  readonly charset: readonly string[]
  /** Index of the CTC blank symbol within `charset`. */
  readonly blankIndex: number
  /** Path to the .onnx file served from /public. */
  readonly modelUrl: string
}

/** Result of running the model over one window. */
export interface LipReaderResult {
  readonly text: string
  /** 0..1 if the engine can estimate it, otherwise undefined. */
  readonly confidence?: number
}

/**
 * The pluggable recognition backend. Implementations: ONNX (real) and Mock.
 * Lifecycle: `init()` once → many `infer()` calls → `dispose()`.
 */
export interface LipReaderEngine {
  readonly name: string
  /** Whether this engine is backed by a real model (vs. a mock/stub). */
  readonly isReal: boolean
  init(): Promise<void>
  /** Run recognition over exactly `spec.windowFrames` frames. */
  infer(window: readonly LipFrame[]): Promise<LipReaderResult>
  dispose(): void
}
