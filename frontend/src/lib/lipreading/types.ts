/**
 * Core contracts for the lip-reading pipeline (see .context/phase-a-design.md).
 *
 * Capture → face detector (4 keypoints) → utterance recorder → crop pipeline (exact port of the
 * Python preprocessing) → Recognizer (local ONNX "speed" or hosted "accuracy") → transcript.
 * The UI depends only on these types, never on a concrete model.
 */

/** [x, y] in source-frame pixels. */
export type Point = readonly [number, number]

/**
 * The 4 BlazeFace keypoints the model's crop is defined by, in this order:
 * right eye, left eye, nose tip, mouth centre (MediaPipe FaceKeyPoint 0..3).
 * Pixel coordinates truncated to integers, like the Python `int(x * iw)`.
 */
export type Keypoints = readonly [Point, Point, Point, Point]

/** A frame a detector can read: the live video, a canvas, or (in a worker) a transferred bitmap. */
export type FrameSource =
  | HTMLVideoElement
  | HTMLCanvasElement
  | ImageBitmap
  | OffscreenCanvas

/** Finds the 4 keypoints of the largest face in a video frame. */
export interface MouthDetector {
  init(): Promise<void>
  /** `timestampMs` must increase monotonically (MediaPipe VIDEO mode). */
  detect(source: FrameSource, timestampMs: number): Keypoints | null
  dispose(): void
}

/** One captured video frame inside an utterance. */
export interface CapturedFrame {
  /** Capture time in ms (performance.now() or video time × 1000). */
  readonly tMs: number
  readonly width: number
  readonly height: number
  /** Grayscale pixels, round(0.299R + 0.587G + 0.114B), row-major, length width*height. */
  readonly gray: Uint8Array
  /** null when no face was detected in this frame (or, with `tracked: false`, not looked for). */
  readonly keypoints: Keypoints | null
  /**
   * false when face detection skipped this frame (the background tracker was busy): its keypoints
   * get interpolated like a missed detection, but it doesn't count against face coverage.
   */
  readonly tracked?: boolean
}

/** A finished push-to-talk segment. */
export interface Utterance {
  readonly frames: readonly CapturedFrame[]
  readonly startedAt: number
  readonly endedAt: number
}

/** Output of the crop pipeline: 25 fps aligned mouth crops. */
export interface CropResult {
  /** One 96*96 uint8 grayscale patch per output frame (25 fps). */
  readonly patches: readonly Uint8Array[]
  /** Fraction of captured frames that had a face (0..1). */
  readonly faceCoverage: number
  /** Smoothed keypoints actually used per output frame (debug / parity). */
  readonly keypoints: readonly Keypoints[]
}

/** Thrown by the crop pipeline when < 50% of frames have a face. */
export class NoFaceError extends Error {
  constructor(message = "no usable face track in the clip") {
    super(message)
    this.name = "NoFaceError"
  }
}

export type RecognitionMode = "speed" | "accuracy"

/** One beam-search reading; a higher `score` is better (log-probability scale, not 0..1). */
export interface Alternative {
  readonly text: string
  readonly score: number
}

/** A saved phrase's model score: (log P(phrase) − log P(reading)) per frame; higher = likelier. */
export interface PhraseScore {
  readonly text: string
  readonly margin: number
}

export interface WordConfidence {
  readonly text: string
  readonly confidence: number
}

export interface RecognitionResult {
  /** Text as returned by the model (uppercase SentencePiece output). */
  readonly text: string
  /** 0..1 when the recognizer can estimate it (greedy CTC), else undefined. */
  readonly confidence?: number
  /**
   * Beam search only: up to 3 distinct readings, best first ([0] is `text`), for the "pick the
   * right one" UI. Empty or absent for greedy recognizers.
   */
  readonly alternatives?: readonly Alternative[]
  /** Per word of `text`, how sure the reader was (0..1): CTC frame probs, or n-best agreement. */
  readonly words?: readonly WordConfidence[]
  /**
   * Rank saved phrases by how well the model thinks each explains these frames (CTC margin,
   * `phrases/ctcScore.ts` on-device, `POST /lipread/phrases` for server reads); null when the
   * scorer can't load or the server can't score.
   */
  readonly scorePhrases?: (
    reading: string,
    phrases: readonly string[]
  ) => Promise<readonly PhraseScore[] | null>
  /** Which mode actually produced this result. */
  readonly mode: RecognitionMode
  /** True when `accuracy` was requested but we fell back to `speed`. */
  readonly fellBack?: boolean
  /** Wall-clock ms for recognize() (excludes capture), plus optional server breakdown. */
  readonly latencyMs: number
  readonly serverLatencyMs?: Readonly<Record<string, number>>
  /** Human-readable engine label, e.g. "ONNX · webgpu" or "RunPod · beam". */
  readonly engine: string
}

/**
 * A recognition backend. Implementations: OnnxRecognizer (speed), HttpRecognizer (accuracy),
 * MockRecognizer (neither available). Lifecycle: init() once → many recognize() → dispose().
 */
export interface Recognizer {
  readonly mode: RecognitionMode
  readonly name: string
  /** False if init failed or the backend is not configured (e.g. model file / URL missing). */
  readonly available: boolean
  readonly isReal: boolean
  init(): Promise<void>
  recognize(crops: CropResult, signal?: AbortSignal): Promise<RecognitionResult>
  dispose(): void
}

/** Model + preprocessing constants shared by the crop pipeline and the local recognizer. */
export interface LipModelSpec {
  readonly name: string
  readonly fps: 25
  /** Patch size produced by the crop pipeline. */
  readonly patchSize: 96
  /** Centre crop fed to the model. */
  readonly inputSize: 88
  readonly mean: number
  readonly std: number
  readonly blankIndex: 0
  readonly modelUrl: string
  readonly tokensUrl: string
  readonly minSeconds: number
  readonly maxSeconds: number
  readonly minFaceCoverage: number
}
