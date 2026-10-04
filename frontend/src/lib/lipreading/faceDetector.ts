import {
  FaceDetector,
  FilesetResolver,
  type Detection,
  type FaceDetectorResult,
} from "@mediapipe/tasks-vision"

import type { Keypoints, MouthDetector, Point } from "./types"

/** Served from /public like the rest of the MediaPipe runtime — no CDN. */
const WASM_BASE = "/mediapipe/wasm"
const MODEL_URL = "/mediapipe/blaze_face_short_range.tflite"

export interface BlazeFaceDetectorOptions {
  /** Same threshold as the Python `FaceDetection(min_detection_confidence=0.5)`. */
  minDetectionConfidence?: number
  /** CPU stays closest to the Python (TFLite CPU) reference; BlazeFace is tiny anyway. */
  delegate?: "CPU" | "GPU"
  modelUrl?: string
  wasmBase?: string
}

/**
 * MediaPipe Tasks FaceDetector (BlazeFace short-range, VIDEO mode) → the 4 keypoints the crop
 * pipeline is defined by, as integer source pixels. JS twin of
 * `ml/src/lipread/vendor/mediapipe/detector.py`.
 *
 * Keypoint order: tasks-vision only types `Detection.keypoints` as `NormalizedKeypoint[]` (labels
 * come back empty) and copies them in model output order, which is BlazeFace's FaceKeyPoint order:
 * 0 right eye, 1 left eye, 2 nose tip, 3 mouth centre (4/5 ear tragions, unused). "Right" is the
 * subject's right = smaller x in the un-mirrored camera frame, matching STABLE_REFERENCE. Checked
 * with the Tasks FaceDetector (same graph as this WASM build) + this .tflite on square and 4:3
 * frontal photos: identical integer points to the legacy Python solution.
 *
 * Short-range only. Python's `MouthCropper` also runs short-range first (full-range only when
 * short-range finds a face in < 50% of frames, D36), so both sides crop from the same detector:
 * on the CPU delegate the keypoints match Python's at integer level on identical frames.
 */
export class BlazeFaceDetector implements MouthDetector {
  private readonly options: Required<BlazeFaceDetectorOptions>
  private detector: FaceDetector | null = null
  private disposed = false
  private lastTimestampMs = Number.NEGATIVE_INFINITY
  private warned = false

  constructor(options: BlazeFaceDetectorOptions = {}) {
    this.options = {
      minDetectionConfidence: options.minDetectionConfidence ?? 0.5,
      delegate: options.delegate ?? "CPU",
      modelUrl: options.modelUrl ?? MODEL_URL,
      wasmBase: options.wasmBase ?? WASM_BASE,
    }
  }

  get ready(): boolean {
    return this.detector !== null
  }

  async init(): Promise<void> {
    if (this.detector || this.disposed) return
    const { wasmBase, modelUrl, delegate, minDetectionConfidence } =
      this.options
    const fileset = await FilesetResolver.forVisionTasks(wasmBase)
    const detector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate },
      runningMode: "VIDEO",
      minDetectionConfidence,
    })
    // dispose() can land while we were loading (React StrictMode unmounts immediately).
    if (this.disposed) {
      detector.close()
      return
    }
    this.detector = detector
  }

  detect(
    source: HTMLVideoElement | HTMLCanvasElement,
    timestampMs: number
  ): Keypoints | null {
    const detector = this.detector
    if (!detector) return null
    const [width, height] = sourceSize(source)
    if (width === 0 || height === 0) return null

    // VIDEO mode throws on non-increasing timestamps (looping clips, duplicate frames): nudge.
    const ts =
      timestampMs > this.lastTimestampMs
        ? timestampMs
        : this.lastTimestampMs + 1
    this.lastTimestampMs = ts

    let result: FaceDetectorResult
    try {
      result = detector.detectForVideo(source, ts)
    } catch (err) {
      if (!this.warned) {
        this.warned = true
        console.warn("[BlazeFaceDetector] detectForVideo failed:", err)
      }
      return null
    }

    const face = largestFace(result.detections)
    if (!face || face.keypoints.length < 4) return null
    const k = face.keypoints
    // Python: int(x * iw) — truncation, not rounding.
    const px = (i: number): Point => [
      Math.trunc(k[i].x * width),
      Math.trunc(k[i].y * height),
    ]
    return [px(0), px(1), px(2), px(3)]
  }

  dispose(): void {
    this.disposed = true
    this.detector?.close()
    this.detector = null
  }
}

function sourceSize(
  source: HTMLVideoElement | HTMLCanvasElement
): [number, number] {
  return "videoWidth" in source
    ? [source.videoWidth, source.videoHeight]
    : [source.width, source.height]
}

/**
 * Largest face by box area. (The Python detector ranks by `(w - x) + (h - y)`, an upstream quirk;
 * identical whenever only one face is in view, which is the supported case.)
 */
function largestFace(detections: readonly Detection[]): Detection | null {
  let best: Detection | null = null
  let bestArea = -1
  for (const d of detections) {
    const box = d.boundingBox
    const area = box ? box.width * box.height : 0
    if (area > bestArea) {
      best = d
      bestArea = area
    }
  }
  return best
}
