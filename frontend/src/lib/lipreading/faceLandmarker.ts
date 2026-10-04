import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision"

import type {
  LipTrackingFrame as LipFrame,
  LipTrackingSpec as LipModelSpec,
} from "./lipTrackingTypes"

/** Assets are served locally from /public (copied at setup) — no CDN needed. */
const WASM_BASE = "/mediapipe/wasm"
const MODEL_URL = "/mediapipe/face_landmarker.task"

/**
 * MediaPipe Face Mesh lip landmark indices (outer + inner lip contour).
 * Used both to draw the debug overlay and to compute the mouth crop box.
 */
export const LIP_LANDMARK_INDICES: readonly number[] = [
  // outer lips
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
  39, 40, 185,
  // inner lips
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82,
  81, 80, 191,
]

export interface MouthDetection {
  /** Preprocessed crop ready for the ring buffer, or null if no face. */
  readonly frame: LipFrame | null
  /** Normalized (0..1) lip points for overlay drawing. */
  readonly lipPoints: readonly { x: number; y: number }[]
}

/**
 * Wraps MediaPipe Face Landmarker: detect a face in a video frame, extract the
 * normalized lip landmarks, and produce a model-ready mouth crop.
 */
export class LipLandmarker {
  private readonly spec: LipModelSpec
  private landmarker: FaceLandmarker | null = null
  // Reused scratch canvas for cropping/resizing — avoids per-frame allocation.
  private readonly cropCanvas = document.createElement("canvas")
  private readonly cropCtx: CanvasRenderingContext2D

  constructor(spec: LipModelSpec) {
    this.spec = spec
    this.cropCanvas.width = spec.cropWidth
    this.cropCanvas.height = spec.cropHeight
    const ctx = this.cropCanvas.getContext("2d", { willReadFrequently: true })
    if (!ctx) throw new Error("2D canvas context unavailable")
    this.cropCtx = ctx
  }

  async init(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    })
  }

  /**
   * Detect + crop the mouth region for one video frame.
   * `timestampMs` must be monotonically increasing (MediaPipe VIDEO mode).
   */
  detect(video: HTMLVideoElement, timestampMs: number): MouthDetection {
    if (!this.landmarker) {
      return { frame: null, lipPoints: [] }
    }

    let result: FaceLandmarkerResult
    try {
      result = this.landmarker.detectForVideo(video, timestampMs)
    } catch {
      return { frame: null, lipPoints: [] }
    }

    const faces = result.faceLandmarks
    if (!faces || faces.length === 0) {
      return { frame: null, lipPoints: [] }
    }

    const landmarks = faces[0]
    const lipPoints = LIP_LANDMARK_INDICES.map((i) => ({
      x: landmarks[i].x,
      y: landmarks[i].y,
    }))

    const frame = this.cropMouth(video, lipPoints)
    return { frame, lipPoints }
  }

  /** Crop a padded box around the lips, resize to the model input, normalize. */
  private cropMouth(
    video: HTMLVideoElement,
    lipPoints: readonly { x: number; y: number }[]
  ): LipFrame | null {
    const vw = video.videoWidth
    const vh = video.videoHeight
    if (vw === 0 || vh === 0) return null

    let minX = 1
    let minY = 1
    let maxX = 0
    let maxY = 0
    for (const p of lipPoints) {
      if (p.x < minX) minX = p.x
      if (p.y < minY) minY = p.y
      if (p.x > maxX) maxX = p.x
      if (p.y > maxY) maxY = p.y
    }

    // Pad the box around the mouth so we keep context (teeth, jaw motion).
    const padX = (maxX - minX) * 0.35
    const padY = (maxY - minY) * 0.5
    const sx = Math.max(0, (minX - padX) * vw)
    const sy = Math.max(0, (minY - padY) * vh)
    const sw = Math.min(vw - sx, (maxX - minX + 2 * padX) * vw)
    const sh = Math.min(vh - sy, (maxY - minY + 2 * padY) * vh)
    if (sw <= 0 || sh <= 0) return null

    const { cropWidth, cropHeight, channels, mean, std } = this.spec
    this.cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, cropWidth, cropHeight)
    const { data } = this.cropCtx.getImageData(0, 0, cropWidth, cropHeight)

    const out = new Float32Array(cropWidth * cropHeight * channels)
    if (channels === 1) {
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        // Rec. 601 luma.
        const gray = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) / 255
        out[p] = (gray - mean[0]) / std[0]
      }
    } else {
      for (let i = 0, p = 0; i < data.length; i += 4, p += 3) {
        out[p] = (data[i] / 255 - mean[0]) / std[0]
        out[p + 1] = (data[i + 1] / 255 - mean[1]) / std[1]
        out[p + 2] = (data[i + 2] / 255 - mean[2]) / std[2]
      }
    }

    return { data: out, width: cropWidth, height: cropHeight, channels }
  }

  dispose(): void {
    this.landmarker?.close()
    this.landmarker = null
  }
}
