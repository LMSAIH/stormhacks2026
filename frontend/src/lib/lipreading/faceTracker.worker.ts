/**
 * Background worker running both face trackers off the main thread, so the camera loop keeps its
 * frame rate (plan D52). Per frame (an `ImageBitmap` transferred in):
 * - BlazeFace → the 4 keypoints the model crop is defined by (same `BlazeFaceDetector` as before);
 * - the teammate's lip tracking: FaceLandmarker with his settings (GPU delegate, 1 face) and his 40
 *   `LIP_LANDMARK_INDICES` for the overlay dots, plus mouth openness for pause detection.
 * His `LipLandmarker` can't run here (it makes a DOM canvas for its unused crop), so this builds the
 * same graph directly; `faceLandmarker.ts` stays untouched.
 */
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision"

import { BlazeFaceDetector, type WasmFileset } from "./faceDetector"
import { LIP_LANDMARK_INDICES } from "./faceLandmarker"
import type { TrackerRequest, TrackerResponse } from "./faceTracker"

const WASM_BASE = "/mediapipe/wasm"
const LANDMARKER_URL = "/mediapipe/face_landmarker.task"
// Face-mesh points for mouth openness: inner lip centre top/bottom, and the mouth corners.
const INNER_TOP = 13
const INNER_BOTTOM = 14
const CORNER_LEFT = 61
const CORNER_RIGHT = 291

const ctx = self as unknown as {
  postMessage(message: TrackerResponse): void
  onmessage: ((event: MessageEvent<TrackerRequest>) => void) | null
}

let blazeFace: BlazeFaceDetector | null = null
let landmarker: FaceLandmarker | null = null
let lastLipTs = Number.NEGATIVE_INFINITY

type Factory = (...args: unknown[]) => unknown
const scope = self as unknown as { ModuleFactory?: Factory }

/**
 * MediaPipe loads its WASM loader with a dynamic `import()` in module workers, which Vite's dev
 * server refuses for files under public/. Load the ES-module loader ourselves instead, and hand
 * MediaPipe a fileset without a loader path: it then uses the global `ModuleFactory` we set
 * (and clears it after each task, hence `withFactory` around every create).
 */
async function loadFileset(): Promise<{ fileset: WasmFileset; factory: Factory }> {
  const resolved = await FilesetResolver.forVisionTasks(WASM_BASE, true) // the ES-module variant
  const loaderUrl = new URL(resolved.wasmLoaderPath, self.location.href).href
  const loader = (await import(/* @vite-ignore */ loaderUrl)) as { default: Factory }
  return {
    fileset: { ...resolved, wasmLoaderPath: "" } as WasmFileset,
    factory: loader.default,
  }
}

async function withFactory<T>(factory: Factory, create: () => Promise<T>): Promise<T> {
  scope.ModuleFactory = factory
  return create()
}

async function createLandmarker(fileset: WasmFileset, factory: Factory): Promise<FaceLandmarker> {
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: LANDMARKER_URL, delegate },
    runningMode: "VIDEO" as const,
    numFaces: 1,
  })
  try {
    return await withFactory(factory, () => FaceLandmarker.createFromOptions(fileset, options("GPU")))
  } catch {
    // no GPU in this worker; the failed attempt may have consumed the factory, so set it again
    return withFactory(factory, () => FaceLandmarker.createFromOptions(fileset, options("CPU")))
  }
}

function track(bitmap: ImageBitmap, tMs: number): TrackerResponse {
  const started = performance.now()
  const keypoints = blazeFace?.detect(bitmap, tMs) ?? null
  let lipPoints: { x: number; y: number }[] = []
  let openness: number | null = null
  if (landmarker) {
    lastLipTs = tMs > lastLipTs ? tMs : lastLipTs + 1 // VIDEO mode wants increasing timestamps
    try {
      const face = landmarker.detectForVideo(bitmap, lastLipTs).faceLandmarks?.[0]
      if (face) {
        lipPoints = LIP_LANDMARK_INDICES.map((i) => ({ x: face[i].x, y: face[i].y }))
        // Pixel distances: landmarks are normalized per axis, so scale by the frame size.
        const dist = (a: number, b: number) =>
          Math.hypot((face[a].x - face[b].x) * bitmap.width, (face[a].y - face[b].y) * bitmap.height)
        const width = dist(CORNER_LEFT, CORNER_RIGHT)
        openness = width > 0 ? dist(INNER_TOP, INNER_BOTTOM) / width : null
      }
    } catch {
      // a failed frame just has no lip points
    }
  }
  bitmap.close()
  return { type: "result", tMs, keypoints, lipPoints, openness, detectMs: performance.now() - started }
}

ctx.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === "init") {
    try {
      const { fileset, factory } = await loadFileset()
      const detector = new BlazeFaceDetector({ fileset })
      await withFactory(factory, () => detector.init()) // one at a time: each create clears the factory
      blazeFace = detector
      landmarker = await createLandmarker(fileset, factory)
      ctx.postMessage({ type: "ready" })
    } catch (err) {
      ctx.postMessage({ type: "error", message: String(err) })
    }
  } else if (msg.type === "frame") {
    ctx.postMessage(track(msg.bitmap, msg.tMs))
  } else if (msg.type === "dispose") {
    blazeFace?.dispose()
    blazeFace = null
    landmarker?.close()
    landmarker = null
  }
}
