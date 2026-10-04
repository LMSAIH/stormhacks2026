import { useCallback, useEffect, useRef, useState } from "react"

import { cropUtterance, rgbaToGray } from "@/lib/lipreading/crop"
import {
  startRecognizers,
  type Recognizers,
  type RecognizersLoading,
} from "@/lib/lipreading/createRecognizers"
import { BlazeFaceDetector } from "@/lib/lipreading/faceDetector"
import { LipLandmarker } from "@/lib/lipreading/faceLandmarker"
import { toSentenceCase } from "@/lib/lipreading/format"
import { LIP_TRACKING_SPEC } from "@/lib/lipreading/lipTrackingTypes"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"
import { drawFaceOverlay, type NormalizedPoint } from "@/lib/lipreading/overlay"
import {
  NoFaceError,
  type CapturedFrame,
  type CropResult,
  type Keypoints,
  type RecognitionMode,
  type RecognitionResult,
  type Recognizer,
  type Utterance,
} from "@/lib/lipreading/types"

export type CameraStatus = "idle" | "starting" | "on" | "error"

export interface EngineStatus {
  name: string
  available: boolean
  isReal: boolean
  /** init() still running (the local model can take minutes on a cold cache). */
  loading: boolean
}

export interface LipTranscriptItem {
  id: string
  /** What to show / speak: the recognizer output read as a sentence ("I THINK" → "I think"). */
  text: string
  /** Recognizer output as returned (uppercase SentencePiece text). */
  raw: string
  /** performance.now() at utterance start — orders it alongside listening utterances. */
  at: number
  /** Mode that actually produced the text (see `fellBack`). */
  mode: RecognitionMode
  latencyMs: number
  /** Accuracy was requested but failed/unavailable, so speed produced this. */
  fellBack?: boolean
  engine: string
  confidence?: number
}

type EngineFlags = Record<RecognitionMode, boolean>

interface ActiveRecording {
  readonly startedAt: number
  readonly frames: CapturedFrame[]
}

const MODE_STORAGE_KEY = "lipread.mode"
const MODE_LABEL: Record<RecognitionMode, string> = {
  speed: "Speed",
  accuracy: "Accuracy",
}
const UNAVAILABLE_MESSAGE: Record<RecognitionMode, string> = {
  speed: "On-device model unavailable",
  accuracy: "Accuracy server unavailable",
}
const NO_FACE_MESSAGE = "No face — keep your face in frame"
const NO_LIP_POINTS: readonly NormalizedPoint[] = []
/**
 * While recording, the lip tracking (the costly graph: ~25 ms a frame on a laptop GPU) runs on
 * every 3rd frame only, so the capture rate the 25 fps resample relies on holds up; its dots reuse
 * the last points in between. While idle it runs on every frame.
 */
const LIP_TRACKING_RECORDING_STRIDE = 3
/** Keep the 25 fps resample at ≤ maxSeconds·fps frames (the server's cap) despite capture jitter. */
const MAX_SPAN_MS = ACTIVE_SPEC.maxSeconds * 1000 - 1000 / ACTIVE_SPEC.fps
const MODES: readonly RecognitionMode[] = ["speed", "accuracy"]
const ENGINES_LOADING: Record<RecognitionMode, EngineStatus> = {
  speed: { name: "loading…", available: false, isReal: false, loading: true },
  accuracy: {
    name: "loading…",
    available: false,
    isReal: false,
    loading: true,
  },
}

/** Accuracy was requested, the service can't serve it and there is no local model to fall back to. */
class NoEngineError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "NoEngineError"
  }
}

export interface UseLipReaderOptions {
  /**
   * Whether push-to-talk and recognition are on (default true). While false, Space and the button
   * do nothing, an utterance being recorded is discarded and one still being recognized is
   * dropped. The camera, face detector and overlay keep running, so switching back on is instant.
   */
  active?: boolean
}

/**
 * Camera + push-to-talk lip reading (.context/phase-a-design.md §1–2, §9):
 * camera → two MediaPipe graphs per frame: BlazeFace (the 4 keypoints the model crop is defined
 * by) and the FaceLandmarker lip tracking (the lip dots on the overlay, `mouthDetected`; only every
 * 3rd frame while recording, to keep the capture rate up) → while the user holds Space or the
 * on-screen button, gray frames + keypoints are buffered → on release the utterance is cropped
 * once and sent to the recognizer for the current mode (accuracy falls back to speed on failure)
 * → one transcript item per utterance.
 *
 * Call it as `useLipReader()` or `useLipReader({ active })`. `inferring` is `busy` under the name
 * the app screen uses for "a recognition is in flight".
 */
export function useLipReader({ active = true }: UseLipReaderOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)

  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle")
  const [ready, setReady] = useState(false)
  /** Lips found by the lip tracking (BlazeFace's face while that is unavailable). */
  const [mouthDetected, setMouthDetected] = useState(false)
  const [fps, setFps] = useState(0)
  const [mode, setModeState] = useState<RecognitionMode>(readStoredMode)
  const [engines, setEngines] = useState(ENGINES_LOADING)
  const [enginesReady, setEnginesReady] = useState(false)
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<LipTranscriptItem[]>([])

  // Pipeline state the frame loop and event handlers read without re-rendering.
  const detectorRef = useRef<BlazeFaceDetector | null>(null)
  const loadRef = useRef<RecognizersLoading | null>(null)
  /** Latest engine pair: the ones loading, then the final pair (speed may become a mock). */
  const recognizersRef = useRef<Recognizers | null>(null)
  /** Which engines have finished init(). */
  const settledRef = useRef<EngineFlags>({ speed: false, accuracy: false })
  const recordingRef = useRef<ActiveRecording | null>(null)
  const autoStopTimerRef = useRef<number | undefined>(undefined)
  const busyRef = useRef(false)
  const modeRef = useRef(mode)
  const activeRef = useRef(active)
  const abortRef = useRef<AbortController | null>(null)

  /** Re-read engine status from the live recognizers (names / availability change on use). */
  const showEngines = useCallback(() => {
    const recognizers = recognizersRef.current
    if (!recognizers) return
    setEngines((prev) => refreshEngines(prev, recognizers, settledRef.current))
  }, [])

  const setMode = useCallback(
    (next: RecognitionMode) => {
      const select = () => {
        modeRef.current = next
        setModeState(next)
        try {
          window.localStorage.setItem(MODE_STORAGE_KEY, next)
        } catch {
          // Storage blocked (private mode, sandboxed iframe): the choice just isn't remembered.
        }
      }
      const recognizers = recognizersRef.current
      // Still loading, or usable: switch now (an utterance waits for a loading engine).
      if (
        !recognizers ||
        !settledRef.current[next] ||
        recognizers[next].available
      ) {
        select()
        return
      }
      // Unavailable engine: re-probe first (the hosted pod may have come up since page load).
      const engine = recognizers[next]
      const signal = abortRef.current?.signal
      setLastError(`Checking ${MODE_LABEL[next]} engine…`)
      engine
        .init()
        .catch(() => undefined)
        .then(() => {
          if (signal?.aborted) return
          showEngines()
          if (engine.available) {
            select()
            setLastError(null)
          } else {
            setLastError(UNAVAILABLE_MESSAGE[next])
          }
        })
    },
    [showEngines]
  )

  const clearTranscript = useCallback(() => setTranscript([]), [])

  const recognizeUtterance = useCallback(
    async (utterance: Utterance) => {
      const signal = abortRef.current?.signal
      busyRef.current = true
      setBusy(true)
      try {
        // Let the busy state paint before the synchronous crop work.
        await new Promise((resolve) => window.setTimeout(resolve, 0))
        let crops: CropResult
        try {
          crops = cropUtterance(utterance, ACTIVE_SPEC)
        } catch (err) {
          if (isNoFaceError(err)) {
            setLastError(NO_FACE_MESSAGE)
            return
          }
          throw err
        }

        const load = loadRef.current
        if (!load) return
        // Waits only for the engine(s) this utterance needs: accuracy is usable after one health
        // check, speed once the local model has loaded.
        const result = await recognizeWithFallback(
          load,
          modeRef.current,
          crops,
          signal
        )
        if (signal?.aborted) return
        showEngines()
        // Switched off while this was in flight: drop it, like an utterance that was never made.
        if (!activeRef.current) return

        const raw = result.text.trim()
        if (!raw) {
          setLastError("Didn't catch that — try again")
          return
        }
        setTranscript((prev) => [
          ...prev,
          {
            id: `lip-${Math.round(utterance.startedAt)}-${prev.length}`,
            text: toSentenceCase(raw),
            raw,
            at: utterance.startedAt,
            mode: result.mode,
            latencyMs: result.latencyMs,
            fellBack: result.fellBack,
            engine: result.engine,
            confidence: result.confidence,
          },
        ])
      } catch (err) {
        if (signal?.aborted) return
        console.error("[useLipReader] recognition failed:", err)
        showEngines()
        setLastError(
          err instanceof NoEngineError
            ? err.message
            : err instanceof Error && err.name === "HttpRecognizerError"
              ? "Accuracy server failed (on-device model not loaded)"
              : "Recognition failed — try again"
        )
      } finally {
        busyRef.current = false
        setBusy(false)
      }
    },
    [showEngines]
  )

  const stopUtterance = useCallback(() => {
    const current = recordingRef.current
    if (!current) return
    recordingRef.current = null
    window.clearTimeout(autoStopTimerRef.current)
    setRecording(false)

    const { frames, startedAt } = current
    const spanMs =
      frames.length >= 2 ? frames[frames.length - 1].tMs - frames[0].tMs : 0
    if (spanMs < ACTIVE_SPEC.minSeconds * 1000) {
      setLastError("Too short — hold while you speak")
      return
    }
    void recognizeUtterance({ frames, startedAt, endedAt: performance.now() })
  }, [recognizeUtterance])

  /** Throw away the utterance being recorded without recognizing it. */
  const cancelUtterance = useCallback(() => {
    if (!recordingRef.current) return
    recordingRef.current = null
    window.clearTimeout(autoStopTimerRef.current)
    setRecording(false)
  }, [])

  const startUtterance = useCallback(() => {
    // One utterance at a time; nothing to record while switched off or before the detector is up.
    if (
      !activeRef.current ||
      recordingRef.current ||
      busyRef.current ||
      !detectorRef.current
    ) {
      return
    }
    recordingRef.current = { startedAt: performance.now(), frames: [] }
    // Backstop for the frame-time cap in the capture loop (frames stop when the tab is hidden).
    window.clearTimeout(autoStopTimerRef.current)
    autoStopTimerRef.current = window.setTimeout(
      stopUtterance,
      ACTIVE_SPEC.maxSeconds * 1000 + 250
    )
    setLastError(null)
    setRecording(true)
  }, [stopUtterance])

  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  // Handlers read `active` through the ref; leaving the active state discards a recording.
  useEffect(() => {
    activeRef.current = active
    if (!active) return
    return cancelUtterance
  }, [active, cancelUtterance])

  // Lets in-flight recognition notice unmount (and HTTP requests get cancelled).
  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller
    return () => {
      controller.abort()
      abortRef.current = null
    }
  }, [])

  // --- Recognizers (speed = local ONNX, accuracy = hosted), each reported as soon as it is up ---
  useEffect(() => {
    let cancelled = false
    const load = acquireRecognizers()
    const settled: EngineFlags = { speed: false, accuracy: false }
    loadRef.current = load
    recognizersRef.current = load.engines
    settledRef.current = settled

    for (const m of MODES) {
      void load.ready[m].then(() => {
        if (cancelled) return
        settled[m] = true
        showEngines()
      })
    }
    load.done.then(
      (recognizers) => {
        if (cancelled) return
        settled.speed = settled.accuracy = true
        recognizersRef.current = recognizers // speed may have become the mock
        showEngines()
        setEnginesReady(true)
        // Remembered mode unavailable here (no hosted URL / no local model): use the other one
        // for this session without overwriting the stored preference.
        const current = modeRef.current
        const other: RecognitionMode =
          current === "speed" ? "accuracy" : "speed"
        if (!recognizers[current].available && recognizers[other].available) {
          modeRef.current = other
          setModeState(other)
        }
      },
      (err: unknown) => {
        if (cancelled) return
        console.error("[useLipReader] recognizers failed to load:", err)
        const failed: EngineStatus = {
          name: "failed to load",
          available: false,
          isReal: false,
          loading: false,
        }
        setEngines({ speed: failed, accuracy: failed })
        setEnginesReady(true)
        setLastError("Recognition engines failed to load")
      }
    )
    return () => {
      cancelled = true
      loadRef.current = null
      recognizersRef.current = null
      settledRef.current = { speed: false, accuracy: false }
      releaseRecognizers()
    }
  }, [showEngines])

  // --- Camera + face detector + lip tracking + per-frame capture loop ---
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let cancelled = false
    let stream: MediaStream | null = null
    let videoFrameHandle: number | null = null
    let animationFrameHandle: number | null = null
    let lastVideoTime = -1
    let lastMouth = false
    let lastFpsUpdate = 0
    const frameTimes: number[] = []

    const detector = new BlazeFaceDetector()
    // The teammate's lip tracking: a second MediaPipe graph (FaceLandmarker mesh) over the same
    // frames. VIDEO mode wants strictly increasing timestamps per graph; BlazeFace nudges its own,
    // this one is nudged below. It only drives the lip dots and `mouthDetected`, never the crop.
    const lipTracker = new LipLandmarker(LIP_TRACKING_SPEC)
    let lipTrackerReady = false
    let lastLipTs = Number.NEGATIVE_INFINITY
    /** Latest lip points: drawn every frame, refreshed only on the frames the tracker runs. */
    let lipPoints = NO_LIP_POINTS
    let framesSinceLipTracking = 0
    // Reused full-resolution scratch canvas for reading pixels while recording.
    const scratch = document.createElement("canvas")
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true })

    const processFrame = (tMs: number) => {
      const width = video.videoWidth
      const height = video.videoHeight
      if (
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        width === 0
      ) {
        return
      }

      let keypoints: Keypoints | null
      const current = recordingRef.current
      if (current && scratchCtx) {
        const first = current.frames[0]
        if (first && tMs - first.tMs >= MAX_SPAN_MS) {
          stopUtterance()
          keypoints = detector.detect(video, tMs)
        } else {
          if (scratch.width !== width) scratch.width = width
          if (scratch.height !== height) scratch.height = height
          scratchCtx.drawImage(video, 0, 0, width, height)
          // Detect on the exact pixels we store, so keypoints and gray frame always match.
          keypoints = detector.detect(scratch, tMs)
          const last = current.frames[current.frames.length - 1]
          if (!last || tMs > last.tMs) {
            const { data } = scratchCtx.getImageData(0, 0, width, height)
            current.frames.push({
              tMs,
              width,
              height,
              gray: rgbaToGray(data, width, height),
              keypoints,
            })
          }
        }
      } else {
        keypoints = detector.detect(video, tMs)
      }

      // Every frame while idle; every LIP_TRACKING_RECORDING_STRIDE-th while recording.
      if (
        lipTrackerReady &&
        (!current || ++framesSinceLipTracking >= LIP_TRACKING_RECORDING_STRIDE)
      ) {
        framesSinceLipTracking = 0
        lastLipTs = tMs > lastLipTs ? tMs : lastLipTs + 1
        lipPoints = lipTracker.detect(video, lastLipTs).lipPoints
      }
      drawFaceOverlay(overlayRef.current, width, height, lipPoints, keypoints)

      // Lips found by the lip tracking; BlazeFace's face stands in while that isn't up.
      const mouth = lipTrackerReady ? lipPoints.length > 0 : keypoints !== null
      if (mouth !== lastMouth) {
        lastMouth = mouth
        setMouthDetected(mouth)
      }
      const now = performance.now()
      frameTimes.push(now)
      while (frameTimes.length > 0 && now - frameTimes[0] > 1000)
        frameTimes.shift()
      if (now - lastFpsUpdate >= 500) {
        lastFpsUpdate = now
        setFps(frameTimes.length)
      }
    }

    const useVideoFrameCallback =
      typeof video.requestVideoFrameCallback === "function"
    const scheduleNextFrame = () => {
      if (cancelled) return
      if (useVideoFrameCallback) {
        videoFrameHandle = video.requestVideoFrameCallback(onVideoFrame)
      } else {
        animationFrameHandle = requestAnimationFrame(onAnimationFrame)
      }
    }
    // One callback per new camera frame; captureTime (same clock as performance.now) gives the
    // most even spacing for the 25 fps resample.
    const onVideoFrame: VideoFrameRequestCallback = (now, metadata) => {
      videoFrameHandle = null
      processFrame(metadata.captureTime ?? metadata.expectedDisplayTime ?? now)
      scheduleNextFrame()
    }
    // rAF runs at display rate, faster than the camera: only process frames that advanced.
    const onAnimationFrame = (now: number) => {
      animationFrameHandle = null
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime
        processFrame(now)
      }
      scheduleNextFrame()
    }

    const start = async () => {
      setCameraStatus("starting")
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        })
      } catch (err) {
        if (cancelled) return
        console.error("[useLipReader] camera failed:", err)
        setCameraStatus("error")
        setLastError(cameraErrorMessage(err))
        return
      }
      if (cancelled) {
        stopTracks(stream)
        return
      }
      video.srcObject = stream
      await video.play().catch(() => undefined)
      if (cancelled) return
      setCameraStatus("on")
      scheduleNextFrame()

      // Both graphs load in parallel and each starts working as soon as it is up. Lip tracking is
      // the overlay only: if it can't load (say, no WebGL for its GPU delegate) the model
      // pipeline still works, so that is a warning, not a failed camera.
      void lipTracker.init().then(
        () => {
          // Unmounted while loading (StrictMode remount): never used, so close it here.
          if (cancelled) lipTracker.dispose()
          else lipTrackerReady = true
        },
        (err: unknown) => {
          if (!cancelled)
            console.warn("[useLipReader] lip tracking failed to load:", err)
        }
      )
      try {
        await detector.init()
      } catch (err) {
        if (cancelled) return
        console.error("[useLipReader] face detector failed to load:", err)
        setLastError("Face detector failed to load")
        return
      }
      if (cancelled) return
      detectorRef.current = detector
      setReady(true)
    }

    void start()

    return () => {
      cancelled = true
      if (videoFrameHandle !== null)
        video.cancelVideoFrameCallback(videoFrameHandle)
      if (animationFrameHandle !== null)
        cancelAnimationFrame(animationFrameHandle)
      stopTracks(stream)
      video.srcObject = null
      detector.dispose()
      lipTrackerReady = false
      lipTracker.dispose()
      detectorRef.current = null
      recordingRef.current = null
      window.clearTimeout(autoStopTimerRef.current)
      setReady(false)
      setRecording(false)
      setCameraStatus("idle")
    }
  }, [stopUtterance])

  // --- Push-to-talk on Space (outside text fields); not even listening while inactive, so Space
  // keeps its normal page behaviour then ---
  useEffect(() => {
    if (!active) return
    let spaceHeld = false
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.code !== "Space" ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey ||
        e.isComposing
      ) {
        return
      }
      if (isEditableTarget(e.target)) return
      e.preventDefault() // no page scroll, no click on a focused button
      if (e.repeat || spaceHeld) return
      spaceHeld = true
      startUtterance()
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== "Space" || !spaceHeld) return
      e.preventDefault()
      spaceHeld = false
      stopUtterance()
    }
    // Key-up never arrives if the window loses focus mid-utterance.
    const onBlur = () => {
      if (!spaceHeld) return
      spaceHeld = false
      stopUtterance()
    }
    // Capture phase so focused widgets can't swallow Space before us.
    window.addEventListener("keydown", onKeyDown, true)
    window.addEventListener("keyup", onKeyUp, true)
    window.addEventListener("blur", onBlur)
    return () => {
      window.removeEventListener("keydown", onKeyDown, true)
      window.removeEventListener("keyup", onKeyUp, true)
      window.removeEventListener("blur", onBlur)
    }
  }, [active, startUtterance, stopUtterance])

  return {
    videoRef,
    overlayRef,
    cameraStatus,
    ready,
    mouthDetected,
    fps,
    mode,
    setMode,
    engines,
    enginesReady,
    recording,
    startUtterance,
    stopUtterance,
    busy,
    /** Alias of `busy`: a recognition is in flight. */
    inferring: busy,
    lastError,
    transcript,
    clearTranscript,
  }
}

/**
 * Recognize with the requested mode. Accuracy waits only for its own health check; if the service
 * is unavailable or the request fails it falls back to speed (waiting for the local model if it is
 * still loading) and marks the result `fellBack`.
 */
async function recognizeWithFallback(
  load: RecognizersLoading,
  mode: RecognitionMode,
  crops: CropResult,
  signal?: AbortSignal
): Promise<RecognitionResult> {
  if (mode === "accuracy") {
    await load.ready.accuracy
    signal?.throwIfAborted()
    const { accuracy } = load.engines
    let failure: unknown = null
    if (accuracy.available) {
      try {
        return await accuracy.recognize(crops, signal)
      } catch (err) {
        if (signal?.aborted) throw err
        failure = err
      }
    }
    const { speed } = await load.done
    signal?.throwIfAborted()
    // Nothing to fall back to: surface the hosted error.
    if (!speed.available) {
      throw (
        failure ??
        new NoEngineError(
          "Accuracy server unavailable and the on-device model isn't loaded"
        )
      )
    }
    if (failure) {
      console.warn(
        "[useLipReader] accuracy mode failed, falling back to speed:",
        failure
      )
    }
    const result = await speed.recognize(crops, signal)
    return { ...result, fellBack: true }
  }
  const { speed } = await load.done
  signal?.throwIfAborted()
  if (!speed.available) {
    throw new NoEngineError(
      `${UNAVAILABLE_MESSAGE.speed}${load.engines.accuracy.available ? " — try Accuracy" : ""}`
    )
  }
  return speed.recognize(crops, signal)
}

/** Engine status from the live recognizers; keeps `prev` when nothing changed. */
function refreshEngines(
  prev: Record<RecognitionMode, EngineStatus>,
  recognizers: Recognizers,
  settled: EngineFlags
): Record<RecognitionMode, EngineStatus> {
  const status = (m: RecognitionMode, r: Recognizer): EngineStatus => ({
    name: r.name,
    available: r.available,
    isReal: r.isReal,
    loading: !settled[m],
  })
  const next = {
    speed: status("speed", recognizers.speed),
    accuracy: status("accuracy", recognizers.accuracy),
  }
  const same = (a: EngineStatus, b: EngineStatus) =>
    a.name === b.name &&
    a.available === b.available &&
    a.isReal === b.isReal &&
    a.loading === b.loading
  return same(prev.speed, next.speed) && same(prev.accuracy, next.accuracy)
    ? prev
    : next
}

/**
 * One recognizer pair per page. StrictMode (dev) mounts → unmounts → remounts effects; deferring
 * dispose lets the remount reuse the pair instead of loading the large local model twice.
 */
let sharedRecognizers: {
  load: RecognizersLoading
  users: number
  disposeTimer: number | undefined
} | null = null

function acquireRecognizers(): RecognizersLoading {
  if (!sharedRecognizers) {
    sharedRecognizers = {
      load: startRecognizers(ACTIVE_SPEC),
      users: 0,
      disposeTimer: undefined,
    }
  }
  const entry = sharedRecognizers
  window.clearTimeout(entry.disposeTimer)
  entry.disposeTimer = undefined
  entry.users += 1
  return entry.load
}

function releaseRecognizers(): void {
  const entry = sharedRecognizers
  if (!entry) return
  entry.users -= 1
  if (entry.users > 0) return
  entry.disposeTimer = window.setTimeout(() => {
    if (sharedRecognizers !== entry || entry.users > 0) return
    sharedRecognizers = null
    void entry.load.done.then((recognizers) => {
      recognizers.speed.dispose()
      recognizers.accuracy.dispose()
    })
  }, 1000)
}

function readStoredMode(): RecognitionMode {
  try {
    const stored = window.localStorage.getItem(MODE_STORAGE_KEY)
    if (stored === "speed" || stored === "accuracy") return stored
  } catch {
    // Storage blocked: fall through to the default.
  }
  return "speed"
}

function isNoFaceError(err: unknown): boolean {
  return (
    err instanceof NoFaceError ||
    (err instanceof Error && err.name === "NoFaceError")
  )
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT"
}

function stopTracks(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop())
}

function cameraErrorMessage(err: unknown): string {
  const name = err instanceof Error ? err.name : ""
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission denied"
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No camera found"
  }
  if (name === "NotReadableError") return "Camera is busy in another app"
  return "Camera unavailable"
}
