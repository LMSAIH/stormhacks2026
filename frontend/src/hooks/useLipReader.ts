import { useCallback, useEffect, useRef, useState } from "react"

import { cropUtterance, rgbaToGray } from "@/lib/lipreading/crop"
import {
  startRecognizers,
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
} from "@/lib/lipreading/types"

export type CameraStatus = "idle" | "starting" | "on" | "error"

export interface LipTranscriptItem {
  id: string
  /** What to show / speak: the recognizer output read as a sentence ("I THINK" → "I think"). */
  text: string
  /** Recognizer output as returned (uppercase SentencePiece text). */
  raw: string
  /** performance.now() at the start of the utterance — orders it alongside listening utterances. */
  at: number
  latencyMs: number
  engine: string
  confidence?: number
}

export interface UseLipReaderOptions {
  /**
   * Whether streaming recognition is on (default true). While false the camera, face detector and
   * overlay keep running (so switching back on is instant) but nothing is buffered or recognized.
   */
  active?: boolean
}

const NO_FACE_MESSAGE = "No face — keep your face in frame"
const NO_LIP_POINTS: readonly NormalizedPoint[] = []

/**
 * While capturing, the lip tracking (the costly FaceLandmarker graph, ~25 ms/frame on a laptop GPU)
 * runs on every 3rd frame only so the capture rate the 25 fps resample relies on holds up; its dots
 * reuse the last points in between.
 */
const LIP_TRACKING_STRIDE = 3

// --- Visual voice-activity detection (VAD) ---------------------------------
// We measure how much the lips *deform* between tracked frames (motion with the overall mouth
// translation removed, normalised by mouth width — so head movement doesn't register, only the
// lips changing shape does). Sustained deformation = speaking; a quiet spell ends the utterance.
/** EMA smoothing factor for the raw per-frame activity. */
const ACTIVITY_EMA = 0.6
/** Smoothed activity above this starts an utterance. */
const ACTIVITY_START = 0.045
/** Lower bar to *stay* speaking (hysteresis), so brief pauses mid-word don't cut it. */
const ACTIVITY_KEEP = 0.025
/** Lips quiet for this long ends the utterance and sends it. */
const SILENCE_MS = 600
/** Include a little before detected speech so the first phoneme isn't clipped. */
const LEAD_MS = 250

/**
 * Camera + continuous lip reading with visual VAD.
 *
 * Two decoupled loops:
 *  - Capture loop (every camera frame): BlazeFace keypoints + gray frame → rolling buffer, plus the
 *    FaceLandmarker lip dots on the overlay. It also runs visual VAD on the lip landmarks.
 *  - Recognition: triggered when the lips go still after speaking (not on a fixed timer) — the
 *    buffered utterance is cropped once (the exact Python crop pipeline) and run through the
 *    on-device ONNX recognizer, appending one transcript item.
 *
 * `speaking` is the live VAD state; `inferring` is true while a recognition is in flight.
 */
export function useLipReader({ active = true }: UseLipReaderOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)

  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle")
  const [ready, setReady] = useState(false)
  /** Lips found by the lip tracking (BlazeFace's face while that is unavailable). */
  const [mouthDetected, setMouthDetected] = useState(false)
  /** Live VAD state: the user is currently speaking. */
  const [speaking, setSpeaking] = useState(false)
  const [fps, setFps] = useState(0)
  const [engineName, setEngineName] = useState("loading…")
  const [engineReady, setEngineReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<LipTranscriptItem[]>([])

  // Pipeline state the loops read without re-rendering.
  const detectorRef = useRef<BlazeFaceDetector | null>(null)
  const loadRef = useRef<RecognizersLoading | null>(null)
  /** Rolling capture buffer; a flushed utterance is sliced out of it by timestamp. */
  const bufferRef = useRef<CapturedFrame[]>([])
  const busyRef = useRef(false)
  const activeRef = useRef(active)
  const engineReadyRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)

  const clearTranscript = useCallback(() => setTranscript([]), [])

  // --- Recognize one utterance (the frames between speech onset and the silence that ended it) ---
  const recognizeUtterance = useCallback(async (frames: CapturedFrame[]) => {
    const signal = abortRef.current?.signal
    busyRef.current = true
    setBusy(true)
    try {
      // Let the busy state paint before the synchronous crop work.
      await new Promise((resolve) => window.setTimeout(resolve, 0))
      const startedAt = frames[0].tMs
      const endedAt = frames[frames.length - 1].tMs

      let crops: CropResult
      try {
        crops = cropUtterance({ frames, startedAt, endedAt }, ACTIVE_SPEC)
      } catch (err) {
        if (isNoFaceError(err)) {
          setLastError(NO_FACE_MESSAGE)
          return
        }
        throw err
      }

      const load = loadRef.current
      if (!load) return
      const { speed } = await load.done
      if (signal?.aborted || !activeRef.current) return

      const result = await speed.recognize(crops, signal)
      if (signal?.aborted || !activeRef.current) return

      setEngineName(speed.name)
      const raw = result.text.trim()
      if (!raw) return // unreadable utterance — skip rather than nag

      setTranscript((prev) => [
        ...prev,
        {
          id: `lip-${Math.round(startedAt)}-${prev.length}`,
          text: toSentenceCase(raw),
          raw,
          at: startedAt,
          latencyMs: result.latencyMs,
          engine: result.engine,
          confidence: result.confidence,
        },
      ])
      setLastError(null)
    } catch (err) {
      if (signal?.aborted) return
      console.error("[useLipReader] recognition failed:", err)
      setLastError("Recognition failed")
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }, [])

  // The capture loop calls recognition through a ref so its effect can stay mount-only.
  const recognizeRef = useRef(recognizeUtterance)
  useEffect(() => {
    recognizeRef.current = recognizeUtterance
  }, [recognizeUtterance])

  // Handlers read `active` through the ref; leaving the active state drops the buffer.
  useEffect(() => {
    activeRef.current = active
    if (!active) bufferRef.current = []
  }, [active])

  // Lets in-flight recognition notice unmount (and cancels HTTP requests, if any).
  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller
    return () => {
      controller.abort()
      abortRef.current = null
    }
  }, [])

  // --- Recognizer (on-device ONNX; mock only if the model can't load) ---
  useEffect(() => {
    let cancelled = false
    const load = acquireRecognizers()
    loadRef.current = load
    load.done.then(
      (recognizers) => {
        if (cancelled) return
        setEngineName(recognizers.speed.name)
        engineReadyRef.current = true
        setEngineReady(true)
      },
      (err: unknown) => {
        if (cancelled) return
        console.error("[useLipReader] recognizer failed to load:", err)
        setEngineName("failed to load")
        engineReadyRef.current = true
        setEngineReady(true)
        setLastError("Recognition engine failed to load")
      }
    )
    return () => {
      cancelled = true
      loadRef.current = null
      releaseRecognizers()
    }
  }, [])

  // --- Camera + face detector + lip tracking + per-frame capture + VAD ---
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let cancelled = false
    let stream: MediaStream | null = null
    let videoFrameHandle: number | null = null
    let animationFrameHandle: number | null = null
    let lastVideoTime = -1
    let lastMouth = false
    let lastSpeaking = false
    let lastFpsUpdate = 0
    const frameTimes: number[] = []

    const detector = new BlazeFaceDetector()
    const lipTracker = new LipLandmarker(LIP_TRACKING_SPEC)
    let lipTrackerReady = false
    let lastLipTs = Number.NEGATIVE_INFINITY
    let lipPoints = NO_LIP_POINTS
    let framesSinceLipTracking = 0
    const scratch = document.createElement("canvas")
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true })
    const maxSpanMs = ACTIVE_SPEC.maxSeconds * 1000
    const minSpanMs = ACTIVE_SPEC.minSeconds * 1000

    // VAD state.
    let prevLip: readonly NormalizedPoint[] | null = null
    let activityEma = 0
    let isSpeaking = false
    let speechStartTms = 0
    let lastActiveTms = 0

    /** Slice an utterance out of the buffer by timestamp and send it to recognition. */
    const flush = (startTms: number, endTms: number) => {
      const all = bufferRef.current
      const seg = all.filter((f) => f.tMs >= startTms && f.tMs <= endTms)
      bufferRef.current = all.filter((f) => f.tMs > endTms)
      if (seg.length < 2) return
      if (seg[seg.length - 1].tMs - seg[0].tMs < minSpanMs) return
      recognizeRef.current(seg)
    }

    const processFrame = (tMs: number) => {
      const width = video.videoWidth
      const height = video.videoHeight
      if (
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        width === 0
      ) {
        return
      }

      const capturing = activeRef.current && engineReadyRef.current
      let keypoints: Keypoints | null
      if (capturing && scratchCtx) {
        if (scratch.width !== width) scratch.width = width
        if (scratch.height !== height) scratch.height = height
        scratchCtx.drawImage(video, 0, 0, width, height)
        // Detect on the exact pixels we store, so keypoints and gray frame always match.
        keypoints = detector.detect(scratch, tMs)
        const buffer = bufferRef.current
        const last = buffer[buffer.length - 1]
        if (!last || tMs > last.tMs) {
          const { data } = scratchCtx.getImageData(0, 0, width, height)
          buffer.push({
            tMs,
            width,
            height,
            gray: rgbaToGray(data, width, height),
            keypoints,
          })
        }
      } else {
        keypoints = detector.detect(video, tMs)
      }

      // Lip tracking (overlay + VAD source) every LIP_TRACKING_STRIDE-th frame.
      let lipFresh = false
      if (lipTrackerReady && ++framesSinceLipTracking >= LIP_TRACKING_STRIDE) {
        framesSinceLipTracking = 0
        lastLipTs = tMs > lastLipTs ? tMs : lastLipTs + 1
        lipPoints = lipTracker.detect(video, lastLipTs).lipPoints
        lipFresh = true
      }
      drawFaceOverlay(overlayRef.current, width, height, lipPoints, keypoints)

      // --- Visual VAD on fresh lip landmarks ---
      if (capturing && lipFresh) {
        if (prevLip && lipPoints.length >= 11) {
          const activity = mouthActivity(prevLip, lipPoints)
          activityEma += ACTIVITY_EMA * (activity - activityEma)
          const bar = isSpeaking ? ACTIVITY_KEEP : ACTIVITY_START
          if (activityEma > bar) {
            if (!isSpeaking) {
              isSpeaking = true
              speechStartTms = tMs - LEAD_MS
            }
            lastActiveTms = tMs
          }
        }
        if (lipPoints.length) prevLip = lipPoints
      }

      // End-of-utterance / safety checks (every frame while capturing).
      if (capturing) {
        if (isSpeaking) {
          if (tMs - lastActiveTms > SILENCE_MS || tMs - speechStartTms > maxSpanMs) {
            isSpeaking = false
            flush(speechStartTms, tMs)
          }
        } else {
          // Not speaking (or lip tracking unavailable): don't let the buffer grow without bound.
          const buf = bufferRef.current
          if (buf.length >= 2 && buf[buf.length - 1].tMs - buf[0].tMs > maxSpanMs) {
            flush(buf[0].tMs, buf[buf.length - 1].tMs)
          }
        }
      }
      if (isSpeaking !== lastSpeaking) {
        lastSpeaking = isSpeaking
        setSpeaking(isSpeaking)
      }

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

      void lipTracker.init().then(
        () => {
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
      bufferRef.current = []
      setReady(false)
      setSpeaking(false)
      setCameraStatus("idle")
    }
  }, [])

  return {
    videoRef,
    overlayRef,
    cameraStatus,
    ready,
    mouthDetected,
    speaking,
    fps,
    engineName,
    engineReady,
    busy,
    /** Alias of `busy`: a recognition is in flight. */
    inferring: busy,
    lastError,
    transcript,
    clearTranscript,
  }
}

/**
 * How much the lips changed shape between two tracked frames, with the overall mouth translation
 * removed (so head movement doesn't register) and normalised by mouth width (scale-invariant).
 * `points` are the FaceLandmarker lip contour; index 0/10 are the mouth corners.
 */
function mouthActivity(
  prev: readonly NormalizedPoint[],
  cur: readonly NormalizedPoint[]
): number {
  const n = Math.min(prev.length, cur.length)
  if (n === 0) return 0
  let mdx = 0
  let mdy = 0
  for (let i = 0; i < n; i++) {
    mdx += cur[i].x - prev[i].x
    mdy += cur[i].y - prev[i].y
  }
  mdx /= n
  mdy /= n
  let sum = 0
  for (let i = 0; i < n; i++) {
    const ex = cur[i].x - prev[i].x - mdx
    const ey = cur[i].y - prev[i].y - mdy
    sum += Math.hypot(ex, ey)
  }
  const deform = sum / n
  const width = Math.hypot(cur[0].x - cur[10].x, cur[0].y - cur[10].y) || 1
  return deform / width
}

function isNoFaceError(err: unknown): boolean {
  return (
    err instanceof NoFaceError ||
    (err instanceof Error && err.name === "NoFaceError")
  )
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

/**
 * One recognizer load per page. StrictMode (dev) mounts → unmounts → remounts effects; deferring
 * dispose lets the remount reuse the warmed model instead of loading the large model twice.
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
