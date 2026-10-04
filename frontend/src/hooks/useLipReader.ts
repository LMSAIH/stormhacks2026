import { useCallback, useEffect, useRef, useState } from "react"

import { cropUtterance, rgbaToGray } from "@/lib/lipreading/crop"
import {
  startRecognizers,
  type RecognizersLoading,
} from "@/lib/lipreading/createRecognizers"
import { BlazeFaceDetector } from "@/lib/lipreading/faceDetector"
import { LipLandmarker } from "@/lib/lipreading/faceLandmarker"
import {
  CAMERA_CONSTRAINTS,
  FACE_ISSUE_TEXT,
  faceIssues,
  meanBrightness,
} from "@/lib/lipreading/faceQuality"
import { FaceTracker, type TrackedFace } from "@/lib/lipreading/faceTracker"
import { toSentenceCase } from "@/lib/lipreading/format"
import { LIP_TRACKING_SPEC } from "@/lib/lipreading/lipTrackingTypes"
import {
  LIP_MODES,
  LONG_PAUSE_MS,
  readStoredMode,
  SHORT_PAUSE_MS,
  storeMode,
  type LipMode,
} from "@/lib/lipreading/modes"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"
import { drawFaceOverlay, type NormalizedPoint } from "@/lib/lipreading/overlay"
import {
  NoFaceError,
  type CapturedFrame,
  type CropResult,
  type Keypoints,
  type RecognitionResult,
} from "@/lib/lipreading/types"
import { rankChoices } from "@/lib/phrases/snap"
import { createPhraseStore, type PhraseStore } from "@/lib/phrases/store"

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
  /** Up to 3 readings to pick from (beam alternatives + saved phrases), best first; [0] = `text`. */
  choices?: readonly string[]
  /** The mode that read it; `fellBack` = Quality's server failed and this device read it. */
  mode?: LipMode
  fellBack?: boolean
}

/** The sentence still being spoken: quick drafts of its pieces, replaced when it locks. */
export interface LipDraft {
  id: string
  text: string
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
 * Fallback only (the background worker failed to start): the main-thread lip tracking runs on
 * every 3rd frame while capturing so the capture rate the 25 fps resample relies on holds up.
 */
const LIP_TRACKING_STRIDE = 3

// --- Visual voice-activity detection (VAD) ---------------------------------
// We measure how much the lips *deform* between tracked frames (motion with the overall mouth
// translation removed, normalised by mouth width — so head movement doesn't register, only the
// lips changing shape does). Sustained deformation = speaking; quiet spells end pieces/sentences
// (SHORT_PAUSE_MS / LONG_PAUSE_MS in modes.ts).
/** EMA smoothing factor for the raw per-frame activity. */
const ACTIVITY_EMA = 0.6
/** Smoothed activity above this starts an utterance. */
const ACTIVITY_START = 0.045
/** Lower bar to *stay* speaking (hysteresis), so brief pauses mid-word don't cut it. */
const ACTIVITY_KEEP = 0.025
/** Include a little before detected speech so the first phoneme isn't clipped. */
const LEAD_MS = 250
/** While nobody speaks, keep this much buffered (≥ LEAD_MS) instead of reading silence. */
const IDLE_KEEP_MS = 1000
/** How often the face-quality hint is re-evaluated. */
const HINT_INTERVAL_MS = 500

/**
 * Camera + continuous lip reading with visual VAD.
 *
 * - Capture loop (every camera frame): gray frame → rolling buffer; both face trackers run in a
 *   background worker (`FaceTracker`), so this loop never waits on them (main-thread fallback if
 *   the worker can't start). Tracker results patch keypoints into the buffered frames and feed VAD.
 * - Sentences: speech starts a sentence; a short pause (SHORT_PAUSE_MS) reads the new piece as a
 *   draft (`draft`); a long pause (LONG_PAUSE_MS) or the mode's length cap locks it and runs the
 *   final read for the mode (`LIP_MODES`): instant = no drafts, each burst is final; normal = the
 *   whole sentence reread on this device; quality = reread on the GPU server (beam + LM), falling
 *   back to this device. Only locked sentences reach `transcript` (the page speaks those).
 * - Final lines carry up to 3 `choices` (beam alternatives + saved phrases); `pickChoice` swaps
 *   one in and remembers it.
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
  const [draft, setDraft] = useState<LipDraft | null>(null)
  const [mode, setModeState] = useState<LipMode>(readStoredMode)
  const [cloudAvailable, setCloudAvailable] = useState(false)
  /** Why the current view will read badly (too far / dark / turned), or null. */
  const [faceHint, setFaceHint] = useState<string | null>(null)

  // Pipeline state the loops read without re-rendering.
  const loadRef = useRef<RecognizersLoading | null>(null)
  /** Rolling capture buffer; pieces and sentences are sliced out of it by timestamp. */
  const bufferRef = useRef<CapturedFrame[]>([])
  const activeRef = useRef(active)
  const modeRef = useRef(mode)
  const engineReadyRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  /**
   * Two lanes: on-device reads run one at a time (one ONNX session); server reads have their own
   * lane so a slow server final never holds up the drafts. Drafts of locked sentences are skipped.
   */
  const queuesRef = useRef<Record<"local" | "cloud", Promise<void>>>({
    local: Promise.resolve(),
    cloud: Promise.resolve(),
  })
  const pendingRef = useRef(0)
  const lockedRef = useRef<Set<string>>(new Set())
  const draftPiecesRef = useRef<{ id: string; pieces: string[] } | null>(null)

  const clearTranscript = useCallback(() => setTranscript([]), [])

  const setMode = useCallback((next: LipMode) => {
    modeRef.current = next
    setModeState(next)
    storeMode(next)
  }, [])


  /** Run recognition work in order within a lane; `busy` while anything is queued or running. */
  const enqueue = useCallback((lane: "local" | "cloud", job: () => Promise<void>) => {
    pendingRef.current += 1
    setBusy(true)
    const queues = queuesRef.current
    queues[lane] = queues[lane].then(job).finally(() => {
      pendingRef.current -= 1
      if (pendingRef.current === 0) setBusy(false)
    })
  }, [])

  /** Crop + read frames with one engine; null when there's no face or no usable text. */
  const read = useCallback(
    async (
      frames: CapturedFrame[],
      engine: "speed" | "accuracy"
    ): Promise<{ result: RecognitionResult; crops: CropResult } | null> => {
      const signal = abortRef.current?.signal
      const startedAt = frames[0].tMs
      const endedAt = frames[frames.length - 1].tMs
      let crops: CropResult
      try {
        crops = cropUtterance({ frames, startedAt, endedAt }, ACTIVE_SPEC)
      } catch (err) {
        if (isNoFaceError(err)) {
          setLastError(NO_FACE_MESSAGE)
          return null
        }
        throw err
      }
      const load = loadRef.current
      if (!load) return null
      const recognizers = await load.done
      if (signal?.aborted || !activeRef.current) return null
      const result = await recognizers[engine].recognize(crops, signal)
      if (signal?.aborted || !activeRef.current) return null
      return result.text.trim() ? { result, crops } : null
    },
    []
  )

  // --- A short pause: read the new piece and add it to the live draft ---
  const readDraft = useCallback(
    (sentenceId: string, frames: CapturedFrame[]) => {
      enqueue("local", async () => {
        if (lockedRef.current.has(sentenceId)) return // the sentence already locked: final covers it
        try {
          const out = await read(frames, "speed")
          if (!out || lockedRef.current.has(sentenceId)) return
          const prev = draftPiecesRef.current
          const pieces = prev?.id === sentenceId ? [...prev.pieces, out.result.text.trim()] : [out.result.text.trim()]
          draftPiecesRef.current = { id: sentenceId, pieces }
          setDraft({ id: sentenceId, text: toSentenceCase(pieces.join(" ")) })
          setLastError(null)
        } catch (err) {
          console.warn("[useLipReader] draft failed:", err)
        }
      })
    },
    [enqueue, read]
  )

  // --- A long pause / length cap: lock the sentence and run the mode's final read ---
  const lockSentence = useCallback(
    (sentenceId: string, frames: CapturedFrame[], lockedMode: LipMode) => {
      lockedRef.current.add(sentenceId)
      const startedAt = frames[0].tMs

      const clearDraft = () => {
        if (draftPiecesRef.current?.id === sentenceId) draftPiecesRef.current = null
        setDraft((d) => (d?.id === sentenceId ? null : d))
      }
      const fail = (err: unknown) => {
        if (abortRef.current?.signal.aborted) return
        console.error("[useLipReader] recognition failed:", err)
        setLastError("Recognition failed")
      }

      /** Rank against saved phrases (plan D53/D62) and add the line, in time order. */
      const finish = async (result: RecognitionResult, fellBack: boolean) => {
        const raw = result.text.trim()
        const readings = [raw, ...(result.alternatives ?? []).slice(1).map((a) => a.text)]
        const hits = await phrases().search(raw).catch(() => [])
        const ranked = rankChoices(readings, hits)
        const best = ranked.snap?.text ?? raw
        setEngineName(result.engine)
        setTranscript((prev) =>
          [
            ...prev,
            {
              id: `lip-${Math.round(startedAt)}`,
              text: toSentenceCase(best),
              raw,
              at: startedAt,
              latencyMs: result.latencyMs,
              engine: result.engine,
              confidence: result.confidence,
              choices: ranked.choices.map((c) => toSentenceCase(c.text)),
              mode: lockedMode,
              fellBack,
            },
          ].sort((a, b) => a.at - b.at)
        )
        void phrases().add(best, "accepted").catch(() => undefined)
        setLastError(null)
      }

      /** Read on this device (normal/instant, or Quality's fallback). */
      const localFinal = (fellBack: boolean) =>
        enqueue("local", async () => {
          try {
            const out = await read(frames, "speed")
            if (out) await finish(out.result, fellBack)
          } catch (err) {
            fail(err)
          } finally {
            clearDraft()
          }
        })

      if (LIP_MODES[lockedMode].final !== "cloud") return localFinal(false)

      // Quality: the server reads it in its own lane; drafts keep going meanwhile.
      enqueue("cloud", async () => {
        let done = false
        try {
          const load = loadRef.current
          if (load && (await load.done).accuracy.available) {
            const out = await read(frames, "accuracy")
            if (out) await finish(out.result, false)
            done = true // read, or no face/text: either way don't reread locally
            clearDraft()
          }
        } catch (err) {
          console.warn("[useLipReader] cloud read failed, reading on this device:", err)
        }
        if (!done) localFinal(true)
      })
    },
    [enqueue, read]
  )

  /** Swap a line for one of its choices (a new id, so the page speaks the fixed sentence). */
  const pickChoice = useCallback((itemId: string, text: string) => {
    setTranscript((prev) =>
      prev.map((item) =>
        item.id === itemId && item.text !== text
          ? { ...item, id: `${item.id}~${text.length}`, text }
          : item
      )
    )
    void phrases().add(text, "picked").catch(() => undefined)
  }, [])

  // The capture loop reaches the latest callbacks through refs so its effect stays mount-only.
  const draftRef = useRef(readDraft)
  const lockRef = useRef(lockSentence)
  useEffect(() => {
    draftRef.current = readDraft
    lockRef.current = lockSentence
  }, [readDraft, lockSentence])

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

  // --- Recognizers (on-device ONNX + the GPU server; mock only if neither loads) ---
  useEffect(() => {
    let cancelled = false
    const load = acquireRecognizers()
    loadRef.current = load
    void load.ready.accuracy.then(() => {
      if (!cancelled) setCloudAvailable(load.engines.accuracy.available)
    })
    load.done.then(
      (recognizers) => {
        if (cancelled) return
        setEngineName(recognizers.speed.name)
        setCloudAvailable(recognizers.accuracy.available)
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

  // --- Camera + face trackers + per-frame capture + VAD ---
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
    let lastHintUpdate = 0
    let lastHint: string | null = null
    const frameTimes: number[] = []

    // Trackers: the background worker, or the main-thread pair if it can't start.
    const tracker = new FaceTracker()
    let trackerReady = false
    const detector = new BlazeFaceDetector()
    let detectorReady = false
    const lipTracker = new LipLandmarker(LIP_TRACKING_SPEC)
    let lipTrackerReady = false
    let lastLipTs = Number.NEGATIVE_INFINITY
    let framesSinceLipTracking = 0
    /** Latest tracking, drawn every frame and used for the face hint. */
    let lipPoints = NO_LIP_POINTS
    let latestKeypoints: Keypoints | null = null

    const scratch = document.createElement("canvas")
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true })
    const minSpanMs = ACTIVE_SPEC.minSeconds * 1000

    // VAD + sentence state.
    let prevLip: readonly NormalizedPoint[] | null = null
    let activityEma = 0
    let isSpeaking = false
    let lastActiveTms = 0
    let sentence: { id: string; startTms: number; pieceStartTms: number; paused: boolean } | null =
      null

    const framesBetween = (startTms: number, endTms: number) =>
      bufferRef.current.filter((f) => f.tMs >= startTms && f.tMs <= endTms)
    const spanOk = (frames: CapturedFrame[]) =>
      frames.length >= 2 && frames[frames.length - 1].tMs - frames[0].tMs >= minSpanMs

    /** Fresh lip landmarks (worker or fallback) → speech activity. */
    const onLipPoints = (points: readonly NormalizedPoint[], tMs: number) => {
      if (!(activeRef.current && engineReadyRef.current)) return
      if (prevLip && points.length >= 11) {
        const activity = mouthActivity(prevLip, points)
        activityEma += ACTIVITY_EMA * (activity - activityEma)
        const bar = isSpeaking ? ACTIVITY_KEEP : ACTIVITY_START
        if (activityEma > bar) {
          if (!isSpeaking) {
            isSpeaking = true
            const start = tMs - LEAD_MS
            sentence = { id: `s-${Math.round(start)}`, startTms: start, pieceStartTms: start, paused: false }
          }
          lastActiveTms = tMs
          if (sentence) sentence.paused = false
        }
      }
      if (points.length) prevLip = points
    }

    /** Short/long pause and length-cap checks, every captured frame. */
    const checkPauses = (tMs: number) => {
      const s = sentence
      if (!isSpeaking || !s) {
        // Idle with lip tracking up: drop old frames instead of reading silence. Without lip
        // tracking there's no VAD, so fall back to reading fixed windows like before.
        const buf = bufferRef.current
        if (buf.length < 2) return
        const haveLips = trackerReady || lipTrackerReady
        if (haveLips) {
          if (tMs - buf[0].tMs > IDLE_KEEP_MS)
            bufferRef.current = buf.filter((f) => f.tMs >= tMs - IDLE_KEEP_MS)
        } else if (buf[buf.length - 1].tMs - buf[0].tMs > LIP_MODES[modeRef.current].maxSeconds * 1000) {
          const frames = buf.slice()
          bufferRef.current = []
          if (spanOk(frames)) lockRef.current(`w-${Math.round(frames[0].tMs)}`, frames, modeRef.current)
        }
        return
      }
      const info = LIP_MODES[modeRef.current]
      const quiet = tMs - lastActiveTms
      const capMs = info.maxSeconds * 1000 - 1000 / ACTIVE_SPEC.fps
      const endOfSentence =
        quiet > LONG_PAUSE_MS || tMs - s.startTms > capMs || (!info.drafts && quiet > SHORT_PAUSE_MS)
      if (endOfSentence) {
        isSpeaking = false
        sentence = null
        const frames = framesBetween(s.startTms, tMs)
        bufferRef.current = bufferRef.current.filter((f) => f.tMs > tMs)
        if (spanOk(frames)) lockRef.current(s.id, frames, modeRef.current)
      } else if (info.drafts && quiet > SHORT_PAUSE_MS && !s.paused) {
        s.paused = true // one draft per pause
        const frames = framesBetween(s.pieceStartTms, tMs)
        if (spanOk(frames)) {
          s.pieceStartTms = tMs
          draftRef.current(s.id, frames)
        }
      }
    }

    /** Worker result: patch the buffered frame's keypoints, update the overlay state, run VAD. */
    tracker.onResult = (face: TrackedFace) => {
      latestKeypoints = face.keypoints
      lipPoints = face.lipPoints
      const buf = bufferRef.current
      for (let i = buf.length - 1; i >= 0; i--) {
        if (buf[i].tMs === face.tMs) {
          buf[i] = { ...buf[i], keypoints: face.keypoints, tracked: true }
          break
        }
        if (buf[i].tMs < face.tMs) break
      }
      onLipPoints(face.lipPoints, face.tMs)
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
      let gray: Uint8Array | null = null
      if (capturing && scratchCtx) {
        if (scratch.width !== width) scratch.width = width
        if (scratch.height !== height) scratch.height = height
        scratchCtx.drawImage(video, 0, 0, width, height)
        let keypoints: Keypoints | null = null
        let tracked = true
        if (trackerReady) {
          // Track the exact pixels we store; skipped frames get interpolated keypoints later.
          tracked = false
          tracker.submit(scratch, tMs)
        } else if (detectorReady) {
          keypoints = detector.detect(scratch, tMs)
          latestKeypoints = keypoints
        }
        const buffer = bufferRef.current
        const last = buffer[buffer.length - 1]
        if (!last || tMs > last.tMs) {
          const { data } = scratchCtx.getImageData(0, 0, width, height)
          gray = rgbaToGray(data, width, height)
          buffer.push({ tMs, width, height, gray, keypoints, tracked })
        }
      } else if (trackerReady) {
        tracker.submit(video, tMs)
      } else if (detectorReady) {
        latestKeypoints = detector.detect(video, tMs)
      }

      // Fallback lip tracking on the main thread (overlay + VAD) every LIP_TRACKING_STRIDE-th frame.
      if (!trackerReady && lipTrackerReady && ++framesSinceLipTracking >= LIP_TRACKING_STRIDE) {
        framesSinceLipTracking = 0
        lastLipTs = tMs > lastLipTs ? tMs : lastLipTs + 1
        lipPoints = lipTracker.detect(video, lastLipTs).lipPoints
        onLipPoints(lipPoints, tMs)
      }
      drawFaceOverlay(overlayRef.current, width, height, lipPoints, latestKeypoints)

      if (capturing) checkPauses(tMs)
      if (isSpeaking !== lastSpeaking) {
        lastSpeaking = isSpeaking
        setSpeaking(isSpeaking)
      }

      const haveLips = trackerReady || lipTrackerReady
      const mouth = haveLips ? lipPoints.length > 0 : latestKeypoints !== null
      if (mouth !== lastMouth) {
        lastMouth = mouth
        setMouthDetected(mouth)
      }
      const now = performance.now()
      if ((trackerReady || detectorReady) && now - lastHintUpdate >= HINT_INTERVAL_MS) {
        lastHintUpdate = now
        const issue = faceIssues(latestKeypoints, width, gray ? meanBrightness(gray) : undefined)[0]
        const hint = issue ? FACE_ISSUE_TEXT[issue] : null
        if (hint !== lastHint) {
          lastHint = hint
          setFaceHint(hint)
        }
      }
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

    /** Main-thread trackers, used only when the worker can't start. */
    const startFallbackTrackers = async () => {
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
      detectorReady = true
      setReady(true)
    }

    const start = async () => {
      setCameraStatus("starting")
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: CAMERA_CONSTRAINTS,
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

      try {
        await tracker.init()
        if (cancelled) return
        trackerReady = true
        setReady(true)
      } catch (err) {
        if (cancelled) return
        console.warn("[useLipReader] background face tracking unavailable, using the main thread:", err)
        tracker.dispose()
        await startFallbackTrackers()
      }
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
      trackerReady = false
      tracker.onResult = null
      tracker.dispose()
      detectorReady = false
      detector.dispose()
      lipTrackerReady = false
      lipTracker.dispose()
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
    /** The sentence still being spoken (drafts), or null. */
    draft,
    mode,
    setMode,
    /** The GPU server answers (Quality reads there; otherwise it falls back to this device). */
    cloudAvailable,
    pickChoice,
    faceHint,
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

/** The user's phrase bank (browser stand-in, or the TiDB service with VITE_PHRASES_URL). */
let phraseStore: PhraseStore | null = null
function phrases(): PhraseStore {
  return (phraseStore ??= createPhraseStore())
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
