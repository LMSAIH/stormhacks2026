import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { useSearchParams } from "react-router-dom"
import { cn } from "cn"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cropUtterance, rgbaToGray } from "@/lib/lipreading/crop"
import { createRecognizers } from "@/lib/lipreading/createRecognizers"
import { BlazeFaceDetector } from "@/lib/lipreading/faceDetector"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"
import {
  NoFaceError,
  type CapturedFrame,
  type CropResult,
  type Keypoints,
  type RecognitionMode,
  type Recognizer,
} from "@/lib/lipreading/types"

/**
 * /lab — deterministic end-to-end harness for the lip-reading pipeline (driven by Playwright).
 *
 * No camera and no wall clock: a clip is stepped frame by frame at its native rate through the
 * same detector → crop → recognizers path the app uses, and every intermediate result is
 * published on `window.__lipLab`. Query params: `auto=1` runs on load, `url=` picks the clip,
 * `fps=` overrides the measured frame rate.
 */

const DEFAULT_CLIP_URL = "/test/clip.mp4"
/** Used when requestVideoFrameCallback can't measure the clip's rate. */
const DEFAULT_FPS = 25
const FILMSTRIP_STRIDE = 4
const LOAD_TIMEOUT_MS = 30_000
const SEEK_TIMEOUT_MS = 10_000
/** Grace after "seeked" for the frame to be presented (none comes if the frame didn't change). */
const PRESENT_TIMEOUT_MS = 300
const FPS_PROBE_TIMEOUT_MS = 2_000
const FPS_PROBE_SAMPLES = 10
/** Container rates a measured fps snaps to (nearest within 0.05% — 29.97 and 30 are 0.1% apart). */
const STANDARD_FPS = [
  24000 / 1001,
  24,
  25,
  30000 / 1001,
  30,
  50,
  60000 / 1001,
  60,
]
const MODES: readonly RecognitionMode[] = ["speed", "accuracy"]

export type LabStatus = "idle" | "running" | "done" | "error"
/** Where `fps` came from: measured via requestVideoFrameCallback, the `fps=` param, or assumed. */
export type FpsSource = "rvfc" | "param" | "default"

interface RecognizerInfo {
  readonly name: string
  readonly available: boolean
  readonly isReal: boolean
}

export type LabRecognition =
  | (RecognizerInfo & {
      readonly text: string
      readonly engine: string
      readonly latencyMs: number
      readonly confidence?: number
      /** Mode that actually produced the text (differs from the slot when it fell back). */
      readonly mode: RecognitionMode
      readonly fellBack: boolean
      readonly serverLatencyMs?: Readonly<Record<string, number>>
    })
  | (RecognizerInfo & { readonly error: string })

export interface LipLabState {
  status: LabStatus
  error?: string
  /** Increments per run so automation can wait for a specific run to finish. */
  runId: number
  url: string
  /** Human-readable progress, e.g. "decode + detect 12/75". */
  stage: string
  width: number
  height: number
  durationS: number
  /** Source frames captured at the clip's native rate. */
  frames: number
  fps: number
  fpsSource: FpsSource
  /**
   * Media time (s) of the frame actually presented for each captured frame — should be i/fps.
   * null where the browser gave no presentation callback (frame not verified).
   */
  frameMediaTimes: readonly (number | null)[]
  faceCoverage: number
  /** Raw detector output per captured frame (null = no face). */
  keypoints: readonly (Keypoints | null)[]
  /** Smoothed keypoints the crop used, one per 25 fps output frame. */
  smoothedKeypoints: readonly Keypoints[]
  patchSize: number
  /** Number of 25 fps mouth crops. */
  T: number
  /** Base64 of the T crops concatenated (T * patchSize * patchSize uint8, row-major). */
  cropsB64: string
  speed: LabRecognition | null
  accuracy: LabRecognition | null
  /** Stage durations in ms (see TIMING_ROWS). */
  timings: Record<string, number>
}

declare global {
  interface Window {
    __lipLab?: LipLabState
  }
}

interface FilmstripItem {
  index: number
  patch: Uint8Array
}

function idleState(url: string, runId = 0): LipLabState {
  return {
    status: "idle",
    runId,
    url,
    stage: "idle",
    width: 0,
    height: 0,
    durationS: 0,
    frames: 0,
    fps: 0,
    fpsSource: "default",
    frameMediaTimes: [],
    faceCoverage: 0,
    keypoints: [],
    smoothedKeypoints: [],
    patchSize: ACTIVE_SPEC.patchSize,
    T: 0,
    cropsB64: "",
    speed: null,
    accuracy: null,
    timings: {},
  }
}

// --- Pipeline (plain async code; the component only wires it to state) ---

type Recognizers = Awaited<ReturnType<typeof createRecognizers>>

let recognizersPromise: Promise<Recognizers> | null = null

/** Created on first use and shared by every run — the int8 speed model is ~203 MB. */
function loadRecognizers(): Promise<Recognizers> {
  recognizersPromise ??= createRecognizers(ACTIVE_SPEC).catch(
    (err: unknown) => {
      recognizersPromise = null
      throw err
    }
  )
  return recognizersPromise
}

interface RunOptions {
  url: string
  runId: number
  fpsOverride: number | null
  /** Element the hidden <video> is mounted in while the run lasts. */
  host: HTMLElement
  signal: AbortSignal
  onState: (state: LipLabState) => void
  onFilmstrip: (items: FilmstripItem[]) => void
}

async function runLab(opts: RunOptions): Promise<void> {
  const { url, runId, fpsOverride, host, signal } = opts
  const timings: Record<string, number> = {}
  let state: LipLabState = {
    ...idleState(url, runId),
    status: "running",
    stage: "loading clip",
  }
  // A superseded (aborted) run never writes again, so it can't clobber a newer one.
  const publish = (patch: Partial<LipLabState>) => {
    if (signal.aborted) return
    state = { ...state, ...patch, timings: roundTimings(timings) }
    window.__lipLab = state
    opts.onState(state)
  }

  const startedAt = performance.now()
  const video = createHiddenVideo(host)
  let detector: BlazeFaceDetector | null = null
  publish({})

  try {
    let t0 = performance.now()
    await openClip(video, url, signal)
    timings.load = performance.now() - t0
    const { videoWidth: width, videoHeight: height } = video
    if (!width || !height) throw new Error("clip has no video track")
    publish({
      stage: "measuring frame rate",
      width,
      height,
      durationS: video.duration,
    })

    t0 = performance.now()
    const measured = fpsOverride === null ? await probeFps(video, signal) : null
    timings.fpsProbe = performance.now() - t0
    const fps = fpsOverride ?? measured ?? DEFAULT_FPS
    const fpsSource: FpsSource =
      fpsOverride !== null ? "param" : measured !== null ? "rvfc" : "default"
    publish({ stage: "loading face detector", fps, fpsSource })

    t0 = performance.now()
    detector = new BlazeFaceDetector()
    await detector.init()
    signal.throwIfAborted()
    timings.detectorInit = performance.now() - t0

    t0 = performance.now()
    const { frames, mediaTimes, detectMs } = await captureFrames(
      video,
      detector,
      fps,
      signal,
      (done, total) =>
        publish({ stage: `decode + detect ${done}/${total}`, frames: done })
    )
    timings.decodeDetect = performance.now() - t0
    timings.detect = detectMs
    const keypoints = frames.map((f) => f.keypoints)
    publish({
      stage: "cropping",
      frames: frames.length,
      frameMediaTimes: mediaTimes,
      keypoints,
      faceCoverage: keypoints.filter(Boolean).length / frames.length,
    })

    t0 = performance.now()
    const crops = cropUtterance(
      { frames, startedAt: 0, endedAt: (frames.length * 1000) / fps },
      ACTIVE_SPEC
    )
    timings.crop = performance.now() - t0
    if (!signal.aborted) {
      opts.onFilmstrip(
        crops.patches.flatMap((patch, index) =>
          index % FILMSTRIP_STRIDE === 0 ? [{ index, patch }] : []
        )
      )
    }
    publish({
      stage: "loading recognizers",
      T: crops.patches.length,
      faceCoverage: crops.faceCoverage,
      smoothedKeypoints: crops.keypoints,
      cropsB64: bytesToBase64(concatBytes(crops.patches)),
    })

    t0 = performance.now()
    let recognizers: Recognizers | null = null
    let loadError = ""
    try {
      recognizers = await loadRecognizers()
    } catch (err) {
      loadError = describeError(err)
    }
    signal.throwIfAborted()
    timings.recognizersInit = performance.now() - t0

    for (const mode of MODES) {
      publish({ stage: `recognizing (${mode})` })
      t0 = performance.now()
      const result: LabRecognition = recognizers
        ? await recognizeWith(recognizers[mode], crops, signal)
        : {
            name: "none",
            available: false,
            isReal: false,
            error: `createRecognizers failed: ${loadError}`,
          }
      signal.throwIfAborted()
      timings[`${mode}Recognize`] = performance.now() - t0
      publish(mode === "speed" ? { speed: result } : { accuracy: result })
    }

    timings.total = performance.now() - startedAt
    publish({ status: "done", stage: "done" })
  } catch (err) {
    if (signal.aborted) return
    timings.total = performance.now() - startedAt
    publish({
      status: "error",
      stage: "error",
      error: isNoFaceError(err)
        ? `no face: ${(err as Error).message}`
        : describeError(err),
    })
  } finally {
    try {
      detector?.dispose()
    } catch (err) {
      console.warn("[lab] detector dispose failed:", err)
    }
    disposeVideo(video)
  }
}

/**
 * Step the clip one source frame at a time: seek, draw, grayscale, detect.
 * Frame i is stamped i/fps — never wall-clock time — so reruns are identical.
 */
async function captureFrames(
  video: HTMLVideoElement,
  detector: BlazeFaceDetector,
  fps: number,
  signal: AbortSignal,
  onProgress: (done: number, total: number) => void
): Promise<{
  frames: CapturedFrame[]
  mediaTimes: (number | null)[]
  detectMs: number
}> {
  const { videoWidth: width, videoHeight: height } = video
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext("2d", { willReadFrequently: true })
  if (!ctx) throw new Error("2D canvas context unavailable")

  const maxFrames = Math.floor(ACTIVE_SPEC.maxSeconds * fps + 1e-6)
  // Lean towards one frame too many: a repeat of the last frame is trimmed below.
  const total = Number.isFinite(video.duration)
    ? Math.max(1, Math.min(maxFrames, Math.ceil(video.duration * fps - 0.01)))
    : maxFrames

  const frames: CapturedFrame[] = []
  const mediaTimes: (number | null)[] = []
  let waitForPresent = typeof video.requestVideoFrameCallback === "function"
  let detectMs = 0
  for (let i = 0; i < total; i++) {
    // Mid-frame target: seeking exactly onto a frame boundary can present frame i-1
    // (float rounding, and Chrome keeps a frame whose end time equals the seek time).
    const mediaTime = await seekAndDraw(
      video,
      (i + 0.5) / fps,
      ctx,
      signal,
      waitForPresent
    )
    // The first seek always changes frame (the fps probe left playback further in), so no
    // callback here means this browser doesn't report paused seeks: stop waiting for one.
    if (i === 0 && mediaTime === null) waitForPresent = false
    const { data } = ctx.getImageData(0, 0, width, height)
    const gray = rgbaToGray(data, width, height)
    const tMs = (i * 1000) / fps
    const t0 = performance.now()
    const keypoints = detector.detect(canvas, tMs)
    detectMs += performance.now() - t0
    frames.push({ tMs, width, height, gray, keypoints })
    mediaTimes.push(mediaTime)
    onProgress(i + 1, total)
  }

  // A duration padded by a longer audio track (or unknown) makes the last seeks clamp onto the
  // final frame; drop those repeats so the count matches a decoder that reads every frame.
  while (
    frames.length > 1 &&
    equalBytes(frames[frames.length - 1].gray, frames[frames.length - 2].gray)
  ) {
    frames.pop()
    mediaTimes.pop()
  }
  return { frames, mediaTimes, detectMs }
}

async function recognizeWith(
  recognizer: Recognizer,
  crops: CropResult,
  signal: AbortSignal
): Promise<LabRecognition> {
  // Read after the call: a recognizer may only learn it is unavailable when it is used.
  const info = (): RecognizerInfo => ({
    name: recognizer.name,
    available: recognizer.available,
    isReal: recognizer.isReal,
  })
  try {
    const r = await recognizer.recognize(crops, signal)
    return {
      ...info(),
      text: r.text,
      engine: r.engine,
      latencyMs: r.latencyMs,
      confidence: r.confidence,
      mode: r.mode,
      fellBack: r.fellBack ?? false,
      serverLatencyMs: r.serverLatencyMs,
    }
  } catch (err) {
    return { ...info(), error: describeError(err) }
  }
}

// --- Video helpers ---

function createHiddenVideo(host: HTMLElement): HTMLVideoElement {
  const video = document.createElement("video")
  video.muted = true
  video.playsInline = true
  video.crossOrigin = "anonymous"
  video.preload = "auto"
  // Stays in the document (1px, near-transparent host) so the compositor keeps presenting
  // frames, which requestVideoFrameCallback depends on.
  video.style.width = "1px"
  video.style.height = "1px"
  host.append(video)
  return video
}

function disposeVideo(video: HTMLVideoElement) {
  video.pause()
  video.removeAttribute("src")
  video.load()
  video.remove()
}

async function openClip(
  video: HTMLVideoElement,
  url: string,
  signal: AbortSignal
): Promise<void> {
  const loaded = waitForMediaEvent(video, "loadeddata", signal, LOAD_TIMEOUT_MS)
  video.src = url
  try {
    await loaded
  } catch (err) {
    signal.throwIfAborted()
    throw new Error(`could not load ${url}: ${describeError(err)}`, {
      cause: err,
    })
  }
}

async function seekTo(
  video: HTMLVideoElement,
  timeS: number,
  signal: AbortSignal
): Promise<void> {
  const seeked = waitForMediaEvent(video, "seeked", signal, SEEK_TIMEOUT_MS)
  video.currentTime = timeS
  await seeked
}

/**
 * Seek, then draw the frame the seek presents into `ctx`. Drawing inside the
 * requestVideoFrameCallback is what makes this deterministic: right after "seeked", Chrome can
 * still hand drawImage the previous frame (seen ~1 in 400 seeks). Returns the presented frame's
 * media time, or null when no new frame was presented (seek landed on the frame already shown,
 * or `waitForPresent` is off) — the frame on screen is then drawn after "seeked".
 */
async function seekAndDraw(
  video: HTMLVideoElement,
  timeS: number,
  ctx: CanvasRenderingContext2D,
  signal: AbortSignal,
  waitForPresent: boolean
): Promise<number | null> {
  const { videoWidth: width, videoHeight: height } = video
  if (!waitForPresent) {
    await seekTo(video, timeS, signal)
    ctx.drawImage(video, 0, 0, width, height)
    return null
  }

  let handle = 0
  let timer = 0
  const presented = new Promise<number>((resolve) => {
    handle = video.requestVideoFrameCallback((_now, metadata) => {
      ctx.drawImage(video, 0, 0, width, height)
      resolve(metadata.mediaTime)
    })
  })
  try {
    await seekTo(video, timeS, signal)
    // The callback lands around "seeked" (either side); the grace period starts after the
    // seek so slow long-GOP seeks don't fall back to the racy path.
    const mediaTime = await Promise.race([
      presented,
      new Promise<null>((resolve) => {
        timer = window.setTimeout(() => resolve(null), PRESENT_TIMEOUT_MS)
      }),
    ])
    if (mediaTime === null) ctx.drawImage(video, 0, 0, width, height)
    return mediaTime
  } finally {
    clearTimeout(timer)
    video.cancelVideoFrameCallback(handle)
  }
}

/** Resolve on `type`; reject on the element's error event, abort, or timeout. */
function waitForMediaEvent(
  media: HTMLMediaElement,
  type: "loadeddata" | "seeked",
  signal: AbortSignal,
  timeoutMs: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason)
      return
    }
    const listeners = new AbortController()
    const settle = (err?: unknown) => {
      listeners.abort()
      clearTimeout(timer)
      if (err === undefined) resolve()
      else reject(err)
    }
    const timer = setTimeout(
      () =>
        settle(
          new Error(`timed out after ${timeoutMs} ms waiting for "${type}"`)
        ),
      timeoutMs
    )
    const options = { signal: listeners.signal }
    media.addEventListener(type, () => settle(), options)
    media.addEventListener(
      "error",
      () => settle(new Error(mediaErrorMessage(media))),
      options
    )
    signal.addEventListener("abort", () => settle(signal.reason), options)
  })
}

function mediaErrorMessage(media: HTMLMediaElement): string {
  const codes = [
    "",
    "aborted",
    "network error",
    "decode error",
    "unsupported source",
  ]
  const err = media.error
  if (!err) return "media error"
  return `video ${codes[err.code] ?? "error"}${err.message ? `: ${err.message}` : ""}`
}

/**
 * Measure the clip's frame rate from the media timestamps requestVideoFrameCallback reports
 * while it plays briefly (muted). Returns null when the browser can't tell us.
 */
async function probeFps(
  video: HTMLVideoElement,
  signal: AbortSignal
): Promise<number | null> {
  if (typeof video.requestVideoFrameCallback !== "function") return null

  const mediaTimes: number[] = []
  const listeners = new AbortController()
  let handle = 0
  let timer = 0
  const sampled = new Promise<void>((resolve) => {
    const onFrame: VideoFrameRequestCallback = (_now, metadata) => {
      mediaTimes.push(metadata.mediaTime)
      if (mediaTimes.length >= FPS_PROBE_SAMPLES) resolve()
      else handle = video.requestVideoFrameCallback(onFrame)
    }
    handle = video.requestVideoFrameCallback(onFrame)
    timer = window.setTimeout(resolve, FPS_PROBE_TIMEOUT_MS)
    video.addEventListener("ended", () => resolve(), {
      signal: listeners.signal,
    })
    signal.addEventListener("abort", () => resolve(), {
      signal: listeners.signal,
    })
    // A refused play() (autoplay policy) just means no measurement.
    video.play().catch(() => resolve())
  })

  try {
    await sampled
  } finally {
    listeners.abort()
    clearTimeout(timer)
    video.cancelVideoFrameCallback(handle)
    video.pause()
  }
  signal.throwIfAborted()
  return fpsFromMediaTimes(mediaTimes)
}

/**
 * Smallest gap between presented frames → fps (dropped frames only ever widen a gap).
 * Snaps to the nearest standard rate within 0.05%, else rounds to 3 decimals.
 */
function fpsFromMediaTimes(mediaTimes: readonly number[]): number | null {
  let minGap = Infinity
  let gaps = 0
  for (let i = 1; i < mediaTimes.length; i++) {
    const gap = mediaTimes[i] - mediaTimes[i - 1]
    if (gap > 1e-4) {
      gaps++
      minGap = Math.min(minGap, gap)
    }
  }
  if (gaps < 2) return null
  const fps = 1 / minGap
  if (fps < 1 || fps > 240) return null
  let best: number | null = null
  for (const rate of STANDARD_FPS) {
    const err = Math.abs(rate - fps) / rate
    if (err < 0.0005 && (best === null || err < Math.abs(best - fps) / best)) {
      best = rate
    }
  }
  return best ?? Math.round(fps * 1000) / 1000
}

// --- Small utilities ---

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}

function bytesToBase64(bytes: Uint8Array): string {
  // Native Uint8Array#toBase64 where available (not yet in our TS lib), chunked btoa otherwise.
  const native = (bytes as Uint8Array & { toBase64?: () => string }).toBase64
  if (typeof native === "function") return native.call(bytes)
  let binary = ""
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(binary)
}

function roundTimings(timings: Record<string, number>): Record<string, number> {
  return Object.fromEntries(
    Object.entries(timings).map(([key, ms]) => [key, Math.round(ms * 10) / 10])
  )
}

function describeError(err: unknown): string {
  if (err instanceof Error) {
    return err.name && err.name !== "Error"
      ? `${err.name}: ${err.message}`
      : err.message
  }
  return String(err)
}

function isNoFaceError(err: unknown): boolean {
  return (
    err instanceof NoFaceError ||
    (err instanceof Error && err.name === "NoFaceError")
  )
}

function parseFpsParam(value: string | null): number | null {
  if (value === null) return null
  const fps = Number(value)
  return Number.isFinite(fps) && fps >= 1 && fps <= 240 ? fps : null
}

// --- Page ---

export function LabPage() {
  const [searchParams] = useSearchParams()
  const initialUrl = searchParams.get("url") ?? DEFAULT_CLIP_URL
  const autoRun = searchParams.get("auto") === "1"
  const fpsOverride = parseFpsParam(searchParams.get("fps"))

  const [url, setUrl] = useState(initialUrl)
  const [lab, setLab] = useState<LipLabState>(() => idleState(initialUrl))
  const [filmstrip, setFilmstrip] = useState<FilmstripItem[]>([])
  const hostRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const runIdRef = useRef(0)

  const run = useCallback(
    (clipUrl: string) => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      runIdRef.current += 1
      setFilmstrip([])
      void runLab({
        url: clipUrl,
        runId: runIdRef.current,
        fpsOverride,
        host: hostRef.current ?? document.body,
        signal: controller.signal,
        onState: setLab,
        onFilmstrip: setFilmstrip,
      })
    },
    [fpsOverride]
  )

  const abortRun = useCallback(() => controllerRef.current?.abort(), [])

  useEffect(() => {
    window.__lipLab = idleState(initialUrl)
    // Deferred so StrictMode's mount → unmount → mount starts one run, not an aborted extra.
    const timer = autoRun ? setTimeout(() => run(initialUrl), 0) : undefined
    return () => {
      clearTimeout(timer)
      abortRun()
    }
  }, [autoRun, initialUrl, run, abortRun])

  const start = () => run(url.trim() || DEFAULT_CLIP_URL)
  const running = lab.status === "running"
  const facesFound = lab.keypoints.filter(Boolean).length
  // Frames whose presented media time is i/fps (within a quarter frame).
  const alignedFrames = lab.frameMediaTimes.filter(
    (t, i) => t !== null && Math.abs(t * lab.fps - i) < 0.25
  ).length

  const facts: [string, string][] = [
    ["Resolution", lab.width ? `${lab.width}×${lab.height}` : "—"],
    [
      "Duration",
      !lab.durationS
        ? "—"
        : Number.isFinite(lab.durationS)
          ? `${lab.durationS.toFixed(3)} s`
          : "unknown",
    ],
    [
      "Frame rate",
      lab.fps ? `${+lab.fps.toFixed(3)} fps (${lab.fpsSource})` : "—",
    ],
    ["Frames", lab.frames ? String(lab.frames) : "—"],
    [
      "Frames verified",
      lab.frameMediaTimes.length
        ? `${alignedFrames}/${lab.frameMediaTimes.length} at i/fps`
        : "—",
    ],
    [
      "Face coverage",
      lab.keypoints.length
        ? `${Math.round(lab.faceCoverage * 100)}% (${facesFound}/${lab.keypoints.length} raw)`
        : "—",
    ],
    ["Crops T (25 fps)", lab.T ? String(lab.T) : "—"],
  ]

  return (
    <div className="mx-auto flex min-h-svh max-w-5xl flex-col gap-5 p-6 text-xs">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h1 className="text-sm font-semibold">Lip-reading lab</h1>
          <p className="text-muted-foreground">
            Deterministic clip → detect → crop → recognize harness. Query:{" "}
            <code>auto=1</code> runs on load, <code>url=</code> picks the clip,{" "}
            <code>fps=</code> overrides the measured rate. Results:{" "}
            <code>window.__lipLab</code>.
          </p>
        </div>
        <StatusBadge status={lab.status} stage={lab.stage} />
      </header>

      <form
        className="flex items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          start()
        }}
      >
        <label className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="font-medium text-muted-foreground">Clip URL</span>
          <Input
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            inputMode="url"
            spellCheck={false}
            autoComplete="off"
          />
        </label>
        <Button type="button" onClick={start}>
          {running ? "Restart" : "Run"}
        </Button>
      </form>

      {lab.error && (
        <p
          role="alert"
          className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive"
        >
          {lab.error}
        </p>
      )}

      <Section title="Clip">
        <dl className="grid grid-cols-1 gap-x-6 sm:grid-cols-2 lg:grid-cols-3">
          {facts.map(([label, value]) => (
            <div
              key={label}
              className="flex justify-between gap-3 border-b border-border py-1"
            >
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-medium tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <div className="grid gap-3 sm:grid-cols-2">
        <RecognitionCard
          title="Speed"
          hint="on-device ONNX · greedy CTC"
          result={lab.speed}
          pending={running}
        />
        <RecognitionCard
          title="Accuracy"
          hint="hosted · beam + LM"
          result={lab.accuracy}
          pending={running}
        />
      </div>

      <Section
        title={`Mouth crops — every ${FILMSTRIP_STRIDE}th of ${lab.T || "T"}`}
      >
        {filmstrip.length === 0 ? (
          <p className="text-muted-foreground">No crops yet.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {filmstrip.map(({ index, patch }) => (
              <figure
                key={index}
                className="flex flex-col items-center gap-0.5"
              >
                <PatchCanvas patch={patch} size={ACTIVE_SPEC.patchSize} />
                <figcaption className="text-[0.625rem] text-muted-foreground tabular-nums">
                  {index}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </Section>

      <Section title="Timings">
        <TimingsTable timings={lab.timings} frames={lab.frames} />
      </Section>

      {/* Host for the hidden <video> each run creates (see createHiddenVideo). */}
      <div
        ref={hostRef}
        aria-hidden
        className="pointer-events-none fixed right-0 bottom-0 size-px overflow-hidden opacity-[0.01]"
      />
    </div>
  )
}

export default LabPage

// --- Presentational pieces ---

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-[0.625rem] font-semibold tracking-wide text-muted-foreground uppercase">
        {title}
      </h2>
      {children}
    </section>
  )
}

function StatusBadge({ status, stage }: { status: LabStatus; stage: string }) {
  return (
    <span
      aria-live="polite"
      className={cn(
        "rounded px-1.5 py-0.5 font-medium tabular-nums",
        status === "idle" && "bg-muted text-muted-foreground",
        status === "running" && "bg-primary/10 text-primary",
        status === "done" &&
          "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
        status === "error" && "bg-destructive/10 text-destructive"
      )}
    >
      {status === "running" ? stage : status}
    </span>
  )
}

function RecognitionCard({
  title,
  hint,
  result,
  pending,
}: {
  title: string
  hint: string
  result: LabRecognition | null
  pending: boolean
}) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border bg-card p-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-xs font-semibold">{title}</h2>
        <span className="text-[0.625rem] text-muted-foreground">{hint}</span>
      </div>

      {!result ? (
        <p className="text-muted-foreground">
          {pending ? "waiting…" : "not run"}
        </p>
      ) : "error" in result ? (
        <p className="break-words text-destructive">{result.error}</p>
      ) : (
        <>
          <p className="font-mono text-sm break-words">
            {result.text || (
              <span className="text-muted-foreground">(empty)</span>
            )}
          </p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
            <dt className="text-muted-foreground">Engine</dt>
            <dd>{result.engine}</dd>
            <dt className="text-muted-foreground">Latency</dt>
            <dd className="tabular-nums">{formatMs(result.latencyMs)}</dd>
            <dt className="text-muted-foreground">Confidence</dt>
            <dd className="tabular-nums">
              {result.confidence === undefined
                ? "—"
                : `${(result.confidence * 100).toFixed(1)}%`}
            </dd>
            <dt className="text-muted-foreground">Produced by</dt>
            <dd>
              {result.fellBack ? `${result.mode} (fell back)` : result.mode}
            </dd>
            {result.serverLatencyMs && (
              <>
                <dt className="text-muted-foreground">Server</dt>
                <dd className="tabular-nums">
                  {Object.entries(result.serverLatencyMs)
                    .map(([stage, ms]) => `${stage} ${Math.round(ms)}`)
                    .join(" · ")}
                </dd>
              </>
            )}
          </dl>
        </>
      )}

      {result && (
        <div className="flex flex-wrap items-center gap-1 text-[0.625rem]">
          {("error" in result || result.name !== result.engine) && (
            <span className="text-muted-foreground">{result.name}</span>
          )}
          <Tag on={result.isReal}>{result.isReal ? "model" : "mock"}</Tag>
          {!result.available && <Tag on={false}>unavailable</Tag>}
        </div>
      )}
    </div>
  )
}

function Tag({ on, children }: { on: boolean; children: ReactNode }) {
  return (
    <span
      className={cn(
        "rounded px-1 font-medium",
        on ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
      )}
    >
      {children}
    </span>
  )
}

function PatchCanvas({ patch, size }: { patch: Uint8Array; size: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    const image = ctx.createImageData(size, size)
    for (let p = 0, q = 0; p < patch.length; p++, q += 4) {
      image.data[q] = patch[p]
      image.data[q + 1] = patch[p]
      image.data[q + 2] = patch[p]
      image.data[q + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [patch, size])

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="size-16 rounded-sm bg-muted"
    />
  )
}

const TIMING_ROWS: readonly (readonly [key: string, label: string])[] = [
  ["load", "Load clip"],
  ["fpsProbe", "Frame-rate probe"],
  ["detectorInit", "Detector init"],
  ["decodeDetect", "Decode + detect (all frames)"],
  ["detect", "↳ detect only"],
  ["crop", "Crop"],
  ["recognizersInit", "Recognizers init (cached after first run)"],
  ["speedRecognize", "Speed recognize"],
  ["accuracyRecognize", "Accuracy recognize"],
  ["total", "Total"],
]

function TimingsTable({
  timings,
  frames,
}: {
  timings: Record<string, number>
  frames: number
}) {
  return (
    <table className="w-full max-w-lg tabular-nums">
      <tbody>
        {TIMING_ROWS.map(([key, label]) => {
          const ms = timings[key]
          const perFrame =
            ms !== undefined &&
            frames > 0 &&
            (key === "decodeDetect" || key === "detect")
          return (
            <tr key={key} className="border-b border-border last:border-0">
              <td className="py-1 pr-4 text-muted-foreground">{label}</td>
              <td className="py-1 text-right font-medium">
                {ms === undefined ? "—" : formatMs(ms)}
              </td>
              <td className="py-1 pl-3 text-right text-muted-foreground">
                {perFrame ? `${(ms / frames).toFixed(1)} ms/frame` : ""}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`
}
