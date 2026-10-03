import { useCallback, useEffect, useRef, useState } from "react"

import { createLipReaderEngine } from "@/lib/lipreading/createEngine"
import { LipLandmarker } from "@/lib/lipreading/faceLandmarker"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"
import { FrameRingBuffer } from "@/lib/lipreading/ringBuffer"
import type { LipReaderEngine } from "@/lib/lipreading/types"

export type CameraStatus = "idle" | "starting" | "on" | "error"

export interface LipTranscriptItem {
  id: string
  text: string
  /** performance.now() timestamp for ordering alongside listening utterances. */
  at: number
}

interface UseLipReaderOptions {
  /** When true, run recognition over the rolling window. Camera runs regardless. */
  active: boolean
  /** How often to run inference, in ms (inference is the expensive step). */
  inferenceIntervalMs?: number
}

/**
 * Owns the camera "entrypoint" and the full lip-reading pipeline:
 * camera → MediaPipe landmarks → mouth crop → ring buffer → engine → transcript.
 *
 * Two decoupled loops (see architecture): the capture loop runs every animation
 * frame to keep the overlay smooth; inference runs on a slower interval so a slow
 * model never stalls the camera.
 */
export function useLipReader({
  active,
  inferenceIntervalMs = 1500,
}: UseLipReaderOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)

  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle")
  const [engineName, setEngineName] = useState("")
  const [engineReal, setEngineReal] = useState(false)
  const [ready, setReady] = useState(false)
  const [mouthDetected, setMouthDetected] = useState(false)
  const [fps, setFps] = useState(0)
  const [inferring, setInferring] = useState(false)
  const [transcript, setTranscript] = useState<LipTranscriptItem[]>([])

  // Long-lived pipeline objects — refs so the loops see them without re-rendering.
  const landmarkerRef = useRef<LipLandmarker | null>(null)
  const engineRef = useRef<LipReaderEngine | null>(null)
  const bufferRef = useRef<FrameRingBuffer | null>(null)
  const activeRef = useRef(active)
  activeRef.current = active

  const clearTranscript = useCallback(() => setTranscript([]), [])

  // --- Setup: camera + landmarker + engine + capture loop (runs once) ---
  useEffect(() => {
    let cancelled = false
    let rafId = 0
    let stream: MediaStream | null = null
    let lastTs = 0
    let frameTimes: number[] = []

    const buffer = new FrameRingBuffer(ACTIVE_SPEC.windowFrames)
    bufferRef.current = buffer

    const captureLoop = () => {
      const video = videoRef.current
      const landmarker = landmarkerRef.current
      if (video && landmarker && video.readyState >= 2) {
        const now = performance.now()
        const ts = now > lastTs ? now : lastTs + 1
        lastTs = ts

        const { frame, lipPoints } = landmarker.detect(video, ts)
        if (frame) buffer.push(frame)
        setMouthDetected(Boolean(frame))
        drawOverlay(overlayRef.current, video, lipPoints)

        // Rolling fps over the last ~30 frames.
        frameTimes.push(now)
        frameTimes = frameTimes.filter((t) => now - t < 1000)
        setFps(frameTimes.length)
      }
      if (!cancelled) rafId = requestAnimationFrame(captureLoop)
    }

    const setup = async () => {
      try {
        setCameraStatus("starting")
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        })
        if (cancelled) return
        const video = videoRef.current
        if (video) {
          video.srcObject = stream
          await video.play().catch(() => undefined)
        }
        setCameraStatus("on")

        const landmarker = new LipLandmarker(ACTIVE_SPEC)
        await landmarker.init()
        if (cancelled) {
          landmarker.dispose()
          return
        }
        landmarkerRef.current = landmarker

        const engine = await createLipReaderEngine(ACTIVE_SPEC)
        if (cancelled) {
          engine.dispose()
          return
        }
        engineRef.current = engine
        setEngineName(engine.name)
        setEngineReal(engine.isReal)
        setReady(true)

        rafId = requestAnimationFrame(captureLoop)
      } catch (err) {
        console.error("[useLipReader] setup failed:", err)
        if (!cancelled) setCameraStatus("error")
      }
    }

    void setup()

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      stream?.getTracks().forEach((t) => t.stop())
      landmarkerRef.current?.dispose()
      landmarkerRef.current = null
      engineRef.current?.dispose()
      engineRef.current = null
      bufferRef.current = null
      setReady(false)
      setCameraStatus("idle")
    }
  }, [])

  // --- Inference loop: only while `active` and the engine is ready ---
  useEffect(() => {
    if (!active || !ready) return

    let stopped = false
    let running = false

    const tick = async () => {
      const engine = engineRef.current
      const buffer = bufferRef.current
      if (running || !engine || !buffer || !buffer.isFull || !activeRef.current) {
        return
      }
      running = true
      setInferring(true)
      try {
        const result = await engine.infer(buffer.snapshot())
        if (!stopped && result.text.trim()) {
          const now = performance.now()
          setTranscript((prev) => [
            ...prev,
            { id: `${now}-${prev.length}`, text: result.text.trim(), at: now },
          ])
        }
      } catch (err) {
        console.error("[useLipReader] inference failed:", err)
      } finally {
        running = false
        if (!stopped) setInferring(false)
      }
    }

    const interval = setInterval(tick, inferenceIntervalMs)
    return () => {
      stopped = true
      clearInterval(interval)
      setInferring(false)
    }
  }, [active, ready, inferenceIntervalMs])

  return {
    videoRef,
    overlayRef,
    cameraStatus,
    engineName,
    engineReal,
    ready,
    mouthDetected,
    fps,
    inferring,
    transcript,
    clearTranscript,
  }
}

/** Draw lip landmarks onto the overlay canvas (debug visualization). */
function drawOverlay(
  canvas: HTMLCanvasElement | null,
  video: HTMLVideoElement,
  lipPoints: readonly { x: number; y: number }[]
) {
  if (!canvas) return
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  const w = video.clientWidth || canvas.width
  const h = video.clientHeight || canvas.height
  if (canvas.width !== w) canvas.width = w
  if (canvas.height !== h) canvas.height = h

  ctx.clearRect(0, 0, w, h)
  if (lipPoints.length === 0) return

  ctx.fillStyle = "oklch(0.72 0.19 150)"
  for (const p of lipPoints) {
    ctx.beginPath()
    ctx.arc(p.x * w, p.y * h, 1.5, 0, Math.PI * 2)
    ctx.fill()
  }
}
