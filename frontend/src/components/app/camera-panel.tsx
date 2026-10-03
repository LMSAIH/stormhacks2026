import type { RefObject } from "react"
import { cn } from "cn"

import type { CameraStatus, EngineStatus } from "@/hooks/useLipReader"
import {
  INPUT_SIZE,
  PATCH_SIZE,
  applyAffine,
  estimateSimilarity,
  patchOrigin,
} from "@/lib/lipreading/crop"
import type { Keypoints, Point, RecognitionMode } from "@/lib/lipreading/types"

interface LastResult {
  latencyMs: number
  mode: RecognitionMode
  engine: string
  fellBack?: boolean
}

interface CameraPanelProps {
  videoRef: RefObject<HTMLVideoElement | null>
  overlayRef: RefObject<HTMLCanvasElement | null>
  cameraStatus: CameraStatus
  mouthDetected: boolean
  fps: number
  recording: boolean
  busy: boolean
  mode: RecognitionMode
  /** Engine behind the current mode. */
  engine: EngineStatus
  /** Most recent recognition, for the latency readout. */
  last?: LastResult
  className?: string
}

const MODE_LABEL: Record<RecognitionMode, string> = {
  speed: "Speed",
  accuracy: "Accuracy",
}

/**
 * Small debug camera — the pipeline entrypoint. Mirrored feed, the 4 BlazeFace keypoints + the
 * mouth crop the model sees, and live pipeline status (face, fps, mode, engine, last latency).
 */
export function CameraPanel({
  videoRef,
  overlayRef,
  cameraStatus,
  mouthDetected,
  fps,
  recording,
  busy,
  mode,
  engine,
  last,
  className,
}: CameraPanelProps) {
  const badge = engine.loading
    ? { label: "loading", tone: "muted" as const }
    : !engine.available
      ? { label: "unavailable", tone: "warn" as const }
      : engine.isReal
        ? { label: "model", tone: "primary" as const }
        : { label: "mock", tone: "muted" as const }

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border bg-card shadow-sm transition-colors",
        recording ? "border-red-500/60" : "border-border",
        className
      )}
    >
      <div className="relative aspect-[4/3] bg-muted">
        {/* Mirror video and overlay together (the frames themselves stay un-mirrored). */}
        <div className="absolute inset-0 -scale-x-100">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full object-cover"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        </div>

        {cameraStatus !== "on" && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
            {cameraStatus === "error"
              ? "Camera unavailable"
              : "Starting camera…"}
          </div>
        )}

        <div className="absolute top-1.5 left-1.5 flex items-center gap-1">
          <StatusDot
            on={mouthDetected}
            label={mouthDetected ? "face" : "no face"}
          />
          {recording && <StatusDot on label="rec" tone="red" pulse />}
          {busy && <StatusDot on label="reading" pulse />}
        </div>
        <div className="absolute top-1.5 right-1.5 rounded bg-background/70 px-1 text-[0.625rem] font-medium text-muted-foreground tabular-nums backdrop-blur">
          {fps} fps
        </div>
      </div>

      <div className="space-y-0.5 px-2 py-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="min-w-0 truncate text-[0.625rem] text-muted-foreground">
            <span className="font-medium text-foreground">
              {MODE_LABEL[mode]}
            </span>
            {" · "}
            {engine.name}
          </span>
          <span
            className={cn(
              "shrink-0 rounded px-1 text-[0.625rem] font-medium",
              badge.tone === "primary" && "bg-primary/10 text-primary",
              badge.tone === "muted" && "bg-muted text-muted-foreground",
              badge.tone === "warn" &&
                "bg-amber-500/15 text-amber-700 dark:text-amber-300"
            )}
          >
            {badge.label}
          </span>
        </div>
        <p className="truncate text-[0.625rem] text-muted-foreground tabular-nums">
          {last ? (
            <>
              last{" "}
              <span className="font-medium text-foreground">
                {formatLatency(last.latencyMs)}
              </span>
              {" · "}
              {last.fellBack
                ? "speed (fallback)"
                : MODE_LABEL[last.mode].toLowerCase()}
              {" · "}
              {last.engine}
            </>
          ) : (
            "no utterances yet"
          )}
        </p>
      </div>
    </div>
  )
}

function StatusDot({
  on,
  label,
  pulse,
  tone = "green",
}: {
  on: boolean
  label: string
  pulse?: boolean
  tone?: "green" | "red"
}) {
  return (
    <span className="flex items-center gap-1 rounded bg-background/70 px-1 py-0.5 text-[0.625rem] font-medium text-muted-foreground backdrop-blur">
      <span
        className={cn(
          "size-1.5 rounded-full",
          !on && "bg-muted-foreground/40",
          on && tone === "green" && "bg-emerald-500",
          on && tone === "red" && "bg-red-500",
          pulse && "animate-pulse"
        )}
      />
      {label}
    </span>
  )
}

export function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

const MOUTH_COLOR = "oklch(0.72 0.19 150)"
const POINT_COLOR = "oklch(0.97 0 0)"

/**
 * Draw the 4 keypoints and the mouth region the model sees onto the overlay canvas. Called from
 * the capture loop every frame (not a React render). `frameWidth/Height` are the video's pixel
 * size; the mapping matches the video's `object-cover` fit. The canvas sits in the same mirrored
 * box as the video, so drawing in un-mirrored frame coordinates lines up.
 */
export function drawFaceOverlay(
  canvas: HTMLCanvasElement | null,
  frameWidth: number,
  frameHeight: number,
  keypoints: Keypoints | null
): void {
  if (!canvas) return
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const cssWidth = canvas.clientWidth
  const cssHeight = canvas.clientHeight
  if (cssWidth === 0 || cssHeight === 0) return
  const dpr = window.devicePixelRatio || 1
  const width = Math.round(cssWidth * dpr)
  const height = Math.round(cssHeight * dpr)
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  ctx.clearRect(0, 0, width, height)
  if (!keypoints || frameWidth === 0 || frameHeight === 0) return

  const scale = Math.max(width / frameWidth, height / frameHeight)
  const offsetX = (width - frameWidth * scale) / 2
  const offsetY = (height - frameHeight * scale) / 2
  const toCanvas = ([x, y]: Point): Point => [
    offsetX + x * scale,
    offsetY + y * scale,
  ]

  const box = modelInputCorners(keypoints)
  if (box) {
    ctx.beginPath()
    box.forEach((corner, i) => {
      const [x, y] = toCanvas(corner)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.lineWidth = 1.5 * dpr
    ctx.setLineDash([4 * dpr, 3 * dpr])
    ctx.strokeStyle = MOUTH_COLOR
    ctx.stroke()
    ctx.setLineDash([])
  }

  ctx.lineWidth = dpr
  ctx.strokeStyle = "rgb(0 0 0 / 0.45)"
  keypoints.forEach((point, i) => {
    const [x, y] = toCanvas(point)
    ctx.beginPath()
    ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2)
    ctx.fillStyle = i === 3 ? MOUTH_COLOR : POINT_COLOR
    ctx.fill()
    ctx.stroke()
  })
}

/**
 * Source-pixel corners of the 88² model input for one frame's raw keypoints, using the crop
 * pipeline's own geometry (similarity onto the mean face → 96² patch origin → centre 88²) mapped
 * back through the inverse transform. Approximate only because the real crop smooths keypoints
 * over ±6 frames first.
 */
function modelInputCorners(kps: Keypoints): Point[] | null {
  const m = estimateSimilarity(kps)
  const [a, b, c, d, e, f] = m
  const det = a * e - b * d
  if (!m.every(Number.isFinite) || det === 0) return null
  const [cx, cy] = applyAffine(m, kps[3])
  const [px, py] = patchOrigin(cx, cy)
  const margin = (PATCH_SIZE - INPUT_SIZE) / 2
  const x0 = px + margin
  const y0 = py + margin

  const toSource = (u: number, v: number): Point => [
    (e * (u - c) - b * (v - f)) / det,
    (a * (v - f) - d * (u - c)) / det,
  ]
  return [
    toSource(x0, y0),
    toSource(x0 + INPUT_SIZE, y0),
    toSource(x0 + INPUT_SIZE, y0 + INPUT_SIZE),
    toSource(x0, y0 + INPUT_SIZE),
  ]
}
