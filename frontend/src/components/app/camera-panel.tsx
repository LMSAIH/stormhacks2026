import type { RefObject } from "react"
import { cn } from "cn"

import type { CameraStatus } from "@/hooks/useLipReader"

interface CameraPanelProps {
  videoRef: RefObject<HTMLVideoElement | null>
  overlayRef: RefObject<HTMLCanvasElement | null>
  cameraStatus: CameraStatus
  mouthDetected: boolean
  fps: number
  engineName: string
  engineReal: boolean
  inferring: boolean
}

/**
 * Small debug camera — the pipeline entrypoint. Shows the mirrored feed, the
 * lip-landmark overlay, and live pipeline status. Intentionally minimal.
 */
export function CameraPanel({
  videoRef,
  overlayRef,
  cameraStatus,
  mouthDetected,
  fps,
  engineName,
  engineReal,
  inferring,
}: CameraPanelProps) {
  return (
    <div className="w-56 overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="relative aspect-[4/3] bg-muted">
        {/* Mirror both video and overlay together. */}
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
            {cameraStatus === "error" ? "Camera unavailable" : "Starting camera…"}
          </div>
        )}

        <div className="absolute top-1.5 left-1.5 flex items-center gap-1">
          <StatusDot
            on={mouthDetected}
            label={mouthDetected ? "mouth" : "no face"}
          />
          {inferring && <StatusDot on label="infer" pulse />}
        </div>
        <div className="absolute top-1.5 right-1.5 rounded bg-background/70 px-1 text-[0.625rem] font-medium text-muted-foreground tabular-nums backdrop-blur">
          {fps} fps
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 px-2 py-1.5">
        <span className="truncate text-[0.625rem] text-muted-foreground">
          {engineName || "loading engine…"}
        </span>
        <span
          className={cn(
            "shrink-0 rounded px-1 text-[0.625rem] font-medium",
            engineReal
              ? "bg-primary/10 text-primary"
              : "bg-muted text-muted-foreground"
          )}
        >
          {engineReal ? "model" : "mock"}
        </span>
      </div>
    </div>
  )
}

function StatusDot({
  on,
  label,
  pulse,
}: {
  on: boolean
  label: string
  pulse?: boolean
}) {
  return (
    <span className="flex items-center gap-1 rounded bg-background/70 px-1 py-0.5 text-[0.625rem] font-medium text-muted-foreground backdrop-blur">
      <span
        className={cn(
          "size-1.5 rounded-full",
          on ? "bg-emerald-500" : "bg-muted-foreground/40",
          pulse && "animate-pulse"
        )}
      />
      {label}
    </span>
  )
}
