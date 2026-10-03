import type { RefObject } from "react"

import type { CameraStatus } from "@/hooks/useLipReader"

interface CameraPanelProps {
  videoRef: RefObject<HTMLVideoElement | null>
  overlayRef: RefObject<HTMLCanvasElement | null>
  cameraStatus: CameraStatus
}

/**
 * Camera feed — the pipeline entrypoint. Shows the mirrored video with the
 * lip-landmark overlay. Intentionally minimal (debug visualization).
 */
export function CameraPanel({
  videoRef,
  overlayRef,
  cameraStatus,
}: CameraPanelProps) {
  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="relative flex-1 bg-muted">
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
      </div>
    </div>
  )
}
