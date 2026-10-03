import { CircleAlert } from "lucide-react"
import { cn } from "cn"

import { ModeToggle } from "@/components/app/mode-toggle"
import { PttButton } from "@/components/app/ptt-button"
import type { EngineStatus, LipTranscriptItem } from "@/hooks/useLipReader"
import { formatLatency } from "@/lib/lipreading/format"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"
import type { RecognitionMode } from "@/lib/lipreading/types"

interface LipControlsProps {
  mode: RecognitionMode
  onModeChange: (mode: RecognitionMode) => void
  /** Per-engine status (drives the dimmed / "unavailable" labels on the mode toggle). */
  engines: Record<RecognitionMode, EngineStatus>
  recording: boolean
  busy: boolean
  /** Camera is on and the face detector is loaded — an utterance can be recorded. */
  captureReady: boolean
  onStart: () => void
  onStop: () => void
  lastError: string | null
  /** Most recent recognition, for the latency readout. */
  last?: Pick<LipTranscriptItem, "latencyMs" | "mode" | "fellBack" | "engine">
  className?: string
}

/**
 * Lip-reading controls under the camera: Speed / Accuracy toggle, push-to-talk, and a one-line
 * status (the last error, else how the last utterance was read). Holding Space does the same as
 * the button (see useLipReader).
 */
export function LipControls({
  mode,
  onModeChange,
  engines,
  recording,
  busy,
  captureReady,
  onStart,
  onStop,
  lastError,
  last,
  className,
}: LipControlsProps) {
  return (
    <div className={cn("shrink-0", className)}>
      <div className="flex flex-wrap items-stretch gap-3">
        <ModeToggle
          className="shrink-0"
          mode={mode}
          onChange={onModeChange}
          engines={engines}
        />
        <PttButton
          className="min-w-44 flex-1"
          recording={recording}
          busy={busy}
          disabled={!captureReady}
          busyLabel={engines[mode].loading ? "Loading model…" : "Reading lips…"}
          maxSeconds={ACTIVE_SPEC.maxSeconds}
          onStart={onStart}
          onStop={onStop}
        />
      </div>

      {/* One status line; takes no space (keeping the page's even gaps) until there is something
          to say. The live region itself stays mounted so changes are announced. */}
      <div
        className={cn(
          "px-1 text-[0.625rem] leading-snug",
          (lastError || last) && "mt-1.5"
        )}
      >
        <p
          role="status"
          aria-live="polite"
          className="flex items-start gap-1 text-amber-700 dark:text-amber-300"
        >
          {lastError && (
            <>
              <CircleAlert className="mt-px size-3 shrink-0" />
              {lastError}
            </>
          )}
        </p>
        {!lastError && last && (
          <p className="truncate text-muted-foreground tabular-nums">
            last{" "}
            <span className="font-medium text-foreground">
              {formatLatency(last.latencyMs)}
            </span>
            {" · "}
            {last.fellBack ? "speed (fallback)" : last.mode}
            {" · "}
            {last.engine}
          </p>
        )}
      </div>
    </div>
  )
}
