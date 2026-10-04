import { useEffect, useState, type PointerEvent } from "react"
import { LoaderCircle, Mic } from "lucide-react"
import { cn } from "cn"

interface PttButtonProps {
  recording: boolean
  busy: boolean
  /** Capture pipeline not ready yet (camera / face detector). */
  disabled?: boolean
  /** Shown with the spinner while busy (e.g. still loading the model). */
  busyLabel?: string
  maxSeconds: number
  onStart: () => void
  onStop: () => void
  className?: string
}

/**
 * Press-and-hold to record one utterance (mirrors holding Space). Shows elapsed time and how much
 * of the max length is used while recording, and a spinner while the utterance is recognized.
 */
export function PttButton({
  recording,
  busy,
  disabled = false,
  busyLabel = "Reading lips…",
  maxSeconds,
  onStart,
  onStop,
  className,
}: PttButtonProps) {
  const elapsedMs = useElapsedWhile(recording)
  const progress = Math.min(1, elapsedMs / (maxSeconds * 1000))
  const inactive = disabled || (busy && !recording)

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (inactive || (e.pointerType === "mouse" && e.button !== 0)) return
    // Keep receiving the release even if the pointer slides off the button.
    e.currentTarget.setPointerCapture(e.pointerId)
    onStart()
  }

  return (
    <button
      type="button"
      disabled={inactive}
      aria-pressed={recording}
      onPointerDown={onPointerDown}
      onPointerUp={onStop}
      onPointerCancel={onStop}
      onLostPointerCapture={onStop}
      onContextMenu={(e) => e.preventDefault()}
      className={cn(
        "relative flex w-full touch-none flex-col items-center justify-center gap-0.5 overflow-hidden rounded-xl border px-3 py-2.5 shadow-sm transition-colors outline-none select-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-60",
        recording
          ? "border-red-500/50 bg-red-500/10 text-red-700 dark:text-red-300"
          : "border-border bg-card text-foreground enabled:hover:bg-muted",
        className
      )}
    >
      <span className="flex items-center gap-1.5 text-xs font-medium">
        {busy ? (
          <LoaderCircle className="size-3.5 animate-spin" />
        ) : (
          <Mic className={cn("size-3.5", recording && "animate-pulse")} />
        )}
        {busy
          ? busyLabel
          : recording
            ? `Recording ${(elapsedMs / 1000).toFixed(1)} s`
            : disabled
              ? "Getting ready…"
              : "Hold to talk"}
      </span>
      <span className="text-[0.625rem] text-muted-foreground">
        {recording ? (
          `release to send · max ${maxSeconds} s`
        ) : (
          <>
            or hold{" "}
            <kbd className="rounded border border-border bg-muted px-1 font-sans text-[0.5625rem]">
              Space
            </kbd>
          </>
        )}
      </span>

      {recording && (
        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-red-500/15">
          <span
            className="block h-full bg-red-500 transition-[width] duration-100 ease-linear"
            style={{ width: `${progress * 100}%` }}
          />
        </span>
      )}
    </button>
  )
}

/** Milliseconds since `active` became true (0 while inactive), ticking every 100 ms. */
function useElapsedWhile(active: boolean): number {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!active) return
    const start = performance.now()
    const id = window.setInterval(
      () => setElapsed(performance.now() - start),
      100
    )
    return () => {
      window.clearInterval(id)
      setElapsed(0)
    }
  }, [active])
  return active ? elapsed : 0
}
