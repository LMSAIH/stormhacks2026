import { useRef, type KeyboardEvent } from "react"
import { Cloud, Zap, type LucideIcon } from "lucide-react"
import { cn } from "cn"

import type { EngineStatus } from "@/hooks/useLipReader"
import type { RecognitionMode } from "@/lib/lipreading/types"

// Latencies measured end to end in the browser (Phase A): ONNX WASM/WebGPU ~0.8–1.2 s, RunPod
// beam ~1.8–3 s round trip (Vancouver → pod) with gzipped crops.
const OPTIONS: readonly {
  mode: RecognitionMode
  label: string
  detail: string
  icon: LucideIcon
}[] = [
  {
    mode: "speed",
    label: "Speed",
    detail: "on-device · private · ~1 s",
    icon: Zap,
  },
  {
    mode: "accuracy",
    label: "Accuracy",
    detail: "cloud beam + LM · ~2–3 s",
    icon: Cloud,
  },
]

interface ModeToggleProps {
  mode: RecognitionMode
  /** Selecting an unavailable side asks the hook to re-probe it (e.g. the pod came back up). */
  onChange: (mode: RecognitionMode) => void
  /** Per-engine status; a side still `loading` is not known to be unavailable yet. */
  engines: Record<RecognitionMode, EngineStatus>
  className?: string
}

/**
 * Speed / Accuracy segmented control. A side whose engine is unavailable (no model file, hosted
 * service not configured or down) is dimmed and labelled — clicking it retries; mock engines are
 * labelled too. Arrow keys switch sides (Space is push-to-talk app-wide).
 */
export function ModeToggle({
  mode,
  onChange,
  engines,
  className,
}: ModeToggleProps) {
  const buttonsRef = useRef<
    Partial<Record<RecognitionMode, HTMLButtonElement | null>>
  >({})

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key))
      return
    e.preventDefault()
    const other: RecognitionMode = mode === "speed" ? "accuracy" : "speed"
    onChange(other)
    buttonsRef.current[other]?.focus()
  }

  return (
    <div
      role="radiogroup"
      aria-label="Recognition mode"
      onKeyDown={onKeyDown}
      className={cn(
        "inline-flex gap-0.5 rounded-lg border border-border bg-muted/60 p-0.5",
        className
      )}
    >
      {OPTIONS.map(({ mode: option, label, detail, icon: Icon }) => {
        const engine = engines[option]
        const selected = option === mode
        const unavailable = !engine.loading && !engine.available
        const note = engine.loading
          ? "loading"
          : unavailable
            ? "unavailable"
            : engine.isReal
              ? null
              : "mock"
        return (
          <button
            key={option}
            ref={(el) => {
              buttonsRef.current[option] = el
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            title={
              unavailable ? `${engine.name} — click to retry` : engine.name
            }
            onClick={() => onChange(option)}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/30",
              selected
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
              unavailable && !selected && "opacity-60"
            )}
          >
            <Icon className="size-3.5 shrink-0" />
            <span className="flex flex-col leading-tight">
              <span className="flex items-center gap-1 text-xs font-medium">
                {label}
                {note && (
                  <span
                    className={cn(
                      "rounded px-1 text-[0.5625rem] font-medium",
                      note === "unavailable"
                        ? "bg-amber-500/15 text-amber-700 dark:text-amber-300"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    {note}
                  </span>
                )}
              </span>
              <span className="hidden text-[0.625rem] text-muted-foreground sm:block">
                {detail}
              </span>
            </span>
          </button>
        )
      })}
    </div>
  )
}
