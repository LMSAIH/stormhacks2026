import { useEffect, useRef, useState } from "react"
import { Check, Cloud, Gauge, Laptop } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  LIP_MODE_ORDER,
  LIP_MODES,
  type LipMode,
} from "@/lib/lipreading/modes"

interface LipModeMenuProps {
  mode: LipMode
  onChange: (mode: LipMode) => void
  /** The GPU server answers; when false, Quality falls back to reading on this device. */
  cloudAvailable: boolean
}

/** Click-to-open picker for how lip reading runs, marking which modes are local vs cloud. */
export function LipModeMenu({ mode, onChange, cloudAvailable }: LipModeMenuProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false)
    window.addEventListener("mousedown", onDown)
    window.addEventListener("keydown", onKey)
    return () => {
      window.removeEventListener("mousedown", onDown)
      window.removeEventListener("keydown", onKey)
    }
  }, [open])

  const current = LIP_MODES[mode]
  return (
    <div ref={ref} className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen((o) => !o)}
        aria-label="Lip reading mode"
        aria-expanded={open}
      >
        <Gauge />
        {current.label}
      </Button>

      {open && (
        <div className="absolute top-full left-0 z-50 mt-1.5 w-72 overflow-hidden rounded-xl border border-border bg-popover p-1 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95">
          {LIP_MODE_ORDER.map((m) => {
            const info = LIP_MODES[m]
            const offline = info.where === "Cloud" && !cloudAvailable
            const Where = info.where === "Cloud" ? Cloud : Laptop
            return (
              <button
                key={m}
                type="button"
                onClick={() => {
                  onChange(m)
                  setOpen(false)
                }}
                className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted"
              >
                <Where className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                <span className="flex-1">
                  <span className="flex items-center gap-1.5">
                    {info.label}
                    <span className="rounded border border-border px-1 text-[0.625rem] text-muted-foreground">
                      {info.where} · up to {info.maxSeconds} s
                    </span>
                  </span>
                  <span className="block text-xs text-muted-foreground">
                    {offline
                      ? "Server offline: reads on this device instead"
                      : info.detail}
                  </span>
                </span>
                {m === mode && <Check className="mt-0.5 size-3.5 text-foreground" />}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
