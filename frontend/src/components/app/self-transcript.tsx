import { useEffect, useRef, useState } from "react"
import { Loader2 } from "lucide-react"
import { cn } from "cn"

import { Avatar } from "@/components/app/avatar"
import type { LipTranscriptItem } from "@/hooks/useLipReader"

interface SelfTranscriptProps {
  items: LipTranscriptItem[]
  ready: boolean
  inferring: boolean
  /** The sentence still being spoken (quick drafts), shown greyed after the finished ones. */
  draft?: string | null
  /** Pick one of the newest line's choices (top 3 readings). */
  onPick?: (itemId: string, text: string) => void
  /** Why the camera view reads badly (too far / dark / turned), if anything. */
  hint?: string | null
}

/** How close to the bottom (px) still counts as "stuck to bottom". */
const STICK_THRESHOLD = 48

/** Box under the video: a running transcript of what *you* are saying (lips). */
export function SelfTranscript({
  items,
  ready,
  inferring,
  draft,
  onPick,
  hint,
}: SelfTranscriptProps) {
  const newest = items[items.length - 1]
  const choices = newest?.choices?.filter((c) => c !== newest.text) ?? []
  const scrollRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)
  const [atBottom, setAtBottom] = useState(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight
    stickToBottom.current = distance < STICK_THRESHOLD
    setAtBottom(distance < 8)
  }

  useEffect(() => {
    if (!stickToBottom.current) return
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
    setAtBottom(true)
  }, [items, draft])

  return (
    <div className="relative flex shrink-0 flex-col gap-1.5 overflow-hidden rounded-xl border border-border bg-card p-3 shadow-sm">
      <div className="flex items-center gap-2">
        <Avatar label="You" color="var(--primary)" className="size-6 text-[0.625rem]" />
        <span className="text-xs font-semibold">You</span>
        {inferring && (
          <Loader2 className="size-3 animate-spin text-muted-foreground" />
        )}
        {hint && (
          <span className="ml-auto text-xs text-amber-600 dark:text-amber-400">{hint}</span>
        )}
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="no-scrollbar max-h-24 overflow-y-auto text-sm leading-relaxed"
      >
        {items.length === 0 && !draft ? (
          <p className="text-muted-foreground">
            {ready ? "Mouth words to the camera…" : "Warming up…"}
          </p>
        ) : (
          <p>
            {items.map((item, i) => (
              <span
                key={item.id}
                className={
                  i === items.length - 1
                    ? "text-foreground"
                    : "text-muted-foreground"
                }
              >
                {item.text}
                {i < items.length - 1 || draft ? " · " : ""}
              </span>
            ))}
            {draft && (
              <span className="italic text-muted-foreground/70">{draft}…</span>
            )}
          </p>
        )}
        {onPick && newest && choices.length > 0 && !draft && (
          <div className="mt-1 flex flex-wrap gap-1">
            {choices.map((text) => (
              <button
                key={text}
                type="button"
                onClick={() => onPick(newest.id, text)}
                className="rounded-md border border-border px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                title="Use this reading instead"
              >
                {text}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Blur hint: fades in at the bottom while content remains below. */}
      <div
        aria-hidden
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-card/90 to-transparent backdrop-blur-[2px] transition-opacity duration-200",
          atBottom ? "opacity-0" : "opacity-100"
        )}
        style={{
          maskImage: "linear-gradient(to top, black 40%, transparent)",
          WebkitMaskImage: "linear-gradient(to top, black 40%, transparent)",
        }}
      />
    </div>
  )
}
