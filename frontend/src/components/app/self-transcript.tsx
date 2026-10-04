import { useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Check, Loader2, X } from "lucide-react"
import { cn } from "cn"

import { UserAvatar } from "@/components/app/user-avatar"
import { useAuth } from "@/hooks/useAuth"
import type { LineEdit, LipTranscriptItem } from "@/hooks/useLipReader"
import {
  lineSegments,
  replaceSpan,
  replaceSpanConfidence,
  splitWords,
  type LineSegment,
} from "@/lib/lipreading/wordSpans"

type PickHandler = (itemId: string, text: string, edit?: LineEdit) => void

interface SelfTranscriptProps {
  items: LipTranscriptItem[]
  ready: boolean
  inferring: boolean
  /** The sentence still being spoken (quick drafts), shown greyed after the finished ones. */
  draft?: string | null
  /** Swap or delete the unsure words of a line (click / hover their box). */
  onPick?: PickHandler
  /** Why the camera view reads badly (too far / dark / turned), if anything. */
  hint?: string | null
}

/** How close to the bottom (px) still counts as "stuck to bottom". */
const STICK_THRESHOLD = 48

/**
 * Teleprompter under the video: a tall, clean, centered readout of what *you* are saying. The newest
 * line reads large and in focus; older lines dim and dissolve toward the top edge.
 */
export function SelfTranscript({
  items,
  ready,
  inferring,
  draft,
  onPick,
  hint,
}: SelfTranscriptProps) {
  const { user } = useAuth()
  const scrollRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight
    stickToBottom.current = distance < STICK_THRESHOLD
  }

  useEffect(() => {
    if (!stickToBottom.current) return
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items, draft])

  const empty = items.length === 0 && !draft

  return (
    <div className="flex shrink-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        {user ? (
          <UserAvatar user={user} className="size-6" />
        ) : (
          <span className="flex size-6 items-center justify-center rounded-full bg-primary text-[0.625rem] font-semibold text-primary-foreground">
            Y
          </span>
        )}
        <span className="text-xs font-semibold">You</span>
        {inferring && (
          <Loader2 className="size-3 animate-spin text-muted-foreground" />
        )}
        {hint && (
          <span className="ml-auto text-xs text-amber-600 dark:text-amber-400">
            {hint}
          </span>
        )}
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="no-scrollbar flex h-44 flex-col overflow-y-auto px-5 py-4 text-center"
        style={{
          maskImage:
            "linear-gradient(to bottom, transparent, black 2rem, black 100%)",
          WebkitMaskImage:
            "linear-gradient(to bottom, transparent, black 2rem, black 100%)",
        }}
      >
        {empty ? (
          <p className="m-auto text-sm text-muted-foreground">
            {ready ? "Mouth words to the camera…" : "Warming up…"}
          </p>
        ) : (
          <div className="mt-auto space-y-1.5">
            {items.map((item, i) => {
              const latest = i === items.length - 1 && !draft
              return (
                <p
                  key={item.id}
                  data-lip-line // the app eval reads finished lines by this (ml/scripts/app_eval)
                  className={cn(
                    "leading-snug transition-colors",
                    latest
                      ? "text-lg font-medium text-foreground"
                      : "text-base text-muted-foreground"
                  )}
                >
                  <TranscriptLine item={item} onPick={onPick} />
                </p>
              )
            })}
            {draft && (
              <p className="text-lg leading-snug text-muted-foreground/60 italic">
                {draft}…
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * One finished line. Unsure words (`lineSegments`) sit in an amber dashed box; every other word
 * shows a faint box on hover. Any box: hover for a red X that deletes it, click for a dropdown
 * (under it, or above near the screen bottom) with the other readings, a field to type the right
 * words, and Delete. The dropdown is portalled with fixed positioning so the scrolling transcript
 * doesn't clip it.
 */
function TranscriptLine({
  item,
  onPick,
}: {
  item: LipTranscriptItem
  onPick?: PickHandler
}) {
  if (!onPick) return <>{item.text}</>
  const segments = lineSegments(item.text, item.wordConfidence, (item.choices ?? []).slice(1))
  // Sure words are edited one at a time; unsure spans as a whole.
  const pieces = segments.flatMap((seg) =>
    seg.flagged
      ? [seg]
      : splitWords(seg.text).map((word, k) => ({
          start: seg.start + k,
          end: seg.start + k + 1,
          text: word,
          flagged: false,
          options: [word],
        }))
  )

  const edit = (seg: LineSegment, replacement: string, typed = false) => {
    const conf = item.wordConfidence ?? splitWords(item.text).map(() => null)
    onPick(item.id, replaceSpan(item.text, seg.start, seg.end, replacement), {
      wordConfidence: replaceSpanConfidence(conf, seg.start, seg.end, replacement),
      deleted: replacement === "",
      typed,
    })
  }
  return (
    <>
      {pieces.map((seg, i) => (
        <span key={`${seg.start}-${seg.text}`}>
          {i > 0 && " "}
          <SpanBox
            segment={seg}
            onReplace={(text, typed) => edit(seg, text, typed)}
            onDelete={() => edit(seg, "")}
          />
        </span>
      ))}
    </>
  )
}

/** One editable span: dashed amber box when unsure, faint box on hover otherwise. */
function SpanBox({
  segment,
  onReplace,
  onDelete,
}: {
  segment: LineSegment
  onReplace: (text: string, typed: boolean) => void
  onDelete: () => void
}) {
  const [anchor, setAnchor] = useState<DOMRect | null>(null)
  const [typed, setTyped] = useState("")
  const boxRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const open = anchor !== null
  const options = segment.options

  useEffect(() => {
    if (!open) return
    const close = () => setAnchor(null)
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node
      if (!boxRef.current?.contains(t) && !menuRef.current?.contains(t)) close()
    }
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close()
    // The transcript auto-scrolls as lines arrive: keep the menu on its words.
    const follow = () => {
      const r = boxRef.current?.getBoundingClientRect()
      setAnchor(r && r.height > 0 ? r : null)
    }
    window.addEventListener("mousedown", onDown)
    window.addEventListener("keydown", onKey)
    window.addEventListener("scroll", follow, true)
    window.addEventListener("resize", follow)
    return () => {
      window.removeEventListener("mousedown", onDown)
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("scroll", follow, true)
      window.removeEventListener("resize", follow)
    }
  }, [open])

  const submitTyped = () => {
    const text = typed.trim()
    if (text && text !== segment.text) onReplace(text, true)
    setTyped("")
    setAnchor(null)
  }

  return (
    <span className="group/span relative inline-block">
      <button
        ref={boxRef}
        type="button"
        onClick={() => {
          setTyped("")
          setAnchor((a) => (a ? null : (boxRef.current?.getBoundingClientRect() ?? null)))
        }}
        aria-haspopup="listbox"
        aria-expanded={open}
        data-unsure={segment.flagged || undefined}
        title={segment.flagged ? "Not sure about these words: click to fix" : "Click to fix"}
        className={cn(
          "rounded-md border px-1 text-left transition-colors hover:bg-muted",
          segment.flagged
            ? "border-dashed border-amber-500/60"
            : "-mx-1 border-transparent hover:border-border",
          open && "border-solid border-foreground/40 bg-muted"
        )}
      >
        {segment.text}
      </button>
      <button
        type="button"
        onClick={onDelete}
        aria-label={`Delete "${segment.text}"`}
        title="Delete"
        className="absolute -top-1.5 -right-1.5 z-10 hidden size-4 items-center justify-center rounded-full bg-red-600 text-white shadow-sm group-hover/span:flex hover:bg-red-700"
      >
        <X className="size-2.5" strokeWidth={3} />
      </button>
      {anchor &&
        createPortal(
          <div
            ref={menuRef}
            role="listbox"
            aria-label="Fix these words"
            style={menuPosition(anchor, options.length + 2)}
            className="z-50 flex min-w-52 max-w-80 flex-col overflow-hidden rounded-xl border border-border bg-popover p-1 text-sm text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95"
          >
            {/* Only the alternative readings are actionable — the current word (index 0) did
                nothing, so it's omitted. */}
            {options.slice(1).map((text) => (
              <button
                key={text}
                type="button"
                role="option"
                onClick={() => {
                  onReplace(text, false)
                  setAnchor(null)
                }}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-muted"
              >
                <span className="flex-1">{text}</span>
              </button>
            ))}
            <form
              className="flex items-center gap-1 px-1 py-1"
              onSubmit={(e) => {
                e.preventDefault()
                submitTyped()
              }}
            >
              <input
                autoFocus
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                placeholder="Type the right words…"
                aria-label="Type the right words"
                className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-sm outline-none focus:border-foreground/40"
              />
              <button
                type="submit"
                disabled={!typed.trim()}
                aria-label="Use what I typed"
                className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40"
              >
                <Check className="size-4" />
              </button>
            </form>
            <button
              type="button"
              onClick={() => {
                onDelete()
                setAnchor(null)
              }}
              className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-red-600 transition-colors hover:bg-red-600/10 dark:text-red-400"
            >
              <span className="flex size-4 items-center justify-center rounded-full bg-red-600 text-white">
                <X className="size-2.5" strokeWidth={3} />
              </span>
              Delete
            </button>
          </div>,
          document.body
        )}
    </span>
  )
}

const MENU_WIDTH = 320
const OPTION_HEIGHT = 36

/**
 * Under the line when it fits, else above it (the transcript sits near the bottom of the screen);
 * kept inside the window horizontally.
 */
function menuPosition(anchor: DOMRect, options: number): React.CSSProperties {
  const height = options * OPTION_HEIGHT + 8
  const left = Math.max(8, Math.min(anchor.left, window.innerWidth - MENU_WIDTH - 8))
  return anchor.bottom + 4 + height <= window.innerHeight
    ? { position: "fixed", left, top: anchor.bottom + 4 }
    : { position: "fixed", left, bottom: window.innerHeight - anchor.top + 4 }
}
