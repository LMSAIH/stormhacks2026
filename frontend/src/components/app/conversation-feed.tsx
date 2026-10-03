import { useEffect, useRef, useState } from "react"
import { MessagesSquare } from "lucide-react"
import { cn } from "cn"

import { Avatar } from "@/components/app/avatar"
import { SpeakerName } from "@/components/app/speaker-name"
import type { Speaker } from "@/lib/listening/types"

export interface FeedMessage {
  id: string
  speakerId: string
  text: string
  final: boolean
  at: number
  /** True for the local user's own lip-read speech. */
  isSelf: boolean
}

interface ConversationFeedProps {
  messages: FeedMessage[]
  /** Detected (non-self) speakers, keyed by id — used for editable names. */
  speakers: Record<string, Speaker>
  onRename: (id: string, name: string) => void
}

/** How close to the bottom (px) still counts as "stuck to bottom". */
const STICK_THRESHOLD = 80

/**
 * Single always-on conversation: diarized speech from people nearby (left) plus
 * the user's own lip-read utterances as "You" (right), merged in time order.
 */
export function ConversationFeed({
  messages,
  speakers,
  onRename,
}: ConversationFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  // Only auto-scroll on new messages if the user is already near the bottom.
  const stickToBottom = useRef(true)
  // Drives the bottom blur hint — shown when there's content below the fold.
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
  }, [messages])

  return (
    <div className="relative flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="no-scrollbar flex-1 space-y-3 overflow-y-auto px-3 py-3"
      >
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <span className="flex size-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <MessagesSquare className="size-5" />
            </span>
            <p className="max-w-[16rem] text-sm text-muted-foreground">
              Waiting for the room. Nearby speech will appear here, each person in
              their own color.
            </p>
          </div>
        ) : (
          messages.map((m) =>
            m.isSelf ? (
              <SelfMessage key={m.id} text={m.text} final={m.final} />
            ) : (
              <OtherMessage
                key={m.id}
                text={m.text}
                final={m.final}
                speaker={speakers[m.speakerId]}
                onRename={onRename}
              />
            )
          )
        )}
      </div>

      {/* Blur hint: fades in at the bottom while content remains below. */}
      <div
        aria-hidden
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-card/90 to-transparent backdrop-blur-[2px] transition-opacity duration-200",
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

function OtherMessage({
  text,
  final,
  speaker,
  onRename,
}: {
  text: string
  final: boolean
  speaker: Speaker | undefined
  onRename: (id: string, name: string) => void
}) {
  return (
    <div className="flex items-start gap-2.5">
      <Avatar
        label={speaker?.name ?? "?"}
        color={speaker?.colorVar ?? "var(--muted-foreground)"}
        className="mt-0.5 size-7 text-[0.6875rem]"
      />
      <div className="min-w-0">
        {speaker && (
          <div className="mb-0.5">
            <SpeakerName speaker={speaker} onRename={onRename} />
          </div>
        )}
        <p
          className={cn(
            "w-fit max-w-full rounded-lg rounded-tl-sm bg-muted/60 px-2.5 py-1.5 text-sm leading-relaxed",
            !final && "opacity-65"
          )}
        >
          {text}
          {!final && <span className="animate-pulse">▍</span>}
        </p>
      </div>
    </div>
  )
}

function SelfMessage({ text, final }: { text: string; final: boolean }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="px-1 text-xs font-semibold text-primary">You</span>
      <p
        className={cn(
          "max-w-[80%] rounded-lg bg-primary/10 px-2.5 py-1.5 text-sm leading-relaxed text-foreground",
          !final && "opacity-65"
        )}
      >
        {text}
      </p>
    </div>
  )
}
