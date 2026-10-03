import { useEffect, useRef } from "react"
import { Eraser } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"
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
  engineName: string
  onRename: (id: string, name: string) => void
  onClear: () => void
}

/**
 * Single always-on conversation: diarized speech from people nearby (left) plus
 * the user's own lip-read utterances as "You" (right), merged in time order.
 */
export function ConversationFeed({
  messages,
  speakers,
  engineName,
  onRename,
  onClear,
}: ConversationFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    })
  }, [messages])

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium">
            <span className="size-1.5 animate-pulse rounded-full bg-emerald-500" />
            Listening
          </span>
          <span className="text-[0.625rem] text-muted-foreground">{engineName}</span>
        </div>
        <Button
          size="xs"
          variant="ghost"
          onClick={onClear}
          disabled={messages.length === 0}
        >
          <Eraser />
          Clear
        </Button>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Listening for speech, and reading your lips…
          </p>
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
    <div className="flex flex-col items-start gap-0.5">
      <div className="flex items-center gap-2">
        <span
          className="size-2 shrink-0 rounded-full"
          style={{ backgroundColor: speaker?.colorVar }}
        />
        {speaker && <SpeakerName speaker={speaker} onRename={onRename} />}
      </div>
      <p
        className={cn(
          "ml-4 max-w-[80%] rounded-lg bg-muted/60 px-2.5 py-1.5 text-sm leading-relaxed",
          !final && "opacity-65"
        )}
      >
        {text}
        {!final && <span className="animate-pulse">▍</span>}
      </p>
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
