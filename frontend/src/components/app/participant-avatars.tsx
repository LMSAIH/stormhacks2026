import { Avatar } from "@/components/app/avatar"
import type { Participant } from "@/lib/conversations/types"

interface ParticipantAvatarsProps {
  participants: readonly Participant[]
  max?: number
}

/** Overlapping avatar stack for a conversation's participants. */
export function ParticipantAvatars({
  participants,
  max = 4,
}: ParticipantAvatarsProps) {
  const shown = participants.slice(0, max)
  const extra = participants.length - shown.length

  return (
    <div className="flex -space-x-2">
      {shown.map((p) => (
        <Avatar
          key={p.id}
          label={p.name}
          color={p.colorVar}
          className="size-7 text-[0.625rem] ring-2 ring-card"
        />
      ))}
      {extra > 0 && (
        <span className="flex size-7 items-center justify-center rounded-full bg-muted text-[0.625rem] font-semibold text-muted-foreground ring-2 ring-card">
          +{extra}
        </span>
      )}
    </div>
  )
}
