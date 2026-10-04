/** A person in a conversation (incl. "You"). */
export interface Participant {
  readonly id: string
  readonly name: string
  /** CSS color (hex or var) for the avatar/accent. */
  readonly colorVar: string
}

/** One finalized sentence in a conversation. */
export interface NoteEntry {
  readonly id: string
  readonly speakerId: string
  readonly text: string
  /** Wall-clock time (epoch ms) the sentence began. */
  readonly at: number
}

/** A past conversation: participants + timestamped sentences. */
export interface Conversation {
  readonly id: string
  readonly title: string
  /** Epoch ms when the conversation started. */
  readonly startedAt: number
  readonly participants: readonly Participant[]
  readonly entries: readonly NoteEntry[]
}

/** Convenience: look up a participant by id. */
export function findParticipant(
  conversation: Conversation,
  speakerId: string
): Participant | undefined {
  return conversation.participants.find((p) => p.id === speakerId)
}
