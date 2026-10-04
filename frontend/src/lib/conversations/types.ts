/** A person in a conversation (incl. "You"). */
export interface Participant {
  readonly id: string
  /** Display name — editable (renaming a speaker updates it). */
  name: string
  /** CSS color (hex or var) for the avatar/accent. */
  readonly colorVar: string
}

/** One finalized message in a conversation, created as soon as the utterance completes. */
export interface NoteEntry {
  readonly id: string
  readonly speakerId: string
  readonly text: string
  /** Wall-clock time (epoch ms) the message was recorded. */
  readonly at: number
}

/** A recorded conversation: participants + timestamped messages. */
export interface Conversation {
  readonly id: string
  /** Default name (date-based); editable. */
  title: string
  /** Epoch ms when the conversation started. */
  readonly startedAt: number
  participants: Participant[]
  entries: NoteEntry[]
}

/** Convenience: look up a participant by id. */
export function findParticipant(
  conversation: Conversation,
  speakerId: string
): Participant | undefined {
  return conversation.participants.find((p) => p.id === speakerId)
}
