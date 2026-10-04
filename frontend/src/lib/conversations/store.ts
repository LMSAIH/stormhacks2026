import type { Conversation } from "./types"

/**
 * Pure helpers for building and filtering conversations. Persistence lives in `./api.ts`
 * (backed by `/api/chats` → Timescale); these are the local, side-effect-free utilities the
 * recorder and search use.
 */

/** Default title involving the date/time; editable later. */
export function defaultTitle(at: number): string {
  return `Conversation · ${new Date(at).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })}`
}

/** A fresh, empty conversation that already knows about "You". */
export function newConversation(at: number): Conversation {
  return {
    id: `conv-${at}`,
    title: defaultTitle(at),
    startedAt: at,
    participants: [{ id: "you", name: "You", colorVar: "var(--primary)" }],
    entries: [],
  }
}

/** Case-insensitive match over title, participant names, and message text. */
export function conversationMatches(conv: Conversation, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return [
    conv.title,
    ...conv.participants.map((p) => p.name),
    ...conv.entries.map((e) => e.text),
  ]
    .join(" ")
    .toLowerCase()
    .includes(q)
}
