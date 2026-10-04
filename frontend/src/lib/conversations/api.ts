import { SAMPLE_CONVERSATIONS } from "./data"
import type { Conversation } from "./types"

/**
 * Pluggable conversations data access. Today this reads local sample data; the
 * UI depends only on these functions, so pointing them at the backend later is
 * a one-file change.
 */

const byNewest = (a: Conversation, b: Conversation) => b.startedAt - a.startedAt

export async function listConversations(): Promise<Conversation[]> {
  await delay()
  return [...SAMPLE_CONVERSATIONS].sort(byNewest)
}

export async function getConversation(id: string): Promise<Conversation | null> {
  await delay()
  return SAMPLE_CONVERSATIONS.find((c) => c.id === id) ?? null
}

/**
 * Search conversations by a text query.
 *
 * TODO: replace this simple client-side substring match with a backend call
 * (vector/semantic search): `await fetch("/api/notes/search?q=" + ...)`.
 * Keeping the same signature means the UI won't change when we swap it.
 */
export async function searchConversations(query: string): Promise<Conversation[]> {
  await delay()
  const q = query.trim().toLowerCase()
  const all = [...SAMPLE_CONVERSATIONS].sort(byNewest)
  if (!q) return all

  return all.filter((c) => {
    const haystack = [
      c.title,
      ...c.participants.map((p) => p.name),
      ...c.entries.map((e) => e.text),
    ]
      .join(" ")
      .toLowerCase()
    return haystack.includes(q)
  })
}

function delay(ms = 150): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}
