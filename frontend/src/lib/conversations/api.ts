import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/backend/client"
import type { Conversation, NoteEntry, Participant } from "./types"

/**
 * Backend-backed conversation access (`/api/chats`), scoped to the signed-in user by the
 * session cookie. Conversations live in Timescale/Postgres; this module maps between the
 * frontend {@link Conversation} shape and the server's chat payload. The UI depends only on
 * these functions, so the storage layer is swappable here alone.
 */

interface ChatSpeaker {
  id: string
  name: string
  color?: string | null
}

interface ChatMessage {
  id?: string | null
  speaker_id: string
  text: string
  at?: number | null
}

interface ChatPayload {
  title: string
  speakers: ChatSpeaker[]
  messages: ChatMessage[]
}

interface ChatRecord extends ChatPayload {
  id: string
  created_at: string
  updated_at?: string
}

interface CreatedChat {
  id: string
  title: string
  created_at: string
}

/** Frontend conversation → server payload. */
export function toPayload(conv: Conversation): ChatPayload {
  return {
    title: conv.title,
    speakers: conv.participants.map((p) => ({
      id: p.id,
      name: p.name,
      color: p.colorVar,
    })),
    messages: conv.entries.map((e) => ({
      id: e.id,
      speaker_id: e.speakerId,
      text: e.text,
      at: e.at,
    })),
  }
}

/** Server record → frontend conversation. */
function toConversation(record: ChatRecord): Conversation {
  const startedAt = Date.parse(record.created_at) || Date.now()
  const participants: Participant[] = record.speakers.map((s) => ({
    id: s.id,
    name: s.name,
    colorVar: s.color ?? "var(--muted-foreground)",
  }))
  const entries: NoteEntry[] = record.messages.map((m, i) => ({
    id: m.id ?? `${record.id}-${i}`,
    speakerId: m.speaker_id,
    text: m.text,
    at: m.at ?? startedAt,
  }))
  return { id: record.id, title: record.title, startedAt, participants, entries }
}

/** All of the signed-in user's conversations, newest first. */
export async function listConversations(): Promise<Conversation[]> {
  const { chats } = await apiGet<{ chats: ChatRecord[] }>("/api/chats")
  return chats.map(toConversation)
}

export async function getConversation(id: string): Promise<Conversation | null> {
  return apiGet<ChatRecord>(`/api/chats/${id}`).then(toConversation)
}

/** Create a conversation; returns the server-assigned id and start time. */
export async function createConversation(
  conv: Conversation
): Promise<{ id: string; startedAt: number }> {
  const created = await apiPost<CreatedChat>("/api/chats", toPayload(conv))
  return { id: created.id, startedAt: Date.parse(created.created_at) || conv.startedAt }
}

/** Overwrite an existing conversation (messages, participants, title). */
export async function saveConversation(id: string, conv: Conversation): Promise<void> {
  await apiPut<{ saved: boolean }>(`/api/chats/${id}`, toPayload(conv))
}

export async function deleteConversation(id: string): Promise<void> {
  await apiDelete<{ deleted: boolean }>(`/api/chats/${id}`)
}
