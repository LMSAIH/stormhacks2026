import { ApiError, apiGet, apiPost, apiPut } from "@/lib/backend/client"
import { colorForString } from "@/lib/palette"
import type { Conversation } from "./types"

interface ChatSummary {
  id: string
  title: string
  created_at: string
}

interface BackendChat extends ChatSummary {
  speakers: { id: string; name: string }[]
  messages: { speaker_id: string; text: string; at?: number }[]
}

export interface ConversationPayload {
  speakers: { id: string; name: string }[]
  messages: { speaker_id: string; text: string; at: number }[]
}

const byNewest = (a: Conversation, b: Conversation) => b.startedAt - a.startedAt

export async function listConversations(): Promise<Conversation[]> {
  const { chats } = await apiGet<{ chats: ChatSummary[] }>("/api/chats")
  const details = await Promise.all(
    chats.map(({ id }) =>
      apiGet<BackendChat>(`/api/chats/${encodeURIComponent(id)}`)
    )
  )
  return details.map(toConversation).sort(byNewest)
}

export async function getConversation(
  id: string
): Promise<Conversation | null> {
  try {
    const chat = await apiGet<BackendChat>(
      `/api/chats/${encodeURIComponent(id)}`
    )
    return toConversation(chat)
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null
    throw error
  }
}

export async function searchConversations(
  query: string
): Promise<Conversation[]> {
  const q = query.trim().toLowerCase()
  const all = await listConversations()
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

export function createConversation(
  payload: ConversationPayload
): Promise<{ id: string; title: string; created_at: string }> {
  return apiPost("/api/chats", payload)
}

export function updateConversation(
  id: string,
  payload: ConversationPayload
): Promise<{ saved: true }> {
  return apiPut(`/api/chats/${encodeURIComponent(id)}`, payload)
}

function toConversation(chat: BackendChat): Conversation {
  const parsedStart = Date.parse(chat.created_at)
  const startedAt = Number.isFinite(parsedStart) ? parsedStart : Date.now()

  return {
    id: chat.id,
    title: chat.title,
    startedAt,
    participants: chat.speakers.map((speaker) => ({
      ...speaker,
      colorVar: colorForString(speaker.id),
    })),
    entries: chat.messages.map((message, index) => ({
      id: `${chat.id}:${index}`,
      speakerId: message.speaker_id,
      text: message.text,
      at: message.at ?? startedAt + index,
    })),
  }
}
