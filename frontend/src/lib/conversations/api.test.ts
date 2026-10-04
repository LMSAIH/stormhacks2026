import { beforeEach, describe, expect, it, vi } from "vitest"

import { apiGet, apiPost, apiPut } from "@/lib/backend/client"
import {
  createConversation,
  searchConversations,
  updateConversation,
  type ConversationPayload,
} from "./api"

vi.mock("@/lib/backend/client", () => ({
  ApiError: class ApiError extends Error {
    status: number
    constructor(status: number, message: string) {
      super(message)
      this.status = status
    }
  },
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiPut: vi.fn(),
}))

const payload: ConversationPayload = {
  speakers: [{ id: "speaker-1", name: "Morgan" }],
  messages: [
    { speaker_id: "speaker-1", text: "Meeting notes", at: 1791028800000 },
  ],
}

beforeEach(() => vi.clearAllMocks())

describe("conversation backend API", () => {
  it("loads and searches backend chat details", async () => {
    vi.mocked(apiGet)
      .mockResolvedValueOnce({
        chats: [
          {
            id: "chat-1",
            title: "Meeting notes",
            created_at: "2026-10-03T12:00:00Z",
          },
        ],
      } as never)
      .mockResolvedValueOnce({
        id: "chat-1",
        title: "Meeting notes",
        created_at: "2026-10-03T12:00:00Z",
        speakers: [{ id: "speaker-1", name: "Morgan" }],
        messages: [
          {
            speaker_id: "speaker-1",
            text: "Discuss launch",
            at: 1791028800000,
          },
        ],
      } as never)

    const conversations = await searchConversations("launch")

    expect(apiGet).toHaveBeenNthCalledWith(1, "/api/chats")
    expect(apiGet).toHaveBeenNthCalledWith(2, "/api/chats/chat-1")
    expect(conversations[0]).toMatchObject({
      id: "chat-1",
      title: "Meeting notes",
      participants: [{ id: "speaker-1", name: "Morgan" }],
      entries: [
        { speakerId: "speaker-1", text: "Discuss launch", at: 1791028800000 },
      ],
    })
  })

  it("creates and updates backend conversations", async () => {
    vi.mocked(apiPost).mockResolvedValueOnce({ id: "chat-1" } as never)
    vi.mocked(apiPut).mockResolvedValueOnce({ saved: true } as never)

    await createConversation(payload)
    await updateConversation("chat-1", payload)

    expect(apiPost).toHaveBeenCalledWith("/api/chats", payload)
    expect(apiPut).toHaveBeenCalledWith("/api/chats/chat-1", payload)
  })
})
