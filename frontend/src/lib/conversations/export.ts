import { formatDate, formatDateTime } from "./format"
import { findParticipant, type Conversation } from "./types"

/** Export a conversation as a downloadable, timestamp-sorted plain-text file. */
export function exportConversation(conversation: Conversation): void {
  const sorted = [...conversation.entries].sort((a, b) => a.at - b.at)

  const header = [
    conversation.title,
    formatDate(conversation.startedAt),
    `Participants: ${conversation.participants.map((p) => p.name).join(", ")}`,
    "",
  ].join("\n")

  const body = sorted
    .map((e) => {
      const name = findParticipant(conversation, e.speakerId)?.name ?? "Unknown"
      return `[${formatDateTime(e.at)}] ${name}: ${e.text}`
    })
    .join("\n")

  const blob = new Blob([header + body + "\n"], {
    type: "text/plain;charset=utf-8",
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `${conversation.id}-notes.txt`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
