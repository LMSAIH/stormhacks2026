import { Link, useNavigate, useParams } from "react-router-dom"
import { ArrowLeft, Download, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Avatar } from "@/components/app/avatar"
import { useConversation } from "@/hooks/useConversation"
import { exportConversation } from "@/lib/conversations/export"
import { formatDate, formatTime } from "@/lib/conversations/format"
import { findParticipant, type Conversation } from "@/lib/conversations/types"

/** /notes/:id — read-only detail of one past conversation. */
export function NoteDetailPage() {
  const { id } = useParams()
  const { conversation, loading } = useConversation(id)
  const navigate = useNavigate()

  return (
    <div className="no-scrollbar h-svh overflow-y-auto">
      <div className="mx-auto max-w-4xl px-5 py-6">
        <Button
          size="sm"
          variant="outline"
          onClick={() => navigate("/notes")}
          className="mb-5"
        >
          <ArrowLeft />
          Back
        </Button>

        <header className="mb-6">
          {loading ? (
            <div className="flex h-9 items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading…
            </div>
          ) : conversation ? (
            <ConversationHeader conversation={conversation} />
          ) : (
            <div>
              <h1 className="text-lg font-semibold">Conversation not found</h1>
              <Link to="/notes" className="text-xs text-primary underline-offset-4 hover:underline">
                Back to conversations
              </Link>
            </div>
          )}
        </header>

        {conversation && (
          <ol className="flex flex-col gap-4">
            {conversation.entries.map((entry) => {
              const p = findParticipant(conversation, entry.speakerId)
              return (
                <li key={entry.id} className="flex items-start gap-3">
                  <Avatar
                    label={p?.name ?? "?"}
                    color={p?.colorVar ?? "var(--muted-foreground)"}
                    className="mt-0.5 size-8 text-[0.6875rem]"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span
                        className="text-sm font-semibold"
                        style={{ color: p?.colorVar }}
                      >
                        {p?.name ?? "Unknown"}
                      </span>
                      <time
                        className="text-[0.625rem] text-muted-foreground tabular-nums"
                        dateTime={new Date(entry.at).toISOString()}
                      >
                        {formatTime(entry.at)}
                      </time>
                    </div>
                    <p className="mt-0.5 text-sm leading-relaxed">{entry.text}</p>
                  </div>
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </div>
  )
}

function ConversationHeader({ conversation }: { conversation: Conversation }) {
  return (
    <div className="flex flex-1 items-start justify-between gap-3">
      <div className="min-w-0">
        <h1 className="truncate text-lg font-semibold">{conversation.title}</h1>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {formatDate(conversation.startedAt)} ·{" "}
          {conversation.participants.map((p) => p.name).join(", ")}
        </p>
      </div>
      <Button
        size="sm"
        variant="outline"
        onClick={() => exportConversation(conversation)}
      >
        <Download />
        Export
      </Button>
    </div>
  )
}
