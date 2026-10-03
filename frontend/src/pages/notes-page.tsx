import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { ArrowLeft, Loader2, NotebookPen, Search } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ParticipantAvatars } from "@/components/app/participant-avatars"
import { useConversations } from "@/hooks/useConversations"
import { formatDate } from "@/lib/conversations/format"
import { findParticipant, type Conversation } from "@/lib/conversations/types"

/** /notes — searchable index of past conversations. */
export function NotesPage() {
  const [query, setQuery] = useState("")
  const { conversations, loading } = useConversations(query)
  const navigate = useNavigate()

  return (
    <div className="no-scrollbar h-svh overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-8">
        <header className="mb-6 flex items-center gap-3">
          <Button
            size="icon"
            variant="ghost"
            onClick={() => navigate("/app")}
            aria-label="Back to app"
          >
            <ArrowLeft />
          </Button>
          <div>
            <h1 className="text-lg font-semibold">Conversations</h1>
            <p className="text-xs text-muted-foreground">
              Your past conversations, timestamped and searchable.
            </p>
          </div>
        </header>

        <div className="relative mb-4">
          <Search className="absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations, people, or what was said…"
            className="h-9 pl-8"
          />
        </div>

        {loading ? (
          <div className="flex h-40 items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading…
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-3 text-center">
            <span className="flex size-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <NotebookPen className="size-5" />
            </span>
            <p className="text-sm text-muted-foreground">
              {query ? `No conversations match “${query}”.` : "No conversations yet."}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {conversations.map((c) => (
              <ConversationCard key={c.id} conversation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ConversationCard({ conversation }: { conversation: Conversation }) {
  const first = conversation.entries[0]
  const firstName = first
    ? findParticipant(conversation, first.speakerId)?.name
    : undefined

  return (
    <Link
      to={`/notes/${conversation.id}`}
      className="block rounded-xl border border-border bg-card p-4 shadow-sm transition-colors hover:border-foreground/20 hover:bg-muted/30"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold">{conversation.title}</h2>
          <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
            {formatDate(conversation.startedAt)} · {conversation.entries.length}{" "}
            {conversation.entries.length === 1 ? "sentence" : "sentences"}
          </p>
        </div>
        <ParticipantAvatars participants={conversation.participants} />
      </div>

      {first && (
        <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">
          {firstName && <span className="font-medium">{firstName}: </span>}
          {first.text}
        </p>
      )}
    </Link>
  )
}
