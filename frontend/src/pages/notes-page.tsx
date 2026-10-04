import { useMemo, useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { ArrowDownAZ, ArrowLeft, Clock, Loader2, NotebookPen, Search } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ParticipantAvatars } from "@/components/app/participant-avatars"
import { useConversations } from "@/hooks/useConversations"
import { formatDate } from "@/lib/conversations/format"
import { findParticipant, type Conversation } from "@/lib/conversations/types"

type SortMode = "date" | "alpha"

/** /notes — searchable, sortable index of past conversations. */
export function NotesPage() {
  const [query, setQuery] = useState("")
  const [sort, setSort] = useState<SortMode>("date")
  const { conversations, loading } = useConversations(query)
  const navigate = useNavigate()

  const sorted = useMemo(() => {
    const list = [...conversations]
    if (sort === "alpha") list.sort((a, b) => a.title.localeCompare(b.title))
    else list.sort((a, b) => b.startedAt - a.startedAt)
    return list
  }, [conversations, sort])

  return (
    <div className="no-scrollbar h-svh overflow-y-auto">
      <div className="mx-auto max-w-4xl px-5 py-6">
        <Button
          size="sm"
          variant="outline"
          onClick={() => navigate("/app")}
          className="mb-5"
        >
          <ArrowLeft />
          Back
        </Button>

        <div className="mb-5">
          <h1 className="text-lg font-semibold">Conversations</h1>
          <p className="text-xs text-muted-foreground">
            Your past conversations, timestamped and searchable.
          </p>
        </div>

        <div className="mb-5 flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations, people, or what was said…"
              className="h-9 pl-8"
            />
          </div>
          <SortToggle sort={sort} onChange={setSort} />
        </div>

        {loading ? (
          <div className="flex h-40 items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading…
          </div>
        ) : sorted.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-3 text-center">
            <span className="flex size-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <NotebookPen className="size-5" />
            </span>
            <p className="text-sm text-muted-foreground">
              {query ? `No conversations match “${query}”.` : "No conversations yet."}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {sorted.map((c) => (
              <ConversationCard key={c.id} conversation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function SortToggle({
  sort,
  onChange,
}: {
  sort: SortMode
  onChange: (s: SortMode) => void
}) {
  return (
    <div className="flex shrink-0 items-center gap-0.5 rounded-lg border border-border bg-muted/40 p-0.5">
      <Button
        size="sm"
        variant={sort === "date" ? "default" : "ghost"}
        onClick={() => onChange("date")}
        className={cn(sort !== "date" && "text-muted-foreground")}
      >
        <Clock />
        Recent
      </Button>
      <Button
        size="sm"
        variant={sort === "alpha" ? "default" : "ghost"}
        onClick={() => onChange("alpha")}
        className={cn(sort !== "alpha" && "text-muted-foreground")}
      >
        <ArrowDownAZ />
        A–Z
      </Button>
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
      className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-sm transition-colors hover:border-foreground/20 hover:bg-muted/30"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold">{conversation.title}</h2>
          <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
            {formatDate(conversation.startedAt)}
          </p>
        </div>
        <ParticipantAvatars participants={conversation.participants} max={3} />
      </div>

      {first && (
        <p className="line-clamp-2 text-sm text-muted-foreground">
          {firstName && <span className="font-medium">{firstName}: </span>}
          {first.text}
        </p>
      )}
    </Link>
  )
}
