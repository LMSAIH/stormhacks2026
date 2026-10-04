import { useEffect, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { ArrowLeft, Download, Loader2, Pencil, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { CopyButton } from "@/components/app/copy-button"
import { Input } from "@/components/ui/input"
import { Avatar } from "@/components/app/avatar"
import { UserAvatar } from "@/components/app/user-avatar"
import { useAuth } from "@/hooks/useAuth"
import { useConversation } from "@/hooks/useConversation"
import { exportConversation } from "@/lib/conversations/export"
import { formatDate, formatTime } from "@/lib/conversations/format"
import { findParticipant, type Conversation } from "@/lib/conversations/types"

/** /notes/:id — detail of one recorded conversation (editable title). */
export function NoteDetailPage() {
  const { id } = useParams()
  const { conversation, loading, rename, remove } = useConversation(id)
  const { user } = useAuth()
  const navigate = useNavigate()
  const [confirmOpen, setConfirmOpen] = useState(false)

  const handleDelete = async () => {
    await remove()
    navigate("/notes")
  }

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
            <ConversationHeader
              conversation={conversation}
              onRename={rename}
              onDelete={() => setConfirmOpen(true)}
            />
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
                  {entry.speakerId === "you" && user ? (
                    <UserAvatar user={user} className="mt-0.5 size-8" />
                  ) : (
                    <Avatar
                      label={p?.name ?? "?"}
                      color={p?.colorVar ?? "var(--muted-foreground)"}
                      className="mt-0.5 size-8 text-[0.6875rem]"
                    />
                  )}
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

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Delete conversation?"
        description={
          conversation
            ? `“${conversation.title}” will be permanently deleted. This can't be undone.`
            : undefined
        }
        onConfirm={handleDelete}
      />
    </div>
  )
}

function ConversationHeader({
  conversation,
  onRename,
  onDelete,
}: {
  conversation: Conversation
  onRename: (title: string) => void
  onDelete: () => void
}) {
  return (
    <div className="flex flex-1 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        <EditableTitle title={conversation.title} onRename={onRename} />
        <p className="mt-0.5 text-xs text-muted-foreground">
          {formatDate(conversation.startedAt)} ·{" "}
          {conversation.participants.map((p) => p.name).join(", ")}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <CopyButton
          size="sm"
          variant="outline"
          label="Copy"
          title="Copy transcript"
          getText={() =>
            conversation.entries
              .map((e) => {
                const who = findParticipant(conversation, e.speakerId)?.name ?? "Speaker"
                return `${who}: ${e.text}`
              })
              .join("\n")
          }
        />
        <Button
          size="sm"
          variant="outline"
          onClick={() => exportConversation(conversation)}
        >
          <Download />
          Export
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={onDelete}
          className="text-muted-foreground hover:border-destructive/40 hover:text-destructive"
        >
          <Trash2 />
          Delete
        </Button>
      </div>
    </div>
  )
}

/** Click the title (or the pencil) to rename the conversation. */
function EditableTitle({
  title,
  onRename,
}: {
  title: string
  onRename: (title: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(title)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) inputRef.current?.select()
  }, [editing])

  const commit = () => {
    const next = draft.trim()
    if (next) onRename(next)
    else setDraft(title)
    setEditing(false)
  }

  if (editing) {
    return (
      <Input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit()
          if (e.key === "Escape") {
            setDraft(title)
            setEditing(false)
          }
        }}
        className="h-8 max-w-sm text-lg font-semibold"
      />
    )
  }

  return (
    <button
      type="button"
      onClick={() => {
        setDraft(title)
        setEditing(true)
      }}
      className="group flex max-w-full items-center gap-2 text-left"
      title="Click to rename"
    >
      <span className="truncate text-lg font-semibold">{title}</span>
      <Pencil className="size-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
    </button>
  )
}
