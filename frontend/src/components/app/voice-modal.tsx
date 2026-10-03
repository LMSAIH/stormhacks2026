import { useEffect, useMemo, useState } from "react"
import { Check, Loader2, Search, X } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar } from "@/components/app/avatar"
import { colorForString } from "@/lib/palette"
import type { Voice } from "@/lib/voices/types"

interface VoiceModalProps {
  open: boolean
  onClose: () => void
  voices: Voice[]
  loading: boolean
  selectedId: string | null
  onSelect: (id: string) => void
}

const cap = (s: string) =>
  s ? s.charAt(0).toUpperCase() + s.slice(1) : s

/** Beautiful, searchable voice picker modal. */
export function VoiceModal({
  open,
  onClose,
  voices,
  loading,
  selectedId,
  onSelect,
}: VoiceModalProps) {
  const [query, setQuery] = useState("")
  const [pending, setPending] = useState<string | null>(selectedId)

  // Sync pending selection + reset search each time the modal opens.
  useEffect(() => {
    if (open) {
      setPending(selectedId)
      setQuery("")
    }
  }, [open, selectedId])

  // Escape to close + lock background scroll while open.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      window.removeEventListener("keydown", onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return voices
    return voices.filter((v) =>
      [
        v.name,
        v.tagline,
        v.description,
        v.gender,
        v.age,
        v.accent,
        v.useCase,
        v.descriptive,
      ]
        .join(" ")
        .toLowerCase()
        .includes(q)
    )
  }, [voices, query])

  if (!open) return null

  const confirm = () => {
    if (pending) onSelect(pending)
    onClose()
  }

  const pendingVoice = voices.find((v) => v.id === pending)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm animate-in fade-in-0"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative flex max-h-[82vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl animate-in fade-in-0 zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            <h2 className="text-base font-semibold">Choose a voice</h2>
            <p className="text-xs text-muted-foreground">
              Pick the voice that reads your conversation aloud.
            </p>
          </div>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={onClose}
            aria-label="Close"
          >
            <X />
          </Button>
        </div>

        {/* Search */}
        <div className="border-b border-border px-5 py-3">
          <div className="relative">
            <Search className="absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name, accent, vibe…"
              className="h-8 pl-8"
            />
          </div>
        </div>

        {/* List */}
        <div className="no-scrollbar flex-1 overflow-y-auto px-5 py-4">
          {loading ? (
            <div className="flex h-40 items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading voices…
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              No voices match “{query}”.
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              {filtered.map((voice) => (
                <VoiceCard
                  key={voice.id}
                  voice={voice}
                  selected={pending === voice.id}
                  onClick={() => setPending(voice.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between gap-3 border-t border-border px-5 py-3">
          <span className="truncate text-xs text-muted-foreground">
            {pendingVoice ? (
              <>
                Selected <span className="font-medium text-foreground">{pendingVoice.name}</span>
              </>
            ) : (
              "No voice selected"
            )}
          </span>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button size="sm" onClick={confirm} disabled={!pending}>
              Use this voice
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function VoiceCard({
  voice,
  selected,
  onClick,
}: {
  voice: Voice
  selected: boolean
  onClick: () => void
}) {
  const chips = [voice.gender, voice.age, voice.accent, voice.useCase].filter(
    Boolean
  )

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group relative flex items-start gap-3 rounded-xl border p-3 text-left transition-colors",
        selected
          ? "border-primary bg-primary/5 ring-2 ring-primary/25"
          : "border-border hover:border-foreground/20 hover:bg-muted/40"
      )}
    >
      <Avatar label={voice.name} color={colorForString(voice.id)} />

      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-baseline gap-2 truncate">
            <span className="text-sm font-semibold">{voice.name}</span>
            {voice.descriptive && (
              <span className="truncate text-[0.625rem] tracking-wide text-muted-foreground capitalize">
                {voice.descriptive}
              </span>
            )}
          </span>
          <span
            className={cn(
              "flex size-4 shrink-0 items-center justify-center rounded-full transition-colors",
              selected
                ? "bg-primary text-primary-foreground"
                : "border border-border text-transparent group-hover:border-foreground/30"
            )}
          >
            <Check className="size-2.5" />
          </span>
        </div>

        {(voice.description || voice.tagline) && (
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
            {voice.description || voice.tagline}
          </p>
        )}

        {chips.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {chips.map((c, i) => (
              <span
                key={i}
                className="rounded-md bg-muted px-1.5 py-0.5 text-[0.625rem] font-medium text-muted-foreground"
              >
                {cap(c)}
              </span>
            ))}
          </div>
        )}
      </div>
    </button>
  )
}
