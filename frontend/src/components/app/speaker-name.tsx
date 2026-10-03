import { useEffect, useRef, useState } from "react"

import { Input } from "@/components/ui/input"
import type { Speaker } from "@/lib/listening/types"

interface SpeakerNameProps {
  speaker: Speaker
  onRename: (id: string, name: string) => void
}

/** Click-to-edit speaker label used in the diarized chat. */
export function SpeakerName({ speaker, onRename }: SpeakerNameProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(speaker.name)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) inputRef.current?.select()
  }, [editing])

  const commit = () => {
    const name = draft.trim()
    if (name) onRename(speaker.id, name)
    else setDraft(speaker.name)
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
            setDraft(speaker.name)
            setEditing(false)
          }
        }}
        className="h-5 w-28 px-1 text-xs font-medium"
      />
    )
  }

  return (
    <button
      type="button"
      onClick={() => {
        setDraft(speaker.name)
        setEditing(true)
      }}
      className="rounded px-1 text-xs font-semibold transition-colors hover:bg-muted"
      style={{ color: speaker.colorVar }}
      title="Click to rename"
    >
      {speaker.name}
    </button>
  )
}
