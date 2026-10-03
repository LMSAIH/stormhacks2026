import { useState } from "react"
import { AudioLines } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/app/theme-toggle"
import { VoiceModal } from "@/components/app/voice-modal"
import { Avatar } from "@/components/app/avatar"
import { ActionsMenu } from "@/components/app/actions-menu"
import { colorForString } from "@/lib/palette"
import { useVoices } from "@/hooks/useVoices"

interface OptionsBoxProps {
  fps: number
}

/** Actions bar: menu (navigation), voice selection, fps, theme. */
export function OptionsBox({ fps }: OptionsBoxProps) {
  const { voices, loading } = useVoices()
  const [open, setOpen] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const selected = voices.find((v) => v.id === selectedId)

  return (
    <div className="flex w-full items-center justify-between gap-2 rounded-xl border border-border bg-card px-3 py-2 shadow-sm">
      <div className="flex items-center gap-2">
        <ActionsMenu />
        <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
          {selected ? (
            <Avatar
              label={selected.name}
              color={colorForString(selected.id)}
              className="size-4 text-[0.5rem]"
            />
          ) : (
            <AudioLines />
          )}
          {selected ? selected.name : "Choose voice"}
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <span className="px-1 text-xs tabular-nums text-muted-foreground">
          {fps} fps
        </span>
        <div className="h-4 w-px bg-border" />
        <ThemeToggle />
      </div>

      <VoiceModal
        open={open}
        onClose={() => setOpen(false)}
        voices={voices}
        loading={loading}
        selectedId={selectedId}
        onSelect={setSelectedId}
      />
    </div>
  )
}
