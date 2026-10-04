import { useState } from "react"
import { AudioLines, LogIn, Volume2, VolumeX } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/app/theme-toggle"
import { VoiceModal } from "@/components/app/voice-modal"
import { Avatar } from "@/components/app/avatar"
import { ActionsMenu } from "@/components/app/actions-menu"
import { LipModeMenu } from "@/components/app/lip-mode-menu"
import type { LipMode } from "@/lib/lipreading/modes"
import { colorForString } from "@/lib/palette"
import type { Voice } from "@/lib/voices/types"

interface OptionsBoxProps {
  fps: number
  voices: Voice[]
  voicesLoading: boolean
  selectedVoiceId: string | null
  onSelectVoice: (id: string) => void
  /** Signed in — enables voice output. */
  authed: boolean
  onSignIn: () => void
  muted: boolean
  onToggleMute: () => void
  /** Lip reading mode picker (instant / normal / quality); hidden when not passed. */
  lipMode?: LipMode
  onLipMode?: (mode: LipMode) => void
  cloudAvailable?: boolean
  /** Opt-in to share picked fixes as training clips (shown in the mode menu). */
  shareClips?: boolean
  onShareClips?: (on: boolean) => void
  canShareClips?: boolean
}

/** Actions bar: menu, voice selection, voice-output controls, fps, theme. */
export function OptionsBox({
  fps,
  voices,
  voicesLoading,
  selectedVoiceId,
  onSelectVoice,
  authed,
  onSignIn,
  muted,
  onToggleMute,
  lipMode,
  onLipMode,
  cloudAvailable = false,
  shareClips,
  onShareClips,
  canShareClips,
}: OptionsBoxProps) {
  const [open, setOpen] = useState(false)
  const selected = voices.find((v) => v.id === selectedVoiceId)

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
        {lipMode && onLipMode && (
          <LipModeMenu
            mode={lipMode}
            onChange={onLipMode}
            cloudAvailable={cloudAvailable}
            shareClips={shareClips}
            onShareClips={onShareClips}
            canShareClips={canShareClips}
          />
        )}
      </div>

      <div className="flex items-center gap-2">
        {authed ? (
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={onToggleMute}
            aria-label={muted ? "Unmute voice" : "Mute voice"}
            title={muted ? "Voice muted" : "Voice on"}
            className="text-muted-foreground"
          >
            {muted ? <VolumeX /> : <Volume2 />}
          </Button>
        ) : (
          <Button size="sm" variant="outline" onClick={onSignIn}>
            <LogIn />
            Sign in
          </Button>
        )}
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
        loading={voicesLoading}
        selectedId={selectedVoiceId}
        onSelect={onSelectVoice}
      />
    </div>
  )
}
