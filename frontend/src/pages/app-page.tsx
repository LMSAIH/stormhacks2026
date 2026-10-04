import { useEffect, useMemo, useRef, useState } from "react"

import { CameraPanel } from "@/components/app/camera-panel"
import { OptionsBox } from "@/components/app/options-box"
import { SelfTranscript } from "@/components/app/self-transcript"
import {
  ConversationFeed,
  type FeedMessage,
} from "@/components/app/conversation-feed"
import { useLipReader } from "@/hooks/useLipReader"
import { useListening } from "@/hooks/useListening"
import { useAuth } from "@/hooks/useAuth"
import { useVoices } from "@/hooks/useVoices"
import { useVoiceOutput } from "@/hooks/useVoiceOutput"
import {
  setConversationContext,
  useCondomEnabled,
  warmCondom,
  WARM_EVERY_MS,
} from "@/lib/agenticCondom"
import { setDefaultVoice } from "@/lib/voices/api"

/**
 * Live app screen: camera + your transcription (left), actions bar + diarized conversation (right).
 *
 * Voice output: each finalized lip-read utterance is streamed to the backend TTS and played aloud
 * in the selected voice (requires sign-in). Past conversations live under /notes.
 */
export function AppPage() {
  const lip = useLipReader({ active: true })
  const listening = useListening({ active: true })

  const { user, signIn } = useAuth()
  const authed = !!user
  const { voices, defaultVoiceId, loading: voicesLoading } = useVoices(authed)
  const [voiceId, setVoiceId] = useState<string | null>(null)
  const [muted, setMuted] = useState(false)

  // Adopt the backend's default voice once it loads (unless the user already picked one).
  useEffect(() => {
    if (defaultVoiceId && !voiceId) setVoiceId(defaultVoiceId)
  }, [defaultVoiceId, voiceId])

  const { speak } = useVoiceOutput({ authed, voiceId })

  const selectVoice = (id: string) => {
    setVoiceId(id)
    if (authed) void setDefaultVoice(id).catch(() => undefined)
  }

  // Speak each new finalized utterance exactly once.
  const spokenRef = useRef<Set<string>>(new Set())
  useEffect(() => {
    for (const item of lip.transcript) {
      if (spokenRef.current.has(item.id)) continue
      spokenRef.current.add(item.id)
      if (!muted) speak(item.text)
    }
  }, [lip.transcript, muted, speak])

  // The Agentic Condom reads the last few finished lines, yours and the captions, as context.
  useEffect(() => {
    setConversationContext([
      ...lip.transcript.map((item) => ({ who: "user" as const, text: item.text, at: item.at })),
      ...listening.utterances
        .filter((u) => u.final)
        .map((u) => ({ who: "other" as const, text: u.text, at: u.at })),
    ])
  }, [lip.transcript, listening.utterances])

  // ...and keeps its server connection warm while it can run (one GET /health per ~45 s).
  const condomOn = useCondomEnabled()
  useEffect(() => {
    if (!condomOn || lip.mode === "instant") return
    warmCondom()
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") warmCondom()
    }, WARM_EVERY_MS + 1000)
    return () => window.clearInterval(timer)
  }, [condomOn, lip.mode])

  const messages = useMemo<FeedMessage[]>(
    () =>
      listening.utterances
        .map((u) => ({
          id: u.id,
          speakerId: u.speakerId,
          text: u.text,
          final: u.final,
          at: u.at,
          isSelf: false,
        }))
        .sort((a, b) => a.at - b.at),
    [listening.utterances]
  )

  return (
    <div className="flex h-svh flex-col">
      <main className="flex flex-1 flex-col gap-4 overflow-hidden p-4 sm:flex-row">
        {/* Camera + your transcription — left half */}
        <div className="flex h-1/2 min-h-0 flex-col gap-4 sm:h-full sm:w-1/2">
          <div className="min-h-0 flex-1">
            <CameraPanel
              videoRef={lip.videoRef}
              overlayRef={lip.overlayRef}
              cameraStatus={lip.cameraStatus}
            />
          </div>
          <SelfTranscript
            items={lip.transcript}
            ready={lip.ready}
            inferring={lip.inferring}
            draft={lip.draft?.text}
            onPick={lip.pickChoice}
            hint={lip.faceHint}
          />
        </div>

        {/* Actions bar + diarized conversation — right half */}
        <div className="flex h-1/2 min-h-0 flex-col gap-4 sm:h-full sm:w-1/2">
          <OptionsBox
            voices={voices}
            voicesLoading={voicesLoading}
            selectedVoiceId={voiceId}
            onSelectVoice={selectVoice}
            authed={authed}
            onSignIn={signIn}
            user={user}
            muted={muted}
            onToggleMute={() => setMuted((m) => !m)}
            lipMode={lip.mode}
            onLipMode={lip.setMode}
            cloudAvailable={lip.cloudAvailable}
            shareClips={lip.shareClips}
            onShareClips={lip.setShareClips}
            canShareClips={lip.canShareClips}
          />
          <div className="min-h-0 flex-1">
            <ConversationFeed
              messages={messages}
              speakers={listening.speakers}
              onRename={listening.renameSpeaker}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default AppPage
