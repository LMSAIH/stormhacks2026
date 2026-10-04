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
  createConversation,
  updateConversation,
  type ConversationPayload,
} from "@/lib/conversations/api"
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
  const [savedChatId, setSavedChatId] = useState<string | null>(null)
  const [savingChat, setSavingChat] = useState(false)
  const [chatSaved, setChatSaved] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

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

  useEffect(() => {
    setChatSaved(false)
    setSaveError(null)
  }, [messages])

  const saveConversation = async () => {
    const finalized = messages.filter(
      (message) => message.final && message.text.trim()
    )
    if (!finalized.length || savingChat) return

    const speakerIds = [
      ...new Set(finalized.map((message) => message.speakerId)),
    ]
    const payload: ConversationPayload = {
      speakers: speakerIds.map((id) => ({
        id,
        name: listening.speakers[id]?.name ?? id,
      })),
      messages: finalized.map((message) => ({
        speaker_id: message.speakerId,
        text: message.text,
        at: Math.round(performance.timeOrigin + message.at),
      })),
    }

    setSavingChat(true)
    setChatSaved(false)
    setSaveError(null)
    try {
      if (savedChatId) {
        await updateConversation(savedChatId, payload)
      } else {
        const created = await createConversation(payload)
        setSavedChatId(created.id)
      }
      setChatSaved(true)
    } catch {
      setSaveError(
        "Conversation could not be saved. Check your connection and try again."
      )
    } finally {
      setSavingChat(false)
    }
  }

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
            fps={lip.fps}
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
              onSave={saveConversation}
              canSave={
                authed &&
                messages.some((message) => message.final && message.text.trim())
              }
              saving={savingChat}
              saved={chatSaved}
              saveError={saveError}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default AppPage
