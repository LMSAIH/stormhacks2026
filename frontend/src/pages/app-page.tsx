import { useMemo } from "react"

import { CameraPanel } from "@/components/app/camera-panel"
import { LipControls } from "@/components/app/lip-controls"
import { OptionsBox } from "@/components/app/options-box"
import { SelfTranscript } from "@/components/app/self-transcript"
import {
  ConversationFeed,
  type FeedMessage,
} from "@/components/app/conversation-feed"
import { useLipReader } from "@/hooks/useLipReader"
import { useListening } from "@/hooks/useListening"

/**
 * Live app screen: camera + your transcription (left), actions bar + diarized
 * conversation (right). Past conversations live under /notes.
 */
export function AppPage() {
  const lip = useLipReader({ active: true })
  const listening = useListening({ active: true })

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
          {/* Speed / Accuracy toggle + push-to-talk (or hold Space) */}
          <LipControls
            mode={lip.mode}
            onModeChange={lip.setMode}
            engines={lip.engines}
            recording={lip.recording}
            busy={lip.busy}
            captureReady={lip.cameraStatus === "on" && lip.ready}
            onStart={lip.startUtterance}
            onStop={lip.stopUtterance}
            lastError={lip.lastError}
            last={lip.transcript.at(-1)}
          />
          <SelfTranscript
            items={lip.transcript}
            ready={lip.ready}
            inferring={lip.inferring}
          />
        </div>

        {/* Actions bar + diarized conversation — right half */}
        <div className="flex h-1/2 min-h-0 flex-col gap-4 sm:h-full sm:w-1/2">
          <OptionsBox fps={lip.fps} />
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
