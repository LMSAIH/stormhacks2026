import { useMemo } from "react"

import { CameraPanel } from "@/components/app/camera-panel"
import {
  ConversationFeed,
  type FeedMessage,
} from "@/components/app/conversation-feed"
import { useLipReader } from "@/hooks/useLipReader"
import { useListening } from "@/hooks/useListening"

/**
 * Main app screen — a single always-on conversation.
 *
 * The camera (top-left) is the pipeline entrypoint. Both pipelines run at once:
 *  - Listening: diarizes people speaking nearby (editable names).
 *  - Lip reading: transcribes what *you* utter, shown as "You".
 * Both streams merge into one time-ordered feed.
 */
export function AppPage() {
  const lip = useLipReader({ active: true })
  const listening = useListening({ active: true })

  const messages = useMemo<FeedMessage[]>(() => {
    const others: FeedMessage[] = listening.utterances.map((u) => ({
      id: u.id,
      speakerId: u.speakerId,
      text: u.text,
      final: u.final,
      at: u.at,
      isSelf: false,
    }))
    const mine: FeedMessage[] = lip.transcript.map((t) => ({
      id: t.id,
      speakerId: "self",
      text: t.text,
      final: true,
      at: t.at,
      isSelf: true,
    }))
    return [...others, ...mine].sort((a, b) => a.at - b.at)
  }, [listening.utterances, lip.transcript])

  const clearAll = () => {
    listening.clear()
    lip.clearTranscript()
  }

  return (
    <div className="flex h-svh flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-baseline gap-2">
          <h1 className="text-sm font-semibold">Lipreader</h1>
          <span className="text-[0.625rem] text-muted-foreground">
            live conversation
          </span>
        </div>
      </header>

      <main className="relative flex-1 overflow-hidden p-4">
        {/* Debug camera — top-left entrypoint */}
        <div className="absolute top-4 left-4 z-10">
          <CameraPanel
            videoRef={lip.videoRef}
            overlayRef={lip.overlayRef}
            cameraStatus={lip.cameraStatus}
            mouthDetected={lip.mouthDetected}
            fps={lip.fps}
            engineName={lip.engineName}
            engineReal={lip.engineReal}
            inferring={lip.inferring}
          />
        </div>

        <div className="mx-auto h-full max-w-2xl pl-0 sm:pl-60">
          <ConversationFeed
            messages={messages}
            speakers={listening.speakers}
            engineName={listening.engineName}
            onRename={listening.renameSpeaker}
            onClear={clearAll}
          />
        </div>
      </main>
    </div>
  )
}

export default AppPage
