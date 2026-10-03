import { useMemo } from "react"
import { CircleAlert } from "lucide-react"

import { CameraPanel, formatLatency } from "@/components/app/camera-panel"
import {
  ConversationFeed,
  type FeedMessage,
} from "@/components/app/conversation-feed"
import { ModeToggle } from "@/components/app/mode-toggle"
import { PttButton } from "@/components/app/ptt-button"
import { useLipReader, type LipTranscriptItem } from "@/hooks/useLipReader"
import { useListening } from "@/hooks/useListening"
import { ACTIVE_SPEC } from "@/lib/lipreading/modelSpec"

/**
 * Main app screen — a single always-on conversation.
 *
 * The camera (top-left) is the pipeline entrypoint. Both pipelines run at once:
 *  - Listening: diarizes people speaking nearby (editable names).
 *  - Lip reading: hold Space / the talk button, release → what *you* mouthed, shown as "You".
 * Both streams merge into one time-ordered feed. Speed (on-device) vs Accuracy (hosted beam)
 * is a toggle in the header.
 */
export function AppPage() {
  const lip = useLipReader()
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
      text: `${toSentenceCase(t.text)} · ${resultSuffix(t)}`,
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

  const lastResult = lip.transcript.at(-1)
  const captureReady = lip.cameraStatus === "on" && lip.ready

  return (
    <div className="flex h-svh flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-2">
        <div className="flex items-baseline gap-2">
          <h1 className="text-sm font-semibold">Lipreader</h1>
          <span className="hidden text-[0.625rem] text-muted-foreground sm:inline">
            live conversation
          </span>
        </div>
        <ModeToggle
          mode={lip.mode}
          onChange={lip.setMode}
          engines={lip.engines}
        />
      </header>

      <main className="flex min-h-0 flex-1 flex-col gap-4 p-4 sm:flex-row">
        {/* Debug camera (pipeline entrypoint) with push-to-talk under it. */}
        <aside className="flex shrink-0 gap-3 sm:w-56 sm:flex-col">
          <CameraPanel
            className="w-40 shrink-0 sm:w-full"
            videoRef={lip.videoRef}
            overlayRef={lip.overlayRef}
            cameraStatus={lip.cameraStatus}
            mouthDetected={lip.mouthDetected}
            fps={lip.fps}
            recording={lip.recording}
            busy={lip.busy}
            mode={lip.mode}
            engine={lip.engines[lip.mode]}
            last={lastResult}
          />
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <PttButton
              recording={lip.recording}
              busy={lip.busy}
              disabled={!captureReady}
              busyLabel={
                lip.engines[lip.mode].loading
                  ? "Loading model…"
                  : "Reading lips…"
              }
              maxSeconds={ACTIVE_SPEC.maxSeconds}
              onStart={lip.startUtterance}
              onStop={lip.stopUtterance}
            />
            <p
              role="status"
              aria-live="polite"
              className="flex min-h-4 items-start gap-1 px-1 text-[0.625rem] leading-snug text-amber-700 dark:text-amber-300"
            >
              {lip.lastError && (
                <>
                  <CircleAlert className="mt-px size-3 shrink-0" />
                  {lip.lastError}
                </>
              )}
            </p>
          </div>
        </aside>

        <div className="mx-auto min-h-0 w-full max-w-2xl flex-1">
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

/** The model emits UPPERCASE; read it as a sentence ("I" stays capitalised). */
function toSentenceCase(text: string): string {
  if (text !== text.toUpperCase()) return text
  const lower = text.toLowerCase().replace(/\bi\b/g, "I")
  return lower.charAt(0).toUpperCase() + lower.slice(1)
}

function resultSuffix(t: LipTranscriptItem): string {
  const mode = t.fellBack ? "speed (fallback)" : t.mode
  return `${mode} ${formatLatency(t.latencyMs)}`
}

export default AppPage
