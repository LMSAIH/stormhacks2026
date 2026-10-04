import { useCallback, useEffect, useRef, useState } from "react"

import { createListeningEngine } from "@/lib/listening/createEngine"
import type { ListeningEngine, Speaker, Utterance } from "@/lib/listening/types"

interface UseListeningOptions {
  active: boolean
}

/**
 * Drives "listening" mode: starts the ASR+diarization engine while active,
 * collects diarized utterances, and exposes editable speaker names.
 */
export function useListening({ active }: UseListeningOptions) {
  const [speakers, setSpeakers] = useState<Record<string, Speaker>>({})
  const [utterances, setUtterances] = useState<Utterance[]>([])
  const [engineName, setEngineName] = useState("")
  const engineRef = useRef<ListeningEngine | null>(null)
  // Local name overrides survive even if the engine re-emits a speaker.
  const nameOverrides = useRef<Record<string, string>>({})

  useEffect(() => {
    if (!active) return
    let stopped = false

    const start = async () => {
      const engine = await createListeningEngine()
      if (stopped) return
      engineRef.current = engine
      setEngineName(engine.name)

      await engine.start({
        onSpeaker: (speaker) => {
          setSpeakers((prev) => {
            if (prev[speaker.id]) return prev
            const name = nameOverrides.current[speaker.id] ?? speaker.name
            return { ...prev, [speaker.id]: { ...speaker, name } }
          })
        },
        onUtterance: (utterance) => {
          setUtterances((prev) => {
            const idx = prev.findIndex((u) => u.id === utterance.id)
            if (idx === -1) return [...prev, utterance]
            const next = prev.slice()
            // Keep the first-seen timestamp so ordering stays stable as it updates.
            next[idx] = { ...utterance, at: prev[idx].at }
            return next
          })
        },
        onDrop: (id) => {
          setUtterances((prev) => prev.filter((u) => u.id !== id))
        },
      })
    }

    start().catch((err) => {
      console.error("[useListening] failed to start:", err)
      setEngineName("unavailable")
    })

    return () => {
      stopped = true
      engineRef.current?.stop()
      engineRef.current = null
    }
  }, [active])

  const renameSpeaker = useCallback((id: string, name: string) => {
    nameOverrides.current[id] = name
    setSpeakers((prev) =>
      prev[id] ? { ...prev, [id]: { ...prev[id], name } } : prev
    )
  }, [])

  const clear = useCallback(() => {
    setUtterances([])
  }, [])

  return { speakers, utterances, engineName, renameSpeaker, clear }
}
