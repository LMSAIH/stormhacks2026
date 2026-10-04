import { useCallback, useEffect, useRef, useState } from "react"

import { TtsClient, type TtsStatus } from "@/lib/backend/tts"

interface UseVoiceOutputOptions {
  /** Only connect/speak while signed in (the TTS socket requires auth). */
  authed: boolean
  /** Reconnect when this changes so the backend picks up the new voice. */
  voiceId: string | null
}

/**
 * Owns the streaming TTS client: connects while signed in, reconnects when the voice changes, and
 * resumes the AudioContext on the first user gesture (autoplay policy). Exposes `speak(text)`.
 */
export function useVoiceOutput({ authed, voiceId }: UseVoiceOutputOptions) {
  const clientRef = useRef<TtsClient | null>(null)
  const [status, setStatus] = useState<TtsStatus>("idle")

  // Create the client once; resume audio on the first interaction.
  useEffect(() => {
    const client = new TtsClient(setStatus)
    clientRef.current = client
    const resume = () => void client.ensureAudio()
    window.addEventListener("pointerdown", resume)
    window.addEventListener("keydown", resume)
    return () => {
      window.removeEventListener("pointerdown", resume)
      window.removeEventListener("keydown", resume)
      client.dispose()
      clientRef.current = null
    }
  }, [])

  // (Re)connect when auth or the selected voice changes.
  useEffect(() => {
    const client = clientRef.current
    if (!client) return
    if (authed) client.reconnect()
    else client.disconnect()
  }, [authed, voiceId])

  const speak = useCallback(
    (text: string) => {
      if (!authed) return
      void clientRef.current?.speak(text)
    },
    [authed]
  )

  return { speak, status }
}
