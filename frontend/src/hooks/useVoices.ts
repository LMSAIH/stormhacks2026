import { useEffect, useState } from "react"

import { fetchVoices } from "@/lib/voices/api"
import type { Voice } from "@/lib/voices/types"

/**
 * Loads the available voices (from the backend, falling back to the bundled snapshot). Re-runs when
 * `reloadKey` changes — e.g. after sign-in, so the authenticated list + default voice load in.
 */
export function useVoices(reloadKey: unknown = null) {
  const [voices, setVoices] = useState<Voice[]>([])
  const [defaultVoiceId, setDefaultVoiceId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchVoices()
      .then((result) => {
        if (cancelled) return
        setVoices(result.voices)
        setDefaultVoiceId(result.defaultVoiceId)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [reloadKey])

  return { voices, defaultVoiceId, loading }
}
