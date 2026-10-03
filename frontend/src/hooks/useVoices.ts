import { useEffect, useState } from "react"

import { fetchVoices } from "@/lib/voices/api"
import type { Voice } from "@/lib/voices/types"

/** Loads the available voices once (via the pluggable fetchVoices API). */
export function useVoices() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchVoices()
      .then((v) => {
        if (!cancelled) setVoices(v)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { voices, loading, error }
}
