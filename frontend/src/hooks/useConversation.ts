import { useEffect, useState } from "react"

import { getConversation } from "@/lib/conversations/api"
import type { Conversation } from "@/lib/conversations/types"

/** Loads a single conversation by id. */
export function useConversation(id: string | undefined) {
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) {
      setConversation(null)
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    getConversation(id)
      .then((c) => {
        if (!cancelled) setConversation(c)
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(
            reason instanceof Error
              ? reason.message
              : "Could not load conversation"
          )
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [id])

  return { conversation, loading, error }
}
