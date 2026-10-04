import { useEffect, useState } from "react"

import { searchConversations } from "@/lib/conversations/api"
import type { Conversation } from "@/lib/conversations/types"

/**
 * Loads conversations matching `query` (empty = all). Re-runs on query change.
 * Backed by the pluggable searchConversations (simple now, vector search later).
 */
export function useConversations(query: string) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    searchConversations(query)
      .then((c) => {
        if (!cancelled) setConversations(c)
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(
            reason instanceof Error
              ? reason.message
              : "Could not load conversations"
          )
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [query])

  return { conversations, loading, error }
}
