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

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    searchConversations(query)
      .then((c) => {
        if (!cancelled) setConversations(c)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [query])

  return { conversations, loading }
}
