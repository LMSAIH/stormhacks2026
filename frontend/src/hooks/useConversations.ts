import { useCallback, useEffect, useMemo, useState } from "react"

import { useAuth } from "@/hooks/useAuth"
import { deleteConversation, listConversations } from "@/lib/conversations/api"
import { conversationMatches } from "@/lib/conversations/store"
import type { Conversation } from "@/lib/conversations/types"

/** The signed-in user's recorded conversations, filtered by `query`. Newest first. */
export function useConversations(query: string) {
  const { user } = useAuth()
  const [all, setAll] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const uid = user?.id

  const refresh = useCallback(async () => {
    if (!uid) {
      setAll([])
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      setAll(await listConversations())
    } catch {
      setAll([])
    }
    setLoading(false)
  }, [uid])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const remove = useCallback(async (id: string) => {
    setAll((prev) => prev.filter((c) => c.id !== id))
    try {
      await deleteConversation(id)
    } catch {
      // Re-sync if the delete failed, so the card reappears.
      void refresh()
    }
  }, [refresh])

  const conversations = useMemo(
    () => all.filter((c) => conversationMatches(c, query)),
    [all, query]
  )

  return { conversations, loading, remove, refresh }
}
