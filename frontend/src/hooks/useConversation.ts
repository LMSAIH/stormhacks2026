import { useCallback, useEffect, useState } from "react"

import { useAuth } from "@/hooks/useAuth"
import {
  deleteConversation,
  getConversation,
  saveConversation,
} from "@/lib/conversations/api"
import type { Conversation } from "@/lib/conversations/types"

/** Loads one of the user's conversations by id, and lets its title be renamed or the whole thing deleted. */
export function useConversation(id: string | undefined) {
  const { user } = useAuth()
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [loading, setLoading] = useState(true)
  const uid = user?.id

  useEffect(() => {
    let cancelled = false
    if (!uid || !id) {
      setConversation(null)
      setLoading(false)
      return
    }
    setLoading(true)
    getConversation(id)
      .then((c) => {
        if (!cancelled) setConversation(c)
      })
      .catch(() => {
        if (!cancelled) setConversation(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [uid, id])

  const rename = useCallback(
    (title: string) => {
      if (!uid || !id) return
      setConversation((prev) => {
        const next = prev ? { ...prev, title } : prev
        if (next) void saveConversation(id, next)
        return next
      })
    },
    [uid, id]
  )

  const remove = useCallback(async () => {
    if (!uid || !id) return
    await deleteConversation(id)
  }, [uid, id])

  return { conversation, loading, rename, remove }
}
