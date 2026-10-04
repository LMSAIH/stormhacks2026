import { useEffect, useState } from "react"

import { fetchMe, signInWithGoogle, type User } from "@/lib/backend/auth"

/** Loads the current user once and exposes a sign-in action. */
export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetchMe()
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { user, loading, signIn: signInWithGoogle }
}
