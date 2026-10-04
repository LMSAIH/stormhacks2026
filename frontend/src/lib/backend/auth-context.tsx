import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"

import {
  fetchMe,
  logout as apiLogout,
  signInWithGoogle,
  type User,
} from "./auth"

interface AuthState {
  user: User | null
  /** True until the first /api/auth/me check resolves. */
  loading: boolean
  signIn: () => void
  signOut: () => Promise<void>
  reload: () => Promise<void>
}

const AuthContext = createContext<AuthState | undefined>(undefined)

/** App-wide auth state: loads the current user once, exposes sign-in/out. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  const reload = useCallback(async () => {
    const u = await fetchMe()
    setUser(u)
  }, [])

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

  const signOut = useCallback(async () => {
    await apiLogout()
    setUser(null)
  }, [])

  const value = useMemo<AuthState>(
    () => ({ user, loading, signIn: signInWithGoogle, signOut, reload }),
    [user, loading, signOut, reload]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider")
  return ctx
}
