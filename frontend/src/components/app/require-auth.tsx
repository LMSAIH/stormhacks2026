import { Loader2 } from "lucide-react"
import { Navigate, Outlet } from "react-router-dom"

import { useAuth } from "@/lib/backend/auth-context"

/** Gate for protected routes: redirect to /login when signed out. */
export function RequireAuth() {
  const { user, loading } = useAuth()

  // Dev only: VITE_SKIP_AUTH=1 in .env.local opens the app without the backend. Never in builds.
  if (import.meta.env.DEV && import.meta.env.VITE_SKIP_AUTH === "1") return <Outlet />

  if (loading) {
    return (
      <div className="flex h-svh items-center justify-center text-muted-foreground">
        <Loader2 className="size-5 animate-spin" />
      </div>
    )
  }

  if (!user) return <Navigate to="/login" replace />

  return <Outlet />
}
