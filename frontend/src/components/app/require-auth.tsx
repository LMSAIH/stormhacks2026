import { Loader2 } from "lucide-react"
import { Navigate, Outlet } from "react-router-dom"

import { useAuth } from "@/lib/backend/auth-context"

/** Gate for protected routes: redirect to /login when signed out. */
export function RequireAuth() {
  const { user, loading } = useAuth()

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
