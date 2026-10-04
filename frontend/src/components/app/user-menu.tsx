import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { LogOut } from "lucide-react"

import { UserAvatar } from "@/components/app/user-avatar"
import { useAuth } from "@/lib/backend/auth-context"

/** Click the profile picture → a small menu with your account and Sign out. */
export function UserMenu() {
  const { user, signOut } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false)
    window.addEventListener("mousedown", onDown)
    window.addEventListener("keydown", onKey)
    return () => {
      window.removeEventListener("mousedown", onDown)
      window.removeEventListener("keydown", onKey)
    }
  }, [open])

  if (!user) return null

  const handleSignOut = async () => {
    setOpen(false)
    await signOut()
    navigate("/login", { replace: true })
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account"
        className="flex rounded-full outline-none ring-offset-background transition-shadow focus-visible:ring-2 focus-visible:ring-ring/40"
      >
        <UserAvatar user={user} className="size-6" />
      </button>

      {open && (
        <div className="absolute top-full right-0 z-50 mt-1.5 w-56 overflow-hidden rounded-xl border border-border bg-popover p-1 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95">
          <div className="flex items-center gap-2 px-2 py-1.5">
            <UserAvatar user={user} className="size-8" />
            <div className="min-w-0">
              {user.name && (
                <p className="truncate text-xs font-medium">{user.name}</p>
              )}
              <p className="truncate text-[0.625rem] text-muted-foreground">
                {user.email ?? "Signed in"}
              </p>
            </div>
          </div>

          <div className="my-1 h-px bg-border" />

          <button
            type="button"
            onClick={handleSignOut}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-muted"
          >
            <LogOut className="size-3.5 text-muted-foreground" />
            <span className="flex-1 text-left">Sign out</span>
          </button>
        </div>
      )}
    </div>
  )
}
