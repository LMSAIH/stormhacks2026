import { useEffect, useRef, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import {
  Check,
  Menu,
  MessagesSquare,
  NotebookPen,
  type LucideIcon,
} from "lucide-react"

import { Button } from "@/components/ui/button"

/** Dropdown menu in the actions bar: navigate between app sections. */
export function ActionsMenu() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const { pathname } = useLocation()

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

  const go = (to: string) => {
    navigate(to)
    setOpen(false)
  }

  return (
    <div ref={ref} className="relative">
      <Button
        size="icon-sm"
        variant="ghost"
        onClick={() => setOpen((o) => !o)}
        aria-label="Menu"
        aria-expanded={open}
      >
        <Menu />
      </Button>

      {open && (
        <div className="absolute top-full left-0 z-50 mt-1.5 w-56 overflow-hidden rounded-xl border border-border bg-popover p-1 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95">
          <MenuItem
            icon={MessagesSquare}
            label="Live conversation"
            active={pathname.startsWith("/app")}
            onClick={() => go("/app")}
          />
          <MenuItem
            icon={NotebookPen}
            label="Notes"
            active={pathname.startsWith("/notes")}
            onClick={() => go("/notes")}
          />
        </div>
      )}
    </div>
  )
}

function MenuItem({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: LucideIcon
  label: string
  active?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-muted"
    >
      <Icon className="size-3.5 text-muted-foreground" />
      <span className="flex-1 text-left">{label}</span>
      {active && <Check className="size-3.5 text-foreground" />}
    </button>
  )
}
