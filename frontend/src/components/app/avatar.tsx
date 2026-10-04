import { cn } from "cn"

interface AvatarProps {
  /** Text to derive the initial from. */
  label: string
  /** Any CSS color (hex like "#0891b2" or a var like "var(--primary)"). */
  color: string
  /** Text color override. Defaults to white (or primary-foreground for primary). */
  textColor?: string
  className?: string
}

/**
 * Monogram avatar: a solid accent-colored circle with the initial in white.
 * Gives speakers and voices a clear, consistent identity.
 */
export function Avatar({ label, color, textColor, className }: AvatarProps) {
  const initial = (label.trim()[0] || "?").toUpperCase()
  const fg =
    textColor ?? (color === "var(--primary)" ? "var(--primary-foreground)" : "#fff")
  return (
    <span
      aria-hidden
      className={cn(
        "flex size-8 shrink-0 select-none items-center justify-center rounded-full text-xs font-semibold",
        className
      )}
      style={{ backgroundColor: color, color: fg }}
    >
      {initial}
    </span>
  )
}
