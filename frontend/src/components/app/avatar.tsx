import { cn } from "cn"

interface AvatarProps {
  /** Text to derive the initial from. */
  label: string
  /** Any CSS color (hex like "#0891b2" or a var like "var(--primary)"). */
  color: string
  className?: string
}

/**
 * Monogram avatar: a soft-tinted circle with the initial in the accent color.
 * Gives speakers and voices a calm, consistent identity.
 */
export function Avatar({ label, color, className }: AvatarProps) {
  const initial = (label.trim()[0] || "?").toUpperCase()
  return (
    <span
      aria-hidden
      className={cn(
        "flex size-8 shrink-0 select-none items-center justify-center rounded-full text-xs font-semibold",
        className
      )}
      style={{
        backgroundColor: `color-mix(in oklab, ${color} 16%, transparent)`,
        color,
      }}
    >
      {initial}
    </span>
  )
}
