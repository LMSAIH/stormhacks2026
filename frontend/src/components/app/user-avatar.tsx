import { useState } from "react"
import { cn } from "cn"

import { Avatar } from "@/components/app/avatar"
import { colorForString } from "@/lib/palette"
import type { User } from "@/lib/backend/auth"

/**
 * The signed-in user's Google photo, falling back to a monogram avatar if the image is missing or
 * blocked (profile images are cross-origin; our COEP headers can reject them).
 */
export function UserAvatar({
  user,
  className,
}: {
  user: User
  className?: string
}) {
  const [failed, setFailed] = useState(false)

  if (user.picture && !failed) {
    return (
      <img
        src={user.picture}
        alt={user.name ?? "Profile"}
        crossOrigin="anonymous"
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
        className={cn("size-7 shrink-0 rounded-full object-cover", className)}
      />
    )
  }

  return (
    <Avatar
      label={user.name ?? user.email ?? "?"}
      color={colorForString(user.id)}
      className={cn("size-7", className)}
    />
  )
}
