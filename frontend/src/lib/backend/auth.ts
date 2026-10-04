import { apiGet, isUnauthorized } from "./client"
import { googleLoginUrl } from "./config"

export interface User {
  id: string
  email?: string
  name?: string
  picture?: string
}

/** Current signed-in user, or null if not authenticated / backend unreachable. */
export async function fetchMe(): Promise<User | null> {
  try {
    const { user } = await apiGet<{ user: User }>("/api/auth/me")
    return user
  } catch (err) {
    if (isUnauthorized(err)) return null
    // Backend down / CORS / network — treat as signed-out rather than throwing.
    console.warn("[auth] /api/auth/me failed:", err)
    return null
  }
}

/** Full-page redirect into Google OAuth; returns to the app afterwards. */
export function signInWithGoogle(): void {
  window.location.href = googleLoginUrl()
}
