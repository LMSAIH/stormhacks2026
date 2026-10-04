import { apiGet, isUnauthorized } from "./client"
import { API_URL, googleLoginUrl } from "./config"

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

/**
 * Clear the server session. A bodiless POST with no custom headers is a CORS "simple request"
 * (no preflight), so it works even though the backend only lists GET/PUT in allow_methods.
 */
export async function logout(): Promise<void> {
  try {
    await fetch(`${API_URL}/api/auth/logout`, {
      method: "POST",
      credentials: "include",
    })
  } catch (err) {
    console.warn("[auth] logout failed:", err)
  }
}
