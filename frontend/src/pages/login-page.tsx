import { Loader2 } from "lucide-react"
import { Link, Navigate } from "react-router-dom"

import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/backend/auth-context"

/** Dedicated sign-in screen. Signed-out users are redirected here. */
export function LoginPage() {
  const { user, loading, signIn } = useAuth()

  if (loading) {
    return (
      <div className="flex h-svh items-center justify-center text-muted-foreground">
        <Loader2 className="size-5 animate-spin" />
      </div>
    )
  }

  // Already signed in — don't show the login screen.
  if (user) return <Navigate to="/app" replace />

  return (
    <div className="flex h-svh items-center justify-center p-6">
      <div className="w-full max-w-sm rounded-2xl border border-border bg-card p-8 text-center shadow-sm">
        <h1 className="text-3xl font-semibold italic tracking-tight">heard</h1>
        <p className="mx-auto mt-2 max-w-[16rem] text-sm text-muted-foreground">
          Sign in to read lips, hear them spoken aloud, and keep your conversations.
        </p>

        <Button className="mt-6 w-full" size="lg" onClick={signIn}>
          <GoogleGlyph />
          Continue with Google
        </Button>

        <p className="mt-4 text-[0.625rem] text-muted-foreground">
          We use your Google account only to sign you in. See our{" "}
          <Link
            to="/privacy"
            className="underline underline-offset-4 transition-colors hover:text-foreground"
          >
            Privacy Policy
          </Link>
          .
        </p>
      </div>
    </div>
  )
}

/** Small inline Google "G" so the button reads as an official provider action. */
function GoogleGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
      <path
        fill="#4285F4"
        d="M23.52 12.27c0-.79-.07-1.54-.2-2.27H12v4.3h6.47a5.53 5.53 0 0 1-2.4 3.63v3h3.88c2.27-2.09 3.57-5.17 3.57-8.66z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.24 0 5.96-1.08 7.95-2.91l-3.88-3c-1.08.72-2.45 1.16-4.07 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.09A12 12 0 0 0 12 24z"
      />
      <path
        fill="#FBBC05"
        d="M5.27 14.29A7.2 7.2 0 0 1 4.89 12c0-.8.14-1.57.38-2.29V6.62H1.29A12 12 0 0 0 0 12c0 1.94.46 3.77 1.29 5.38l3.98-3.09z"
      />
      <path
        fill="#EA4335"
        d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.44-3.44C17.95 1.19 15.24 0 12 0A12 12 0 0 0 1.29 6.62l3.98 3.09C6.22 6.86 8.87 4.75 12 4.75z"
      />
    </svg>
  )
}
