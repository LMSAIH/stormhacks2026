// Auth lives in a shared context so the router guard, pages, and menu agree on one source of truth.
export { useAuth } from "@/lib/backend/auth-context"
