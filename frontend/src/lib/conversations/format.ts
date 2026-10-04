/** Shared, locale-aware time/date formatting for conversations. */

export function formatTime(at: number): string {
  return new Date(at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function formatDate(at: number): string {
  return new Date(at).toLocaleDateString([], {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  })
}

export function formatDateTime(at: number): string {
  return new Date(at).toLocaleString()
}
