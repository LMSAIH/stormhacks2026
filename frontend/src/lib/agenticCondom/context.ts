/**
 * The conversation so far, for the Agentic Condom: the page feeds it the finished lines (yours and
 * the other people's captions); each correction sends the last few, so the LLM can pick the word
 * that fits what was just said.
 */
export interface ContextLine {
  readonly who: "user" | "other"
  readonly text: string
}

/** Lines kept (the server's prompt uses this many). */
export const MAX_CONTEXT = 6
/** A long caption keeps only its last this-many characters (the most recent words). */
export const MAX_CONTEXT_CHARS = 300

let lines: readonly ContextLine[] = []

/** Replace the conversation: finished lines with when they started (`at`, performance.now()). */
export function setConversationContext(
  entries: readonly (ContextLine & { readonly at: number })[]
): void {
  lines = entries
    .filter((e) => e.text.trim())
    .sort((a, b) => a.at - b.at)
    .slice(-MAX_CONTEXT)
    .map(({ who, text }) => ({ who, text: clip(text) }))
}

/** The last MAX_CONTEXT lines, oldest first. */
export function conversationContext(): readonly ContextLine[] {
  return lines
}

function clip(text: string): string {
  const flat = text.trim().replace(/\s+/g, " ")
  if (flat.length <= MAX_CONTEXT_CHARS) return flat
  const tail = flat.slice(-MAX_CONTEXT_CHARS)
  const space = tail.indexOf(" ")
  return space >= 0 ? tail.slice(space + 1) : tail
}
