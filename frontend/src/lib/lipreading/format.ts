/** "850 ms" below a second, "1.25 s" above. */
export function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

/**
 * The model emits UPPERCASE; read it as a sentence ("I" stays capitalised). Text that already has
 * lowercase letters (e.g. from a corrector) is left alone.
 */
export function toSentenceCase(text: string): string {
  if (text !== text.toUpperCase()) return text
  const lower = text.toLowerCase().replace(/\bi\b/g, "I")
  return lower.charAt(0).toUpperCase() + lower.slice(1)
}
