/**
 * Vivid, harmonic accent palette (Tailwind ~600 level). Shared by speaker
 * diarization and voice avatars so color identity is consistent app-wide.
 */
export const ACCENT_COLORS: readonly string[] = [
  "#0891b2", // cyan-600
  "#7c3aed", // violet-600
  "#e11d48", // rose-600
  "#059669", // emerald-600
  "#ea580c", // orange-600
  "#2563eb", // blue-600
  "#c026d3", // fuchsia-600
  "#ca8a04", // yellow-600
]

/** Deterministically map any key (e.g. a voice id) to a stable accent color. */
export function colorForString(key: string): string {
  let hash = 0
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0
  }
  return ACCENT_COLORS[hash % ACCENT_COLORS.length]
}
