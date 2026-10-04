/**
 * The three ways a sentence gets read (plan D55, D67, D69). Caps come from the length table
 * (.context/streaming-length-table.md): errors don't rise with length, only the wait does
 * (browser ≈ 0.36 s per second of video).
 */
export type LipMode = "instant" | "normal" | "quality"

export interface LipModeInfo {
  readonly label: string
  /** Where the final reading runs: this device, or the GPU server. */
  readonly where: "Local" | "Cloud"
  /** A sentence locks at this length even without a pause. */
  readonly maxSeconds: number
  /** Show quick drafts of each piece while the sentence is still going. */
  readonly drafts: boolean
  /** How the locked sentence is read: not again (instant), on this device, or on the server. */
  readonly final: "none" | "local" | "cloud"
  readonly detail: string
}

export const LIP_MODES: Readonly<Record<LipMode, LipModeInfo>> = {
  instant: {
    label: "Instant",
    where: "Local",
    maxSeconds: 2,
    drafts: false,
    final: "none",
    detail: "Each short burst is read and shown right away",
  },
  normal: {
    label: "Normal",
    where: "Local",
    maxSeconds: 6,
    drafts: true,
    final: "local",
    detail: "Drafts while you talk, then the sentence is reread on this device",
  },
  quality: {
    label: "Quality",
    where: "Cloud",
    maxSeconds: 20,
    drafts: true,
    final: "cloud",
    detail: "Drafts on this device; the final read runs on the GPU server",
  },
}

export const LIP_MODE_ORDER: readonly LipMode[] = ["instant", "normal", "quality"]
const STORAGE_KEY = "lipread.mode"

export function readStoredMode(): LipMode {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    if (value && value in LIP_MODES) return value as LipMode
  } catch {
    // storage blocked (private mode, sandboxed preview): use the default
  }
  return "normal"
}

export function storeMode(mode: LipMode): void {
  try {
    localStorage.setItem(STORAGE_KEY, mode)
  } catch {
    // not remembered; fine
  }
}

// --- Pause timing (plan D70: extends the teammate's lip-motion detector) ---
/** Lips still this long mid-sentence: read the new piece as a draft. */
export const SHORT_PAUSE_MS = 300
/** Lips still this long: the sentence is over; lock it and run the final read. */
export const LONG_PAUSE_MS = 800
