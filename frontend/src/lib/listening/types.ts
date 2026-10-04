/**
 * Contracts for "listening" mode: background transcription + speaker
 * diarization of people speaking around you. As with lip reading, the UI
 * depends only on these interfaces so a real ASR+diarization backend can
 * replace the mock without UI changes.
 */

import { ACCENT_COLORS } from "@/lib/palette"

export interface Speaker {
  readonly id: string
  /** Editable display name (defaults to "Speaker 1", etc.). */
  name: string
  /** Stable accent color token (chart-1..chart-5) for the UI. */
  readonly colorVar: string
}

export interface Utterance {
  readonly id: string
  readonly speakerId: string
  /** Text so far — may grow while `final` is false. */
  text: string
  /** performance.now() timestamp (for cross-source ordering / display). */
  readonly at: number
  /** False while the ASR is still refining this utterance. */
  final: boolean
}

export interface ListeningEvents {
  /** A new speaker was detected. */
  onSpeaker(speaker: Speaker): void
  /** An utterance was created or updated (partial or final). */
  onUtterance(utterance: Utterance): void
  /** A previously-shown partial turned out to be echo/noise — remove it. */
  onDrop?(id: string): void
}

export interface ListeningEngine {
  readonly name: string
  readonly isReal: boolean
  start(events: ListeningEvents): Promise<void>
  stop(): void
}

/** Per-speaker accent colors, cycled by speaker index (shared app palette). */
export const SPEAKER_COLORS = ACCENT_COLORS
