/**
 * Contracts for "listening" mode: background transcription + speaker
 * diarization of people speaking around you. As with lip reading, the UI
 * depends only on these interfaces so a real ASR+diarization backend can
 * replace the mock without UI changes.
 */

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
}

export interface ListeningEngine {
  readonly name: string
  readonly isReal: boolean
  start(events: ListeningEvents): Promise<void>
  stop(): void
}

/** chart-1..chart-5 cycled for speaker accent colors. */
export const SPEAKER_COLORS: readonly string[] = [
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-1)",
  "var(--chart-4)",
  "var(--chart-5)",
]
