/** A selectable TTS voice (ElevenLabs-shaped). */
export interface Voice {
  readonly id: string
  readonly name: string
  /** Short descriptor from the voice name, e.g. "Laid-Back, Casual, Resonant". */
  readonly tagline: string
  /** One-line description of the voice's character/use. */
  readonly description: string
  readonly gender: string
  readonly age: string
  readonly accent: string
  /** Intended use case, e.g. "conversational", "narration". */
  readonly useCase: string
  /** One-word vibe, e.g. "sassy", "classy". */
  readonly descriptive: string
}