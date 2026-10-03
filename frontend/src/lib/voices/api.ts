import { ELEVENLABS_VOICES } from "./data"
import type { Voice } from "./types"

/**
 * Fetch the list of selectable voices.
 *
 * TODO: point this at our own backend once it exists, e.g.
 *   const res = await fetch("/api/voices")
 *   return res.json()
 * For now it returns a snapshot of ElevenLabs premade voices (see ./data.ts).
 * The UI depends only on this function, so wiring the real API is a one-file change.
 */
export async function fetchVoices(): Promise<Voice[]> {
  // Small delay so loading states are exercised like a real request.
  await new Promise((r) => setTimeout(r, 200))
  return ELEVENLABS_VOICES
}
