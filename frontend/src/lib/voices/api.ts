import { apiGet, apiPut } from "@/lib/backend/client"
import { ELEVENLABS_VOICES } from "./data"
import type { Voice } from "./types"

export interface VoicesResult {
  voices: Voice[]
  /** The voice the backend currently uses for TTS, if signed in. */
  defaultVoiceId: string | null
}

interface BackendVoice {
  voice_id: string
  name?: string
  description?: string
  labels?: Record<string, string>
}

/** Map a raw ElevenLabs voice (as the backend returns them) to our Voice shape. */
function mapVoice(v: BackendVoice): Voice {
  const name = v.name ?? ""
  const dash = name.indexOf(" - ")
  const labels = v.labels ?? {}
  return {
    id: v.voice_id,
    name: (dash === -1 ? name : name.slice(0, dash)).trim(),
    tagline: dash === -1 ? "" : name.slice(dash + 3).trim(),
    description: v.description ?? "",
    gender: labels.gender ?? "",
    age: (labels.age ?? "").replace(/_/g, " "),
    accent: labels.accent ?? "",
    useCase: (labels.use_case ?? "").replace(/_/g, " "),
    descriptive: labels.descriptive ?? "",
  }
}

/**
 * Fetch selectable voices from the backend (requires sign-in). Falls back to the bundled snapshot
 * when the backend is unreachable or the user isn't authenticated, so the picker still works.
 */
export async function fetchVoices(): Promise<VoicesResult> {
  try {
    const data = await apiGet<{
      voices: BackendVoice[]
      default_voice_id: string
    }>("/api/voices")
    return {
      voices: data.voices.map(mapVoice),
      defaultVoiceId: data.default_voice_id ?? null,
    }
  } catch {
    return { voices: ELEVENLABS_VOICES, defaultVoiceId: null }
  }
}

/** Persist the backend's default TTS voice. */
export async function setDefaultVoice(voiceId: string): Promise<void> {
  await apiPut("/api/voice", { voice_id: voiceId })
}
