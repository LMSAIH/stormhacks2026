import { SttListeningEngine } from "./sttEngine"
import type { ListeningEngine } from "./types"

/**
 * Returns the listening backend: live ElevenLabs Scribe + diarization over the backend `/ws/stt`
 * socket. The UI depends only on `ListeningEngine`, so swapping implementations is local to here.
 */
export async function createListeningEngine(): Promise<ListeningEngine> {
  return new SttListeningEngine()
}
