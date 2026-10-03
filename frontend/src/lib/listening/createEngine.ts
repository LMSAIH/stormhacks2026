import { MockListeningEngine } from "./mockEngine"
import type { ListeningEngine } from "./types"

/**
 * Returns the listening backend. Today this is always the mock diarizer; when a
 * real streaming ASR + diarization backend exists (local model or a WS to a
 * server), branch here — the UI consuming `ListeningEngine` won't change.
 */
export async function createListeningEngine(): Promise<ListeningEngine> {
  return new MockListeningEngine()
}
