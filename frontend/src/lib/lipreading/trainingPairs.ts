import { gzip } from "./httpRecognizer"
import type { CropResult } from "./types"

/** How the confirmed text was given: picked from the top 3, typed, or kept as read. */
export type PairSource = "picked" | "typed" | "accepted"

const PATCH = 96
const STORAGE_KEY = "lipread.shareClips"

/** The lip-read server that stores pairs (same as accuracy mode); "" when not configured. */
export function pairsBaseUrl(): string {
  const raw: unknown = import.meta.env.VITE_LIPREAD_URL
  return typeof raw === "string" ? raw.trim().replace(/\/+$/, "") : ""
}

/** The user's opt-in (plan D59/D61): off unless they turned it on. */
export function readShareClips(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1"
  } catch {
    return false
  }
}

export function storeShareClips(on: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, on ? "1" : "0")
  } catch {
    // not remembered; fine
  }
}

/**
 * Send one (mouth clip, confirmed text) pair to `POST /training-pairs` for fine-tuning. The clip is
 * the 96x96 aligned patches the model read (no face image beyond the mouth), gzipped.
 */
export async function sendTrainingPair(
  baseUrl: string,
  crops: CropResult,
  text: string,
  source: PairSource,
  fetchImpl: typeof fetch = fetch
): Promise<void> {
  const t = crops.patches.length
  const raw = new Uint8Array(t * PATCH * PATCH)
  crops.patches.forEach((p, i) => raw.set(p, i * PATCH * PATCH))
  const packed = await gzip(raw)
  const query = new URLSearchParams({ t: String(t), h: String(PATCH), w: String(PATCH), text, source })
  const headers: Record<string, string> = { "Content-Type": "application/octet-stream" }
  if (packed) headers["Content-Encoding"] = "gzip"
  const res = await fetchImpl(`${baseUrl}/training-pairs?${query}`, {
    method: "POST",
    headers,
    body: packed ?? raw,
  })
  if (!res.ok) throw new Error(`training pair upload: HTTP ${res.status}`)
}
