import type { LipFrame, LipReaderEngine, LipReaderResult } from "./types"

/** Canned phrases cycled through so the UI is demoable before the model lands. */
const PHRASES = [
  "hello there",
  "can you read my lips",
  "this is a demo",
  "lip reading works",
  "place blue at g nine now",
  "testing the pipeline",
]

/**
 * Stand-in engine used when no .onnx model is present. Returns a canned phrase,
 * but only when it actually receives mouth-crop frames — so the full capture →
 * window → infer loop is exercised exactly as it will be with the real model.
 */
export class MockLipReaderEngine implements LipReaderEngine {
  readonly name = "Mock (no model loaded)"
  readonly isReal = false
  private cursor = 0

  async init(): Promise<void> {
    // Simulate a short load so loading states are visible.
    await new Promise((r) => setTimeout(r, 150))
  }

  async infer(window: readonly LipFrame[]): Promise<LipReaderResult> {
    // Only "recognize" if a mouth was actually detected this window.
    const hasSignal = window.some((f) => f.data.some((v) => v !== 0))
    if (!hasSignal) return { text: "" }

    const phrase = PHRASES[this.cursor % PHRASES.length]
    this.cursor++
    return { text: phrase, confidence: 0.5 }
  }

  dispose(): void {
    this.cursor = 0
  }
}
