import type {
  CropResult,
  RecognitionMode,
  RecognitionResult,
  Recognizer,
} from "./types"

/** Canned phrases (uppercase, like the real model) so the UI is demoable with no model at all. */
const PHRASES = [
  "HELLO THERE",
  "CAN YOU READ MY LIPS",
  "THIS IS A DEMO",
  "LIP READING WORKS",
  "TESTING THE PIPELINE",
  "NICE TO MEET YOU",
]

/**
 * Stand-in used only when neither the local model nor the hosted service is available. It still
 * goes through capture → crop, and only "recognizes" an utterance that produced mouth crops.
 */
export class MockRecognizer implements Recognizer {
  readonly name = "Mock (no model loaded)"
  readonly available = true
  readonly isReal = false
  readonly mode: RecognitionMode
  private readonly delayMs: number
  private cursor = 0

  /** `delayMs` simulates load / inference time so loading states are visible. */
  constructor(mode: RecognitionMode = "speed", delayMs = 150) {
    this.mode = mode
    this.delayMs = delayMs
  }

  async init(): Promise<void> {
    await sleep(this.delayMs)
  }

  async recognize(
    crops: CropResult,
    signal?: AbortSignal
  ): Promise<RecognitionResult> {
    const started = performance.now()
    await sleep(this.delayMs, signal)
    const text =
      crops.patches.length > 0 ? PHRASES[this.cursor++ % PHRASES.length] : ""
    return {
      text,
      confidence: text ? 0.5 : undefined,
      mode: this.mode,
      latencyMs: performance.now() - started,
      engine: this.name,
    }
  }

  dispose(): void {
    this.cursor = 0
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason)
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    signal?.addEventListener("abort", onAbort, { once: true })
  })
}
