import { HttpRecognizer } from "./httpRecognizer"
import { MockRecognizer } from "./mockRecognizer"
import { ACTIVE_SPEC } from "./modelSpec"
import { OnnxRecognizer } from "./onnxRecognizer"
import type { LipModelSpec, RecognitionMode, Recognizer } from "./types"

export interface Recognizers {
  readonly speed: Recognizer
  readonly accuracy: Recognizer
}

/** Both backends initialising independently — see {@link startRecognizers}. */
export interface RecognizersLoading {
  /** The engines being initialised. `done` may still swap `speed` for a mock. */
  readonly engines: Recognizers
  /** Settles (never rejects) once that engine's init() has, i.e. its `available` is known. */
  readonly ready: Readonly<Record<RecognitionMode, Promise<void>>>
  /** The final pair — what {@link createRecognizers} resolves to. */
  readonly done: Promise<Recognizers>
}

/**
 * Build both recognition backends and start initialising them in parallel (design §1):
 * - `accuracy` — HttpRecognizer; `available=false` when VITE_LIPREAD_URL is unset or the service
 *   is down (call its `init()` again to re-probe). Ready after one /health round trip.
 * - `speed` — OnnxRecognizer. If the model can't load (file not published, ORT init failed) it
 *   stays as an unavailable OnnxRecognizer while `accuracy` works — so the UI can say so and an
 *   accuracy→speed fallback never produces fake text — and becomes a MockRecognizer only when
 *   neither backend is available. Ready once the ~775 MB model is downloaded and a session warmed
 *   up, which can take minutes on a cold cache.
 *
 * Per-engine `ready` lets callers use accuracy mode without waiting for the local model.
 */
export function startRecognizers(
  spec: LipModelSpec = ACTIVE_SPEC
): RecognizersLoading {
  const speed = new OnnxRecognizer(spec)
  const accuracy = new HttpRecognizer()
  const ready = {
    speed: speed.init().catch((err: unknown) => {
      console.warn(
        `[lipreading] local model unavailable (${spec.modelUrl}):`,
        err
      )
    }),
    // HttpRecognizer.init() reports problems through `available`; never let it reject `done`.
    accuracy: accuracy.init().catch((err: unknown) => {
      console.warn("[lipreading] lip-read service check failed:", err)
    }),
  }
  const done = Promise.all([ready.speed, ready.accuracy]).then(
    async (): Promise<Recognizers> => {
      if (speed.available || accuracy.available) return { speed, accuracy }
      speed.dispose()
      const mock = new MockRecognizer("speed")
      await mock.init()
      return { speed: mock, accuracy }
    }
  )
  return { engines: { speed, accuracy }, ready, done }
}

/** Build and initialise both backends; resolves once both inits have settled. */
export function createRecognizers(
  spec: LipModelSpec = ACTIVE_SPEC
): Promise<Recognizers> {
  return startRecognizers(spec).done
}
