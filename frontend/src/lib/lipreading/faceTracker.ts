import type { NormalizedPoint } from "./overlay"
import type { Keypoints } from "./types"

/** One frame's tracking, reported back by the worker. */
export interface TrackedFace {
  /** The capture timestamp the frame was submitted with: match it to the recorded frame. */
  readonly tMs: number
  /** BlazeFace's 4 keypoints (integer source pixels), or null when no face. */
  readonly keypoints: Keypoints | null
  /** The 40 lip points for the overlay (0..1 of the frame), empty when no face. */
  readonly lipPoints: readonly NormalizedPoint[]
  /** Inner-lip gap / mouth width: ~0 closed, ~0.3+ open. Null without lip tracking. */
  readonly openness: number | null
  /** Worker time for both trackers on this frame. */
  readonly detectMs: number
}

export type TrackerRequest =
  | { type: "init" }
  | { type: "frame"; tMs: number; bitmap: ImageBitmap }
  | { type: "dispose" }

export type TrackerResponse =
  | { type: "ready" }
  | { type: "error"; message: string }
  | ({ type: "result" } & TrackedFace)

/**
 * Runs both face trackers in a background worker (`faceTracker.worker.ts`) so the capture loop
 * never waits on them (plan D52). `submit()` hands a frame over only when the worker is free and
 * returns false otherwise: every camera frame still gets recorded, and frames the worker skipped
 * have no keypoints, which the crop pipeline already fills by interpolation (so under load this
 * becomes "detect every 2nd or 3rd frame" by itself). Results arrive in `onResult`.
 */
/** A frame unanswered this long is considered lost. */
const STALL_MS = 1000

export class FaceTracker {
  onResult: ((face: TrackedFace) => void) | null = null
  private worker: Worker | null = null
  private busy = false
  private busySince = 0
  private ready = false

  async init(): Promise<void> {
    const worker = new Worker(new URL("./faceTracker.worker.ts", import.meta.url), {
      type: "module",
    })
    this.worker = worker
    await new Promise<void>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<TrackerResponse>) => {
        const msg = event.data
        if (msg.type === "ready") {
          this.ready = true
          resolve()
        } else if (msg.type === "error") {
          reject(new Error(`face tracker worker: ${msg.message}`))
        } else {
          this.busy = false
          this.onResult?.(msg)
        }
      }
      worker.onerror = (event) => reject(new Error(`face tracker worker: ${event.message}`))
      this.post({ type: "init" })
    })
  }

  /** Track this frame if the worker is idle; false = skipped (worker busy or not ready). */
  submit(source: HTMLVideoElement | HTMLCanvasElement, tMs: number): boolean {
    if (!this.ready || !this.worker) return false
    // A frame the worker never answered (crash, lost message) would otherwise leave it "busy"
    // forever and freeze the lip dots; after STALL_MS, give up on it and send a fresh one.
    if (this.busy && performance.now() - this.busySince < STALL_MS) return false
    this.busy = true
    this.busySince = performance.now()
    createImageBitmap(source).then(
      (bitmap) => {
        if (this.worker) this.post({ type: "frame", tMs, bitmap }, [bitmap])
        else bitmap.close()
      },
      () => {
        this.busy = false // e.g. the video had no frame yet
      }
    )
    return true
  }

  dispose(): void {
    if (this.worker) this.post({ type: "dispose" })
    this.worker?.terminate()
    this.worker = null
    this.ready = false
  }

  private post(message: TrackerRequest, transfer: Transferable[] = []): void {
    this.worker?.postMessage(message, transfer)
  }
}
