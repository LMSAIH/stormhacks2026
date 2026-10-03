import type { LipFrame } from "./types"

/**
 * Fixed-capacity ring buffer of mouth-crop frames.
 *
 * The capture loop pushes one frame per video tick; the inference loop reads a
 * contiguous snapshot of the most recent `capacity` frames. Decoupling them this
 * way keeps the camera smooth even when inference is slow (see useLipReader).
 */
export class FrameRingBuffer {
  private readonly capacity: number
  private readonly frames: (LipFrame | undefined)[]
  private writeIndex = 0
  private count = 0

  constructor(capacity: number) {
    this.capacity = capacity
    this.frames = new Array<LipFrame | undefined>(capacity)
  }

  push(frame: LipFrame): void {
    this.frames[this.writeIndex] = frame
    this.writeIndex = (this.writeIndex + 1) % this.capacity
    this.count = Math.min(this.count + 1, this.capacity)
  }

  /** True once we have a full window ready to infer on. */
  get isFull(): boolean {
    return this.count === this.capacity
  }

  get size(): number {
    return this.count
  }

  /**
   * Oldest-to-newest snapshot of all buffered frames. Returns a fresh array so
   * the inference loop can read it without racing the capture loop.
   */
  snapshot(): LipFrame[] {
    const out: LipFrame[] = []
    const start = this.count < this.capacity ? 0 : this.writeIndex
    for (let i = 0; i < this.count; i++) {
      const frame = this.frames[(start + i) % this.capacity]
      if (frame) out.push(frame)
    }
    return out
  }

  clear(): void {
    this.frames.fill(undefined)
    this.writeIndex = 0
    this.count = 0
  }
}
