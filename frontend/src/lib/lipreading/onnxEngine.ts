import * as ort from "onnxruntime-web"

import { greedyCtcDecode } from "./ctc"
import type { LipFrame, LipModelSpec, LipReaderEngine, LipReaderResult } from "./types"

// Serve the ORT wasm/webgpu binaries locally (copied into /public/ort).
ort.env.wasm.wasmPaths = "/ort/"

/**
 * Real recognition backend backed by an .onnx model via onnxruntime-web.
 *
 * Tries the WebGPU execution provider first, falls back to WASM. Builds an
 * [1, T, H, W, C] input tensor from the mouth-crop window (the LipNet layout);
 * if the team's model expects a different layout, adjust `buildInput` only.
 */
export class OnnxLipReaderEngine implements LipReaderEngine {
  readonly name: string
  readonly isReal = true
  private readonly spec: LipModelSpec
  private session: ort.InferenceSession | null = null

  constructor(spec: LipModelSpec) {
    this.spec = spec
    this.name = `ONNX · ${spec.name}`
  }

  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.spec.modelUrl, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    })
  }

  async infer(window: readonly LipFrame[]): Promise<LipReaderResult> {
    if (!this.session) throw new Error("OnnxLipReaderEngine not initialized")

    const input = this.buildInput(window)
    const feeds: Record<string, ort.Tensor> = {
      [this.session.inputNames[0]]: input,
    }
    const output = await this.session.run(feeds)
    const logitsTensor = output[this.session.outputNames[0]]
    const logits = logitsTensor.data as Float32Array

    const vocab = this.spec.charset.length
    const timesteps = Math.floor(logits.length / vocab)
    const text = greedyCtcDecode(logits, timesteps, this.spec)
    return { text }
  }

  /** Stack the window into a single [1, T, H, W, C] Float32 tensor. */
  private buildInput(window: readonly LipFrame[]): ort.Tensor {
    const { windowFrames, cropHeight, cropWidth, channels } = this.spec
    const frameSize = cropHeight * cropWidth * channels
    const data = new Float32Array(windowFrames * frameSize)

    for (let t = 0; t < windowFrames; t++) {
      // Pad short windows by repeating the last frame.
      const frame = window[t] ?? window[window.length - 1]
      if (frame) data.set(frame.data, t * frameSize)
    }

    return new ort.Tensor("float32", data, [
      1,
      windowFrames,
      cropHeight,
      cropWidth,
      channels,
    ])
  }

  dispose(): void {
    this.session?.release()
    this.session = null
  }
}
