import { MockLipReaderEngine } from "./mockEngine"
import { ACTIVE_SPEC } from "./modelSpec"
import { OnnxLipReaderEngine } from "./onnxEngine"
import type { LipModelSpec, LipReaderEngine } from "./types"

/**
 * Pick the recognition engine: use the real ONNX model if one is actually
 * served at `spec.modelUrl`, otherwise fall back to the mock so the app still
 * runs. This is the single seam the model team integrates against — drop a
 * `lipreader.onnx` into /public/models and this returns the real engine.
 */
export async function createLipReaderEngine(
  spec: LipModelSpec = ACTIVE_SPEC
): Promise<LipReaderEngine> {
  if (await modelExists(spec.modelUrl)) {
    try {
      const engine = new OnnxLipReaderEngine(spec)
      await engine.init()
      return engine
    } catch (err) {
      console.warn("[lipreading] ONNX init failed, using mock:", err)
    }
  }

  const mock = new MockLipReaderEngine()
  await mock.init()
  return mock
}

async function modelExists(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { method: "HEAD" })
    return res.ok && res.headers.get("content-type") !== "text/html"
  } catch {
    return false
  }
}
