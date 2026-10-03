import { beforeEach, describe, expect, it, vi } from "vitest"

import { createRecognizers, startRecognizers } from "./createRecognizers"

const backend = vi.hoisted(() => ({
  onnxLoads: true,
  httpUp: true,
  onnxDisposed: 0,
  /** When set, the local model's init() waits for it (a slow 775 MB load). */
  onnxGate: null as Promise<void> | null,
}))

vi.mock("./onnxRecognizer", () => ({
  OnnxRecognizer: class {
    readonly mode = "speed"
    readonly isReal = true
    available = false
    name = "ONNX · not loaded"
    async init() {
      if (backend.onnxGate) await backend.onnxGate
      if (!backend.onnxLoads) {
        this.name = "ONNX · model not found"
        throw new Error("model missing")
      }
      this.available = true
      this.name = "ONNX · wasm"
    }
    recognize() {
      return Promise.reject(new Error("not used"))
    }
    dispose() {
      backend.onnxDisposed++
    }
  },
}))

vi.mock("./httpRecognizer", () => ({
  HttpRecognizer: class {
    readonly mode = "accuracy"
    readonly isReal = true
    available = false
    name = "RunPod · not configured"
    async init() {
      this.available = backend.httpUp
      if (backend.httpUp) this.name = "RunPod · beam (cuda)"
    }
    recognize() {
      return Promise.reject(new Error("not used"))
    }
    dispose() {}
  },
}))

beforeEach(() => {
  Object.assign(backend, {
    onnxLoads: true,
    httpUp: true,
    onnxDisposed: 0,
    onnxGate: null,
  })
  vi.spyOn(console, "warn").mockImplementation(() => {})
})

describe("createRecognizers", () => {
  it("uses the local model for speed and the service for accuracy", async () => {
    const { speed, accuracy } = await createRecognizers()
    expect(speed).toMatchObject({
      mode: "speed",
      name: "ONNX · wasm",
      available: true,
      isReal: true,
    })
    expect(accuracy).toMatchObject({ mode: "accuracy", available: true })
  })

  it("keeps speed as an unavailable ONNX engine (not a mock) while accuracy works", async () => {
    backend.onnxLoads = false
    const { speed, accuracy } = await createRecognizers()
    expect(speed).toMatchObject({
      name: "ONNX · model not found",
      available: false,
      isReal: true,
    })
    expect(accuracy.available).toBe(true)
  })

  it("keeps the local model when the service is down", async () => {
    backend.httpUp = false
    const { speed, accuracy } = await createRecognizers()
    expect(speed).toMatchObject({ available: true, isReal: true })
    expect(accuracy.available).toBe(false)
  })

  it("falls back to the mock only when neither backend is available", async () => {
    Object.assign(backend, { onnxLoads: false, httpUp: false })
    const { speed, accuracy } = await createRecognizers()
    expect(speed).toMatchObject({
      mode: "speed",
      available: true,
      isReal: false,
    })
    expect(accuracy.available).toBe(false)
    expect(backend.onnxDisposed).toBe(1)
  })
})

describe("startRecognizers", () => {
  it("reports accuracy ready without waiting for the local model", async () => {
    let releaseModel = () => {}
    backend.onnxGate = new Promise<void>((resolve) => {
      releaseModel = resolve
    })
    const load = startRecognizers()
    let speedSettled = false
    void load.ready.speed.then(() => {
      speedSettled = true
    })

    await load.ready.accuracy
    expect(load.engines.accuracy.available).toBe(true)
    expect(speedSettled).toBe(false)
    expect(load.engines.speed.available).toBe(false)

    releaseModel()
    const { speed, accuracy } = await load.done
    expect(speed).toBe(load.engines.speed)
    expect(speed.available).toBe(true)
    expect(accuracy).toBe(load.engines.accuracy)
  })

  it("never rejects ready or done when the local model fails", async () => {
    backend.onnxLoads = false
    const load = startRecognizers()
    await expect(load.ready.speed).resolves.toBeUndefined()
    await expect(load.done).resolves.toMatchObject({
      speed: { available: false },
      accuracy: { available: true },
    })
  })
})
