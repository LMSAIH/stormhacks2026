/**
 * Real-model golden test: OnnxRecognizer (onnxruntime-web, WASM EP, in Node) on an LRS3 clip's
 * pre-made crops must produce exactly the text PyTorch's greedy decode gives.
 *
 *   uv run --directory ml python ../frontend/src/lib/lipreading/__fixtures__/engine/make_fixture.py
 *   ml/scripts/publish_frontend_model.sh        # or set LIPREAD_ONNX_PATH
 *   LIPREAD_ONNX_TEST=1 pnpm exec vitest run src/lib/lipreading/onnxRecognizer.golden.test.ts
 *
 * Skipped unless LIPREAD_ONNX_TEST=1: needs the 775 MB model and the gitignored LRS3 fixture.
 */
import { describe, expect, it, vi } from "vitest"

import { toModelInput } from "@/lib/lipreading/crop"

import { OnnxRecognizer } from "./onnxRecognizer"
import type { CropResult } from "./types"

/** Python `to_model_input`: centre-crop 88, /255, (x − 0.421) / 0.165, laid out [1, 1, T, 88, 88]. */
function referenceModelInput(crops: CropResult): {
  data: Float32Array
  dims: [1, 1, number, 88, 88]
} {
  const size = 88
  const offset = 4
  const frames = crops.patches.length
  const data = new Float32Array(frames * size * size)
  crops.patches.forEach((patch, t) => {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const v = patch[(y + offset) * 96 + x + offset]
        data[(t * size + y) * size + x] = (v / 255 - 0.421) / 0.165
      }
    }
  })
  return { data, dims: [1, 1, frames, size, size] }
}

// Use the crop agent's toModelInput when it exists, else the reference above.
const cropImpl = vi.hoisted(() => ({ real: true }))
vi.mock("@/lib/lipreading/crop", async (importOriginal) => {
  try {
    return await importOriginal<typeof import("@/lib/lipreading/crop")>()
  } catch {
    cropImpl.real = false
    return { toModelInput: referenceModelInput }
  }
})

interface NodeFs {
  existsSync(path: URL | string): boolean
  readFileSync(path: URL | string): Uint8Array
}
interface Expected {
  frames: number
  height: number
  width: number
  text: string
  confidence: number
  label: string
  onnx_text: string
}

const env =
  (globalThis as { process?: { env: Record<string, string | undefined> } })
    .process?.env ?? {}
const here = (path: string) => new URL(path, import.meta.url)
const FIXTURE = "./__fixtures__/engine/"

describe.skipIf(env.LIPREAD_ONNX_TEST !== "1")(
  "OnnxRecognizer golden (real model, WASM)",
  () => {
    it(
      "decodes an LRS3 clip exactly like PyTorch greedy",
      { timeout: 600_000 },
      async () => {
        const fsModule = "node:fs" // non-literal keeps Node types out of the app tsconfig
        const fs = (await import(/* @vite-ignore */ fsModule)) as NodeFs
        const modelPath = [
          env.LIPREAD_ONNX_PATH,
          here("../../../public/models/lipread_ctc.onnx"),
          here("../../../../ml/artifacts/lipread_ctc.onnx"),
        ].find((p) => p !== undefined && fs.existsSync(p))
        if (!modelPath)
          throw new Error(
            "model not found: run ml/scripts/publish_frontend_model.sh"
          )
        const tokensPath = new URL(
          "tokens.json",
          modelPath instanceof URL ? modelPath : `file://${modelPath}`
        )
        if (!fs.existsSync(here(`${FIXTURE}crops.bin`))) {
          throw new Error(
            `fixture missing: run ${FIXTURE}make_fixture.py (see header)`
          )
        }

        const read = (p: URL) => new TextDecoder().decode(fs.readFileSync(p))
        const expected = JSON.parse(
          read(here(`${FIXTURE}expected.json`))
        ) as Expected
        const tokens = JSON.parse(read(tokensPath)) as string[]
        const raw = fs.readFileSync(here(`${FIXTURE}crops.bin`))
        const size = expected.height * expected.width
        expect(raw.length).toBe(expected.frames * size)
        const crops: CropResult = {
          patches: Array.from({ length: expected.frames }, (_, i) =>
            raw.subarray(i * size, (i + 1) * size)
          ),
          faceCoverage: 1,
          keypoints: [],
        }

        if (cropImpl.real) {
          // The crop module's toModelInput must agree with Python's to_model_input layout + maths.
          const ours = toModelInput(crops)
          const ref = referenceModelInput(crops)
          expect(ours.dims).toEqual(ref.dims)
          let maxDiff = 0
          for (let i = 0; i < ref.data.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(ours.data[i] - ref.data[i]))
          }
          expect(maxDiff).toBeLessThan(1e-5)
        }

        const rec = new OnnxRecognizer(undefined, {
          loadRuntime: () => import("onnxruntime-web"),
          wasmPaths: null,
          executionProviders: ["wasm"],
          loadAssets: async () => ({
            model: fs.readFileSync(modelPath),
            tokens,
          }),
        })
        const t0 = performance.now()
        await rec.init()
        const loadMs = performance.now() - t0
        expect(rec.available).toBe(true)
        expect(rec.name).toBe("ONNX · wasm")

        const first = await rec.recognize(crops)
        const second = await rec.recognize(crops) // warm
        rec.dispose()
        console.info(
          `[golden] toModelInput=${cropImpl.real ? "crop module" : "test reference"} ` +
            `load=${Math.round(loadMs)}ms infer=${Math.round(first.latencyMs)}/${Math.round(second.latencyMs)}ms ` +
            `T=${expected.frames} text=${JSON.stringify(first.text)} (python ${JSON.stringify(expected.text)}, ` +
            `label ${JSON.stringify(expected.label)}) conf=${first.confidence?.toFixed(4)}/${expected.confidence.toFixed(4)}`
        )

        expect(first.text).toBe(expected.text)
        expect(second.text).toBe(expected.text)
        expect(first.confidence).toBeCloseTo(expected.confidence, 3)
        expect(first).toMatchObject({ mode: "speed", engine: "ONNX · wasm" })
      }
    )
  }
)
