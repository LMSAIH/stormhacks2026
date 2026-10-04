/**
 * Real-model golden test: OnnxRecognizer (onnxruntime-web, WASM EP, in Node) on an LRS3 clip's
 * pre-made crops, with the int8 model the app ships (`lipread_ctc.int8.onnx` = the quantized
 * `ml/artifacts/lipread_ctc.dyn-pw8-rn16.onnx`), must produce exactly the text that *native*
 * onnxruntime (Python, CPU EP) greedy-decodes from the same file on the same crops. That reference
 * is the `quantized` block of the gitignored `expected.json`, which records the model's sha256; the
 * test fails early when the model it loads is not that file. The text must match exactly; the
 * confidence only to within 0.01, because the int8 kernels of ORT-web's WASM build and of native
 * onnxruntime round differently. (The same clip through PyTorch fp32 gives the same text,
 * `expected.text`; it is logged for comparison, not asserted.)
 *
 *   uv run --directory ml python ../frontend/src/lib/lipreading/__fixtures__/engine/make_fixture.py
 *       # crops.bin + expected.json; `--quantized-only` re-derives just the `quantized` block
 *   ml/scripts/publish_frontend_model.sh        # or set LIPREAD_ONNX_PATH
 *   LIPREAD_ONNX_TEST=1 pnpm exec vitest run src/lib/lipreading/onnxRecognizer.golden.test.ts
 *
 * Skipped unless LIPREAD_ONNX_TEST=1: needs the 203 MB model and the gitignored LRS3 fixture.
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
interface NodeCrypto {
  createHash(algorithm: "sha256"): {
    update(data: Uint8Array): { digest(encoding: "hex"): string }
  }
}
/** What native onnxruntime gives for the quantized model (make_fixture.py `quantized_golden`). */
interface QuantizedExpected {
  model: string
  sha256: string
  onnxruntime: string
  text: string
  confidence: number
}
interface Expected {
  frames: number
  height: number
  width: number
  /** PyTorch fp32 greedy decode (informational for this test). */
  text: string
  confidence: number
  label: string
  quantized?: QuantizedExpected
}

const env =
  (globalThis as { process?: { env: Record<string, string | undefined> } })
    .process?.env ?? {}
const here = (path: string) => new URL(path, import.meta.url)
const FIXTURE = "./__fixtures__/engine/"

describe.skipIf(env.LIPREAD_ONNX_TEST !== "1")(
  "OnnxRecognizer golden (real int8 model, WASM)",
  () => {
    it(
      "decodes an LRS3 clip exactly like native onnxruntime on the same int8 model",
      { timeout: 600_000 },
      async () => {
        const fsModule = "node:fs" // non-literal keeps Node types out of the app tsconfig
        const cryptoModule = "node:crypto"
        const fs = (await import(/* @vite-ignore */ fsModule)) as NodeFs
        const { createHash } = (await import(
          /* @vite-ignore */ cryptoModule
        )) as NodeCrypto
        const modelPath = [
          env.LIPREAD_ONNX_PATH,
          here("../../../public/models/lipread_ctc.int8.onnx"),
          here("../../../../ml/artifacts/lipread_ctc.dyn-pw8-rn16.onnx"),
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
        const reference = expected.quantized
        if (!reference) {
          throw new Error(
            `${FIXTURE}expected.json has no \`quantized\` block: run ${FIXTURE}make_fixture.py --quantized-only`
          )
        }
        const modelBytes = fs.readFileSync(modelPath)
        expect(
          createHash("sha256").update(modelBytes).digest("hex"),
          `the model is not the file expected.json was generated for (${reference.model}): ` +
            `regenerate it with ${FIXTURE}make_fixture.py --quantized-only`
        ).toBe(reference.sha256)
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
          loadAssets: async () => ({ model: modelBytes, tokens }),
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
            `T=${expected.frames} ort-web=${JSON.stringify(first.text)} conf=${first.confidence?.toFixed(4)} | ` +
            `native int8 (ort ${reference.onnxruntime})=${JSON.stringify(reference.text)} conf=${reference.confidence.toFixed(4)} | ` +
            `pytorch fp32=${JSON.stringify(expected.text)} conf=${expected.confidence.toFixed(4)} | ` +
            `label ${JSON.stringify(expected.label)}`
        )

        expect(first.text).toBe(reference.text)
        expect(second.text).toBe(reference.text)
        // The int8 kernels of ORT-web's WASM build and of native onnxruntime round differently
        // (confidence here: 0.9316 vs 0.9301; fp32 PyTorch 0.9247), so only the text is exact.
        expect(
          Math.abs((first.confidence ?? Number.NaN) - reference.confidence)
        ).toBeLessThan(0.01)
        expect(first).toMatchObject({ mode: "speed", engine: "ONNX · wasm" })
      }
    )
  }
)
