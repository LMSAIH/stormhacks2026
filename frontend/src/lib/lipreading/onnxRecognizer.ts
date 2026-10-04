import type * as Ort from "onnxruntime-web"

import { toModelInput } from "@/lib/lipreading/crop"

import { greedyCtcDecode } from "./ctc"
import { loadPieceScores, rankByModel, type PieceScores } from "@/lib/phrases/ctcScore"
import { ACTIVE_SPEC } from "./modelSpec"
import type {
  CropResult,
  LipModelSpec,
  RecognitionResult,
  Recognizer,
} from "./types"

export type OrtModule = typeof Ort
export type OnnxExecutionProvider = "webgpu" | "wasm"

/** The model or tokens file is not being served — run `ml/scripts/publish_frontend_model.sh`. */
export class ModelMissingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ModelMissingError"
  }
}

export interface OnnxModelAssets {
  /** URL for ORT to fetch, or the model bytes. */
  readonly model: string | Uint8Array
  readonly tokens: readonly string[]
}

export interface OnnxRecognizerOptions {
  /** Supplies onnxruntime-web; default {@link RUNTIME_FILE} from `public/`. Tests pass the npm one. */
  readonly loadRuntime?: () => Promise<OrtModule>
  /** Where ORT fetches its .wasm. Default `${BASE_URL}ort/`; null keeps ORT's default (Node). */
  readonly wasmPaths?: string | null
  /** Tried in order; an EP that can't create *and run* the model falls through to the next. */
  readonly executionProviders?: readonly OnnxExecutionProvider[]
  /** Default: `spec.modelUrl` via Cache Storage, fetch `spec.tokensUrl`. Tests read files instead. */
  readonly loadAssets?: (spec: LipModelSpec) => Promise<OnnxModelAssets>
}

/**
 * The `onnxruntime-web/webgpu` build (copied in public/ort, byte-identical to the installed
 * 1.30.0 dist). Its native WebGPU EP runs this model; the default build's JSEP WebGPU EP creates
 * the session but fails on the first 3D conv ("Unsupported padding 2,3,3,2,3,3"). Its WASM EP is
 * as fast as the plain WASM build, so one runtime serves both EPs.
 *
 * Imported at runtime rather than bundled: the threaded WASM spawns its workers from its own
 * module URL, which has to be ORT itself — not a Vite chunk that may pull in app code.
 */
export const RUNTIME_FILE = "ort/ort.webgpu.bundle.min.mjs"

const INPUT = "video"
const OUTPUT = "log_probs"
const WARMUP_DIMS = [1, 1, 8, 88, 88] as const

/**
 * "speed" mode: the Auto-AVSR encoder + CTC head (the int8 quantization, see modelSpec.ts) on
 * onnxruntime-web (multi-threaded WASM; WebGPU is opt-in), greedy CTC decode, fully on-device.
 * Input [1, 1, T, 88, 88] from the crop pipeline → `log_probs` [T, vocab].
 */
export class OnnxRecognizer implements Recognizer {
  readonly mode = "speed" as const
  readonly isReal = true
  private readonly spec: LipModelSpec
  private readonly options: OnnxRecognizerOptions
  private ort: OrtModule | null = null
  private session: Ort.InferenceSession | null = null
  private model: string | Uint8Array | null = null
  private tokens: readonly string[] = []
  private ep: OnnxExecutionProvider | null = null
  private status = "not loaded"
  private loading: Promise<void> | null = null
  private queue: Promise<unknown> = Promise.resolve()

  constructor(
    spec: LipModelSpec = ACTIVE_SPEC,
    options: OnnxRecognizerOptions = {}
  ) {
    this.spec = spec
    this.options = options
  }

  get name(): string {
    return `ONNX · ${this.ep ?? this.status}`
  }

  get available(): boolean {
    return this.session !== null
  }

  /** Execution provider of the live session (null until init succeeds). */
  get executionProvider(): OnnxExecutionProvider | null {
    return this.ep
  }

  /** Checks the model is served, loads ORT, creates + warms up a session. Throws on failure. */
  init(): Promise<void> {
    this.loading ??= this.load().catch((err: unknown) => {
      this.loading = null
      this.status =
        err instanceof ModelMissingError ? "model not found" : "failed to load"
      throw err
    })
    return this.loading
  }

  recognize(
    crops: CropResult,
    signal?: AbortSignal
  ): Promise<RecognitionResult> {
    const started = performance.now()
    // One session can't run concurrently; queue calls instead of failing them.
    const job = this.queue.then(() => this.run(crops, started, signal))
    this.queue = job.catch(() => undefined)
    return job
  }

  dispose(): void {
    const session = this.session
    this.session = null
    this.ep = null
    this.model = null
    this.loading = null
    this.status = "disposed"
    void session?.release().catch(() => undefined)
  }

  private async load(): Promise<void> {
    this.status = "loading"
    // Assets first: a missing model fails fast, before ORT and its ~27 MB .wasm are fetched.
    const assets = await (this.options.loadAssets ?? fetchAssets)(this.spec)
    const ort = await (this.options.loadRuntime ?? loadPublicRuntime)()
    const { wasmPaths = publicPath("ort/") } = this.options
    if (wasmPaths !== null) ort.env.wasm.wasmPaths = wasmPaths
    ort.env.webgpu.powerPreference = "high-performance" // hybrid laptops: the discrete GPU
    const providers = this.options.executionProviders ?? defaultExecutionProviders()
    // Run WASM inference in ORT's own worker: on the main thread a 1-2 s read blocks the camera
    // loop, so frames said during it were never recorded and the lip dots froze. (WebGPU can't
    // be proxied; tests in Node have no workers.)
    if (wasmPaths !== null && typeof Worker !== "undefined" && !providers.includes("webgpu"))
      ort.env.wasm.proxy = true
    this.ort = ort
    this.model = assets.model
    this.tokens = assets.tokens
    await this.openSession(providers)
  }

  /**
   * One EP per attempt so we know which one is live (given ["webgpu", "wasm"], ORT silently drops
   * an unavailable WebGPU backend). The CPU EP still runs any op WebGPU lacks.
   */
  private async openSession(
    eps: readonly OnnxExecutionProvider[]
  ): Promise<void> {
    const { ort, model } = this
    if (!ort || model === null)
      throw new Error("OnnxRecognizer: runtime not loaded")
    const errors: string[] = []
    for (const ep of eps) {
      if (ep === "webgpu" && !hasWebGpu()) {
        errors.push("webgpu: not supported here")
        continue
      }
      const options: Ort.InferenceSession.SessionOptions = {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
        // ORT prints warnings (e.g. shape ops placed on CPU) via console.error; keep real errors.
        logSeverityLevel: 3,
      }
      const t0 = performance.now()
      let session: Ort.InferenceSession | null = null
      try {
        session =
          typeof model === "string"
            ? await ort.InferenceSession.create(model, options)
            : await ort.InferenceSession.create(model, options)
        // A session can be created and still fail on its first run, so prove the EP with a tiny
        // inference. Also compiles WebGPU shaders before the first real utterance.
        const warmup = await this.infer(
          new Float32Array(8 * 88 * 88),
          WARMUP_DIMS,
          session
        )
        warmup.dispose()
      } catch (err) {
        void session?.release().catch(() => undefined)
        errors.push(`${ep}: ${messageOf(err)}`)
        console.warn(
          `[lipreading] ONNX on ${ep} unusable, trying the next EP:`,
          err
        )
        continue
      }
      this.session = session
      this.ep = ep
      console.info(
        `[lipreading] ONNX ready on ${ep} in ${Math.round(performance.now() - t0)} ms`
      )
      return
    }
    throw new Error(`no ONNX execution provider worked (${errors.join("; ")})`)
  }

  private async run(
    crops: CropResult,
    started: number,
    signal?: AbortSignal
  ): Promise<RecognitionResult> {
    signal?.throwIfAborted()
    if (!this.session)
      throw new Error("OnnxRecognizer is not initialised (call init())")
    const { data, dims } = toModelInput(crops)
    let logProbs: Ort.Tensor
    try {
      logProbs = await this.infer(data, dims)
    } catch (err) {
      if (this.ep !== "webgpu") throw err
      // WebGPU can still fail later (device lost, out of memory): rebuild on WASM, retry once.
      console.warn(
        "[lipreading] WebGPU inference failed, switching to WASM:",
        err
      )
      void this.session?.release().catch(() => undefined)
      this.session = null
      this.ep = null
      await this.openSession(["wasm"])
      logProbs = await this.infer(data, dims)
    }
    signal?.throwIfAborted()

    const timesteps = logProbs.dims[0]
    const { text, confidence, words } = greedyCtcDecode(
      logProbs.data as Float32Array,
      timesteps,
      this.tokens
    )
    // Copied out of the tensor (freed right after) for model-scored phrase snapping.
    const kept = new Float32Array(logProbs.data as Float32Array)
    logProbs.dispose()
    const tokens = this.tokens
    return {
      text,
      confidence,
      words,
      scorePhrases: async (reading, phrases) => {
        const sp = await pieceScoresOnce()
        return sp ? rankByModel(kept, timesteps, tokens, sp, reading, phrases) : null
      },
      mode: this.mode,
      latencyMs: performance.now() - started,
      engine: this.name,
    }
  }

  private async infer(
    data: Float32Array,
    dims: readonly number[],
    session = this.session
  ): Promise<Ort.Tensor> {
    const { ort } = this
    if (!ort || !session)
      throw new Error("OnnxRecognizer is not initialised (call init())")
    const input = session.inputNames.includes(INPUT)
      ? INPUT
      : session.inputNames[0]
    const output = session.outputNames.includes(OUTPUT)
      ? OUTPUT
      : session.outputNames[0]
    const results = await session.run({
      [input]: new ort.Tensor("float32", data, dims),
    })
    return results[output]
  }
}

let publicRuntime: Promise<OrtModule> | null = null

function loadPublicRuntime(): Promise<OrtModule> {
  publicRuntime ??= (
    import(/* @vite-ignore */ publicPath(RUNTIME_FILE)) as Promise<OrtModule>
  ).catch((err: unknown) => {
    publicRuntime = null
    throw err
  })
  return publicRuntime
}

/** Absolute URL of a file in `public/`, honouring Vite's `base`. */
function publicPath(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, document.baseURI).href
}

async function fetchAssets(spec: LipModelSpec): Promise<OnnxModelAssets> {
  const model = await loadModel(spec.modelUrl)
  const res = await fetch(spec.tokensUrl)
  if (!isServed(res)) {
    throw new ModelMissingError(
      `${spec.tokensUrl} is not being served (HTTP ${res.status})`
    )
  }
  const tokens: unknown = await res.json()
  if (
    !Array.isArray(tokens) ||
    tokens.length < 3 ||
    !tokens.every((t) => typeof t === "string")
  ) {
    throw new Error(`${spec.tokensUrl}: expected a JSON array of token strings`)
  }
  return { model, tokens }
}

/** Cache Storage bucket for model bytes; bump the suffix to drop every cached model at once. */
const MODEL_CACHE = "lipread-models-v1"

/**
 * The model bytes, from Cache Storage when this exact URL was downloaded before (the default URL
 * is pinned to a Hugging Face commit, so a URL always means the same bytes), else downloaded once
 * and stored. Without Cache Storage (old browsers, some private modes) ORT just fetches the URL.
 */
async function loadModel(url: string): Promise<string | Uint8Array> {
  const cache = await openModelCache()
  const hit = await cache?.match(url).catch(() => undefined)
  if (hit) return new Uint8Array(await hit.arrayBuffer())

  const res = await fetch(url)
  if (!isServed(res)) {
    throw new ModelMissingError(
      `${url} is not being served (HTTP ${res.status}) — check VITE_LIPREAD_MODEL_BASE, or run ` +
        "ml/scripts/publish_frontend_model.sh for a local copy"
    )
  }
  const bytes = new Uint8Array(await res.arrayBuffer())
  // A full disk / quota only costs the next page load a re-download.
  await cache
    ?.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }))
    .catch((err: unknown) => {
      console.warn("[lipreading] could not cache the model:", err)
    })
  return bytes
}

async function openModelCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(MODEL_CACHE)
  } catch {
    return null
  }
}

/** Vite and most SPA hosts answer a missing file with index.html and 200. */
function isServed(res: Response): boolean {
  return (
    res.ok && !(res.headers.get("content-type") ?? "").includes("text/html")
  )
}

/**
 * WASM by default: it is the tested path (the int8 model's MatMulInteger / DynamicQuantizeLinear
 * are CPU kernels). VITE_ORT_WEBGPU=1 opts in to WebGPU for benchmarking (Phase B). Checked with
 * the int8 model: on an Intel Xe adapter the session is created on WebGPU and reads the test clip
 * correctly; with no adapter (`navigator.gpu` present, `requestAdapter()` null) or a failing
 * session, openSession() falls through to WASM; a WebGPU failure mid-utterance rebuilds on WASM.
 */
function defaultExecutionProviders(): OnnxExecutionProvider[] {
  return import.meta.env.VITE_ORT_WEBGPU === "1" ? ["webgpu", "wasm"] : ["wasm"]
}

function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

let pieceScoresLoad: Promise<PieceScores | null> | null = null
/** The phrase scorer's piece table (112 KB), fetched on first use; null if it can't load. */
function pieceScoresOnce(): Promise<PieceScores | null> {
  pieceScoresLoad ??= loadPieceScores().catch((err: unknown) => {
    console.warn("[lipreading] phrase scorer unavailable:", err)
    return null
  })
  return pieceScoresLoad
}
