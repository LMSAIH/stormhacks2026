import type {
  Alternative,
  WordConfidence,
  CropResult,
  RecognitionResult,
  Recognizer,
} from "./types"

const PATCH = 96
// Only the centre 88×88 reaches the model (the server's CenterCrop(88) is a no-op on 88 px), so
// sending it is bit-identical server-side and 16% less upload.
const SENT = 88
const OFFSET = (PATCH - SENT) / 2
const HEALTH_TIMEOUT_MS = 5_000
const REQUEST_TIMEOUT_MS = 10_000
// Beam + LM time grows with the sentence: measured from a laptop, p95 ~5 s for a 2 s clip and
// ~22 s for 20 s (.context/streaming-length-table.md), so allow 1 s more per second of video.
const REQUEST_MS_PER_VIDEO_SECOND = 1_000
const FPS = 25

export type HttpRecognizerErrorKind =
  "not_configured" | "network" | "timeout" | "http" | "bad_response"

/** Accuracy mode failed; the caller falls back to speed mode (design §1). */
export class HttpRecognizerError extends Error {
  readonly kind: HttpRecognizerErrorKind
  readonly status?: number

  constructor(kind: HttpRecognizerErrorKind, message: string, status?: number) {
    super(message)
    this.name = "HttpRecognizerError"
    this.kind = kind
    this.status = status
  }
}

export interface HttpRecognizerOptions {
  /** Service base URL. Default `VITE_LIPREAD_URL`; unset/empty → not configured. */
  readonly baseUrl?: string
  readonly healthTimeoutMs?: number
  /** Base request timeout; `requestMsPerSecond` is added per second of video. */
  readonly requestTimeoutMs?: number
  readonly requestMsPerSecond?: number
  /** Server-side decode; beam 40 + LM is what accuracy mode is for. */
  readonly decode?: "beam" | "greedy"
}

/** The parts of the `POST /lipread/crops` response we use (same JSON as `/lipread`, brief §5). */
interface LipreadResponse {
  text: string
  raw_text?: string
  confidence?: number
  latency_ms?: Record<string, unknown>
  alternatives: Alternative[]
  words: WordConfidence[]
}

/**
 * "accuracy" mode: ships the locally-cropped mouth patches (centre 88×88, raw uint8, gzipped) to the
 * hosted service, which runs beam search + LM on a GPU. `available` reflects the last health probe
 * or request; call `init()` again to re-probe.
 */
export class HttpRecognizer implements Recognizer {
  readonly mode = "accuracy" as const
  readonly isReal = true
  private readonly baseUrl: string
  private readonly healthTimeoutMs: number
  private readonly requestTimeoutMs: number
  private readonly requestMsPerSecond: number
  private readonly decode: "beam" | "greedy"
  private ok = false
  private device: string | null = null
  private status: string

  constructor(options: HttpRecognizerOptions = {}) {
    const raw: unknown = options.baseUrl ?? import.meta.env.VITE_LIPREAD_URL
    this.baseUrl = typeof raw === "string" ? raw.trim().replace(/\/+$/, "") : ""
    this.healthTimeoutMs = options.healthTimeoutMs ?? HEALTH_TIMEOUT_MS
    this.requestTimeoutMs = options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS
    this.requestMsPerSecond = options.requestMsPerSecond ?? REQUEST_MS_PER_VIDEO_SECOND
    this.decode = options.decode ?? "beam"
    this.status = this.baseUrl ? "not checked" : "not configured"
  }

  get name(): string {
    if (!this.ok) return `RunPod · ${this.status}`
    return `RunPod · ${this.decode}${this.device ? ` (${this.device})` : ""}`
  }

  get available(): boolean {
    return this.ok
  }

  /** GET /health. Never throws: an offline or unconfigured service just stays unavailable. */
  async init(): Promise<void> {
    if (!this.baseUrl) return
    try {
      const res = await this.request(
        `${this.baseUrl}/health`,
        {},
        this.healthTimeoutMs
      )
      const body: unknown = res.ok ? await res.json().catch(() => null) : null
      const health = isRecord(body) ? body : null
      if (
        !res.ok ||
        !health ||
        (health.status !== undefined && health.status !== "ok")
      ) {
        this.markDown(`offline (HTTP ${res.status})`)
        return
      }
      this.device = typeof health.device === "string" ? health.device : null
      this.ok = true
    } catch (err) {
      this.markDown(
        err instanceof HttpRecognizerError && err.kind === "timeout"
          ? "timed out"
          : "offline"
      )
    }
  }

  async recognize(
    crops: CropResult,
    signal?: AbortSignal
  ): Promise<RecognitionResult> {
    signal?.throwIfAborted()
    if (!this.baseUrl) {
      throw new HttpRecognizerError(
        "not_configured",
        "VITE_LIPREAD_URL is not set"
      )
    }
    const started = performance.now()
    const frames = crops.patches.length
    const raw = packPatches(crops.patches)
    const gz = await gzip(raw)
    const query = new URLSearchParams({
      t: String(frames),
      h: String(SENT),
      w: String(SENT),
      decode: this.decode,
      correct: "false",
    })
    const headers: Record<string, string> = {
      "Content-Type": "application/octet-stream",
    }
    if (gz) headers["Content-Encoding"] = "gzip"

    let body: LipreadResponse
    try {
      const json = await this.request(
        `${this.baseUrl}/lipread/crops?${query}`,
        { method: "POST", headers, body: gz ?? raw },
        this.requestTimeoutMs + (this.requestMsPerSecond * frames) / FPS,
        signal,
        async (r): Promise<unknown> => {
          if (!r.ok) throw await httpError(r)
          return r.json()
        }
      )
      body = parseResponse(json)
    } catch (err) {
      if (!signal?.aborted && !answered(err)) this.markDown("offline")
      throw err
    }
    this.ok = true

    return {
      text: body.raw_text ?? body.text, // corrector is off: raw VSR output
      confidence: body.confidence,
      alternatives: body.alternatives,
      words: body.words,
      mode: this.mode,
      latencyMs: performance.now() - started,
      serverLatencyMs: numericEntries(body.latency_ms),
      engine: this.name,
    }
  }

  dispose(): void {
    this.ok = false
  }

  private markDown(status: string): void {
    this.ok = false
    this.status = status
  }

  /**
   * fetch with a timeout that also honours the caller's signal. `read` consumes the body inside
   * the timeout window. Caller aborts surface as the signal's reason (an AbortError).
   */
  private async request<T = Response>(
    url: string,
    init: RequestInit,
    timeoutMs: number,
    outer?: AbortSignal,
    read?: (res: Response) => Promise<T>
  ): Promise<T> {
    outer?.throwIfAborted()
    const ctrl = new AbortController()
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      ctrl.abort()
    }, timeoutMs)
    const forward = () => ctrl.abort(outer?.reason)
    outer?.addEventListener("abort", forward, { once: true })
    try {
      const res = await fetch(url, { ...init, signal: ctrl.signal })
      return read ? await read(res) : (res as T)
    } catch (err) {
      if (outer?.aborted) throw outer.reason ?? err
      if (timedOut) {
        throw new HttpRecognizerError(
          "timeout",
          `no response from ${url} within ${timeoutMs / 1000} s`
        )
      }
      if (err instanceof HttpRecognizerError) throw err
      throw new HttpRecognizerError(
        "network",
        `request to ${url} failed: ${messageOf(err)}`
      )
    } finally {
      clearTimeout(timer)
      outer?.removeEventListener("abort", forward)
    }
  }
}

/** Centre 88×88 of each 96×96 patch, back to back: the `t*88*88` uint8 body `/lipread/crops` expects. */
function packPatches(patches: readonly Uint8Array[]): Uint8Array<ArrayBuffer> {
  const size = PATCH * PATCH
  const sent = SENT * SENT
  if (patches.length === 0) throw new Error("no frames to send")
  const out = new Uint8Array(patches.length * sent)
  patches.forEach((p, i) => {
    if (p.length !== size)
      throw new Error(`patch ${i} has ${p.length} bytes, expected ${size}`)
    for (let y = 0; y < SENT; y++) {
      const src = (y + OFFSET) * PATCH + OFFSET
      out.set(p.subarray(src, src + SENT), i * sent + y * SENT)
    }
  })
  return out
}

/** gzip via CompressionStream (~2× smaller for mouth crops); null where unsupported. */
export async function gzip(
  bytes: Uint8Array<ArrayBuffer>
): Promise<Uint8Array<ArrayBuffer> | null> {
  if (typeof CompressionStream === "undefined") return null
  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new CompressionStream("gzip"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

async function httpError(res: Response): Promise<HttpRecognizerError> {
  const body: unknown = await res.json().catch(() => null)
  const detail = isRecord(body) ? body.detail : undefined
  let reason = res.statusText || "request failed"
  if (typeof detail === "string") reason = detail
  else if (isRecord(detail) && typeof detail.error === "string")
    reason = detail.error
  else if (
    Array.isArray(detail) &&
    isRecord(detail[0]) &&
    typeof detail[0].msg === "string"
  ) {
    reason = detail[0].msg
  }
  return new HttpRecognizerError(
    "http",
    `lip-read service: ${reason} (HTTP ${res.status})`,
    res.status
  )
}

function parseResponse(body: unknown): LipreadResponse {
  if (!isRecord(body) || typeof body.text !== "string") {
    throw new HttpRecognizerError(
      "bad_response",
      "lip-read service returned an unexpected body"
    )
  }
  return {
    text: body.text,
    raw_text: typeof body.raw_text === "string" ? body.raw_text : undefined,
    confidence:
      typeof body.confidence === "number" ? body.confidence : undefined, // null for beam
    latency_ms: isRecord(body.latency_ms) ? body.latency_ms : undefined,
    alternatives: Array.isArray(body.alternatives)
      ? body.alternatives.filter(
          (a): a is Alternative =>
            isRecord(a) && typeof a.text === "string" && typeof a.score === "number"
        )
      : [],
    words: Array.isArray(body.words)
      ? body.words.filter(
          (w): w is WordConfidence =>
            isRecord(w) && typeof w.text === "string" && typeof w.confidence === "number"
        )
      : [],
  }
}

/** The service responded (4xx such as clip_too_short, or a malformed 200) — it is still up. */
function answered(err: unknown): boolean {
  if (!(err instanceof HttpRecognizerError)) return false
  return (
    err.kind === "bad_response" ||
    (err.kind === "http" && (err.status ?? 500) < 500)
  )
}

function numericEntries(
  rec?: Record<string, unknown>
): Record<string, number> | undefined {
  if (!rec) return undefined
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(rec))
    if (typeof v === "number") out[k] = v
  return out
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v)
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
