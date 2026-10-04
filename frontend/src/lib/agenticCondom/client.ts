/**
 * The Agentic Condom client: one `POST {condomBaseUrl()}/correct` per final line (Normal and
 * Quality), where an LLM may fix the words the reader was unsure of before the line is spoken.
 *
 * It never throws and never waits past the mode's budget (CONDOM_BUDGET_MS, the whole round trip).
 * A timeout, a failure, an answer for another line or one that breaks rule 1 (`gate.ts`, re-checked
 * here whatever the server says) gives the line back as read. After BREAKER_FAILURES network
 * failures or timeouts in a row it rests for BREAKER_PAUSE_MS, so a dead server doesn't cost every
 * line the full budget.
 */
import type { LipMode } from "@/lib/lipreading/modes"
import type { PhraseScore } from "@/lib/lipreading/types"
import { splitWords } from "@/lib/lipreading/wordSpans"
import { lookalike } from "@/lib/phrases/lookalike"

import { conversationContext, MAX_CONTEXT, type ContextLine } from "./context"
import { applyEdits, bracketed, checkCorrection, editedWords, type CondomEdit } from "./gate"
import { condomEnabled } from "./settings"

/** The whole round trip may take this long per mode; past it the line goes out as read. */
export const CONDOM_BUDGET_MS = { normal: 500, quality: 1000 } as const
/** Shorter lines go out as read: the LLM would have to invent the rest. */
export const MIN_WORDS = 2
/** This many network failures or timeouts in a row rest the condom for BREAKER_PAUSE_MS. */
export const BREAKER_FAILURES = 2
export const BREAKER_PAUSE_MS = 30_000
/** At most one keep-alive `GET /health` per this long. */
export const WARM_EVERY_MS = 45_000
/** Phrase-memory candidates sent per line. */
export const MAX_PHRASES = 5
/** Look-alike candidates come without a score, so they must at least look this alike (0..1). */
const MIN_LOOKALIKE = 0.5
const MAX_ALTERNATIVES = 5
// The server's /correct limits (past them it answers 422).
const MAX_TEXT_CHARS = 2000
const MAX_WORDS = 200
const MAX_PHRASE_CHARS = 300
const WARM_TIMEOUT_MS = 5_000
/** Corrected lines remembered for the transcript's hover hint. */
const MAX_MARKS = 50

/**
 * corrected / unchanged: the LLM answered (and passed rule 1). skipped: nothing it may change (under
 * MIN_WORDS words, no word under CONDOM_FLAG_BELOW). rejected: the answer broke rule 1. timeout: no answer within the
 * budget. error: network failure, HTTP error, or an answer that isn't for this line. off: switched
 * off, no server, Instant mode, or no LLM on the server. offline: resting after failures (breaker).
 * aborted: the caller gave up (unmount).
 */
export type CondomStatus =
  | "corrected"
  | "unchanged"
  | "skipped"
  | "rejected"
  | "timeout"
  | "error"
  | "off"
  | "offline"
  | "aborted"

/** Statuses the server may report for a line it gave back as read. */
const SERVER_STATUSES: readonly CondomStatus[] = ["unchanged", "skipped", "rejected", "timeout", "error", "off"]

export interface CondomPhrase {
  readonly text: string
  /** The model's margin for it (`PhraseScore`); null when only ranked by look-alike. */
  readonly score: number | null
}

export interface CondomRequest {
  /** The line to correct (after phrase snapping), as read. */
  readonly text: string
  /** Per word of `text`, how sure the reader was (`confidenceFor`); null = unknown, i.e. sure. */
  readonly confidence: readonly (number | null)[]
  /** The other readings of the line. */
  readonly alternatives?: readonly string[]
  /** Phrase-memory candidates, best first (`condomPhrases`). */
  readonly phrases?: readonly CondomPhrase[]
  /** The conversation before this line, oldest first. Default: `conversationContext()`. */
  readonly context?: readonly ContextLine[]
}

export interface CondomOptions {
  /** Sets the budget; Instant never corrects. Default "normal". */
  readonly mode?: LipMode
  /** Give up (unmount): the line goes out as read. */
  readonly signal?: AbortSignal
  /** The transcript line's id: a correction is remembered for its hover hint (`condomChangedWords`). */
  readonly lineId?: string
  // Overrides (tests): the switch, the server, fetch, the clock, the budget.
  readonly enabled?: boolean
  readonly baseUrl?: string
  readonly fetch?: typeof fetch
  readonly now?: () => number
  readonly budgetMs?: number
}

export interface CondomOutcome {
  /** What to show and speak: the corrected line, or `raw` as is. */
  readonly text: string
  /** The line as sent. */
  readonly raw: string
  /** What changed, in `raw`'s word indices; empty unless corrected. */
  readonly edits: CondomEdit[]
  readonly status: CondomStatus
  readonly changed: boolean
}

const breaker = { failures: 0, restUntil: Number.NEGATIVE_INFINITY }
let lastWarm = Number.NEGATIVE_INFINITY
const marks = new Map<string, { readonly text: string; readonly words: readonly number[] }>()

const defaultNow = () => performance.now()
const trimBase = (url: unknown) => (typeof url === "string" ? url.trim().replace(/\/+$/, "") : "")

/*
 * Build-time settings (frontend/.env.local), both optional:
 * - VITE_CONDOM_URL: the condom's server when it isn't the lip-read server; else VITE_LIPREAD_URL.
 *   Neither set → the condom is off (and its menu switch disabled).
 * - VITE_CONDOM_BUDGET_MS: one budget (ms) for both modes instead of CONDOM_BUDGET_MS, e.g. for
 *   evals over a slow link.
 */

/** The server that runs the condom; "" when not configured. */
export function condomBaseUrl(): string {
  return trimBase(import.meta.env.VITE_CONDOM_URL) || trimBase(import.meta.env.VITE_LIPREAD_URL)
}

/** The round-trip budget for a mode: VITE_CONDOM_BUDGET_MS when set, else CONDOM_BUDGET_MS. */
export function condomBudgetMs(mode: "normal" | "quality"): number {
  const raw: unknown = import.meta.env.VITE_CONDOM_BUDGET_MS
  const ms = typeof raw === "string" && raw.trim() ? Number(raw) : Number.NaN
  return Number.isFinite(ms) && ms > 0 ? ms : CONDOM_BUDGET_MS[mode]
}

/** Ask the condom to fix `req.text`. Never throws: any trouble gives the line back as read. */
export async function correctLine(req: CondomRequest, opts: CondomOptions = {}): Promise<CondomOutcome> {
  const raw = req.text
  const asRead = (status: CondomStatus): CondomOutcome => ({ text: raw, raw, edits: [], status, changed: false })
  try {
    const mode = opts.mode ?? "normal"
    const base = trimBase(opts.baseUrl ?? condomBaseUrl())
    if (mode === "instant" || !base || !(opts.enabled ?? condomEnabled())) return asRead("off")
    const tokens = splitWords(raw)
    const confidence = tokens.map((_, i) => finiteOrNull(req.confidence[i]))
    if (
      tokens.length < MIN_WORDS ||
      tokens.length > MAX_WORDS ||
      raw.length > MAX_TEXT_CHARS ||
      !confidence.some((c) => bracketed(c))
    )
      return asRead("skipped")
    const now = opts.now ?? defaultNow
    if (now() < breaker.restUntil) return asRead("offline")
    if (opts.signal?.aborted) return asRead("aborted")

    const answer = await post(
      `${base}/correct`,
      requestBody(req, tokens, confidence, mode),
      opts.budgetMs ?? condomBudgetMs(mode),
      opts.fetch ?? fetch,
      opts.signal
    )
    if (answer.kind === "aborted") return asRead("aborted")
    if (answer.kind === "timeout" || answer.kind === "network" || (answer.kind === "http" && answer.status >= 500)) {
      breaker.failures += 1
      if (breaker.failures >= BREAKER_FAILURES) breaker.restUntil = now() + BREAKER_PAUSE_MS
      return asRead(answer.kind === "timeout" ? "timeout" : "error")
    }
    breaker.failures = 0 // it answered
    if (answer.kind !== "ok") return asRead("error")
    const reply = parseReply(answer.body)
    if (!reply || splitWords(reply.raw).join(" ") !== tokens.join(" ")) return asRead("error")
    if (reply.status !== undefined && reply.status !== "corrected") {
      const status = SERVER_STATUSES.find((s) => s === reply.status)
      return asRead(status ?? "unchanged")
    }
    const verdict = checkCorrection(raw, reply.text, confidence)
    if (!verdict.ok) return asRead("rejected")
    if (verdict.edits.length === 0) return asRead("unchanged") // case or punctuation only
    const upper = raw === raw.toUpperCase()
    const edits = verdict.edits.map((e) => ({ ...e, to: inCase(e.to, e.from, upper) }))
    const text = applyEdits(raw, edits)
    if (opts.lineId) remember(opts.lineId, text, editedWords(edits))
    return { text, raw, edits, status: "corrected", changed: true }
  } catch {
    return asRead("error")
  }
}

/**
 * Phrase-memory candidates for a line, best first: the model's own ranking when it scored the saved
 * phrases (`scored`, CTC margins), else `own` by look-alike to the reading, without a score.
 */
export function condomPhrases(
  reading: string,
  scored: readonly PhraseScore[] | null | undefined,
  own: readonly string[]
): CondomPhrase[] {
  const fits = (text: string) => text.trim() !== "" && text.length <= MAX_PHRASE_CHARS
  if (scored)
    return scored
      .filter((p) => fits(p.text) && Number.isFinite(p.margin))
      .sort((a, b) => b.margin - a.margin)
      .slice(0, MAX_PHRASES)
      .map((p) => ({ text: p.text, score: p.margin }))
  return own
    .filter(fits)
    .map((text) => ({ text, alike: lookalike(reading, text) }))
    .filter((p) => p.alike >= MIN_LOOKALIKE)
    .sort((a, b) => b.alike - a.alike)
    .slice(0, MAX_PHRASES)
    .map((p) => ({ text: p.text, score: null }))
}

/**
 * Keep the connection to the condom's server warm, so a line's POST doesn't pay for a new TLS
 * connection: a cheap `GET /health` (a CORS simple request), at most once per WARM_EVERY_MS.
 * Fire and forget; never throws.
 */
export function warmCondom(opts: Pick<CondomOptions, "enabled" | "baseUrl" | "fetch" | "now"> = {}): void {
  try {
    const base = trimBase(opts.baseUrl ?? condomBaseUrl())
    if (!base || !(opts.enabled ?? condomEnabled())) return
    const t = (opts.now ?? defaultNow)()
    if (t - lastWarm < WARM_EVERY_MS) return
    lastWarm = t
    const fetchImpl = opts.fetch ?? fetch
    void fetchImpl(`${base}/health`, { signal: AbortSignal.timeout(WARM_TIMEOUT_MS) })
      .then((res) => res.text())
      .catch(() => undefined)
  } catch {
    // best effort
  }
}

/** Word indices of a transcript line that the condom put there; [] once the line was edited. */
export function condomChangedWords(lineId: string, text: string): readonly number[] {
  const mark = marks.get(lineId)
  return mark && mark.text.toLowerCase() === text.toLowerCase() ? mark.words : []
}

/** Forget the breaker, the warm-up clock and the remembered corrections (tests). */
export function resetCondomClient(): void {
  breaker.failures = 0
  breaker.restUntil = Number.NEGATIVE_INFINITY
  lastWarm = Number.NEGATIVE_INFINITY
  marks.clear()
}

function remember(lineId: string, text: string, words: readonly number[]): void {
  marks.set(lineId, { text, words })
  while (marks.size > MAX_MARKS) marks.delete(marks.keys().next().value as string)
}

function requestBody(
  req: CondomRequest,
  tokens: readonly string[],
  confidence: readonly (number | null)[],
  mode: "normal" | "quality"
): string {
  const seen = new Set([tokens.join(" ").toLowerCase()])
  const alternatives = (req.alternatives ?? [])
    .map((a) => splitWords(a).join(" "))
    .filter((a) => {
      const key = a.toLowerCase()
      if (!a || seen.has(key)) return false
      seen.add(key)
      return true
    })
    .slice(0, MAX_ALTERNATIVES)
  return JSON.stringify({
    text: req.text,
    words: tokens.map((text, i) => ({ text, confidence: confidence[i] })),
    alternatives,
    phrases: (req.phrases ?? []).filter((p) => p.text.length <= MAX_PHRASE_CHARS).slice(0, MAX_PHRASES),
    context: (req.context ?? conversationContext()).slice(-MAX_CONTEXT),
    mode,
  })
}

type Answer =
  | { readonly kind: "ok"; readonly body: unknown }
  | { readonly kind: "http"; readonly status: number }
  | { readonly kind: "malformed" | "network" | "timeout" | "aborted" }

/**
 * POST the JSON as text/plain: a CORS simple request, so no preflight round trip inside the
 * budget. Settles by the deadline even if fetch ignores its signal, then cancels the request.
 */
async function post(
  url: string,
  body: string,
  budgetMs: number,
  fetchImpl: typeof fetch,
  signal?: AbortSignal
): Promise<Answer> {
  const ctrl = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<Answer>((resolve) => {
    timer = setTimeout(() => resolve({ kind: "timeout" }), budgetMs)
  })
  let giveUp!: () => void
  const aborted = new Promise<Answer>((resolve) => {
    giveUp = () => resolve({ kind: "aborted" })
  })
  signal?.addEventListener("abort", giveUp, { once: true })
  const attempt = (async (): Promise<Answer> => {
    let res: Response
    try {
      res = await fetchImpl(url, {
        method: "POST",
        headers: { "Content-Type": "text/plain;charset=UTF-8" },
        body,
        signal: ctrl.signal,
      })
    } catch {
      return { kind: "network" }
    }
    if (!res.ok) return { kind: "http", status: res.status }
    try {
      return { kind: "ok", body: (await res.json()) as unknown }
    } catch {
      return { kind: "malformed" }
    }
  })()
  try {
    return await Promise.race([attempt, deadline, aborted])
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener("abort", giveUp)
    ctrl.abort() // whatever is still in flight
  }
}

function parseReply(body: unknown): { text: string; raw: string; status?: string } | null {
  if (!isRecord(body) || typeof body.text !== "string" || typeof body.raw !== "string") return null
  return { text: body.text, raw: body.raw, status: typeof body.status === "string" ? body.status : undefined }
}

/**
 * A replacement in the line's case: capitals in a line read in capitals (so toSentenceCase still
 * applies), else lowercase unless the word it replaces was capitalised. Case only, never the words.
 */
function inCase(to: string, from: string, upperLine: boolean): string {
  if (upperLine) return to.toUpperCase()
  const word = to === to.toUpperCase() ? to.toLowerCase().replace(/\bi\b/g, "I") : to
  return /^[A-Z]/.test(from) ? word.charAt(0).toUpperCase() + word.slice(1) : word
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v)
}
