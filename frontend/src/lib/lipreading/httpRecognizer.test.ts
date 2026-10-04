import { afterEach, describe, expect, it, vi } from "vitest"

import { HttpRecognizer } from "./httpRecognizer"
import type { CropResult } from "./types"

const BASE = "https://pod.example/" // trailing slash must be stripped
const PATCH = 96 * 96
const SENT = 88 * 88 // only the centre crop is uploaded

function crops(frames: number): CropResult {
  const patches = Array.from({ length: frames }, (_, i) =>
    new Uint8Array(PATCH).fill(i + 1)
  )
  return { patches, faceCoverage: 1, keypoints: [] }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  })
}

/** A fetch that never answers, but rejects like the real one when its signal aborts. */
function hangingFetch(_url: string, init?: RequestInit): Promise<Response> {
  return new Promise((_, reject) => {
    init?.signal?.addEventListener("abort", () =>
      reject(new DOMException("aborted", "AbortError"))
    )
  })
}

async function gunzip(body: BodyInit): Promise<Uint8Array> {
  const stream = new Blob([body as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

const HEALTHY = { status: "ok", model: "auto_avsr_lrs3_v19.1", device: "cuda" }
const RESULT = {
  text: "Hello there.",
  raw_text: "HELLO THERE",
  confidence: null,
  frames: 3,
  latency_ms: { load: 1.5, crop: 0.4, vsr: 861, correct: 0, total: 870 },
}

async function healthy(fetchMock = vi.fn()): Promise<HttpRecognizer> {
  fetchMock.mockResolvedValueOnce(json(HEALTHY))
  vi.stubGlobal("fetch", fetchMock)
  const rec = new HttpRecognizer({ baseUrl: BASE, requestTimeoutMs: 50, requestMsPerSecond: 0 })
  await rec.init()
  return rec
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("HttpRecognizer", () => {
  it("is unavailable and never touches the network without a URL", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    const rec = new HttpRecognizer({ baseUrl: "  " })
    await rec.init()
    expect(rec.available).toBe(false)
    expect(rec.name).toBe("RunPod · not configured")
    await expect(rec.recognize(crops(13))).rejects.toMatchObject({
      kind: "not_configured",
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("probes /health and reports the device", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    expect(fetchMock.mock.calls[0][0]).toBe("https://pod.example/health")
    expect(rec.available).toBe(true)
    expect(rec.mode).toBe("accuracy")
    expect(rec.isReal).toBe(true)
    expect(rec.name).toBe("RunPod · beam (cuda)")
  })

  it.each([
    [
      "HTTP 502",
      () => Promise.resolve(json({ error: "bad gateway" }, 502)),
      "RunPod · offline (HTTP 502)",
    ],
    [
      "an HTML page",
      () => Promise.resolve(new Response("<html></html>")),
      "RunPod · offline (HTTP 200)",
    ],
    [
      "a network error",
      () => Promise.reject(new TypeError("Failed to fetch")),
      "RunPod · offline",
    ],
    ["a hung server", hangingFetch, "RunPod · timed out"],
  ])("stays unavailable when /health gives %s", async (_, impl, name) => {
    vi.stubGlobal("fetch", vi.fn(impl))
    const rec = new HttpRecognizer({ baseUrl: BASE, healthTimeoutMs: 30 })
    await rec.init()
    expect(rec.available).toBe(false)
    expect(rec.name).toBe(name)
  })

  it("POSTs gzipped uint8 crops and maps the response", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const input = crops(3)
    const result = await rec.recognize(input)

    const [url, init] = fetchMock.mock.calls[1] as [string, RequestInit]
    expect(url).toBe(
      "https://pod.example/lipread/crops?t=3&h=88&w=88&decode=beam&correct=false"
    )
    expect(init.method).toBe("POST")
    expect(init.headers).toEqual({
      "Content-Type": "application/octet-stream",
      "Content-Encoding": "gzip",
    })
    const sent = await gunzip(init.body as BodyInit)
    expect(sent.length).toBe(3 * SENT)
    input.patches.forEach((_, i) =>
      expect(sent.subarray(i * SENT, (i + 1) * SENT)).toEqual(
        new Uint8Array(SENT).fill(i + 1)
      )
    )

    expect(result).toMatchObject({
      text: "HELLO THERE", // raw_text: the corrector is off
      mode: "accuracy",
      engine: "RunPod · beam (cuda)",
      serverLatencyMs: RESULT.latency_ms,
    })
    expect(result.confidence).toBeUndefined()
    expect(result.fellBack).toBeUndefined()
    expect(result.latencyMs).toBeGreaterThanOrEqual(0)
  })

  it("falls back to `text` when there is no raw_text and keeps a numeric confidence", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(
      json({ text: "HI", confidence: 0.8, latency_ms: {} })
    )
    await expect(rec.recognize(crops(13))).resolves.toMatchObject({
      text: "HI",
      confidence: 0.8,
    })
  })

  it("surfaces the server's error detail and stays available on 4xx", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(
      json({ detail: { error: "clip_too_short" } }, 422)
    )
    await expect(rec.recognize(crops(3))).rejects.toMatchObject({
      kind: "http",
      status: 422,
      message: expect.stringContaining("clip_too_short"),
    })
    expect(rec.available).toBe(true)
  })

  it.each([
    ["HTTP 503", () => Promise.resolve(json({ detail: "busy" }, 503)), "http"],
    [
      "a network error",
      () => Promise.reject(new TypeError("Failed to fetch")),
      "network",
    ],
    ["a hung server", hangingFetch, "timeout"],
  ])("marks the service down after %s", async (_, impl, kind) => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockImplementationOnce(impl)
    await expect(rec.recognize(crops(3))).rejects.toMatchObject({ kind })
    expect(rec.available).toBe(false)
    expect(rec.name).toBe("RunPod · offline")
  })

  it("honours the caller's abort signal without marking the service down", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockImplementationOnce(hangingFetch)
    const ctrl = new AbortController()
    const pending = rec.recognize(crops(3), ctrl.signal)
    setTimeout(() => ctrl.abort(), 5)
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(rec.available).toBe(true)
  })

  it("uploads exactly the centre 88×88 of each 96×96 patch", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const patch = new Uint8Array(PATCH)
    for (let y = 0; y < 96; y++)
      for (let x = 0; x < 96; x++) patch[y * 96 + x] = (y * 7 + x) & 255
    await rec.recognize({ patches: [patch], faceCoverage: 1, keypoints: [] })

    const [, init] = fetchMock.mock.calls[1] as [string, RequestInit]
    const sent = await gunzip(init.body as BodyInit)
    expect(sent.length).toBe(SENT)
    for (let y = 0; y < 88; y++)
      for (let x = 0; x < 88; x++)
        expect(sent[y * 88 + x]).toBe(((y + 4) * 7 + (x + 4)) & 255)
  })

  it("scores phrases on the server from the same crops, best first", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const result = await rec.recognize(crops(3))
    fetchMock.mockResolvedValueOnce(
      json({
        phrases: [
          { text: "Hello where", margin: -0.31 },
          { text: "Hello there", margin: -0.02 },
        ],
        frames: 3,
        latency_ms: { score: 12 },
      })
    )

    await expect(
      result.scorePhrases?.("HELLO THERE", ["Hello where", "Hello there"])
    ).resolves.toEqual([
      { text: "Hello there", margin: -0.02 },
      { text: "Hello where", margin: -0.31 },
    ])
    const [, read] = fetchMock.mock.calls[1] as [string, RequestInit]
    const [url, init] = fetchMock.mock.calls[2] as [string, RequestInit]
    expect(url).toBe("https://pod.example/lipread/phrases?t=3&h=88&w=88")
    expect(init.method).toBe("POST")
    expect(init.headers).toBeUndefined() // multipart: fetch sets the boundary, no CORS preflight
    const form = init.body as FormData
    expect(form.get("reading")).toBe("HELLO THERE")
    expect(form.getAll("phrases")).toEqual(["Hello where", "Hello there"])
    const sent = form.get("crops") as Blob
    expect(sent.type).toBe("application/gzip")
    expect(await gunzip(sent)).toEqual(await gunzip(read.body as BodyInit))
  })

  it("leaves out phrases the server would refuse", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const result = await rec.recognize(crops(3))
    fetchMock.mockResolvedValueOnce(json({ phrases: [] }))
    const many = Array.from({ length: 600 }, (_, i) => `phrase ${i}`)
    await result.scorePhrases?.("HI", ["x".repeat(301), ...many])
    const form = (fetchMock.mock.calls[2] as [string, RequestInit])[1].body as FormData
    expect(form.getAll("phrases")).toEqual(many.slice(0, 500))

    await expect(result.scorePhrases?.("HI", ["x".repeat(301)])).resolves.toEqual([])
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it.each([
    ["a server without the endpoint", () => Promise.resolve(json({ detail: "Not Found" }, 404))],
    ["a server error", () => Promise.resolve(json({ detail: "boom" }, 500))],
    ["a network error", () => Promise.reject(new TypeError("Failed to fetch"))],
    ["a hung server", hangingFetch],
    ["a malformed body", () => Promise.resolve(json({ text: "HI" }))],
  ])("gives null (look-alike ranking) after %s, still available", async (_, impl) => {
    const fetchMock = vi.fn()
    fetchMock.mockResolvedValueOnce(json(HEALTHY))
    vi.stubGlobal("fetch", fetchMock)
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const rec = new HttpRecognizer({ baseUrl: BASE, scoreTimeoutMs: 30 })
    await rec.init()
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const result = await rec.recognize(crops(3))
    fetchMock.mockImplementationOnce(impl)
    await expect(result.scorePhrases?.("HI", ["Hi there"])).resolves.toBeNull()
    expect(rec.available).toBe(true)
  })

  it("gives null when the caller's read is aborted", async () => {
    const fetchMock = vi.fn()
    const rec = await healthy(fetchMock)
    fetchMock.mockResolvedValueOnce(json(RESULT))
    const ctrl = new AbortController()
    const result = await rec.recognize(crops(3), ctrl.signal)
    fetchMock.mockImplementationOnce(hangingFetch)
    const pending = result.scorePhrases?.("HI", ["Hi there"])
    setTimeout(() => ctrl.abort(), 5)
    await expect(pending).resolves.toBeNull()
  })

  it("rejects patches that are not 96×96", async () => {
    const rec = await healthy()
    const bad: CropResult = {
      patches: [new Uint8Array(88 * 88)],
      faceCoverage: 1,
      keypoints: [],
    }
    await expect(rec.recognize(bad)).rejects.toThrow(/expected 9216/)
  })
})
