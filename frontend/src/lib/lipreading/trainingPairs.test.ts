import { describe, expect, it, vi } from "vitest"

import { sendTrainingPair } from "./trainingPairs"
import type { CropResult } from "./types"

const crops = (t: number): CropResult => ({
  patches: Array.from({ length: t }, (_, i) => new Uint8Array(96 * 96).fill(i)),
  faceCoverage: 1,
  keypoints: [],
})

describe("sendTrainingPair", () => {
  it("posts the 96x96 patches with t, text and source in the query", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("{}", { status: 200 }))
    await sendTrainingPair("http://srv", crops(3), "I think", "picked", fetchImpl)
    const [url, init] = fetchImpl.mock.calls[0]
    const u = new URL(String(url))
    expect(u.pathname).toBe("/training-pairs")
    expect(Object.fromEntries(u.searchParams)).toEqual({
      t: "3",
      h: "96",
      w: "96",
      text: "I think",
      source: "picked",
    })
    expect(init?.method).toBe("POST")
    const headers = init?.headers as Record<string, string>
    // gzipped when CompressionStream exists (Node 18+), else the raw 3*96*96 bytes
    const body = new Uint8Array(await new Response(init?.body as BodyInit).arrayBuffer())
    if (headers["Content-Encoding"] === "gzip") expect(body.length).toBeLessThan(3 * 96 * 96)
    else expect(body.length).toBe(3 * 96 * 96)
  })

  it("throws on a server error", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("no", { status: 422 }))
    await expect(sendTrainingPair("http://srv", crops(2), "x", "picked", fetchImpl)).rejects.toThrow(
      "HTTP 422"
    )
  })
})
