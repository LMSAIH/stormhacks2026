/**
 * Test-only loader for crop fixtures written by
 * `ml/scripts/dump_crop_reference.py` (meta.json + frames.bin + crops.bin).
 */
import type { CapturedFrame, Keypoints } from "../../types"

export interface CropFixtureMeta {
  source: string
  cv2: string
  T: number
  H: number
  W: number
  fps: number
  patchSize: number
  stableReference: number[][]
  /** Raw detector keypoints per 25 fps frame; null = no face. */
  keypoints: (Keypoints | null)[]
  /** Everything below was captured inside the Python VideoProcess. */
  smoothed: Keypoints[]
  transforms: number[][]
  inliers: number[][]
  mouth256: number[][]
  origins: number[][]
}

export interface CropFixture {
  meta: CropFixtureMeta
  /** T·H·W gray frames exactly as Python warped them. */
  frames: Uint8Array
  /** T·96·96 Python crops. */
  crops: Uint8Array
}

interface NodeFsPromises {
  readFile(path: URL | string): Promise<Uint8Array>
}

// Resolved at runtime only: vitest runs in Node, while tsconfig.app.json
// (browser) has no Node types and the app never imports this file.
const fs = (await import(
  /* @vite-ignore */ "node:fs/promises" as string
)) as NodeFsPromises

/** `dir`: a directory URL (ending in "/") or a filesystem path. */
export async function loadCropFixture(dir: URL | string): Promise<CropFixture> {
  const at = (name: string) =>
    typeof dir === "string"
      ? `${dir.replace(/\/$/, "")}/${name}`
      : new URL(name, dir)
  const meta = JSON.parse(
    new TextDecoder().decode(await fs.readFile(at("meta.json")))
  ) as CropFixtureMeta
  const frames = new Uint8Array(await fs.readFile(at("frames.bin")))
  const crops = new Uint8Array(await fs.readFile(at("crops.bin")))
  const { T, H, W, patchSize } = meta
  if (frames.length !== T * H * W || crops.length !== T * patchSize ** 2) {
    throw new Error(`fixture ${String(dir)}: sizes do not match meta.json`)
  }
  return { meta, frames, crops }
}

/** Fixture frames as captured frames at exactly 25 fps (tMs = i·40). */
export function fixtureFrames({ meta, frames }: CropFixture): CapturedFrame[] {
  const { T, H, W } = meta
  return Array.from({ length: T }, (_, i) => ({
    tMs: i * 40,
    width: W,
    height: H,
    gray: frames.subarray(i * H * W, (i + 1) * H * W),
    keypoints: meta.keypoints[i],
  }))
}
