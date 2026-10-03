import type { LipModelSpec } from "./types"

/**
 * Default spec targeting a LipNet-style GRID model.
 *
 * These numbers match the canonical LipNet pipeline (75 RGB mouth frames at
 * 25fps, 100x50). When the real model lands, adjust ONLY this object — if the
 * team ships, say, a 96x96 grayscale Auto-AVSR model, change the fields here
 * and the rest of the pipeline follows.
 */

// GRID/LipNet charset: a-z, space, apostrophe, then CTC blank last.
const LIPNET_CHARSET: readonly string[] = [
  ..."abcdefghijklmnopqrstuvwxyz".split(""),
  " ",
  "'",
  "<blank>",
]

export const LIPNET_SPEC: LipModelSpec = {
  name: "LipNet (GRID)",
  windowFrames: 75,
  targetFps: 25,
  cropWidth: 100,
  cropHeight: 50,
  channels: 3,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  charset: LIPNET_CHARSET,
  blankIndex: LIPNET_CHARSET.length - 1,
  modelUrl: "/models/lipreader.onnx",
}

/** The spec the app currently builds against. */
export const ACTIVE_SPEC: LipModelSpec = LIPNET_SPEC
