import type { LipModelSpec } from "./types"

/**
 * Auto-AVSR `LRS3_V_WER19.1` (visual-only), encoder + CTC head exported by
 * `ml/scripts/export_onnx.py` and int8-quantized by `ml/scripts/quantize_onnx.py --variant
 * dyn-pw8-rn16` (203 MB instead of the 775 MB fp32 export; same I/O and tokens):
 * `video` float32 [1, 1, T, 88, 88] → `log_probs` float32 [T, 5049], T ≤ 500 frames.
 * Preprocessing constants mirror `ml/src/lipread/preprocess.py`. The model files are not in git:
 * `ml/scripts/publish_frontend_model.sh` copies them into `frontend/public/models/`.
 */
export const AUTO_AVSR_LRS3_SPEC: LipModelSpec = {
  name: "Auto-AVSR LRS3 (19.1)",
  fps: 25,
  patchSize: 96,
  inputSize: 88,
  mean: 0.421,
  std: 0.165,
  blankIndex: 0,
  modelUrl: "/models/lipread_ctc.int8.onnx",
  tokensUrl: "/models/tokens.json",
  minSeconds: 0.5,
  maxSeconds: 10,
  minFaceCoverage: 0.5,
}

/** The spec the app currently builds against. */
export const ACTIVE_SPEC: LipModelSpec = AUTO_AVSR_LRS3_SPEC
