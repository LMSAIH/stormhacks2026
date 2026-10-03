import type { LipModelSpec } from "./types"

/**
 * Auto-AVSR `LRS3_V_WER19.1` (visual-only), encoder + CTC head exported by
 * `ml/scripts/export_onnx.py`: `video` float32 [1, 1, T, 88, 88] → `log_probs` float32 [T, 5049].
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
  modelUrl: "/models/lipread_ctc.onnx",
  tokensUrl: "/models/tokens.json",
  minSeconds: 0.5,
  maxSeconds: 10,
  minFaceCoverage: 0.5,
}

/** The spec the app currently builds against. */
export const ACTIVE_SPEC: LipModelSpec = AUTO_AVSR_LRS3_SPEC
