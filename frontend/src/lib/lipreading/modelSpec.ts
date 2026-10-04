import type { LipModelSpec } from "./types"

/**
 * Auto-AVSR `LRS3_V_WER19.1` (visual-only), encoder + CTC head exported by
 * `ml/scripts/export_onnx.py` and int8-quantized by `ml/scripts/quantize_onnx.py --variant
 * dyn-pw8-rn16` (203 MB instead of the 775 MB fp32 export; same I/O and tokens):
 * `video` float32 [1, 1, T, 88, 88] → `log_probs` float32 [T, 5049], T ≤ 500 frames.
 * Preprocessing constants mirror `ml/src/lipread/preprocess.py`.
 *
 * The model files are not in git. By default they load from the Hugging Face repo, pinned to the
 * commit holding the regression-locked file (sha256 da02d72e…f195b, `ml/tests/quantized_baseline.json`),
 * so the bytes can't change underneath the app; the browser keeps them in Cache Storage after the
 * first download. `VITE_LIPREAD_MODEL_BASE=/models` serves them from `public/models/` instead
 * (offline; `ml/scripts/publish_frontend_model.sh` copies them there).
 */
const HF_MODEL_BASE =
  "https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/9359b251b8d9b8e2d63bada99013bb92eee3b087"
const MODEL_BASE = (
  import.meta.env.VITE_LIPREAD_MODEL_BASE || HF_MODEL_BASE
).replace(/\/+$/, "")

export const AUTO_AVSR_LRS3_SPEC: LipModelSpec = {
  name: "Auto-AVSR LRS3 (19.1)",
  fps: 25,
  patchSize: 96,
  inputSize: 88,
  mean: 0.421,
  std: 0.165,
  blankIndex: 0,
  modelUrl: `${MODEL_BASE}/lipread_ctc.int8.onnx`,
  tokensUrl: `${MODEL_BASE}/tokens.json`,
  minSeconds: 0.5,
  maxSeconds: 10,
  minFaceCoverage: 0.5,
}

/** The spec the app currently builds against. */
export const ACTIVE_SPEC: LipModelSpec = AUTO_AVSR_LRS3_SPEC
