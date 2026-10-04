import type { LipModelSpec } from "./types"

/**
 * Auto-AVSR `LRS3_V_WER19.1` (visual-only) fine-tuned on the team's own clips (B2 `FT_v1`, WiSE
 * blend α 0.5 with the stock weights; `.context/b2-report.md`), encoder + CTC head exported by
 * `ml/scripts/export_onnx.py` and int8-quantized by `ml/scripts/quantize_onnx.py --variant
 * dyn-pw8-rn16` (203 MB instead of the 775 MB fp32 export; same I/O and tokens as stock):
 * `video` float32 [1, 1, T, 88, 88] → `log_probs` float32 [T, 5049], T ≤ 500 frames.
 * Preprocessing constants mirror `ml/src/lipread/preprocess.py`.
 *
 * The model files are not in git. By default they load from the Hugging Face repo's `finetuned-v1`
 * branch, pinned to the commit holding the regression-locked file (sha256 55143d51…65979,
 * `ml/tests/quantized_baseline.json`), so the bytes can't change underneath the app; the browser keeps
 * them in Cache Storage after the first download. The stock model stays on `main` (commit 9359b251…,
 * sha256 da02d72e…). `VITE_LIPREAD_MODEL_BASE=/models` serves them from `public/models/` instead
 * (offline; `ml/scripts/publish_frontend_model.sh` copies them there).
 */
const HF_MODEL_BASE =
  "https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/397241eb133cae2ce692a8752eebc0f6359e936d"
const MODEL_BASE = (
  import.meta.env.VITE_LIPREAD_MODEL_BASE || HF_MODEL_BASE
).replace(/\/+$/, "")

export const AUTO_AVSR_LRS3_SPEC: LipModelSpec = {
  name: "Auto-AVSR LRS3 (19.1), B2 fine-tune α 0.5",
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
