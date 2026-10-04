# Phase A design — wire the Auto-AVSR model into the frontend (speed / accuracy modes)

Branch `feat/frontend-lipread` (= `ml/model-pipeline` + master's frontend). Frontend = Vite web app
(`frontend/`, pnpm ONLY, never npm). Model facts: brief §4/§5/§11. This doc is the contract every
implementer codes against; if you must deviate, say so in your final report.

## 1. Modes (user-facing toggle, persisted in localStorage — wrap access in try/catch)

| Mode | Where | Decode | Measured (brief §11) |
|---|---|---|---|
| `speed` (default) | on-device: onnxruntime-web, EPs `["webgpu","wasm"]` | greedy CTC | 28.6% WER, ~0.24 s laptop CPU (ORT native); browser TBD |
| `accuracy` | hosted: `POST {VITE_LIPREAD_URL}/lipread/crops` | beam 40 + LM | 21.7% WER, ~1.3 s p50 RTT from Vancouver |

- Both modes share ONE capture + crop pipeline; only the final `recognize(crops)` differs.
- `accuracy` failure (no URL configured, network error, non-200, >15 s timeout) → fall back to
  `speed` for that utterance and mark the result `fellBack: true`.
- `speed` unavailable (model file missing / ORT init failed) → engine reports `available=false`;
  UI shows it; accuracy still works. Mock engine only when neither is available.

## 2. Utterance segmenting (replaces the 75-frame ring buffer + 1.5 s interval)

Push-to-talk: hold **Space** (when focus is not in an input/textarea) or press-and-hold the on-screen
button → record → release → recognize once → append one transcript item. Hard cap 10 s (auto-stop);
discard < 0.5 s. While recording, per video frame store `{ tMs, gray: Uint8Array(W*H), kps|null }`
(gray = Rec.601 `round(0.299R+0.587G+0.114B)`, same as cv2 RGB2GRAY), where kps = 4 keypoints
from the face detector in **pixel coords truncated to int** (Python does `int(x*iw)`).

## 3. Crop pipeline — exact port of `ml/src/lipread/preprocess.py` + vendored `video_process.py`

Pure TS, no DOM, in `frontend/src/lib/lipreading/crop/` so it runs in Node tests:

1. **Resample to 25 fps by timestamp**: output n = max(1, round(durationS*25)) frames,
   duration = (last.tMs - first.tMs)/1000 + 1/srcFps; pick nearest source frame for k*40 ms
   (mirror `lipread/video.py::resample_fps`: idx = min(round(k*src_fps/25), len-1)).
2. **Coverage gate**: if fewer than 50% of frames have kps → `NoFaceError` (UI: "no face").
3. **Interpolate** missing kps linearly between detected frames; copy first/last detection to the
   leading/trailing gaps (vendored `interpolate_landmarks`).
4. **Smooth** per frame i: `m = min(6, i, n-1-i)`; `s = mean(kps[i-m..i+m])`; then
   `s += mean(kps[i]) - mean(s)` (per-axis means over the 4 points) — exactly `crop_patch`.
5. **Similarity transform** (rotation+uniform scale+translation, no shear/reflection) mapping the
   4 smoothed points → `STABLE_REFERENCE` (least squares / Umeyama; Python uses
   `cv2.estimateAffinePartial2D(..., LMEDS)` — equal to LSQ when all 4 points are inliers):
   ```
   STABLE_REFERENCE = [[102.073943, 94.272304],   // right eye
                       [156.361305, 93.578156],   // left eye
                       [129.003738, 135.90343],   // nose tip
                       [129.313373, 157.822996]]  // mouth centre   (256x256 space)
   ```
6. **Warp + cut**: transformed mouth point `(cx, cy)` = M·kps[3]. Patch = 96×96 at
   `x0 = round(clip(cx-48, 0, 256))`, `y0 = round(clip(cy-48, 0, 256))` (keep 96 wide: shift inward
   if clipped). For each patch pixel (u,v): 256-space point (x0+u, y0+v) → source via M⁻¹ →
   **bilinear** sample of the gray frame, border = 0 (cv2 INTER_LINEAR, BORDER_CONSTANT) →
   round to uint8. Output `Uint8Array(96*96)` per frame.
7. **Model input**: centre-crop 88 (offset 4,4) → `/255` → `(x-0.421)/0.165` →
   `Float32Array` laid out **[1, 1, T, 88, 88]** (N, C, T, H, W).

## 4. Model I/O (local)

- Files (gitignored, copied by `ml/scripts/publish_frontend_model.sh`): `frontend/public/models/lipread_ctc.int8.onnx`
  (203 MB: the `dyn-pw8-rn16` int8 quantization of the 775 MB fp32 export, D39; same I/O, T ≤ 500 frames)
  + `frontend/public/models/tokens.json` (array of 5049 strings).
- Input `video` float32 [1,1,T,88,88], T dynamic. Output `log_probs` float32 [T,5049].
- Greedy: argmax per frame → collapse repeats → drop blank (**index 0**) → drop `<eos>` (last
  index) → join pieces → replace `▁` with space → trim. Model emits UPPERCASE; display as-is or
  sentence-case (UI choice). Confidence = mean max-prob over non-blank frames (exp of log-prob).

## 5. Hosted endpoint (new, Python) — `POST /lipread/crops`

Query: `t` (frames), `h`=96, `w`=96, `decode`=greedy|beam (default beam), `correct`=false.
Body: raw uint8 `t*h*w` bytes (row-major frames), optionally gzip with header
`Content-Encoding: gzip` (browser `CompressionStream("gzip")`). Validates size, 0.5–10 s at 25 fps,
`h,w ∈ {88,96}` (88 → treat as already centre-cropped). Response = same JSON as `/lipread`
(`text, raw_text, confidence, frames, latency_ms{load,crop,vsr,correct,total}`). CORS already `*`.
Frontend base URL from `import.meta.env.VITE_LIPREAD_URL` (`frontend/.env.example` documents it;
real value goes in `frontend/.env.local`, gitignored).

## 6. Frontend contracts (`frontend/src/lib/lipreading/types.ts` — authored before build)

See the file. Key types: `Keypoints`, `CapturedFrame`, `Utterance`, `CropResult`,
`RecognitionMode`, `RecognitionResult`, `Recognizer` (local ONNX / hosted HTTP / mock),
`MouthDetector`.

## 7. File ownership (parallel build — touch ONLY your files)

| Agent | Owns |
|---|---|
| crop | `src/lib/lipreading/crop/**`, `src/lib/lipreading/crop/*.test.ts`, `ml/scripts/dump_crop_reference.py` |
| engines | `src/lib/lipreading/{onnxRecognizer,httpRecognizer,mockRecognizer,ctc,modelSpec,createRecognizers}.ts`, `ml/scripts/publish_frontend_model.sh` |
| detector+hook+ui | `src/lib/lipreading/faceDetector.ts`, `src/hooks/useLipReader.ts`, `src/components/app/{camera-panel,mode-toggle,ptt-button}.tsx`, `src/pages/app-page.tsx` |
| server | `ml/src/lipread/serve/app.py`, `ml/scripts/smoke_checks.py` (new check), brief §5 text |
| lab | `src/pages/lab-page.tsx`, route in `src/main.tsx` |
| media | `ml/data/**` (gitignored), `frontend/public/test/**` (gitignored), `frontend/public/mediapipe/blaze_face_*.tflite` |
| (me) | `types.ts`, `eslint.config.js`, `.gitignore`s, package.json devDeps (vitest), integration |

Legacy files removed in integration: `ringBuffer.ts`, old `onnxEngine.ts`, `mockEngine.ts`,
`createEngine.ts`. **`faceLandmarker.ts` + `face_landmarker.task` are the teammate's lip tracking and
must stay** (restored after the master merge; it drives the lip-dot overlay, BlazeFace drives the crop).

## 9. Cross-module exports (import exactly these names; all paths under `frontend/src/`)

```ts
// lib/lipreading/crop/index.ts            (crop agent)
export function rgbaToGray(rgba: Uint8ClampedArray, width: number, height: number): Uint8Array
export function cropUtterance(u: Utterance, spec?: LipModelSpec): CropResult   // throws NoFaceError
export function toModelInput(crops: CropResult): { data: Float32Array; dims: [1, 1, number, 88, 88] }
export const STABLE_REFERENCE: Keypoints
// + pure helpers (resampleIndices, interpolateKeypoints, smoothKeypoints,
//   estimateSimilarity, warpPatch) exported for tests

// lib/lipreading/modelSpec.ts             (engines agent)
export const ACTIVE_SPEC: LipModelSpec      // modelUrl "/models/lipread_ctc.int8.onnx", tokensUrl "/models/tokens.json"
// lib/lipreading/ctc.ts
export function greedyCtcDecode(logProbs: Float32Array, timesteps: number, tokens: readonly string[]):
  { text: string; confidence?: number }
// lib/lipreading/createRecognizers.ts
export function createRecognizers(spec?: LipModelSpec): Promise<{ speed: Recognizer; accuracy: Recognizer }>
//   speed = OnnxRecognizer (falls back to MockRecognizer if model missing); accuracy = HttpRecognizer
//   (available=false when VITE_LIPREAD_URL unset). Each class in its own file.

// lib/lipreading/faceDetector.ts          (detector+hook+ui agent)
export class BlazeFaceDetector implements MouthDetector   // tasks-vision FaceDetector, VIDEO mode

// hooks/useLipReader.ts                   (detector+hook+ui agent)
export function useLipReader(): {
  videoRef; overlayRef; cameraStatus: "idle"|"starting"|"on"|"error"; ready: boolean;
  mouthDetected: boolean; fps: number;
  mode: RecognitionMode; setMode(m: RecognitionMode): void;
  engines: Record<RecognitionMode, { name: string; available: boolean; isReal: boolean }>;
  recording: boolean; startUtterance(): void; stopUtterance(): void; busy: boolean;
  lastError: string | null;
  transcript: { id: string; text: string; at: number; mode: RecognitionMode; latencyMs: number;
                fellBack?: boolean; engine: string; confidence?: number }[];
  clearTranscript(): void;
}
```

## 8. Acceptance

- `pnpm lint && pnpm typecheck && pnpm build && pnpm test` green; `./smoke.sh` green.
- Node test: TS crop of Python-dumped gray frames + kps vs Python crops → mean |Δ| ≤ 1.5 levels,
  max |Δ| ≤ 8 (bilinear rounding), transforms within 1e-3.
- Browser `/lab` (Playwright, headless Chromium → WASM EP): real face clip → speed text non-empty,
  accuracy text from the pod; JS keypoints vs Python keypoints mean error reported.
