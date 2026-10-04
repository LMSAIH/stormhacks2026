# Demo checklist: start 30 minutes before recording or judging

Work down the list in order. Commands assume the repo root on the demo laptop. Never paste an API
key or token into a chat, a commit or a screen recording.

## Fastest path: the live site

The whole stack runs at https://tryheard.tech (app), https://ml.tryheard.tech (our ML server) and
https://api.tryheard.tech (the backend), so the demo needs only Chromium (`ml/runpod/README-deploy.md`).

- [ ] `scripts/check_live.sh` passes (run it from anywhere; its exit code is the number of failed
      checks).
- [ ] Open https://tryheard.tech, sign in with Google, pick the voice, and check that mute is off.
- [ ] Open the mode menu: Quality reads "Cloud · up to 20 s" and does not say "Server offline".
- [ ] The camera, light and phrase-memory steps below (T−12, T−10) still apply to this browser.
- [ ] If a check fails: on the pod, `RESTART=ml bash /workspace/stormhacks2026/ml/runpod/up.sh`
      restarts one part (`backend` or `tunnel` for the others). If the pod itself is gone,
      `BRANCH=master bash ml/runpod/create_pod.sh` (needs `RUNPOD_API_KEY`) recreates it in about
      3 minutes. Then run `check_live.sh` again and reload the page.

The steps below are the local path (Vite on the laptop, a pod or uvicorn of your own). Use them if
the live site is down.

## T−30: start and warm the GPU pod (Quality mode)

- [ ] Start the serving pod from the RunPod console (pod ID in the latest handoff: `qa5oi7o7g46n4q`
      on 2026-10-04). If it says "not enough free GPUs on the host machine", create a new
      secure-cloud RTX 4090 pod instead (D80), with HTTP port 8000 exposed, and use its ID below.
- [ ] Open a terminal on the pod (RunPod console → Connect: web terminal or SSH) and run:

  ```bash
  # existing pod (a restart wipes everything outside /workspace, so re-run bootstrap; it's idempotent)
  BRANCH=master bash /workspace/stormhacks2026/ml/runpod/bootstrap.sh && bash /workspace/stormhacks2026/ml/runpod/serve.sh
  # new pod (bootstrap.sh's default branch is stale, so pass BRANCH=master)
  curl -fsSL https://raw.githubusercontent.com/LMSAIH/stormhacks2026/master/ml/runpod/bootstrap.sh | BRANCH=master bash
  bash /workspace/stormhacks2026/ml/runpod/serve.sh
  ```

  `serve.sh` loads the model at start-up (`LIPREAD_WARM=1`) and waits for `/health`.
- [ ] From the laptop: `curl -s https://<POD_ID>-8000.proxy.runpod.net/health` shows
      `"status":"ok"`, `"loaded":true` and `"device":"cuda:0"`.
- [ ] The pod costs US$0.74/h (about CA$1.05/h at 1.4250 CAD per USD, 2 October 2026 close). Stop it
      in the RunPod console when the demo is over.

## T−25: app settings

- [ ] Put a local copy of the model in `frontend/public/models/` (gitignored), so a page reload
      doesn't depend on Hugging Face. The app caches the model after one download but fetches
      `tokens.json` again on every load.

  ```bash
  mkdir -p frontend/public/models
  BASE=https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/9359b251b8d9b8e2d63bada99013bb92eee3b087
  curl -fL -o frontend/public/models/lipread_ctc.int8.onnx "$BASE/lipread_ctc.int8.onnx"
  curl -fL -o frontend/public/models/tokens.json "$BASE/tokens.json"
  shasum -a 256 frontend/public/models/lipread_ctc.int8.onnx   # must start with da02d72e
  ```

- [ ] `frontend/.env.local`:

  ```bash
  VITE_LIPREAD_URL=https://<POD_ID>-8000.proxy.runpod.net
  VITE_LIPREAD_MODEL_BASE=/models
  ```

  No trailing slash on the URL. Leave `VITE_SKIP_AUTH` unset: the voice only plays for a signed-in
  user. If the backend runs in docker compose, also set `VITE_API_URL=http://localhost:4000`.
- [ ] Restart `pnpm dev` (in `frontend/`) so the new settings are picked up.

## T−20: backend and voice

- [ ] The backend team's server is running: `python main.py` (REST on :5000, TTS WebSocket on
      :8765) or `docker compose up -d` (REST on :4000). Its `FRONTEND_ORIGINS` must include
      `http://localhost:5173`.
- [ ] In the demo browser profile (not incognito: it drops the cached model and phrase memory), open
      `http://localhost:5173/login` (or "Sign in" on the landing page), sign in with Google, pick the
      voice, and check that mute is off.

## T−15: model loaded, both readers live

- [ ] Open `http://localhost:5173/app` with DevTools open and wait for
      `[lipreading] ONNX ready on wasm in … ms` in the console, with no `local model unavailable`
      warning. The transcript box's "Mouth words to the camera…" only means face tracking is up;
      the main screen doesn't show whether the model has loaded or which reader is active.
- [ ] If neither the model nor the server loads, the app falls back to a mock that prints canned
      phrases ("HELLO THERE", "NICE TO MEET YOU"). Never demo in that state.
- [ ] Reload once: it should be ready again within a few seconds.
- [ ] Open the mode menu: Quality reads "Cloud · up to 20 s" and does not say "Server offline".
      The app checks the server only when the page loads, so start the pod before loading the page.
      The menu never changes after that, even if the server fails later.

## T−12: camera and light

- [ ] The app asks the webcam for 640×480 at 30 fps, and `/app` no longer shows a frame-rate
      readout. Paste this into the DevTools console on `/app`; after 3 s it prints the camera's
      size and frame rate, for example `camera: 640×480 at 29.9 fps`:

  ```js
  (() => {
    const v = document.querySelector("video")
    const t0 = performance.now()
    let n = 0
    const tick = () => {
      n++
      if (performance.now() - t0 < 3000) v.requestVideoFrameCallback(tick)
      else console.log(`camera: ${v.videoWidth}×${v.videoHeight} at ${(n / ((performance.now() - t0) / 1000)).toFixed(1)} fps`)
    }
    v.requestVideoFrameCallback(tick)
  })()
  ```

- [ ] Aim for 25–30 fps with your face in the lit position. Below about 20: add light (webcams slow
      down in dim rooms), close other apps using the camera, plug in the laptop. Fewer frames cost
      accuracy: offline, 15 fps had 48.3% of words wrong against 38.1% at 25 fps
      (`.context/project-brief.md` §11).
- [ ] Even light from the front, no window behind you: in the app test, contrast around the mouth
      was what tracked errors. Face fills about a third of the frame, eyes on the lens, mouth fully
      visible, head turned less than 15°, hands away from the face.
- [ ] Camera at eye level, not below the face: on the 62-person test a camera below the face added
      21.6 points (on the 4 takes that could be compared). Prop the laptop up if needed.
- [ ] No amber hint next to "You" under the camera ("Move closer to the camera", "Too dark — add
      some light", "Face the camera straight on").

## T−10: phrase memory with the demo lines

Phrase memory lives in this browser profile only. Either mouth each demo line once in Normal mode
and fix any misread by clicking a word and typing, or paste this into the DevTools console on the
app's page (it writes the same records as `frontend/src/lib/phrases/store.ts`; running it twice is
harmless):

```js
(async () => {
  const lines = [
    "Hi, nice to meet you", "Can you help me please", "I would like a glass of water",
    "Thank you so much", "I am not feeling well", "Please call my family", "Where is the bathroom",
    "I need a few more minutes", "Yes, that sounds good to me", "No, I do not want that",
    "Can you repeat that more slowly", "This app reads my lips and speaks for me",
  ]
  // Same rule as normalizeText in frontend/src/lib/phrases/lookalike.ts.
  const norm = (t) => t.toUpperCase().replace(/[^A-Z0-9' ]+/g, " ").replace(/\s+/g, " ").trim()
  const db = await new Promise((resolve, reject) => {
    const req = indexedDB.open("lipread-phrases", 1)
    req.onupgradeneeded = () => {
      const store = req.result.createObjectStore("phrases", { keyPath: "id" })
      store.createIndex("norm", "norm", { unique: true })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  let added = 0
  for (const text of lines) {
    added += await new Promise((resolve) => {
      const tx = db.transaction("phrases", "readwrite")
      const req = tx.objectStore("phrases").add({
        id: crypto.randomUUID(), text, norm: norm(text), count: 1, source: "typed", lastUsed: Date.now(),
      })
      req.onerror = (e) => e.preventDefault() // already saved: the norm index is unique
      tx.oncomplete = () => resolve(req.error ? 0 : 1)
      tx.onabort = () => resolve(0)
    })
  }
  db.close()
  console.log(`phrase memory: ${added} added, ${lines.length - added} already there`)
})()
```

- [ ] The console prints `phrase memory: 12 added` (or `already there` on a second run).

## T−8: warm up and rehearse

- [ ] In Quality mode, mouth two or three throwaway sentences: the first beam reads after a server
      start are the slow ones.
- [ ] In Normal mode, run through the demo lines from `demo-script.md` once. Pause about a second
      between sentences (a sentence ends after 0.8 s of still lips). Swap out any line that misreads
      twice.

## T−3: recording setup

- [ ] OBS (or the screen recorder): the browser window at 1920×1080, desktop audio on so the
      ElevenLabs voice is recorded, microphone muted for the silent segments.
- [ ] Notifications off, other tabs closed, laptop plugged in.

## Which mode to use

| Mode | Use it for | Why |
|---|---|---|
| Normal (default) | The demo lines | The fine-tuned model on the laptop, with drafts and model-scored phrase memory: 27.0% of words wrong in the app test; about 1.9 s to text |
| Quality | One natural sentence that isn't in phrase memory | Beam + LM on the RTX 4090: 25.8% in the app test; 35.0% against 41.3% on-device on 62 unseen people; about 2.2 s to text |
| Instant | Not in the demo | No boxes and no phrase memory; 30.0% in the app test |

App test: 20 strangers' clips, two runs each, and runs of the same model differ by up to 8 points.
Numbers: `.context/b2-report.md`, `.context/eval-v2.md`.

## If something breaks

| Symptom | What to do |
|---|---|
| The pod stops answering mid-demo | Nothing on screen changes: Quality sentences are read on the laptop instead, the console logs `[useLipReader] cloud read failed, reading on this device` once, and the mode menu keeps its page-load state. Keep going, since the laptop reads every sentence. Check `curl -s https://<POD_ID>-8000.proxy.runpod.net/health`, run `bash /workspace/stormhacks2026/ml/runpod/serve.sh` on the pod, then reload the page: after a network error, timeout or 5xx the app stops trying the server until it reloads |
| Mode menu says "Server offline" | The server was down when the page loaded. Start it (T−30 steps), check `/health`, reload the page |
| The pod won't start | Create a new secure-cloud RTX 4090 pod with HTTP port 8000 exposed, bootstrap it as above, update `VITE_LIPREAD_URL`, restart `pnpm dev`, reload |
| No pod at all | Run the server on the laptop: `./ml/scripts/download_checkpoints.sh` (about 1.3 GB, once), then `uv run --directory ml uvicorn lipread.serve.app:app --port 8000` (the first run also installs the Python environment) and `VITE_LIPREAD_URL=http://127.0.0.1:8000`. Beam search on a laptop is slower than on the 4090, so prefer Normal |
| No voice | Check that you are signed in and not muted, and that the backend is running |
| Lip dots frozen | Reload the page |
| "Camera is busy in another app" | Close the other app (video calls, OBS virtual camera), then reload |
| Several misreads in a row | Check the light and distance, re-run the camera line from T−12, pause a full second between sentences, use a line from phrase memory |
| Live judging and nothing works | Play the recorded demo video |
