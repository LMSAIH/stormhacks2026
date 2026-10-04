# System

The whole system as it runs for the demo. The browser does the camera work, face tracking, mouth
crop and the on-device read, and no camera video leaves it: Quality reads and opted-in training
pairs send grayscale mouth crops only. Green is ours (`frontend/`, `ml/`), yellow is the backend
team's (`backend/`), grey is outside services. A dashed box is planned but not built; a dashed arrow
is optional (opt-in, or only when configured).

```mermaid
flowchart TB
  subgraph browser["Laptop browser: our app (frontend/)"]
    direction LR
    io["Webcam 640×480, 30 fps requested<br/>Microphone, speakers"]
    app["React app, /app page<br/>face trackers in a worker, mouth crop,<br/>on-device int8 reader (onnxruntime-web WASM),<br/>phrase memory in IndexedDB"]
    io <--> app
  end

  subgraph ml["Our ML server (ml/): RunPod RTX 4090, or uvicorn on the laptop"]
    direction LR
    api["FastAPI lipread.serve.app<br/>/health, /lipread/crops, /training-pairs<br/>Auto-AVSR LRS3_V_WER19.1: beam 40 + RNN LM"]
    pairs[("Training pairs on disk<br/>LIPREAD_PAIRS_DIR")]
    api --> pairs
  end

  subgraph be["Backend team (backend/), not ours"]
    direction LR
    backend["FastAPI: REST :5000 (Google sign-in, voices, notes)<br/>TTS WebSocket :8765<br/>/ws/stt captions, speaker labels if DIARIZATION=1"]
    db[("Postgres<br/>TIMESCALE_SERVICE_URL")]
    backend --> db
  end

  subgraph hf["Hugging Face Hub"]
    direction LR
    model[("eschmechel/auto-avsr-lrs3-vsr-int8-onnx<br/>public, pinned commit")]
    ckpt[("Amanvir/LRS3_V_WER19.1<br/>+ Amanvir/lm_en_subword")]
    pairsds[("Training-pairs dataset<br/>LIPREAD_PAIRS_REPO, public")]
    recds[("Team recordings dataset<br/>private, fine-tune only")]
  end

  subgraph ext["Outside services"]
    direction TB
    el["ElevenLabs<br/>TTS, voices, Scribe speech-to-text"]
    google["Google OAuth"]
  end

  tidb["TiDB phrase service<br/>backend team, not built"]

  model -->|"int8 model 203 MB, cached after the first load<br/>tokens.json, fetched on every load"| app
  app -->|"POST /lipread/crops<br/>gzipped 88×88 gray mouth crops, no video"| api
  api -->|"text, up to 3 readings,<br/>per-word confidence"| app
  app -.->|"opt-in, off by default: POST /training-pairs<br/>96×96 mouth crops + confirmed text"| api
  pairs -.->|"background upload,<br/>server's HF token"| pairsds
  ckpt -->|"downloaded at pod setup"| api
  app -->|"finished-line text (signed in only)<br/>mic audio, 16 kHz PCM<br/>sign-in, voices, notes"| backend
  backend -->|"speech audio, 24 kHz PCM<br/>captions"| app
  backend -->|"server-side API keys"| ext
  app -.->|"VITE_PHRASES_URL<br/>unset today: IndexedDB"| tidb

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
  classDef backendteam fill:#fef7e0,stroke:#e37400,color:#000
  classDef outside fill:#f1f3f4,stroke:#5f6368,color:#000
  classDef planned fill:#ffffff,stroke:#80868b,stroke-dasharray:5 5,color:#5f6368
  class io,app,api,pairs ours
  class backend,db backendteam
  class el,google,model,ckpt,pairsds,recds outside
  class tidb planned
```

## What crosses the network

| From → to | What | When |
|---|---|---|
| Hugging Face → browser | `lipread_ctc.int8.onnx` (203 MB), from a pinned commit | First load only; after that the browser's Cache Storage (`lipread-models-v1`) |
| Hugging Face → browser | `tokens.json` (78 KB), same commit | Every page load: the app does not cache it |
| Browser → ML server | `GET /health` | Once at page load (5 s timeout); the app does not re-check later |
| Browser → ML server | `POST /lipread/crops`: t × 88 × 88 uint8 gray mouth crops, gzipped | Each Quality sentence when it locks |
| ML server → browser | `text`, up to 3 `alternatives`, per-word confidence (`words`), `latency_ms` | Reply to the above |
| Browser → ML server | `POST /training-pairs`: t × 96 × 96 mouth crops + the confirmed text | Only with the opt-in on, when the user picks or types a fix |
| ML server → Hugging Face | The pair as `id.npz` + `id.txt` + `id.json` | Only if `LIPREAD_PAIRS_REPO` is set on the server |
| Browser → backend | Text of each finished line over the TTS WebSocket | Signed in and not muted |
| Backend → browser | 24 kHz 16-bit mono PCM | Reply to the above |
| Browser → backend | Microphone audio, 16 kHz 16-bit mono PCM in ~100 ms frames, over `/ws/stt` | While the listening panel runs |
| Backend → ElevenLabs | Text to speak, audio to transcribe, voice list | API key stays on the backend |
| Browser → TiDB phrase service | Phrase search and saves | Not built: needs the backend team's service and `VITE_PHRASES_URL` |

## Source of truth

- `frontend/src/pages/app-page.tsx`
- `frontend/src/hooks/useLipReader.ts`
- `frontend/src/lib/lipreading/createRecognizers.ts`
- `frontend/src/lib/lipreading/onnxRecognizer.ts`
- `frontend/src/lib/lipreading/httpRecognizer.ts`
- `frontend/src/lib/lipreading/modelSpec.ts`
- `frontend/src/lib/lipreading/trainingPairs.ts`
- `frontend/src/lib/lipreading/faceTracker.worker.ts`
- `frontend/src/lib/lipreading/faceQuality.ts`
- `frontend/src/lib/phrases/store.ts`
- `frontend/src/lib/backend/config.ts`
- `frontend/src/lib/backend/tts.ts`
- `frontend/src/lib/listening/sttEngine.ts`
- `ml/src/lipread/serve/app.py`
- `ml/src/lipread/model.py`
- `ml/scripts/download_checkpoints.sh`
- `backend/main.py`
- `backend/config.py`
- `backend/websocket_server.py`
- `backend/api/stt.py`
