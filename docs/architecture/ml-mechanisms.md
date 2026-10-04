# ML mechanisms — how *heard* works

A mechanism-level view of the three ML pipelines behind the app, plus the model build/train
pipeline that feeds them. Green = ours (`frontend/`, `ml/`); yellow = backend team (`backend/`);
grey = outside services. Dashed = planned / optional / opt-in.

---

## 1. System overview — the two live flows

*heard* runs two independent real-time loops at once:

- **You → voice** (silent speech): you mouth words at the webcam, the browser lip-reads them
  on-device, and ElevenLabs speaks the text aloud in your chosen voice.
- **Them → captions** (listening): the mic streams to the server, which transcribes *and*
  diarizes nearby speech into speaker-labeled captions. A toggle can pause this.

Everything the two loops produce is recorded, message-by-message, into a conversation in Timescale.

```mermaid
flowchart LR
  subgraph browser["Browser — our React app (frontend/)"]
    cam["Webcam 640×480"]
    lips["On-device lip reader<br/>(int8 ONNX, onnxruntime-web)"]
    mic["Microphone"]
    feed["Live captions + your transcript"]
    rec["Conversation recorder"]
    spk(["Speakers 🔊"])
    cam --> lips
  end

  subgraph be["Backend (backend/) — one asyncio process"]
    tts["TTS WS :8765"]
    stt["STT+diarization WS<br/>:5000 /ws/stt"]
    rest["REST :5000<br/>auth · voices · /api/chats"]
    db[("TimescaleDB<br/>user_chats, voice prefs")]
    rest --> db
  end

  subgraph ext["Outside services"]
    el["ElevenLabs<br/>TTS · Scribe STT"]
    google["Google OAuth"]
  end

  mlsrv["Our ML server (ml/)<br/>FastAPI on RunPod 4090<br/>beam + LM (accuracy path)"]:::ours

  lips -->|"finalized text"| tts
  tts --> el --> tts -->|"24 kHz PCM"| spk
  mic -->|"16 kHz PCM"| stt
  stt <--> el
  stt -->|"speaker-labeled utterances"| feed
  lips -.->|"opt-in: 88×88 crops, beam re-decode"| mlsrv
  lips --> rec
  feed --> rec
  rec -->|"POST/PUT /api/chats"| rest
  rest <--> google

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
  classDef backendteam fill:#fef7e0,stroke:#e37400,color:#000
  classDef outside fill:#f1f3f4,stroke:#80868b,color:#000
  class browser,lips,rec ours
  class be,tts,stt,rest,db backendteam
  class ext,el,google,mlsrv outside
```

---

## 2. Lip-reading pipeline (visual speech recognition)

The whole read happens **in the browser** — no camera video ever leaves the device. Face tracking
runs in a Web Worker so the capture loop never stalls; a **visual VAD** (lip *deformation*, head
motion removed) decides when an utterance starts and stops, so it's streaming, not push-to-talk.
The crop is a bit-exact port of the Python training preprocessing, so the ONNX model sees exactly
what it was trained on.

```mermaid
flowchart TB
  cap["Capture loop<br/>requestVideoFrameCallback<br/>→ grayscale (Rec.601 fixed-point)"]

  subgraph worker["Face tracker (Web Worker)"]
    blaze["BlazeFace detector<br/>4 keypoints: eyes, nose, mouth"]
    mesh["FaceLandmarker (mesh)<br/>40 lip points + openness"]
  end

  vad{"Visual VAD<br/>lip deformation / mouth width<br/>EMA 0.6 · start 0.035 / keep 0.018<br/>hysteresis + 1 s lead"}

  subgraph crop["Mouth-crop pipeline (crop/) — bit-exact port of preprocess.py"]
    rs["Resample → 25 fps"]
    interp["Interpolate missing keypoints"]
    smooth["Temporal smoothing ±6 frames"]
    sim["Similarity warp → 20-word mean face<br/>(estimateAffinePartial2D / LMEDS)"]
    patch["Cut 96×96 mouth patch"]
    rs --> interp --> smooth --> sim --> patch
  end

  norm["Normalize<br/>center-crop 96→88 · /255<br/>mean 0.421, std 0.165<br/>tensor [1,1,T,88,88]"]

  onnx["int8 ONNX encoder + CTC head<br/>dyn-pw8-rn16, 203 MB<br/>onnxruntime-web WASM (WebGPU opt-in)<br/>HF eschmechel/...-int8-onnx (pinned)"]
  greedy["Greedy CTC decode<br/>argmax → merge repeats → drop blank<br/>SentencePiece → text + per-word confidence"]
  out["Transcript line<br/>text · up to 3 choices · confidence<br/>phrase-snapping · word-confidence boxes"]

  cap --> worker
  blaze --> crop
  mesh --> vad
  vad -->|"utterance window"| crop
  patch --> norm --> onnx --> greedy --> out

  acc["ml/ FastAPI (accuracy path)<br/>POST /lipread/crops?decode=beam<br/>beam 20 + RNN-LM + attention decoder<br/>same model, GPU on RunPod"]:::ours
  norm -.->|"quality mode / opt-in: gzipped 88×88 crops"| acc
  acc -.->|"text, n-best, word confidences"| out

  tts["→ ElevenLabs TTS (speak aloud)"]
  out --> tts

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
  class acc ours
```

**Two tiers, one model.** Speed path = int8 ONNX + greedy CTC in the browser. Accuracy path =
the *same* Auto-AVSR encoder/CTC in PyTorch on the server, but decoded with beam search + RNN-LM +
the attention decoder (none of which are in the ONNX graph). I/O, tokens (`unigram5000`, 5049), and
preprocessing are kept byte-identical so the two are interchangeable.

---

## 3. Listening pipeline (STT + diarization)

The mic streams 16 kHz PCM to one WebSocket (`/ws/stt`). The server fans it to **two** consumers in
parallel: ElevenLabs Scribe for the words, and our diarizer for *who said them*. Word-level
timestamps from Scribe are aligned against the diarizer's speaker timeline to split each committed
transcript into per-speaker pieces. Diarization is feature-gated (`DIARIZATION=1`); when off, the
captions still work, just without speaker labels.

```mermaid
flowchart TB
  micin["Browser mic<br/>echoCancellation on, AGC/NS off<br/>resample → 16 kHz s16le, ~100 ms frames"]
  ws["/ws/stt (FastAPI :5000)<br/>SttSession — cookie-auth"]
  micin -->|"PCM frames"| ws

  subgraph scribe["Words (ElevenLabs Scribe)"]
    sc["scribe_v2_realtime<br/>pcm_16000 · commit=vad<br/>language_code=en (English only)<br/>include_timestamps when diarizing"]
  end

  subgraph diar["Diarization (diarization/) — runs in a 2-thread pool"]
    v["Silero VAD<br/>512-sample frames, speech/silence"]
    emb["ECAPA-TDNN embedder<br/>192-d voiceprint, 3 s window, L2-norm"]
    clus["Online centroid clustering<br/>cosine sim ≥ threshold 0.45 → join<br/>else new speaker_N<br/>2-recheck switch confirm"]
    v --> emb --> clus
  end

  ws --> scribe
  ws --> diar

  merge["Align words ↔ speaker timeline<br/>split_by_speaker (word midpoint time)<br/>group contiguous → Piece(speaker, text)"]
  gate{"TTS echo gate<br/>drop our own spoken output"}
  utt["utterance events<br/>{id, text, final, speaker}"]
  capfeed["Speaker-labeled caption feed<br/>Speaker 1/2… + accent colors"]

  scribe -->|"committed words + timestamps"| merge
  clus -->|"speaker_at(t)"| merge
  merge --> gate --> utt --> capfeed

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
```

**Live config** (deployed `.env`): Silero VAD + ECAPA-TDNN, cosine threshold `0.45`, 3 s voiceprint
window, 2 s min speech, 0.8 s hangover. Speaker centroids adapt over time (running weighted average)
so a voice keeps matching as it drifts.

---

## 4. Text-to-speech path (speaking your words)

A dedicated raw-WebSocket server on `:8765`. Each finalized lip-read line is sent as text + a
`\x` terminator that flushes it to ElevenLabs; audio streams back as 24 kHz PCM and plays gaplessly
via Web Audio. The voice is resolved from the user's saved preference at connect time.

```mermaid
flowchart LR
  line["Finalized lip-read line"]
  client["TtsClient (browser)<br/>text + '\\x' terminator<br/>plays 24 kHz PCM gaplessly"]
  srv["TTS WS :8765 — cookie-auth<br/>resolve user voice → backend<br/>split on terminator, flush=true"]
  flash["FlashWsBackend (default)<br/>eleven_flash_v2_5<br/>stream-input, pcm_24000"]
  el["ElevenLabs TTS"]

  line --> client -->|"text"| srv --> flash <--> el
  el -->|"PCM audio"| client -->|"🔊"| spk(["Speakers"])

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
  class client ours
```

---

## 5. Model build & training pipeline (offline)

How the browser's int8 model and the server's fine-tuned checkpoints are produced. The int8 export
is gated by a regression test (WER delta, argmax agreement, a sha + exact-text lock) before it can
ship. The **LLM corrector** (Llama-3.2-3B LoRA via Unsloth) is a *passthrough hook* today — the
plumbing exists, the fine-tuned model is post-MVP.

```mermaid
flowchart TB
  ckpt[("PyTorch checkpoint<br/>HF Amanvir/LRS3_V_WER19.1<br/>+ lm_en_subword RNN-LM")]
  exp["export_onnx.py<br/>encoder+CTC → fp32 ONNX (~775 MB)<br/>opset 17, dynamic T"]
  quant["quantize_onnx.py<br/>recipe dyn-pw8-rn16 → 203 MB int8<br/>(WASM-safe ops only)"]
  gate{"regress_quantized.py (gate)<br/>size ≤220 MB · WER Δ ≤+1.0<br/>argmax ≥0.95 · sha+text lock"}
  hf[("HF eschmechel/<br/>auto-avsr-lrs3-vsr-int8-onnx<br/>pinned commit")]
  browser["Browser speed path"]:::ours

  ckpt --> exp --> quant --> gate -->|"pass"| hf --> browser

  subgraph ft["Fine-tune (RunPod 4090)"]
    rec[("Team recordings +<br/>opt-in training pairs")]
    train["finetune.py (auto_avsr recipe)<br/>→ WiSE-FT blend"]
    conv["convert_ckpt.py<br/>auto_avsr ↔ Chaplin layout"]
    serve["ml/ FastAPI accuracy path"]
    rec --> train --> conv --> serve
  end
  conv -.->|"re-export / re-quantize"| exp

  corr["Corrector (post-MVP)<br/>Llama-3.2-3B LoRA / Unsloth / RunPod<br/>passthrough hook until built"]:::planned
  serve -.-> corr

  classDef ours fill:#e6f4ea,stroke:#1e8e3e,color:#000
  classDef planned fill:#fff,stroke:#80868b,color:#000,stroke-dasharray:5 5
  class exp,quant,gate,ft,train,conv,serve,rec ours
```

---

## Source of truth

- `frontend/src/hooks/useLipReader.ts`
- `frontend/src/lib/lipreading/`
- `frontend/src/lib/listening/`
- `frontend/src/lib/backend/`
- `frontend/src/hooks/useConversationRecorder.ts`
- `backend/main.py`
- `backend/websocket_server.py`
- `backend/api/`
- `backend/stt/`
- `backend/diarization/`
- `backend/tts/`
- `ml/src/lipread/`
- `ml/scripts/`
- `ml/runpod/`
