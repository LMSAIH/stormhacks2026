# TTS WebSocket server

Run: `uvicorn app.main:app --port 8000` from `backend/` (needs `ELEVENLABS_API_KEY` in `backend/.env`).

- `GET /health` -> `{"status":"ok","backends":["flash","v4"]}`
- `WS /ws/tts?backend=flash|v4` (default: `TTS_BACKEND` env, else `flash`)

One ElevenLabs session per connection, opened (pre-warmed) on connect and kept
across many turns. It is closed when the socket closes.

## Server -> client
1. First, JSON: `{"type":"ready","backend":"flash","sample_rate":24000,"codec":"pcm","encoding":"s16le","channels":1}`
   Wait for this before sending (the session is warm).
2. BINARY frames: raw PCM s16le mono chunks, forwarded as they arrive. Play them in order (queue them into an AudioWorklet / scheduled buffers).
3. JSON `{"type":"error","message":"..."}` on failures or bad input. Invalid JSON keeps the connection open.

## Client -> server (JSON text frames)
- `{"type":"text","text":"hello there","final":true}` - split on whitespace into words. `final` defaults to true; `final:false` (partials) are ignored, so send only committed words.
- `{"type":"end_turn"}` - flush remaining buffered text at the end of an utterance. The connection stays open for the next turn.

There is no barge-in: to interrupt, stop playback client-side (discard queued audio).
