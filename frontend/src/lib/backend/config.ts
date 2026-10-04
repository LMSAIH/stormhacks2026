/** Backend endpoints. Override via Vite env (VITE_API_URL / VITE_TTS_WS_URL). */

export const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:5000"

export const TTS_WS_URL: string =
  import.meta.env.VITE_TTS_WS_URL ?? "ws://localhost:8765"

/** Listening-mode STT/diarization socket (FastAPI route on the API port). */
export const STT_WS_URL: string =
  import.meta.env.VITE_STT_WS_URL ?? `${API_URL.replace(/^http/, "ws")}/ws/stt`

/** The STT endpoint wants 16 kHz mono signed-16-bit PCM. */
export const STT_SAMPLE_RATE = 16000

/** The backend streams pcm_24000 (24 kHz, 16-bit LE, mono). */
export const TTS_SAMPLE_RATE = 24000

/** Flush marker the TTS WebSocket splits text on (backend `TERMINATOR = "\x"`). */
export const TTS_TERMINATOR = "\\x"

/** Full URL that starts Google OAuth (full-page redirect). */
export const googleLoginUrl = (): string => `${API_URL}/api/auth/google/login`
