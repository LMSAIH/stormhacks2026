import { TTS_SAMPLE_RATE, TTS_TERMINATOR, TTS_WS_URL } from "./config"

export type TtsStatus =
  | "idle"
  | "connecting"
  | "ready"
  | "needs-auth"
  | "error"

/**
 * Streaming text-to-speech client.
 *
 * Opens the backend WebSocket, sends each utterance as `text + TERMINATOR`, and plays the PCM
 * (24 kHz, 16-bit LE, mono) it streams back, scheduled gaplessly through the Web Audio API.
 * The backend reads the selected voice when the socket connects, so changing voice means
 * `reconnect()`.
 */
export class TtsClient {
  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  /** When the next scheduled chunk should start (ctx clock). */
  private nextStartTime = 0
  /** Text queued while the socket is still opening. */
  private pending: string[] = []
  private closedByUs = false
  private readonly onStatus: (status: TtsStatus) => void

  constructor(onStatus: (status: TtsStatus) => void) {
    this.onStatus = onStatus
  }

  /** Create/resume the AudioContext — must be called from a user gesture at least once. */
  async ensureAudio(): Promise<void> {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext
      this.ctx = new Ctor()
    }
    if (this.ctx.state === "suspended") await this.ctx.resume()
  }

  connect(): void {
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return
    }
    this.closedByUs = false
    this.onStatus("connecting")
    const ws = new WebSocket(TTS_WS_URL)
    ws.binaryType = "arraybuffer"
    this.ws = ws

    ws.onopen = () => {
      this.onStatus("ready")
      const queued = this.pending
      this.pending = []
      for (const text of queued) this.sendNow(text)
    }
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) this.playPcm(ev.data)
    }
    ws.onclose = (ev) => {
      this.ws = null
      // 4401 = backend "Sign-in required" close code.
      if (ev.code === 4401) this.onStatus("needs-auth")
      else if (!this.closedByUs) this.onStatus("idle")
    }
    ws.onerror = () => {
      if (!this.closedByUs) this.onStatus("error")
    }
  }

  /** Reconnect (e.g. after the selected voice changed). */
  reconnect(): void {
    this.disconnect()
    this.connect()
  }

  /** Speak one utterance. Resumes audio, connects if needed, queues until open. */
  async speak(text: string): Promise<void> {
    const trimmed = text.trim()
    if (!trimmed) return
    await this.ensureAudio()
    if (!this.ws || this.ws.readyState > WebSocket.OPEN) this.connect()
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.sendNow(trimmed)
    else this.pending.push(trimmed)
  }

  private sendNow(text: string): void {
    this.ws?.send(text + TTS_TERMINATOR)
  }

  private playPcm(buffer: ArrayBuffer): void {
    const ctx = this.ctx
    if (!ctx) return
    // 16-bit samples: ignore a trailing odd byte if a chunk split mid-sample.
    const samples = buffer.byteLength >> 1
    if (samples === 0) return
    const int16 = new Int16Array(buffer, 0, samples)
    const audioBuffer = ctx.createBuffer(1, samples, TTS_SAMPLE_RATE)
    const channel = audioBuffer.getChannelData(0)
    for (let i = 0; i < samples; i++) channel[i] = int16[i] / 32768

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)
    const start = Math.max(ctx.currentTime, this.nextStartTime)
    source.start(start)
    this.nextStartTime = start + audioBuffer.duration
  }

  disconnect(): void {
    this.closedByUs = true
    this.nextStartTime = 0
    this.pending = []
    const ws = this.ws
    this.ws = null
    ws?.close()
  }

  dispose(): void {
    this.disconnect()
    void this.ctx?.close()
    this.ctx = null
  }
}
