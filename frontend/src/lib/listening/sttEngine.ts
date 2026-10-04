import { STT_SAMPLE_RATE, STT_WS_URL } from "@/lib/backend/config"
import { ACCENT_COLORS } from "@/lib/palette"
import type { ListeningEngine, ListeningEvents } from "./types"

/** ~100 ms of 16 kHz audio per frame, the cadence the backend expects. */
const FRAME_SAMPLES = STT_SAMPLE_RATE / 10

interface SttEvent {
  type: "ready" | "utterance" | "drop" | "error"
  id?: string
  text?: string
  final?: boolean
  speaker?: string | null
  message?: string
}

/**
 * Live listening via the backend `/ws/stt` (ElevenLabs Scribe + optional speaker diarization).
 *
 * Captures the mic, resamples to 16 kHz mono PCM16, and streams ~100 ms frames continuously; parses
 * the server's utterance/drop events into diarized utterances. Speaker labels only appear when the
 * backend runs with DIARIZATION=1 — otherwise `speaker` is null and everything shows unlabeled.
 */
export class SttListeningEngine implements ListeningEngine {
  readonly name = "Scribe (live)"
  readonly isReal = true

  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private processor: ScriptProcessorNode | null = null
  private events: ListeningEvents | null = null
  private readonly seenSpeakers = new Set<string>()
  private carry: number[] = []
  private stopped = false

  async start(events: ListeningEvents): Promise<void> {
    this.events = events
    this.stopped = false

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: false,
    })
    if (this.stopped) return

    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext
    const ctx = new Ctor()
    this.ctx = ctx
    if (ctx.state === "suspended") await ctx.resume()

    this.connect()

    const inRate = ctx.sampleRate
    const ratio = STT_SAMPLE_RATE / inRate
    this.source = ctx.createMediaStreamSource(this.stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    this.processor = processor
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0)
      // Linear resample input → 16 kHz, accumulate, flush in 100 ms frames.
      const outLen = Math.floor(input.length * ratio)
      for (let i = 0; i < outLen; i++) {
        const pos = i / ratio
        const i0 = Math.floor(pos)
        const i1 = Math.min(i0 + 1, input.length - 1)
        this.carry.push(input[i0] + (input[i1] - input[i0]) * (pos - i0))
      }
      while (this.carry.length >= FRAME_SAMPLES) {
        this.sendPcm(this.carry.splice(0, FRAME_SAMPLES))
      }
    }
    // Pull the graph without routing mic to the speakers (gain 0 = no feedback).
    const mute = ctx.createGain()
    mute.gain.value = 0
    this.source.connect(processor)
    processor.connect(mute)
    mute.connect(ctx.destination)
  }

  private connect(): void {
    const ws = new WebSocket(STT_WS_URL)
    ws.binaryType = "arraybuffer"
    this.ws = ws
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") this.handleEvent(JSON.parse(ev.data))
    }
    ws.onclose = (ev) => {
      if (!this.stopped && ev.code === 4401) {
        console.warn("[stt] listening socket rejected (sign-in required)")
      }
    }
    ws.onerror = () => {
      if (!this.stopped) console.warn("[stt] listening socket error")
    }
  }

  private sendPcm(floats: number[]): void {
    const ws = this.ws
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const buf = new ArrayBuffer(floats.length * 2)
    const view = new DataView(buf)
    for (let i = 0; i < floats.length; i++) {
      const s = Math.max(-1, Math.min(1, floats[i]))
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }
    ws.send(buf)
  }

  private handleEvent(msg: SttEvent): void {
    const events = this.events
    if (!events) return
    switch (msg.type) {
      case "utterance":
        if (msg.id && typeof msg.text === "string") {
          this.emitUtterance(msg.id, msg.text, !!msg.final, msg.speaker ?? null)
        }
        break
      case "drop":
        if (msg.id) events.onDrop?.(msg.id)
        break
      case "error":
        console.warn("[stt]", msg.message)
        break
    }
  }

  private emitUtterance(
    id: string,
    text: string,
    final: boolean,
    speaker: string | null
  ): void {
    const events = this.events
    if (!events) return

    if (speaker && !this.seenSpeakers.has(speaker)) {
      const index = speakerIndex(speaker, this.seenSpeakers.size)
      this.seenSpeakers.add(speaker)
      events.onSpeaker({
        id: speaker,
        name: `Speaker ${index + 1}`,
        colorVar: ACCENT_COLORS[index % ACCENT_COLORS.length],
      })
    }

    events.onUtterance({
      id,
      speakerId: speaker ?? "unknown",
      text,
      at: performance.now(),
      final,
    })
  }

  stop(): void {
    this.stopped = true
    if (this.processor) this.processor.onaudioprocess = null
    this.processor?.disconnect()
    this.source?.disconnect()
    this.ws?.close()
    this.stream?.getTracks().forEach((t) => t.stop())
    void this.ctx?.close()
    this.processor = null
    this.source = null
    this.ws = null
    this.stream = null
    this.ctx = null
    this.events = null
    this.carry = []
    this.seenSpeakers.clear()
  }
}

/** "speaker_0" → 0; falls back to appearance order when there's no number. */
function speakerIndex(speaker: string, fallback: number): number {
  const m = /(\d+)/.exec(speaker)
  return m ? Number(m[1]) : fallback
}
