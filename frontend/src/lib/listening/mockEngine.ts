import {
  SPEAKER_COLORS,
  type ListeningEngine,
  type ListeningEvents,
  type Speaker,
} from "./types"

/** Scripted conversation used to drive the diarized chat UI before real ASR. */
const SCRIPT: { speaker: number; text: string }[] = [
  { speaker: 0, text: "Hey, did you get a chance to look at the designs?" },
  { speaker: 1, text: "Yeah, I think the camera panel should stay small." },
  { speaker: 0, text: "Agreed. Let's keep it in the corner for debugging." },
  { speaker: 2, text: "What about the listening mode transcript?" },
  { speaker: 1, text: "That's the chat view with editable speaker names." },
  { speaker: 0, text: "Nice. We can rename people as we figure out who's who." },
  { speaker: 2, text: "Perfect, that makes the diarization way more useful." },
]

let idCounter = 0
const nextId = () => `mock-${idCounter++}`

/**
 * Emits a scripted, diarized conversation on a timer — streaming each line in
 * word-by-word (partial → final) so the UI's interim/finalized states are
 * exercised exactly as they will be with a real streaming ASR backend.
 */
export class MockListeningEngine implements ListeningEngine {
  readonly name = "Mock diarization (no ASR loaded)"
  readonly isReal = false
  private timer: ReturnType<typeof setTimeout> | null = null
  private lineIndex = 0
  private readonly speakers = new Map<number, Speaker>()

  async start(events: ListeningEvents): Promise<void> {
    this.lineIndex = 0
    this.scheduleNextLine(events)
  }

  private scheduleNextLine(events: ListeningEvents): void {
    const line = SCRIPT[this.lineIndex % SCRIPT.length]
    this.lineIndex++

    const speaker = this.ensureSpeaker(line.speaker, events)
    const utteranceId = nextId()
    const words = line.text.split(" ")
    let shown = 0

    const streamWord = () => {
      shown++
      const text = words.slice(0, shown).join(" ")
      const final = shown === words.length
      events.onUtterance({
        id: utteranceId,
        speakerId: speaker.id,
        text,
        at: performance.now(),
        final,
      })

      if (!final) {
        this.timer = setTimeout(streamWord, 90 + Math.min(shown * 7, 120))
      } else {
        // Pause, then move to the next line.
        this.timer = setTimeout(() => this.scheduleNextLine(events), 900)
      }
    }

    this.timer = setTimeout(streamWord, 400)
  }

  private ensureSpeaker(index: number, events: ListeningEvents): Speaker {
    const existing = this.speakers.get(index)
    if (existing) return existing

    const speaker: Speaker = {
      id: `speaker-${index}`,
      name: `Speaker ${index + 1}`,
      colorVar: SPEAKER_COLORS[index % SPEAKER_COLORS.length],
    }
    this.speakers.set(index, speaker)
    events.onSpeaker(speaker)
    return speaker
  }

  stop(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
    this.speakers.clear()
  }
}
