import { ACCENT_COLORS } from "@/lib/palette"
import type { Conversation } from "./types"

/**
 * Sample past conversations. Temporary seed data until the backend is wired up
 * (see ./api.ts). Timestamps use Date.UTC so they're deterministic.
 */

const YOU = { id: "you", name: "You", colorVar: "var(--primary)" }

// minutes/seconds helper for readable offsets.
const t = (base: number, min: number, sec = 0) =>
  base + min * 60_000 + sec * 1000

const c1Start = Date.UTC(2026, 9, 2, 16, 30, 0) // Oct 2, 2026
const c2Start = Date.UTC(2026, 9, 1, 9, 0, 0) // Oct 1, 2026
const c3Start = Date.UTC(2026, 8, 28, 11, 15, 0) // Sep 28, 2026

export const SAMPLE_CONVERSATIONS: Conversation[] = [
  {
    id: "design-review",
    title: "Design review",
    startedAt: c1Start,
    participants: [
      YOU,
      { id: "maya", name: "Maya", colorVar: ACCENT_COLORS[0] },
      { id: "theo", name: "Theo", colorVar: ACCENT_COLORS[1] },
    ],
    entries: [
      { id: "c1-1", speakerId: "maya", text: "Thanks for joining — let's look at the new conversation layout.", at: t(c1Start, 0, 12) },
      { id: "c1-2", speakerId: "you", text: "I kept the camera on the left and the transcript on the right.", at: t(c1Start, 0, 28) },
      { id: "c1-3", speakerId: "theo", text: "The split feels balanced. What about the speaker colors?", at: t(c1Start, 0, 51) },
      { id: "c1-4", speakerId: "you", text: "Each speaker gets a vivid accent from a shared palette.", at: t(c1Start, 1, 9) },
      { id: "c1-5", speakerId: "maya", text: "Love the avatars. They make diarization much clearer.", at: t(c1Start, 1, 33) },
      { id: "c1-6", speakerId: "theo", text: "Let's ship it and revisit the notes archive next week.", at: t(c1Start, 2, 2) },
    ],
  },
  {
    id: "morning-standup",
    title: "Morning standup",
    startedAt: c2Start,
    participants: [
      YOU,
      { id: "priya", name: "Priya", colorVar: ACCENT_COLORS[2] },
      { id: "sam", name: "Sam", colorVar: ACCENT_COLORS[3] },
      { id: "leo", name: "Leo", colorVar: ACCENT_COLORS[4] },
    ],
    entries: [
      { id: "c2-1", speakerId: "priya", text: "Quick round — what's everyone on today?", at: t(c2Start, 0, 8) },
      { id: "c2-2", speakerId: "you", text: "Wiring the notes route and the search box.", at: t(c2Start, 0, 20) },
      { id: "c2-3", speakerId: "sam", text: "I'm finishing the export format for transcripts.", at: t(c2Start, 0, 37) },
      { id: "c2-4", speakerId: "leo", text: "Looking into the lip-reading model conversion.", at: t(c2Start, 0, 55) },
      { id: "c2-5", speakerId: "priya", text: "Great. Any blockers before we wrap?", at: t(c2Start, 1, 14) },
      { id: "c2-6", speakerId: "you", text: "None on my end, search is nearly done.", at: t(c2Start, 1, 26) },
    ],
  },
  {
    id: "coffee-chat",
    title: "Coffee chat with Nina",
    startedAt: c3Start,
    participants: [
      YOU,
      { id: "nina", name: "Nina", colorVar: ACCENT_COLORS[5] },
    ],
    entries: [
      { id: "c3-1", speakerId: "nina", text: "How's the hackathon project coming along?", at: t(c3Start, 0, 10) },
      { id: "c3-2", speakerId: "you", text: "Good! Real-time lip reading in the browser.", at: t(c3Start, 0, 24) },
      { id: "c3-3", speakerId: "nina", text: "That's wild. Does it run fully on device?", at: t(c3Start, 0, 40) },
      { id: "c3-4", speakerId: "you", text: "Yes, ONNX Runtime with MediaPipe for the mouth crop.", at: t(c3Start, 0, 58) },
      { id: "c3-5", speakerId: "nina", text: "Can't wait to try it. Send me the link!", at: t(c3Start, 1, 20) },
    ],
  },
]
