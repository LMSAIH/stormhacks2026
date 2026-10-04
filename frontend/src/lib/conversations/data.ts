import { colorForString } from "@/lib/palette"
import type { Conversation, Participant } from "./types"

/**
 * Sample past conversations. Temporary seed data until the backend is wired up
 * (see ./api.ts). Deterministic timestamps (no Date.now()).
 */

const YOU: Participant = { id: "you", name: "You", colorVar: "var(--primary)" }
const person = (name: string): Participant => ({
  id: name.toLowerCase(),
  name,
  colorVar: colorForString(name),
})

const DAY = 86_400_000
const BASE = Date.UTC(2026, 9, 3, 9, 0, 0) // Oct 3, 2026 09:00 UTC

/** Build a conversation from terse [speaker, text] lines. */
function conv(
  id: string,
  title: string,
  daysAgo: number,
  hour: number,
  lines: [string, string][]
): Conversation {
  const start = BASE - daysAgo * DAY + hour * 3_600_000
  const participants = new Map<string, Participant>([["you", YOU]])
  const entries = lines.map(([who, text], i) => {
    const sp = who === "You" ? YOU : person(who)
    participants.set(sp.id, sp)
    return { id: `${id}-${i}`, speakerId: sp.id, text, at: start + i * 18_000 }
  })
  return { id, title, startedAt: start, participants: [...participants.values()], entries }
}

export const SAMPLE_CONVERSATIONS: Conversation[] = [
  conv("design-review", "Design review", 1, 7, [
    ["Maya", "Let's look at the new conversation layout."],
    ["You", "Camera on the left, transcript on the right."],
    ["Theo", "The split feels balanced. What about speaker colors?"],
    ["You", "Each speaker gets a vivid accent from a shared palette."],
    ["Maya", "The avatars make diarization much clearer."],
  ]),
  conv("morning-standup", "Morning standup", 2, 0, [
    ["Priya", "Quick round — what's everyone on today?"],
    ["You", "Wiring the notes route and the search box."],
    ["Sam", "Finishing the export format for transcripts."],
    ["Leo", "Looking into the lip-reading model conversion."],
  ]),
  conv("coffee-chat", "Coffee chat with Nina", 5, 2, [
    ["Nina", "How's the hackathon project coming along?"],
    ["You", "Good! Real-time lip reading in the browser."],
    ["Nina", "Does it run fully on device?"],
    ["You", "Yes, ONNX Runtime with MediaPipe for the mouth crop."],
  ]),
  conv("client-onboarding", "Client onboarding call", 3, 5, [
    ["Dana", "Thanks for setting up our workspace."],
    ["You", "Of course — let's walk through the basics."],
    ["Dana", "How do we invite the rest of the team?"],
    ["You", "From settings, under members."],
  ]),
  conv("sprint-retro", "Sprint retro", 4, 6, [
    ["Omar", "What went well this sprint?"],
    ["You", "The pipeline refactor landed cleanly."],
    ["Grace", "Testing coverage went up a lot."],
    ["Omar", "Let's keep the momentum next sprint."],
  ]),
  conv("one-on-one-priya", "1:1 with Priya", 6, 3, [
    ["Priya", "How are you feeling about the roadmap?"],
    ["You", "Confident, but the timeline is tight."],
    ["Priya", "Let's find something to cut scope on."],
  ]),
  conv("product-roadmap", "Product roadmap", 7, 4, [
    ["Theo", "Q4 is all about the notes archive."],
    ["You", "And vector search on the backend."],
    ["Maya", "We should prototype search first."],
    ["Theo", "Agreed, it de-risks the rest."],
  ]),
  conv("bug-triage", "Bug triage", 8, 1, [
    ["Sam", "We have three P1s open."],
    ["You", "I'll take the camera init race condition."],
    ["Leo", "I'll look at the overlay flicker."],
  ]),
  conv("user-interview-alex", "User interview — Alex", 9, 8, [
    ["Alex", "I'd use this in noisy meetings for sure."],
    ["You", "What matters most — speed or accuracy?"],
    ["Alex", "Accuracy, as long as it's close to live."],
  ]),
  conv("marketing-sync", "Marketing sync", 10, 2, [
    ["Grace", "We need a tagline for the launch."],
    ["You", "Something about reading the room."],
    ["Dana", "I like that direction."],
  ]),
  conv("investor-update", "Investor update", 12, 5, [
    ["You", "Usage doubled since the beta."],
    ["Rosa", "What's driving retention?"],
    ["You", "The notes archive keeps people coming back."],
  ]),
  conv("architecture-discussion", "Architecture discussion", 13, 6, [
    ["Leo", "Should search live on the edge or a server?"],
    ["You", "Server — we want vector search."],
    ["Sam", "We can keep the client interface identical."],
    ["Leo", "That makes the swap painless."],
  ]),
  conv("hiring-debrief", "Hiring debrief", 14, 3, [
    ["Priya", "Thoughts on the frontend candidate?"],
    ["You", "Strong on React, great design sense."],
    ["Omar", "Agreed, let's move to an offer."],
  ]),
  conv("weekly-all-hands", "Weekly all-hands", 15, 7, [
    ["Rosa", "Big week — the notes feature shipped."],
    ["You", "Thanks to everyone who pushed on search."],
    ["Grace", "The demo got a great response."],
  ]),
  conv("support-escalation", "Support escalation", 16, 1, [
    ["Dana", "A user can't export their transcript."],
    ["You", "Looking now — likely a blob issue."],
    ["Dana", "Thanks, keep me posted."],
  ]),
  conv("brainstorm-naming", "Brainstorm: naming", 18, 4, [
    ["Maya", "Working names for the product?"],
    ["You", "Lipreader is clear, maybe too literal."],
    ["Theo", "Let's shortlist five and vote."],
  ]),
  conv("demo-rehearsal", "Demo rehearsal", 20, 6, [
    ["Omar", "Run it top to bottom once."],
    ["You", "Starting with the live conversation view."],
    ["Grace", "Remember to show the export."],
    ["Omar", "Looks tight. Ship it."],
  ]),
  conv("lunch-with-theo", "Lunch with Theo", 22, 5, [
    ["Theo", "The avatars really pull the UI together."],
    ["You", "Solid colors read better than tints."],
    ["Theo", "Totally. Clean and recognizable."],
  ]),
]
