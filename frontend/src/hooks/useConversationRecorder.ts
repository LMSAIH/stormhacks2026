import { useCallback, useEffect, useRef } from "react"

import type { LipTranscriptItem } from "@/hooks/useLipReader"
import type { Speaker, Utterance } from "@/lib/listening/types"
import { createConversation, saveConversation } from "@/lib/conversations/api"
import { newConversation } from "@/lib/conversations/store"
import type { Conversation, NoteEntry, Participant } from "@/lib/conversations/types"

interface RecorderInput {
  /** The signed-in user's id; conversations are scoped to it. */
  userId: string | undefined
  /** Your finalized lip-read utterances (each is a completed message). */
  lipItems: LipTranscriptItem[]
  /** Nearby speakers' utterances (we record the final ones). */
  listeningUtterances: Utterance[]
  /** Detected speakers, for names/colors (and live renames). */
  speakers: Record<string, Speaker>
}

const UNKNOWN: Participant = {
  id: "unknown",
  name: "Speaker",
  colorVar: "var(--muted-foreground)",
}

/**
 * Records the current session as a conversation in the backend (`/api/chats` → Timescale),
 * scoped to the signed-in user.
 *
 * A new conversation is created when the app mounts; each completed utterance — yours ("You") and
 * everyone else's — is appended as its own timestamped message as soon as it finalizes, and the
 * conversation is saved automatically. It only persists once it has at least one message, so idle
 * sessions don't litter the archive. Writes are serialized through a promise chain so the first
 * message creates the row (yielding a server id) and every later message updates it.
 */
export function useConversationRecorder({
  userId,
  lipItems,
  listeningUtterances,
  speakers,
}: RecorderInput): void {
  const convRef = useRef<Conversation | null>(null)
  const appended = useRef<Set<string>>(new Set())
  const serverId = useRef<string | null>(null)
  const saveChain = useRef<Promise<void>>(Promise.resolve())

  const persist = useCallback(() => {
    if (!userId) return
    // Serialize writes: create the row on the first flush, then update it.
    saveChain.current = saveChain.current.then(async () => {
      const conv = convRef.current
      if (!conv || conv.entries.length === 0) return
      try {
        if (!serverId.current) {
          const { id, startedAt } = await createConversation(conv)
          serverId.current = id
          if (convRef.current) convRef.current = { ...convRef.current, id, startedAt }
        } else {
          await saveConversation(serverId.current, conv)
        }
      } catch {
        // Best-effort: a failed write is retried on the next completed utterance.
      }
    })
  }, [userId])

  // New conversation per session (once the user is known).
  useEffect(() => {
    convRef.current = userId ? newConversation(Date.now()) : null
    appended.current = new Set()
    serverId.current = null
    saveChain.current = Promise.resolve()
  }, [userId])

  // Your lip-read lines → "You" messages.
  useEffect(() => {
    const conv = convRef.current
    if (!conv || !userId) return
    const additions: NoteEntry[] = []
    for (const item of lipItems) {
      const mid = `you-${item.id}`
      if (appended.current.has(mid)) continue
      appended.current.add(mid)
      additions.push({ id: mid, speakerId: "you", text: item.text, at: Date.now() })
    }
    if (additions.length) {
      convRef.current = { ...conv, entries: [...conv.entries, ...additions] }
      persist()
    }
  }, [lipItems, userId, persist])

  // Others' finalized captions → speaker messages.
  useEffect(() => {
    const conv = convRef.current
    if (!conv || !userId) return
    const newEntries: NoteEntry[] = []
    const participants = [...conv.participants]
    for (const u of listeningUtterances) {
      if (!u.final) continue
      const mid = `sp-${u.id}`
      if (appended.current.has(mid)) continue
      appended.current.add(mid)
      const sp = speakers[u.speakerId]
      const p = sp ? { id: sp.id, name: sp.name, colorVar: sp.colorVar } : UNKNOWN
      if (!participants.find((x) => x.id === p.id)) participants.push(p)
      newEntries.push({ id: mid, speakerId: p.id, text: u.text, at: Date.now() })
    }
    if (newEntries.length) {
      convRef.current = { ...conv, participants, entries: [...conv.entries, ...newEntries] }
      persist()
    }
  }, [listeningUtterances, speakers, userId, persist])

  // Keep participant names in sync when a speaker is renamed live.
  useEffect(() => {
    const conv = convRef.current
    if (!conv || !userId) return
    let changed = false
    const participants = conv.participants.map((p) => {
      const sp = speakers[p.id]
      if (sp && sp.name !== p.name) {
        changed = true
        return { ...p, name: sp.name }
      }
      return p
    })
    if (changed) {
      convRef.current = { ...conv, participants }
      persist()
    }
  }, [speakers, userId, persist])
}
