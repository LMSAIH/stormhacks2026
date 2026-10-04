import { useSyncExternalStore } from "react"

/**
 * The Agentic Condom's on/off switch (lip-mode menu), remembered in this browser as
 * `lipread.condom` ("1"/"0"; the app eval sets it). It only runs where its server is configured
 * (`condomBaseUrl`), and never in Instant mode.
 */
export const CONDOM_DEFAULT_ON = true
const STORAGE_KEY = "lipread.condom"

const listeners = new Set<() => void>()
/** The switch for this session when storage is blocked (private mode): it still works until reload. */
let unsaved: boolean | null = null

export function condomEnabled(): boolean {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === "1" ? true : stored === "0" ? false : CONDOM_DEFAULT_ON
  } catch {
    return unsaved ?? CONDOM_DEFAULT_ON
  }
}

export function setCondomEnabled(on: boolean): void {
  unsaved = on
  try {
    localStorage.setItem(STORAGE_KEY, on ? "1" : "0")
  } catch {
    // not remembered past this session; fine
  }
  for (const listener of listeners) listener()
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange)
  return () => {
    listeners.delete(onChange)
  }
}

/** The switch, re-rendering when it flips. */
export function useCondomEnabled(): boolean {
  return useSyncExternalStore(subscribe, condomEnabled, () => CONDOM_DEFAULT_ON)
}
