import { normalizeText } from "./lookalike"

/** Where a saved phrase came from: read and kept, picked from the top 3, or typed as a fix. */
export type PhraseSource = "accepted" | "picked" | "typed"

export interface Phrase {
  readonly id: string
  readonly text: string
  readonly count: number
  readonly source: PhraseSource
  readonly lastUsed: number
}

/** A phrase returned by `search`; `distance` is the store's own (vector) distance when it has one. */
export interface PhraseHit extends Phrase {
  readonly distance?: number
}

/**
 * The user's phrase bank. `IndexedDbPhraseStore` keeps it in the browser (stand-in, works offline);
 * `HttpPhraseStore` talks to the backend team's TiDB service. Same calls either way, so the app
 * switches with one setting (`VITE_PHRASES_URL`).
 */
export interface PhraseStore {
  /** Save a phrase, or bump its count if the same words are already there. */
  add(text: string, source: PhraseSource): Promise<Phrase>
  /** Candidates for a reading. The browser store returns everything; TiDB returns its nearest k. */
  search(reading: string, k?: number): Promise<PhraseHit[]>
  list(limit?: number): Promise<Phrase[]>
  remove(id: string): Promise<void>
}

const DB_NAME = "lipread-phrases"
const STORE = "phrases"

interface StoredPhrase extends Phrase {
  readonly norm: string
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export class IndexedDbPhraseStore implements PhraseStore {
  private db: Promise<IDBDatabase> | null = null
  private readonly factory: IDBFactory

  constructor(factory: IDBFactory = indexedDB) {
    this.factory = factory
  }

  private open(): Promise<IDBDatabase> {
    this.db ??= new Promise((resolve, reject) => {
      const req = this.factory.open(DB_NAME, 1)
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore(STORE, { keyPath: "id" })
        store.createIndex("norm", "norm", { unique: true })
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
    return this.db
  }

  private async tx(mode: IDBTransactionMode): Promise<IDBObjectStore> {
    return (await this.open()).transaction(STORE, mode).objectStore(STORE)
  }

  async add(text: string, source: PhraseSource): Promise<Phrase> {
    const norm = normalizeText(text)
    const store = await this.tx("readwrite")
    const existing = (await request(store.index("norm").get(norm))) as StoredPhrase | undefined
    const phrase: StoredPhrase = existing
      ? { ...existing, count: existing.count + 1, source, lastUsed: Date.now() }
      : { id: crypto.randomUUID(), text: text.trim(), norm, count: 1, source, lastUsed: Date.now() }
    await request(store.put(phrase))
    return strip(phrase)
  }

  async search(): Promise<PhraseHit[]> {
    return this.list() // small bank: rank everything with the look-alike score (snap.ts)
  }

  async list(limit = 500): Promise<Phrase[]> {
    const all = (await request((await this.tx("readonly")).getAll())) as StoredPhrase[]
    return all
      .sort((a, b) => b.count - a.count || b.lastUsed - a.lastUsed)
      .slice(0, limit)
      .map(strip)
  }

  async remove(id: string): Promise<void> {
    await request((await this.tx("readwrite")).delete(id))
  }
}

function strip(p: StoredPhrase): Phrase {
  return { id: p.id, text: p.text, count: p.count, source: p.source, lastUsed: p.lastUsed }
}

/** The backend team's TiDB phrase service (spec: .context/streaming-plan.md, "backend"). */
export class HttpPhraseStore implements PhraseStore {
  private readonly baseUrl: string
  private readonly userId: string
  private readonly fetchImpl: typeof fetch

  constructor(baseUrl: string, userId: string, fetchImpl: typeof fetch = fetch) {
    this.baseUrl = baseUrl
    this.userId = userId
    this.fetchImpl = fetchImpl
  }

  private async call<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl.replace(/\/+$/, "")}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    })
    if (!res.ok) throw new Error(`phrase service ${path}: HTTP ${res.status}`)
    return (res.status === 204 ? undefined : await res.json()) as T
  }

  add(text: string, source: PhraseSource): Promise<Phrase> {
    return this.call("/phrases", {
      method: "POST",
      body: JSON.stringify({ user_id: this.userId, text, source }),
    })
  }

  search(reading: string, k = 10): Promise<PhraseHit[]> {
    const q = new URLSearchParams({ user_id: this.userId, q: reading, k: String(k) })
    return this.call(`/phrases/search?${q}`)
  }

  list(limit = 500): Promise<Phrase[]> {
    const q = new URLSearchParams({ user_id: this.userId, limit: String(limit) })
    return this.call(`/phrases?${q}`)
  }

  async remove(id: string): Promise<void> {
    await this.call(`/phrases/${encodeURIComponent(id)}`, { method: "DELETE" })
  }
}

/** TiDB service when `VITE_PHRASES_URL` is set, else the browser stand-in. */
export function createPhraseStore(): PhraseStore {
  const url = import.meta.env.VITE_PHRASES_URL as string | undefined
  return url
    ? new HttpPhraseStore(url, (import.meta.env.VITE_PHRASES_USER as string | undefined) || "demo")
    : new IndexedDbPhraseStore()
}
