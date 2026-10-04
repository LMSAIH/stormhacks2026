/**
 * Model-scored phrase snapping: how well does a saved phrase explain the lip frames?
 *
 * `snap.ts` compares the *text* of a reading with saved phrases. This scores each phrase against the
 * model's own per-frame `log_probs` instead: log P(phrase | video), summed over all CTC alignments,
 * minus the same for what greedy decoding read, per frame:
 *
 *   margin = (log P(phrase) − log P(reading)) / T        (≤ ~0; 0 = as likely as the reading)
 *
 * Port of `ml/src/lipread/phrases.py`; parity is checked against sentencepiece and torch ctc_loss
 * (`__fixtures__/ctcScore.json`, regenerate with `ml/scripts/export_phrase_assets.py`).
 * Offline (`ml/scripts/bench_phrase_snap.py`, LRS3 test 100-399, 50 saved phrases + 3 near-copies
 * each): margin ≥ −0.2 fixed 29/50 in-list readings with 2 wrong snaps in 300 clips; the look-alike
 * rule at SNAP_THRESHOLD 0.75 fixed 17 with 4.
 */
import { normalizeText } from "./lookalike"

/** Below this margin a phrase doesn't replace the reading. */
export const MODEL_SNAP_MARGIN = -0.2

const BLANK = 0
const WORD_START = "▁"
const UNK = "<unk>"
/** SentencePiece's penalty for an unknown character, below the lowest piece score. */
const UNK_PENALTY = 10

/** SentencePiece unigram piece → log-probability score (`public/phrases/pieceScores.json`). */
export interface PieceScores {
  readonly scores: ReadonlyMap<string, number>
  readonly maxLength: number
  readonly unkScore: number
}

export function pieceScores(entries: Record<string, number>): PieceScores {
  const scores = new Map(Object.entries(entries))
  let maxLength = 1
  let min = 0
  for (const [piece, score] of scores) {
    maxLength = Math.max(maxLength, piece.length)
    min = Math.min(min, score)
  }
  return { scores, maxLength, unkScore: min - UNK_PENALTY }
}

export async function loadPieceScores(
  url = `${import.meta.env.BASE_URL}phrases/pieceScores.json`
): Promise<PieceScores> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`piece scores: HTTP ${res.status} for ${url}`)
  return pieceScores((await res.json()) as Record<string, number>)
}

/**
 * Text → SentencePiece pieces, as `SentencePieceProcessor.EncodeAsPieces` does for the model's
 * unigram5000 vocabulary: "▁" marks each word start, then the segmentation with the best total
 * piece score wins (Viterbi). A character no piece covers stays as itself and maps to `<unk>`.
 */
export function tokenizePieces(text: string, sp: PieceScores): string[] {
  const norm = normalizeText(text)
  if (!norm) return []
  const s = WORD_START + norm.replaceAll(" ", WORD_START)
  const n = s.length
  const best = new Float64Array(n + 1).fill(-Infinity)
  const from = new Int32Array(n + 1).fill(-1)
  const piece: string[] = new Array<string>(n + 1)
  best[0] = 0
  for (let j = 0; j < n; j++) {
    if (best[j] === -Infinity) continue
    let covered = false
    for (let i = j + 1; i <= Math.min(n, j + sp.maxLength); i++) {
      const p = s.slice(j, i)
      const score = sp.scores.get(p)
      if (score === undefined) continue
      if (i === j + 1) covered = true
      if (best[j] + score > best[i]) {
        best[i] = best[j] + score
        from[i] = j
        piece[i] = p
      }
    }
    if (!covered && best[j] + sp.unkScore > best[j + 1]) {
      best[j + 1] = best[j] + sp.unkScore
      from[j + 1] = j
      piece[j + 1] = s[j] // like sentencepiece: the raw character, whose id is <unk>
    }
  }
  const out: string[] = []
  for (let i = n; i > 0; i = from[i]) out.push(piece[i])
  return out.reverse()
}

function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b))
}

/**
 * log P(ids | video) under CTC, summed over alignments (the forward algorithm, as torch's ctc_loss).
 * `logProbs` is row-major [timesteps, vocab]. −Infinity when the frames can't fit the ids.
 */
export function ctcLogLikelihood(
  logProbs: Float32Array,
  timesteps: number,
  vocab: number,
  ids: readonly number[]
): number {
  const L = ids.length
  if (L === 0 || timesteps === 0) return -Infinity
  const S = 2 * L + 1
  const label = (s: number) => (s % 2 === 0 ? BLANK : ids[(s - 1) >> 1])
  let alpha = new Float64Array(S).fill(-Infinity)
  let next = new Float64Array(S)
  alpha[0] = logProbs[label(0)]
  alpha[1] = logProbs[label(1)]
  for (let t = 1; t < timesteps; t++) {
    const row = t * vocab
    for (let s = 0; s < S; s++) {
      const l = label(s)
      let a = alpha[s]
      if (s > 0) a = logAdd(a, alpha[s - 1])
      if (s > 1 && l !== BLANK && l !== label(s - 2)) a = logAdd(a, alpha[s - 2])
      next[s] = a + logProbs[row + l]
    }
    ;[alpha, next] = [next, alpha]
  }
  return logAdd(alpha[S - 1], alpha[S - 2])
}

export interface ModelScore {
  readonly text: string
  /** (log P(phrase) − log P(reading)) / frames; higher = the model finds it more likely. */
  readonly margin: number
}

/** Token ids (indices into the model's tokens.json) for a text. */
export function phraseIds(text: string, tokens: readonly string[], sp: PieceScores): number[] {
  const index = tokenIndex(tokens)
  const unk = index.get(UNK) ?? BLANK
  return tokenizePieces(text, sp).map((p) => index.get(p) ?? unk)
}

const indexCache = new WeakMap<readonly string[], Map<string, number>>()
function tokenIndex(tokens: readonly string[]): Map<string, number> {
  let index = indexCache.get(tokens)
  if (!index) {
    index = new Map(tokens.map((t, i) => [t, i]))
    indexCache.set(tokens, index)
  }
  return index
}

/** Saved phrases, best first, by how well the model thinks each explains the frames. */
export function rankByModel(
  logProbs: Float32Array,
  timesteps: number,
  tokens: readonly string[],
  sp: PieceScores,
  reading: string,
  phrases: readonly string[]
): ModelScore[] {
  if (phrases.length === 0) return []
  const ll = (text: string) => ctcLogLikelihood(logProbs, timesteps, tokens.length, phraseIds(text, tokens, sp))
  const lls = phrases.map(ll)
  const own = reading.trim() ? ll(reading) : -Infinity
  const base = Number.isFinite(own) ? own : Math.max(...lls)
  const frames = Math.max(timesteps, 1)
  return phrases
    .map((text, i) => ({ text, margin: (lls[i] - base) / frames }))
    .sort((a, b) => b.margin - a.margin)
}

/** The saved phrase to replace `reading` with, or undefined. Never the reading itself. */
export function modelSnap(
  ranked: readonly ModelScore[],
  reading: string,
  margin = MODEL_SNAP_MARGIN
): ModelScore | undefined {
  const top = ranked[0]
  return top && top.margin >= margin && normalizeText(top.text) !== normalizeText(reading) ? top : undefined
}
