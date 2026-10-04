/** CTC blank id (the model's `<blank>` token). */
const BLANK = 0

/** SentencePiece word-start marker. */
const WORD_START = "▁"

/**
 * Greedy CTC decode — a line-for-line port of `ml/src/lipread/model.py`
 * (`LipReader.greedy` = `collapse_ctc` + `ids_to_text`), so browser and Python agree exactly.
 *
 * `logProbs` is the model's `log_probs` output, row-major [timesteps, vocab] with
 * vocab === tokens.length. Per frame: argmax (first index wins ties, like torch) → merge repeats
 * → drop blank (0) and `<eos>` (last id; dropped *after* merging, as in Python) → join pieces →
 * "▁" to space → trim. SentencePiece pieces never contain raw whitespace, so `trim()` matches
 * Python's `strip()`. `<unk>` is kept, as in Python.
 *
 * `confidence` = mean max-probability (exp of the max log-prob) over non-blank frames;
 * undefined when every frame is blank.
 */
export function greedyCtcDecode(
  logProbs: Float32Array,
  timesteps: number,
  tokens: readonly string[]
): { text: string; confidence?: number } {
  const vocab = tokens.length
  if (logProbs.length !== timesteps * vocab) {
    throw new Error(
      `log_probs has ${logProbs.length} values, expected ${timesteps} × ${vocab} ` +
        "(timesteps × tokens.json length) — is tokens.json from the same export as the model?"
    )
  }
  const eos = vocab - 1
  const pieces: string[] = []
  let prev = BLANK
  let probSum = 0
  let nonBlank = 0

  for (let t = 0; t < timesteps; t++) {
    const row = t * vocab
    let best = 0
    let bestLogp = logProbs[row]
    for (let v = 1; v < vocab; v++) {
      const x = logProbs[row + v]
      if (x > bestLogp) {
        bestLogp = x
        best = v
      }
    }

    if (best !== BLANK) {
      probSum += Math.exp(bestLogp)
      nonBlank++
      if (best !== prev && best !== eos) pieces.push(tokens[best])
    }
    prev = best
  }

  const text = pieces.join("").replaceAll(WORD_START, " ").trim()
  return nonBlank > 0 ? { text, confidence: probSum / nonBlank } : { text }
}
