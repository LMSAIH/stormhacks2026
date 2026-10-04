# Model-scored phrase snapping (2026-10-04, cloud session, branch `ml/b2-finetune`)

## What it is
Rank saved phrases by how well the model thinks each one explains the lip frames: CTC
log-likelihood of the phrase minus that of the greedy reading, per frame (`margin`, ≤ ~0). The app
today ranks by text look-alike (`phrases/lookalike.ts` + `snap.ts`).

- Python: `ml/src/lipread/phrases.py` (`rank_phrases`, `snap`, `DEFAULT_MARGIN = −0.2`)
- Browser: `frontend/src/lib/phrases/ctcScore.ts` (`loadPieceScores`, `rankByModel`, `modelSnap`,
  `MODEL_SNAP_MARGIN`), parity-tested vs sentencepiece (665 texts) and torch ctc_loss. Piece scores
  are fetched from `public/phrases/pieceScores.json` (112 KB, not bundled).
- Wired for on-device reads (Instant excluded) on `cloud/phrase-scoring`, 2026-10-04: app eval in
  `.context/app-eval.md` (Normal ≈23% vs 28.7%). Only the user's own phrases are model-ranked; swear
  seeds keep look-alike.
- Quality (server) reads, `ml/quality-server`, 2026-10-04: `POST /lipread/phrases` takes the same
  crops + the phrases + the reading and returns `rank_phrases` margins from the encoder's CTC
  log-probs (brief §5); `httpRecognizer.ts` attaches it as `RecognitionResult.scorePhrases`, so the
  hook's model snapping covers Quality too (null on any failure → look-alike). Margins are against
  the reading the client shows (the beam reading), like on-device (the greedy reading).

## Offline result (`ml/scripts/bench_phrase_snap.py`)
LRS3 test idx 100–399 (not the LRS3-100 gate), greedy WER before snapping 34.4%, 50 clips' sentences
saved as phrases.

| memory | rule | right phrase 1st | fixed | wrong snaps | WER after |
|---|---|---|---|---|---|
| 50 distinct phrases | look-alike ≥ 0.75 (app today) | 50/50 | 20/50 | 1 | 32.8% |
| | look-alike ≥ 0.6 | | 31/50 | 1 | 31.0% |
| | model ≥ −0.2 | 49/50 | 29/50 | 2 | 31.2% |
| 50 + 3 one-word-swap decoys each | look-alike ≥ 0.75 | **41/50** | 17/50 | 4 | 32.9% |
| | model ≥ −0.2 | **49/50** | **29/50** | **2** | 31.2% |
| | model ≥ −0.3 | | 34/50 | 4 | 30.1% |

With distinct phrases the two tie; with similar saved phrases (the realistic case for one user's
memory) look-alike picks the wrong one and the model doesn't. No rule ever broke a correct reading.
Not modelled: the new `snapAllowed` 0.9 confidence gate, short app-style phrases.

## How to wire it (for whoever owns `useLipReader.ts`)
1. `onnxRecognizer.ts`: keep the last `log_probs` (Float32Array + timesteps) on the result instead of
   disposing it straight away, e.g. `RecognitionResult.scorePhrases?: (reading, phrases) => ModelScore[]`
   closing over them (keeps the tensor out of React state).
2. Where the app snaps: `rankByModel(...)` over the phrase-memory candidates → `modelSnap(ranked,
   reading)`; keep `snapAllowed` as an extra guard (model picks the phrase, the gate protects sure
   words). Use the model ranking for the top-3 picker too.
3. Quality mode: done as a separate `POST /lipread/phrases` (the phrases are only known once the
   reading is back), see above.

Re-run `ml/scripts/bench_phrase_snap.py` (CPU, ~5 min) after changes; `--decoys 3` is the hard case.
