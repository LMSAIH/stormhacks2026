#!/usr/bin/env bash
# B2 fine-tune job on the pod: prep → fine-tune → smoke-test the result → bench vs stock.
# Run detached so SSH/Jupyter disconnects don't kill it:
#   setsid bash ml/runpod/b2_finetune.sh > /workspace/b2/job.log 2>&1 < /dev/null &
#
#   PHASE=1 (default): 12 LRS3-test clips (idx 600+, never the LRS3-100 gate set), 2 epochs —
#                      proves the loop only; its WER numbers mean nothing.
#   PHASE=2:           CLIPS=/workspace/b2/recordings HOLDOUT=<speaker> NAME=FT_v1 [EPOCHS=3 LR=1e-4]
#                      [PAIRS=/workspace/b2/training_pairs PAIR_SOURCES=typed,picked]
#                      → also benches stock vs fine-tuned on LRS3-100 + the held-out speaker.
set -euo pipefail
cd "$(dirname "$0")/.."   # ml/
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
PHASE="${PHASE:-1}" B2="${B2:-/workspace/b2}"
mkdir -p "$B2"
# uv sync is exact: smoke.sh's sync (export+dev) uninstalls the train extra, so re-add it here
uv sync --quiet --extra export --extra dev --extra train

if [[ "$PHASE" == 1 ]]; then
  NAME="${NAME:-FT_phase1}" CLIPS="$B2/lrs3_tiny" HOLDOUT=lrs2
  EPOCHS="${EPOCHS:-2}" EXTRA=(--val-frac 0.25)
  uv run python scripts/prepare_finetune_data.py dump-lrs3 data/lrs3_test/0000.parquet "$CLIPS" \
    --start 600 --n 12 --speakers 3
else
  : "${CLIPS:?set CLIPS=dir of <speaker>_<nnn>.mp4 + .txt}" "${HOLDOUT:?set HOLDOUT=<speaker>}"
  # defaults from the GRID rehearsal (.context/b2-report.md): frozen BatchNorm, lr 1e-4, 3 epochs,
  # then WiSE-FT blends; plain 15-epoch fine-tuning forgot open speech (LRS3-100 28.6% → 70.8%)
  NAME="${NAME:-FT_v1}" EPOCHS="${EPOCHS:-3}" EXTRA=()
  # PAIRS=dir of the app's opt-in training pairs (POST /training-pairs) → added to train only
  [[ -n "${PAIRS:-}" ]] && EXTRA+=(--pairs "$PAIRS" --pair-sources "${PAIR_SOURCES:-typed,picked}")
fi
ROOT="$B2/data_$NAME"

echo "== prep $CLIPS → $ROOT (holdout $HOLDOUT)"
rm -rf "$ROOT"
uv run python scripts/prepare_finetune_data.py prepare "$CLIPS" "$ROOT" --holdout-speaker "$HOLDOUT" \
  "${EXTRA[@]}"

echo "== fine-tune $NAME ($EPOCHS epochs)"
uv run python scripts/finetune.py --root "$ROOT" --name "$NAME" --exp-dir "$B2/exp" \
  --epochs "$EPOCHS" --lr "${LR:-1e-4}" --precision "${PRECISION:-bf16-mixed}" \
  --max-frames "${MAX_FRAMES:-1600}" ${FREEZE_FRONTEND:+--freeze-frontend} \
  $([[ "${FREEZE_BN:-1}" == 1 ]] && echo --freeze-bn)

echo "== lipread on the result"
clip="$(ls "$ROOT"/cstm/cstm_video/*.npy | head -1)"
LIPREAD_MODEL="$NAME" uv run python - "$clip" <<'EOF'
import sys, numpy as np
from lipread.model import LipReader
from lipread.preprocess import to_model_input
r = LipReader()
x = to_model_input(np.load(sys.argv[1]))
print("greedy:", r.transcribe(x, "greedy").text)
print("beam:  ", r.transcribe(x, "beam").text)
EOF

if [[ "$PHASE" == 2 ]]; then
  echo "== bench: stock vs $NAME on LRS3-100 (unseen faces) and the held-out speaker"
  mkdir -p "$B2/heldout"
  for spk in ${HOLDOUT//,/ }; do for f in "$CLIPS/${spk}"_*; do ln -sf "$f" "$B2/heldout/"; done; done
  ALPHAS="${ALPHAS:-0.25 0.35 0.4 0.5}"
  uv run python scripts/interpolate_ckpt.py "$NAME" --alphas $ALPHAS
  blends=(); for a in $ALPHAS; do blends+=("${NAME}_a$a"); done
  for m in LRS3_V_WER19.1 "$NAME" "${blends[@]}"; do
    LIPREAD_MODEL="$m" uv run python scripts/bench.py --lrs3-parquet data/lrs3_test/0000.parquet \
      --n 100 --backend local --decode greedy beam --tag "lrs3_$m" --out "$B2/bench"
    LIPREAD_MODEL="$m" uv run python scripts/bench.py --clips "$B2/heldout" --n 1000 \
      --backend local --decode greedy beam --tag "heldout_$m" --out "$B2/bench"
  done
fi
echo "== done $NAME"
