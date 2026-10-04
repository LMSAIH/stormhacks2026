#!/usr/bin/env bash
# B2 rehearsal on public data while team recordings are pending: fine-tune 19.1 on GRID speakers
# (CC BY 4.0, raw frontal video → our MouthCropper), score unseen GRID speakers and the LRS3-100
# regression gate, stock vs fine-tuned. GRID's fixed grammar makes its own gain mostly vocabulary;
# the point is the LRS3-100 number (does the recipe forget open speech?) and picking lr/freeze.
#   setsid bash ml/runpod/b2_grid.sh > /workspace/b2/grid.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # ml/
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
B2="${B2:-/workspace/b2}" TRAIN="${TRAIN:-s1,s2,s3,s4,s7,s5}" HOLDOUT="${HOLDOUT:-s6,s16}"
PER="${PER:-200}" EPOCHS="${EPOCHS:-5}"
uv sync --quiet --extra export --extra dev --extra train

echo "== fetch GRID ($TRAIN | holdout $HOLDOUT, $PER clips each)"
uv run python scripts/fetch_grid.py "$B2/grid_raw" --speakers "$TRAIN,$HOLDOUT" --per-speaker "$PER" \
  --cache "$B2/grid_zips"
echo "== prep"
rm -rf "$B2/data_grid"
for spk in ${TRAIN//,/ } ${HOLDOUT//,/ }; do
  ls "$B2/grid_raw/${spk}"_*.mpg >/dev/null 2>&1 || { echo "speaker $spk missing (alignment mismatch?)"; exit 1; }
done
uv run python scripts/prepare_finetune_data.py prepare "$B2/grid_raw" "$B2/data_grid" \
  --holdout-speaker "$HOLDOUT" --workers "${WORKERS:-16}" 2>&1 | grep -v "^ok "

# name|extra finetune args
RUNS=("FT_grid_lr1e-4|--lr 1e-4" "FT_grid_lr5e-5_frozen|--lr 5e-5 --freeze-frontend")
for run in "${RUNS[@]}"; do
  name="${run%%|*}" args="${run#*|}"
  echo "== fine-tune $name ($args, $EPOCHS epochs)"
  # shellcheck disable=SC2086
  uv run python scripts/finetune.py --root "$B2/data_grid" --name "$name" --exp-dir "$B2/exp" \
    --epochs "$EPOCHS" --precision bf16-mixed --max-frames 1600 $args 2>&1 \
    | grep -E "stock 19.1|picked|greedy WER|min ·|Error|error"
done

echo "== bench: LRS3-100 (unseen faces, open speech) + 100 held-out GRID clips, greedy + beam"
mkdir -p "$B2/grid_heldout"
for spk in ${HOLDOUT//,/ }; do
  for f in $(ls "$B2/grid_raw/${spk}"_*.mpg | head -50); do ln -sf "$f" "${f%.mpg}.txt" "$B2/grid_heldout/"; done
done
for m in LRS3_V_WER19.1 "${RUNS[@]%%|*}"; do
  for set in lrs3 grid; do
    src=(--lrs3-parquet data/lrs3_test/0000.parquet --n 100)
    [[ $set == grid ]] && src=(--clips "$B2/grid_heldout" --n 100)
    echo "-- $m on $set"
    LIPREAD_MODEL="$m" uv run python scripts/bench.py "${src[@]}" --backend local --decode greedy beam \
      --warmup 1 --tag "${set}_$m" --out "$B2/bench" 2>&1 | grep -E "^\|" | head -4
  done
done
echo "== done"
