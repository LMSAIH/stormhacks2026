#!/usr/bin/env bash
# Forgetting sweep after b2_grid.sh: WiSE-FT blends of a fine-tuned model + low-lr runs, greedy WER
# on LRS3-100 (open speech, the regression gate) and 100 held-out GRID clips. Prints one table.
#   setsid bash ml/runpod/b2_sweep.sh > /workspace/b2/sweep.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # ml/
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
B2="${B2:-/workspace/b2}" BASE_FT="${BASE_FT:-FT_grid_lr1e-4}" ALPHAS="${ALPHAS:-0.1 0.2 0.3 0.5}"
uv sync --quiet --extra export --extra dev --extra train

uv run python scripts/interpolate_ckpt.py "$BASE_FT" --alphas $ALPHAS
models=()
for a in $ALPHAS; do models+=("${BASE_FT}_a$a"); done
# low-lr runs: name|args
LOWLR=("FT_grid_lr1e-5_e2|--lr 1e-5 --epochs 2" "FT_grid_lr3e-6_e1|--lr 3e-6 --epochs 1 --warmup-epochs 0")
for run in "${LOWLR[@]}"; do
  name="${run%%|*}" args="${run#*|}"
  echo "== fine-tune $name ($args)"
  # shellcheck disable=SC2086
  uv run python scripts/finetune.py --root "$B2/data_grid" --name "$name" --exp-dir "$B2/exp" \
    --precision bf16-mixed --max-frames 1600 $args 2>&1 | grep -E "picked|greedy WER|Error"
  models+=("$name")
done

echo "== bench (greedy)"
printf '| model | LRS3-100 greedy | GRID held-out greedy |\n|---|---|---|\n' > "$B2/sweep.md"
for m in LRS3_V_WER19.1 "$BASE_FT" "${models[@]}"; do
  w=()
  for src in "--lrs3-parquet data/lrs3_test/0000.parquet" "--clips $B2/grid_heldout"; do
    # shellcheck disable=SC2086
    w+=("$(LIPREAD_MODEL="$m" uv run python scripts/bench.py $src --n 100 --backend local \
      --decode greedy --warmup 1 --tag "sweep_${m}_${#w[@]}" --out "$B2/bench" 2>/dev/null \
      | awk -F'|' '/^\| greedy/ {gsub(/ /,"",$5); print $5}')")
  done
  echo "| $m | ${w[0]} | ${w[1]} |" | tee -a "$B2/sweep.md"
done
echo "== done"
