#!/usr/bin/env bash
# Forgetting sweep after b2_grid.sh: fine-tune variants + WiSE-FT blends, greedy WER on LRS3-100
# (open speech = the regression gate) and 100 held-out GRID clips → $B2/sweep.md.
#   RUNS="name|finetune args;name2|args2"  BLEND="model:α1,α2 model2:α"  BENCH="extra models"
#   setsid bash ml/runpod/b2_sweep.sh > /workspace/b2/sweep.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # ml/
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
B2="${B2:-/workspace/b2}"
uv sync --quiet --extra export --extra dev --extra train
models=(${BENCH:-})

IFS=';' read -ra runs <<< "${RUNS:-}"
for run in "${runs[@]}"; do
  [[ -z "$run" ]] && continue
  name="${run%%|*}" args="${run#*|}"
  echo "== fine-tune $name ($args)"
  # shellcheck disable=SC2086
  uv run python scripts/finetune.py --root "$B2/data_grid" --name "$name" --exp-dir "$B2/exp" \
    --precision bf16-mixed --max-frames 1600 $args 2>&1 | grep -E "picked|greedy WER|Error"
  models+=("$name")
done
for spec in ${BLEND:-}; do
  m="${spec%%:*}" alphas="${spec#*:}"
  uv run python scripts/interpolate_ckpt.py "$m" --alphas ${alphas//,/ }
  for a in ${alphas//,/ }; do models+=("${m}_a$a"); done
done

echo "== bench (greedy)"
[[ -s "$B2/sweep.md" ]] || printf '| model | LRS3-100 greedy | GRID held-out greedy |\n|---|---|---|\n' > "$B2/sweep.md"
for m in "${models[@]}"; do
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
