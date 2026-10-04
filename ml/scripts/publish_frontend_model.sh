#!/usr/bin/env bash
# Publish the quantized ONNX lip-reading model to the web frontend ("speed" mode).
#
#   ml/scripts/publish_frontend_model.sh
#
# Copies into frontend/public/models/ (gitignored — the model never goes in git):
#   ml/artifacts/lipread_ctc.dyn-pw8-rn16.onnx  →  lipread_ctc.int8.onnx   (203 MB; the fp32 export is 775 MB)
#   ml/artifacts/tokens.json                    →  tokens.json
# The int8 file is the one frontend/src/lib/lipreading/modelSpec.ts points at (ACTIVE_SPEC.modelUrl).
#
# Missing pieces are built first: scripts/export_onnx.py (fp32 export + tokens.json) if either is
# absent, then scripts/quantize_onnx.py --variant dyn-pw8-rn16 (see its docstring for the recipe and
# measured accuracy; ml/tests/quantized_baseline.json pins the exact model + texts). Re-running is cheap:
# files whose size and mtime already match are skipped.
#
# Also removes a stale frontend/public/models/lipread_ctc.onnx: earlier versions of this script
# published the 775 MB fp32 model under that name, and `vite build` would copy it into dist/.
set -euo pipefail

ML="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ML/artifacts"
DST="$ML/../frontend/public/models"
VARIANT="dyn-pw8-rn16"
LOCK="$ML/tests/quantized_baseline.json"
# source file (in $SRC)  →  published name (in $DST)
PUBLISH=("lipread_ctc.$VARIANT.onnx:lipread_ctc.int8.onnx" "tokens.json:tokens.json")

sig() { stat -c '%s %Y' "$1" 2>/dev/null || stat -f '%z %m' "$1"; }  # size + mtime (GNU / BSD)
sha256() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" | cut -d' ' -f1; }

if [[ ! -f "$SRC/lipread_ctc.$VARIANT.onnx" ]]; then
  if [[ ! -f "$SRC/lipread_ctc.onnx" ]]; then
    echo "── $SRC is missing the fp32 export; running scripts/export_onnx.py"
    uv run --directory "$ML" python scripts/export_onnx.py
  fi
  echo "── building lipread_ctc.$VARIANT.onnx (scripts/quantize_onnx.py --variant $VARIANT)"
  uv run --directory "$ML" python scripts/quantize_onnx.py --variant "$VARIANT"
fi
if [[ ! -f "$SRC/tokens.json" ]]; then
  echo "── $SRC/tokens.json is missing; running scripts/export_onnx.py"
  uv run --directory "$ML" python scripts/export_onnx.py
fi

mkdir -p "$DST"
DST="$(cd "$DST" && pwd)"
for pair in "${PUBLISH[@]}"; do
  src="$SRC/${pair%%:*}" dst="$DST/${pair##*:}"
  if [[ -f "$dst" && "$(sig "$src")" == "$(sig "$dst")" ]]; then
    echo "up to date  $dst"
    continue
  fi
  # Reflink (instant, no extra disk) where the filesystem supports it; plain copy otherwise.
  cp --reflink=auto -p "$src" "$dst.tmp" 2>/dev/null || cp -p "$src" "$dst.tmp"
  mv "$dst.tmp" "$dst"
  echo "copied      $dst ($(du -h "$dst" | cut -f1))"
done

if [[ -f "$DST/lipread_ctc.onnx" ]]; then
  rm -f "$DST/lipread_ctc.onnx"
  echo "removed     $DST/lipread_ctc.onnx (stale 775 MB fp32 copy)"
fi

# Warn (not fail) when the published model is not the one the regression lock was recorded for.
if [[ -f "$LOCK" ]] && command -v python3 >/dev/null; then
  locked="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["sha256"])' "$LOCK" 2>/dev/null || true)"
  actual="$(sha256 "$DST/lipread_ctc.int8.onnx")"
  if [[ -n "$locked" && "$locked" != "$actual" ]]; then
    echo "WARNING: lipread_ctc.int8.onnx sha256 ${actual:0:12}… differs from the lock ${locked:0:12}… in" \
         "ml/tests/quantized_baseline.json (re-quantized?) — run scripts/regress_quantized.py before shipping it" >&2
  fi
fi

echo "✓ frontend/public/models ready — served as /models/lipread_ctc.int8.onnx + /models/tokens.json"
