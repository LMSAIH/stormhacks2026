#!/usr/bin/env bash
# Publish the exported ONNX lip-reading model to the web frontend ("speed" mode).
#
#   ml/scripts/publish_frontend_model.sh
#
# Copies ml/artifacts/{lipread_ctc.onnx,tokens.json} → frontend/public/models/ (gitignored; the
# model is 775 MB fp32 and never goes in git). Runs scripts/export_onnx.py first if either is
# missing. Re-running is cheap: files whose size and mtime already match are skipped.
set -euo pipefail

ML="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ML/artifacts"
DST="$ML/../frontend/public/models"
FILES=(lipread_ctc.onnx tokens.json)

sig() { stat -c '%s %Y' "$1" 2>/dev/null || stat -f '%z %m' "$1"; }  # size + mtime (GNU / BSD)

if [[ ! -f "$SRC/lipread_ctc.onnx" || ! -f "$SRC/tokens.json" ]]; then
  echo "── $SRC is missing the export; running scripts/export_onnx.py"
  uv run --directory "$ML" python scripts/export_onnx.py
fi

mkdir -p "$DST"
DST="$(cd "$DST" && pwd)"
for f in "${FILES[@]}"; do
  src="$SRC/$f" dst="$DST/$f"
  if [[ -f "$dst" && "$(sig "$src")" == "$(sig "$dst")" ]]; then
    echo "up to date  $dst"
    continue
  fi
  # Reflink (instant, no extra disk) where the filesystem supports it; plain copy otherwise.
  cp --reflink=auto -p "$src" "$dst.tmp" 2>/dev/null || cp -p "$src" "$dst.tmp"
  mv "$dst.tmp" "$dst"
  echo "copied      $dst ($(du -h "$dst" | cut -f1))"
done

echo "✓ frontend/public/models ready — served as /models/lipread_ctc.onnx + /models/tokens.json"
