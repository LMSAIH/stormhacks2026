#!/usr/bin/env bash
# Download the pretrained Auto-AVSR VSR checkpoint (LRS3_V_WER19.1) and the subword RNN-LM
# used for beam search. Same HF mirrors Chaplin uses. Idempotent; ~1.3 GB total.
# Licence: research / non-commercial (LRS3 / BBC terms) — hackathon use only.
set -euo pipefail

DEST="${LIPREAD_CKPT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"
HF="https://huggingface.co"

fetch() {  # fetch <repo> <file> <out_dir>
  local repo="$1" file="$2" out="$3/$2"
  mkdir -p "$3"
  if [[ -s "$out" ]]; then
    echo "have  $out"
    return
  fi
  echo "fetch $repo/$file"
  curl -fL --retry 3 --progress-bar -o "$out.part" "$HF/$repo/resolve/main/$file"
  mv "$out.part" "$out"
}

fetch Amanvir/LRS3_V_WER19.1 model.json "$DEST/LRS3_V_WER19.1"
fetch Amanvir/LRS3_V_WER19.1 model.pth  "$DEST/LRS3_V_WER19.1"
if [[ "${SKIP_LM:-0}" != "1" ]]; then
  fetch Amanvir/lm_en_subword model.json "$DEST/lm_en_subword"
  fetch Amanvir/lm_en_subword model.pth  "$DEST/lm_en_subword"
fi
echo "checkpoints in $DEST"
