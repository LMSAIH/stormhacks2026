#!/usr/bin/env bash
# One-shot pod setup (idempotent). Everything heavy lives on the pod volume (/workspace) so it
# survives pod restarts. Run as root on a RunPod GPU pod:
#   curl -fsSL https://raw.githubusercontent.com/LMSAIH/stormhacks2026/ml/model-pipeline/ml/runpod/bootstrap.sh | bash
# or, if the repo is already there:  bash /workspace/stormhacks2026/ml/runpod/bootstrap.sh
set -euo pipefail

WS="${WORKSPACE:-/workspace}"
REPO="${REPO:-https://github.com/LMSAIH/stormhacks2026.git}"
BRANCH="${BRANCH:-ml/model-pipeline}"
DIR="$WS/stormhacks2026"
export UV_CACHE_DIR="$WS/.cache/uv" UV_PYTHON_INSTALL_DIR="$WS/.cache/python" UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

echo "== CUDA preflight (some community hosts show the GPU in nvidia-smi but cuInit fails)"
if ! python3 -c "import ctypes, sys; sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))"; then
  echo "CUDA preflight FAILED (cuInit != 0) — bad host, recreate the pod" >&2
  exit 3
fi

echo "== system packages (opencv/mediapipe runtime libs)"
if ! ldconfig -p | grep -q libGL.so.1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 libegl1 >/dev/null
fi

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== repo $BRANCH"
if [[ -d "$DIR/.git" ]]; then
  git -C "$DIR" fetch -q origin "$BRANCH" && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q --ff-only
else
  git clone -q --branch "$BRANCH" "$REPO" "$DIR"
fi

cd "$DIR/ml"
echo "== uv sync"
uv sync --extra export --extra dev

echo "== checkpoints"
bash scripts/download_checkpoints.sh  # via bash: exec bits can be lost (repo is edited on a fileMode=false FS)

echo "== LRS3 test shard (pre-made crops, for bench.py)"
mkdir -p data/lrs3_test
[[ -s data/lrs3_test/0000.parquet ]] || curl -fsSL -o data/lrs3_test/0000.parquet \
  "https://huggingface.co/datasets/mattymchen/lrs3-test/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

echo "== nvidia"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "bootstrap done: $DIR"
