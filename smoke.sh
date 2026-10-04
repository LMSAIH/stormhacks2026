#!/usr/bin/env bash
# Repo smoke gate — run after any product-code change; report `smoke: N/N`.
#   ./smoke.sh            all parts
#   ./smoke.sh ml         only ml/   (or: frontend, backend)
#   ./smoke.sh app        opt-in, ~6 min: the app eval gate (20 real faces through /app; not in "all")
#   SMOKE_REQUIRE_CUDA=1 ./smoke.sh   fail if torch can't see a GPU (use on the RunPod pod)
# Parts that don't exist yet are skipped, not failed. Uses pnpm (never npm) and uv.
set -uo pipefail
cd "$(dirname "$0")"

ONLY="${1:-all}"
pass=0; total=0; failed=()

step() {  # step <name> <cmd...>
  local name="$1"; shift
  total=$((total + 1))
  printf '\n── %s\n' "$name"
  if "$@"; then
    pass=$((pass + 1)); echo "✓ $name"
  else
    failed+=("$name"); echo "✗ $name"
  fi
}

want() { [[ "$ONLY" == all || "$ONLY" == "$1" ]]; }

if want frontend && [[ -f frontend/package.json ]] && ! command -v pnpm >/dev/null; then
  echo "── frontend: SKIP (pnpm not installed — e.g. on a GPU pod)"
elif want frontend && [[ -f frontend/package.json ]]; then
  step "frontend: install" pnpm --dir frontend install --frozen-lockfile --silent
  step "frontend: lint"    pnpm --dir frontend lint
  # Unit + parity tests; the real-model golden test also runs with LIPREAD_ONNX_TEST=1.
  step "frontend: test"    pnpm --dir frontend test
  step "frontend: build"   pnpm --dir frontend build
fi

if want backend && [[ -f backend/pyproject.toml ]]; then
  # Placeholder until the infra team lands backend/: import check only.
  step "backend: sync" uv sync --project backend --quiet
fi

if want ml && [[ -f ml/pyproject.toml ]]; then
  step "ml: sync"   uv sync --project ml --extra export --extra dev --quiet
  step "ml: checks" uv run --directory ml python scripts/smoke_checks.py
fi

# Opt-in: plays 20 real-face clips through the real /app in headless Chromium and fails if Normal
# or Instant reads more than 5 pts worse than the gate line in .context/app-eval.md.
if [[ "$ONLY" == app ]] && ! command -v pnpm >/dev/null; then
  echo "── app: SKIP (pnpm not installed — e.g. on a GPU pod)"
elif [[ "$ONLY" == app ]]; then
  step "app: eval gate" uv run --directory ml python scripts/app_eval/gate.py
fi

echo
if ((total == 0)); then echo "smoke: nothing to check"; exit 1; fi
echo "smoke: $pass/$total"
((${#failed[@]})) && printf '  failed: %s\n' "${failed[@]}"
exit $((total - pass))
