#!/usr/bin/env bash
# Repo smoke gate — run after any product-code change; report `smoke: N/N`.
#   ./smoke.sh            all parts
#   ./smoke.sh ml         only ml/   (or: frontend, backend)
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

echo
if ((total == 0)); then echo "smoke: nothing to check"; exit 1; fi
echo "smoke: $pass/$total"
((${#failed[@]})) && printf '  failed: %s\n' "${failed[@]}"
exit $((total - pass))
