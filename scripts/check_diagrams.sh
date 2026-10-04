#!/usr/bin/env bash
# Which architecture diagrams are probably stale? Warning only: always exits 0, never blocks.
#
#   scripts/check_diagrams.sh            # compare with origin/master
#   scripts/check_diagrams.sh <base>     # compare with another branch or commit
#
# Lists the files changed since the merge base with <base> (commits, uncommitted edits and new
# untracked files), matches them against the "## Source of truth" list at the end of each
# docs/architecture/*.md, and prints the diagrams whose sources changed while the diagram did not.
# List entries are `- \`path\`` lines; a path ending in / covers everything under it, * is a glob.
# Run `git fetch origin master` first if your origin/master is old. Not part of smoke.sh.
set -u

cd "$(dirname "$0")/.." 2>/dev/null || { echo "check_diagrams: cannot find the repo root"; exit 0; }
base="${1:-origin/master}"
dir="docs/architecture"

if ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
  echo "check_diagrams: '$base' not found (try: git fetch origin master). Nothing checked."
  exit 0
fi
mb="$(git merge-base "$base" HEAD 2>/dev/null)" || {
  echo "check_diagrams: no merge base with '$base'. Nothing checked."
  exit 0
}
if ! ls "$dir"/*.md >/dev/null 2>&1; then
  echo "check_diagrams: no diagrams in $dir. Nothing checked."
  exit 0
fi

changed="$({ git diff --name-only "$mb"; git ls-files --others --exclude-standard; } | sort -u)"
if [[ -z "$changed" ]]; then
  echo "check_diagrams: no files changed since $(git rev-parse --short "$mb") ($base)."
  exit 0
fi

# Source-of-truth entries of one diagram, one path per line.
entries() {
  awk '
    /^## / { in_list = ($0 ~ /^## Source of truth/); next }
    in_list && /^- `/ { s = $0; sub(/^- `/, "", s); sub(/`.*/, "", s); print s }
  ' "$1"
}

# Does a changed file match an entry? Entry ending in / = directory prefix, else exact or glob.
matches() {
  local file="$1" entry="$2"
  if [[ "$entry" == */ ]]; then
    [[ "$file" == "$entry"* ]]
  else
    # shellcheck disable=SC2053  # unquoted on purpose: * in an entry is a glob
    [[ "$file" == $entry ]]
  fi
}

stale=0
covered=""
for diagram in "$dir"/*.md; do
  hits=""
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    if [[ ! -e "$entry" ]] && ! compgen -G "$entry" >/dev/null; then
      echo "note: $diagram lists '$entry', which does not exist"
    fi
    while IFS= read -r file; do
      if matches "$file" "$entry"; then
        hits="$hits
$file"
        covered="$covered
$file"
      fi
    done <<< "$changed"
  done < <(entries "$diagram")
  [[ -z "$hits" ]] && continue
  hits="$(printf '%s\n' "$hits" | sed '/^$/d' | sort -u | tr '\n' ' ')"
  hits="${hits% }"
  if printf '%s\n' "$changed" | grep -qxF "$diagram"; then
    echo "updated:        $diagram (sources changed: $hits)"
  else
    echo "probably stale: $diagram"
    echo "                sources changed: $hits"
    stale=$((stale + 1))
  fi
done

# Changed code that no diagram lists: maybe a new component that needs a diagram.
uncovered=""
while IFS= read -r file; do
  case "$file" in
    frontend/src/*.test.ts|frontend/src/*/__fixtures__/*) continue ;;
    frontend/src/*|ml/src/*|ml/scripts/*|ml/runpod/*) ;;
    *) continue ;;
  esac
  printf '%s\n' "$covered" | grep -qxF "$file" || uncovered="$uncovered
$file"
done <<< "$changed"
if [[ -n "$uncovered" ]]; then
  echo "not in any diagram's Source of truth (fine for small helpers; new components need one):"
  printf '%s\n' "$uncovered" | sed '/^$/d; s/^/  /'
fi

if ((stale)); then
  echo "check_diagrams: $stale diagram(s) probably stale. Update them, or say in the PR why not."
else
  echo "check_diagrams: no stale diagrams found."
fi
exit 0
