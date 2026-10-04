#!/usr/bin/env bash
# Is https://tryheard.tech up, end to end, from outside? Curls the three hostnames, checks the
# headers and CORS the browser relies on, both WebSockets, and sends one real /lipread/crops read.
# Needs curl and gzip only; runs from any machine. Exit code = number of failed checks.
#
#   scripts/check_live.sh
#   ML=https://<pod>-8000.proxy.runpod.net scripts/check_live.sh   # one part against another host
set -uo pipefail
cd "$(dirname "$0")/.."

SITE="${SITE:-https://tryheard.tech}"
API="${API:-https://api.tryheard.tech}"
ML="${ML:-https://ml.tryheard.tech}"
WSAPI="${API/#https/wss}"
CROPS=frontend/src/lib/lipreading/crop/__fixtures__/synthetic/crops.bin  # 24 × 96 × 96 uint8
pass=0; fail=0
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

ok()  { pass=$((pass + 1)); printf '  ok    %s\n' "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL  %s\n' "$*"; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
code() { curl -sS -o /dev/null -m "${TIMEOUT:-20}" -w '%{http_code}' "$@" 2>/dev/null; }
hdr()  { curl -sS -m 20 -D - -o /dev/null "$@" 2>/dev/null | tr -d '\r'; }
# A WebSocket upgrade through Cloudflare: 101 = the socket server accepted it (it then closes with
# 4401 when signed out). A route nobody serves answers 403/404 instead.
ws_code() {
  curl -sS -o /dev/null -m 5 -w '%{http_code}' --http1.1 -H "Origin: $SITE" \
    -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' \
    -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' "${1/#wss/https}" 2>/dev/null
}

echo "== $SITE (Cloudflare Pages)"
h=$(hdr "$SITE/")
check "GET / → 200" grep -q '^HTTP/[0-9.]* 200' <<<"$h"
check "COOP same-origin" grep -qi '^cross-origin-opener-policy: same-origin' <<<"$h"
check "COEP require-corp" grep -qi '^cross-origin-embedder-policy: require-corp' <<<"$h"
check "GET /app → 200 (SPA route)" test "$(code "$SITE/app")" = 200
check "speed-mode runtime /ort/ort.wasm.bundle.min.mjs → 200" test "$(code "$SITE/ort/ort.wasm.bundle.min.mjs")" = 200
check "ORT wasm /ort/ort-wasm-simd-threaded.wasm → 200" test "$(code "$SITE/ort/ort-wasm-simd-threaded.wasm")" = 200
check "www → apex redirect" grep -qiE "^location: $SITE/?" <<<"$(hdr "${SITE/:\/\//://www.}/")"

echo "== $ML (our ML server)"
health=$(curl -sS -m 20 "$ML/health" 2>/dev/null)
check "GET /health → status ok, model loaded" grep -q '"status":"ok".*"loaded":true' <<<"$health"
echo "        $health"
check "CORS for $SITE" grep -qi '^access-control-allow-origin: \(\*\|'"$SITE"'\)' \
  <<<"$(hdr -H "Origin: $SITE" "$ML/health")"
gzip -c "$CROPS" > "$tmp/crops.gz"
read_out=$(curl -sS -m 60 -w '\n%{http_code} %{time_total}' -X POST \
  -H 'Content-Type: application/octet-stream' -H 'Content-Encoding: gzip' -H "Origin: $SITE" \
  --data-binary @"$tmp/crops.gz" "$ML/lipread/crops?t=24&h=96&w=96&decode=beam" 2>/dev/null)
read_status=$(tail -1 <<<"$read_out")
check "POST /lipread/crops (beam, 24 synthetic frames) → 200 with text" \
  bash -c '[[ "$1" == 200\ * ]] && grep -q "\"text\"" <<<"$2"' _ "$read_status" "$(head -1 <<<"$read_out")"
echo "        HTTP $read_status s round trip; $(head -1 <<<"$read_out" | grep -o '"latency_ms":{[^}]*}' || true)"

echo "== $API (backend team's server)"
check "GET /openapi.json → 200" test "$(code "$API/openapi.json")" = 200
check "GET /api/auth/me signed out → 401" test "$(code "$API/api/auth/me")" = 401
pre=$(hdr -X OPTIONS -H "Origin: $SITE" -H 'Access-Control-Request-Method: GET' "$API/api/auth/me")
check "CORS preflight allows $SITE with credentials" bash -c \
  'grep -qi "^access-control-allow-origin: $2" <<<"$1" && grep -qi "^access-control-allow-credentials: true" <<<"$1"' _ "$pre" "$SITE"
login=$(hdr "$API/api/auth/google/login")
enc="${API//:/%3A}"; enc="${enc//\//%2F}"
check "Google login redirect_uri is $API/api/auth/google/callback" \
  grep -q "redirect_uri=${enc}%2Fapi%2Fauth%2Fgoogle%2Fcallback" <<<"$login"
check "WebSocket $WSAPI/ws/tts upgrades (TTS server :8765)" test "$(ws_code "$WSAPI/ws/tts")" = 101
check "WebSocket $WSAPI/ws/stt upgrades (REST app /ws/stt)" test "$(ws_code "$WSAPI/ws/stt")" = 101

echo
echo "check_live: $pass/$((pass + fail))"
exit "$fail"
