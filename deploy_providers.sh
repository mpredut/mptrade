#!/usr/bin/env bash
# Deploy the checkout and refresh BOTH fleet processes and independent bots.
# --check is strictly local/offline: validate imports/configuration and print scope.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$ROOT/procs.conf"
cd "$ROOT"
case "${1:-}" in ""|--check) ;; *) echo "Usage: $0 [--check]"; exit 2;; esac
PY=""
for candidate in .venv myenv; do
    if [ -x "$ROOT/$candidate/bin/python" ]; then PY="$ROOT/$candidate/bin/python"; break; fi
done
[ -n "$PY" ] || { echo "No virtual environment found"; exit 1; }
source "$ROOT/process_control.sh"

if [ "${1:-}" != --check ]; then
    # pipefail + errexit: an unsuccessful pull must never be followed by a restart.
    git pull --ff-only | tail -5
fi
"$PY" instrument_registry.py
BINANCE_AUTO_START_WEBSOCKETS=0 "$PY" -c 'from providers.market_api import api; import tradeall, assetguardian; from binance_api import trailing_stop; print("Import preflight OK")'

declare -a patterns=() directories=() roles=() old_pids=()
while IFS='|' read -r pat dir cmd label hblog hbstale role; do
    [ -z "$pat" ] && continue
    case "$pat" in \#*) continue;; esac
    case "$role" in fleet|bot) ;; *) echo "Invalid role for $label"; exit 1;; esac
    dir=$(eval echo "$dir")
    [ -d "$dir" ] || { echo "Missing directory for $label"; exit 1; }
    if [ "$pat" = rtrade.py ]; then
        "$PY" -c 'from instrument_registry import single_symbol_for; single_symbol_for("binance", "rtrade")'
    fi
    patterns+=("$pat"); directories+=("$dir"); roles+=("$role")
    old_pids+=("$(manifest_pids "$pat" "$dir")")
    printf '  %-18s role=%s\n' "$label" "$role"
done < "$MANIFEST"
[ "${#patterns[@]}" -gt 0 ] || { echo "Empty process manifest"; exit 1; }
if [ "${1:-}" = --check ]; then
    echo "Preflight only: no pull, stop, launch, API call, or production health claim."
    exit 0
fi

# The active fleet supervisor revives role=fleet processes. Do not launch a second
# supervisor or delete caches/state. A missing supervisor will fail verification.
for index in "${!patterns[@]}"; do
    if [ "${roles[$index]}" = fleet ]; then
        stop_manifest_process "${patterns[$index]}" "${directories[$index]}"
    fi
done
# Keep daemon-inherited descriptors off the caller's output pipe (upstream fix).
# Do not mask launcher failure: successful import/pull is not successful deployment.
mkdir -p "$ROOT/logs"
if ! bash "$ROOT/bots_start.sh" >"$ROOT/logs/deploy_bots_start.log" 2>&1; then
    tail -20 "$ROOT/logs/deploy_bots_start.log"
    exit 1
fi
tail -3 "$ROOT/logs/deploy_bots_start.log"

# Require one replacement per manifest entry and fresh caches on three consecutive
# checks. Presence alone, an old trailing PID, or an old success log is insufficient.
stable=0
deadline=$((SECONDS + 90))
while [ "$SECONDS" -lt "$deadline" ]; do
    ready=1
    for index in "${!patterns[@]}"; do
        current="$(manifest_pids "${patterns[$index]}" "${directories[$index]}")"
        count=$(printf '%s\n' "$current" | awk 'NF {n++} END {print n+0}')
        if [ "$count" -ne 1 ]; then ready=0; continue; fi
        state="$(ps -o stat= -p "$current" 2>/dev/null | tr -d '[:space:]')" || { ready=0; continue; }
        case "$state" in Z*|T*) ready=0; continue;; esac
        while read -r previous; do
            [ -z "$previous" ] && continue
            [ "$current" != "$previous" ] || ready=0
        done <<< "${old_pids[$index]}"
    done
    if [ "$ready" -eq 1 ] && "$PY" verify_tools/check_cache_coherence.py; then
        stable=$((stable + 1))
        if [ "$stable" -ge 3 ]; then
            echo "Deployment verified: replacement processes and fresh disk caches."
            echo "This does not prove fills, account health, or trailing activation; inspect runtime state."
            exit 0
        fi
    else
        stable=0
    fi
    sleep 2
done
echo "Deployment verification FAILED: process replacement or cache freshness not confirmed." >&2
exit 1
