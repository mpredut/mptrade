#!/usr/bin/env bash
# run_backtest_parallel.sh -- runs backtest commands from a file in parallel on DEV (N cores).
# Usage: offline/runners/run_backtest_parallel.sh <command_file> [JOBS]
set -euo pipefail

CALLER_PWD="$(pwd)"
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-$ROOT}"
cd "$REPO_ROOT"

RAW_CMDFILE="${1:?give a file with commands (one per line)}"
if [[ "$RAW_CMDFILE" = /* ]]; then
    CMDFILE="$RAW_CMDFILE"
elif [ -f "$CALLER_PWD/$RAW_CMDFILE" ]; then
    CMDFILE="$CALLER_PWD/$RAW_CMDFILE"
else
    CMDFILE="$REPO_ROOT/$RAW_CMDFILE"
fi
if [ ! -f "$CMDFILE" ] || [ ! -r "$CMDFILE" ]; then
    echo "[parallel] ERROR: command file not found or unreadable: $RAW_CMDFILE" >&2
    exit 2
fi
JOBS="${2:-$(nproc)}"
[[ "$JOBS" =~ ^[0-9]+$ ]] && [ "$JOBS" -ge 1 ] || JOBS=1
OUTDIR="$REPO_ROOT/logs/backtest_parallel/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

mapfile -t CMDS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CMDFILE" || true)
if [ "${#CMDS[@]}" -eq 0 ]; then
    echo "[parallel] ERROR: command file contains 0 executable commands: $CMDFILE" >&2
    exit 1
fi
echo "[parallel] ${#CMDS[@]} commands, JOBS=$JOBS -> $OUTDIR"
: > "$OUTDIR/manifest.tsv"

idx=0
pids=()
for cmd in "${CMDS[@]}"; do
  idx=$((idx + 1))
  log="$OUTDIR/$(printf '%02d' "$idx").log"
  printf '%02d\t%s\n' "$idx" "$cmd" >> "$OUTDIR/manifest.tsv"
  echo "=== [$idx] $cmd ===" > "$log"
  ( rc=0; eval "$cmd" >> "$log" 2>&1 || rc=$?; echo "[exit $rc]" >> "$log"; exit "$rc" ) &
  pids+=("$!")
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
echo "[parallel] DONE (failed=$failed/${#CMDS[@]}). Logs in $OUTDIR"
sed 's/^/  /' "$OUTDIR/manifest.tsv"
[ "$failed" -eq 0 ]
