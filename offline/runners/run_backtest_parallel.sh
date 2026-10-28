#!/usr/bin/env bash
# run_backtest_parallel.sh — runs backtest commands from a file in parallel on DEV (N cores).
# Usage: offline/runners/run_backtest_parallel.sh <command_file> [JOBS]
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-$ROOT}"
cd "$REPO_ROOT"

CMDFILE="${1:?give a file with commands (one per line)}"
JOBS="${2:-$(nproc)}"
[[ "$JOBS" =~ ^[0-9]+$ ]] && [ "$JOBS" -ge 1 ] || JOBS=1
OUTDIR="$REPO_ROOT/logs/backtest_parallel/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

mapfile -t CMDS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CMDFILE")
echo "[parallel] ${#CMDS[@]} commands, JOBS=$JOBS -> $OUTDIR"

idx=0
for cmd in "${CMDS[@]}"; do
  idx=$((idx + 1))
  log="$OUTDIR/$(printf '%02d' "$idx").log"
  printf '%02d\t%s\n' "$idx" "$cmd" >> "$OUTDIR/manifest.tsv"
  echo "=== [$idx] $cmd ===" > "$log"
  ( rc=0; eval "$cmd" >> "$log" 2>&1 || rc=$?; echo "[exit $rc]" >> "$log" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
done
wait || true
echo "[parallel] DONE. Logs in $OUTDIR"
sed 's/^/  /' "$OUTDIR/manifest.tsv"
