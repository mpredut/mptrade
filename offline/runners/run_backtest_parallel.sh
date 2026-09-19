#!/usr/bin/env bash
# run_backtest_parallel.sh — runs backtests IN PARALLEL on dev (N cores).
# The backtest is CPU-bound and independent per parameter => embarrassingly parallel.
#
# Usage: offline/runners/run_backtest_parallel.sh <command_file> [JOBS]
#   <command_file>: one command per line (# = a comment, blank lines are ignored).
#   JOBS: how many at once (default nproc).
# Each command runs in its own process, with the output in
#   logs/backtest_parallel/<timestamp>/NN.log  (plus manifest.tsv with the NN->command mapping).
#
# Runs on DEV (the backtest machine). It does NOT touch the live config or processes.
set -uo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BINANCE_REPO_ROOT:-${ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}}"
cd "$REPO_ROOT"
CMDFILE="${1:?give a file with commands (one per line)}"
JOBS="${2:-$(nproc)}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
  JOBS=1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$REPO_ROOT/logs/backtest_parallel/$STAMP"; mkdir -p "$OUTDIR"

mapfile -t CMDS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CMDFILE")
echo "[parallel] ${#CMDS[@]} commands, JOBS=$JOBS -> $OUTDIR"

idx=0
for cmd in "${CMDS[@]}"; do
  idx=$((idx + 1))
  log="$OUTDIR/$(printf '%02d' "$idx").log"
  printf '%02d\t%s\n' "$idx" "$cmd" | tee -a "$OUTDIR/manifest.tsv" >/dev/null
  echo "=== [$idx] $cmd ===" > "$log"
  ( eval "$cmd" >> "$log" 2>&1; echo "[exit $?]" >> "$log" ) &
  # limit concurrency to JOBS
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait

echo "[parallel] DONE. Logs in $OUTDIR"
echo "[parallel] manifest:"; cat "$OUTDIR/manifest.tsv" | sed 's/^/  /'
