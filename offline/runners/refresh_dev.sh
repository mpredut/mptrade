#!/usr/bin/env bash
# refresh_dev.sh — brings the BACKTEST machine (dev) up to date with prod via two channels:
#   1. CODE: `git pull --ff-only` on dev (code travels through GitHub, not this script).
#   2. DATA (cachedb, untracked by git): rsync prod -> dev. Only the price archives
#      (cache_price_*.jsonl) are strictly needed by backtesting; we sync all of
#      cachedb so that dev is a real mirror (rsync transfers only the delta, cheaply).
#
# Runs on PROD (it holds the SSH key to dev). It does NOT touch prod and does NOT git pull on
# prod (the live machine with real money — the code is pulled there DELIBERATELY, never automatically).
# Idempotent; safe to put in cron on prod.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"

# A MUTEX (4 Aug): refresh_dev can be started CONCURRENTLY — once from its own cron (*/30,
# so also at 00:00/12:00) AND once called by trigger_backtest_dev.sh (0 */12 = 00:00/12:00).
# Two simultaneous `git pull --ff-only` runs on dev -> FETCH_HEAD written by both -> 'Cannot
# fast-forward to multiple branches'. flock serialises them: the second waits (at most 300s) for
# finish first, then runs (git pull becomes 'already up to date', cheap). No races.
exec 9>"/tmp/refresh_dev.lock"
if ! flock -w 300 9; then
  echo "[refresh_dev $(date '+%F %T')] another refresh_dev has run for >300s — skipping"
  exit 0
fi

REPO_ROOT="${BINANCE_REPO_ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}"
SSH="ssh -o BatchMode=yes -p $DEV_PORT"

echo "[refresh_dev $(date '+%F %T')] code: git pull --ff-only on dev"
$SSH "$DEV_USER@$DEV_HOST" "cd ~/$DEV_PATH && git pull --ff-only origin $DEV_CODE_BRANCH" 2>&1 | sed 's/^/  /'

echo "[refresh_dev $(date '+%F %T')] data: rsync cachedb/ prod -> dev"
# --exclude '*.tmp': cacheManager writes atomically through temporary .tmp files that come and go
# in real time; there is no point copying them. We tolerate the codes 23/24 (partial or
# vanished files) — benign on a live source, NOT a real sync failure.
rc=0
rsync -a --info=stats1 --exclude '*.tmp' -e "$SSH" \
  "$REPO_ROOT/cachedb/" "$DEV_USER@$DEV_HOST:$DEV_PATH/cachedb/" 2>&1 | sed 's/^/  /' || rc=$?
if [ "$rc" != 0 ] && [ "$rc" != 24 ] && [ "$rc" != 23 ]; then
  echo "[refresh_dev] rsync failed with code $rc"; exit "$rc"
fi

echo "[refresh_dev $(date '+%F %T')] done"
