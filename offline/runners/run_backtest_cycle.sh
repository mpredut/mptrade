#!/usr/bin/env bash
# run_backtest_cycle.sh — a complete backtest cycle on DEV: it runs the pilot in
# --propose mode (it applies nothing) and publishes the proposals to the git branch.
# Uses a separate git WORKTREE to push to `backtest-proposals` without touching main.
# Triggered from prod (trigger_backtest_dev.sh through ssh) or by hand.
#
# The data (cachedb) and the code (git pull) are brought up to date by refresh_dev.sh RUN
# ON PROD before this cycle is triggered — here we only run and publish.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-${ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}}"
source "$REPO_ROOT/orchestratorOS/lib/env_common.sh"
ONLY="${PILOT_ONLY:-}"            # empty = every key; e.g. "maxage,hardtp"
cd "$REPO_ROOT"

# 1. Run scheduled pilot
echo "[cycle $(date '+%F %T')] pilot --propose (only='${ONLY:-all}')"
args=(--propose)
[ -n "$ONLY" ] && args+=(--only "$ONLY")
"$PYTHON_BIN" offline/research/monitortrades_backtest/scheduled_pilot.py "${args[@]}"

# 2. Publish proposals
echo "[cycle $(date '+%F %T')] publish proposals to git"
DEFAULT_WT="$(dirname "$REPO_ROOT")/mptrade-proposals"
if [ ! -d "$DEFAULT_WT" ] && [ -d "$(dirname "$REPO_ROOT")/binance-proposals" ]; then
  DEFAULT_WT="$(dirname "$REPO_ROOT")/binance-proposals"
fi
WT="${WT:-$DEFAULT_WT}"
BRANCH="$BACKTEST_PROPOSALS_BRANCH"
SRC_FILE="$REPO_ROOT/backtest_proposals.json"

[ -f "$SRC_FILE" ] || { echo "[publish] is missing $SRC_FILE — run the pilot with --propose first"; exit 1; }

git -C "$REPO_ROOT" fetch -q origin || true

if ! git -C "$REPO_ROOT" worktree list --porcelain | grep -q "worktree $WT"; then
  rm -rf "$WT"
  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git -C "$REPO_ROOT" worktree add -q "$WT" "$BRANCH"
  else
    git -C "$REPO_ROOT" worktree add -q -b "$BRANCH" "$WT" origin/main
  fi
else
  git -C "$WT" pull -q --ff-only origin "$BRANCH" 2>/dev/null || true
fi

cp "$SRC_FILE" "$WT/backtest_proposals.json"
git -C "$WT" add -f backtest_proposals.json
if git -C "$WT" diff --cached --quiet; then
  echo "[publish] the proposals are unchanged — nothing to push"
else
  n=$(grep -c '"full_key"' "$WT/backtest_proposals.json" || true)
  git -C "$WT" commit -q -m "backtest proposals $(date '+%F %T') [$(git -C "$REPO_ROOT" rev-parse --short HEAD)] — $n proposal(s)"
  git -C "$WT" push -q origin "$BRANCH"
  echo "[publish] $n proposal(s) published on branch $BRANCH"
fi

echo "[cycle $(date '+%F %T')] done"
