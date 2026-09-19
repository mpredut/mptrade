#!/usr/bin/env bash
# publish_proposals.sh — publishes backtest_proposals.json (written by scheduled_pilot
# --propose) to the dedicated `backtest-proposals` git branch, by pushing. Runs on
# DEV. It uses a separate git WORKTREE (~/binance-proposals) so it does NOT touch
# the working tree on main (where the backtests run).
#
# Flow (UNIFIED_BACKTEST_PLAN.md §9): dev -> github (proposals branch) -> prod
# pulls and applies them with guardrails. The file is gitignored on main; on the proposals
# branch it is tracked with `git add -f`.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-${ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}}"
DEFAULT_WT="$(dirname "$REPO_ROOT")/mptrade-proposals"
if [ ! -d "$DEFAULT_WT" ] && [ -d "$(dirname "$REPO_ROOT")/binance-proposals" ]; then
  DEFAULT_WT="$(dirname "$REPO_ROOT")/binance-proposals"
fi
WT="${WT:-$DEFAULT_WT}"
BRANCH="$BACKTEST_PROPOSALS_BRANCH"
SRC_FILE="$REPO_ROOT/backtest_proposals.json"

[ -f "$SRC_FILE" ] || { echo "[publish] is missing $SRC_FILE — run the pilot with --propose first"; exit 1; }

git -C "$REPO_ROOT" fetch -q origin || true

# Create the worktree if missing: from origin/$BRANCH if it exists on the remote,
# otherwise a new branch started from origin/main.
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
