#!/usr/bin/env bash
# run_backtest_cycle.sh — runs scheduled_pilot.py in --propose mode on DEV and publishes
# proposals to the backtest-proposals git branch using a separate git worktree.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-$ROOT}"
ONLY="${PILOT_ONLY:-}"
cd "$REPO_ROOT"

exec 9>"/tmp/mptrade_backtest_cycle.lock"
flock -n 9 || { echo "[cycle $(date '+%F %T')] Another backtest cycle is already running — skipping"; exit 0; }

echo "[cycle $(date '+%F %T')] pilot --propose (only='${ONLY:-all}')"
args=(--propose)
[ -n "$ONLY" ] && args+=(--only "$ONLY")
"$PYTHON_BIN" offline/research/monitortrades_backtest/scheduled_pilot.py "${args[@]}"

echo "[cycle $(date '+%F %T')] publish proposals to git"
DEFAULT_WT="$(dirname "$REPO_ROOT")/mptrade-proposals"
[ ! -d "$DEFAULT_WT" ] && [ -d "$(dirname "$REPO_ROOT")/binance-proposals" ] && DEFAULT_WT="$(dirname "$REPO_ROOT")/binance-proposals"
WT="${WT:-$DEFAULT_WT}"
BRANCH="$BACKTEST_PROPOSALS_BRANCH"
SRC_FILE="$REPO_ROOT/backtest_proposals.json"
[ -f "$SRC_FILE" ] || { echo "[publish] is missing $SRC_FILE — run the pilot with --propose first"; exit 1; }

export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2}"
timeout 30 git -C "$REPO_ROOT" fetch -q origin || true

if ! git -C "$REPO_ROOT" worktree list --porcelain | grep -q "worktree $WT"; then
  rm -rf "$WT"
  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git -C "$REPO_ROOT" worktree add -q "$WT" "$BRANCH"
  else
    git -C "$REPO_ROOT" worktree add -q -b "$BRANCH" "$WT" origin/main
  fi
else
  timeout 30 git -C "$WT" pull -q --ff-only origin "$BRANCH" 2>/dev/null || true
fi

cp "$SRC_FILE" "$WT/backtest_proposals.json"
git -C "$WT" add -f backtest_proposals.json
if git -C "$WT" diff --cached --quiet; then
  echo "[publish] the proposals are unchanged — nothing to push"
else
  n=$(grep -c '"full_key"' "$WT/backtest_proposals.json" || true)
  git -C "$WT" commit -q -m "backtest proposals $(date '+%F %T') [$(git -C "$REPO_ROOT" rev-parse --short HEAD)] — $n proposal(s)"
  timeout 30 git -C "$WT" push -q origin "$BRANCH"
  echo "[publish] $n proposal(s) published on branch $BRANCH"
fi
echo "[cycle $(date '+%F %T')] done"
