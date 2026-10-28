#!/usr/bin/env bash
# manage_gitautodeploy.sh — safe git auto-pull and conditional fleet restart on PROD.
# Cron: */5 * * * *
# Respects FLEET_PAUSED, active order locks, and AUTODEPLOY_MODE (off|shadow|active).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$SCRIPT_DIR/../lib/os_notify.sh"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"

LOCK="/tmp/mptrade_gitautodeploy.lock"
LAST_MARK="/tmp/mptrade_gitautodeploy_last_sha"
LOG="$ROOT/logs/gitautodeploy.log"

exec 9>"$LOCK"
flock -n 9 || exit 0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

AUTODEPLOY_MODE="$(env_get AUTODEPLOY_MODE "active")"
[ "$AUTODEPLOY_MODE" = "off" ] || [ -f "$ROOT/FLEET_PAUSED" ] && exit 0

LAST_PROCESSED=""
[ -f "$LAST_MARK" ] && LAST_PROCESSED="$(cat "$LAST_MARK" 2>/dev/null || true)"

export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o ConnectTimeout=10 -o BatchMode=yes}"
if ! timeout 25 git fetch -q origin main 2>>"$LOG"; then
  log "WARN: git fetch origin main failed"
  exit 0
fi

LOCAL_SHA="$(git rev-parse HEAD 2>/dev/null || echo "")"
REMOTE_SHA="$(git rev-parse origin/main 2>/dev/null || echo "")"

if [ -z "$LOCAL_SHA" ] || [ -z "$REMOTE_SHA" ] || [ "$LOCAL_SHA" = "$REMOTE_SHA" ] || [ "$REMOTE_SHA" = "$LAST_PROCESSED" ]; then
  exit 0
fi

if ! git merge-base --is-ancestor "$LOCAL_SHA" "$REMOTE_SHA" 2>/dev/null; then
  log "ERROR: origin/main ($REMOTE_SHA) is not a fast-forward of HEAD ($LOCAL_SHA). Skipping."
  echo "$REMOTE_SHA" > "$LAST_MARK"
  send_os_ntfy "AutoDeploy Skipped ($(hostname))" "origin/main (${REMOTE_SHA:0:8}) is not a fast-forward of HEAD (${LOCAL_SHA:0:8}). Manual intervention required." "high"
  exit 1
fi

CHANGED_FILES="$(git diff --name-only "$LOCAL_SHA" "$REMOTE_SHA" 2>/dev/null || true)"
COMMIT_SUMMARY="$(git log --oneline "${LOCAL_SHA}..${REMOTE_SHA}" 2>/dev/null | head -n 5)"

if ! timeout 90 "$PYTHON_BIN" "$ROOT/orchestratorOS/livecheck/verify_deploy_gate.py" --target-sha "$REMOTE_SHA" >>"$LOG" 2>&1; then
  echo "$REMOTE_SHA" > "$LAST_MARK"
  log "BLOCKED: verify_deploy_gate.py vetoed commit ${REMOTE_SHA:0:8}."
  send_os_ntfy "AutoDeploy VETOED ($(hostname))" "Commit ${REMOTE_SHA:0:8} failed verify_deploy_gate.py on $(hostname). Deploy aborted.
Commits:
$COMMIT_SUMMARY" "high"
  exit 1
fi

NEED_RESTART=0
echo "$CHANGED_FILES" | grep -qE '\.(py|conf|env)$|^procs\.conf$|^orchestrator' && NEED_RESTART=1

if [ "$AUTODEPLOY_MODE" = "shadow" ]; then
  echo "$REMOTE_SHA" > "$LAST_MARK"
  log "SHADOW: new commit ${LOCAL_SHA:0:8} -> ${REMOTE_SHA:0:8} (need_restart=$NEED_RESTART)"
  send_os_ntfy "AutoDeploy Shadow ($(hostname))" "New commit on origin/main: ${LOCAL_SHA:0:8} -> ${REMOTE_SHA:0:8} (need_restart=$NEED_RESTART)
$COMMIT_SUMMARY" "default"
  exit 0
fi

git reset --hard "$REMOTE_SHA" >>"$LOG" 2>&1
echo "$REMOTE_SHA" > "$LAST_MARK"
log "DEPLOYED: ${LOCAL_SHA:0:8} -> ${REMOTE_SHA:0:8} (need_restart=$NEED_RESTART)"

if [ "$NEED_RESTART" -eq 1 ]; then
  PY_CHANGED=$(echo "$CHANGED_FILES" | grep -E '\.py$' || true)
  if [ -n "$PY_CHANGED" ]; then
    for f in $PY_CHANGED; do
      if [ -f "$ROOT/$f" ] && ! "$PYTHON_BIN" -m py_compile "$ROOT/$f" 2>>"$LOG"; then
        log "ERROR: Syntax error in $f! Rolling back to $LOCAL_SHA"
        git reset --hard "$LOCAL_SHA" >>"$LOG" 2>&1
        send_os_ntfy "AutoDeploy Rollback ($(hostname))" "Syntax error in $f on commit ${REMOTE_SHA:0:8}. Rolled back to ${LOCAL_SHA:0:8}." "urgent"
        exit 1
      fi
    done
  fi
  log "Touching trigger_restart.flag for graceful orchestrator reload."
  touch "$ROOT/trigger_restart.flag"
  send_os_ntfy "AutoDeploy Applied ($(hostname))" "Updated ${LOCAL_SHA:0:8} -> ${REMOTE_SHA:0:8} and triggered graceful reload.
$COMMIT_SUMMARY" "default"
fi
