#!/usr/bin/env bash
# sync_wsl_cache.sh — Unpacks the latest PROD backup archive (when --unpack-backup is passed)
# and rsyncs live cachedb/ data (including high-density 365d/24h JSONL and .npz archives)
# from PROD and DEV into the local WSL repository (/home/mariusp/mptrade).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_TAR="${BACKUP_TAR:-$HOME/mptrade-secrets-backup.tar.gz}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/mptrade-secrets-backup}"

exec 9>"/tmp/sync_wsl_cache.lock"
if ! flock -w 120 9; then
    echo "[sync_wsl_cache $(date '+%F %T')] another sync is running — skipping"
    exit 0
fi

if [[ "${1:-}" == "--unpack-backup" ]] && [ -f "$BACKUP_TAR" ]; then
    echo "[sync_wsl_cache $(date '+%F %T')] unpacking $BACKUP_TAR -> $BACKUP_DIR and $REPO_ROOT"
    rm -rf "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    tar xzf "$BACKUP_TAR" -C "$BACKUP_DIR"

    tar cf - -C "$BACKUP_DIR" \
        --exclude='.gemini_backup' \
        --exclude='.ssh_backup' \
        --exclude='.rclone_backup' \
        --exclude='.host_secrets' \
        --exclude='.claude' \
        --exclude='.venv' \
        --exclude='myenv' \
        . | tar xf - -C "$REPO_ROOT"
fi

echo "[sync_wsl_cache $(date '+%F %T')] rsync cachedb/ from mptrade-prod -> $REPO_ROOT/cachedb/"
rc=0
rsync -a --info=stats1 \
    --exclude '*.tmp' \
    --exclude '*.tmp.*' \
    --exclude '*.previous' \
    --exclude '*.lock' \
    "mptrade-prod:/home/predut/mptrade/cachedb/" "$REPO_ROOT/cachedb/" 2>&1 || rc=$?
if [ "$rc" != 0 ] && [ "$rc" != 23 ] && [ "$rc" != 24 ]; then
    echo "[sync_wsl_cache $(date '+%F %T')] WARN: rsync from mptrade-prod exited with $rc" >&2
fi

echo "[sync_wsl_cache $(date '+%F %T')] rsync *.npz and dev_regression_status.json from mptrade-dev -> $REPO_ROOT/cachedb/"
rc_dev=0
rsync -a --info=stats1 \
    --include '*.npz' \
    --include 'dev_regression_status.json' \
    --exclude '*' \
    "mptrade-dev:/home/predut/mptrade/cachedb/" "$REPO_ROOT/cachedb/" 2>&1 || rc_dev=$?
if [ "$rc_dev" != 0 ] && [ "$rc_dev" != 23 ] && [ "$rc_dev" != 24 ]; then
    echo "[sync_wsl_cache $(date '+%F %T')] WARN: rsync from mptrade-dev exited with $rc_dev" >&2
fi

echo "[sync_wsl_cache $(date '+%F %T')] done"
