#!/usr/bin/env bash
# sync_wsl_cache.sh — Unpacks the latest PROD backup archive (when --unpack-backup is passed)
# and rsyncs live cachedb/ data (including high-density 365d/24h JSONL and .npz archives)
# from PROD and DEV into the local WSL repository (/home/mariusp/mptrade).
set -euo pipefail

# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/orchestratorOS/lib/backup_sync_common.sh"

BACKUP_TAR="${BACKUP_TAR:-$HOME/mptrade-secrets-backup.tar.gz}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/mptrade-secrets-backup}"

exec 9>"/tmp/sync_wsl_cache.lock"
flock -w 120 9 || { echo "[sync_wsl_cache $(date '+%F %T')] another sync is running — skipping"; exit 0; }

if [[ "${1:-}" == "--unpack-backup" ]] && [ -f "$BACKUP_TAR" ]; then
    echo "[sync_wsl_cache $(date '+%F %T')] unpacking $BACKUP_TAR -> $BACKUP_DIR and $REPO_ROOT"
    rm -rf "$BACKUP_DIR" && mkdir -p "$BACKUP_DIR"
    tar xzf "$BACKUP_TAR" -C "$BACKUP_DIR"
    unpack_repo_payload "$BACKUP_DIR" "$REPO_ROOT" --exclude='logs' --exclude='lock'
fi

echo "[sync_wsl_cache $(date '+%F %T')] rsync cachedb/ from mptrade-prod -> $REPO_ROOT/cachedb/"
rsync_cachedb "mptrade-prod:/home/predut/mptrade/cachedb/" "$REPO_ROOT/cachedb/" -e "$SSH_FAST" || true

echo "[sync_wsl_cache $(date '+%F %T')] rsync *.npz and dev_regression_status.json from mptrade-dev -> $REPO_ROOT/cachedb/"
rsync_cachedb "mptrade-dev:/home/predut/mptrade/cachedb/" "$REPO_ROOT/cachedb/" -e "$SSH_FAST" \
    --include '*.npz' --include 'dev_regression_status.json' --exclude '*' || true

echo "[sync_wsl_cache $(date '+%F %T')] done"
