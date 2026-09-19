#!/usr/bin/env bash
# backup_remote.sh — backup local (backup_local.sh) + upload CRIPTAT off-site, descentralizat (Storj).
# rclone does the encryption (a 'crypt' remote wrapping the Storj remote) -> what reaches Storj is
# ONLY ciphertext. You keep the encryption password SEPARATELY (off-server) so you can decrypt at restore time.
# It overwrites the last version (no bloat). See docs/DISASTER_RECOVERY.md for the config plus the restore.
#
# The archive name follows the checkout directory unless BACKUP_NAME overrides it.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RCLONE="${RCLONE:-$HOME/bin/rclone}"
command -v "$RCLONE" >/dev/null 2>&1 || RCLONE=rclone
REMOTE="${RCLONE_REMOTE:-storj-crypt:}"          # the crypt remote (over Storj)
BACKUP_NAME="${BACKUP_NAME:-$(basename "$ROOT")-secrets-backup}"
TAR="${BACKUP_TAR:-$HOME/$BACKUP_NAME.tar.gz}"
DEST="${REMOTE}${BACKUP_NAME}.tar.gz"

echo "$(date '+%F %T') === backup_remote ==="
# 1. fresh local backup (folder + tarball) — reuses existing script
"$ROOT/tools/admin/backup_local.sh" >/dev/null
[ -f "$TAR" ] || { echo "❌ local tarball missing: $TAR"; exit 1; }

# 2. an ENCRYPTED upload into Storj (it overwrites the last version)
"$RCLONE" copyto "$TAR" "$DEST" --transfers 1
echo "$(date '+%F %T') ✔ encrypted upload -> $DEST  ($("$RCLONE" size "$DEST" 2>/dev/null | tr '\n' ' '))"
