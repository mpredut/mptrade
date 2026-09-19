#!/usr/bin/env bash
# manage_backups.sh — Unified script for backup and disaster recovery.
# Usage:
#   ./manage_backups.sh local                     # Create a local backup of secrets and state
#   ./manage_backups.sh remote                    # Create a local backup and upload encrypted to Storj
#   ./manage_backups.sh restore <secrets_folder>  # Rebuild the machine using a backup folder

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cmd="${1:-}"
shift || true

backup_local() {
    local OUT="${1:-$HOME/$(basename "$ROOT")-secrets-backup}"
    case "$OUT" in "$ROOT"|"$ROOT"/*) echo "❌ the destination can NOT be inside the repo: $OUT"; exit 1;; esac
    
    cd "$ROOT"
    local LIST
    LIST="$(git ls-files --others --ignored --exclude-standard | grep -vE '^(myenv|\.venv)/' | grep -vE '(__pycache__|\.pyc$|\.log($|\.)|\.lock$|^index\.html$|^\.claude/)')"
    [ -n "$LIST" ] || { echo "❌ nothing to save"; exit 1; }
    
    rm -rf "$OUT"; mkdir -p "$OUT"
    printf '%s\n' "$LIST" | tar cf - -C "$ROOT" -T - | tar xf - -C "$OUT"
    
    mkdir -p "$OUT/_machine"
    local PIA_TOKEN="${PIA_DIP_TOKEN:-$HOME/piatoken.txt}"
    if [ -f "$PIA_TOKEN" ]; then
        install -m 0600 "$PIA_TOKEN" "$OUT/_machine/piatoken.txt"
    fi
    
    tar czf "$OUT.tar.gz" -C "$OUT" .
    chmod -R go-rwx "$OUT" 2>/dev/null || true
    chmod 600 "$OUT.tar.gz"
    
    local KEEP="${BACKUP_KEEP:-7}"
    local DATED="$OUT-$(date +%Y%m%d).tar.gz"
    cp -p "$OUT.tar.gz" "$DATED" && chmod 600 "$DATED"
    ls -1t "$OUT"-????????.tar.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f
    
    local N
    N="$(find "$OUT" -type f | wc -l)"
    echo "=== COMPLETE backup: $N files ==="
    echo "Folder : $OUT"
    echo "Tarball: $OUT.tar.gz + history keeping last $KEEP"
}

backup_remote() {
    local RCLONE="${RCLONE:-$HOME/bin/rclone}"
    command -v "$RCLONE" >/dev/null 2>&1 || RCLONE=rclone
    local REMOTE="${RCLONE_REMOTE:-storj-crypt:}"
    local BACKUP_NAME="${BACKUP_NAME:-$(basename "$ROOT")-secrets-backup}"
    local TAR="${BACKUP_TAR:-$HOME/$BACKUP_NAME.tar.gz}"
    local DEST="${REMOTE}${BACKUP_NAME}.tar.gz"
    
    echo "$(date '+%F %T') === backup_remote ==="
    backup_local "$HOME/$BACKUP_NAME" >/dev/null
    [ -f "$TAR" ] || { echo "❌ local tarball missing: $TAR"; exit 1; }
    
    "$RCLONE" copyto "$TAR" "$DEST" --transfers 1
    echo "$(date '+%F %T') ✔ encrypted upload -> $DEST"
}

restore_backup() {
    local SECRETS="${1:-}"
    fail() { echo "❌ $*" >&2; exit 1; }
    
    echo "===== RESTORE @ $ROOT ====="
    [ -n "$SECRETS" ] || fail "Usage: $0 restore <secrets_folder>"
    [ -d "$SECRETS" ] || fail "The secrets folder does not exist: $SECRETS"
    command -v python3 >/dev/null || fail "python3 is missing"
    
    echo "--- [1/5] restoring the secrets plus the state from $SECRETS ---"
    tar cf - --exclude='./_machine' -C "$SECRETS" . | tar xf - -C "$ROOT"
    if [ -f "$SECRETS/_machine/piatoken.txt" ]; then
        install -m 0600 "$SECRETS/_machine/piatoken.txt" "$HOME/piatoken.txt"
    fi
    echo "    ✔ restored"
    
    echo "--- [2/5] venv (myenv) + dependencies ---"
    [ -x "$ROOT/myenv/bin/python" ] || python3 -m venv "$ROOT/myenv" || fail "cannot create the venv"
    "$ROOT/myenv/bin/pip" install -q --upgrade pip
    "$ROOT/myenv/bin/pip" install -q -r "$ROOT/requirements.txt" || fail "pip install failed"
    echo "    ✔ dependencies installed"
    
    echo "--- [3/5] systemd + DNS + SSH + cron (needs sudo) ---"
    if sudo -v 2>/dev/null; then
        sudo env TRADING_ROOT="$ROOT" TRADING_USER="$(id -un)" TRADING_PYTHON="$ROOT/myenv/bin/python" bash "$ROOT/systemd/install_prod.sh"
        echo "    ✔ PROD profile installed"
    else
        echo "    ! no sudo — by hand: sudo bash systemd/install_prod.sh"
    fi
    
    crontab -l >/dev/null 2>&1 && echo "--- [4/5] cron check: installed ---"
    echo "--- [5/5] DONE ---"
}

case "$cmd" in
    local) backup_local "$@" ;;
    remote) backup_remote "$@" ;;
    restore) restore_backup "$@" ;;
    *) echo "Usage: $0 {local|remote|restore}"; exit 1 ;;
esac
