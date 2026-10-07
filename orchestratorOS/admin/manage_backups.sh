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
    
    # Preserve Antigravity CLI (agy) OAuth credentials, settings, and project configs for Disaster Recovery
    if [ -f "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" ] || [ -d "$HOME/.gemini/config" ]; then
        mkdir -p "$OUT/.gemini_backup/antigravity-cli" "$OUT/.gemini_backup/config"
        [ -f "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" ] && cp -p "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" "$OUT/.gemini_backup/antigravity-cli/" 2>/dev/null || true
        [ -f "$HOME/.gemini/antigravity-cli/installation_id" ] && cp -p "$HOME/.gemini/antigravity-cli/installation_id" "$OUT/.gemini_backup/antigravity-cli/" 2>/dev/null || true
        [ -f "$HOME/.gemini/antigravity-cli/settings.json" ] && cp -p "$HOME/.gemini/antigravity-cli/settings.json" "$OUT/.gemini_backup/antigravity-cli/" 2>/dev/null || true
        [ -f "$HOME/.gemini/config/config.json" ] && cp -p "$HOME/.gemini/config/config.json" "$OUT/.gemini_backup/config/" 2>/dev/null || true
        [ -f "$HOME/.gemini/config/mcp_config.json" ] && cp -p "$HOME/.gemini/config/mcp_config.json" "$OUT/.gemini_backup/config/" 2>/dev/null || true
        if [ -d "$HOME/.gemini/config/projects" ]; then
            mkdir -p "$OUT/.gemini_backup/config/projects"
            cp -a "$HOME/.gemini/config/projects/." "$OUT/.gemini_backup/config/projects/" 2>/dev/null || true
        fi
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

log() { echo "[$(date -u +%FT%TZ)] $*"; }

restore_backup() {
    local SECRETS="${1:-}"
    fail() { echo "❌ $*" >&2; exit 1; }
    
    echo "===== RESTORE @ $ROOT ====="
    [ -n "$SECRETS" ] || fail "Usage: $0 restore <secrets_folder_or_tarball>"
    
    local TMP_DIR=""
    if [ -f "$SECRETS" ]; then
        TMP_DIR="$(mktemp -d)"
        trap 'rm -rf "$TMP_DIR"' EXIT
        echo "--- extracting archive $SECRETS -> $TMP_DIR ---"
        tar xzf "$SECRETS" -C "$TMP_DIR"
        SECRETS="$TMP_DIR"
    fi
    
    [ -d "$SECRETS" ] || fail "The secrets location does not exist or is invalid: $SECRETS"
    command -v python3 >/dev/null || fail "python3 is missing"
    
    echo "--- [1/5] restoring secrets plus state from $SECRETS ---"
    tar cf - -C "$SECRETS" --exclude='.gemini_backup' . | tar xf - -C "$ROOT"
    
    if [ -d "$SECRETS/.gemini_backup" ]; then
        echo "    ✔ restoring Antigravity CLI (agy) OAuth credentials, projects, and config to ~/.gemini"
        mkdir -p "$HOME/.gemini/antigravity-cli" "$HOME/.gemini/config"
        [ -d "$SECRETS/.gemini_backup/antigravity-cli" ] && cp -a "$SECRETS/.gemini_backup/antigravity-cli/." "$HOME/.gemini/antigravity-cli/" 2>/dev/null || true
        [ -d "$SECRETS/.gemini_backup/config" ] && cp -a "$SECRETS/.gemini_backup/config/." "$HOME/.gemini/config/" 2>/dev/null || true
        chmod 600 "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" 2>/dev/null || true
    fi
    
    echo "--- [2/5] venv + dependencies ---"
    local VENV_DIR="$ROOT/myenv"
    [ -d "$ROOT/.venv" ] && VENV_DIR="$ROOT/.venv"
    [ -x "$VENV_DIR/bin/python" ] || python3 -m venv "$VENV_DIR" || fail "cannot create the venv"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    if [ -f "$ROOT/requirements.txt" ]; then
        "$VENV_DIR/bin/pip" install -q -r "$ROOT/requirements.txt" || fail "pip install failed"
    fi
    echo "    ✔ dependencies installed in $VENV_DIR"
    
    echo "--- [3/5] systemd + DNS + SSH + cron (needs sudo) ---"
    local TARGET_USER="${TRADING_USER:-${SUDO_USER:-$(stat -c %U "$ROOT")}}"
    local TARGET_GROUP
    TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
    if [ "$(id -u)" -eq 0 ]; then
        chown -R "$TARGET_USER:$TARGET_GROUP" "$ROOT" 2>/dev/null || true
        env TRADING_ROOT="$ROOT" TRADING_USER="$TARGET_USER" TRADING_PYTHON="$VENV_DIR/bin/python" bash "$ROOT/systemd/install_prod.sh"
        echo "    ✔ PROD profile installed"
    elif sudo -v 2>/dev/null; then
        sudo env TRADING_ROOT="$ROOT" TRADING_USER="$TARGET_USER" TRADING_PYTHON="$VENV_DIR/bin/python" bash "$ROOT/systemd/install_prod.sh"
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
