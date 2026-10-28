#!/usr/bin/env bash
# manage_backups.sh — Unified script for backup and disaster recovery.
# Usage:
#   ./manage_backups.sh local                     # Create a local full DR backup of secrets, state, and cachedb
#   ./manage_backups.sh remote                    # Create local full DR backup + upload raw price-only archive to Storj
#   ./manage_backups.sh pull-prices               # Download and unpack raw price-only archive from Storj into cachedb/
#   ./manage_backups.sh restore <secrets_folder>  # Rebuild the machine using a backup folder or tarball

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cmd="${1:-}"
shift || true

read_env_var() {
    local key="$1" file="${2:-$ROOT/.env}"
    [ -f "$file" ] || return 0
    grep -E "^[[:space:]]*${key}=" "$file" | tail -n 1 | cut -d= -f2- | sed -E 's/^[[:space:]]*["'\'']?//; s/["'\'']?[[:space:]]*$//'
}

ensure_rclone_storj_config() {
    local access_key secret_key endpoint bucket
    access_key="${STORJ_ACCESS_KEY:-${STORJ_ACCES_KEY:-$(read_env_var STORJ_ACCESS_KEY)}}"
    [ -n "$access_key" ] || access_key="$(read_env_var STORJ_ACCES_KEY)"
    secret_key="${STORJ_SECRET_KEY:-$(read_env_var STORJ_SECRET_KEY)}"
    endpoint="${STORJ_ENDPOINT:-$(read_env_var STORJ_ENDPOINT)}"
    bucket="${STORJ_BUCKET:-$(read_env_var STORJ_BUCKET)}"

    # Sanitize accidental leading characters before http(s):// (e.g. Yhttps://...)
    if [[ "$endpoint" == *http* ]]; then
        endpoint="http${endpoint#*http}"
    fi
    endpoint="${endpoint:-https://gateway.storjshare.io}"
    export STORJ_BUCKET_RESOLVED="${bucket:-mptrade-price-data}"

    if [ -n "$access_key" ] && [ -n "$secret_key" ]; then
        mkdir -p "$HOME/.config/rclone"
        chmod 700 "$HOME/.config/rclone" 2>/dev/null || true
        local conf="$HOME/.config/rclone/rclone.conf"
        if [ -f "$conf" ] && grep -q '^\[storj\]' "$conf" 2>/dev/null; then
            # Remove existing [storj] stanza before writing updated credentials
            awk '
                /^\[storj\]$/ { skip=1; next }
                /^\[/ { skip=0 }
                !skip { print }
            ' "$conf" > "${conf}.tmp" && mv "${conf}.tmp" "$conf"
        fi
        cat >> "$conf" <<EOF
[storj]
type = s3
provider = Storj
access_key_id = ${access_key}
secret_access_key = ${secret_key}
endpoint = ${endpoint}
EOF
        chmod 600 "$conf" 2>/dev/null || true
    fi
}

backup_local() {
    local OUT="${1:-$HOME/$(basename "$ROOT")-secrets-backup}"
    case "$OUT" in "$ROOT"|"$ROOT"/*) echo "❌ the destination can NOT be inside the repo: $OUT"; exit 1;; esac
    
    ensure_rclone_storj_config
    cd "$ROOT"
    local LIST
    LIST="$(git ls-files --others --ignored --exclude-standard | grep -vE '^(myenv|\.venv)/' | grep -vE '(__pycache__|\.pyc$|\.log($|\.)|\.lock$|\.tmp($|\.)|\.previous$|^index\.html$|^\.claude/)')"
    [ -n "$LIST" ] || { echo "❌ nothing to save"; exit 1; }
    
    rm -rf "$OUT"; mkdir -p "$OUT"
    printf '%s\n' "$LIST" | tar cf - --exclude='*.tmp' --exclude='*.tmp.*' --exclude='*.previous' --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T - | tar xf - -C "$OUT"
    
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

    # Preserve SSH keys, client config, and authorized_keys for GitHub autodeploy & PROD->DEV orchestration
    if [ -d "$HOME/.ssh" ]; then
        mkdir -p "$OUT/.ssh_backup"
        for ssh_item in id_ed25519 id_ed25519.pub id_rsa id_rsa.pub authorized_keys config known_hosts; do
            [ -f "$HOME/.ssh/$ssh_item" ] && cp -p "$HOME/.ssh/$ssh_item" "$OUT/.ssh_backup/" 2>/dev/null || true
        done
    fi

    # Preserve rclone config (Storj encrypted backup credentials) and PIA dedicated IP token if present
    if [ -f "$HOME/.config/rclone/rclone.conf" ]; then
        mkdir -p "$OUT/.rclone_backup"
        cp -p "$HOME/.config/rclone/rclone.conf" "$OUT/.rclone_backup/rclone.conf" 2>/dev/null || true
    fi
    if [ -f "$HOME/piatoken.txt" ]; then
        mkdir -p "$OUT/.host_secrets"
        cp -p "$HOME/piatoken.txt" "$OUT/.host_secrets/piatoken.txt" 2>/dev/null || true
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

backup_price_data_only() {
    local PRICE_TAR="${1:-$HOME/mptrade-price-data.tar.gz}"
    cd "$ROOT"
    local PRICE_FILES
    PRICE_FILES="$(find cachedb -maxdepth 1 -type f \( \
        -name 'cache_365_price_*.jsonl' -o \
        -name 'cache_365_price_*.jsonl.meta' -o \
        -name 'cache_24_price_*.json' -o \
        -name 'cache_24_price_*.json.meta' -o \
        -name 'cache_sparse_price_*.jsonl' -o \
        -name 'cache_sparse_price_*.jsonl.meta' -o \
        -name 'cache_prices_multi.json' -o \
        -name 'cache_prices_multi.json.meta' \
    \) | sort)"
    [ -n "$PRICE_FILES" ] || { echo "❌ no raw price files found in cachedb/"; exit 1; }

    printf '%s\n' "$PRICE_FILES" | tar czf "$PRICE_TAR" --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T -
    chmod 600 "$PRICE_TAR"
    local N
    N="$(printf '%s\n' "$PRICE_FILES" | wc -l)"
    echo "=== PRICE-ONLY archive: $N raw price files -> $PRICE_TAR ==="
}

backup_remote() {
    ensure_rclone_storj_config
    local RCLONE="${RCLONE:-$HOME/bin/rclone}"
    command -v "$RCLONE" >/dev/null 2>&1 || RCLONE=rclone
    local BUCKET="${STORJ_BUCKET_RESOLVED:-mptrade-price-data}"
    local DEFAULT_REMOTE="storj:${BUCKET}/"
    local REMOTE="${RCLONE_REMOTE:-$DEFAULT_REMOTE}"
    local LOCAL_BACKUP_NAME="${BACKUP_NAME:-$(basename "$ROOT")-secrets-backup}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"
    local DATED_NAME="${PRICE_NAME}-$(date +%Y%m%d).tar.gz"
    local DEST="${REMOTE}${PRICE_NAME}.tar.gz"
    local DEST_DATED="${REMOTE}${DATED_NAME}"
    
    echo "$(date '+%F %T') === backup_remote (local full DR + Storj raw price data only) ==="
    backup_local "$HOME/$LOCAL_BACKUP_NAME"
    backup_price_data_only "$PRICE_TAR"
    [ -f "$PRICE_TAR" ] || { echo "❌ price tarball missing: $PRICE_TAR"; exit 1; }
    
    if [[ "$REMOTE" == storj:* ]]; then
        "$RCLONE" mkdir "${REMOTE%/}" --contimeout 15s --timeout 60s 2>/dev/null || true
    fi
    "$RCLONE" copyto "$PRICE_TAR" "$DEST" --transfers 1 --retries 3 --low-level-retries 5 --contimeout 15s --timeout 120s
    echo "$(date '+%F %T') ✔ remote price upload -> $DEST"
    "$RCLONE" copyto "$PRICE_TAR" "$DEST_DATED" --transfers 1 --retries 3 --low-level-retries 5 --contimeout 15s --timeout 120s
    echo "$(date '+%F %T') ✔ remote dated price upload -> $DEST_DATED"

    local KEEP="${BACKUP_KEEP:-7}"
    local old_files
    old_files="$("$RCLONE" lsf "$REMOTE" --include "${PRICE_NAME}-????????.tar.gz" --contimeout 15s --timeout 60s 2>/dev/null | sort -r | tail -n +"$((KEEP + 1))" || true)"
    if [ -n "$old_files" ]; then
        while IFS= read -r old_f; do
            [ -n "$old_f" ] && "$RCLONE" deletefile "${REMOTE}${old_f}" --contimeout 15s --timeout 60s 2>/dev/null || true
        done <<< "$old_files"
    fi
}

pull_remote_price_data() {
    ensure_rclone_storj_config
    local RCLONE="${RCLONE:-$HOME/bin/rclone}"
    command -v "$RCLONE" >/dev/null 2>&1 || RCLONE=rclone
    local BUCKET="${STORJ_BUCKET_RESOLVED:-mptrade-price-data}"
    local REMOTE="${RCLONE_REMOTE:-storj:${BUCKET}/}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"

    echo "$(date '+%F %T') === pull_remote_price_data from ${REMOTE}${PRICE_NAME}.tar.gz ==="
    "$RCLONE" copyto "${REMOTE}${PRICE_NAME}.tar.gz" "$PRICE_TAR" --transfers 1 --retries 3 --low-level-retries 5 --contimeout 15s --timeout 120s
    [ -f "$PRICE_TAR" ] || { echo "❌ failed to download $PRICE_TAR from Storj"; exit 1; }
    mkdir -p "$ROOT/cachedb"
    tar xzf "$PRICE_TAR" -C "$ROOT"
    echo "$(date '+%F %T') ✔ restored raw price archives into $ROOT/cachedb"
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
    tar cf - -C "$SECRETS" \
        --exclude='.gemini_backup' \
        --exclude='.ssh_backup' \
        --exclude='.rclone_backup' \
        --exclude='.host_secrets' \
        . | tar xf - -C "$ROOT"
    
    if [ -d "$SECRETS/.gemini_backup" ]; then
        echo "    ✔ restoring Antigravity CLI (agy) OAuth credentials, projects, and config to ~/.gemini"
        mkdir -p "$HOME/.gemini/antigravity-cli" "$HOME/.gemini/config"
        [ -d "$SECRETS/.gemini_backup/antigravity-cli" ] && cp -a "$SECRETS/.gemini_backup/antigravity-cli/." "$HOME/.gemini/antigravity-cli/" 2>/dev/null || true
        [ -d "$SECRETS/.gemini_backup/config" ] && cp -a "$SECRETS/.gemini_backup/config/." "$HOME/.gemini/config/" 2>/dev/null || true
        chmod 600 "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" 2>/dev/null || true
    fi

    if [ -d "$SECRETS/.ssh_backup" ]; then
        echo "    ✔ restoring SSH keys and config to ~/.ssh"
        mkdir -p "$HOME/.ssh"
        chmod 700 "$HOME/.ssh"
        for ssh_item in id_ed25519 id_ed25519.pub id_rsa id_rsa.pub config known_hosts; do
            [ -f "$SECRETS/.ssh_backup/$ssh_item" ] && cp -p "$SECRETS/.ssh_backup/$ssh_item" "$HOME/.ssh/$ssh_item" 2>/dev/null || true
        done
        if [ -f "$SECRETS/.ssh_backup/authorized_keys" ]; then
            touch "$HOME/.ssh/authorized_keys"
            while IFS= read -r key_line; do
                [ -n "$key_line" ] && ! grep -Fqx "$key_line" "$HOME/.ssh/authorized_keys" 2>/dev/null && printf '%s\n' "$key_line" >> "$HOME/.ssh/authorized_keys"
            done < "$SECRETS/.ssh_backup/authorized_keys"
        fi
        chmod 600 "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa" "$HOME/.ssh/config" "$HOME/.ssh/authorized_keys" 2>/dev/null || true
        chmod 644 "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/known_hosts" 2>/dev/null || true
    fi

    if [ -f "$SECRETS/.rclone_backup/rclone.conf" ]; then
        echo "    ✔ restoring rclone configuration to ~/.config/rclone/rclone.conf"
        mkdir -p "$HOME/.config/rclone"
        cp -p "$SECRETS/.rclone_backup/rclone.conf" "$HOME/.config/rclone/rclone.conf" 2>/dev/null || true
        chmod 600 "$HOME/.config/rclone/rclone.conf" 2>/dev/null || true
    fi

    if [ -f "$SECRETS/.host_secrets/piatoken.txt" ]; then
        echo "    ✔ restoring PIA token to ~/piatoken.txt"
        cp -p "$SECRETS/.host_secrets/piatoken.txt" "$HOME/piatoken.txt" 2>/dev/null || true
        chmod 600 "$HOME/piatoken.txt" 2>/dev/null || true
    fi

    # If high-density 365d price archives are missing from the restored backup, pull them from Storj if available
    if ! ls "$ROOT"/cachedb/cache_365_price_*.jsonl >/dev/null 2>&1; then
        echo "    ! 365d price archives missing locally; attempting pull from Storj..."
        pull_remote_price_data || echo "    ! Storj price pull skipped or unavailable"
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
    pull-prices) pull_remote_price_data "$@" ;;
    restore) restore_backup "$@" ;;
    *) echo "Usage: $0 {local|remote|pull-prices|restore}"; exit 1 ;;
esac
