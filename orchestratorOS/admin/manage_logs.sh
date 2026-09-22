#!/usr/bin/env bash
# manage_logs.sh — combined log rotation and retention
# 1. logrotate: caps the size of continuous nohup console logs via copytruncate.
# 2. retention: age-based compression and deletion for dated files in logger/
#
# A suggested cron (hourly, offset so it does not clash with the healthcheck at :*0/:*5):
# Schedule this script through the rendered production crontab.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== manage_logs $(date '+%Y-%m-%d %H:%M:%S') ==="

# 1. Rotate continuous stream logs via logrotate
POLICY="$ROOT/config.env"
if [ -r "$POLICY" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$POLICY"
    set +a
    LOGROTATE="$(command -v logrotate || echo /usr/sbin/logrotate)"
    if [ -x "$LOGROTATE" ]; then
        # Ensure archive subdirectories exist for stream logs
        mkdir -p "$ROOT/logs/archive" \
                 "$ROOT/kraken/archive" \
                 "$ROOT/hyperliquid/archive" \
                 "$ROOT/212trading/archive" \
                 "$ROOT/binance_api/archive"

        CONF="$(mktemp)"
        {
            cat <<EOF
$ROOT/logs/*.log
$ROOT/kraken/*.log
$ROOT/hyperliquid/*.log
$ROOT/212trading/*.log
$ROOT/binance_api/*.log
$ROOT/*.log
{
    size ${LOGROTATE_MAX_SIZE:-20M}
    rotate ${LOGROTATE_KEEP:-5}
    missingok
    notifempty
    compress
    copytruncate
    olddir archive
}
EOF
        } > "$CONF"
        "$LOGROTATE" -s "$ROOT/.logrotate.state" "$CONF"
        rm -f "$CONF"
        echo "✔ logrotate complete."
    else
        echo "⚠ logrotate not found, skipping."
    fi
else
    echo "⚠ policy $POLICY missing, logrotate skipped."
fi

# 2. Age-based retention for dated logs in logger/
LOGGER_DIR="$ROOT/logger"
LOGGER_ARCHIVE_DIR="$LOGGER_DIR/archive"
COMPRESS_AFTER_DAYS="${LOG_COMPRESS_AFTER_DAYS:-1}"
DELETE_AFTER_DAYS="${LOG_DELETE_AFTER_DAYS:-45}"

if [ -d "$LOGGER_DIR" ]; then
    mkdir -p "$LOGGER_ARCHIVE_DIR"
    echo "  before retention: $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"

    # Compress dated logs older than COMPRESS_AFTER_DAYS (skips today's active file)
    find "$LOGGER_DIR" -maxdepth 1 -name "*.log" -mtime +"$COMPRESS_AFTER_DAYS" -print0 \
        | xargs -0 -r gzip -f

    # Move any compressed .gz files from logger/ root into logger/archive/
    find "$LOGGER_DIR" -maxdepth 1 -name "*.log*.gz" -print0 \
        | while IFS= read -r -d '' gz_file; do
            mv -f "$gz_file" "$LOGGER_ARCHIVE_DIR/"
        done

    # Move any loose .gz files in logs/ into logs/archive/ if left behind
    if [ -d "$ROOT/logs" ]; then
        find "$ROOT/logs" -maxdepth 1 -name "*.log*.gz" -print0 \
            | while IFS= read -r -d '' gz_file; do
                mv -f "$gz_file" "$ROOT/logs/archive/"
            done
    fi

    # Delete archives older than DELETE_AFTER_DAYS from all archive directories
    find "$LOGGER_ARCHIVE_DIR" -maxdepth 1 -name "*.gz" -mtime +"$DELETE_AFTER_DAYS" -print0 \
        | xargs -0 -r rm -f

    for arch_dir in "$ROOT/logs/archive" "$ROOT/kraken/archive" "$ROOT/hyperliquid/archive" "$ROOT/212trading/archive" "$ROOT/binance_api/archive"; do
        if [ -d "$arch_dir" ]; then
            find "$arch_dir" -maxdepth 1 -name "*.gz" -mtime +"$DELETE_AFTER_DAYS" -print0 \
                | xargs -0 -r rm -f
        fi
    done

    echo "  after retention:  $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"
fi
