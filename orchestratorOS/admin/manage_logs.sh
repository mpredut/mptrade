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

# 1. Rotate logs via logrotate
POLICY="$ROOT/config.env"
if [ -r "$POLICY" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$POLICY"
    set +a
    LOGROTATE="$(command -v logrotate || echo /usr/sbin/logrotate)"
    if [ -x "$LOGROTATE" ]; then
        CONF="$(mktemp)"
        {
            cat <<EOF
$ROOT/logs/*.log
$ROOT/kraken/*.log
$ROOT/hyperliquid/*.log
$ROOT/212trading/*.log
$ROOT/binance_api/*.log
$ROOT/logger/*.log
$ROOT/logger/*.jsonl
$ROOT/logger/execution_audit/*.jsonl
$ROOT/logs/*.jsonl
$ROOT/logs/shadow_live/*.jsonl
$ROOT/logs/hyperliquid_shadow/*.jsonl
$ROOT/*.log
{
    size ${LOGROTATE_MAX_SIZE:-20M}
    rotate ${LOGROTATE_KEEP:-5}
    missingok
    notifempty
    compress
    copytruncate
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

# 2. Age-based retention
LOGGER_DIR="$ROOT/logger"
COMPRESS_AFTER_DAYS=3
DELETE_AFTER_DAYS=45

if [ -d "$LOGGER_DIR" ]; then
    echo "  before retention: $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"

    find "$LOGGER_DIR" -maxdepth 1 -name "*.log" -mtime +$COMPRESS_AFTER_DAYS -print0 \
        | xargs -0 -r gzip -f

    find "$LOGGER_DIR" -maxdepth 1 -name "*.log.gz" -mtime +$DELETE_AFTER_DAYS -print0 \
        | xargs -0 -r rm -f

    echo "  after retention:  $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"
fi
