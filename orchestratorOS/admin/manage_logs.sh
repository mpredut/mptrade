#!/usr/bin/env bash
# manage_logs.sh — combined log rotation, logger/ retention, and 3h agy-bot thread pruning.
# 1. logrotate: caps the size of continuous nohup console logs via copytruncate.
# 2. retention: age-based compression and deletion for dated files in logger/ and stream archives.
# 3. prune_llm_threads.py: runs every 3 hours to prune expired automated agy-bot threads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env_common.sh
source "$SCRIPT_DIR/../lib/env_common.sh"

echo "=== manage_logs $(date '+%Y-%m-%d %H:%M:%S') ==="

STREAM_DIRS=(logs kraken hyperliquid 212trading binance_api)
for d in "${STREAM_DIRS[@]}"; do
    mkdir -p "$ROOT/$d/archive"
done

# 1. Rotate continuous stream logs via logrotate
POLICY="$ROOT/config.env"
if [ -r "$POLICY" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$POLICY"
    set +a
    LOGROTATE="$(command -v logrotate || echo /usr/sbin/logrotate)"
    if [ -x "$LOGROTATE" ]; then
        CONF="$(mktemp)"
        trap 'rm -f "$CONF"' EXIT
        cat > "$CONF" <<EOF
$ROOT/logs/*.log
$ROOT/kraken/*.log
$ROOT/hyperliquid/*.log
$ROOT/212trading/*.log
$ROOT/binance_api/*.log
{
    size ${LOGROTATE_MAX_SIZE:-20M}
    rotate ${LOGROTATE_KEEP:-5}
    missingok
    notifempty
    compress
    copytruncate
    olddir archive
}
$ROOT/*.log
{
    size ${LOGROTATE_MAX_SIZE:-20M}
    rotate ${LOGROTATE_KEEP:-5}
    missingok
    notifempty
    compress
    copytruncate
    olddir $ROOT/logs/archive
}
EOF
        "$LOGROTATE" -s "$ROOT/.logrotate.state" "$CONF"
        rm -f "$CONF"
        echo "✔ logrotate complete."
    else
        echo "⚠ logrotate not found, skipping."
    fi
else
    echo "⚠ policy $POLICY missing, logrotate skipped."
fi

# 2. Age-based retention for dated logs in logger/ and stream archives
LOGGER_DIR="$ROOT/logger"
LOGGER_ARCHIVE_DIR="$LOGGER_DIR/archive"
COMPRESS_AFTER_DAYS="${LOG_COMPRESS_AFTER_DAYS:-1}"
DELETE_AFTER_DAYS="${LOG_DELETE_AFTER_DAYS:-45}"

if [ -d "$LOGGER_DIR" ]; then
    mkdir -p "$LOGGER_ARCHIVE_DIR"
    echo "  before retention: $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"

    find "$LOGGER_DIR" -maxdepth 1 -name "*.log" -mtime +"$COMPRESS_AFTER_DAYS" -print0 | xargs -0 -r gzip -f

    for dir_pair in "$LOGGER_DIR:$LOGGER_ARCHIVE_DIR" "$ROOT/logs:$ROOT/logs/archive"; do
        src_d="${dir_pair%%:*}"; dst_d="${dir_pair#*:}"
        [ -d "$src_d" ] && find "$src_d" -maxdepth 1 -name "*.log*.gz" -exec mv -f {} "$dst_d/" \;
    done

    for arch_dir in "$LOGGER_ARCHIVE_DIR" "${STREAM_DIRS[@]/#/$ROOT/}"; do
        [[ "$arch_dir" != */archive ]] && arch_dir="$arch_dir/archive"
        [ -d "$arch_dir" ] && find "$arch_dir" -maxdepth 1 -name "*.gz" -mtime +"$DELETE_AFTER_DAYS" -print0 | xargs -0 -r rm -f
    done

    BACKTEST_RETENTION_DAYS="${LOG_BACKTEST_DELETE_AFTER_DAYS:-14}"
    for run_dir in "$LOGGER_DIR/backtest" "$ROOT/logs/overnight_runs"; do
        [ -d "$run_dir" ] && find "$run_dir" -mindepth 1 -maxdepth 1 -mtime +"$BACKTEST_RETENTION_DAYS" -exec rm -rf {} + 2>/dev/null || true
    done

    find /tmp -maxdepth 1 -user "$(id -un)" \( -name "tmp*" -o -name "pytest-of-*" -o -name "binance-tests-*" -o -name "mptrade-tests-*" -o -name "rtrade-*" -o -name "test_reentry_state_*" \) -mtime +1 -exec rm -rf {} + 2>/dev/null || true
    echo "  after retention:  $(du -sh "$LOGGER_DIR" 2>/dev/null | cut -f1)"
fi

# 3. Retention for autonomous LLM CLI threads (every 3 hours)
if [ $(( $(date +%-H) % 3 )) -eq 0 ] && [ -f "$ROOT/orchestratorOS/admin/prune_llm_threads.py" ]; then
    echo "  pruning automated agy-bot CLI threads (3h cycle)..."
    "$ROOT/orchestratorOS/run_python.sh" "$ROOT/orchestratorOS/admin/prune_llm_threads.py" || true
fi
