#!/usr/bin/env bash
# manage_logs.sh — rotates, compresses, and prunes application logs across all venue folders.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env_common.sh
source "$SCRIPT_DIR/../lib/env_common.sh"

LOG_RETENTION_DAYS="$(env_get LOG_RETENTION_DAYS 30)"
LOG_MAX_SIZE="$(env_get LOG_MAX_SIZE 50M)"
LOG_ROTATE_COUNT="$(env_get LOG_ROTATE_COUNT 14)"

STATE_FILE="$ROOT/logs/.logrotate.state"
CONF_FILE="/tmp/mptrade_logrotate_$$.conf"
trap 'rm -f "$CONF_FILE"' EXIT

STREAM_DIRS=(logs kraken hyperliquid 212trading binance_api)
for d in "${STREAM_DIRS[@]}"; do
  mkdir -p "$ROOT/$d/archive"
done

# 1. Move stray rotated .gz files into their respective archive/ folder
for d in logger "${STREAM_DIRS[@]}"; do
  target_arch="$ROOT/${d/logger/logs}/archive"
  if [ -d "$ROOT/$d" ]; then
    find "$ROOT/$d" -maxdepth 1 -type f -name "*.log*.gz" -exec mv -f {} "$target_arch/" \; 2>/dev/null || true
  fi
done

# 2. Generate logrotate config and run rotation if logrotate is installed
if command -v logrotate >/dev/null 2>&1; then
  : > "$CONF_FILE"
  for d in "${STREAM_DIRS[@]}"; do
    cat >> "$CONF_FILE" <<EOF
"$ROOT/$d/*.log" {
    daily
    maxsize $LOG_MAX_SIZE
    rotate $LOG_ROTATE_COUNT
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
    olddir $ROOT/$d/archive
}
EOF
  done
  logrotate -s "$STATE_FILE" "$CONF_FILE" 2>/dev/null || true
fi

# 3. Direct size-cap fallback (truncate any single active log > 200MB if logrotate skipped it)
for d in "${STREAM_DIRS[@]}"; do
  find "$ROOT/$d" -maxdepth 1 -type f -name "*.log" -size +200M | while read -r biglog; do
    ts="$(date '+%Y%m%d_%H%M%S')"
    base="$(basename "$biglog")"
    gzip -c "$biglog" > "$ROOT/$d/archive/${base}.${ts}.gz" && : > "$biglog"
  done
done

# 4. Enforce retention window on archived .gz logs and /tmp test artifacts
for d in "${STREAM_DIRS[@]}"; do
  find "$ROOT/$d/archive" -maxdepth 1 -type f -name "*.gz" -mtime "+$LOG_RETENTION_DAYS" -delete 2>/dev/null || true
done
find /tmp -maxdepth 1 -name "pytest-of-*" -mtime +3 -exec rm -rf {} + 2>/dev/null || true
