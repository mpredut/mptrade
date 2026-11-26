#!/bin/bash
# deadman_switch.sh -- "dead man's switch" via ntfy scheduled delivery ("In: 35m") + healthchecks.io.
# If the server or cron dies, the scheduled ntfy message is not pushed forward and delivers itself 35m later.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$HERE/../lib/os_notify.sh"

TOPIC="$(env_get NTFY_TOPIC_SERVER "$(env_get NTFY_TOPIC_DEADMAN "$(ntfy_error_topic)")")"
ntfy_load_auth_hdr
HOST="$(hostname)"

DOWN_AFTER_SEC=$((35 * 60))
mkdir -p "$ROOT/logs"
BEAT_FILE="$ROOT/logs/.deadman_last_beat"
NOW=$(date +%s)
LAST=$(cat "$BEAT_FILE" 2>/dev/null)
echo "$NOW" > "$BEAT_FILE"
if [[ "$LAST" =~ ^[0-9]+$ ]] && (( NOW - LAST >= DOWN_AFTER_SEC )); then
    GAP_MIN=$(( (NOW - LAST) / 60 ))
    send_os_email "SERVER DOWN ($HOST) - back up after ${GAP_MIN} min" \
        "No heartbeat from $(date -d "@$LAST" '+%F %H:%M') to $(date -d "@$NOW" '+%F %H:%M') (${GAP_MIN} min): crash / reboot / power loss / cron stopped. The server is running again."
    echo "$(date '+%H:%M') deadman: heartbeat gap ${GAP_MIN} min -> outage email queued"
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 10 --retry 2 --retry-delay 5 --retry-connrefused \
    "${AUTH_HDR[@]}" \
    -H "In: 35m" -H "Priority: urgent" -H "Title: SERVER DOWN ($HOST)" \
    -d "No heartbeat for 35 minutes -- check the server (crash / reboot / power loss)." \
    "https://ntfy.sh/$TOPIC/server-alive" || echo "000")
if [[ "$HTTP_CODE" =~ ^2 ]]; then
    echo "$(date '+%H:%M') deadman: pushed heartbeat (+35m)"
elif [ "$HTTP_CODE" = "429" ]; then
    echo "$(date '+%H:%M') deadman: ntfy rate-limited (HTTP 429); relying on healthchecks.io"
else
    echo "$(date '+%H:%M') deadman: ntfy curl HTTP $HTTP_CODE after retries"
fi

HC_URL="$(env_get HC_PING_URL)"
if [ -n "$HC_URL" ]; then
    curl -fsS -m 10 --retry 3 --retry-delay 3 --retry-all-errors "$HC_URL" >/dev/null 2>&1 \
        && echo "$(date '+%H:%M') deadman: hc ping OK" \
        || echo "$(date '+%H:%M') deadman: hc ping FAILED (healthchecks.io will alarm if it persists)"
fi

BOOT_ID_SRC="${DEADMAN_BOOT_ID_PATH:-/proc/sys/kernel/random/boot_id}"
BOOT_FILE="$ROOT/logs/.deadman_last_boot_id"
BOOT_ID="$(cat "$BOOT_ID_SRC" 2>/dev/null || true)"
if [ -n "$BOOT_ID" ]; then
    PREV_BOOT_ID="$(cat "$BOOT_FILE" 2>/dev/null || true)"
    if [ -z "$PREV_BOOT_ID" ]; then
        echo "$BOOT_ID" > "$BOOT_FILE"
    elif [ "$BOOT_ID" != "$PREV_BOOT_ID" ]; then
        UP_SINCE="$(uptime -s 2>/dev/null || date '+%F %H:%M')"
        GAP_INFO=""
        if [[ "$LAST" =~ ^[0-9]+$ ]]; then
            GAP_INFO=" (last heartbeat $(date -d "@$LAST" '+%F %H:%M'), gap $(( (NOW - LAST) / 60 )) min)"
        fi
        REBOOT_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -m 10 --retry 2 --retry-delay 5 --retry-connrefused \
            "${AUTH_HDR[@]}" \
            -H "Priority: urgent" -H "Title: SERVER REBOOTED ($HOST)" \
            -d "Server booted at ${UP_SINCE}${GAP_INFO} -- check for crash / reboot / power loss." \
            "https://ntfy.sh/$TOPIC" || echo "000")
        if [[ "$REBOOT_HTTP" =~ ^2 ]]; then
            echo "$BOOT_ID" > "$BOOT_FILE"
            echo "$(date '+%H:%M') deadman: reboot detected (up since ${UP_SINCE}) -> push sent"
        else
            echo "$(date '+%H:%M') deadman: reboot detected (up since ${UP_SINCE}) -> ntfy HTTP $REBOOT_HTTP (will retry)"
        fi
    fi
fi
