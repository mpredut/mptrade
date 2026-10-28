#!/bin/bash
# deadman_switch.sh — "dead man's switch" via ntfy scheduled delivery ("In: 35m") + healthchecks.io.
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

curl --fail-with-body -sS -m 10 --retry 4 --retry-delay 5 --retry-all-errors --retry-connrefused \
    "${AUTH_HDR[@]}" \
    -H "In: 35m" -H "Title: SERVER DOWN ($HOST)" \
    -d "No heartbeat for 35 minutes — check the server (crash / reboot / power loss)." \
    "https://ntfy.sh/$TOPIC/server-alive" >/dev/null \
    && echo "$(date '+%H:%M') deadman: pushed heartbeat (+35m)" \
    || echo "$(date '+%H:%M') deadman: curl ERROR after retries (prolonged DNS/net blip or quota?)"

HC_URL="$(env_get HC_PING_URL)"
if [ -n "$HC_URL" ]; then
    curl -fsS -m 10 --retry 3 --retry-delay 3 --retry-all-errors "$HC_URL" >/dev/null 2>&1 \
        && echo "$(date '+%H:%M') deadman: hc ping OK" \
        || echo "$(date '+%H:%M') deadman: hc ping FAILED (healthchecks.io will alarm if it persists)"
fi
