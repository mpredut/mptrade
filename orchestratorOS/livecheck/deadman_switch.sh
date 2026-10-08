#!/bin/bash
# deadman_switch.sh — an ntfy alert if the Linux server dies (crash/reboot/power-off),
# not just when a bot/process dies (healthcheck.sh --supervise already covers that).
#
# How it works: on every run (cron every 15 min) we push a SCHEDULED ntfy message
# (In: 35m) further into the future, using the same sequence id in the URL
# (ntfy.sh/<topic>/server-alive). Each update is still a request counted
# against the ntfy quota; the old */2 cadence produced up to 720 requests/day and exceeded
# the free quota. 96/day leaves room for real alerts. Pattern is documented as
# "dead man's switch": https://docs.ntfy.sh/publish/#scheduled-delivery
#
# If the server dies (or just cron does), nobody pushes the queued ntfy message
# further out and it delivers itself 35 minutes later — the alert arrives even if
# the machine is completely off or without power.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/env_common.sh"

TOPIC=$(env_get NTFY_TOPIC_SERVER)
[ -n "$TOPIC" ] || TOPIC=$(env_get NTFY_TOPIC_DEADMAN)
[ -n "$TOPIC" ] || TOPIC=$(env_get NTFY_TOPIC_ERROR)
[ -n "$TOPIC" ] || TOPIC=$(env_get NTFY_TOPIC "ntfy-server-1978")

TOKEN="${NTFY_TOKEN:-$(env_get NTFY_TOKEN)}"

AUTH_HDR=()
[ -n "$TOKEN" ] && AUTH_HDR=(-H "Authorization: Bearer $TOKEN")

HOST=$(hostname)

# Email side of the switch. ntfy does NOT email scheduled ("In:") messages, and a dead
# server cannot send mail, so: while DOWN, healthchecks.io (HC_PING_URL, below) emails
# from outside; once back UP, the heartbeat gap is detected here and the outage window
# is emailed through the shared mailer (orchestratorOS/lib/os_notify.sh).
DOWN_AFTER_SEC=$((35 * 60))   # keep in sync with the "In: 35m" header below
BEAT_FILE="$ROOT/logs/.deadman_last_beat"
NOW=$(date +%s)
LAST=$(cat "$BEAT_FILE" 2>/dev/null)
echo "$NOW" > "$BEAT_FILE"
if [[ "$LAST" =~ ^[0-9]+$ ]] && (( NOW - LAST >= DOWN_AFTER_SEC )); then
    GAP_MIN=$(( (NOW - LAST) / 60 ))
    # shellcheck source=../lib/os_notify.sh
    source "$ROOT/orchestratorOS/lib/os_notify.sh"
    send_os_email "SERVER DOWN ($HOST) - back up after ${GAP_MIN} min" \
        "No heartbeat from $(date -d "@$LAST" '+%F %H:%M') to $(date -d "@$NOW" '+%F %H:%M') (${GAP_MIN} min): crash / reboot / power loss / cron stopped. The server is running again."
    echo "$(date '+%H:%M') deadman: heartbeat gap ${GAP_MIN} min -> outage email queued"
fi

# --retry 4 --retry-all-errors: retries on transient DNS/network blips
curl --fail-with-body -sS -m 10 --retry 4 --retry-delay 5 --retry-all-errors --retry-connrefused \
    "${AUTH_HDR[@]}" \
    -H "In: 35m" -H "Title: SERVER DOWN ($HOST)" \
    -d "No heartbeat for 35 minutes — check the server (crash / reboot / power loss)." \
    "https://ntfy.sh/$TOPIC/server-alive" >/dev/null \
    && echo "$(date '+%H:%M') deadman: pushed heartbeat (+35m)" \
    || echo "$(date '+%H:%M') deadman: curl ERROR after retries (prolonged DNS/net blip or quota?)"

# Second, INDEPENDENT dead-man's switch on healthchecks.io. It does NOT share ntfy's free
# quota, so it keeps working when ntfy is 429-throttled.
# Optional: create a check (period 15m, grace ~20m, e-mail/phone set THERE) and put its ping
# URL in .env as HC_PING_URL=... (secret, gitignored). Absent -> this block is a no-op.
HC_URL=$(env_get HC_PING_URL)
if [ -n "$HC_URL" ]; then
    curl -fsS -m 10 --retry 3 --retry-delay 3 --retry-all-errors "$HC_URL" >/dev/null 2>&1 \
        && echo "$(date '+%H:%M') deadman: hc ping OK" \
        || echo "$(date '+%H:%M') deadman: hc ping FAILED (healthchecks.io will alarm if it persists)"
fi
