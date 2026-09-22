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
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOPIC=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
[ -n "$TOPIC" ] || TOPIC=$(grep -hs -m1 '^NTFY_TOPIC=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
if [ -z "$TOPIC" ]; then
    echo "$(date '+%H:%M') deadman: no NTFY_TOPIC(_ERROR) found in $ROOT/.env or $ROOT/config.env"
    exit 1
fi

TOKEN="${NTFY_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    TOKEN=$(grep -hs -m1 '^NTFY_TOKEN=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
fi

AUTH_HDR=()
[ -n "$TOKEN" ] && AUTH_HDR=(-H "Authorization: Bearer $TOKEN")

HOST=$(hostname)
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
HC_URL=$(grep -hs -m1 '^HC_PING_URL=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d '" ')
if [ -n "$HC_URL" ]; then
    curl -fsS -m 10 --retry 3 --retry-delay 3 --retry-all-errors "$HC_URL" >/dev/null 2>&1 \
        && echo "$(date '+%H:%M') deadman: hc ping OK" \
        || echo "$(date '+%H:%M') deadman: hc ping FAILED (healthchecks.io will alarm if it persists)"
fi
