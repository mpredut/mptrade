#!/bin/bash
# deadman_switch.sh — an ntfy alert if the Linux server dies (crash/reboot/power-off),
# not just when a bot/process dies (healthcheck.sh --supervise already covers that).
#
# How it works: on every run (cron every 15 min) we push a SCHEDULED ntfy message
# (In: 35m) further into the future, using the same sequence id in the URL
# (ntfy.sh/<topic>/server-alive). Each update is still a request counted
# against the ntfy quota; the old */2 cadence produced up to 720 requests/day and exceeded it on its own
# limita gratuita. 96/zi lasa loc alertelor reale. Pattern-ul este documentat ca
# "dead man's switch": https://docs.ntfy.sh/publish/#scheduled-delivery
#
# If the server dies (or just cron does), nobody pushes the queued ntfy message
# further out and it delivers itself 35 minutes later — the alert arrives even if
# the machine is completely off or without power.
ROOT="$(cd "$(dirname "$0")" && pwd)"
TOPIC=$(grep -hs '^NTFY_TOPIC_ERROR=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '" ')
[ -z "$TOPIC" ] && TOPIC=$(grep -hs '^NTFY_TOPIC=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '" ')
if [ -z "$TOPIC" ]; then
    echo "$(date '+%H:%M') deadman: no NTFY_TOPIC(_ERROR) found in $ROOT/.env"
    exit 1
fi

HOST=$(hostname)
# --retry 4 --retry-all-errors (8 Aug): it retries the push on transient DNS/network blips too
# (NameResolutionError), not only on 5xx. A typical blip (~30-40s) is ridden out in one run
# -> avoids a false alert when the server is alive but DNS resolution dropped briefly.
# Worst-case ~4x(10s+5s)=60s, mult sub cadenta de 15 min.
curl --fail-with-body -sS -m 10 --retry 4 --retry-delay 5 --retry-all-errors --retry-connrefused \
    -H "In: 35m" -H "Title: SERVER DOWN ($HOST)" \
    -d "No heartbeat for 35 minutes — check the server (crash / reboot / power loss)." \
    "https://ntfy.sh/$TOPIC/server-alive" >/dev/null \
    && echo "$(date '+%H:%M') deadman: impins (+35m)" \
    || echo "$(date '+%H:%M') deadman: curl ERROR after the retries (a prolonged DNS/net blip?)"

# Second, INDEPENDENT dead-man's switch on healthchecks.io. It does NOT share ntfy's free
# quota, so it keeps working when ntfy is 429-throttled -- exactly the gap that swallowed the
# alerts on 18-19 Sep (deadman + Notifier + selfheal + healthcheck all funnel through one
# ntfy topic and hit the limit during the incident). The ALARM fires on healthchecks.io's
# side when pings STOP, so a failed ping here (server down / no net) is what triggers it.
# Optional: create a check (period 15m, grace ~20m, e-mail/phone set THERE) and put its ping
# URL in .env as HC_PING_URL=... (secret, gitignored). Absent -> this block is a no-op.
HC_URL=$(grep -hs '^HC_PING_URL=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '" ')
if [ -n "$HC_URL" ]; then
    curl -fsS -m 10 --retry 3 --retry-delay 3 --retry-all-errors "$HC_URL" >/dev/null 2>&1 \
        && echo "$(date '+%H:%M') deadman: hc ping OK" \
        || echo "$(date '+%H:%M') deadman: hc ping FAILED (healthchecks.io will alarm if it persists)"
fi
