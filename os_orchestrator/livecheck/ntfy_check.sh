#!/bin/bash
# ntfy_check.sh — checks the ntfy topics for ALARM messages (monitoring from dev,
# without SSH to the server). Used manually or by the Claude session's monitoring job.
# Usage: ./ntfy_check.sh [since]   (default: 40m; ex. 12h)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../env_common.sh"
SINCE="${1:-40m}"

# Read from config.env without exposing the secrets in the output.
PHONE_URL=$(grep -E '^\s*(export\s+)?PHONE_ALERT_URL=' "$ROOT/config.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '" ')
NT_TOPIC=$(grep -E '^\s*(export\s+)?NTFY_TOPIC_ERROR=' "$ROOT/config.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '" ')

check_url() {
    local url="$1" label="$2"
    [ -z "$url" ] && { echo "$label: (topic missing from config.env)"; return; }
    curl -s -m 15 "$url/json?poll=1&since=$SINCE" | "$PYTHON_BIN" -c "
import sys, json, datetime
alarms, info = [], 0
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: m = json.loads(line)
    except Exception: continue
    if m.get('event') != 'message': continue
    title = (m.get('title') or ''); body = (m.get('message') or '')[:100]
    ts = datetime.datetime.fromtimestamp(m.get('time', 0)).strftime('%d %H:%M')
    low = (title + ' ' + body).lower()
    if any(k in low for k in ('dead','hung','stopped','stale','error','fail','crash','absent')):
        alarms.append(f'{ts} [{title}] {body}')
    else:
        info += 1
print(f'$label: informative={info} ALARMS={len(alarms)}')
for a in alarms[-8:]:
    print('  !! ' + a)
"
}

check_url "$PHONE_URL" "crypto-alerts"
check_url "https://ntfy.sh/$NT_TOPIC" "flota/healthcheck"
