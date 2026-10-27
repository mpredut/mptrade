#!/bin/bash
# ntfy_check.sh — checks the ntfy topics for ALARM messages (monitoring from dev,
# without SSH to the server). Used manually or by the Claude session's monitoring job.
# Usage: ./ntfy_check.sh [since]   (default: 40m; ex. 12h)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/env_common.sh"
SINCE="${1:-40m}"

# Read from .env or config.env without exposing secrets in output.
PHONE_URL=$(env_get PHONE_ALERT_URL)
NT_TOPIC=$(env_get NTFY_TOPIC_ERROR "ntfy-error-1978")
TOKEN="${NTFY_TOKEN:-$(env_get NTFY_TOKEN)}"
AUTH_HDR=()
[ -n "$TOKEN" ] && AUTH_HDR=(-H "Authorization: Bearer $TOKEN")

check_url() {
    local url="$1" label="$2"
    [ -z "$url" ] && { echo "$label: (topic missing from .env/config.env)"; return; }
    curl -s -m 15 "${AUTH_HDR[@]}" "$url/json?poll=1&since=$SINCE" | "$PYTHON_BIN" -c '
import sys, json, datetime
label = sys.argv[1] if len(sys.argv) > 1 else ""
alarms, info = [], 0
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: m = json.loads(line)
    except Exception: continue
    if not isinstance(m, dict) or m.get("event") != "message": continue
    title = str(m.get("title") or ""); body = str(m.get("message") or "")[:100]
    try:
        raw_t = m.get("time")
        ts = datetime.datetime.fromtimestamp(float(raw_t) if raw_t is not None and not isinstance(raw_t, bool) else 0.0).strftime("%d %H:%M")
    except Exception:
        ts = "--:--"
    low = (title + " " + body).lower()
    if any(k in low for k in ("dead","hung","stopped","stale","error","fail","crash","absent")):
        alarms.append(f"{ts} [{title}] {body}")
    else:
        info += 1
print(f"{label}: informative={info} ALARMS={len(alarms)}")
for a in alarms[-8:]:
    print("  !! " + a)
' "$label"
}

check_url "$PHONE_URL" "crypto-alerts"
check_url "https://ntfy.sh/$NT_TOPIC" "flota/healthcheck"
