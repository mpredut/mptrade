#!/usr/bin/env bash
# os_notify.sh — shared ntfy notification helper for OS/System scripts.
# Routes notifications to NTFY_TOPIC_ERROR with deduplication, 5-min anti-burst gate, and email mirroring.

_OS_NOTIFY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_common.sh
source "$_OS_NOTIFY_DIR/env_common.sh"

ntfy_error_topic() {
  env_get NTFY_TOPIC_ERROR "$(env_get NTFY_TOPIC "ntfy-error-1978")"
}

ntfy_direct_curl() {  # $1=topic $2=priority $3=tags $4=title $5=body [$6=timeout_sec]
  local topic="${1:-$(ntfy_error_topic)}" prio="${2:-default}" tags="${3:-warning,computer}"
  local title="${4:-OS Alert}" body="${5:-}" timeout_s="${6:-8}"
  [ -n "$topic" ] || return 1
  local token auth_hdr=()
  token="$(env_get NTFY_TOKEN "")"
  [ -n "$token" ] && auth_hdr=(-H "Authorization: Bearer $token")
  curl -sS --max-time "$timeout_s" \
    "${auth_hdr[@]}" \
    -H "Title: $title" \
    -H "Priority: $prio" \
    -H "Tags: $tags" \
    -d "$body" \
    "https://ntfy.sh/$topic" >/dev/null 2>&1
}

send_os_ntfy() {  # $1=title $2=body [$3=priority] [$4=topic]
  local title="${1:-OS Alert}" body="${2:-}" prio="${3:-default}"
  local topic="${4:-$(ntfy_error_topic)}"
  [ -n "$topic" ] || return 0

  local state_file="${OS_NOTIFY_STATE_FILE:-/tmp/mptrade_os_notify_state.json}"
  local dedup_sec="${OS_NOTIFY_DEDUP_SECONDS:-300}"
  local gate_sec="${OS_NOTIFY_GATE_SECONDS:-300}"

  if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
    local gate_res
    gate_res=$(OS_NOTIFY_TITLE="$title" OS_NOTIFY_BODY="$body" OS_NOTIFY_TOPIC="$topic" \
               OS_NOTIFY_STATE="$state_file" OS_NOTIFY_DEDUP="$dedup_sec" OS_NOTIFY_GATE="$gate_sec" \
               "$PYTHON_BIN" - <<'PY' 2>/dev/null || echo "SEND"
import hashlib, json, os, re, time
state_path = os.environ["OS_NOTIFY_STATE"]
title, body, topic = os.environ["OS_NOTIFY_TITLE"], os.environ["OS_NOTIFY_BODY"], os.environ["OS_NOTIFY_TOPIC"]
dedup_s, gate_s, now = float(os.environ["OS_NOTIFY_DEDUP"]), float(os.environ["OS_NOTIFY_GATE"]), time.time()
norm = re.sub(r"\b\d+(\.\d+)?\b", "<N>", f"{topic}|{title}|{body}".lower())
sig = hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:20]
try:
    with open(state_path, "r", encoding="utf-8") as fh:
        st = json.load(fh)
    if not isinstance(st, dict):
        st = {}
except Exception:
    st = {}
sigs = {k: v for k, v in st.get("sigs", {}).items() if now - float(v) < max(dedup_s, 3600.0)}
last_sent = float(st.get("last_sent_ts", 0.0))
decision = "SEND"
if dedup_s > 0 and sig in sigs and (now - float(sigs[sig])) < dedup_s:
    decision = "SKIP_DUP"
elif gate_s > 0 and (now - last_sent) < gate_s:
    decision = "SKIP_GATE"
else:
    sigs[sig] = now
    st["last_sent_ts"] = now
st["sigs"] = sigs
try:
    tmp = f"{state_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, state_path)
except Exception:
    pass
print(decision)
PY
    )
    [ "$gate_res" = "SKIP_DUP" ] || [ "$gate_res" = "SKIP_GATE" ] && return 0
  fi

  local tags="warning,computer"
  [ "$prio" = "high" ] || [ "$prio" = "urgent" ] && tags="rotating_light,warning"
  ntfy_direct_curl "$topic" "$prio" "$tags" "$title" "$body" 8 || return 1

  if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
    PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
    OS_ERR_TITLE="$title" OS_ERR_BODY="$body" OS_ERR_PRIO="$prio" OS_ERR_TOPIC="$topic" \
    "$PYTHON_BIN" -c '
import os
from orchestratorOS.notify import email_mirror
email_mirror.send_error_email(
    title=os.environ.get("OS_ERR_TITLE", ""),
    body=os.environ.get("OS_ERR_BODY", ""),
    priority=os.environ.get("OS_ERR_PRIO", "default"),
    topic=os.environ.get("OS_ERR_TOPIC", ""),
    category="os_shell",
)
' >/dev/null 2>&1 &
  fi
}
