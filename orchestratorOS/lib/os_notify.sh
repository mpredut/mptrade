#!/usr/bin/env bash
# os_notify.sh — unified helper for ntfy.sh push notifications from bash.
# Topics that the mirror policy copies to email (ERROR, DEADMAN — see
# notify_engine/mailer.py) are also emailed, in the background, best-effort.
# Usage:
#   Direct:  ./os_notify.sh "Title" "Body" [priority] [topic]
#   Sourced: source os_notify.sh && send_os_ntfy "Title" "Body" [priority] [topic]
# Returns curl's exit status for the push (0 = delivered to ntfy).

set -u

if ! declare -f env_get >/dev/null 2>&1; then
    _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck source=env_common.sh
    source "$_script_dir/env_common.sh"
fi

ntfy_error_topic() {
    env_get NTFY_TOPIC_ERROR "$(env_get NTFY_TOPIC "ntfy-error-1978")"
}

ntfy_server_topic() {
    env_get NTFY_TOPIC_SERVER "$(env_get NTFY_TOPIC_DEADMAN "$(ntfy_error_topic)")"
}

ntfy_load_auth_hdr() {
    local token="${NTFY_TOKEN:-$(env_get NTFY_TOKEN)}"
    AUTH_HDR=()
    [ -n "$token" ] && AUTH_HDR=(-H "Authorization: Bearer $token") || true
}

# Email copy via the Python mailer (single SMTP choke point + mirror policy).
# Detached so a dead line / slow SMTP never blocks the caller. When run as root
# (vpn/autodeploy), drop to the repo owner so shared state files keep their owner.
send_os_email() {  # $1=title $2=body [$3=topic: email only if that topic is mirrored]
    local log_file="$ROOT/logs/mailer.log"
    local cmd=("$ROOT/orchestratorOS/run_python.sh" "$ROOT/notify_engine/mailer.py")
    [ -n "${3:-}" ] && cmd+=(--topic "$3")
    cmd+=("$1" "${2:-}")
    if [ "$(id -u)" = "0" ]; then
        local owner; owner="$(stat -c %U "$ROOT" 2>/dev/null || echo "")"
        if [ -n "$owner" ] && [ "$owner" != "root" ]; then
            touch "$log_file" 2>/dev/null && chown "$owner:$owner" "$log_file" 2>/dev/null || true
            chmod 0664 "$log_file" 2>/dev/null || true
            cmd=(runuser -u "$owner" -- "${cmd[@]}")
        fi
    fi
    ( cd "$ROOT" && "${cmd[@]}" >>"$log_file" 2>&1 ) </dev/null &
}

send_os_ntfy() {
    local title="${1:-OS Alert}" body="${2:-}" priority="${3:-default}" topic="${4:-}"
    [ -z "${title// /}" ] && title="OS Alert"
    if [ -z "$topic" ]; then
        case "$title" in
            *FAILED*|*ERROR*|*CRITICAL*|*DOWN*|*DEAD*) topic="$(ntfy_error_topic)" ;;
            autodeploy*|*Deadman*|*"Server Status"*) topic="$(ntfy_server_topic)" ;;
            *) topic="$(ntfy_error_topic)" ;;
        esac
    fi

    # Per-topic batching + central single-gate dispatcher + similarity elimination
    local stamp_file="" gate_file="" topic_gate_file="" bulk_file="" now_ts="" used_bulk=0
    local dedup_sec="${OS_NOTIFY_DEDUP_SECONDS:-900}"
    local gate_sec="${OS_NOTIFY_GATE_SECONDS:-$(env_get NOTIFICATION_GATE_SECONDS 180)}"
    local topic_bulk_sec="${OS_NOTIFY_TOPIC_BULK_SECONDS:-$(env_get NOTIFICATION_TOPIC_BULK_SECONDS 300)}"
    local topic_max_items="${OS_NOTIFY_TOPIC_MAX_ITEMS:-$(env_get NOTIFICATION_TOPIC_BULK_MAX_ITEMS 10)}"
    if [ "$dedup_sec" -gt 0 ] 2>/dev/null; then
        local dedup_dir="$ROOT/logs/.os_notify_dedup"
        mkdir -p "$dedup_dir" 2>/dev/null || true
        chmod 0777 "$dedup_dir" 2>/dev/null || true
        find "$dedup_dir" -maxdepth 1 -type f \( -name '[0-9a-f]*' -o -name '.pending_bulk_*' \) -mmin +60 -delete 2>/dev/null || true
        local norm_title norm_body key_hash safe_topic
        norm_title="$(printf '%s' "$title" | sed -E 's/[0-9]+([.][0-9]+)?/<N>/g')"
        norm_body="$(printf '%s' "$body" | head -n 1 | sed -E 's/\b[0-9a-f]{7,40}\b/<SHA>/g; s/[0-9]+([.][0-9]+)?/<N>/g')"
        key_hash="$(printf '%s|%s|%s' "$topic" "$norm_title" "$norm_body" | md5sum 2>/dev/null | awk '{print $1}')"
        safe_topic="$(printf '%s' "$topic" | tr -c 'A-Za-z0-9_-' '_')"
        if [ -n "$key_hash" ] && [ -d "$dedup_dir" ]; then
            stamp_file="$dedup_dir/$key_hash"
            gate_file="$dedup_dir/.global_gate_ts"
            topic_gate_file="$dedup_dir/.topic_gate_${safe_topic}"
            bulk_file="$dedup_dir/.pending_bulk_${safe_topic}"
            now_ts="$(date +%s)"
            if [ -f "$stamp_file" ]; then
                local last_ts; last_ts="$(cat "$stamp_file" 2>/dev/null || echo 0)"
                [ -n "$last_ts" ] && [ $((now_ts - last_ts)) -lt "$dedup_sec" ] 2>/dev/null && return 0
            fi
            local should_buffer=0
            if [ "$gate_sec" -gt 0 ] 2>/dev/null && [ -f "$gate_file" ]; then
                local gate_ts; gate_ts="$(cat "$gate_file" 2>/dev/null || echo 0)"
                if [ -n "$gate_ts" ] && [ $((now_ts - gate_ts)) -lt "$gate_sec" ] 2>/dev/null; then
                    should_buffer=1
                fi
            fi
            if [ "$topic_bulk_sec" -gt 0 ] 2>/dev/null && [ -f "$topic_gate_file" ]; then
                local t_gate_ts; t_gate_ts="$(cat "$topic_gate_file" 2>/dev/null || echo 0)"
                if [ -n "$t_gate_ts" ] && [ $((now_ts - t_gate_ts)) -lt "$topic_bulk_sec" ] 2>/dev/null; then
                    should_buffer=1
                fi
            fi
            case "$title" in
                *RECOVERED*|*repaired*|*RESTORED*)
                    should_buffer=0
                    local rec_prefix="${title% RECOVERED*}"
                    rec_prefix="${rec_prefix% RESTORED*}"
                    if [ -n "$rec_prefix" ] && [ "$rec_prefix" != "$title" ] && [ -f "$bulk_file" ]; then
                        if grep -vF "${rec_prefix} FAILED:" "$bulk_file" > "${bulk_file}.tmp" 2>/dev/null; then
                            mv -f "${bulk_file}.tmp" "$bulk_file"
                        else
                            rm -f "$bulk_file" "${bulk_file}.tmp"
                        fi
                    fi
                    ;;
            esac
            if [ "$should_buffer" -eq 1 ]; then
                printf '%s\n' "$now_ts" > "$stamp_file" 2>/dev/null || true
                chmod 0666 "$stamp_file" 2>/dev/null || true
                local cur_lines=0
                if [ -f "$bulk_file" ]; then
                    cur_lines="$(wc -l < "$bulk_file" 2>/dev/null || echo 0)"
                fi
                if [ "${cur_lines:-0}" -lt "${topic_max_items:-10}" ] 2>/dev/null; then
                    printf '%s: %s\n' "$title" "$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-300)" >> "$bulk_file" 2>/dev/null || true
                    chmod 0666 "$bulk_file" 2>/dev/null || true
                fi
                return 0
            fi
            if [ -s "$bulk_file" ]; then
                local bulk_count=$(( $(wc -l < "$bulk_file" 2>/dev/null || echo 0) + 1 ))
                local topic_label="ERROR"
                case "${topic,,}" in
                    *deadman*) topic_label="DEADMAN" ;;
                    *error*)   topic_label="ERROR" ;;
                    *guard*)   topic_label="GUARD" ;;
                    *trade*)   topic_label="TRADES" ;;
                    *macro*)   topic_label="MACRO" ;;
                    *price*)   topic_label="PRICE" ;;
                    *server*)  topic_label="SERVER" ;;
                esac
                local common_summary
                common_summary="$(
                    { awk -F': ' '{print $1}' "$bulk_file" 2>/dev/null; printf '%s\n' "$title"; } | awk -v fallback="$topic_label" '
                    {
                        lines[++n] = $0
                        delete seen_line
                        for (i = 1; i <= NF; i++) {
                            tok = $i
                            gsub(/^:+|:+$/, "", tok)
                            tl = tolower(tok)
                            if (length(tok) > 1 && tl !~ /^(in|on|to|by|for|the|of|and|or|at|is|are|with|from|a|an)$/ && !(tl in seen_line)) {
                                seen_line[tl] = 1
                                counts[tl]++
                                if (n == 1) {
                                    order[++m] = tl
                                    display[tl] = tok
                                }
                            }
                        }
                    }
                    END {
                        out = ""
                        for (j = 1; j <= m; j++) {
                            tl = order[j]
                            if (counts[tl] == n) {
                                out = (out == "" ? display[tl] : out " " display[tl])
                            }
                        }
                        print (out != "" ? out : (fallback != "" ? fallback : "Alerts"))
                    }'
                )"
                [ -z "$common_summary" ] && common_summary="$topic_label"
                local flat_body
                flat_body="$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-300)"
                body="$(
                    { cat "$bulk_file" 2>/dev/null; printf '%s: %s\n' "$title" "$flat_body"; } \
                    | awk '{ if (NR > 1) printf "\n\n"; printf "%d. %s", NR, $0 } END { printf "\n" }'
                )"
                title="📦 Bulk ($bulk_count): $common_summary"
                used_bulk=1
            fi
        fi
    fi

    local AUTH_HDR=()
    ntfy_load_auth_hdr
    send_os_email "$title" "$body" "$topic"

    if curl --fail -sS -m 10 --retry 2 --retry-delay 3 --retry-all-errors -X POST "https://ntfy.sh/$topic" \
        -H "Title: $title" -H "Priority: $priority" "${AUTH_HDR[@]}" -d "$body" >/dev/null 2>&1; then
        if [ -n "$stamp_file" ]; then
            printf '%s\n' "$now_ts" > "$stamp_file" 2>/dev/null || true
            printf '%s\n' "$now_ts" > "$gate_file" 2>/dev/null || true
            [ -n "$topic_gate_file" ] && printf '%s\n' "$now_ts" > "$topic_gate_file" 2>/dev/null || true
            chmod 0666 "$stamp_file" "$gate_file" ${topic_gate_file:+"$topic_gate_file"} 2>/dev/null || true
            [ "$used_bulk" -eq 1 ] && rm -f "$bulk_file" 2>/dev/null || true
            if [ "$(id -u)" = "0" ]; then
                rm -rf "$ROOT/logs/.os_notify_dedup.root_old" 2>/dev/null || true
                local owner; owner="$(stat -c %U "$ROOT" 2>/dev/null || echo "")"
                [ -n "$owner" ] && [ "$owner" != "root" ] && chown -R "$owner:$owner" "$ROOT/logs/.os_notify_dedup" 2>/dev/null || true
            fi
        fi
        return 0
    fi
    return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    send_os_ntfy "${1:-OS Alert}" "${2:-}" "${3:-default}" "${4:-}"
fi
