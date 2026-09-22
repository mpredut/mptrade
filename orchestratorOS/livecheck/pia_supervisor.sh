#!/bin/bash
# pia_supervisor.sh — Private Internet Access Dedicated IP & WireGuard supervisor daemon.
#
# Usage:
#   systemd service daemon: runs continuously supervising the tunnel and probing health.
#   Manual diagnostics:    ./pia_supervisor.sh --check  (or -c / --status)

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/config.env" ] && {
    set -a
    . "$ROOT/config.env" 2>/dev/null || true
    set +a
}

HEALTH_INTERVAL="${PIA_HEALTH_INTERVAL:-30}"
FAILURE_LIMIT="${PIA_FAILURE_LIMIT:-3}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-7}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
VPN_IF="${PIA_VPN_IF:-wgpia0}"
PREWARM_INTERVAL="${PIA_PREWARM_INTERVAL:-300}"
BACKOFF_MIN_SEC="${PIA_BACKOFF_MIN_SEC:-${PIA_FALLBACK_COOLDOWN_SEC:-60}}"
BACKOFF_MAX_SEC="${PIA_BACKOFF_MAX_SEC:-7200}"

failures=0
last_prewarm=0
current_cooldown_sec="$BACKOFF_MIN_SEC"

REG_DED_DE=""
REG_DED_BE=""
REG_DYN_DE="de-frankfurt"

send_ntfy() {
    local title="$1"
    local body="$2"
    local priority="${3:-default}"
    local topic; topic=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
    [ -n "$topic" ] || topic="ntfy-error-941582"
    local token; token=$(grep -hs -m1 '^NTFY_TOKEN=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
    local auth_hdr=()
    [ -n "$token" ] && auth_hdr=(-H "Authorization: Bearer $token")
    curl -s -m 10 -X POST "https://ntfy.sh/$topic" \
        -H "Title: $title" \
        -H "Priority: $priority" \
        "${auth_hdr[@]}" \
        -d "$body" >/dev/null 2>&1 &
}

isp_healthy() {
    # Test if direct physical uplink has working internet (without VPN)
    curl -4 -s -m 5 -o /dev/null https://api.binance.com/api/v3/time 2>/dev/null
}

pia() {
    timeout "$CLI_TIMEOUT" piactl "$@"
}

vpn_healthy() {
    [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1

    # 1. Resolve Binance IPv4 address (getent/systemd-resolved)
    local binance_ip
    binance_ip=$(getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}')
    [ -n "$binance_ip" ] || binance_ip=$(resolvectl query api.binance.com 2>/dev/null | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    [ -n "$binance_ip" ] || return 1

    # 2. Pass resolved IP directly to curl via --resolve to avoid DNS lookup races
    # and guarantee traffic routes through the VPN interface.
    curl -4 --interface "$VPN_IF" \
        --resolve "api.binance.com:443:$binance_ip" \
        --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || return 1
}

prewarm_dns() {
    # Refresh local systemd-resolved cache for all critical fleet venue endpoints.
    # Runs at most once every PREWARM_INTERVAL seconds (default: 300s / 5 min).
    local now; now=$(date +%s)
    if [ $((now - last_prewarm)) -ge "$PREWARM_INTERVAL" ]; then
        last_prewarm=$now
        local domains=(
            "api.binance.com"
            "stream.binance.com"
            "api.kraken.com"
            "api.hyperliquid.xyz"
            "live.trading212.com"
            "ntfy.sh"
        )
        for d in "${domains[@]}"; do
            getent ahostsv4 "$d" >/dev/null 2>&1 &
        done
    fi
}

discover_regions() {
    REG_DED_DE=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-de-frankfurt" || true)
    REG_DED_BE=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-belgium" || true)
    REG_DYN_DE="de-frankfurt"
}

connect_to_region() {
    local target_region="$1"
    local desc="$2"
    echo "Attempting connection to $desc ($target_region)..."
    pia set region "$target_region" >/dev/null 2>&1 || return 1
    pia set requestportforward true >/dev/null 2>&1 || true
    pia connect >/dev/null 2>&1 || return 1

    local conn_ok=0
    for attempt in $(seq 1 12); do
        sleep 5
        local state vpnip
        state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
        vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
        if [ "$state" = "Connected" ] && echo "$vpnip" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
            conn_ok=1
            break
        fi
        echo "Waiting for IP assignment on $target_region ($attempt/12)... state=${state:-unknown} vpnip=${vpnip:-unknown}"
        if [ "$state" = "Connected" ] && [ "$vpnip" = "Unknown" ] && [ "$attempt" -ge 3 ]; then
            echo "Region $target_region returned vpnip=Unknown after $attempt attempts."
            break
        fi
    done

    if [ "$conn_ok" -eq 1 ] && vpn_healthy; then
        echo "Successfully connected to $target_region (IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
        return 0
    fi

    echo "Connection to $target_region failed or health probe failed."
    pia disconnect >/dev/null 2>&1 || true
    return 1
}

establish_vpn_tunnel() {
    discover_regions

    # 1. Primary: Frankfurt Dedicated IP
    if [ -n "$REG_DED_DE" ]; then
        if connect_to_region "$REG_DED_DE" "Frankfurt Dedicated IP"; then
            return 0
        fi
        echo "Frankfurt Dedicated IP ($REG_DED_DE) failed."
    fi

    # 2. Secondary: Belgium Dedicated IP
    if [ -n "$REG_DED_BE" ]; then
        echo "Falling back to Belgium Dedicated IP ($REG_DED_BE)..."
        send_ntfy "PIA VPN: Frankfurt DIP Failed" "Frankfurt Dedicated IP failed. Falling back to Belgium Dedicated IP ($REG_DED_BE)..." "high"
        if connect_to_region "$REG_DED_BE" "Belgium Dedicated IP"; then
            send_ntfy "PIA VPN: Connected on Belgium DIP" "Successfully connected to Belgium Dedicated IP ($REG_DED_BE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
            return 0
        fi
        echo "Belgium Dedicated IP ($REG_DED_BE) failed."
    fi

    # 3. Tertiary: Dynamic Frankfurt
    echo "Falling back to Dynamic Frankfurt ($REG_DYN_DE)..."
    send_ntfy "PIA VPN: Dedicated IPs Failed" "Dedicated IP failed. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." "high"
    if connect_to_region "$REG_DYN_DE" "Dynamic Frankfurt"; then
        send_ntfy "PIA VPN: Connected on Dynamic" "Successfully connected to Dynamic Frankfurt ($REG_DYN_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
        return 0
    fi

    echo "All PIA regions failed (Frankfurt DIP, Belgium DIP, Dynamic Frankfurt)."
    return 1
}

enter_isp_cooldown() {
    local reason="$1"
    echo "CRITICAL: PIA VPN unavailable ($reason). Disconnecting tunnel to release direct Romanian ISP traffic..."
    pia disconnect >/dev/null 2>&1 || true
    sleep 2

    if isp_healthy; then
        echo "Direct Romanian ISP uplink is healthy. Fleet operating on direct IP."
        send_ntfy "PIA VPN Down -> Direct ISP Fallback (${current_cooldown_sec}s)" \
            "PIA VPN failed ($reason). Tunnel disconnected. Fleet operating directly via Romanian ISP for a ${current_cooldown_sec}s cooldown." \
            "urgent"
    else
        echo "WARNING: Direct ISP uplink also unreachable or experiencing network issues."
        send_ntfy "Network Outage: PIA & ISP Down" \
            "Both PIA and direct ISP uplink probe failed ($reason). Cooldown: ${current_cooldown_sec}s." \
            "urgent"
    fi

    local elapsed=0
    echo "Entering Romanian ISP cooldown mode for ${current_cooldown_sec}s..."
    while [ "$elapsed" -lt "$current_cooldown_sec" ]; do
        sleep 30
        elapsed=$((elapsed + 30))
        prewarm_dns
    done

    echo "Cooldown period (${current_cooldown_sec}s) elapsed. Attempting recovery reconnect..."
    send_ntfy "PIA VPN: Attempting Reconnect" \
        "Cooldown (${current_cooldown_sec}s) elapsed. Attempting PIA recovery across fallback ladder..." \
        "default"
}

# --- Diagnostic / Manual Tool Mode ---
if [ "${1:-}" = "--check" ] || [ "${1:-}" = "-c" ] || [ "${1:-}" = "--status" ]; then
    echo "=== PIA VPN Manual Status Check ==="
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    region=$(pia get region 2>/dev/null | tr -d '\r')
    port=$(pia get portforward 2>/dev/null | tr -d '\r')
    echo "Connection State: ${state:-Unknown}"
    echo "VPN IP:           ${vpnip:-Unknown}"
    echo "Region:           ${region:-Unknown}"
    echo "Port Forward:     ${port:-Unknown}"
    echo "Interface:        $VPN_IF"
    if vpn_healthy; then
        echo "Health Probe:     OK (Binance ping via $VPN_IF succeeded)"
        exit 0
    else
        echo "Health Probe:     FAILED (Interface down or Binance probe unreachable)"
        exit 1
    fi
fi

# --- Daemon Configuration & Initialization ---
# Require .env file and enforce required PIA variables
[ -f "$ROOT/.env" ] || { echo "error: missing required $ROOT/.env configuration" >&2; exit 1; }

PIA_USER=""
PIA_PASS=""
PIA_DIP_TOKEN_FRANKFURT=""
PIA_DIP_TOKEN_BELGIUM=""
while IFS='=' read -r key val; do
    val="${val%\"}"
    val="${val#\"}"
    val="${val%\'}"
    val="${val#\'}"
    case "$key" in
        PIA_USER|PIA_ACCOUNT_USER) PIA_USER="$val" ;;
        PIA_PASS) PIA_PASS="$val" ;;
        PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT) PIA_DIP_TOKEN_FRANKFURT="$val" ;;
        PIA_DIP_TOKEN_BELGIUM) PIA_DIP_TOKEN_BELGIUM="$val" ;;
    esac
done < <(grep -E '^(PIA_USER|PIA_ACCOUNT_USER|PIA_PASS|PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT|PIA_DIP_TOKEN_BELGIUM)=' "$ROOT/.env" 2>/dev/null || true)

[ -n "$PIA_USER" ] || { echo "error: PIA_USER missing in $ROOT/.env" >&2; exit 1; }
[ -n "$PIA_PASS" ] || { echo "error: PIA_PASS missing in $ROOT/.env" >&2; exit 1; }
[ -n "$PIA_DIP_TOKEN_FRANKFURT" ] || { echo "error: PIA_DIP_TOKEN_FRANKFURT missing in $ROOT/.env" >&2; exit 1; }

# Clamp physical uplink MTU before touching PIA.
UPLINK_IF="${PIA_UPLINK_IF:-ens18}"
UPLINK_MTU="${PIA_UPLINK_MTU:-1280}"
ip link set dev "$UPLINK_IF" mtu "$UPLINK_MTU" 2>/dev/null \
    || echo "warning: could not set $UPLINK_IF MTU to $UPLINK_MTU (continuing)"

# Prefer IPv4 system-wide.
if ! grep -qs '^precedence ::ffff:0:0/96' /etc/gai.conf; then
    echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
fi

sleep 5

# Enable background mode for headless server operation.
pia background enable || exit 1
# Allow LAN traffic through kill switch so SSH management is preserved.
pia set allowlan true || true

# WireGuard protocol
pia set protocol wireguard || exit 1

# Ensure PIA login if not already authenticated
if ! pia get connectionstate 2>/dev/null | grep -Eq '^(Connected|Connecting|Disconnected|Reconnecting)$'; then
    echo "Attempting PIA login using .env credentials..."
    cred_tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
    chmod 0600 "$cred_tmp"
    printf "%s\n%s\n" "$PIA_USER" "$PIA_PASS" > "$cred_tmp"
    pia login "$cred_tmp" >/dev/null 2>&1 || true
    rm -f "$cred_tmp"
fi

# Register Dedicated IP from tokens if not present in daemon regions
if [ -n "$PIA_DIP_TOKEN_FRANKFURT" ] && ! pia get regions 2>/dev/null | grep -q "^dedicated-de-frankfurt"; then
    echo "Registering Frankfurt Dedicated IP from .env..."
    tok_tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
    chmod 0600 "$tok_tmp"
    printf "%s\n" "$PIA_DIP_TOKEN_FRANKFURT" > "$tok_tmp"
    if pia dedicatedip add "$tok_tmp" >/dev/null 2>&1; then
        echo "Successfully added Frankfurt Dedicated IP token from .env"
    else
        echo "warning: Failed to add Frankfurt Dedicated IP token from .env" >&2
    fi
    rm -f "$tok_tmp"
fi

if [ -n "$PIA_DIP_TOKEN_BELGIUM" ] && ! pia get regions 2>/dev/null | grep -q "^dedicated-belgium"; then
    echo "Registering Belgium Dedicated IP from .env..."
    tok_tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
    chmod 0600 "$tok_tmp"
    printf "%s\n" "$PIA_DIP_TOKEN_BELGIUM" > "$tok_tmp"
    if pia dedicatedip add "$tok_tmp" >/dev/null 2>&1; then
        echo "Successfully added Belgium Dedicated IP token from .env"
    else
        echo "warning: Failed to add Belgium Dedicated IP token from .env" >&2
    fi
    rm -f "$tok_tmp"
fi

# Establish tunnel on startup across the fallback ladder
while ! establish_vpn_tunnel; do
    echo "Initial VPN connection failed across all regions. Entering Romanian ISP cooldown..."
    enter_isp_cooldown "Initial startup connection failed across all regions"
    current_cooldown_sec=$((current_cooldown_sec * 2))
    if [ "$current_cooldown_sec" -gt "$BACKOFF_MAX_SEC" ]; then
        current_cooldown_sec="$BACKOFF_MAX_SEC"
    fi
done

current_cooldown_sec="$BACKOFF_MIN_SEC"
failures=0

echo "VPN connected with region $(pia get region 2>/dev/null | tr -d '\r'):"
pia get vpnip

sleep 2
PORT=$(pia get portforward 2>/dev/null | tr -d '\r')
echo "Port Forward: $PORT"

# Prime DNS cache on startup
prewarm_dns

# --- Main Supervisor Health Loop ---
while true; do
    sleep "$HEALTH_INTERVAL"
    prewarm_dns

    if vpn_healthy; then
        if [ "$failures" -gt 0 ]; then
            echo "PIA recovered after $failures failed probes"
            send_ntfy "PIA VPN: Recovered" "PIA VPN is healthy again (Region: $(pia get region 2>/dev/null | tr -d '\r'), IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
        fi
        failures=0
        current_cooldown_sec="$BACKOFF_MIN_SEC"
        echo "PIA healthy (tun0 + DNS + HTTPS)"
        continue
    fi

    failures=$((failures + 1))
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    echo "PIA unhealthy: probe $failures/$FAILURE_LIMIT (state=${state:-unknown})"

    if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
        current_reg=$(pia get region 2>/dev/null | tr -d '\r')
        discover_regions

        recovered=0
        # If currently on Frankfurt Dedicated IP:
        if [ -n "$REG_DED_DE" ] && [ "$current_reg" = "$REG_DED_DE" ]; then
            # 1. Fallback to Belgium Dedicated IP
            if [ -n "$REG_DED_BE" ]; then
                echo "Frankfurt Dedicated IP failed $failures consecutive probes. Attempting fallback to Belgium Dedicated IP ($REG_DED_BE)..."
                send_ntfy "PIA VPN: Frankfurt DIP Failing" "Dedicated Frankfurt IP failed $failures probes. Falling back to Belgium Dedicated IP ($REG_DED_BE)..." "high"
                if connect_to_region "$REG_DED_BE" "Belgium Dedicated IP"; then
                    send_ntfy "PIA VPN: Switched to Belgium DIP" "Successfully switched to Belgium Dedicated IP ($REG_DED_BE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
                    failures=0
                    current_cooldown_sec="$BACKOFF_MIN_SEC"
                    recovered=1
                fi
            fi
            # 2. Fallback to Dynamic Frankfurt if Belgium failed or not available
            if [ "$recovered" -eq 0 ]; then
                echo "Dedicated IPs failed $failures consecutive probes. Attempting fallback to Dynamic Frankfurt ($REG_DYN_DE)..."
                send_ntfy "PIA VPN: Dedicated IP Failing" "Dedicated IP failed $failures probes. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." "high"
                if connect_to_region "$REG_DYN_DE" "Dynamic Frankfurt"; then
                    send_ntfy "PIA VPN: Switched to Dynamic Frankfurt" "Successfully switched to Dynamic Frankfurt ($REG_DYN_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
                    failures=0
                    current_cooldown_sec="$BACKOFF_MIN_SEC"
                    recovered=1
                fi
            fi
        # If currently on Belgium Dedicated IP:
        elif [ -n "$REG_DED_BE" ] && [ "$current_reg" = "$REG_DED_BE" ]; then
            # Fallback to Dynamic Frankfurt
            echo "Belgium Dedicated IP failed $failures consecutive probes. Attempting fallback to Dynamic Frankfurt ($REG_DYN_DE)..."
            send_ntfy "PIA VPN: Belgium DIP Failing" "Belgium Dedicated IP failed $failures probes. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." "high"
            if connect_to_region "$REG_DYN_DE" "Dynamic Frankfurt"; then
                send_ntfy "PIA VPN: Switched to Dynamic Frankfurt" "Successfully switched to Dynamic Frankfurt ($REG_DYN_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
                failures=0
                current_cooldown_sec="$BACKOFF_MIN_SEC"
                recovered=1
            fi
        fi

        # If recovered to another region, continue supervising
        if [ "$recovered" -eq 1 ]; then
            continue
        fi

        # All VPN regions failed (or dynamic Frankfurt failed):
        # Enter exponential backoff cooldown on direct Romanian ISP
        enter_isp_cooldown "PIA failed $failures probes on region ${current_reg:-unknown}"

        # Cooldown elapsed, attempt single clean recovery reconnect across ladder
        if establish_vpn_tunnel; then
            echo "PIA successfully reconnected after cooldown! (IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
            send_ntfy "PIA VPN: Reconnected After Cooldown" "PIA reconnected successfully (Region: $(pia get region 2>/dev/null | tr -d '\r'), IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
            current_cooldown_sec="$BACKOFF_MIN_SEC"
            failures=0
        else
            echo "PIA reconnection attempt failed after cooldown. Increasing exponential backoff..."
            current_cooldown_sec=$((current_cooldown_sec * 2))
            if [ "$current_cooldown_sec" -gt "$BACKOFF_MAX_SEC" ]; then
                current_cooldown_sec="$BACKOFF_MAX_SEC"
            fi
            echo "Next cooldown will be ${current_cooldown_sec}s (max: ${BACKOFF_MAX_SEC}s)."
            failures="$FAILURE_LIMIT"
        fi
    fi
done
