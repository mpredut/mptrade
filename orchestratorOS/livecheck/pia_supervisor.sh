#!/bin/bash
# pia_supervisor.sh — Private Internet Access Dedicated IP & WireGuard supervisor daemon.
#
# Usage:
#   systemd service daemon: runs continuously supervising the tunnel and probing health.
#   Manual diagnostics:    ./pia_supervisor.sh --check  (or -c / --status)

set -u

HEALTH_INTERVAL="${PIA_HEALTH_INTERVAL:-30}"
FAILURE_LIMIT="${PIA_FAILURE_LIMIT:-3}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-7}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
VPN_IF="${PIA_VPN_IF:-wgpia0}"
failures=0

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
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

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

# Note on kill switch: user accepts risk of real-IP leak in exchange for persistent
# server reachability if the VPN drops. To disable: pia set killswitch off || true

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

# Register Dedicated IP from tokens if not present
if ! pia get regions 2>/dev/null | grep -q "^dedicated-"; then
    echo "Registering Frankfurt Dedicated IP from .env..."
    tok_tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
    chmod 0600 "$tok_tmp"
    printf "%s\n" "$PIA_DIP_TOKEN_FRANKFURT" > "$tok_tmp"
    token_added=0
    if pia dedicatedip add "$tok_tmp" >/dev/null 2>&1; then
        echo "Successfully added Frankfurt Dedicated IP token from .env"
        token_added=1
    fi
    rm -f "$tok_tmp"

    if [ "$token_added" -eq 0 ] && [ -n "$PIA_DIP_TOKEN_BELGIUM" ]; then
        echo "Frankfurt token failed; trying Belgium token from .env..."
        tok_tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
        chmod 0600 "$tok_tmp"
        printf "%s\n" "$PIA_DIP_TOKEN_BELGIUM" > "$tok_tmp"
        if pia dedicatedip add "$tok_tmp" >/dev/null 2>&1; then
            echo "Successfully added Belgium Dedicated IP token from .env"
            token_added=1
        fi
        rm -f "$tok_tmp"
    fi

    if [ "$token_added" -eq 0 ]; then
        echo "error: Failed to register Dedicated IP tokens from .env" >&2
        exit 1
    fi
fi

DEDICATED=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-")
if [ -z "$DEDICATED" ]; then
    echo "No dedicated IP registered or token failed. Falling back to dynamic de-frankfurt."
    DEDICATED="de-frankfurt"
fi

pia set region "$DEDICATED" || exit 1
pia set requestportforward true || exit 1

if ! pia connect; then
    echo "Connection to $DEDICATED failed! Falling back to dynamic de-frankfurt."
    pia set region "de-frankfurt" || exit 1
    pia connect || exit 1
fi

echo "Waiting for the IP assignment..."
sleep 2
connected=0
for attempt in $(seq 1 12); do
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    if [ "$state" = "Connected" ] && echo "$vpnip" | grep -q '[0-9]'; then
        connected=1
        break
    fi
    if [ "$state" = "Connected" ] && [ "$vpnip" = "Unknown" ] && [ "$attempt" -ge 3 ]; then
        echo "PIA Connected but vpnip=Unknown after $attempt attempts; forcing reconnect."
        pia disconnect >/dev/null 2>&1 || true
        sleep 3
        pia connect >/dev/null 2>&1 || true
        sleep 5
    fi
    sleep 5
    echo "Still waiting for an IP ($attempt/12)... state=$state vpnip=$vpnip"
done
if [ "$connected" -ne 1 ]; then
    echo "PIA did not receive an IP within 60s; systemd will retry."
    exit 1
fi

echo "VPN connected with a dedicated IP:"
pia get vpnip

sleep 2
PORT=$(pia get portforward)
echo "Port Forward: $PORT"

# --- Main Supervisor Health Loop ---
while true; do
    sleep "$HEALTH_INTERVAL"
    if vpn_healthy; then
        if [ "$failures" -gt 0 ]; then
            echo "PIA recovered after $failures failed probes"
        fi
        failures=0
        echo "PIA healthy (tun0 + DNS + HTTPS)"
        continue
    fi

    failures=$((failures + 1))
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    echo "PIA unhealthy: probe $failures/$FAILURE_LIMIT (state=${state:-unknown})"
    if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
        echo "PIA/DNS persistently unavailable. Resetting tunnel; systemd will reconnect."
        pia disconnect >/dev/null 2>&1 || true
        exit 1
    fi
done
