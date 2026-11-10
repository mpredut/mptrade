#!/usr/bin/env bash
# pia_common.sh — Single Source of Truth (SSOT) for Private Internet Access (PIA) VPN
# CLI calls, health/routing probes, DIP reachability, and supervisor ownership state,
# shared by pia_supervisor.sh, vpn_watchdog.sh, and wait_for_vpn.sh.

_PIA_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=os_notify.sh
source "$_PIA_LIB_DIR/os_notify.sh"

# Load PIA credentials, interface, and timeouts from .env / config.env using env_get SSOT
PIA_ACCOUNT_USER="$(env_get PIA_ACCOUNT_USER "$(env_get PIA_USER)")"
PIA_USER="$PIA_ACCOUNT_USER"
PIA_PASS="$(env_get PIA_PASS)"
PIA_DIP_TOKEN_FRANKFURT="$(env_get PIA_DIP_TOKEN_FRANKFURT "$(env_get PIA_DIP_TOKEN)")"
PIA_DIP_TOKEN_BELGIUM="$(env_get PIA_DIP_TOKEN_BELGIUM)"
VPN_IF="${PIA_VPN_IF:-$(env_get PIA_VPN_IF wgpia0)}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-$(env_get PIA_CLI_TIMEOUT 6)}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-$(env_get PIA_PROBE_TIMEOUT 7)}"
PIA_SUPERVISOR_STATE_FILE="${PIA_SUPERVISOR_STATE_FILE:-/run/pia_supervisor.state}"

pia() {
  local _out="" _rc=0
  if [ "$(id -u)" -eq 0 ] && [ -n "${TRADING_USER:-}" ] && [ "$(id -un)" != "$TRADING_USER" ]; then
    _out=$(timeout "$CLI_TIMEOUT" runuser -u "$TRADING_USER" -- piactl "$@" 2>/dev/null) || _rc=$?
  else
    _out=$(timeout "$CLI_TIMEOUT" piactl "$@" 2>/dev/null) || _rc=$?
  fi
  [ -n "$_out" ] && printf '%s\n' "${_out//$'\r'/}"
  return $_rc
}

pia_vpn_route_up() {
  [ "$(pia get connectionstate)" = "Connected" ] || return 1
  ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1
  # Verify kernel policy routing sends normal application traffic via the VPN interface.
  ip route get 1.1.1.1 2>/dev/null | grep -q "dev $VPN_IF" || return 1
}

vpn_healthy() {
  pia_vpn_route_up || return 1

  local binance_ip=""
  if command -v resolvectl >/dev/null 2>&1; then
    binance_ip=$(timeout 4 resolvectl query --cache=no --type=A -i "$VPN_IF" api.binance.com 2>/dev/null | grep -E -o '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n1 || true)
  else
    binance_ip=$(timeout 4 getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}' || true)
  fi
  [ -n "$binance_ip" ] || return 1

  curl -4 --interface "$VPN_IF" \
    --resolve "api.binance.com:443:$binance_ip" \
    --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
    --fail --silent --show-error https://api.binance.com/api/v3/time \
    >/dev/null 2>&1 || return 1
}

dip_endpoint_reachable() {
  local region="$1" dip_ip http_code
  dip_ip=$(printf '%s' "$region" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}$' || true)
  [ -n "$dip_ip" ] || return 0
  http_code=$(curl -4 -k -s -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 4 "https://${dip_ip}:1337/" 2>/dev/null || true)
  [ -n "$http_code" ] && [ "$http_code" != "000" ]
}

pia_write_supervisor_state() {  # $1=mode (starting|healthy|connecting|failover|isp_cooldown)
  local mode="${1:-running}" tmp="${PIA_SUPERVISOR_STATE_FILE}.tmp.$$"
  printf 'SUPERVISOR_TS=%s\nSUPERVISOR_PID=%s\nSUPERVISOR_MODE=%s\n' "$(date +%s)" "$$" "$mode" > "$tmp" 2>/dev/null \
    && mv -f "$tmp" "$PIA_SUPERVISOR_STATE_FILE" 2>/dev/null || rm -f "$tmp" 2>/dev/null || true
}

pia_supervisor_owning_tunnel() {
  local st since now age sup_ts="" mpid=""
  st=$(systemctl show -p ActiveState --value pia.service 2>/dev/null || echo "")
  [ "$st" = "activating" ] && return 0
  [ "$st" = "active" ] || return 1
  if [ -r "$PIA_SUPERVISOR_STATE_FILE" ]; then
    sup_ts=$(awk -F= '$1=="SUPERVISOR_TS"{print $2}' "$PIA_SUPERVISOR_STATE_FILE" 2>/dev/null | tr -cd '0-9' || true)
    [ -n "$sup_ts" ] && [ $(( $(date +%s) - sup_ts )) -lt 180 ]
    return $?
  fi
  since=$(systemctl show -p ActiveEnterTimestampMonotonic --value pia.service 2>/dev/null || echo 0)
  now=$(awk '{printf "%d", $1*1000000}' /proc/uptime 2>/dev/null || echo 0)
  age=$(( (now - ${since:-0}) / 1000000 ))
  if [ "$age" -ge 0 ] && [ "$age" -lt "${CONNECT_SETTLE:-120}" ]; then
    return 0
  fi
  mpid=$(systemctl show -p MainPID --value pia.service 2>/dev/null || echo 0)
  [ "${mpid:-0}" -gt 0 ] && [ -d "/proc/$mpid" ]
}

pia_ensure_lan_rule() {
  if ! ip rule show 2>/dev/null | grep -q "to 192.168.0.0/24 lookup main"; then
    ip rule add to 192.168.0.0/24 lookup main pref 50 2>/dev/null || true
  fi
}

pia_ensure_ipv6_disabled() {
  local uplink_if="${1:-${PIA_UPLINK_IF:-ens18}}"
  [ "$(id -u)" -eq 0 ] || return 0
  if [ -f "$ROOT/systemd/sysctl-90-disable-ipv6.conf" ] && [ ! -f /etc/sysctl.d/90-disable-ipv6.conf ]; then
    install -m 0644 "$ROOT/systemd/sysctl-90-disable-ipv6.conf" /etc/sysctl.d/90-disable-ipv6.conf 2>/dev/null || true
  fi
  sysctl -w net.ipv6.conf.all.disable_ipv6=1 \
            net.ipv6.conf.default.disable_ipv6=1 \
            "net.ipv6.conf.${uplink_if}.disable_ipv6=1" \
            net.ipv6.conf.lo.disable_ipv6=0 >/dev/null 2>&1 || true
}

pia_with_secret_file() {  # $1=secret_content $2..=command (secret file path appended as last arg)
  local content="$1"; shift
  local tmp rc=0
  tmp="$(mktemp -p /dev/shm 2>/dev/null || mktemp)"
  chmod 0600 "$tmp"
  printf "%s\n" "$content" > "$tmp"
  if [ "$(id -u)" -eq 0 ] && [ -n "${TRADING_USER:-}" ]; then
    chown "$TRADING_USER" "$tmp" 2>/dev/null || true
  fi
  "$@" "$tmp" >/dev/null 2>&1 || rc=$?
  rm -f "$tmp"
  return $rc
}

pia_login_from_env() {
  [ -n "${PIA_USER:-}" ] && [ -n "${PIA_PASS:-}" ] || return 1
  echo "Attempting PIA login using .env credentials..."
  pia_with_secret_file "$(printf "%s\n%s" "$PIA_USER" "$PIA_PASS")" pia login
}

pia_register_dip_if_needed() {  # $1=token $2=region_prefix $3=label
  local token="${1:-}" prefix="$2" label="$3"
  [ -n "$token" ] || return 0
  if ! pia get regions 2>/dev/null | grep -q "^${prefix}"; then
    echo "Registering ${label} Dedicated IP from .env..."
    if ! pia_with_secret_file "$token" pia dedicatedip add; then
      pia_login_from_env || true
    fi
    if pia get regions 2>/dev/null | grep -q "^${prefix}" || pia_with_secret_file "$token" pia dedicatedip add; then
      echo "Successfully added ${label} Dedicated IP token from .env"
    else
      echo "warning: Failed to add ${label} Dedicated IP token from .env" >&2
    fi
  fi
}

ensure_dedicated_ips_registered() {
  pia_register_dip_if_needed "${PIA_DIP_TOKEN_FRANKFURT:-}" "dedicated-de-frankfurt" "Frankfurt"
  pia_register_dip_if_needed "${PIA_DIP_TOKEN_BELGIUM:-}" "dedicated-belgium" "Belgium"
}

pia_preferred_dedicated_region() {
  local regions match=""
  regions="$(pia get regions 2>/dev/null || true)"
  [ -n "$regions" ] || return 1
  for prefix in "^dedicated-de-frankfurt" "^dedicated-belgium" "^dedicated-"; do
    match="$(printf '%s\n' "$regions" | grep -m1 "$prefix" || true)"
    if [ -n "$match" ]; then
      printf '%s\n' "$match"
      return 0
    fi
  done
  return 1
}
