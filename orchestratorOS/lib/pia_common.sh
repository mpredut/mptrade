#!/usr/bin/env bash
# pia_common.sh — shared Private Internet Access (PIA) VPN & network helpers
# used by pia_supervisor.sh and vpn_watchdog.sh.

_PIA_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=os_notify.sh
source "$_PIA_LIB_DIR/os_notify.sh"

# Load PIA credentials and tokens from .env / config.env using env_get SSOT
PIA_ACCOUNT_USER="$(env_get PIA_ACCOUNT_USER "$(env_get PIA_USER)")"
PIA_USER="$PIA_ACCOUNT_USER"
PIA_PASS="$(env_get PIA_PASS)"
PIA_DIP_TOKEN_FRANKFURT="$(env_get PIA_DIP_TOKEN_FRANKFURT "$(env_get PIA_DIP_TOKEN)")"
PIA_DIP_TOKEN_BELGIUM="$(env_get PIA_DIP_TOKEN_BELGIUM)"
VPN_IF="${PIA_VPN_IF:-$(env_get PIA_VPN_IF wgpia0)}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-$(env_get PIA_CLI_TIMEOUT 6)}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-$(env_get PIA_PROBE_TIMEOUT 7)}"

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
