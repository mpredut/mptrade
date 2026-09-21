#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TRADING_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SYSTEMD_DIR="$ROOT/systemd"

test -d "$ROOT/.git" || { echo "Repo missing: $ROOT"; exit 1; }

TRADING_USER="${TRADING_USER:-${SUDO_USER:-$(stat -c %U "$ROOT")}}"
id "$TRADING_USER" >/dev/null 2>&1 || {
  echo "Trading user does not exist: $TRADING_USER" >&2; exit 1;
}
TRADING_GROUP="$(id -gn "$TRADING_USER")"
TRADING_HOME="$(getent passwd "$TRADING_USER" | cut -d: -f6)"
[ -n "$TRADING_HOME" ] || { echo "Home missing for $TRADING_USER" >&2; exit 1; }

if [ -n "${TRADING_PYTHON:-}" ]; then
  PYTHON="$TRADING_PYTHON"
else
  PYTHON=""
  for candidate in "$ROOT/.venv/bin/python" "$ROOT/myenv/bin/python"; do
    [ -x "$candidate" ] && { PYTHON="$candidate"; break; }
  done
fi
[ -x "$PYTHON" ] || {
  echo "No trading Python found; set TRADING_PYTHON or create .venv/myenv." >&2
  exit 1
}

if [ "${1:-}" = "--render-only" ]; then
  [ -n "${TRADING_RENDER_DIR:-}" ] || {
    echo "TRADING_RENDER_DIR is required with --render-only." >&2; exit 1;
  }
  TMP_DIR="$TRADING_RENDER_DIR"
  mkdir -p "$TMP_DIR"
else
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
fi
escape_replacement() { printf '%s' "$1" | sed 's/[&|\\]/\\&/g'; }
ROOT_SED="$(escape_replacement "$ROOT")"
USER_SED="$(escape_replacement "$TRADING_USER")"
HOME_SED="$(escape_replacement "$TRADING_HOME")"
PYTHON_SED="$(escape_replacement "$PYTHON")"
render() {
  sed -e "s|@TRADING_ROOT@|$ROOT_SED|g" \
      -e "s|@TRADING_USER@|$USER_SED|g" \
      -e "s|@TRADING_HOME@|$HOME_SED|g" \
      -e "s|@TRADING_PYTHON@|$PYTHON_SED|g" "$1" > "$2"
  ! grep -q '@TRADING_' "$2" || {
    echo "Unresolved deployment placeholder in $1" >&2; return 1;
  }
}

render "$SYSTEMD_DIR/python_orchestrator.service" "$TMP_DIR/python_orchestrator.service"
render "$SYSTEMD_DIR/pia.service" "$TMP_DIR/pia.service"
render "$SYSTEMD_DIR/piavpn.service" "$TMP_DIR/piavpn.service"
render "$SYSTEMD_DIR/crontab.prod.txt" "$TMP_DIR/crontab.prod.txt"
render "$SYSTEMD_DIR/crontab.root.prod.txt" "$TMP_DIR/crontab.root.prod.txt"
render "$SYSTEMD_DIR/bashrc" "$TMP_DIR/bashrc"
render "$SYSTEMD_DIR/sudo.txt" "$TMP_DIR/sudo.txt"
render "$SYSTEMD_DIR/sudoers-trading" "$TMP_DIR/sudoers-trading"
render "$ROOT/hyperliquid/hl-dn.service" "$TMP_DIR/hl-dn.service"
render "$ROOT/kraken/xstock-watch.service" "$TMP_DIR/xstock-watch.service"

if [ "${1:-}" = "--render-only" ]; then
  echo "Rendered deployment files in $TMP_DIR"
  exit 0
fi

test "$(id -u)" -eq 0 || { echo "Run this installer with sudo." >&2; exit 1; }

install -m 0644 "$TMP_DIR/python_orchestrator.service" /etc/systemd/system/python_orchestrator.service
install -m 0644 "$TMP_DIR/pia.service" /etc/systemd/system/pia.service
install -m 0644 "$TMP_DIR/piavpn.service" /etc/systemd/system/piavpn.service
install -d -m 0755 /etc/systemd/resolved.conf.d
install -m 0644 "$SYSTEMD_DIR/resolved-20-trading-cache.conf" \
  /etc/systemd/resolved.conf.d/20-trading-cache.conf
install -d -m 0755 /etc/ssh/sshd_config.d
install -m 0644 "$SYSTEMD_DIR/sshd-20-trading.conf" \
  /etc/ssh/sshd_config.d/20-trading.conf
# Direct-to-router uplink (see netplan-99-force-gateway.yaml and DISASTER_RECOVERY.md).
# Installed but deliberately NOT applied here: `netplan apply` while PIA is connected wipes
# PIA's policy routing. It takes effect on the next reboot (clean order: netplan then
# pia.service); to apply it immediately run `netplan apply` then `systemctl restart
# pia.service` to rebuild the tunnel routing.
install -m 0600 "$SYSTEMD_DIR/netplan-99-force-gateway.yaml" \
  /etc/netplan/99-force-gateway.yaml
# Cap PIA's debug daemon log (debug logging is left ON in production for diagnostics).
install -d -m 0755 /etc/logrotate.d
install -m 0644 "$SYSTEMD_DIR/logrotate-pia-daemon.conf" /etc/logrotate.d/pia-daemon

install -d -o "$TRADING_USER" -g "$TRADING_GROUP" -m 0755 "$ROOT/logs"
crontab -u "$TRADING_USER" "$TMP_DIR/crontab.prod.txt"
# Root crontab is reserved for tasks that require root privileges.
# pia_selfheal.sh is a MANUAL tool only — do NOT add it here; pia.service
# (Restart=always) is the sole automated PIA recovery mechanism.
crontab -u root "$TMP_DIR/crontab.root.prod.txt"

systemctl daemon-reload
sshd -t
systemctl enable cron.service
systemctl enable piavpn.service pia.service python_orchestrator.service
systemctl restart systemd-resolved
systemctl reload ssh
systemctl restart piavpn.service pia.service python_orchestrator.service

systemctl --no-pager --full status python_orchestrator.service pia.service piavpn.service
crontab -u "$TRADING_USER" -l
crontab -u root -l
