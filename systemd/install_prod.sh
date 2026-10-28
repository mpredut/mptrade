#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TRADING_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SYSTEMD_DIR="$ROOT/systemd"

test -d "$ROOT/.git" || { echo "Repo missing: $ROOT"; exit 1; }

TRADING_USER="${TRADING_USER:-${SUDO_USER:-$(stat -c %U "$ROOT")}}"
id "$TRADING_USER" >/dev/null 2>&1 || { echo "Trading user does not exist: $TRADING_USER" >&2; exit 1; }
TRADING_GROUP="$(id -gn "$TRADING_USER")"
TRADING_HOME="$(getent passwd "$TRADING_USER" | cut -d: -f6)"
[ -n "$TRADING_HOME" ] || { echo "Home missing for $TRADING_USER" >&2; exit 1; }

REPO_ROOT="$ROOT" source "$ROOT/orchestratorOS/lib/env_common.sh"
PYTHON="${TRADING_PYTHON:-${PYTHON_BIN:-}}"
[ -x "$PYTHON" ] || { echo "No trading Python found; set TRADING_PYTHON or create .venv/myenv." >&2; exit 1; }

if [ "${1:-}" = "--render-only" ]; then
  [ -n "${TRADING_RENDER_DIR:-}" ] || { echo "TRADING_RENDER_DIR is required with --render-only." >&2; exit 1; }
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
  ! grep -q '@TRADING_' "$2" || { echo "Unresolved deployment placeholder in $1" >&2; return 1; }
}

mkdir -p "$TMP_DIR/antigravity-cli-daemon.service.d"
for tpl in python_orchestrator.service pia.service piavpn.service crontab.prod.txt \
           crontab.root.prod.txt bashrc sudoers-trading antigravity-cli-daemon.service \
           antigravity-cli-daemon.service.d/override.conf \
           antigravity-cli-daemon.service.d/store-fix.conf agy-bot; do
  render "$SYSTEMD_DIR/$tpl" "$TMP_DIR/$tpl"
done

if [ "${1:-}" = "--render-only" ]; then
  echo "Rendered deployment files in $TMP_DIR"
  exit 0
fi

test "$(id -u)" -eq 0 || { echo "Run this installer with sudo." >&2; exit 1; }

for unit in python_orchestrator.service pia.service piavpn.service; do
  install -m 0644 "$TMP_DIR/$unit" "/etc/systemd/system/$unit"
done
install -d -m 0755 /etc/systemd/resolved.conf.d /etc/ssh/sshd_config.d /etc/sysctl.d /etc/logrotate.d /etc/sudoers.d
install -m 0644 "$SYSTEMD_DIR/resolved-20-trading-cache.conf" /etc/systemd/resolved.conf.d/20-trading-cache.conf
install -m 0644 "$SYSTEMD_DIR/sshd-20-trading.conf" /etc/ssh/sshd_config.d/20-trading.conf
install -m 0644 "$SYSTEMD_DIR/sysctl-90-disable-ipv6.conf" /etc/sysctl.d/90-disable-ipv6.conf
sysctl -p /etc/sysctl.d/90-disable-ipv6.conf >/dev/null 2>&1 || true
install -m 0600 "$SYSTEMD_DIR/netplan-99-force-gateway.yaml" /etc/netplan/99-force-gateway.yaml
install -m 0644 "$SYSTEMD_DIR/logrotate-pia-daemon.conf" /etc/logrotate.d/pia-daemon
install -m 0755 "$SYSTEMD_DIR/trading-admin" /usr/local/sbin/trading-admin
install -m 0440 "$TMP_DIR/sudoers-trading" /etc/sudoers.d/trading
visudo -cf /etc/sudoers.d/trading

if systemctl is-enabled binance.service >/dev/null 2>&1 || systemctl is-active binance.service >/dev/null 2>&1; then
  systemctl stop binance.service 2>/dev/null || true
  systemctl disable binance.service 2>/dev/null || true
  rm -f /etc/systemd/system/binance.service /etc/systemd/system/multi-user.target.wants/binance.service
fi

install -d -o "$TRADING_USER" -g "$TRADING_GROUP" -m 0755 "$ROOT/logs"
crontab -u "$TRADING_USER" "$TMP_DIR/crontab.prod.txt"
crontab -u root "$TMP_DIR/crontab.root.prod.txt"

loginctl enable-linger "$TRADING_USER" 2>/dev/null || true
install -d -o "$TRADING_USER" -g "$TRADING_GROUP" -m 0755 \
  "$TRADING_HOME/.config/systemd/user/antigravity-cli-daemon.service.d" "$TRADING_HOME/.local/bin"
for u in antigravity-cli-daemon.service antigravity-cli-daemon.service.d/override.conf antigravity-cli-daemon.service.d/store-fix.conf; do
  install -o "$TRADING_USER" -g "$TRADING_GROUP" -m 0644 "$TMP_DIR/$u" "$TRADING_HOME/.config/systemd/user/$u"
done
install -o "$TRADING_USER" -g "$TRADING_GROUP" -m 0755 "$TMP_DIR/agy-bot" "$TRADING_HOME/.local/bin/agy-bot"

if [ -x "$TRADING_HOME/.local/bin/agy" ]; then
  TRADING_UID="$(id -u "$TRADING_USER")"
  sudo -u "$TRADING_USER" XDG_RUNTIME_DIR="/run/user/$TRADING_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TRADING_UID/bus" \
    bash -c 'systemctl --user daemon-reload && systemctl --user enable --now antigravity-cli-daemon.service' 2>/dev/null || true
fi

systemctl daemon-reload
sshd -t
systemctl enable cron.service piavpn.service pia.service python_orchestrator.service
systemctl restart systemd-resolved
systemctl reload ssh
systemctl restart piavpn.service pia.service python_orchestrator.service
systemctl --no-pager --full status python_orchestrator.service pia.service piavpn.service
crontab -u "$TRADING_USER" -l
crontab -u root -l
