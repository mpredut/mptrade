#!/usr/bin/env bash
# Idempotent, zero-downtime OS clock-sync installer (chrony) for mptrade PROD & DEV hosts.
#
# Usage:
#   sudo bash systemd/install_clock_sync.sh
#
# Deliberately does NOT restart python_orchestrator.service or pia.service so live trading
# continues uninterrupted. Also invoked automatically by systemd/install_prod.sh during full
# host provisioning / disaster recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHRONY_SRC="$SCRIPT_DIR/chrony-trading.conf"
ADMIN_SRC="$SCRIPT_DIR/trading-admin"

test "$(id -u)" -eq 0 || { echo "Run this installer with sudo." >&2; exit 1; }
test -r "$CHRONY_SRC" || { echo "Missing chrony config: $CHRONY_SRC" >&2; exit 1; }

if ! command -v chronyd >/dev/null 2>&1 || ! command -v chronyc >/dev/null 2>&1; then
  echo "[clock-sync] Installing chrony package..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq chrony
fi

# Ensure systemd-timesyncd does not compete with chrony.
if systemctl list-unit-files systemd-timesyncd.service >/dev/null 2>&1; then
  systemctl stop systemd-timesyncd.service 2>/dev/null || true
  systemctl disable systemd-timesyncd.service 2>/dev/null || true
fi

if [ -f /etc/chrony/chrony.conf ] && [ ! -f /etc/chrony/chrony.conf.orig ]; then
  cp -a /etc/chrony/chrony.conf /etc/chrony/chrony.conf.orig
fi

install -d -m 0755 /etc/chrony/conf.d /var/lib/chrony /var/log/chrony
install -m 0644 -o root -g root "$CHRONY_SRC" /etc/chrony/chrony.conf
chronyd -p -f /etc/chrony/chrony.conf >/dev/null

# Keep trading-admin updated with sync-clock support when present or on PROD hosts.
if [ -r "$ADMIN_SRC" ] && { [ -e /usr/local/sbin/trading-admin ] || [ -f /etc/systemd/system/python_orchestrator.service ]; }; then
  install -m 0755 -o root -g root "$ADMIN_SRC" /usr/local/sbin/trading-admin
fi

systemctl enable chrony.service >/dev/null 2>&1 || true
systemctl restart chrony.service
chronyc waitsync 10 0.05 0.0 1 >/dev/null 2>&1 || true

echo "[clock-sync] chrony active; tracking summary:"
chronyc tracking || true
chronyc sources -v || true
