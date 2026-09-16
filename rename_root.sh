#!/usr/bin/env bash
# Rename the trading root folder in ONE automated, self-contained step.
#
#     sudo bash rename_root.sh <new-folder-name>
#
# No preset env vars, nothing typed by hand beyond the new name: the script derives the
# current root from its OWN location, and calls systemd/install_prod.sh with no env --
# install_prod itself auto-derives TRADING_ROOT (its location), TRADING_USER (SUDO_USER)
# and the Python (auto-detects myenv). The venv is made relocatable first so the move
# needs no path edits. Idempotent-ish: refuses if the destination already exists.
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "Run with sudo: systemd/cron install and the root-owned venv need root." >&2; exit 1; }
NEW="${1:?usage: sudo bash rename_root.sh <new-folder-name>}"
case "$NEW" in */*|.*|"") echo "Give a bare folder name (no path, no leading dot)." >&2; exit 1;; esac

CUR="$(cd "$(dirname "$0")" && pwd)"          # current root, derived from this script
PARENT="$(dirname "$CUR")"
DEST="$PARENT/$NEW"
TUSER="${SUDO_USER:-$(stat -c %U "$CUR")}"    # trading user, derived (never from preset env)

[ "$CUR" != "$DEST" ] || { echo "Already named '$NEW'; nothing to do."; exit 0; }
[ ! -e "$DEST" ] || { echo "Destination already exists: $DEST" >&2; exit 1; }
id "$TUSER" >/dev/null 2>&1 || { echo "Derived trading user does not exist: $TUSER" >&2; exit 1; }
echo "Renaming trading root:  $CUR  ->  $DEST   (user=$TUSER)"

echo "== [1/6] stop the fleet + pause supervision (brief trading pause) =="
systemctl stop binance.service 2>/dev/null || true
systemctl stop cron 2>/dev/null || true       # so healthcheck does not respawn at the old path mid-move

echo "== [2/6] rename the folder =="
mv "$CUR" "$DEST"
cd "$DEST"

echo "== [3/6] make the venv relocatable (self-deriving activate; root can write it) =="
bash "$DEST/make_venv_portable.sh"

echo "== [4/6] re-render systemd units + both crontabs for the new path (auto-derived) =="
bash "$DEST/systemd/install_prod.sh"           # no env vars: install_prod derives root/user/python

echo "== [5/6] restart the fleet + relaunch the bots at the new path =="
systemctl start cron
systemctl restart pia.service binance.service 2>/dev/null || systemctl start binance.service
runuser -u "$TUSER" -- bash "$DEST/bots_start.sh" >/dev/null 2>&1 || \
  sudo -u "$TUSER" bash "$DEST/bots_start.sh" >/dev/null 2>&1 || true

echo "== [6/6] verify =="
sleep 3
echo "  binance.service: $(systemctl is-active binance.service 2>/dev/null || echo unknown)"
echo "DONE. Trading root is now: $DEST"
echo "Confirm with:  $DEST/healthcheck.sh --check   (and: git -C $DEST remote -v)"
