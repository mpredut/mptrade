#!/usr/bin/env bash
# Rename the trading root folder in ONE automated, self-contained step.
#
#     sudo bash rename_root.sh <new-folder-name>          # do it
#          bash rename_root.sh --dry-run <new-folder-name>  # preview + validate, no changes
#
# No preset env vars, nothing typed by hand beyond the new name: the script derives the
# current root from its OWN location, and calls systemd/install_prod.sh with no env --
# install_prod itself auto-derives TRADING_ROOT (its location), TRADING_USER (SUDO_USER)
# and the Python (auto-detects myenv). The venv is made relocatable first so the move
# needs no path edits.
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
NEW="${1:?usage: [sudo] bash rename_root.sh [--dry-run] <new-folder-name>}"
case "$NEW" in */*|.*|"") echo "Give a bare folder name (no path, no leading dot)." >&2; exit 1;; esac

CUR="$(cd "$(dirname "$0")/../.." && pwd)"  # current root, derived from this script
PARENT="$(dirname "$CUR")"
DEST="$PARENT/$NEW"
TUSER="${SUDO_USER:-$(stat -c %U "$CUR")}"    # trading user, derived (never from preset env)

[ "$CUR" != "$DEST" ] || { echo "Already named '$NEW'; nothing to do."; exit 0; }
[ ! -e "$DEST" ] || { echo "Destination already exists: $DEST" >&2; exit 1; }
id "$TUSER" >/dev/null 2>&1 || { echo "Derived trading user does not exist: $TUSER" >&2; exit 1; }
echo "Plan: rename trading root  $CUR  ->  $DEST   (user=$TUSER)"

if [ "$DRY" = 1 ]; then
  echo "[dry-run] steps that WOULD run: stop binance.service+cron -> mv -> make_venv_portable"
  echo "[dry-run]   -> systemd/install_prod.sh (no env) -> restart fleet/pia -> bots_start -> verify"
  echo "[dry-run] validating install_prod auto-derivation (render-only, installs nothing)..."
  RD="$(mktemp -d)"
  if TRADING_RENDER_DIR="$RD" bash "$CUR/systemd/install_prod.sh" --render-only >/dev/null 2>&1; then
    hits="$(grep -rl "$CUR" "$RD" 2>/dev/null | wc -l)"
    echo "[dry-run]   render OK: @TRADING_ROOT@ auto-derived to $CUR ($hits rendered files reference it, no preset env)"
  else
    echo "[dry-run]   render-only FAILED -- inspect before the real run" >&2
  fi
  rm -rf "$RD"
  echo "[dry-run] no changes made."
  exit 0
fi

[ "$(id -u)" = 0 ] || { echo "Run the real rename with sudo (systemd/cron + root-owned venv need root)." >&2; exit 1; }

echo "== [1/6] stop the fleet + pause supervision (brief trading pause) =="
systemctl stop python_orchestrator.service 2>/dev/null || true
systemctl stop cron 2>/dev/null || true       # so healthcheck does not respawn at the old path mid-move

echo "== [2/6] rename the folder =="
mv "$CUR" "$DEST"
cd "$DEST"

echo "== [3/6] make the venv relocatable (self-deriving activate; root can write it) =="
bash "$DEST/orchestratorOS/admin/make_venv_portable.sh"

echo "== [4/6] re-render systemd units + both crontabs for the new path (auto-derived) =="
bash "$DEST/systemd/install_prod.sh"           # no env vars: install_prod derives root/user/python

echo "== [5/6] restart the fleet + relaunch the bots at the new path =="
systemctl start cron
systemctl restart pia.service python_orchestrator.service 2>/dev/null || systemctl start python_orchestrator.service

echo "== [6/6] verify =="
sleep 3
echo "  python_orchestrator.service: $(systemctl is-active python_orchestrator.service 2>/dev/null || echo unknown)"
echo "DONE. Trading root is now: $DEST"
echo "Confirm with:  $DEST/healthcheck.sh --check   (and: git -C $DEST remote -v)"
