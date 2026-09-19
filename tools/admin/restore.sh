#!/usr/bin/env bash
# restore.sh — DISASTER RECOVERY: rebuild EVERYTHING on a new machine, in one command.
#
# It assumes: the repo is already cloned (you need it in order to run the script) plus the folder of
# SECRETS copied from your backup (it is NOT in git — made with ./backup_secrets.sh).
#
#   git clone <repository-url> /srv/trading/current
#   cd /srv/trading/current && ./restore.sh /path/to/secrets-backup
#
# The secrets folder MIRRORS the repo structure (.env, hyperliquid/.env, keys/, ...).
# The repository path and account name are detected and passed to the installer.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SECRETS="${1:-}"
fail() { echo "❌ $*" >&2; exit 1; }

echo "===== RESTORE @ $ROOT ====="
[ -n "$SECRETS" ] || fail "Usage: $0 <secrets_folder>  (made with ./backup_secrets.sh)"
[ -d "$SECRETS" ] || fail "The secrets folder does not exist: $SECRETS"
command -v python3 >/dev/null || fail "python3 is missing (apt install python3 python3-venv)"

echo "--- [1/5] restoring the secrets plus the state from $SECRETS ---"
# _machine holds data external to the repo and is restored separately.
tar cf - --exclude='./_machine' -C "$SECRETS" . | tar xf - -C "$ROOT"
if [ -f "$SECRETS/_machine/piatoken.txt" ]; then
    install -m 0600 "$SECRETS/_machine/piatoken.txt" "$HOME/piatoken.txt"
fi
echo "    ✔ restored (.env, keys/, .state*, cachedb/, PIA token)"

echo "--- [2/5] venv (myenv) + dependencies ---"
[ -x "$ROOT/myenv/bin/python" ] || python3 -m venv "$ROOT/myenv" || fail "cannot create the venv"
"$ROOT/myenv/bin/pip" install -q --upgrade pip
"$ROOT/myenv/bin/pip" install -q -r "$ROOT/requirements.txt" || fail "pip install failed"
echo "    ✔ dependencies installed"

echo "--- [3/5] systemd + DNS + SSH + cron (needs sudo) ---"
if sudo -v 2>/dev/null; then
    sudo env TRADING_ROOT="$ROOT" TRADING_USER="$(id -un)" \
        TRADING_PYTHON="$ROOT/myenv/bin/python" \
        bash "$ROOT/systemd/install_prod.sh"
    echo "    ✔ PROD profile installed"
else
    echo "    ! no sudo — by hand: sudo bash systemd/install_prod.sh"
fi

echo "--- [4/5] cron check ---"
crontab -l >/dev/null 2>&1 && echo "    ✔ cron installed by the PROD profile"

echo "--- [5/5] DONE ---"
echo "Still needed (once): install the PIA application and log in, then:"
echo "    sudo systemctl start pia binance     # the fleet starts; the bots arrive through cron"
echo "    ./healthcheck.sh --check             # check that everything is 'ok'"
