#!/usr/bin/env bash
# backup_secrets.sh — a COMPLETE backup of everything that is NOT in git (secrets plus the bot/provider state),
# discovered AUTOMATICALLY from git (nothing hardcoded) minus the regenerable parts (venv/log/pyc/lock/html).
# It catches: the .env files, keys/, ALL the .state_* files (HL/Kraken/T212/xstock/trailing), cachedb/,
# .watchdog_state, trade_cooldown, priceanalysis.json and so on — and future files, automatically.
# Output: folder + tarball OUTSIDE the repo. Copy the tarball OFF-MACHINE.
#
#   ./backup_secrets.sh                 # -> ~/<checkout>-secrets-backup/ + .tar.gz
#   ./backup_secrets.sh /media/usb/bk   # custom destination
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HOME/$(basename "$ROOT")-secrets-backup}"
case "$OUT" in "$ROOT"|"$ROOT"/*) echo "❌ the destination can NOT be inside the repo (secrets would be committed): $OUT"; exit 1;; esac

cd "$ROOT"
# Anything gitignored is not in git, so it needs a backup. Only regenerable files are excluded.
LIST="$(git ls-files --others --ignored --exclude-standard \
    | grep -vE '^(myenv|\.venv)/' \
    | grep -vE '(__pycache__|\.pyc$|\.log($|\.)|\.lock$|^index\.html$|^\.claude/)')"
[ -n "$LIST" ] || { echo "❌ nothing to save (git ls-files empty?)"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
printf '%s\n' "$LIST" | tar cf - -C "$ROOT" -T - | tar xf - -C "$OUT"

# Machine-specific data kept intentionally outside the repo. The PIA token is secret
# and it is required to restore the dedicated IP on a fresh install.
mkdir -p "$OUT/_machine"
PIA_TOKEN="${PIA_DIP_TOKEN:-$HOME/piatoken.txt}"
if [ -f "$PIA_TOKEN" ]; then
    install -m 0600 "$PIA_TOKEN" "$OUT/_machine/piatoken.txt"
fi

tar czf "$OUT.tar.gz" -C "$OUT" .   # latest, stable path for Windows pull
chmod -R go-rwx "$OUT" 2>/dev/null || true
chmod 600 "$OUT.tar.gz"

# HISTORY: also keep a DATED copy (the last KEEP days). Without history, a corruption or
# a deletion of secrets would enter the backup at 03:30 and OVERWRITE the only good copy.
KEEP="${BACKUP_KEEP:-7}"
DATED="$OUT-$(date +%Y%m%d).tar.gz"
cp -p "$OUT.tar.gz" "$DATED" && chmod 600 "$DATED"
ls -1t "$OUT"-????????.tar.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f

N="$(find "$OUT" -type f | wc -l)"
echo "=== a COMPLETE backup: $N files (secrets plus state) ==="
printf '%s\n' "$LIST" | sed 's/^/    /'
echo "Folder : $OUT"
echo "Tarball: $OUT.tar.gz (600)  + history: $DATED (keeping last $KEEP)"
echo "⚠ Copy the tarball OFF-MACHINE. It holds the HL wallet key + every API key. NOT in git!"
