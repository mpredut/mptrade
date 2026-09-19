#!/usr/bin/env bash
# Make the venv relocatable so the repo root can be renamed with a pure `mv` -- no
# hardcoded folder path anywhere. `python -m venv` bakes an absolute VIRTUAL_ENV into
# myenv/bin/activate; this rewrites it to SELF-DERIVE from the script's own location, so
# the fleet (which sources $ROOT/myenv/bin/activate under bash) works at any path.
#
# Idempotent. Re-run after any venv (re)creation. Everything else in the repo already
# derives its root ($ROOT in the shell scripts, __file__ in Python, @TRADING_ROOT@
# auto-derived by systemd/install_prod.sh), so this is the last piece.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_NAME="${1:-}"
if [ -z "$VENV_NAME" ]; then
  if [ -d "$ROOT/myenv" ]; then
    VENV_NAME="myenv"
  elif [ -d "$ROOT/.venv" ]; then
    VENV_NAME=".venv"
  else
    echo "no myenv or .venv directory found in $ROOT" >&2; exit 1
  fi
fi
ACT="$ROOT/$VENV_NAME/bin/activate"
[ -f "$ACT" ] || { echo "no venv activate at $ACT" >&2; exit 1; }

python3 - "$ACT" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path).read()
# Self-derive VIRTUAL_ENV from activate's own location (bash: BASH_SOURCE; fallback $0).
derive = 'export VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"'
# Match the non-cygwin `export VIRTUAL_ENV=<literal path>` line only (leave the
# cygpath Windows branch, which never runs on the Linux server, untouched).
out = re.sub(r'(?m)^(\s*)export VIRTUAL_ENV=(?!\$\(cygpath)(?!")\S.*$', r'\1' + derive, src)
if out != src:
    open(path, "w").write(out)
    print(f"activate patched to self-derive VIRTUAL_ENV: {path}")
else:
    print(f"activate already self-deriving (no change): {path}")
PY

# Prove it: the derived VIRTUAL_ENV must equal the real venv path when sourced here.
# shellcheck disable=SC1090
( . "$ACT" && [ "$VIRTUAL_ENV" = "$ROOT/$VENV_NAME" ] \
  && echo "verified: VIRTUAL_ENV -> $VIRTUAL_ENV" \
  || { echo "VERIFY FAILED: VIRTUAL_ENV=$VIRTUAL_ENV expected $ROOT/$VENV_NAME" >&2; exit 1; } )
