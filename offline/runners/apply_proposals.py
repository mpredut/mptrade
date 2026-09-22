#!/usr/bin/env python3
"""apply_proposals.py — pulls backtest proposals from the git branch
`backtest-proposals` and applies them with guardrails on PROD. It does NOT rerun
the backtest (dev already did). The owning process is NOT restarted here — that
is done by watchdogfor_cacheandconfig when it detects the config change (a user decision).

Guardrails (applied HERE, on PROD, where the live value is authoritative):
  - the current value is RE-READ live from prod (not taken from the proposal) — if
    prod has changed in the meantime, we apply against the real value;
  - AVERAGING, not a jump: new = (current_prod + winner) / 2 (damping, as in the pilot);
  - RATE LIMIT: the same parameter does not change more often than MIN_DAYS_BETWEEN_CHANGES;
  - AUDIT: every decision (applied or not) in logger/backtest_pilot_audit.jsonl;
  - a git commit on main after each change (history plus reversibility).

  --dry-run : show what it would apply, without writing config or committing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.research.monitortrades_backtest import scheduled_pilot as sp  # noqa: E402
from botcore import parse_dotenv  # noqa: E402

# Load .env + config.env so the notifications (PHONE_ALERT_URL/NTFY_TOPIC from .env)
# work — apply runs standalone, not through the fleet (which loads them at startup).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    load_dotenv(os.path.join(ROOT, "config.env"))
except Exception:  # noqa: BLE001
    pass

_DEV_PROFILE = parse_dotenv(os.path.join(ROOT, "offline", "runners", "dev_backtest.env"))
BRANCH = os.environ.get("BACKTEST_PROPOSALS_BRANCH") or _DEV_PROFILE.get(
    "BACKTEST_PROPOSALS_BRANCH")
if not BRANCH:
    raise RuntimeError("Missing BACKTEST_PROPOSALS_BRANCH in dev_backtest.env")


def _read_proposals():
    """Read backtest_proposals.json from origin/BRANCH WITHOUT a checkout (git show)."""
    subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", BRANCH],
                   capture_output=True, timeout=30)
    out = subprocess.run(["git", "-C", ROOT, "show", f"origin/{BRANCH}:backtest_proposals.json"],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        return []
    try:
        return json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="show what it would apply, without writing config or committing")
    args = ap.parse_args()

    if os.environ.get("APPLY_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        print("[apply] APPLY_DISABLED=true -- exiting without doing anything")
        return

    proposals = _read_proposals()
    if not proposals:
        print(f"[apply] no proposal on branch {BRANCH} — nothing to do")
        return

    for p in proposals:
        fk, section, key = p["full_key"], p["section"], p["key"]
        symbol, winner = p["symbol"], float(p["winner_value"])
        current = sp._current_value(section, key)   # LIVE from prod (authoritative)
        rec = {"full_key": fk, "symbol": symbol, "source": "apply_proposals",
               "winner_value": winner, "current_value": current, "dev_commit": p.get("dev_commit")}

        if abs(winner - current) < 1e-9:
            rec["action"] = "no_change"
            rec["reason"] = f"the winner {winner} = the value already configured on prod"
            sp._append_audit(rec)
            print(f"[apply] {fk}: no_change (already {current})")
            continue

        last = sp._last_change_for(fk)
        if last:
            last_ts = datetime.fromisoformat(last["ts"])
            if datetime.now() - last_ts < timedelta(days=sp.MIN_DAYS_BETWEEN_CHANGES):
                rec["action"] = "rate_limited"
                rec["reason"] = f"last changed at {last['ts']} (< {sp.MIN_DAYS_BETWEEN_CHANGES} days)"
                sp._append_audit(rec)
                print(f"[apply] {fk}: rate_limited (the last change was {last['ts']})")
                continue

        new_value = round((current + winner) / 2, 4)
        rec["proposed_new_value"] = new_value
        rec["action"] = "would_apply" if args.dry_run else "applied"
        tag = " [dry-run]" if args.dry_run else ""
        print(f"[apply] {fk}: {current} -> {new_value} (winner backtest {winner}, dev {p.get('dev_commit')}){tag}")

        if not args.dry_run:
            sp._apply_config_change(section, key, current, new_value)
            sp._notify_change(fk, symbol, current, new_value, winner, rec)
            subprocess.run(["git", "-C", ROOT, "add", "instruments.conf"], capture_output=True, timeout=10)
            subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m",
                            f"apply backtest proposal: {fk} {current}->{new_value} "
                            f"(winner {winner}, dev {p.get('dev_commit')})"], capture_output=True, timeout=10)
        sp._append_audit(rec)

    if not args.dry_run:
        print("[apply] done. The owning process restart is performed by watchdogfor_cacheandconfig "
              "(detects instruments.conf change).")


if __name__ == "__main__":
    main()
