#!/usr/bin/env python3
"""apply_proposals.py -- pulls backtest proposals from the git branch
`backtest-proposals` and applies them with guardrails on PROD. It does NOT rerun
the backtest (dev already did). The owning process is NOT restarted here -- that
is done by watchdogfor_cacheandconfig when it detects the config change (a user decision).

Guardrails (applied HERE, on PROD, where the live value is authoritative):
  - the current value is RE-READ live from prod (not taken from the proposal) -- if
    prod has changed in the meantime, we apply against the real value;
  - AVERAGING, not a jump: new = (current_prod + winner) / 2 (damping, as in the pilot);
  - RATE LIMIT: the same parameter does not change more often than MIN_DAYS_BETWEEN_CHANGES;
  - AUDIT: every decision (applied or not) in logger/backtest_pilot_audit.jsonl;
  - a git commit on main after each change (history plus reversibility).

  --dry-run : show what it would apply, without writing config or committing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import FileLock, _parse_bool_flag, bool_env, load_env_stack, optional_env, parse_dotenv, safe_pos_float  # noqa: E402
from state_io import parse_json_payload  # noqa: E402
from offline.research.monitortrades_backtest import scheduled_pilot as sp  # noqa: E402

_DEV_PROFILE = parse_dotenv(os.path.join(ROOT, "offline", "runners", "dev_backtest.env"))
BRANCH = (optional_env("BACKTEST_PROPOSALS_BRANCH") or _DEV_PROFILE.get("BACKTEST_PROPOSALS_BRANCH") or "").strip()
_ALLOWED_PILOT_PARAM_NAMES = frozenset(
    fk.split(".", 1)[1] for fk in sp.PILOT_KEYS if "." in fk
)


def _configured_branch(default: str = "") -> str:
    return (
        BRANCH
        or optional_env("BACKTEST_PROPOSALS_BRANCH")
        or str(_DEV_PROFILE.get("BACKTEST_PROPOSALS_BRANCH") or "")
        or default
    ).strip()


def _validate_branch_name(raw: object) -> str:
    clean = raw.strip() if isinstance(raw, str) and not isinstance(raw, bool) else ""
    if not clean or clean.startswith("-") or ".." in clean or any(ch.isspace() for ch in clean):
        raise RuntimeError(f"Invalid BACKTEST_PROPOSALS_BRANCH: {raw!r}")
    return clean


def _resolve_branch() -> str:
    raw = _configured_branch()
    if not raw:
        raise RuntimeError("Missing BACKTEST_PROPOSALS_BRANCH in dev_backtest.env")
    return _validate_branch_name(raw)


def _read_proposals(branch: str | None = None):
    """Read backtest_proposals.json from origin/BRANCH WITHOUT a checkout (git show)."""
    target_branch = _resolve_branch() if branch is None else _validate_branch_name(branch)
    try:
        fetch_res = subprocess.run(
            ["git", "-C", ROOT, "fetch", "-q", "origin", target_branch],
            capture_output=True, timeout=30,
        )
        if fetch_res.returncode != 0:
            return []
        out = subprocess.run(
            ["git", "-C", ROOT, "show", f"origin/{target_branch}:backtest_proposals.json"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    data = parse_json_payload(out.stdout or "[]", default=[])
    return data if isinstance(data, list) else []


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="show what it would apply, without writing config or committing")
    args = ap.parse_args(argv)

    try:
        load_env_stack()
    except Exception:  # noqa: BLE001
        pass

    if bool_env("APPLY_DISABLED", default=False):
        print("[apply] APPLY_DISABLED=true -- exiting without doing anything")
        return

    proposals = _read_proposals()
    if not proposals:
        branch_label = _configured_branch("backtest-proposals")
        print(f"[apply] no proposal on branch {branch_label} -- nothing to do")
        return

    with FileLock("/tmp/mptrade_apply_proposals.lock"):
        applied_count = 0
        for p in proposals:
            if not isinstance(p, dict):
                continue
            fk, section, key = p.get("full_key"), p.get("section"), p.get("key")
            symbol, raw_winner = p.get("symbol"), p.get("winner_value")
            if (
                not isinstance(fk, str) or not fk.strip()
                or not isinstance(section, str) or not section.strip()
                or not isinstance(key, str) or not key.strip()
                or not isinstance(symbol, str) or not symbol.strip()
                or raw_winner is None or isinstance(raw_winner, bool)
            ):
                continue
            fk = fk.strip()
            section = section.strip()
            key = key.strip()
            symbol = symbol.strip()
            if fk != f"{section}.{key}" or key not in _ALLOWED_PILOT_PARAM_NAMES:
                sp._append_audit({
                    "full_key": fk, "symbol": symbol, "source": "apply_proposals",
                    "action": "skipped_unlisted_key",
                    "reason": f"key {fk!r} outside pilot whitelist or mismatched section/key",
                    "dev_commit": p.get("dev_commit"),
                })
                continue
            try:
                current_raw = sp._current_value(section, key)   # LIVE from prod (authoritative)
            except (KeyError, ValueError) as exc:
                sp._append_audit({
                    "full_key": fk, "symbol": symbol, "source": "apply_proposals",
                    "action": "skipped_missing_key", "reason": str(exc),
                    "dev_commit": p.get("dev_commit"),
                })
                continue
            winner = safe_pos_float(raw_winner, default=None)
            current = safe_pos_float(current_raw, default=None)
            if (
                winner is None
                or current is None
                or (key == "mt.hardtp_fraction" and (winner > 1.0 or current > 1.0))
            ):
                sp._append_audit({
                    "full_key": fk, "symbol": symbol, "source": "apply_proposals",
                    "action": "skipped_invalid_value",
                    "reason": f"winner={raw_winner!r}, current={current_raw!r}",
                    "dev_commit": p.get("dev_commit"),
                })
                continue
            rec = {"full_key": fk, "symbol": symbol, "source": "apply_proposals",
                   "winner_value": winner, "current_value": current, "dev_commit": p.get("dev_commit")}

            new_value = round((current + winner) / 2, 4)
            if abs(winner - current) < 1e-9 or abs(new_value - current) < 1e-9:
                rec["action"] = "no_change"
                rec["reason"] = f"the damped value {new_value} = the value already configured on prod ({current})"
                sp._append_audit(rec)
                print(f"[apply] {fk}: no_change (already {current})")
                continue

            limited, last_ts_str = sp.is_rate_limited(fk)
            if limited:
                rec["action"] = "rate_limited"
                rec["reason"] = f"last changed at {last_ts_str} (< {sp.MIN_DAYS_BETWEEN_CHANGES} days)"
                sp._append_audit(rec)
                print(f"[apply] {fk}: rate_limited (the last change was {last_ts_str})")
                continue

            rec["proposed_new_value"] = new_value
            rec["action"] = "would_apply" if args.dry_run else "applied"
            tag = " [dry-run]" if args.dry_run else ""
            print(f"[apply] {fk}: {current} -> {new_value} (winner backtest {winner}, dev {p.get('dev_commit')}){tag}")

            if not args.dry_run:
                sp._apply_config_change(section, key, current, new_value)
                commit_ok = False
                try:
                    commit_res = subprocess.run(
                        [
                            "git", "-C", ROOT, "commit", "-q", "--only", "instruments.conf",
                            "-m",
                            f"apply backtest proposal: {fk} {current}->{new_value} "
                            f"(winner {winner}, dev {p.get('dev_commit')})",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    commit_ok = commit_res.returncode == 0
                    if not commit_ok:
                        rec["action"] = "commit_failed"
                        rec["reason"] = (commit_res.stderr or commit_res.stdout or f"rc={commit_res.returncode}").strip()
                        print(f"[apply] git commit failed for {fk}: {rec['reason']}")
                except (OSError, subprocess.SubprocessError) as exc:
                    rec["action"] = "commit_failed"
                    rec["reason"] = str(exc)
                    print(f"[apply] git commit warning for {fk}: {exc}")
                if commit_ok:
                    sp._notify_change(fk, symbol, current, new_value, winner, rec)
                    applied_count += 1
            sp._append_audit(rec)

        if not args.dry_run:
            if applied_count > 0:
                try:
                    push_res = subprocess.run(
                        ["git", "-C", ROOT, "push", "-q", "origin", "HEAD:main"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if push_res.returncode != 0:
                        reb_res = subprocess.run(
                            ["git", "-C", ROOT, "pull", "--rebase", "--autostash", "-q", "origin", "main"],
                            capture_output=True, text=True, timeout=30,
                        )
                        if reb_res.returncode != 0:
                            subprocess.run(
                                ["git", "-C", ROOT, "rebase", "--abort"],
                                capture_output=True, text=True, timeout=10,
                            )
                        else:
                            subprocess.run(
                                ["git", "-C", ROOT, "push", "-q", "origin", "HEAD:main"],
                                capture_output=True, text=True, timeout=30,
                            )
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"[apply] git push warning: {exc}")
            print("[apply] done. The owning process restart is performed by watchdogfor_cacheandconfig "
                  "(detects instruments.conf change).")


if __name__ == "__main__":
    main()
