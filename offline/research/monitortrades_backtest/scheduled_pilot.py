#!/usr/bin/env python3
"""
scheduled_pilot.py — A PILOT (a single module: monitortrades.py, a user decision on
23 Jul: "for now we run a pilot test and do not extend it to all modules").

It reads the test ranges from the "# BACKTEST: ..." annotations written DIRECTLY in
instruments.conf (offline/research/backtest_ranges.py, plain text — NOT YAML/JSON,
an explicit user decision), runs a REAL backtest (offline/research/monitortrades_backtest/
run_replay_backtest.py, over cache_price_{symbol}.jsonl) for each value
from the grid, and RECONFIGURES + RESTARTS monitortrades.py if it finds a
value is clearly better — with explicit guardrails (the user asked for complete
automation, but after it was shown today that the same max_budget=5000 gave +$3016 in
one configuration and -$5279 in another, on the SAME history):

  1. CONFIRMATION ON 2 INDEPENDENT WINDOWS: the available history is split in
     half (first/second) — a value is considered a "winner" ONLY
     if it has the best edge against buy & hold (net - buy_hold) in BOTH
     halves, not just one. A result that wins on only one window is
     treated as noise, not signal.
  2. ONLY values from the grid (never extrapolation).
  3. AVERAGING, not a direct jump (a user decision): the value actually applied =
     (value_configured_today + winning_backtest_value) / 2 —
     it damps exactly the kind of instability demonstrated today.
  4. RATE LIMIT: a parameter does not change more often than once every
     PILOT_MIN_DAYS_BETWEEN_CHANGES days (7 by default) — a persistent journal.
  5. AUDIT: every run writes a line into the journal (what was tested, both windows,
     decision, reason) — whether or not anything changed.
  6. NOTIFICATION (alertnotifiers.notify, the channel the fleet already uses): ONLY
     when a real change is applied (not on every "nothing new" run).
  7. KILL SWITCH: the env var PILOT_DISABLED=true stops EVERYTHING, without touching the code.

Running it by hand (recommended before putting it on cron):
    python3 offline/research/monitortrades_backtest/scheduled_pilot.py --dry-run
    python3 offline/research/monitortrades_backtest/scheduled_pilot.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.research.backtest_ranges import scan_backtest_ranges
from offline.research.monitortrades_backtest import run_replay_backtest as rb
from providers.replay_provider import load_price_series

INSTRUMENTS_CONF = os.path.join(ROOT, "instruments.conf")
AUDIT_LOG = os.path.join(ROOT, "logger", "backtest_pilot_audit.jsonl")
MIN_DAYS_BETWEEN_CHANGES = float(os.environ.get("PILOT_MIN_DAYS_BETWEEN_CHANGES", "7"))
# The winner must beat the CURRENT value by at least this much (net-buy_hold,
# USD) on BOTH windows. Otherwise the change is below the noise, or the parameter is inert on
# this history (e.g. found 28 Jul: hard-TP never fires -> every value gives
# identical results, and max() on a tie falsely "applies" the first element of the grid).
MIN_EDGE_MARGIN_USD = float(os.environ.get("PILOT_MIN_EDGE_MARGIN_USD", "1.0"))

# The keys in the pilot's scope (BTC/TAO, monitortrades) — the rest of the annotations in
# instruments.conf are NOT touched without explicitly extending this list (a user
# decision: the pilot is limited to monitortrades, not all modules). They all live as
# mt.* in instruments.conf, so run_replay_backtest already fires them through params
# (verified: is_trend_up neutralised, maxage->since_s, hardtp->inst.param).
#   #4-5  gain/lost   (DONE 23 Jul: TAO mt.lost 4.9->5.25 applied)
#   #16   maxage_days (28 iul)
#   #15   hardtp / hardtp_fraction (28 iul; per-instrument, monitortrades.py:447)
def _build_pilot_keys(conf_path=INSTRUMENTS_CONF):
    """Derive pilot keys dynamically from instruments.conf (SSOT role.mt=yes)."""
    from instrument_registry import select_instruments
    keys = {}
    mt_params = (
        "mt.gain",
        "mt.lost",
        "mt.maxage_days",
        "mt.hardtp",
        "mt.hardtp_fraction",
        "mt.hardtp_cooldown_h",
    )
    for spec in select_instruments(role="mt", path=conf_path).values():
        for p in mt_params:
            if p in spec.params:
                keys[f"{spec.name}.{p}"] = (spec.symbol, spec.base, p)
        if spec.has_role("trailing") and "trailing.pct" in spec.params:
            keys[f"{spec.name}.trailing.pct"] = (spec.symbol, spec.base, "trailing.pct")
    return keys


PILOT_KEYS = _build_pilot_keys()


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _git_head_short():
    """The (short) commit of the code that produced the proposal — so that prod knows
    exactly which engine/config version produced it. '?' if git does not answer."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _append_audit(entry):
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
    entry = dict(entry, ts=_now_iso())
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _last_change_for(full_key):
    """The most recent journal entry with action='applied' for full_key."""
    if not os.path.exists(AUDIT_LOG):
        return None
    last = None
    with open(AUDIT_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("full_key") == full_key and entry.get("action") == "applied":
                last = entry
    return last


def _current_value(section, key):
    """Read today's LIVE value (not the grid) directly from instruments.conf."""
    import configparser
    cp = configparser.ConfigParser()
    cp.read(INSTRUMENTS_CONF)
    return float(cp[section][key])


def _edge(pnl):
    """net - buy_hold — a measure of "how much better than simply holding",
    comparable between different windows (unlike the raw net)."""
    return pnl["net"] - pnl["buy_hold"]


def _num_str(x):
    """A compact representation of a numeric value for the grid (17.0->'17',
    0.5->'0.5', 5.25->'5.25') — used when the LIVE value is added to the test set."""
    return "%g" % x


_SERIES_SPLIT_CACHE: dict[str, tuple[list, list]] = {}
_RUN_ONE_CACHE: dict[tuple, dict | None] = {}


def _norm_params_key(params: dict) -> tuple:
    items = []
    for k, v in sorted(params.items()):
        try:
            items.append((k, float(v)))
        except (TypeError, ValueError):
            items.append((k, str(v)))
    return tuple(items)


def _run_one_cache_key(symbol: str, params: dict, series: list) -> tuple:
    return (
        str(symbol),
        _norm_params_key(params),
        len(series),
        series[0] if series else None,
        series[-1] if series else None,
    )


def _split_series(symbol):
    """The available history, split into 2 halves (first/second) — the 2
    INDEPENDENT windows required by the confirmation guardrail."""
    cached = _SERIES_SPLIT_CACHE.get(symbol)
    if cached is not None:
        return cached
    path = os.path.join(ROOT, "cachedb", f"cache_price_sparse_{symbol}.jsonl")
    series = load_price_series(path, symbol)
    mid = len(series) // 2
    res = (series[:mid], series[mid:])
    if hasattr(rb, "prewarm_trend_series"):
        rb.prewarm_trend_series(symbol, res[0])
        rb.prewarm_trend_series(symbol, res[1])
    _SERIES_SPLIT_CACHE[symbol] = res
    return res


def _run_one(symbol, base, params, series):
    """Run the canonical replay backtest (run_replay_backtest.run_symbol) over a
    pre-sliced in-memory series without writing pnl.json."""
    if not series:
        return None
    ck = _run_one_cache_key(symbol, params, series)
    if ck in _RUN_ONE_CACHE:
        cached = _RUN_ONE_CACHE[ck]
        return dict(cached) if cached is not None else None
    res = rb.run_symbol(symbol, params, base, quiet=True, series=series)
    out = {"net": res["net"], "buy_hold": res["buy_hold"]} if res else None
    _RUN_ONE_CACHE[ck] = out
    return dict(out) if out is not None else None


def _eval_one_half(task: tuple[str, str, dict, int]):
    symbol, base, params, half_idx = task
    halves = _split_series(symbol)
    series = halves[half_idx]
    if len(series) < 100:
        return symbol, params, half_idx, None, {}
    res = _run_one(symbol, base, params, series)
    tk = rb._series_cache_key(symbol, series, rb.DEFAULT_WINDOW_SECONDS)
    diag = dict(
        rb._LAST_RUN_DIAGNOSTICS.get(tk)
        or {"hard_tp_touched": rb._LAST_HARD_TP_TOUCHED.get(tk, True)}
    )
    return symbol, params, half_idx, res, diag


def _eval_symbol_baseline(symbol: str, base: str):
    half1, half2 = _split_series(symbol)
    if len(half1) < 100 or len(half2) < 100:
        return symbol, None, None, None, {}, {}
    base_params = dict(rb.SYMBOLS[symbol]["params"])
    r1 = _run_one(symbol, base, base_params, half1)
    k1 = rb._series_cache_key(symbol, half1, rb.DEFAULT_WINDOW_SECONDS)
    diag1 = dict(rb._LAST_RUN_DIAGNOSTICS.get(k1) or {"hard_tp_touched": rb._LAST_HARD_TP_TOUCHED.get(k1, True)})
    r2 = _run_one(symbol, base, base_params, half2)
    k2 = rb._series_cache_key(symbol, half2, rb.DEFAULT_WINDOW_SECONDS)
    diag2 = dict(rb._LAST_RUN_DIAGNOSTICS.get(k2) or {"hard_tp_touched": rb._LAST_HARD_TP_TOUCHED.get(k2, True)})
    return symbol, base_params, r1, r2, diag1, diag2


def evaluate_key(full_key, symbol, base, key, dry_run=True, propose=False):
    grid_values = scan_backtest_ranges(INSTRUMENTS_CONF).get(full_key)
    if not grid_values:
        return {"full_key": full_key, "action": "skipped", "reason": "no_grid_annotation"}

    section = full_key.split(".", 1)[0]
    current = _current_value(section, key)
    base_params = dict(rb.SYMBOLS[symbol]["params"])

    half1, half2 = _split_series(symbol)
    if len(half1) < 100 or len(half2) < 100:
        return {"full_key": full_key, "action": "skipped", "reason": "not enough history for 2 windows"}

    # Make sure the current LIVE value is among those tested: without it we cannot
    # compares the winner AGAINST it, and an INERT parameter (every value giving
    # identical results) would be falsely "applied" to the first element of the grid
    # (max() on a tie returns the first). Found 28 Jul at #15 hardtp.
    test_values = list(grid_values)
    if not any(abs(float(v) - current) < 1e-9 for v in test_values):
        test_values.append(_num_str(current))

    eval_order = sorted(test_values, key=float) if key == "mt.hardtp" else test_values
    invariant_r1 = None
    invariant_max_htp1 = -1.0
    invariant_r2 = None
    invariant_max_htp2 = -1.0
    raw_results = {}
    for v in eval_order:
        t0 = time.time()
        params = dict(base_params)
        params[key] = v
        v_ratio = float(v) / 100.0 if key == "mt.hardtp" else 0.0
        if key == "mt.hardtp" and invariant_r1 is not None and v_ratio <= invariant_max_htp1:
            r1 = dict(invariant_r1)
            _RUN_ONE_CACHE[_run_one_cache_key(symbol, params, half1)] = r1
        else:
            r1 = _run_one(symbol, base, params, half1)
            if key == "mt.hardtp" and r1 is not None:
                tk1 = rb._series_cache_key(symbol, half1, rb.DEFAULT_WINDOW_SECONDS)
                d1 = rb._LAST_RUN_DIAGNOSTICS.get(tk1) or {}
                if not rb._LAST_HARD_TP_TOUCHED.get(tk1, True):
                    invariant_r1 = r1
                    invariant_max_htp1 = float("inf")
                elif d1.get("min_hard_tp_gain_ratio", 0.0) >= v_ratio:
                    invariant_r1 = r1
                    invariant_max_htp1 = float(d1["min_hard_tp_gain_ratio"])
        if key == "mt.hardtp" and invariant_r2 is not None and v_ratio <= invariant_max_htp2:
            r2 = dict(invariant_r2)
            _RUN_ONE_CACHE[_run_one_cache_key(symbol, params, half2)] = r2
        else:
            r2 = _run_one(symbol, base, params, half2)
            if key == "mt.hardtp" and r2 is not None:
                tk2 = rb._series_cache_key(symbol, half2, rb.DEFAULT_WINDOW_SECONDS)
                d2 = rb._LAST_RUN_DIAGNOSTICS.get(tk2) or {}
                if not rb._LAST_HARD_TP_TOUCHED.get(tk2, True):
                    invariant_r2 = r2
                    invariant_max_htp2 = float("inf")
                elif d2.get("min_hard_tp_gain_ratio", 0.0) >= v_ratio:
                    invariant_r2 = r2
                    invariant_max_htp2 = float(d2["min_hard_tp_gain_ratio"])
        sys.stderr.write(f"  [{full_key}] {key}={v}: half1={r1} half2={r2} ({time.time()-t0:.1f}s)\n")
        if r1 is None or r2 is None:
            continue
        raw_results[v] = {"edge_half1": _edge(r1), "edge_half2": _edge(r2), "pnl1": r1, "pnl2": r2}
    results = {v: raw_results[v] for v in test_values if v in raw_results}

    if not results:
        return {"full_key": full_key, "action": "skipped", "reason": "backtest without results"}

    winner_half1 = max(results, key=lambda v: results[v]["edge_half1"])
    winner_half2 = max(results, key=lambda v: results[v]["edge_half2"])

    entry = {
        "full_key": full_key, "symbol": symbol, "current_value": current,
        "grid": grid_values, "results": results,
        "winner_half1": winner_half1, "winner_half2": winner_half2,
    }

    if winner_half1 != winner_half2:
        entry["action"] = "no_change"
        entry["reason"] = f"unconfirmed: a different winner on the 2 windows ({winner_half1} vs {winner_half2})"
        return entry

    winner = winner_half1
    winner_val = float(winner)
    if abs(winner_val - current) < 1e-9:
        entry["action"] = "no_change"
        entry["reason"] = "the winning value = the value already configured"
        return entry

    # The winner differs from the current value -> it has to BEAT it by a
    # significant margin on BOTH windows. A margin of ~0 means the parameter is inert on this history
    # (e.g. a hard TP that never triggered), and the "winner" is merely a tie-break artefact.
    cur_key = next(
        (v for v in results if math.isclose(float(v), current, rel_tol=1e-5, abs_tol=1e-6)),
        None,
    )
    if cur_key is None or cur_key not in results:
        entry["action"] = "no_change"
        entry["reason"] = f"baseline current value {current} has no backtest results"
        return entry
    margin_h1 = results[winner]["edge_half1"] - results[cur_key]["edge_half1"]
    margin_h2 = results[winner]["edge_half2"] - results[cur_key]["edge_half2"]
    entry["margin_vs_current"] = {"half1": round(margin_h1, 2), "half2": round(margin_h2, 2)}
    if margin_h1 < MIN_EDGE_MARGIN_USD or margin_h2 < MIN_EDGE_MARGIN_USD:
        entry["action"] = "no_change"
        entry["reason"] = (f"negligible gain versus the current value {current} "
                            f"(+{margin_h1:.2f}/+{margin_h2:.2f} < {MIN_EDGE_MARGIN_USD} USD) "
                            f"— the parameter may be inert on this history")
        return entry

    # Mod DEV (--propose): castigatorul a trecut de guardrail-uri (confirmat pe 2
    # windows plus beating the current value by a margin). We do NOT apply or rate-limit here —
    # we propose the RAW winning value; PROD decides at application time (averaging with
    # ITS live value, a rate limit, an audit). See UNIFIED_BACKTEST_PLAN.md §9.
    if propose:
        entry["winner_value"] = winner_val
        entry["action"] = "proposed"
        return entry

    last_change = _last_change_for(full_key)
    if last_change:
        last_ts = datetime.fromisoformat(last_change["ts"])
        if datetime.now() - last_ts < timedelta(days=MIN_DAYS_BETWEEN_CHANGES):
            entry["action"] = "rate_limited"
            entry["reason"] = (f"last changed at {last_change['ts']}, "
                                f"waiting {MIN_DAYS_BETWEEN_CHANGES} days between changes")
            return entry

    new_value = round((current + winner_val) / 2, 4)
    entry["proposed_new_value"] = new_value
    entry["action"] = "would_apply" if dry_run else "applied"

    if not dry_run:
        _apply_config_change(section, key, current, new_value)
        _restart_monitortrades()
        _notify_change(full_key, symbol, current, new_value, winner_val, entry)

    return entry


def _apply_config_change(section, key, old_value, new_value):
    """Replace ONLY the numeric value on the `key = ...` line in
    sectiunea `section`, pastrand tot restul fisierului (comentarii,
    formatare, alte sectiuni) neatins."""
    with open(INSTRUMENTS_CONF, encoding="utf-8") as f:
        lines = f.readlines()
    in_section = False
    key_re = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)([^\s#]+)(.*)$')
    for i, line in enumerate(lines):
        sm = re.match(r'^\s*\[([^\]]+)\]\s*$', line)
        if sm:
            in_section = (sm.group(1) == section)
            continue
        if in_section:
            m = key_re.match(line)
            if m:
                lines[i] = f"{m.group(1)}{new_value}{m.group(3)}\n"
                break
    with open(INSTRUMENTS_CONF, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _restart_monitortrades():
    """Kills the live process — the flota_start.sh supervisor restarts it
    automatically (the same mechanism used by hand throughout the session)."""
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", "python monitortrades.py"],
                              capture_output=True, text=True, timeout=5)
        pids = [p for p in out.stdout.split() if p.isdigit()]
        for pid in pids:
            subprocess.run(["kill", pid], timeout=5)
    except Exception as e:  # noqa: BLE001 — it does not stop the journalling or the notification
        print(f"[scheduled_pilot] error restarting monitortrades: {e}")


def _notify_change(full_key, symbol, old_value, new_value, winner_val, entry):
    try:
        import alertnotifiers as alert
        body = (f"{full_key}: {old_value} -> {new_value} "
                f"(a backtest winner confirmed on 2 windows: {winner_val}, "
                f"averaged with the old value)")
        alert.notify(title="Pilot backtest: config changed", body=body,
                     source="scheduled_pilot.py", symbol=symbol)
    except Exception as e:  # noqa: BLE001
        print(f"[scheduled_pilot] notification error: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="evaluate and report, but do NOT change the config or restart the bot")
    ap.add_argument("--only", default="",
                     help="run ONLY the keys containing one of these substrings "
                          "separated by commas (e.g. 'maxage,hardtp' or 'BINANCE_TAO'); empty = all")
    ap.add_argument("--propose", action="store_true",
                     help="DEV mode: do NOT apply or restart; write confirmed proposals "
                          "to --propose-out for git dev->prod pipeline")
    ap.add_argument("--propose-out", default=os.path.join(ROOT, "backtest_proposals.json"),
                     help="where to write the proposals in --propose mode (default backtest_proposals.json)")
    args = ap.parse_args()

    if os.environ.get("PILOT_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        print("[scheduled_pilot] PILOT_DISABLED=true -- leaving without doing anything")
        return

    lock_file = None
    try:
        import fcntl
        lock_path = (
            "/tmp/mptrade_scheduled_pilot_dryrun.lock"
            if (args.dry_run and not args.propose)
            else "/tmp/mptrade_scheduled_pilot.lock"
        )
        lock_file = open(lock_path, "w", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("[scheduled_pilot] Another instance is already running -- exiting\n")
        return
    except Exception:  # noqa: BLE001
        pass

    # monitor_price_and_trade() is VERY chatty (a print() on every tick) —
    # the pilot runs dozens of variants over hundreds of thousands of ticks, so
    # output suppression is required (otherwise console I/O dominates execution time).
    import log
    log.disable_print()

    only_terms = [t.strip() for t in args.only.split(",") if t.strip()]
    keys = {k: v for k, v in PILOT_KEYS.items()
            if not only_terms or any(t in k for t in only_terms)}
    if not keys:
        print(f"[scheduled_pilot] no key contains '{args.only}' -- leaving")
        return

    # Preload price series and trend tapes once in parent process so forked workers share them via COW.
    unique_sym_base = sorted({(sym, base) for sym, base, _key in keys.values()})
    for sym, _base in unique_sym_base:
        _split_series(sym)

    # Precompute baseline (base_params) once per symbol half across workers so all keys per symbol
    # reuse the cached baseline and balanced workers evaluate both halves concurrently.
    max_workers = os.cpu_count() or 2
    all_ranges = scan_backtest_ranges(INSTRUMENTS_CONF)
    baseline_tasks = []
    for sym, base in unique_sym_base:
        half1, half2 = _split_series(sym)
        if len(half1) >= 100 and len(half2) >= 100:
            bp = dict(rb.SYMBOLS[sym]["params"])
            baseline_tasks.append((sym, base, bp, 0, len(half1)))
            baseline_tasks.append((sym, base, bp, 1, len(half2)))
    baseline_tasks.sort(key=lambda t: -t[4])

    base_results: dict[str, dict] = {}
    if baseline_tasks:
        with _futures.ProcessPoolExecutor(max_workers=min(len(baseline_tasks), max_workers)) as pre_ex:
            for sym, bp, half_idx, res, diag in pre_ex.map(
                _eval_one_half,
                [(s, b, p, h) for s, b, p, h, _ in baseline_tasks],
                chunksize=1,
            ):
                entry_b = base_results.setdefault(sym, {"base_params": bp, "r": [None, None], "diag": [{}, {}]})
                entry_b["r"][half_idx] = res
                entry_b["diag"][half_idx] = diag

    for sym, entry_b in base_results.items():
        base_params = entry_b["base_params"]
        r1, r2 = entry_b["r"]
        diag1, diag2 = entry_b["diag"]
        if base_params is not None:
            half1, half2 = _split_series(sym)
            _RUN_ONE_CACHE[_run_one_cache_key(sym, base_params, half1)] = r1
            _RUN_ONE_CACHE[_run_one_cache_key(sym, base_params, half2)] = r2
            htp1 = diag1.get("hard_tp_touched", True)
            htp2 = diag2.get("hard_tp_touched", True)
            base_htp = float(base_params.get("mt.hardtp") or 0.0)
            base_cd_s = float(base_params.get("mt.hardtp_cooldown_h") or 6.0) * 3600.0
            base_tr = float(base_params.get("trailing.pct") or 0.0)
            base_tr_r = base_tr / 100.0
            for fk, (k_sym, _k_base, k_name) in keys.items():
                if k_sym != sym:
                    continue
                if k_name == "mt.hardtp_fraction":
                    for gv in all_ranges.get(fk, ()):
                        p = dict(base_params, **{k_name: gv})
                        if not htp1:
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half1)] = r1
                        if not htp2:
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half2)] = r2
                elif k_name == "mt.hardtp_cooldown_h":
                    dts1 = diag1.get("hard_tp_cooldown_dts", ())
                    dts2 = diag2.get("hard_tp_cooldown_dts", ())
                    for gv in all_ranges.get(fk, ()):
                        cand_cd_s = float(gv) * 3600.0
                        p = dict(base_params, **{k_name: gv})
                        if not htp1 or all((dt < cand_cd_s) == (dt < base_cd_s) for dt in dts1):
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half1)] = r1
                        if not htp2 or all((dt < cand_cd_s) == (dt < base_cd_s) for dt in dts2):
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half2)] = r2
                elif k_name == "mt.hardtp" and base_htp > 0:
                    min_g1 = diag1.get("min_hard_tp_gain_ratio", 0.0)
                    min_g2 = diag2.get("min_hard_tp_gain_ratio", 0.0)
                    for gv in all_ranges.get(fk, ()):
                        gv_f = float(gv)
                        gv_r = gv_f / 100.0
                        if gv_f >= base_htp:
                            p = dict(base_params, **{k_name: gv})
                            if not htp1 or min_g1 >= gv_r:
                                _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half1)] = r1
                            if not htp2 or min_g2 >= gv_r:
                                _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half2)] = r2
                elif k_name == "trailing.pct" and base_tr > 0:
                    min_tr1 = diag1.get("min_trail_trigger_ratio", 0.0)
                    max_untr1 = diag1.get("max_untriggered_trail_ratio", 1.0)
                    min_tr2 = diag2.get("min_trail_trigger_ratio", 0.0)
                    max_untr2 = diag2.get("max_untriggered_trail_ratio", 1.0)
                    for gv in all_ranges.get(fk, ()):
                        gv_r = float(gv) / 100.0
                        p = dict(base_params, **{k_name: gv})
                        if (gv_r >= base_tr_r and gv_r <= min_tr1) or (gv_r <= base_tr_r and gv_r > max_untr1):
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half1)] = r1
                        if (gv_r >= base_tr_r and gv_r <= min_tr2) or (gv_r <= base_tr_r and gv_r > max_untr2):
                            _RUN_ONE_CACHE[_run_one_cache_key(sym, p, half2)] = r2

    # Pre-evaluate all remaining uncached (symbol, base, params, half_idx) tasks across the worker pool,
    # sorted by descending series length so no single worker is left with a long tail.
    pending_tasks = []
    seen_task_cks = set()
    for fk, (sym, base, key) in keys.items():
        grid_values = all_ranges.get(fk)
        if not grid_values:
            continue
        half1, half2 = _split_series(sym)
        if len(half1) < 100 or len(half2) < 100:
            continue
        section = fk.split(".", 1)[0]
        current = _current_value(section, key)
        base_params = dict(rb.SYMBOLS[sym]["params"])
        test_values = list(grid_values)
        if not any(abs(float(v) - current) < 1e-9 for v in test_values):
            test_values.append(_num_str(current))
        for v in test_values:
            p = dict(base_params, **{key: v})
            for half_idx, h_series in ((0, half1), (1, half2)):
                ck = _run_one_cache_key(sym, p, h_series)
                if ck not in _RUN_ONE_CACHE and ck not in seen_task_cks:
                    seen_task_cks.add(ck)
                    pending_tasks.append((sym, base, p, half_idx, len(h_series)))
    pending_tasks.sort(key=lambda t: -t[4])

    proposals = []
    sys.stderr.write(
        f"[scheduled_pilot] {len(keys)} keys ({len(pending_tasks)} uncached half-window runs) "
        f"on {max_workers} parallel workers\n"
    )
    if pending_tasks:
        with _futures.ProcessPoolExecutor(max_workers=min(len(pending_tasks), max_workers)) as ex:
            for sym, p, half_idx, res, diag in ex.map(
                _eval_one_half,
                [(s, b, prm, h) for s, b, prm, h, _ in pending_tasks],
                chunksize=1,
            ):
                h_series = _split_series(sym)[half_idx]
                _RUN_ONE_CACHE[_run_one_cache_key(sym, p, h_series)] = res
                tk = rb._series_cache_key(sym, h_series, rb.DEFAULT_WINDOW_SECONDS)
                rb._LAST_RUN_DIAGNOSTICS[tk] = diag
                rb._LAST_HARD_TP_TOUCHED[tk] = diag.get("hard_tp_touched", True)

    entries = {}
    for fk, (sym, base, key) in keys.items():
        try:
            entries[fk] = evaluate_key(fk, sym, base, key, args.dry_run, args.propose)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[scheduled_pilot] Error evaluating {fk}: {exc}\n")
            entries[fk] = {"full_key": fk, "action": "error", "reason": str(exc)}
    for full_key, (symbol, base, key) in keys.items():   # Processing in a stable order.
        entry = entries[full_key]
        _append_audit(entry)
        sys.stderr.write(f"=== {full_key} ===\n"
                         + json.dumps({k: v for k, v in entry.items() if k != "results"},
                                      indent=2, default=str) + "\n")
        if args.propose and entry.get("action") == "proposed":
            proposals.append({
                "ts": _now_iso(), "full_key": full_key,
                "section": full_key.split(".", 1)[0], "key": key, "symbol": symbol,
                "current_on_dev": entry["current_value"],
                "winner_value": entry["winner_value"],
                "margin_vs_current": entry.get("margin_vs_current"),
                "dev_commit": _git_head_short(),
            })

    if args.propose:
        # a snapshot of the current proposals (overwriting — the file means "what dev proposes
        # NOW"; prod consumes them and applies them with its own guardrails). Empty = no
        # confirmed signal in this cycle (prod has nothing to apply).
        with open(args.propose_out, "w", encoding="utf-8") as f:
            json.dump(proposals, f, indent=2, default=str)
        print(f"\n[scheduled_pilot] {len(proposals)} proposal(s) written to {args.propose_out}")
        if proposals and os.environ.get("PILOT_NOTIFY_PROPOSALS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            # notify user that new proposals are available to review and apply
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(ROOT, ".env"))
                load_dotenv(os.path.join(ROOT, "config.env"))
                import alertnotifiers as alert
                lines = [f"{p['full_key']}: {p['current_on_dev']} -> winner {p['winner_value']}"
                         for p in proposals]
                alert.notify(title=f"Backtest: {len(proposals)} new config proposal(s)",
                             body="Run offline/runners/apply_proposals.py on prod (with guardrails):\n"
                                  + "\n".join(lines),
                             source="scheduled_pilot.py", symbol="backtest")
            except Exception as e:  # noqa: BLE001
                print(f"[scheduled_pilot] proposals notification error: {e}")


if __name__ == "__main__":
    main()
