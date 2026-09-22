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
PILOT_KEYS = {
    "BINANCE_BTC.mt.gain": ("BTCUSDC", "BTC", "mt.gain"),
    "BINANCE_BTC.mt.lost": ("BTCUSDC", "BTC", "mt.lost"),
    "BINANCE_TAO.mt.gain": ("TAOUSDC", "TAO", "mt.gain"),
    "BINANCE_TAO.mt.lost": ("TAOUSDC", "TAO", "mt.lost"),
    "BINANCE_BTC.mt.maxage_days": ("BTCUSDC", "BTC", "mt.maxage_days"),
    "BINANCE_TAO.mt.maxage_days": ("TAOUSDC", "TAO", "mt.maxage_days"),
    "BINANCE_BTC.mt.hardtp": ("BTCUSDC", "BTC", "mt.hardtp"),
    "BINANCE_TAO.mt.hardtp": ("TAOUSDC", "TAO", "mt.hardtp"),
    "BINANCE_BTC.mt.hardtp_fraction": ("BTCUSDC", "BTC", "mt.hardtp_fraction"),
    "BINANCE_TAO.mt.hardtp_fraction": ("TAOUSDC", "TAO", "mt.hardtp_fraction"),
}


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


def _split_series(symbol):
    """The available history, split into 2 halves (first/second) — the 2
    INDEPENDENT windows required by the confirmation guardrail."""
    path = os.path.join(ROOT, "cachedb", f"cache_price_{symbol}.jsonl")
    series = load_price_series(path, symbol)
    mid = len(series) // 2
    return series[:mid], series[mid:]


def _run_one(symbol, base, params, series):
    """Runs the REAL backtest logic (structurally identical to
    run_replay_backtest.run_symbol()) over a SERIES given directly (not the file
    whole file from disk) — needed so that the 2 halves can be tested separately
    without reading/splitting the file every time. It does NOT write pnl.json (the pilot
    runs dozens of variants -> it would generate dozens of pointless folders in
    logger/backtest/)."""
    if not series:
        return None
    provider = rb.ReplayMarketDataProvider({symbol: series}, fee_pct=rb.FEE_PCT)
    api = rb.MarketApi([provider])
    inst = rb.Instrument(name=symbol, symbol=symbol, provider="replay",
                         base=base, quote="USDC", params=dict(params), api=api)
    maxage_s = int(float(params["mt.maxage_days"]) * 24 * 3600)

    # 23 Jul: the real is_trend_up() reads the LIVE trend cache (contaminating
    # a historical replay with the REAL current state of the market — see
    # run_replay_backtest._neutral_is_trend_up). Neutralizat identic aici.
    orig_is_trend_up = rb.mt.is_trend_up
    rb.mt.is_trend_up = rb._neutral_is_trend_up
    try:
        first_price = provider.advance(symbol)
        if first_price is None:
            return None
        last_price = first_price
        provider.place_order(symbol, "BUY", first_price, rb.SEED_NOTIONAL_USD / first_price)

        while True:
            price = provider.advance(symbol)
            if price is None:
                break
            last_price = price
            buys = provider.get_orders(symbol, "BUY", since_s=maxage_s)
            sells = provider.get_orders(symbol, "SELL", since_s=maxage_s)
            if not buys and not sells:
                provider.place_order(symbol, "BUY", price, rb.SEED_NOTIONAL_USD / price)
                continue
            try:
                rb.mt.monitor_price_and_trade(inst, sbs=rb.SBS, now_fn=lambda: provider.now(symbol))
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[{symbol}] error in monitor_price_and_trade: {e}\n")
    finally:
        rb.mt.is_trend_up = orig_is_trend_up

    all_buys = provider.get_orders(symbol, "BUY", since_s=1e12)
    all_sells = provider.get_orders(symbol, "SELL", since_s=1e12)
    total_bought = sum(o["qty"] * o["price"] for o in all_buys)
    total_sold = sum(o["qty"] * o["price"] for o in all_sells)
    open_qty, _ = provider.position(symbol)
    open_value = open_qty * last_price
    fees = sum(o["qty"] * o["price"] * rb.FEE_PCT / 100 for o in all_buys + all_sells)
    net = total_sold - total_bought + open_value - fees
    bh_qty = rb.SEED_NOTIONAL_USD / first_price
    buy_hold = (last_price - first_price) * bh_qty - 2 * bh_qty * first_price * rb.FEE_PCT / 100
    return {"net": round(net, 2), "buy_hold": round(buy_hold, 2)}


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

    results = {}
    for v in test_values:
        t0 = time.time()
        params = dict(base_params)
        params[key] = v
        r1 = _run_one(symbol, base, params, half1)
        r2 = _run_one(symbol, base, params, half2)
        sys.stderr.write(f"  [{full_key}] {key}={v}: half1={r1} half2={r2} ({time.time()-t0:.1f}s)\n")
        if r1 is None or r2 is None:
            continue
        results[v] = {"edge_half1": _edge(r1), "edge_half2": _edge(r2), "pnl1": r1, "pnl2": r2}

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
    cur_key = next(v for v in results if abs(float(v) - current) < 1e-9)
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

    # monitor_price_and_trade() is VERY chatty (a print() on every tick) —
    # the pilot runs dozens of variants over hundreds of thousands of ticks, so
    # output suppression is required (otherwise console I/O dominates execution time).
    rb.mt.log.disable_print()

    only_terms = [t.strip() for t in args.only.split(",") if t.strip()]
    keys = {k: v for k, v in PILOT_KEYS.items()
            if not only_terms or any(t in k for t in only_terms)}
    if not keys:
        print(f"[scheduled_pilot] no key contains '{args.only}' -- leaving")
        return

    # Independent keys => run in PARALLEL (ProcessPoolExecutor). Fork on Linux =>
    # each key has its own rb.mt (so the is_trend_up monkeypatch from _run_one
    # stays isolated per process). The audit and the proposals are collected in the parent (no race).
    proposals = []
    max_workers = min(len(keys), os.cpu_count() or 2)
    # stderr (not print/stdout): disable_print() plus the ProcessPool buffering swallow
    # the parent's stdout; the per-value lines already use sys.stderr.write and show up correctly.
    sys.stderr.write(f"[scheduled_pilot] {len(keys)} keys on {max_workers} parallel workers\n")
    with _futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2key = {ex.submit(evaluate_key, fk, sym, base, key, args.dry_run, args.propose): fk
                   for fk, (sym, base, key) in keys.items()}
        entries = {}
        for fut in _futures.as_completed(fut2key):
            entries[fut2key[fut]] = fut.result()
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
        if proposals:
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
