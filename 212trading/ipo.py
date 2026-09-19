#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
ipo.py — generic Trading 212 watcher and auto-trader for any symbol configured in .env.

Configure the instrument in .env; SPCX is no longer hard-coded:
    T212_TICKER=NVDA_US_EQ     exact required T212 instrument
    YAHOO_SYMBOL=NVDA          optional Yahoo price symbol, derived from T212_TICKER
    SYMBOL_LABEL=NVIDIA        optional display/notification label
    WAIT_FOR_LAUNCH=false      false for existing trading; true waits for an IPO launch
    EXPECTED_ISIN=US67066G1040 optional; reject an ISIN mismatch

Two trading modes after launch or immediately:
    STRAT_ENABLED=true   -> DCA + take-profit strategy; see strategy.py
    STRAT_ENABLED=false  -> one LIMIT order; see ORDER_* and order_manager.py

Commands:
    python3 ipo.py                          # run using .env configuration
    python3 ipo.py --paper                  # force safe PAPER mode
    python3 ipo.py --symbol NVDA_US_EQ      # override the instrument for this run
    python3 ipo.py --test-notify all        # test notifications
    python3 ipo.py --test-order NVDA_US_EQ  # test one order
    python3 ipo.py --find-ticker nvidia     # find the exact T212 ticker
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ipo_common import (
    load_t212_environment, log, now_str, float_env, ET, required_env,
    required_float_env, required_bool_env,
)
from market_data import check_market, t212_to_yahoo
from t212_client import T212Client
from ipo_notify import notify
from order_manager import resolve_quantity, place_order_with_retry
from strategy import Strategy, StratParams
from credentials import t212_credentials

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def in_market_window() -> bool:
    """Return True between approximately 09:00 and 16:30 ET on weekdays."""
    from datetime import datetime
    n = datetime.now(ET)
    if n.weekday() >= 5:
        return False
    minutes = n.hour * 60 + n.minute
    return 9 * 60 <= minutes <= 16 * 60 + 30


def verify_instrument(client: T212Client, ticker: str, expected_isin: str) -> bool:
    """Best-effort verification that a configured EXPECTED_ISIN matches the ticker.
    Return False only for a proven mismatch identifying the wrong instrument."""
    if not expected_isin:
        return True
    instruments = client.list_instruments()
    if not instruments:
        log("  ! cannot verify the ISIN (metadata unavailable) — continuing on the explicit ticker")
        return True
    match = next((i for i in instruments if str(i.get("ticker", "")).upper() == ticker.upper()), None)
    if not match:
        log(f"  ! {ticker} does not appear in the T212 metadata yet — continuing (it may be a delay)")
        return True
    if str(match.get("isin", "")) != expected_isin:
        log(f"  ! ISIN {match.get('isin')} != asteptat {expected_isin} — OPRESC (instrument gresit).")
        notify(title=f"⚠ {ticker}: ISIN nepotrivit!",
               body=f"Found isin={match.get('isin')}, expected {expected_isin}. Not trading.",
               source="verify")
        return False
    log(f"  [verify] {ticker} confirmat (isin {expected_isin})")
    return True


# ---------------------------------------------------------------------------
# Start trading through the strategy or a single order
# ---------------------------------------------------------------------------
def start_trading(client, t212_ticker, label, strat_enabled, strat_dry,
                  order_price, order_qty, order_budget_ron, order_validity,
                  order_dry, desktop) -> int:
    if strat_enabled:
        log(f"  Pornesc STRATEGIA pe {t212_ticker} ({'PAPER' if strat_dry else '⚠ REAL'})")
        Strategy(client, t212_ticker, StratParams.from_env(),
                 dry_run=strat_dry, desktop=desktop).run()
        return 0

    if order_price:
        qty = resolve_quantity(order_price, order_qty, order_budget_ron)
        if not qty or qty <= 0:
            log("  ! invalid qty/budget — the order was NOT SENT")
            return 1
        ok = place_order_with_retry(client, t212_ticker, qty, order_price,
                                    order_validity, order_dry, desktop=desktop)
        return 0 if ok else 1

    log("  ! neither STRAT_ENABLED nor ORDER_PRICE — nothing to trade.")
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    env_file = os.environ.get("ENV_FILE", os.path.join(here, ".env"))
    profile = os.environ.get("IPO_PROFILE")
    for i, a in enumerate(sys.argv):
        if a == "--env-file" and i + 1 < len(sys.argv):
            env_file = sys.argv[i + 1]
        if a in ("--profile", "-p") and i + 1 < len(sys.argv):
            profile = sys.argv[i + 1]
    profile_file = None
    if profile:                                            # versioned per-profile config: config.<profile>.env
        cfg_dir = os.path.dirname(env_file) or "."
        cfg = os.path.join(cfg_dir, f"config.{profile}.env")
        if not os.path.exists(cfg):
            avail = sorted(os.path.basename(p)[7:-4]
                           for p in glob.glob(os.path.join(cfg_dir, "config.*.env")))
            log(f"! unknown profile '{profile}' (is missing {cfg})")
            log(f"  profile disponibile: {', '.join(avail) or '(niciunul)'}")
            return 2
        profile_file = cfg
    load_t212_environment(env_file, profile_file=profile_file)

    ap = argparse.ArgumentParser(description="Watcher + auto-trade generic pe T212.")
    ap.add_argument("--profile", "-p",     required=True, metavar="NUME",
                    help="REQUIRED: the config profile -> it loads config.<NAME>.env (e.g. spcx, nvda)")
    ap.add_argument("--env-file",          default=env_file)
    ap.add_argument("--symbol",            metavar="T212_TICKER",
                    help="Override the instrument (otherwise T212_TICKER from .env)")
    ap.add_argument("--interval",          type=float,
                    default=required_float_env("POLL_SECONDS"))
    ap.add_argument("--desktop",           action="store_true")
    ap.add_argument("--market-hours-only", action="store_true")
    ap.add_argument("--skip-wait",         action="store_true",
                    help="Skip the launch check (start straight away even if the feed says it is not trading yet)")
    ap.add_argument("--paper",             action="store_true",
                    help="Force PAPER (a safe test, no money), whatever .env says")
    ap.add_argument("--execute",           action="store_true",
                    help="Override: a real order (single-order mode)")
    ap.add_argument("--test-notify",       choices=["market", "trade", "all"], metavar="WHAT")
    ap.add_argument("--test-order",        metavar="T212_TICKER",
                    help="Test a single order on the given ticker and exit")
    ap.add_argument("--test-strategy",     metavar="T212_TICKER",
                    help="Run the strategy NOW on the given ticker (paper if STRAT_EXECUTE!=true)")
    ap.add_argument("--find-ticker",       metavar="NUME",
                    help="Search for an instrument on T212 by name or symbol")
    args = ap.parse_args()
    # The one-shot lifecycle marker is namespaced by profile. Preserve the CLI
    # identity explicitly so parallel profiles/accounts never share one marker.
    os.environ["IPO_PROFILE"] = args.profile.strip()

    # --- .env configuration ---
    credentials = t212_credentials()
    t212_env    = required_env("T212_ENV").lower()
    client = T212Client(
        credentials.key, credentials.secret, env=t212_env,
        min_gap_sec=required_float_env("T212_MIN_GAP_SEC"),
        portfolio_ttl_sec=required_float_env("T212_PORTFOLIO_TTL_SEC"),
    )

    # Generic instrument.
    t212_ticker  = (args.symbol or os.environ.get("T212_TICKER") or "").strip()
    yahoo_symbol = os.environ.get("YAHOO_SYMBOL") or (t212_to_yahoo(t212_ticker) if t212_ticker else "")
    label        = os.environ.get("SYMBOL_LABEL") or yahoo_symbol or t212_ticker
    expected_isin = os.environ.get("EXPECTED_ISIN", "").strip()

    # Strategy / order.
    strat_enabled = required_bool_env("STRAT_ENABLED")
    strat_dry     = args.paper or not required_bool_env("STRAT_EXECUTE")
    order_price      = float_env("ORDER_PRICE")
    order_qty        = float_env("ORDER_QTY")
    order_budget_ron = float_env("ORDER_BUDGET_RON")
    _val             = required_env("ORDER_VALIDITY").upper()
    if _val not in {"DAY", "GTC", "GOOD_TILL_CANCEL"}:
        raise ValueError(f"Invalid ORDER_VALIDITY: {_val!r}")
    order_validity   = "GOOD_TILL_CANCEL" if _val in ("GTC", "GOOD_TILL_CANCEL") else "DAY"
    order_dry        = args.paper or not (args.execute or required_bool_env("ORDER_EXECUTE"))
    interval         = max(args.interval, 30)

    # --- one-shot commands ---
    if args.find_ticker:
        return _cmd_find_ticker(client, args.find_ticker)
    if args.test_notify:
        return _cmd_test_notify(args.test_notify, label, args.desktop)
    if args.test_order:
        return _cmd_test_order(client, args.test_order, order_price, order_qty,
                               order_budget_ron, order_validity, order_dry, args.desktop)
    if args.test_strategy:
        log(f"[TEST STRATEGY] {args.test_strategy}  {'PAPER' if strat_dry else '⚠ REAL'}")
        Strategy(client, args.test_strategy, StratParams.from_env(),
                 dry_run=strat_dry, desktop=args.desktop).run()
        return 0

    # --- banner ---
    if not t212_ticker:
        log("! T212_TICKER missing from .env (or use --symbol). Nothing to trade.")
        return 1
    log("=== Watcher T212 ===")
    log(f"    instrument   : {label}  ({t212_ticker}, price through {yahoo_symbol})")
    log(f"    mediu T212   : {t212_env.upper()}")
    log(f"    mode         : {'STRATEGY (DCA+TP)' if strat_enabled else 'a single order'}")
    log(f"    execution    : {'PAPER (no money)' if (strat_dry if strat_enabled else order_dry) else '⚠ REAL — REAL MONEY'}")
    log(f"    lansare      : checking until {label} is launched (already listed: immediately; IPO: at the open)")
    log(f"    ntfy/email   : {os.environ.get('NTFY_TOPIC_TRADES') or '-'} / {os.environ.get('ALERT_TO_EMAIL') or '-'}")

    # --- PRE-FLIGHT: verify the instrument at STARTUP so configuration errors are caught
    #     immediately, not after waiting days for launch. Stop now on a wrong ISIN.
    #     If an unlaunched IPO ticker is not in metadata yet, warn and continue.
    log("    pre-flight: verific instrumentul pe T212...")
    if not verify_instrument(client, t212_ticker, expected_isin):
        return 1

    # --- IDENTICAL mechanism for EVERY symbol: wait until the instrument has LAUNCHED
    #     with real volume. NVDA passes immediately; SPCX waits for its actual launch.
    #     --skip-wait provides an emergency bypass.
    if not args.skip_wait:
        if not _wait_for_launch(args, yahoo_symbol, label, interval):
            return 130  # interrupted

    # --- FINAL post-launch verification catches ticker reuse before trading ---
    if not verify_instrument(client, t212_ticker, expected_isin):
        return 1

    try:
        return start_trading(client, t212_ticker, label, strat_enabled, strat_dry,
                             order_price, order_qty, order_budget_ron, order_validity,
                             order_dry, args.desktop)
    except KeyboardInterrupt:
        log("Stopped manually.")
        return 130


def _wait_for_launch(args, yahoo_symbol, label, interval) -> bool:
    """Wait until the symbol is actually trading; return False if interrupted."""
    log(f"    Waiting for the launch {label}... (Ctrl+C to stop)")
    try:
        while True:
            if args.market_hours_only and not in_market_window():
                time.sleep(min(interval * 5, 600))
                continue
            m = check_market(yahoo_symbol)
            # 'launched' means the instrument has traded with real volume, even if its
            # market is currently closed. Existing NVDA starts immediately, while a
            # zero-volume SPCX IPO placeholder waits for actual trading to begin.
            if m and m.get("launched"):
                ts = now_str()
                now_open = "trading NOW" if m.get("trading") else f"the market is {m.get('state')}"
                body = (f"{label} e DISPONIBIL pe {m.get('exchange')} ({now_open}).\n"
                        f"Price: {m['price']} {m.get('currency') or ''}  "
                        f"(vol {m.get('volume')}, {m.get('state')})\n{ts}")
                log("############################################")
                log(f">>> {label} E DISPONIBIL — pornesc tranzactionarea <<<")
                log(body.replace("\n", " | "))
                log("############################################")
                notify(title=f"{label} disponibil — pornesc!", body=body,
                       source=m.get("exchange") or "market", price=m.get("price"),
                       desktop=args.desktop)
                return True
            if m:
                log(f"ping - waiting for the launch  |  price={m.get('price')} vol={m.get('volume')} "
                    f"state={m.get('state')} age={m.get('age_min')}min")
            else:
                log("ping - simbol indisponibil pe feed")
            time.sleep(interval)
    except KeyboardInterrupt:
        log("Stopped manually.")
        return False


# ---------------------------------------------------------------------------
# One-shot commands
# ---------------------------------------------------------------------------
def _cmd_find_ticker(client: T212Client, query: str) -> int:
    log(f"[FIND] Caut '{query}' in instrumentele T212...")
    instruments = client.list_instruments()
    if instruments is None:
        log("! cannot list the instruments (auth/network)")
        return 1
    q = query.lower()
    hits = [i for i in instruments
            if q in str(i.get("ticker", "")).lower()
            or q in str(i.get("name", "")).lower()
            or q in str(i.get("shortName", "")).lower()]
    for h in hits:
        log(f"  ticker={h.get('ticker'):<20} name={h.get('name')}  "
            f"currency={h.get('currencyCode')}  isin={h.get('isin')}")
    if not hits:
        log(f"  No result for '{query}'")
    return 0


def _cmd_test_notify(what: str, label: str, desktop: bool) -> int:
    ts = now_str()
    if what in ("market", "all"):
        notify(title=f"[TEST] {label} a inceput tranzactionarea!",
               body=f"{label} IS TRADING.\nLast price: 99.99 USD\n{ts}",
               source="market", price=99.99, desktop=desktop)
    if what in ("trade", "all"):
        notify(title=f"[TEST] Order {label} placed on T212!",
               body=f"LIMIT qty=0.5 @ 99 USD\n{ts}", source="trade", desktop=desktop)
    log("[TEST] Gata.")
    return 0


def _cmd_test_order(client, ticker, order_price, order_qty, order_budget_ron,
                    order_validity, order_dry, desktop) -> int:
    if not order_price:
        log("! ORDER_PRICE missing in .env"); return 1
    if not order_qty and not order_budget_ron:
        log("! ORDER_QTY or ORDER_BUDGET_RON missing from .env"); return 1
    qty = resolve_quantity(order_price, order_qty, order_budget_ron)
    if not qty or qty <= 0:
        log("! invalid quantity"); return 1
    ok = place_order_with_retry(client, ticker, qty, order_price, order_validity,
                                order_dry, desktop=desktop, write_marker=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
