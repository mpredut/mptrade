#!/usr/bin/env python3
"""
kraken_bot.py — Kraken Spot watcher and DCA/take-profit auto-trader.

The same flow as T212 ipo.py, adapted to Kraken. KRAKEN_PAIR configures the pair in
.env. If a future pair is not listed yet, the bot WAITS until it appears before entry,
matching T212 SPCX logic.

Commands:
    python3 kraken_bot.py                 # run from .env
    python3 kraken_bot.py --paper         # force PAPER without money
    python3 kraken_bot.py --pair HYPEEUR  # override pair
    python3 kraken_bot.py --find-pair hype  # find the exact Kraken pair
    python3 kraken_bot.py --price         # show current price
    python3 kraken_bot.py --balance       # show balances; requires credentials
    python3 kraken_bot.py --test-strategy HYPEEUR  # run the strategy NOW
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kraken_common import (log, now_str, required_bool_env, required_env,
                           required_int_env, load_env_stack,
                           single_instance)
from kraken_client import KrakenClient, KrakenError
from market_data import get_price, pair_available
from notify import notify
from strategies.spot_dca import Strategy, StratParams

# Provider-agnostic path B requires the StrategyExecutor contract. Wrap the _BOT client
# in KrakenProvider so Strategy shares its connection/nonce. Balance/find/price CLI
# commands continue using KrakenClient directly.
from providers.kraken_provider import (  # noqa: E402
    KrakenProvider,
    kraken_strategy_dry_run,
)
from providers.execution_audit import AuditedStrategyExecutor  # noqa: E402
from credentials import kraken_credentials  # noqa: E402

def _build_client() -> KrakenClient:
    credentials = kraken_credentials("bot")
    return KrakenClient(credentials.key, credentials.secret)


def _build_executor(client: KrakenClient) -> AuditedStrategyExecutor:
    """Return a strict Kraken executor decorated only with observational JSONL audit."""
    return AuditedStrategyExecutor(KrakenProvider(client=client), venue="Kraken")


def main() -> int:
    env_file = os.environ.get("ENV_FILE", os.path.join(os.path.dirname(__file__), ".env"))
    for i, a in enumerate(sys.argv):
        if a == "--env-file" and i + 1 < len(sys.argv):
            env_file = sys.argv[i + 1]
    load_env_stack(env_file)
    poll_seconds = required_int_env("KRAKEN_BOT_POLL_SEC")

    ap = argparse.ArgumentParser(description="Kraken DCA and take-profit bot.")
    ap.add_argument("--env-file", default=env_file)
    ap.add_argument("--pair", help="Override the pair (otherwise from .env KRAKEN_PAIR)")
    ap.add_argument("--interval", type=int, default=poll_seconds)
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--skip-wait", action="store_true", help="Skip waiting for the listing")
    ap.add_argument(
        "--paper", action="store_true",
        help="Force paper mode without using real money")
    ap.add_argument("--find-pair", metavar="TERM", help="Search Kraken pairs")
    ap.add_argument("--price", action="store_true", help="Show the current price and exit")
    ap.add_argument(
        "--balance", action="store_true",
        help="Show balances (requires API credentials)")
    ap.add_argument("--test-strategy", metavar="PAIR", help="Run the strategy NOW on the given pair")
    args = ap.parse_args()
    if args.test_strategy:
        single_instance(f"kraken_bot_{args.test_strategy.strip()}")
    elif not any(getattr(args, a, None) for a in ("balance", "find_pair", "price")):
        # Use one instance PER PAIR so HYPE, ADA, WIF, etc. can run concurrently with
        # separate locks. The former fixed 'kraken_bot' key blocked a second pair.
        _lock_pair = (args.pair or required_env("KRAKEN_PAIR")).strip()
        single_instance(f"kraken_bot_{_lock_pair}")

    client = _build_client()

    pair        = (args.pair or required_env("KRAKEN_PAIR")).strip()
    label       = required_env("SYMBOL_LABEL")
    strategy_execution = required_bool_env("STRAT_EXECUTE")
    strat_dry = kraken_strategy_dry_run(args.paper, strategy_execution)
    interval    = max(args.interval, 15)

    # --- one-shot commands ---
    if args.find_pair:
        return _cmd_find_pair(client, args.find_pair)
    if args.price:
        p = get_price(client, pair) if pair else None
        log(f"[PRICE] {pair} = {p}")
        return 0 if p else 1
    if args.balance:
        return _cmd_balance(client)
    if args.test_strategy:
        log(
            f"[TEST] strategy for {args.test_strategy}  "
            f"{'PAPER' if strat_dry else '⚠ REAL'}")
        Strategy(_build_executor(client), args.test_strategy, StratParams.from_env(),
                 dry_run=strat_dry, desktop=args.desktop).run()
        return 0

    if not pair:
        log("! KRAKEN_PAIR missing from .env (or --pair). Nothing to trade.")
        return 1

    log("=== Kraken bot ===")
    log(f"    pair         : {label}  ({pair})")
    log(
        f"    credentials  : "
        f"{'yes' if os.environ.get('KRAKEN_API_KEY_BOT') else 'NO (public/paper only)'}")
    log(f"    execution    : {'PAPER (no money)' if strat_dry else '⚠ REAL — REAL MONEY'}")
    log(f"    ntfy/email   : {os.environ.get('NTFY_TOPIC_TRADES') or '-'} / {os.environ.get('ALERT_TO_EMAIL') or '-'}")

    # --- wait until the pair is LISTED and tradable, analogous to launch ---
    if not args.skip_wait:
        if not _wait_for_listing(client, pair, label, interval, args.desktop):
            return 130

    # --- start the strategy ---
    try:
        Strategy(_build_executor(client), pair, StratParams.from_env(), dry_run=strat_dry,
                 desktop=args.desktop).run()
        return 0
    except KeyboardInterrupt:
        log("Stopped manually.")
        return 130


def _wait_for_listing(client, pair, label, interval, desktop) -> bool:
    # Preflight: start immediately if the pair is already listed.
    info = pair_available(client, pair)
    if info:
        log(f"  [verify] {pair} is LISTED and tradable — starting.")
        return True
    log(f"    {pair} not listed on Kraken yet — waiting for it... (Ctrl+C to stop)")
    try:
        while True:
            info = pair_available(client, pair)
            if info:
                p = get_price(client, pair)
                body = f"{label} ({pair}) is available on Kraken, price {p}"
                log("############################################")
                log(f">>> {label} LISTED ON KRAKEN — STARTING <<<")
                log("############################################")
                notify(title=f"{label} listed on Kraken!", body=body,
                       source="kraken", price=p, desktop=desktop)
                return True
            log(f"ping - waiting for the listing {pair}...")
            time.sleep(interval)
    except KeyboardInterrupt:
        log("Stopped manually.")
        return False


def _cmd_find_pair(client: KrakenClient, term: str) -> int:
    try:
        pairs = client.asset_pairs()
    except KrakenError as e:
        log(f"! {e}")
        return 1
    t = term.upper()
    hits = [(k, v) for k, v in pairs.items()
            if t in (k + str(v.get("altname")) + str(v.get("wsname")) + str(v.get("base"))).upper()]
    log(f"[FIND] '{term}' — {len(hits)} results:")
    for k, v in hits[:20]:
        log(f"  altname={v.get('altname'):<12} wsname={v.get('wsname'):<14} "
            f"base={v.get('base')} quote={v.get('quote')} status={v.get('status')}")
    if not hits:
        log("  (none)")
    return 0


def _cmd_balance(client: KrakenClient) -> int:
    try:
        bal = client.balance()
    except KrakenError as e:
        log(f"! balance: {e}")
        return 1
    log("=== Kraken balances ===")
    for asset, amt in bal.items():
        if float(amt) > 0:
            log(f"  {asset:<8} {amt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
