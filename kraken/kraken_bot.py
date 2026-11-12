#!/usr/bin/env python3
"""
kraken_bot.py -- Kraken Spot watcher and DCA/take-profit auto-trader.

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

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    add_paper_skip_wait_cli_args,
    create_env_cli_parser,
    load_cli_env_stack,
    load_env_stack,
    log,
    optional_env,
    required_bool_env,
    required_env,
    required_int_env,
    safe_pos_float,
    single_instance,
)
from kraken.kraken_common import build_kraken_client, notify
from kraken.kraken_client import KrakenClient, KrakenError
from kraken.market_data import check_pair_listing, get_price
from strategies.spot_engine import Strategy, StratParams

# Provider-agnostic path B requires the StrategyExecutor contract. Wrap the _BOT client
# in KrakenProvider so Strategy shares its connection/nonce. Balance/find/price CLI
# commands continue using KrakenClient directly.
from providers.order_ohlc_utils import format_strategy_execution_mode  # noqa: E402
from providers.kraken_provider import KrakenProvider  # noqa: E402
from providers.execution_audit import AuditedStrategyExecutor  # noqa: E402
from credentials import kraken_credentials  # noqa: E402

def _build_client() -> KrakenClient:
    return build_kraken_client(KrakenClient, kraken_credentials, "bot")


def main() -> int:
    env_file = load_cli_env_stack(__file__, load_fn=load_env_stack)
    poll_seconds = required_int_env("KRAKEN_BOT_POLL_SEC")

    ap = create_env_cli_parser("Kraken DCA and take-profit bot.", env_file)
    ap.add_argument("--pair", help="Override the pair (otherwise from .env KRAKEN_PAIR)")
    ap.add_argument("--interval", type=int, default=poll_seconds)
    add_paper_skip_wait_cli_args(
        ap,
        paper_help="Force paper mode without using real money",
        skip_wait_help="Skip waiting for the listing",
        include_desktop=True,
        test_strategy_metavar="PAIR",
        test_strategy_help="Run the strategy NOW on the given pair",
        find_flag="--find-pair",
        find_metavar="TERM",
        find_help="Search Kraken pairs",
        extra_store_true=(
            ("--price", "Show the current price and exit"),
            ("--balance", "Show balances (requires API credentials)"),
        ),
    )
    args = ap.parse_args()
    if args.test_strategy:
        args.test_strategy = args.test_strategy.strip().upper()
        single_instance(f"kraken_bot_{args.test_strategy.strip()}")
    elif not any(getattr(args, a, None) for a in ("balance", "find_pair", "price")):
        # Use one instance PER PAIR so HYPE, ADA, WIF, etc. can run concurrently with
        # separate locks. The former fixed 'kraken_bot' key blocked a second pair.
        _lock_pair = (args.pair or required_env("KRAKEN_PAIR")).strip().upper()
        single_instance(f"kraken_bot_{_lock_pair}")

    # --- one-shot commands ---
    if args.find_pair:
        return _cmd_find_pair(KrakenClient(), args.find_pair)
    if args.price:
        price_pair = (args.pair or required_env("KRAKEN_PAIR")).strip().upper()
        p = get_price(KrakenClient(), price_pair) if price_pair else None
        log(f"[PRICE] {price_pair} = {p}")
        return 0 if p else 1
    if args.balance:
        return _cmd_balance(_build_client())

    client = _build_client()

    pair        = (args.pair or required_env("KRAKEN_PAIR")).strip().upper()
    label       = required_env("SYMBOL_LABEL")
    strategy_execution = required_bool_env("STRAT_EXECUTE")
    strat_dry = KrakenProvider.strategy_dry_run(args.paper, strategy_execution)
    observe_only = required_bool_env("KRAKEN_BOT_OBSERVE_ONLY")
    interval    = max(args.interval, 15)

    if args.test_strategy:
        test_pair = args.test_strategy.strip().upper()
        log(
            f"[TEST] strategy for {test_pair}  "
            f"{'OBSERVE ONLY' if observe_only else ('PAPER' if strat_dry else '⚠ REAL')}")
        Strategy(AuditedStrategyExecutor(KrakenProvider(client=client), venue="Kraken"), test_pair, StratParams.from_env(),
                 dry_run=strat_dry, desktop=args.desktop, observe_only=observe_only).run()
        return 0

    if not pair:
        log("! KRAKEN_PAIR missing from .env (or --pair). Nothing to trade.")
        return 1

    log("=== Kraken bot ===")
    log(f"    pair         : {label}  ({pair})")
    log(
        f"    credentials  : "
        f"{'yes' if optional_env('KRAKEN_API_KEY_BOT') else 'NO (public/paper only)'}")
    log(f"    execution    : {format_strategy_execution_mode(observe_only, strat_dry)}")
    log(f"    ntfy/email   : {optional_env('NTFY_TOPIC_TRADES', '-')} / {optional_env('ALERT_TO_EMAIL', '-')}")

    # --- wait until the pair is LISTED and tradable, analogous to launch ---
    if not args.skip_wait:
        if not _wait_for_listing(client, pair, label, interval, args.desktop):
            return 130

    # --- start the strategy ---
    try:
        Strategy(AuditedStrategyExecutor(KrakenProvider(client=client), venue="Kraken"), pair, StratParams.from_env(), dry_run=strat_dry,
                 desktop=args.desktop, observe_only=observe_only).run()
        return 0
    except KeyboardInterrupt:
        log("Stopped manually.")
        return 130


def _wait_for_listing(client, pair, label, interval, desktop) -> bool:
    # Preflight: start immediately if the pair is already listed.
    info, confirmed_unlisted = check_pair_listing(client, pair)
    if info:
        log(f"  [verify] {pair} is LISTED and tradable — starting.")
        return True
    saw_unlisted = confirmed_unlisted
    if saw_unlisted:
        log(f"    {pair} not listed on Kraken yet — waiting for it... (Ctrl+C to stop)")
    else:
        log(f"    {pair} verification deferred (transient API/network error) — retrying in {interval}s...")
    try:
        while True:
            time.sleep(interval)
            info, confirmed_unlisted = check_pair_listing(client, pair)
            if confirmed_unlisted:
                saw_unlisted = True
            if info:
                if not saw_unlisted:
                    log(f"  [verify] {pair} is LISTED and tradable after API recovery — starting.")
                    return True
                p = get_price(client, pair)
                clean_lbl = label.strip() if isinstance(label, str) else ""
                display_label = pair if clean_lbl.lower() in ("", "kraken") else f"{clean_lbl} ({pair})"
                body = f"{display_label} is available on Kraken, price {p}"
                log("############################################")
                log(f">>> {display_label} LISTED ON KRAKEN — STARTING <<<")
                log("############################################")
                notify(title=f"{display_label} listed on Kraken!", body=body,
                       source="kraken", price=p, desktop=desktop)
                return True
            log(f"ping - waiting for the listing {pair}...")
    except KeyboardInterrupt:
        log("Stopped manually.")
        return False


def _cmd_find_pair(client: KrakenClient, term: str) -> int:
    if not term or isinstance(term, bool) or not str(term).strip():
        log("! invalid search term")
        return 1
    term = str(term).strip()
    try:
        pairs = client.asset_pairs()
    except KrakenError as e:
        log(f"! {e}")
        return 1
    if not isinstance(pairs, dict):
        log("! invalid asset_pairs response")
        return 1
    t = term.upper()
    hits = [(k, v) for k, v in pairs.items()
            if isinstance(v, dict) and t in (k + str(v.get("altname")) + str(v.get("wsname")) + str(v.get("base"))).upper()]
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
    if isinstance(bal, dict):
        for asset, amt in bal.items():
            if safe_pos_float(amt, default=None) is not None:
                log(f"  {asset:<8} {amt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
