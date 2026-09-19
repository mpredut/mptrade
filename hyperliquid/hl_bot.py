#!/usr/bin/env python3
"""
hl_bot.py — Hyperliquid long-only perpetual watcher and DCA/take-profit auto-trader.

IMPORTANT: run with the Hyperliquid SDK/eth_account virtual environment. Easiest:
    ./hl_run.sh            # selects server myenv, local .venv, or python3
Alternatively: source ../myenv/bin/activate; python hl_bot.py

Commands:
    ...python hl_bot.py                  # run from .env
    ...python hl_bot.py --paper          # PAPER without money or wallet
    ...python hl_bot.py --price          # public HYPE price
    ...python hl_bot.py --balance        # available USDC; requires HL_ACCOUNT_ADDRESS
    ...python hl_bot.py --positions      # current position
    ...python hl_bot.py --test-strategy HYPE
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from common import (
    load_env_stack, log, now_str, required_env, required_int_env,
    required_bool_env, single_instance,
)
from hl_client import HLClient, HLError
from market_data import get_price, coin_available
from notify import notify
from strategy import Strategy, StratParams

def _build_client(need_wallet: bool) -> HLClient:
    mainnet = required_bool_env("HL_MAINNET")
    secret = os.environ.get("HL_SECRET_KEY") if need_wallet else None
    addr = os.environ.get("HL_ACCOUNT_ADDRESS")
    return HLClient(secret_key=secret, account_address=addr, mainnet=mainnet)


def main() -> int:
    env_file = os.environ.get("ENV_FILE", os.path.join(os.path.dirname(__file__), ".env"))
    for i, a in enumerate(sys.argv):
        if a == "--env-file" and i + 1 < len(sys.argv):
            env_file = sys.argv[i + 1]
    load_env_stack(env_file)

    ap = argparse.ArgumentParser(description="Bot DCA+TP pe Hyperliquid (perp long-only).")
    ap.add_argument("--env-file", default=env_file)
    ap.add_argument("--coin", help="Override the coin (otherwise HL_COIN from .env)")
    ap.add_argument("--interval", type=int, default=required_int_env("HL_POLL_SECONDS"))
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--paper", action="store_true", help="PAPER (no money, no wallet)")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--signal", action="store_true", help="Arata semnalul de trend/predictie curent")
    ap.add_argument("--test-strategy", metavar="COIN")
    args = ap.parse_args()

    coin      = (args.coin or required_env("HL_COIN")).strip()
    label     = os.environ.get("SYMBOL_LABEL") or coin
    leverage  = required_int_env("HL_LEVERAGE")
    strat_dry = args.paper or not required_bool_env("STRAT_EXECUTE")
    interval  = max(args.interval, 15)

    if args.test_strategy:
        single_instance(f"hl_bot_{args.test_strategy.strip()}")
    elif not any((args.price, args.balance, args.positions, args.signal)):
        single_instance(f"hl_bot_{coin}")

    # A wallet is required only for real trading.
    need_wallet = not strat_dry or args.balance or args.positions
    # RESILIENCE: retry startup network failures instead of terminating with a traceback.
    while True:
        try:
            client = _build_client(need_wallet)
            break
        except HLError as e:
            log(f"! {e}")
            return 1                         # configuration error: do not retry
        except KeyboardInterrupt:
            return 130
        except Exception as e:  # noqa: BLE001
            log(f"! the connection failed ({e.__class__.__name__}) — retrying in 60s")
            time.sleep(60)

    if args.price:
        p = get_price(client, coin); log(f"[PRICE] {coin} = {p}")
        return 0 if p else 1
    if args.balance:
        log(f"[BALANCE] USDC disponibil: {client.withdrawable()}")
        return 0
    if args.positions:
        szi, entry = client.position(coin)
        log(f"[POSITION] {coin}: size={szi} entryPx={entry}")
        return 0
    if args.signal:
        from signals import get_signal
        s = get_signal(client, coin)
        log(f"[SIGNAL] {coin}: trend={s['trend']}  confidence={s['confidence']}  sursa={s['source']}"
            + (f"  ({s['detail']})" if s.get("detail") else ""))
        return 0
    if args.test_strategy:
        log(f"[TEST] strategie {args.test_strategy}  {'PAPER' if strat_dry else '⚠ REAL'}")
        Strategy(client, args.test_strategy, StratParams.from_env(),
                 dry_run=strat_dry, desktop=args.desktop, leverage=leverage).run()
        return 0

    log("=== Hyperliquid bot ===")
    log(f"    coin         : {label}  ({coin} perp, levier {leverage}x)")
    log(f"    wallet       : {'yes' if os.environ.get('HL_SECRET_KEY') else 'NO (public/paper only)'}")
    log(f"    execution    : {'PAPER (no money)' if strat_dry else '⚠ REAL — REAL MONEY'}")
    log(f"    ntfy/email   : {os.environ.get('NTFY_TOPIC_TRADES') or '-'} / {os.environ.get('ALERT_TO_EMAIL') or '-'}")

    if not args.skip_wait:
        if not _wait_for_listing(client, coin, label, interval, args.desktop):
            return 130

    try:
        Strategy(client, coin, StratParams.from_env(), dry_run=strat_dry,
                 desktop=args.desktop, leverage=leverage).run()
        return 0
    except KeyboardInterrupt:
        log("Stopped manually."); return 130


def _wait_for_listing(client, coin, label, interval, desktop) -> bool:
    if coin_available(client, coin):
        log(f"  [verify] {coin} e disponibil pe Hyperliquid — pornesc.")
        return True
    log(f"    {coin} is unavailable on Hyperliquid — waiting... (Ctrl+C to stop)")
    try:
        while True:
            if coin_available(client, coin):
                p = get_price(client, coin)
                log(f">>> {label} is available on Hyperliquid (price {p}) — starting <<<")
                notify(title=f"{label} disponibil pe Hyperliquid!",
                       body=f"{coin} price {p}", source="hyperliquid", price=p, desktop=desktop)
                return True
            log(f"ping - waiting for {coin}...")
            time.sleep(interval)
    except KeyboardInterrupt:
        log("Stopped manually."); return False


if __name__ == "__main__":
    raise SystemExit(main())
