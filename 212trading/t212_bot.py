#!/usr/bin/env python3
"""
t212_bot.py — UNIFIED Trading 212 bot: ONE process, multiple assets.

Replaces `ipo.py --profile X`, which used one process per asset. It discovers every
config.<asset>.env file and starts one THREAD for each:
  * ISOLATED: one asset's error cannot terminate the others; it catches and retries;
  * ONE T212 client with shared throttling reduces rate-limit 429 responses;
  * each asset's configuration comes from a FILE rather than the command line.

Add an asset by creating config.<asset>.env, without new code/process/cron entries.
Remove one by renaming the file, for example config.nvda.env.off.

  python3 t212_bot.py                # run every config.*.env; REAL if STRAT_EXECUTE=true
  python3 t212_bot.py --paper        # force safe PAPER mode for all assets
  python3 t212_bot.py --only nvda    # run one asset for debugging
  python3 t212_bot.py --skip-wait    # bypass launch waiting and start directly
  python3 t212_bot.py --list         # show assets without starting them
"""
from __future__ import annotations

import argparse
import glob
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipo_common import (  # noqa: E402
    load_t212_environment, log, parse_dotenv, single_instance,
    required_env, required_float_env, required_bool_env,
)
from ipo_notify import notify  # noqa: E402
from listing_watcher import wait_for_launch  # noqa: E402
from strategy import Strategy, StratParams  # noqa: E402
from t212_client import T212Client  # noqa: E402
from credentials import t212_credentials  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
STOP = threading.Event()


def discover_assets(cfg_dir: str) -> list[tuple[str, str]]:
    """Return sorted config.<asset>.env files as [(name, path), ...].
    config.nvda.env -> ('nvda', '.../config.nvda.env')."""
    out = []
    for p in sorted(glob.glob(os.path.join(cfg_dir, "config.*.env"))):
        name = os.path.basename(p)[len("config."):-len(".env")]
        out.append((name, p))
    return out


def verify_isin(client: T212Client, ticker: str, expected_isin: str) -> bool:
    """Return False only for a proven ISIN mismatch identifying the wrong instrument.
    Otherwise return True best-effort and continue with the explicit ticker if metadata is absent."""
    if not expected_isin:
        return True
    instruments = client.list_instruments()
    if not instruments:
        return True
    match = next((i for i in instruments if str(i.get("ticker", "")).upper() == ticker.upper()), None)
    if not match:
        return True
    if str(match.get("isin", "")) != expected_isin:
        log(f"  ! [{ticker}] ISIN {match.get('isin')} != the expected {expected_isin} — NOT trading")
        notify(title=f"⚠ {ticker}: ISIN nepotrivit", source="T212",
               body=f"Found {match.get('isin')}, we expected {expected_isin}.", symbol=ticker)
        return False
    return True


def run_asset(name: str, cfg: dict, client: T212Client, force_paper: bool, skip_wait: bool) -> None:
    """Run an asset lifecycle in its isolated thread, catching and retrying errors
    without terminating the process or other bots."""
    ticker = required_env("T212_TICKER", cfg)
    label = required_env("SYMBOL_LABEL", cfg)
    yahoo = required_env("YAHOO_SYMBOL", cfg)
    isin = (cfg.get("EXPECTED_ISIN") or "").strip()
    strat_enabled = required_bool_env("STRAT_ENABLED", cfg)
    strat_dry = force_paper or not required_bool_env("STRAT_EXECUTE", cfg)
    interval = required_float_env("POLL_SECONDS", cfg)
    if interval < 30:
        raise ValueError("POLL_SECONDS must be at least 30")

    if not strat_enabled:
        log(f"  ! [{name}] STRAT_ENABLED!=true — skipping (t212_bot only runs strategies)"); return

    log(f"  ▶ [{label}] {ticker} | price through {yahoo} | {'PAPER' if strat_dry else '⚠ REAL — MONEY'} | poll {interval}s")
    while not STOP.is_set():
        try:
            if not verify_isin(client, ticker, isin):
                return  # invalid configuration: do not retry blindly
            if not skip_wait:
                ok = wait_for_launch(
                    yahoo, label, interval, stop=STOP,
                    on_launch=lambda m: notify(
                        title=f"{label} disponibil — pornesc!", source="T212",
                        body=f"{label} tranzactionabil pe {m.get('exchange')} @ {m.get('price')}",
                        price=m.get("price"), symbol=label))
                if not ok:
                    return  # stop requested
                if not verify_isin(client, ticker, isin):
                    return
            # Block in the strategy's self-healing loop. It returns only after an
            # unexpected stop, at which point verification and startup are retried.
            Strategy(client, ticker, StratParams.from_env(cfg), dry_run=strat_dry).run(stop=STOP)
            return
        except Exception as e:  # noqa: BLE001 — resilience: one asset cannot terminate the process
            log(f"  ! [{label}] eroare ciclu ({e.__class__.__name__}: {e}) — reincerc in 60s")
            STOP.wait(60)
    log(f"  ⏹ [{label}] stopped")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bot unificat T212: un proces, mai multe active (config.*.env).")
    ap.add_argument("--paper", action="store_true", help="Force PAPER on all of them (a safe test)")
    ap.add_argument("--only", metavar="NUME", help="Run only the asset with this name (config.NAME.env)")
    ap.add_argument("--skip-wait", action="store_true", help="Skip waiting for the launch")
    ap.add_argument("--list", action="store_true", help="Show the assets and exit")
    ap.add_argument("--env-file", default=os.path.join(_HERE, ".env"))
    args = ap.parse_args()
    if not args.list:
        single_instance("t212_bot")   # one instance prevents duplicate trading

    load_t212_environment(args.env_file)
    cfg_dir = os.path.dirname(args.env_file) or _HERE
    assets = discover_assets(cfg_dir)
    if args.only:
        assets = [(n, p) for (n, p) in assets if n == args.only]
    if not assets:
        log(f"! no config.*.env found in {cfg_dir}" + (f" pt '{args.only}'" if args.only else ""))
        return 1

    if args.list:
        log(f"=== {len(assets)} active ===")
        for n, p in assets:
            c = parse_dotenv(p)
            real = c.get("STRAT_EXECUTE", "").strip().lower() == "true" and not args.paper
            on = c.get("STRAT_ENABLED", "").strip().lower() == "true"
            log(f"  {n:<8} {c.get('T212_TICKER','?'):<14} "
                f"{'STRAT' if on else 'OFF':<6} {'⚠ REAL' if real else 'PAPER'}")
        return 0

    credentials = t212_credentials()
    client = T212Client(
        credentials.key, credentials.secret,
        env=required_env("T212_ENV"),
        min_gap_sec=required_float_env("T212_MIN_GAP_SEC"),
        portfolio_ttl_sec=required_float_env("T212_PORTFOLIO_TTL_SEC"),
    )

    log(f"=== t212_bot: {len(assets)} assets in ONE process ({'PAPER forced' if args.paper else 'config'}) ===")
    client.list_instruments()  # warm the cache ONCE so threads avoid the expensive call
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    threads = []
    for name, path in assets:
        cfg = parse_dotenv(path)
        t = threading.Thread(target=run_asset, name=name, daemon=True,
                             args=(name, cfg, client, args.paper, args.skip_wait))
        t.start()
        threads.append(t)
        time.sleep(0.5)  # stagger startup to avoid simultaneous API calls

    try:
        while not STOP.is_set() and any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        log("Oprire ceruta (Ctrl+C)...")
        STOP.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
