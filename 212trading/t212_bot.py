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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HERE)
for _p in (_ROOT_DIR, _HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _mod_name in ("ipo_common", "ipo_notify", "listing_watcher", "strategy", "t212_client"):
    _mod_file = getattr(sys.modules.get(_mod_name), "__file__", None)
    if _mod_file and os.path.dirname(os.path.abspath(_mod_file)) != _HERE:
        sys.modules.pop(_mod_name, None)
try:
    from .ipo_common import (  # noqa: E402
        load_t212_environment, log, parse_dotenv, single_instance,
        required_env, required_float_env, required_bool_env, _parse_bool_flag,
    )
    from .ipo_notify import notify  # noqa: E402
    from .listing_watcher import verify_instrument_isin, wait_for_launch  # noqa: E402
    from .strategy import Strategy, StratParams  # noqa: E402
    from .t212_client import T212Client  # noqa: E402
except ImportError:
    from ipo_common import (  # noqa: E402
        load_t212_environment, log, parse_dotenv, single_instance,
        required_env, required_float_env, required_bool_env, _parse_bool_flag,
    )
    from ipo_notify import notify  # noqa: E402
    from listing_watcher import verify_instrument_isin, wait_for_launch  # noqa: E402
    from strategy import Strategy, StratParams  # noqa: E402
    from t212_client import T212Client  # noqa: E402
from credentials import t212_credentials  # noqa: E402

STOP = threading.Event()


def discover_assets(cfg_dir: str) -> list[tuple[str, str]]:
    """Return sorted config.<asset>.env files as [(name, path), ...].
    config.nvda.env -> ('nvda', '.../config.nvda.env')."""
    out = []
    for p in sorted(glob.glob(os.path.join(glob.escape(cfg_dir), "config.*.env"))):
        name = os.path.basename(p)[len("config."):-len(".env")]
        out.append((name, p))
    return out


def verify_isin(client: T212Client, ticker: str, expected_isin: str) -> bool:
    """Return False only for a proven ISIN mismatch identifying the wrong instrument.
    Otherwise return True best-effort and continue with the explicit ticker if metadata is absent."""
    return verify_instrument_isin(
        client,
        ticker,
        expected_isin,
        notify_fn=notify,
        notify_source="T212",
        verbose=False,
    )


def run_asset(name: str, cfg: dict, client: T212Client, force_paper: bool, skip_wait: bool) -> None:
    """Run an asset lifecycle in its isolated thread, catching and retrying errors
    without terminating the process or other bots."""
    ticker = required_env("T212_TICKER", cfg)
    label = required_env("SYMBOL_LABEL", cfg)
    yahoo = required_env("YAHOO_SYMBOL", cfg)
    isin = (cfg.get("EXPECTED_ISIN") or "").strip()
    strat_enabled = required_bool_env("STRAT_ENABLED", cfg)
    strat_dry = force_paper or not required_bool_env("STRAT_EXECUTE", cfg)
    observe_only = _parse_bool_flag(cfg.get("T212_BOT_OBSERVE_ONLY") or os.environ.get("T212_BOT_OBSERVE_ONLY"), False)
    interval = required_float_env("POLL_SECONDS", cfg)
    if not (interval >= 30 and interval <= 86400):
        raise ValueError("POLL_SECONDS must be a finite number between 30 and 86400")

    if not strat_enabled:
        log(f"  ! [{name}] STRAT_ENABLED!=true — skipping (t212_bot only runs strategies)"); return

    exec_mode_str = "OBSERVE ONLY (AI managed)" if observe_only else ("PAPER" if strat_dry else "⚠ REAL — MONEY")
    log(f"  ▶ [{label}] {ticker} | price through {yahoo} | {exec_mode_str} | poll {interval}s")
    while not STOP.is_set():
        try:
            if not verify_isin(client, ticker, isin):
                return  # invalid configuration: do not retry blindly
            if not skip_wait:
                ok = wait_for_launch(yahoo, label, interval, stop=STOP)
                if not ok:
                    return  # stop requested
                if not verify_isin(client, ticker, isin):
                    return
            # Block in the strategy's self-healing loop. It returns only after an
            # unexpected stop, at which point verification and startup are retried.
            Strategy(client, ticker, StratParams.from_env(cfg), dry_run=strat_dry, observe_only=observe_only).run(stop=STOP)
            return
        except Exception as e:  # noqa: BLE001 — resilience: one asset cannot terminate the process
            log(f"  ! [{label}] cycle error ({e.__class__.__name__}: {e}) — retrying in 60s")
            STOP.wait(60)
    log(f"  ⏹ [{label}] stopped")


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified T212 bot: one process, multiple assets (config.*.env).")
    ap.add_argument("--paper", action="store_true", help="Force PAPER on all of them (a safe test)")
    ap.add_argument("--only", metavar="NAME", help="Run only the asset with this name (config.NAME.env)")
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
        only_cf = args.only.strip().casefold()
        assets = [(n, p) for (n, p) in assets if n.casefold() == only_cf]
    if not assets:
        log(f"! no config.*.env found in {cfg_dir}" + (f" for '{args.only}'" if args.only else ""))
        return 1

    if args.list:
        log(f"=== {len(assets)} assets ===")
        for n, p in assets:
            c = parse_dotenv(p)
            real = _parse_bool_flag(c.get("STRAT_EXECUTE"), False) and not args.paper
            on = _parse_bool_flag(c.get("STRAT_ENABLED"), False)
            log(f"  {n:<8} {c.get('T212_TICKER','?'):<14} "
                f"{'STRAT' if on else 'OFF':<6} {'⚠ REAL' if real else 'PAPER'}")
        return 0

    credentials = t212_credentials()
    client = T212Client(
        credentials.key, credentials.secret,
        env=required_env("T212_ENV"),
        min_gap_sec=required_float_env("T212_MIN_GAP_SEC"),
        portfolio_ttl_sec=required_float_env("T212_PORTFOLIO_TTL_SEC"),
        order_min_gap_sec=required_float_env("T212_ORDER_MIN_GAP_SEC"),
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
        log("Shutdown requested (Ctrl+C)...")
        STOP.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
