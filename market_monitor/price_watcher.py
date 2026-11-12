#!/usr/bin/env python3
"""Run new-coin discovery and watch-list price alerts in one process.

``price_watcher.conf`` supplies the watch list, thresholds, scan intervals, sources,
and limits. Importing this module loads environment files, validates required alert
configuration, and prints notification-channel status; only ``main`` starts monitor
threads. Coin discovery additionally requires ``ALERT_NEW_COIN=TRUE``.

Usage: ``python3 price_watcher.py [--config PATH] [--check]``.
"""
import argparse
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from market_monitor.pricechecker import PriceChecker
from market_monitor.pricefetcher import create_cachePriceAll

# Alert orchestration formerly lived in ``run_price_monitor.py``.
from market_monitor.new_coins_discovery import NewCoinsMonitor, NewCoinsFactory, MAX_NEW_COINS_TO_TRACK
from notify_engine.alertnotifiers import AlertNotifier, format_ntfy_url
from botcore import (
    add_store_true_cli_args,
    load_env_stack,
    optional_env,
    required_bool_env,
    required_env,
    safe_int,
    safe_nonneg_float,
    secret_env,
    touch_module_heartbeat,
)
from cachepaths import cache_path
from market_monitor.alerts_config import load_config

load_env_stack()
ALERT_NEW_COIN = required_bool_env("ALERT_NEW_COIN")

# Price alerts route to their dedicated ntfy topic, passed explicitly to the batch
# notifier (no PHONE_ALERT_URL indirection).
_price_topic = required_env("NTFY_TOPIC_PRICE")
PRICE_WEBHOOK_URL = format_ntfy_url(_price_topic)

CMC_API_KEY = required_env("CMC_API_KEY")
TIME_INTERVAL_CLEANUP = 6 * 60 * 60  # 6 hours in seconds
ENABLED_SOURCES = ["coinmarketcap", "coingecko", "binance", "dexscreener"]


print("ENV CONFIGURATION:")
if PRICE_WEBHOOK_URL:
    print(f"   [OK] Phone webhook: ENABLED -> {PRICE_WEBHOOK_URL[:40]}...")
else:
    print("   [DISABLED] Phone webhook: DISABLED (NTFY_TOPIC_PRICE is missing)")

_tg_token = secret_env("TELEGRAM_BOT_TOKEN")
_tg_chat = optional_env("TELEGRAM_CHAT_ID")
if _tg_token and _tg_chat:
    print(f"   [OK] Telegram: ENABLED -> Chat ID: {_tg_chat}")
else:
    _missing_tg = [k for k, v in (("TELEGRAM_BOT_TOKEN", _tg_token), ("TELEGRAM_CHAT_ID", _tg_chat)) if not v]
    print(f"   [DISABLED] Telegram: DISABLED (missing: {', '.join(_missing_tg)})")

_email_user = optional_env("SMTP_USERNAME")
_email_pass = secret_env("SMTP_PASSWORD")
_alert_to_email = optional_env("ALERT_TO_EMAIL")
if _email_user and _email_pass and _alert_to_email:
    print(f"   [OK] Email: ENABLED -> Sender: {_email_user}")
else:
    _missing_em = [
        k for k, v in (
            ("SMTP_USERNAME", _email_user),
            ("SMTP_PASSWORD", _email_pass),
            ("ALERT_TO_EMAIL", _alert_to_email),
        ) if not v
    ]
    print(f"   [DISABLED] Email: DISABLED (missing: {', '.join(_missing_em)})")


def periodic_cleanup(cachePriceAll, new_coins_checker):
    """Run price cleanup every six hours and coin cleanup when a monitor is supplied."""
    while True:
        print(f"sleeping for {TIME_INTERVAL_CLEANUP / 3600:g} hours before next cleanup...")
        time.sleep(TIME_INTERVAL_CLEANUP)
        print("[Periodic] Running cleanup for stale prices...")

        if hasattr(cachePriceAll, 'cleanup_old_prices'):
            try:
                cachePriceAll.cleanup_old_prices()
            except Exception as e:
                print(f"[Periodic] cleanup_old_prices error: {e}")
        else:
            print("[Periodic] cachePriceAll.cleanup_old_prices() does not exist")

        if hasattr(cachePriceAll, 'cleanup_old_symbols'):
            try:
                cachePriceAll.cleanup_old_symbols(max_age_days=7)
            except Exception as e:
                print(f"[Periodic] cleanup_old_symbols error: {e}")
        else:
            print("[Periodic] cachePriceAll.cleanup_old_symbols() does not exist")

        if new_coins_checker and hasattr(new_coins_checker, 'cleanup_old_new_coins'):
            try:
                new_coins_checker.cleanup_old_new_coins()
            except Exception as e:
                print(f"[Periodic] cleanup_old_new_coins error: {e}")
        else:
            print("[Periodic] new_coins_checker.cleanup_old_new_coins() does not exist")

def start_new_coin_checker(cachePriceAll, interval_seconds=3600,
                           max_new_coins=MAX_NEW_COINS_TO_TRACK, sources=None):
    print("\nInitializing new coin checker...")

    def _on_new_coin_alerts(alerts):
        if not alerts:
            return

        print("\n" + "=" * 70)
        print(f"{len(alerts)} NEW COINS DISCOVERED")
        print("=" * 70)

        for coin_info in alerts:
            if not isinstance(coin_info, dict):
                continue
            source = coin_info.get('source', 'unknown')
            has_price = coin_info.get('has_price', False)
            auto_added = coin_info.get('auto_added', False)
            added_at = AlertNotifier.format_human_readable_time(
                coin_info.get('added_at')
            )

            print(
                f"* {coin_info.get('symbol', 'N/A')} - "
                f"{coin_info.get('name', 'N/A')}"
            )
            print(f"   Source: {source}")
            print(f"   Added: {added_at}")

            if has_price:
                px_val = safe_nonneg_float(coin_info.get('price'), default=0.0)
                print(f"   Price: ${px_val:.8f}")
                print(f"   Auto-added: {auto_added}")
            else:
                print("   Informational only")

            if coin_info.get('url'):
                print(f"   URL: {coin_info['url']}")

            print()

        print("=" * 70)

        # Send one notification containing the whole coin batch.
        AlertNotifier.send(
            alerts,
            enable_phone_webhook=True,
            webhook_url=PRICE_WEBHOOK_URL
        )

    factory = NewCoinsFactory(enabled_sources=sources or ENABLED_SOURCES, cmc_api_key=CMC_API_KEY)
    new_coins_checker = NewCoinsMonitor(cachePriceAll, factory=factory)
    new_coins_checker.alert_callbacks.append(_on_new_coin_alerts)
    new_coins_checker.start_monitoring(interval_seconds=interval_seconds)
    print(f"New coin checker started! Active sources: {factory.get_available_sources()}")

    print("\nPerforming initial new coin discovery...")
    new_coins_checker.refresh()

    auto_added_count = 0
    max_new_i = safe_int(max_new_coins, default=None) if isinstance(max_new_coins, (int, float)) else None
    eff_max_new = max(0, max_new_i) if max_new_i is not None else MAX_NEW_COINS_TO_TRACK
    for source_name, coins in new_coins_checker.all_new_coins.items():
        if source_name.lower() == "coinmarketcap":
            for coin in coins[:eff_max_new]:
                if new_coins_checker.add_new_coin_to_watchlist(coin):
                    auto_added_count += 1
        else:
            if coins:
                symbols_list = ', '.join([str(c.get('symbol', 'N/A')) for c in coins[:10] if isinstance(c, dict)])
                if len(coins) > 10:
                    symbols_list += f" and {len(coins) - 10} more"
                print(f"[Startup] Source {source_name}: {len(coins)} new coins: {symbols_list}")

    if auto_added_count > 0:
        print(f"{auto_added_count} new coins auto-added to watchlist from CoinMarketCap")
    else:
        print("No new coins with price were found on CoinMarketCap")

    print(f"\n{'=' * 70}\nSTATUS REPORT\n{'=' * 70}")
    if hasattr(cachePriceAll, 'original_symbols'):
        print(f"\nTracked price symbols: {len(cachePriceAll.original_symbols)}")
        print(f"   First 10: {cachePriceAll.original_symbols[:10]}")
    if new_coins_checker:
        summary = new_coins_checker.get_summary()
        if isinstance(summary, dict):
            print(f"\nNew coins discovered total: {summary.get('total_new_coins', 0)}")
            sources_map = summary.get('sources')
            if isinstance(sources_map, dict):
                for source, data in sources_map.items():
                    cnt = data.get('count', 0) if isinstance(data, dict) else 0
                    print(f"   {source}: {cnt} coins")
            all_syms = summary.get('all_symbols')
            if isinstance(all_syms, list) and all_syms:
                print(f"   New symbols: {all_syms[:10]}")
    print(new_coins_checker.get_report())

    return new_coins_checker

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Alert monitor: new coins plus price thresholds (config-driven).")
    ap.add_argument("--config", default=os.path.join(_HERE, "price_watcher.conf"))
    args = add_store_true_cli_args(
        ap,
        ("--check", "validate the config plus the imports and exit (it does not start the monitor)"),
    ).parse_args()

    cfg = load_config(args.config)
    ac = cfg["alert_config"]
    print("=" * 70)
    print(f"market_alerts -- config: {args.config}")
    print(f"   watchlist : {cfg['watch']}  (max {cfg['max_monitored']})")
    print(f"   default   : up +{ac['default']['up_percent']}% / down -{ac['default']['down_percent']}%")
    print(f"   new_coin  : up +{ac['dynamic']['up_percent']}% / down -{ac['dynamic']['down_percent']}%")
    if ac["per_coin"]:
        pc = ", ".join(f"{k}(+{v['up_percent']}/-{v['down_percent']})" for k, v in ac["per_coin"].items())
        print(f"   per-coin: {pc}")
    print(f"   cooldown {ac['cooldown_minutes']}min | lookback {ac['lookback_hours']}h | "
          f"price scan {cfg['price_scan_seconds']}s | new-coin scan {cfg['new_coins_scan_seconds']}s")
    print("=" * 70)
    print(
        "[OK] --check: the config is valid and the imports are OK. Exiting without starting the monitor."
        if args.check
        else "\nInitialising the price cache..."
    )
    if args.check:
        return

    cmc_key_for_cache = CMC_API_KEY if (cfg.get("discover_new_coins") and ALERT_NEW_COIN) else None
    cachePriceAll = create_cachePriceAll(cmc_api_key=cmc_key_for_cache,
                                         symbols=cfg["watch"], max_symbols=cfg["max_monitored"])
    print("Waiting for the first price sync (5s)...")
    time.sleep(5)

    print("Starting the price-threshold checker...")
    price_checker = PriceChecker(
        cachePriceAll=cachePriceAll,
        alert_callback=lambda alert: AlertNotifier.send(alert, enable_phone_webhook=True, webhook_url=PRICE_WEBHOOK_URL),
        config=ac,
        state_file=cache_path("cache_watcher_all_price.cooldowns.json"),
    )
    price_checker.start_monitoring(cfg["price_scan_seconds"])

    new_coins_checker = None
    if not cfg["discover_new_coins"]:
        print("NEW COIN ALERT DISABLED in config (discover_new_coins = no) -- watchlist only")
    elif ALERT_NEW_COIN:
        print("Starting new coin discovery checker...")
        new_coins_checker = start_new_coin_checker(
            cachePriceAll, interval_seconds=cfg["new_coins_scan_seconds"],
            max_new_coins=cfg["max_new_coins"], sources=cfg["sources"])
    else:
        print("NEW COIN ALERT DISABLED (ALERT_NEW_COIN != TRUE)")

    cleanup_thread = threading.Thread(target=periodic_cleanup, name="periodic_cleanup",
                                      args=(cachePriceAll, new_coins_checker), daemon=True)
    cleanup_thread.start()

    try:
        hb_syms = list(cfg.get("watch") or ())
        hb_itv = {"price_scan": cfg["price_scan_seconds"]}
        touch_module_heartbeat("price_watcher", interval_sec=30.0, force=True, symbols=hb_syms, intervals=hb_itv)
        while True:
            time.sleep(30)
            touch_module_heartbeat("price_watcher", interval_sec=30.0, symbols=hb_syms, intervals=hb_itv)
            print("\nWaiting for alerts... (Ctrl+C to stop)\n", flush=True)
    except KeyboardInterrupt:
        print("\nStopping system...")
        if new_coins_checker is not None:
            new_coins_checker.stop_monitoring()
        price_checker.stop_monitoring()
        print("Done.")


if __name__ == "__main__":
    main()

