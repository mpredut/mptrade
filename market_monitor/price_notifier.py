#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""Run new-coin discovery and watch-list price alerts in one process.

``price_notifier.conf`` supplies the watch list, thresholds, scan intervals, sources,
and limits. Importing this module loads environment files, validates required alert
configuration, and prints notification-channel status; only ``main`` starts monitor
threads. Coin discovery additionally requires ``ALERT_NEW_COIN=TRUE``.

Usage: ``python3 price_notifier.py [--config PATH] [--check]``.
"""

import argparse
import os
import platform
import threading
import time


from pricechecker import start_price_alert_checker
from pricefetcher import create_cachePriceAll
# Alert orchestration formerly lived in ``run_price_monitor.py``.
from market_monitor.new_coins_discovery import create_new_coins_checker, NewCoinsMonitor, NewCoinsFactory, MAX_NEW_COINS_TO_TRACK
from alertnotifiers import AlertNotifier
from botcore import load_env_stack, required_bool_env

_ROOT = os.path.dirname(os.path.abspath(__file__))
load_env_stack(os.path.join(_ROOT, ".env"))
ALERT_NEW_COIN = required_bool_env("ALERT_NEW_COIN")

# Price alerts route to their dedicated ntfy topic, passed explicitly to the batch
# notifier (no PHONE_ALERT_URL indirection).
_price_topic = os.environ.get("NTFY_TOPIC_PRICE")
PRICE_WEBHOOK_URL = f"https://ntfy.sh/{_price_topic}" if _price_topic else None

CMC_API_KEY = os.environ.get('CMC_API_KEY')
TIME_INTERVAL_CLEANUP = 6 * 60 * 60  # 6 hours in seconds
REQUIRED_ENV_VARS = ("CMC_API_KEY",)
ENABLED_SOURCES = ["coinmarketcap", "coingecko", "binance", "dexscreener"]

def validate_required_env():
    missing = []
    for key in REQUIRED_ENV_VARS:
        if not os.environ.get(key):
            missing.append(key)

    if not os.environ.get("NTFY_TOPIC_PRICE"):
        missing.append("NTFY_TOPIC_PRICE")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(sorted(set(missing)))
        )

validate_required_env()
CMC_API_KEY = os.environ.get('CMC_API_KEY')

def print_notification_channels_status():
    print("ENV CONFIGURATION:")
    if PRICE_WEBHOOK_URL:
        print(f"   ✅ Phone webhook: ENABLED -> {PRICE_WEBHOOK_URL[:40]}...")
    else:
        print("   ❌ Phone webhook: DISABLED (NTFY_TOPIC_PRICE is missing)")

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        print(f"   ✅ Telegram: ENABLED -> Chat ID: {tg_chat}")
    else:
        missing = []
        if not tg_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not tg_chat:
            missing.append("TELEGRAM_CHAT_ID")
        print(f"   ❌ Telegram: DISABLED (missing: {', '.join(missing)})")

    email_user = os.environ.get("SMTP_USERNAME")
    email_pass = os.environ.get("SMTP_PASSWORD")
    alert_to_email = os.environ.get("ALERT_TO_EMAIL")
    if email_user and email_pass and alert_to_email:
        print(f"   ✅ Email: ENABLED -> Sender: {email_user}")
    else:
        missing = []
        if not email_user:
            missing.append("SMTP_USERNAME")
        if not email_pass:
            missing.append("SMTP_PASSWORD")
        if not alert_to_email:
            missing.append("ALERT_TO_EMAIL")
        print(f"   ❌ Email: DISABLED (missing: {', '.join(missing)})")



print_notification_channels_status()


def alert_handler(alert):
    AlertNotifier.send(alert, enable_phone_webhook=True, webhook_url=PRICE_WEBHOOK_URL)

def new_coin_alerts_handler(alerts):
    if not alerts:
        return

    print("\n" + "=" * 70)
    print(f"🆕 {len(alerts)} NEW COINS DISCOVERED")
    print("=" * 70)

    for coin_info in alerts:
        source = coin_info.get('source', 'unknown')
        has_price = coin_info.get('has_price', False)
        auto_added = coin_info.get('auto_added', False)
        added_at = AlertNotifier.format_human_readable_time(
            coin_info.get('added_at')
        )

        print(
            f"🆕 {coin_info['symbol']} - "
            f"{coin_info.get('name', 'N/A')}"
        )
        print(f"   📡 Source: {source}")
        print(f"   📅 Added: {added_at}")

        if has_price:
            print(f"   💰 Price: ${coin_info.get('price', 0):.8f}")
            print(f"   ✅ Auto-added: {auto_added}")
        else:
            print("   ⚠️ Informational only")

        if coin_info.get('url'):
            print(f"   🔗 {coin_info['url']}")

        print()

    print("=" * 70)

    # Send one notification containing the whole coin batch.
    AlertNotifier.send(
        alerts,
        enable_phone_webhook=True,
        webhook_url=PRICE_WEBHOOK_URL
    )

def print_new_coin_status(cachePriceAll, new_coins_checker):
    print("\n" + "=" * 70)
    print("📊 STATUS REPORT")
    print("=" * 70)

    if hasattr(cachePriceAll, 'original_symbols'):
        symbols_count = len(cachePriceAll.original_symbols)
        print(f"\n💰 Tracked price symbols: {symbols_count}")
        print(f"   First 10: {cachePriceAll.original_symbols[:10]}")

    if new_coins_checker:
        summary = new_coins_checker.get_summary()
        print(f"\n🆕 New coins discovered total: {summary['total_new_coins']}")
        for source, data in summary['sources'].items():
            print(f"   {source}: {data['count']} coins")
        if summary['all_symbols']:
            print(f"   New symbols: {summary['all_symbols'][:10]}")


def periodic_cleanup(cachePriceAll, new_coins_checker):
    """Run price cleanup every six hours and coin cleanup when a monitor is supplied."""
    while True:
        print(f"sleeping for {TIME_INTERVAL_CLEANUP} hours before next cleanup...")
        time.sleep(TIME_INTERVAL_CLEANUP )
        print("[Periodic] Running cleanup for stale prices...")

        if hasattr(cachePriceAll, 'cleanup_old_prices'):
            cachePriceAll.cleanup_old_prices()
        else:
            print("[Periodic] cachePriceAll.cleanup_old_prices() does not exist")

        if hasattr(cachePriceAll, 'cleanup_old_symbols'):
            cachePriceAll.cleanup_old_symbols(max_age_days=7)
        else:
            print("[Periodic] cachePriceAll.cleanup_old_symbols() does not exist")

        if new_coins_checker and hasattr(new_coins_checker, 'cleanup_old_new_coins'):
            new_coins_checker.cleanup_old_new_coins()
        else:
            print("[Periodic] new_coins_checker.cleanup_old_new_coins() does not exist")

def start_new_coin_checker(cachePriceAll, interval_seconds=3600,
                           max_new_coins=MAX_NEW_COINS_TO_TRACK, sources=None):
    print("\n⏳ Initializing new coin checker...")

    factory = NewCoinsFactory(enabled_sources=sources or ENABLED_SOURCES, cmc_api_key=CMC_API_KEY)
    new_coins_checker = NewCoinsMonitor(cachePriceAll, factory=factory)
    new_coins_checker.register_alerts_callback(new_coin_alerts_handler)
    new_coins_checker.start_monitoring(interval_seconds=interval_seconds)
    print(f"New coin checker started! Active sources: {factory.get_available_sources()}")

    print("\n⏳ Performing initial new coin discovery...")
    new_coins_checker.refresh()

    auto_added_count = 0
    for source_name, coins in new_coins_checker.all_new_coins.items():
        if source_name.lower() == "coinmarketcap":
            for coin in coins[:max_new_coins]:
                if new_coins_checker.add_new_coin_to_watchlist(coin):
                    auto_added_count += 1
        else:
            if coins:
                symbols_list = ', '.join([c['symbol'] for c in coins[:10]])
                if len(coins) > 10:
                    symbols_list += f" and {len(coins) - 10} more"
                print(f"[Startup] ℹ️ Source {source_name}: {len(coins)} new coins: {symbols_list}")

    if auto_added_count > 0:
        print(f"✅ {auto_added_count} new coins auto-added to watchlist from CoinMarketCap")
    else:
        print("ℹ️ No new coins with price were found on CoinMarketCap")

    print_new_coin_status(cachePriceAll, new_coins_checker)
    print(new_coins_checker.get_report())

    return new_coins_checker

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Alert monitor: new coins plus price thresholds (config-driven).")
    ap.add_argument("--config", default=os.path.join(_HERE, "price_notifier.conf"))
    ap.add_argument("--check", action="store_true", help="validate the config plus the imports and exit (it does not start the monitor)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ac = cfg["alert_config"]
    print("=" * 70)
    print(f"⚙️ market_alerts — config: {args.config}")
    print(f"   watchlist : {cfg['watch']}  (max {cfg['max_monitored']})")
    print(f"   default   : up +{ac['default']['up_percent']}% / down -{ac['default']['down_percent']}%")
    print(f"   new_coin  : up +{ac['dynamic']['up_percent']}% / down -{ac['dynamic']['down_percent']}%")
    if ac["per_coin"]:
        pc = ", ".join(f"{k}(+{v['up_percent']}/-{v['down_percent']})" for k, v in ac["per_coin"].items())
        print(f"   per-coin: {pc}")
    print(f"   cooldown {ac['cooldown_minutes']}min | lookback {ac['lookback_hours']}h | "
          f"price scan {cfg['price_scan_seconds']}s | new-coin scan {cfg['new_coins_scan_seconds']}s")
    print("=" * 70)

    if args.check:
        print("✅ --check: the config is valid and the imports are OK. Exiting without starting the monitor.")
        return

    print("\n⏳ Initialising the price cache...")
    cachePriceAll = create_cachePriceAll(cmc_api_key=CMC_API_KEY,
                                         symbols=cfg["watch"], max_symbols=cfg["max_monitored"])
    print("⏳ Waiting for the first price sync (5s)...")
    time.sleep(5)

    print("⏳ Starting the price-threshold checker...")
    price_checker = start_price_alert_checker(
        cachePriceAll=cachePriceAll, alert_callback=alert_handler,
        check_interval_seconds=cfg["price_scan_seconds"], config=ac)

    # The thread captures the arguments supplied here. Because ``None`` is passed before
    # the optional coin monitor is constructed, this thread cleans price state only and
    # never calls ``cleanup_old_new_coins`` on the later local variable.
    cleanup_thread = threading.Thread(target=periodic_cleanup, name="periodic_cleanup",
                                      args=(cachePriceAll, None), daemon=True)
    cleanup_thread.start()

    new_coins_checker = None
    if not cfg["discover_new_coins"]:
        print("NEW COIN ALERT DISABLED in config (discover_new_coins = no) — watchlist only")
    elif ALERT_NEW_COIN:
        print("⏳ Pornesc checker-ul de monede noi...")
        new_coins_checker = start_new_coin_checker(
            cachePriceAll, interval_seconds=cfg["new_coins_scan_seconds"],
            max_new_coins=cfg["max_new_coins"], sources=cfg["sources"])
    else:
        print("NEW COIN ALERT DEZACTIVAT (ALERT_NEW_COIN != TRUE)")

    # Send startup notification so the admin knows the fleet is active.
    # Type "bot_event" gets special rendering in the ntfy channel.
    AlertNotifier.send({
        "type": "bot_event",
        "symbol": "SYSTEM",
        "event_name": f"Binance fleet started on {platform.system()}"
    }, enable_phone_webhook=True, webhook_url=PRICE_WEBHOOK_URL)

    try:
        while True:
            time.sleep(160)
            print("\n👉 Waiting for alerts... (Ctrl+C to stop)\n")
    except KeyboardInterrupt:
        print("\n🛑 Opresc sistemul...")
        if new_coins_checker is not None:
            new_coins_checker.stop_monitoring()
        price_checker.stop_monitoring()
        print("👋 Gata.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
alerts_config.py — load the plain-text price_notifier.conf file for the alert monitor.

Line-oriented format (# starts a full-line or inline comment):
    watch    = BTC, TAO, HYPE          # watchlist (coins that are always monitored)
    sources  = coinmarketcap, coingecko
    default  = 4.1 / 7.5               # default threshold: UP% / DOWN%
    new_coin = 12 / 25                 # threshold for new coins
    BTC      = 6 / 10                  # PER-COIN threshold (any symbol)
    cooldown_minutes = 30              # scalar settings (see _SETTING_KEYS)

The file and every operational key are mandatory. Missing or malformed configuration
aborts startup instead of silently selecting a trading/alerting policy.
"""

import os

# Scalar key -> (type, destination: "ac" in alert_config or "top" in cfg).
_SETTING_KEYS = {
    "cooldown_minutes": (int, "ac"), "lookback_hours": (int, "ac"),
    "max_monitored": (int, "top"), "max_new_coins": (int, "top"),
    "new_coins_scan_seconds": (int, "top"), "price_scan_seconds": (int, "top"),
}
_LIST_KEYS = {"watch": str.upper, "sources": str.lower}
_BUCKET_ALIAS = {"default": "default", "new_coin": "dynamic"}  # conf new_coin -> internal dynamic


def _pair(val: str) -> dict:
    """'6 / 10' -> {'up_percent': 6.0, 'down_percent': 10.0}."""
    up, _, down = val.partition("/")
    return {"up_percent": float(up.strip()), "down_percent": float(down.strip())}


def load_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise ValueError(f"Required alert configuration file is missing: {path}")
    cfg = {"alert_config": {"per_coin": {}}}
    ac = cfg["alert_config"]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"Malformed alert configuration line: {raw.rstrip()!r}")
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            try:
                if key == "discover_new_coins":
                    normalized = val.strip().lower()
                    if normalized not in {"yes", "no", "true", "false", "1", "0", "on", "off"}:
                        raise ValueError("expected an explicit boolean")
                    cfg[key] = normalized in {"yes", "true", "1", "on"}
                elif key in _LIST_KEYS:
                    norm = _LIST_KEYS[key]
                    cfg[key] = [norm(x.strip()) for x in val.split(",") if x.strip()]
                elif key in _SETTING_KEYS:
                    typ, where = _SETTING_KEYS[key]
                    v = typ(float(val))
                    (ac if where == "ac" else cfg)[key] = v
                elif "/" in val:                          # An UP/DOWN threshold.
                    pair = _pair(val)
                    if key in _BUCKET_ALIAS:
                        ac[_BUCKET_ALIAS[key]] = pair
                    else:                                 # Every other name is a per-coin key.
                        ac["per_coin"][key.upper()] = pair
                else:
                    raise ValueError(f"Unknown alert configuration key: {key}")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid alert configuration for {key}: {val!r}") from exc

    required_top = {
        "watch", "sources", "discover_new_coins", "max_monitored",
        "max_new_coins", "new_coins_scan_seconds", "price_scan_seconds",
    }
    required_alert = {"default", "dynamic", "cooldown_minutes", "lookback_hours"}
    missing = sorted(required_top - cfg.keys()) + sorted(required_alert - ac.keys())
    if missing:
        raise ValueError("Missing mandatory alert settings: " + ", ".join(missing))
    if not cfg["watch"] or not cfg["sources"]:
        raise ValueError("Alert watch and sources lists must not be empty")
    return cfg


def resolve(alert_config: dict, symbol: str, is_dynamic: bool) -> dict:
    """Resolve a coin threshold: per_coin, then dynamic for new coins, then default."""
    per = alert_config.get("per_coin", {})
    if symbol in per:
        return per[symbol]
    return alert_config["dynamic"] if is_dynamic else alert_config["default"]


if __name__ == "__main__":
    import json
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "price_notifier.conf"
    print(json.dumps(load_config(p), indent=2))
