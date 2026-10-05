#!/usr/bin/env python3
"""Import selected fleet modules and report import-time failures.

This does not invoke module entry points or start fleet loops, but ordinary Python
import-time side effects in the selected modules still occur. Run it with the repository
virtual environment before restarting the fleet.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODS = [
    "binance_api.bapi", "binance_api.bapi_client", "binance_api.bapi_placeorder",
    "binance_api.bapi_trades", "binance_api.bapi_allorders", "binance_api.bapi_ws",
    "cacheManager", "market_monitor.pricefetcher", "tradeall", "monitortrades",
    "assetguardian", "rtrade", "monitororder", "market_regime",
    "providers.market_api", "symbols",
]

fails = 0
for m in MODS:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(f"FAIL {m}: {e.__class__.__name__}: {e}")

print(f"\n=== {len(MODS)-fails}/{len(MODS)} OK, {fails} failed ===")
sys.exit(1 if fails else 0)
