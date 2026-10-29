#!/usr/bin/env python3
"""
listing_watcher.py — GENERIC component that waits until an asset becomes tradable.

Decoupled from the strategy and any specific asset. For ANY new IPO, listing, or
reactivated ticker, wait for real volume or moving prices without mistaking a
pre-listing placeholder for trading. Reused by t212_bot.py for every awaited asset.
"""
from __future__ import annotations

import time
from datetime import datetime

try:
    from .ipo_common import ET, log, safe_float
    from .market_data import check_market
except ImportError:
    import os as _os
    import sys as _sys
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_HERE)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    if _HERE not in _sys.path:
        _sys.path.append(_HERE)
    from ipo_common import ET, log, safe_float
    try:
        from market_data import check_market
    except ImportError:
        import importlib.util
        _md_spec = importlib.util.spec_from_file_location(
            "t212_listing_watcher_md", _os.path.join(_HERE, "market_data.py")
        )
        _md_mod = importlib.util.module_from_spec(_md_spec)
        _md_spec.loader.exec_module(_md_mod)
        check_market = _md_mod.check_market


def in_market_window() -> bool:
    """Return True during approximate US market hours, 09:00-16:30 ET on weekdays."""
    n = datetime.now(ET)
    if n.weekday() >= 5:
        return False
    minutes = n.hour * 60 + n.minute
    return 9 * 60 <= minutes <= 16 * 60 + 30


def _wait(stop: "threading.Event | None", seconds: float) -> None:
    """Sleep interruptibly when a stop event exists, otherwise sleep normally."""
    if stop is not None:
        stop.wait(seconds)
    else:
        time.sleep(seconds)


def wait_for_launch(yahoo_symbol: str, label: str, interval: int = 60, *,
                    market_hours_only: bool = False,
                    stop: "threading.Event | None" = None,
                    on_launch=None) -> bool:
    """Block until `yahoo_symbol` has LAUNCHED through actual trading.
    The mechanism is identical for every symbol: existing NVDA passes immediately,
    while a zero-volume pre-IPO placeholder waits for trading to open.

    Return True at launch and False if `stop` is set first. Optional `on_launch(info)`
    receives the market dictionary at launch, for example to send a notification.
    """
    eff_interval = max(1.0, safe_float(interval, default=60.0))
    log(f"    [{label}] waiting for the launch... (poll {eff_interval:g}s)")
    while not (stop is not None and stop.is_set()):
        if market_hours_only and not in_market_window():
            _wait(stop, min(eff_interval * 5, 600))
            continue
        m = check_market(yahoo_symbol)
        # 'launched' means prior real-volume trading, even if the market is closed now.
        if m and m.get("launched"):
            now_open = "trading NOW" if m.get("trading") else f"the market is {m.get('state')}"
            log(f">>> [{label}] AVAILABLE on {m.get('exchange')} ({now_open}) "
                f"price={m.get('price')} {m.get('currency') or ''} <<<")
            if on_launch:
                try:
                    on_launch(m)
                except Exception as e:  # noqa: BLE001
                    log(f"    ! [{label}] on_launch failed: {e}")
            return True
        if m:
            log(f"    [{label}] ping — waiting for the launch | price={m.get('price')} "
                f"vol={m.get('volume')} state={m.get('state')} age={m.get('age_min')}min")
        else:
            log(f"    [{label}] ping — symbol unavailable on feed")
        _wait(stop, eff_interval)
    return False
