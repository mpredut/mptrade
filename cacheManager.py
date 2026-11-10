"""Orchestration, singleton registry (``CacheFactory``), and daemon supervisor for the cache subsystem.

The 3-tier polymorphic cache hierarchy is split across 5 focused modules:
1. ``cacheInterface.py`` — Level 1 pure abstract contract: ``CacheManagerInterface(ABC)``
2. ``cacheaccount.py``   — Level 2 ``CacheAccountManagerBase(CacheManagerInterface, ABC)``
                           + Level 3 ``CacheTradeManager``, ``CacheOrderManager``, ``CacheAssetValueManager``
3. ``cacheprice.py``     — Level 2 ``CachePriceManagerBase(CacheManagerInterface, ABC)``
                           + Level 3 ``CacheSparsePriceManager``, ``Cache24PriceManager``,
                           ``Cache365PriceManager``, ``CacheCurrentPriceManager``
                           + Non-Binance price streams (``HyperliquidPriceStream``,
                           ``KrakenPriceStream``, ``_NonBinanceTrendPoller``)
4. ``cachetrend.py``     — Level 2 ``CacheTrendManagerBase(CacheManagerInterface, ABC)``
                           + Level 3 ``CacheLongTrendManager``, ``CacheShortTrendManager``
5. ``cacheManager.py``   — ``CacheFactory`` registry, ``get_*_manager()`` accessors,
                           Binance user-stream WS bridge wiring, and daemon entrypoint.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import threading
import time
from typing import Optional

from binance_api import bapi as api
from binance_api import bapi_allorders as apiorders
from binance_api import bapi_ws_bridge as _bapi_ws_bridge
import binance_cache_health as account_cache_health
from botcore import (
    load_dotenv,
    safe_nonneg_float,
    single_instance,
    touch_module_heartbeat,
)
from cacheInterface import (
    CM_LONG_ARCHIVE_FLUSH_SEC,
    CM_LONG_ARCHIVE_MEMORY_ROWS,
    CM_LONG_ARCHIVE_MONTHS,
    CM_LONG_ARCHIVE_SAMPLE_SEC,
    CacheManagerInterface,
    _CONFIG_ROOT,
    _coerce_symbol_list,
    _monotonic_time,
    _strict_int_field,
)
from cacheaccount import (
    CacheAccountManagerBase,
    CacheAssetValueManager,
    CacheOrderManager,
    CacheTradeManager,
    WS_EVENT_LOG_ENABLED,
    WS_LOSS_TIMEOUT_SEC,
    WS_ONLY_MODE,
    _mark_ws_available,
    _mark_ws_event_received,
    _mark_ws_unhealthy,
    _should_poll_for_manager,
    _ws_available,
    _ws_health_lock,
    _ws_is_healthy,
    _ws_last_event_ts,
)
from cachepaths import migrate_legacy_filename_if_needed
from cacheprice import (
    Cache24PriceManager,
    Cache365PriceManager,
    CacheCurrentPriceManager,
    CachePriceManagerBase,
    CacheSparsePriceManager,
    HyperliquidPriceStream,
    KrakenPriceStream,
    _fetch_price_with_deadline,
    _start_nonbinance_trend_poller,
)
from cachetrend import (
    CM_DYNAMIC_WINDOW_MAX_SEC,
    CM_DYNAMIC_WINDOW_MIN_SEC,
    CM_PUBLISHED_TREND_WINDOWS_SEC,
    CacheLongTrendManager,
    CacheShortTrendManager,
    CacheTrendManagerBase,
)
import instrument_registry as sym
from instrument_registry import tracked_price_instruments
from providers.base import MarketDataProvider, _parse_bool_flag
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
)

load_dotenv()

_ws_health_lock = _bapi_ws_bridge._ws_health_lock
_ws_available = _bapi_ws_bridge._ws_available
_ws_last_event_ts = _bapi_ws_bridge._ws_last_event_ts
_ws_is_healthy = _bapi_ws_bridge._ws_is_healthy

_current_price_instance: Optional[CacheCurrentPriceManager] = None
_current_price_lock = threading.RLock()


def _default_price_tracking():
    """Return ``(symbols, nonbinance_symbols, provider_names)`` from ``instruments.conf``."""
    return tracked_price_instruments()


def get_current_price_manager(
    ws_manager=None,
    symbols=None,
    sync_ts=None,
    start_sync=True,
    provider_names=None,
    read_only=None,
) -> CacheCurrentPriceManager:
    """Return or lazily create the CacheCurrentPriceManager singleton."""
    global _current_price_instance
    start_sync = _parse_bool_flag(start_sync, default=True)
    inst = _current_price_instance
    if (
        inst is not None
        and sync_ts is None
        and not provider_names
        and ws_manager is None
        and (
            not start_sync
            or (
                (thread := inst.thread) is not None
                and thread.is_alive()
                and not inst.read_only
            )
        )
    ):
        return inst
    with _current_price_lock:
        if _current_price_instance is not None:
            if sync_ts is not None:
                _current_price_instance.sync_ts = sync_ts
            if provider_names:
                _current_price_instance.bind_providers(provider_names)
            if ws_manager is not None:
                _current_price_instance.attach_ws_manager(ws_manager)
            if start_sync:
                _current_price_instance.periodic_sync(save_state=False)
            return _current_price_instance
        if symbols is None:
            _syms, _, _default_providers = _default_price_tracking()
            if provider_names is None and _default_providers:
                provider_names = _default_providers
        else:
            _syms = (
                _coerce_symbol_list(symbols) if isinstance(symbols, str) else symbols
            )
        instance = CacheCurrentPriceManager(
            sync_ts=(
                sync_ts
                if sync_ts is not None
                else CURRENT_PRICE_SYNC_INTERVAL_SEC
            ),
            symbols=_syms,
            filename=CacheCurrentPriceManager.DEFAULT_FILENAME,
            ws_manager=ws_manager,
            provider_names=provider_names,
            api_client=api,
            read_only=(
                (not start_sync)
                if read_only is None
                else _parse_bool_flag(read_only, default=False)
            ),
        )
        if start_sync:
            instance.periodic_sync(save_state=False)
        _current_price_instance = instance
    return _current_price_instance


_short_trend_instance = None
_short_trend_lock = threading.RLock()


def get_short_trend_manager(
    symbols=None, filename=CacheShortTrendManager.DEFAULT_FILENAME, writer=False
):
    """Return the CacheShortTrendManager singleton."""
    global _short_trend_instance
    writer = _parse_bool_flag(writer, default=False)
    inst = _short_trend_instance
    if inst is not None and not writer:
        return inst
    with _short_trend_lock:
        if _short_trend_instance is not None:
            if writer:
                _short_trend_instance.become_writer()
            return _short_trend_instance
        if symbols is None:
            _syms = _default_price_tracking()[0]
        else:
            _syms = (
                _coerce_symbol_list(symbols) if isinstance(symbols, str) else symbols
            )
        _short_trend_instance = CacheShortTrendManager(
            _syms,
            filename,
            writer=writer,
            cache24_getter=lambda: get_cache_manager("24Price"),
            current_price_getter=get_current_price_manager,
        )
    return _short_trend_instance


def get_active_current_price_manager() -> Optional[CacheCurrentPriceManager]:
    """Return the active in-memory ``CacheCurrentPriceManager`` singleton without creating one."""
    return _current_price_instance or CacheFactory._instances.get("CurrentPrice")


def get_active_short_trend_manager() -> Optional[CacheShortTrendManager]:
    """Return the active in-memory ``CacheShortTrendManager`` singleton without creating one."""
    return _short_trend_instance


ORDER_SYNC_INTERVAL_SEC = 0.4 * 60
TRADE_SYNC_INTERVAL_SEC = 3 * 60
SPARSE_PRICE_SYNC_INTERVAL_SEC = 3 * 60
PRICE_24_SYNC_INTERVAL_SEC = 30
PRICE_365_SYNC_INTERVAL_SEC = CM_LONG_ARCHIVE_FLUSH_SEC
CURRENT_PRICE_SYNC_INTERVAL_SEC = 30
LONG_TREND_SYNC_INTERVAL_SEC = 10 * 60
ASSET_VALUE_SYNC_INTERVAL_SEC = 10 * 60

_PRICE_TREND_CACHES = frozenset(
    {
        "CurrentPrice",
        "24Price",
        "365Price",
        "SparsePrice",
        "LongTrend",
    }
)


class CacheFactory:
    """Single registry and singleton factory for all domain cache managers."""

    _lock = threading.RLock()
    _instances = {}
    _sync_started = set()
    _CONFIG = {}
    _teardown_hooks: dict = {}

    @classmethod
    def register_teardown_hook(cls, name: str, callback) -> None:
        """Register an optional teardown callback invoked when ``name`` is removed or shut down."""
        if not isinstance(name, str) or not name.strip() or not callable(callback):
            return
        with cls._lock:
            hooks = cls._teardown_hooks.setdefault(name, [])
            if callback not in hooks:
                hooks.append(callback)

    @classmethod
    def register(
        cls,
        name: str,
        manager_class: type,
        *,
        sync_ts,
        filename: Optional[str] = None,
        prefix: Optional[str] = None,
        ext: Optional[str] = None,
        **extra_kwargs,
    ) -> dict:
        """Register or update a domain cache implementation in ``CacheFactory``."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("CacheFactory registration name must be a non-empty string")
        if not callable(sync_ts):
            interval_val = sync_ts
            sync_ts = lambda _v=interval_val: _v
        if filename is None:
            venue_arg = extra_kwargs.get("venue")
            disk_fn = getattr(manager_class, "disk_filename", None)
            if venue_arg and callable(disk_fn):
                filename = disk_fn(venue_arg)
            else:
                filename = getattr(manager_class, "DEFAULT_FILENAME", None)
        if prefix is None:
            prefix = getattr(manager_class, "FILE_PREFIX", None)
        if ext is None:
            ext = getattr(manager_class, "FILE_EXT", None)
        entry = {
            "class": manager_class,
            "filename": filename,
            "sync_ts": sync_ts,
        }
        if prefix is not None:
            entry["prefix"] = prefix
        if ext is not None:
            entry["ext"] = ext
        entry.update(extra_kwargs)
        with cls._lock:
            cls._CONFIG[name] = entry
        return entry

    @staticmethod
    def _iter_managers(value):
        return value.values() if isinstance(value, dict) else (value,)

    @classmethod
    def get(cls, name, symbols=None, *, start_sync=True, provider_names=None):
        start_sync = _parse_bool_flag(start_sync, default=True)
        if isinstance(symbols, str):
            symbols = _coerce_symbol_list(symbols)
        if name == "ShortTrend":
            return get_short_trend_manager(symbols=symbols)
        inst = cls._instances.get(name)
        if (
            inst is not None
            and symbols is None
            and provider_names is None
            and (not start_sync or name in cls._sync_started)
        ):
            return inst
        with cls._lock:
            if name not in cls._CONFIG:
                raise ValueError(f"Unknown cache type: {name}")
            if provider_names is not None and name != "CurrentPrice":
                raise ValueError("provider_names is supported only for CurrentPrice")

            if name in cls._instances and symbols is not None:
                inst = cls._instances[name]
                existing = (
                    set(inst.keys())
                    if isinstance(inst, dict)
                    else set(getattr(inst, "symbols", []))
                )
                requested = set(symbols)
                if requested != existing:
                    missing = requested - existing
                    builtins.print(
                        f"[CacheFactory][WARN] '{name}' already exists with symbols {sorted(existing)}; "
                        f"the request for {sorted(requested)} is IGNORED"
                        + (f" (missing: {sorted(missing)})" if missing else "")
                        + ". Singleton per name — the first instance is used."
                    )
            if name == "CurrentPrice" and name in cls._instances:
                cls._instances[name].bind_providers(provider_names)

            created = name not in cls._instances
            sync_ts = None
            if created:
                config = cls._CONFIG[name]
                manager_class = config["class"]
                sync_ts = config["sync_ts"]()
                extra_kwargs = {
                    key: value
                    for key, value in config.items()
                    if key not in {"class", "filename", "sync_ts", "prefix", "ext"}
                }
                cfg_read_only = bool(extra_kwargs.pop("read_only", False))
                eff_read_only = cfg_read_only or (not start_sync)
                venue_cfg = str(extra_kwargs.get("venue") or "").strip().lower()
                is_non_binance_venue = bool(venue_cfg and venue_cfg != "binance")
                if symbols is None:
                    if name == "AssetValue":
                        symbols = ["TOTAL"]
                    elif name in _PRICE_TREND_CACHES:
                        symbols, _, _default_providers = _default_price_tracking()
                        if (
                            name == "CurrentPrice"
                            and provider_names is None
                            and _default_providers
                        ):
                            provider_names = _default_providers
                    elif is_non_binance_venue:
                        symbols = None
                    else:
                        symbols = sym.symbols

                if name == "CurrentPrice":
                    current_price_sync_ts = (
                        None if _current_price_instance is not None else sync_ts
                    )
                    instance = get_current_price_manager(
                        symbols=symbols,
                        sync_ts=current_price_sync_ts,
                        start_sync=False,
                        provider_names=provider_names,
                        read_only=eff_read_only,
                    )
                    sync_ts = instance.sync_ts
                elif "prefix" in config:
                    prefix, ext = config["prefix"], config["ext"]
                    instance = {
                        s: manager_class(
                            sync_ts=sync_ts,
                            filename=f"{prefix}{s}.{ext}",
                            symbols=[s],
                            api_client=api,
                            read_only=eff_read_only,
                            **extra_kwargs,
                        )
                        for s in symbols
                    }
                else:
                    instance = manager_class(
                        sync_ts=sync_ts,
                        filename=config["filename"],
                        symbols=symbols,
                        api_client=None if is_non_binance_venue else api,
                        read_only=eff_read_only,
                        **extra_kwargs,
                    )

                cls._instances[name] = instance

            if (
                start_sync
                and not cls._CONFIG[name].get("read_only")
                and name not in cls._sync_started
            ):
                if sync_ts is None:
                    sync_ts = cls._CONFIG[name]["sync_ts"]()
                for manager in cls._iter_managers(cls._instances[name]):
                    manager.periodic_sync(sync_ts, False)
                cls._sync_started.add(name)

            return cls._instances[name]

    @classmethod
    def _clear_bapi_trade_manager_ref(cls, trade_mgr) -> None:
        if trade_mgr is None:
            return
        for hook in list(cls._teardown_hooks.get("Trade", ())):
            try:
                hook(trade_mgr)
            except Exception:
                pass
        _bapi_tr = sys.modules.get("binance_api.bapi_trades")
        if (
            _bapi_tr is not None
            and getattr(_bapi_tr, "cache_trade_manager", None) is trade_mgr
        ):
            _bapi_tr.cache_trade_manager = None

    @classmethod
    def shutdown_all(cls, timeout=5.0):
        """Stop every factory-owned synchronization loop and clear the registry."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            current_price = (
                cls._instances.get("CurrentPrice") or _current_price_instance
            )
            trade_mgr = cls._instances.get("Trade")
            short_trend = _short_trend_instance
            managers = []
            seen_ids = set()
            for value in cls._instances.values():
                for manager in cls._iter_managers(value):
                    if id(manager) not in seen_ids:
                        seen_ids.add(id(manager))
                        managers.append(manager)
            if current_price is not None and id(current_price) not in seen_ids:
                seen_ids.add(id(current_price))
                managers.append(current_price)
            if short_trend is not None and id(short_trend) not in seen_ids:
                seen_ids.add(id(short_trend))
                managers.append(short_trend)
            cls._instances = {}
            cls._sync_started = set()
            for manager in managers:
                ev = getattr(manager, "_stop_event", None)
                if ev is not None:
                    ev.set()
            eff_timeout = safe_nonneg_float(timeout, default=5.0)
            if eff_timeout is None:
                eff_timeout = 5.0
            deadline = _monotonic_time() + eff_timeout
            results = [
                manager.shutdown(timeout=max(0.0, deadline - _monotonic_time()))
                for manager in managers
                if hasattr(manager, "shutdown")
            ]
            if (
                current_price is _current_price_instance
                and getattr(current_price, "thread", None) is None
            ):
                _current_price_instance = None
            if (
                short_trend is _short_trend_instance
                and getattr(short_trend, "_full_eval_thread", None) is None
                and getattr(short_trend, "_flush_thread", None) is None
                and getattr(short_trend, "thread", None) is None
            ):
                _short_trend_instance = None
            cls._clear_bapi_trade_manager_ref(trade_mgr)
            return all(results) if results else True

    @classmethod
    def remove(cls, name, timeout=5.0):
        """Remove a singleton cleanly after stopping all threads it owns."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            if name == "ShortTrend":
                inst = _short_trend_instance
                cls._sync_started.discard("ShortTrend")
                if inst is None:
                    return True
                stopped = (
                    bool(inst.shutdown(timeout=timeout))
                    if hasattr(inst, "shutdown")
                    else True
                )
                if stopped and inst is _short_trend_instance:
                    _short_trend_instance = None
                return stopped
            value = cls._instances.pop(name, None)
            if name == "CurrentPrice" and value is None:
                value = _current_price_instance
            cls._sync_started.discard(name)
            if value is None:
                return True
            results = [
                manager.shutdown(timeout=timeout)
                for manager in cls._iter_managers(value)
                if hasattr(manager, "shutdown")
            ]
            stopped = all(results) if results else True
            if (
                name == "CurrentPrice"
                and value is _current_price_instance
                and stopped
            ):
                _current_price_instance = None
            if name == "Trade" and stopped:
                cls._clear_bapi_trade_manager_ref(value)
            return stopped


# Register all canonical domain cache implementations in one place.
CacheFactory.register(
    "Trade",
    CacheTradeManager,
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "Order",
    CacheOrderManager,
    sync_ts=lambda: ORDER_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "SparsePrice",
    CacheSparsePriceManager,
    sync_ts=lambda: SPARSE_PRICE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "24Price",
    Cache24PriceManager,
    sync_ts=lambda: PRICE_24_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "365Price",
    Cache365PriceManager,
    sync_ts=lambda: PRICE_365_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "CurrentPrice",
    CacheCurrentPriceManager,
    sync_ts=lambda: CURRENT_PRICE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "LongTrend",
    CacheLongTrendManager,
    sync_ts=lambda: LONG_TREND_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "AssetValue",
    CacheAssetValueManager,
    sync_ts=lambda: ASSET_VALUE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "KrakenTrade",
    CacheTradeManager,
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
    venue="kraken",
    stream_format="json",
    read_only=True,
)
CacheFactory.register(
    "HyperliquidTrade",
    CacheTradeManager,
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
    venue="hyperliquid",
    stream_format="json",
    read_only=True,
)
CacheFactory.register(
    "T212Trade",
    CacheTradeManager,
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
    venue="t212",
    stream_format="json",
    read_only=True,
)


def get_cache_manager(name, symbols=None, *, start_sync=True):
    return CacheFactory.get(name, symbols, start_sync=start_sync)


# Inject readers and runtime services across the domain base classes and MarketDataProvider.
CacheManagerInterface.bind_runtime_services(
    api_client=api,
    poll_predicate=_should_poll_for_manager,
    ws_resync_clear_hook=_bapi_ws_bridge.clear_ws_rest_resync_for_manager,
)
CachePriceManagerBase.bind_current_price_getter(get_current_price_manager)
CacheTrendManagerBase.bind_service_getters(
    cache24_getter=lambda: get_cache_manager("24Price"),
    current_price_getter=get_current_price_manager,
    api_client=api,
)
if hasattr(MarketDataProvider, "bind_ssot_cache_readers"):
    MarketDataProvider.bind_ssot_cache_readers(
        current_price_reader=get_active_current_price_manager,
        short_trend_reader=get_active_short_trend_manager,
    )


def ensure_account_cache_readers(status):
    """Synchronize this process with both versions in a fresh health marker."""
    try:
        order_version = status.order_cache_version
        trade_version = status.trade_cache_version
        if not order_version or not trade_version:
            raise ValueError("health marker has no durable cache versions")
        get_cache_manager("Order", start_sync=False).ensure_persisted_version(
            order_version
        )
        get_cache_manager("Trade", start_sync=False).ensure_persisted_version(
            trade_version
        )
    except account_cache_health.AccountCacheNotReady:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise account_cache_health.AccountCacheNotReady(
            "account_cache_reader_not_current"
        ) from exc


# ── Real-time Binance user stream -> cache actions ────────────────────────────

_ws_bridge = _bapi_ws_bridge._ws_bridge


def _ws_cache_getter(name, symbols=None, *, start_sync=False):
    """Look up caches from WebSocket callbacks without spawning rogue polling loops in consumers."""
    try:
        if symbols is not None:
            return get_cache_manager(name, symbols=symbols, start_sync=start_sync)
        return get_cache_manager(name, start_sync=start_sync)
    except TypeError:
        if symbols is not None:
            return get_cache_manager(name, symbols)
        return get_cache_manager(name)


def _upsert_order_from_execution_report(event):
    return _bapi_ws_bridge.upsert_order_from_execution_report(
        event, cache_getter=_ws_cache_getter
    )


def _append_trade_from_execution_report(event):
    return _bapi_ws_bridge.append_trade_from_execution_report(
        event, cache_getter=_ws_cache_getter
    )


def _persist_ws_updated_caches(event_type):
    _bapi_ws_bridge.persist_ws_updated_caches(
        event_type, cache_getter=_ws_cache_getter
    )


refresh_symbol_in_cache = _bapi_ws_bridge.refresh_symbol_in_cache


def _handle_binance_ws_event(event):
    _bapi_ws_bridge.handle_binance_ws_event(
        event,
        cache_getter=_ws_cache_getter,
        log_enabled=WS_EVENT_LOG_ENABLED,
        upsert_order_fn=_upsert_order_from_execution_report,
        append_trade_fn=_append_trade_from_execution_report,
        persist_fn=_persist_ws_updated_caches,
    )


def enable_real_ws_event_sync():
    global _ws_bridge
    _bapi_ws_bridge._ws_bridge = _ws_bridge
    _ws_bridge = _bapi_ws_bridge.enable_real_ws_event_sync(
        on_event=_handle_binance_ws_event,
        on_available=_mark_ws_available,
        on_healthy=_mark_ws_event_received,
        on_unhealthy=_mark_ws_unhealthy,
        loss_timeout_sec=WS_LOSS_TIMEOUT_SEC,
    )
    return _ws_bridge


def _initialize_once():
    _bapi_ws_bridge.initialize_ws_bridge_once(enable_fn=enable_real_ws_event_sync)


if __name__ == "__main__":
    single_instance("cacheManager")
    account_cache_health.enable_writer()
    for _account_cache_name in ("Order", "Trade"):
        _acc_mgr = get_cache_manager(_account_cache_name, start_sync=False)
        CacheFactory._sync_started.add(_account_cache_name)
        _ver = str(
            getattr(_acc_mgr, "_persisted_data_version", "")
            or getattr(_acc_mgr, "_loaded_data_version", "")
            or ""
        )
        _cursors = (
            getattr(_acc_mgr, "_rest_sync_cursor_per_symbol", None)
            or getattr(_acc_mgr, "fetchtime_per_symbol", None)
            or {}
        )
        _sync_ms = (
            min(
                (
                    int(v)
                    for v in _cursors.values()
                    if isinstance(v, (int, float)) and v > 0
                ),
                default=0,
            )
            if isinstance(_cursors, dict)
            else 0
        )
        if (
            _ver
            and _sync_ms > 0
            and (int(time.time() * 1000) - _sync_ms)
            <= int(account_cache_health.MAX_AGE_SEC * 1000)
        ):
            account_cache_health.record_successful_sync(
                _account_cache_name, _ver, now_ms=_sync_ms
            )
    _initialize_once()
    threads = []
    _nb_poller = None
    _trend_mgr = None

    _trend_syms, _nb_syms, _nb_provider_names = _default_price_tracking()
    if _nb_syms:
        builtins.print(
            f"[cacheManager] tracking non-Binance instruments from instruments.conf: {_nb_syms}"
        )

    _trend_cpm = None
    try:
        from binance_api import bapi_ws

        _trend_cpm = get_current_price_manager(
            ws_manager=bapi_ws.get_ws_manager(),
            sync_ts=0.8,
            start_sync=False,
        )
    except Exception as _ws_init_err:
        print(f"[cacheManager] Warning pre-attaching price WS: {_ws_init_err}")

    for name, config in CacheFactory._CONFIG.items():
        if config.get("read_only"):
            continue
        if name == "365Price":
            _arch_syms = tracked_price_instruments(role="archive")[0]
            cache = get_cache_manager(
                name, symbols=_arch_syms or None, start_sync=False
            )
        else:
            cache = get_cache_manager(name, start_sync=False)
        interval = (
            cache.sync_ts
            if name == "CurrentPrice" and hasattr(cache, "sync_ts")
            else config["sync_ts"]()
        )

        if isinstance(cache, dict):
            for idx, manager in enumerate(cache.values()):
                if name == "365Price":
                    manager._maint_stagger_sec = float(idx * 300)
                threads.append(manager.periodic_sync(interval))
        else:
            threads.append(cache.periodic_sync(interval))
        CacheFactory._sync_started.add(name)

    try:
        if _trend_cpm is None:
            from binance_api import bapi_ws

            _trend_cpm = get_current_price_manager(
                ws_manager=bapi_ws.get_ws_manager(),
                sync_ts=0.8,
            )
        _trend_cache24 = CacheFactory.get("24Price")
        _trend_cache365 = CacheFactory._instances.get("365Price") or {}
        if isinstance(_trend_cache365, dict):
            for _c365 in _trend_cache365.values():
                _trend_cpm.subscribe_price(_c365)
        _trend_mgr = get_short_trend_manager(writer=True)
        _trend_mgr.start_computation(
            _trend_cache24, _trend_cpm, run_full_eval=True
        )
        print("⚙️ cacheManager: full trend computation started (cache_short_trend.json).")
        if _nb_syms:
            _nb_poller = _start_nonbinance_trend_poller(
                _trend_cpm,
                _nb_syms,
                interval_sec=10,
                provider_names=_nb_provider_names,
            )
    except Exception as e:
        print(f"[cacheManager] Cannot start the trend computation: {e}")

    _stop_requested = threading.Event()

    def _request_stop(_signum=None, _frame=None):
        _stop_requested.set()

    import signal as _signal

    _signal.signal(_signal.SIGTERM, _request_stop)
    _signal.signal(_signal.SIGINT, _request_stop)

    try:
        touch_module_heartbeat(
            "cacheManager",
            interval_sec=30.0,
            force=True,
            symbols=list(_trend_syms or ()),
        )
        while not _stop_requested.is_set() and any(t.is_alive() for t in threads):
            touch_module_heartbeat(
                "cacheManager",
                interval_sec=30.0,
                symbols=list(_trend_syms or ()),
            )
            if _stop_requested.wait(timeout=5.0):
                break
    except KeyboardInterrupt:
        print("Stopped manually.")
    finally:
        account_cache_health.disable_writer()
        print("Cleanup / releasing resources...")
        shutdown_deadline = time.monotonic() + 2.5
        try:
            from binance_api import bapi_ws as _bws

            _bws_mgr = getattr(_bws, "bapi_ws_manager", None)
        except Exception:
            _bws_mgr = None
        for _ws_obj in (_ws_bridge, _bws_mgr):
            if _ws_obj is not None and hasattr(_ws_obj, "_stop_event"):
                try:
                    _ws_obj._stop_event.set()
                except Exception:
                    pass
        if _nb_poller is not None and hasattr(_nb_poller, "stop_event"):
            _nb_poller.stop_event.set()
            for _nb_ws in getattr(_nb_poller, "ws_streams", ()):
                if hasattr(_nb_ws, "_stop_event"):
                    try:
                        _nb_ws._stop_event.set()
                    except Exception:
                        pass
        if _trend_mgr is not None and hasattr(_trend_mgr, "_stop_event"):
            _trend_mgr._stop_event.set()
        _c365_Shutdown_managers = list(
            CacheFactory._iter_managers(
                CacheFactory._instances.get("365Price") or {}
            )
        )
        for _inst in list(CacheFactory._instances.values()):
            for _mgr in CacheFactory._iter_managers(_inst):
                if hasattr(_mgr, "_stop_event"):
                    _mgr._stop_event.set()
        if _current_price_instance is not None and hasattr(
            _current_price_instance, "_stop_event"
        ):
            _current_price_instance._stop_event.set()
        if _nb_poller is not None:
            _nb_poller.stop(
                timeout=min(0.5, max(0.05, shutdown_deadline - time.monotonic()))
            )
        if _trend_mgr is not None:
            _trend_mgr.shutdown(
                timeout=min(0.8, max(0.05, shutdown_deadline - time.monotonic()))
            )
        CacheFactory.shutdown_all(
            timeout=min(1.0, max(0.05, shutdown_deadline - time.monotonic()))
        )
        for _c365_mgr in _c365_Shutdown_managers:
            try:
                _c365_mgr.save_state_to_file_if_enabled()
            except Exception:
                pass
        if _current_price_instance is not None:
            _current_price_instance.shutdown(
                timeout=min(0.3, max(0.05, shutdown_deadline - time.monotonic()))
            )
        for _ws_obj in (_ws_bridge, _bws_mgr):
            if _ws_obj is not None:
                try:
                    _ws_obj.stop(
                        timeout=min(
                            0.3, max(0.05, shutdown_deadline - time.monotonic())
                        )
                    )
                except Exception:
                    pass
        from binance_api import bapi_client

        bapi_client.stop_periodic_resync()
