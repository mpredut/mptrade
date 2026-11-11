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
                           and venue-agnostic daemon supervisor entrypoint.
"""

from __future__ import annotations

import builtins
import os
import threading
import time
from typing import Optional

from botcore import (
    _parse_bool_flag,
    load_dotenv,
    safe_nonneg_float,
    single_instance,
    touch_module_heartbeat,
)
from cacheInterface import (
    CacheManagerInterface,
    _CONFIG_ROOT,
    _coerce_symbol_list,
    _monotonic_time,
)
from cacheaccount import (
    ASSET_VALUE_SYNC_INTERVAL_SEC,
    CacheAccountManagerBase,
    CacheAssetValueManager,
    CacheOrderManager,
    CacheTradeManager,
    ORDER_SYNC_INTERVAL_SEC,
    TRADE_SYNC_INTERVAL_SEC,
    _READONLY_ACCOUNT_READERS,
    account_cache_health,
    api,
    attach_hyperliquid_user_fills_bridge,
    bind_account_cache_getter,
    disable_account_cache_daemon_writer,
    ensure_account_cache_readers,
    get_asset_value_manager,
    get_binance_order_manager,
    get_trade_manager,
    request_stop_account_ws_streams,
    start_account_cache_daemon,
    stop_account_ws_streams,
)
from cacheprice import (
    CM_LONG_ARCHIVE_FLUSH_SEC,
    CM_LONG_ARCHIVE_MEMORY_ROWS,
    CM_LONG_ARCHIVE_MONTHS,
    CM_LONG_ARCHIVE_SAMPLE_SEC,
    CURRENT_PRICE_SYNC_INTERVAL_SEC,
    Cache24PriceManager,
    Cache365PriceManager,
    CacheCurrentPriceManager,
    CachePriceManagerBase,
    CacheSparsePriceManager,
    HyperliquidPriceStream,
    KrakenPriceStream,
    PRICE_24_SYNC_INTERVAL_SEC,
    SPARSE_PRICE_SYNC_INTERVAL_SEC,
    _READONLY_PRICE_READERS,
    _start_nonbinance_trend_poller,
    attach_default_price_ws_stream,
    get_24_price_manager,
    get_current_price_manager,
    get_sparse_price_manager,
    request_stop_price_ws_streams,
    stop_price_ws_streams,
)
from cachetrend import (
    CM_DYNAMIC_WINDOW_MAX_SEC,
    CM_DYNAMIC_WINDOW_MIN_SEC,
    CM_PUBLISHED_TREND_WINDOWS_SEC,
    CacheLongTrendManager,
    CacheShortTrendManager,
    CacheTrendManagerBase,
    LONG_TREND_SYNC_INTERVAL_SEC,
    _READONLY_TREND_READERS,
    get_long_trend_manager,
    get_short_trend_manager,
)
import instrument_registry as sym
from instrument_registry import tracked_price_instruments
from providers.base import MarketDataProvider
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
)

load_dotenv()
HyperliquidPriceStream.register_stream_init_hook(attach_hyperliquid_user_fills_bridge)

_PRICE_TREND_CACHES = frozenset(
    {
        "CurrentPrice",
        "24Price",
        "365Price",
        "SparsePrice",
        "LongTrend",
        "ShortTrend",
    }
)


class CacheFactory:
    """Single registry and singleton factory for all domain cache managers."""

    _lock = threading.RLock()
    _instances = {}
    _sync_started = set()
    _CONFIG = {}

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
    def _resync_readonly_consumer_instance(
        cls, name: str, instance, symbols=None
    ) -> None:
        if name in ("CurrentPrice", "ShortTrend") or instance is None:
            return
        if isinstance(instance, dict) and symbols is not None:
            wanted = {str(s).strip().upper() for s in symbols if s}
            managers = [
                mgr
                for sym_key, mgr in instance.items()
                if str(sym_key).strip().upper() in wanted
            ]
        else:
            managers = cls._iter_managers(instance)
        for manager in managers:
            if (
                getattr(manager, "data_sync_mode", CacheManagerInterface.MODE_DISK_READER)
                == CacheManagerInterface.MODE_DISK_READER
                and getattr(manager, "read_only", False)
                and not getattr(manager, "save_state", False)
                and not getattr(manager, "_is_canonical_account_cache", lambda: False)()
                and hasattr(manager, "resync_mem_file")
            ):
                try:
                    manager.resync_mem_file()
                except Exception:
                    pass

    @classmethod
    def _apply_mode_to_instance(
        cls,
        instance,
        resolved_mode: str,
        save_state_flag: bool,
        *,
        cfg_read_only: bool = False,
    ) -> None:
        if instance is None or cfg_read_only:
            return
        for manager in cls._iter_managers(instance):
            if save_state_flag or resolved_mode == CacheManagerInterface.MODE_PRODUCER:
                if hasattr(manager, "become_writer"):
                    manager.become_writer()
                else:
                    manager.save_state = True
                    manager.read_only = False
                    manager.data_sync_mode = CacheManagerInterface.MODE_PRODUCER
            elif (
                resolved_mode == CacheManagerInterface.MODE_LIVE_READER
                and not getattr(manager, "save_state", False)
                and not getattr(manager, "writer", False)
            ):
                manager.data_sync_mode = CacheManagerInterface.MODE_LIVE_READER

    @classmethod
    def get(
        cls,
        name,
        symbols=None,
        *,
        data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
        provider_names=None,
        ws_manager=None,
        sync_ts=None,
        read_only=None,
        filename=None,
    ):
        resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(
            data_sync_mode
        )
        if (
            name == "CurrentPrice"
            and ws_manager is not None
            and resolved_mode == CacheManagerInterface.MODE_DISK_READER
        ):
            resolved_mode = CacheManagerInterface.MODE_LIVE_READER
        if isinstance(symbols, str):
            symbols = _coerce_symbol_list(symbols)
        inst = cls._instances.get(name)
        if (
            inst is not None
            and symbols is None
            and not provider_names
            and ws_manager is None
            and sync_ts is None
            and not save_state_flag
            and (
                not start_periodic
                or name in cls._sync_started
                or name == "ShortTrend"
                or (
                    name == "CurrentPrice"
                    and (thread := getattr(inst, "thread", None)) is not None
                    and thread.is_alive()
                    and not getattr(inst, "read_only", True)
                )
            )
        ):
            if resolved_mode == CacheManagerInterface.MODE_LIVE_READER:
                cls._apply_mode_to_instance(inst, resolved_mode, False)
            elif (
                resolved_mode == CacheManagerInterface.MODE_DISK_READER
                and name not in cls._sync_started
            ):
                cls._resync_readonly_consumer_instance(name, inst)
            return inst
        with cls._lock:
            if name not in cls._CONFIG:
                raise ValueError(f"Unknown cache type: {name}")
            if provider_names is not None and name != "CurrentPrice":
                raise ValueError("provider_names is supported only for CurrentPrice")

            cfg_read_only = bool(cls._CONFIG[name].get("read_only", False)) and name != "ShortTrend"
            resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(
                data_sync_mode,
                config_read_only=cfg_read_only,
            )
            if (
                name == "CurrentPrice"
                and ws_manager is not None
                and resolved_mode == CacheManagerInterface.MODE_DISK_READER
            ):
                resolved_mode = CacheManagerInterface.MODE_LIVE_READER
            if name == "ShortTrend":
                start_periodic = False

            if name in cls._instances and symbols is not None and name not in ("CurrentPrice", "ShortTrend"):
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
                inst = cls._instances[name]
                if sync_ts is not None:
                    inst.sync_ts = sync_ts
                if provider_names:
                    inst.bind_providers(provider_names)
                if ws_manager is not None:
                    inst.attach_ws_manager(ws_manager)

            created = name not in cls._instances
            eff_sync_ts = sync_ts
            if created:
                cls._sync_started.discard(name)
                config = cls._CONFIG[name]
                manager_class = config["class"]
                if eff_sync_ts is None:
                    eff_sync_ts = config["sync_ts"]()
                extra_kwargs = {
                    key: value
                    for key, value in config.items()
                    if key not in {"class", "filename", "sync_ts", "prefix", "ext"}
                }
                extra_kwargs.pop("read_only", None)
                eff_read_only = (
                    _parse_bool_flag(read_only, default=False)
                    if read_only is not None
                    else (cfg_read_only or (not save_state_flag and not start_periodic))
                )
                uses_shared_symbols = MarketDataProvider.uses_shared_registry_symbols(
                    extra_kwargs.get("venue")
                )
                if symbols is None:
                    if name == "AssetValue":
                        symbols = ["TOTAL"]
                    elif name in _PRICE_TREND_CACHES:
                        symbols, _, _default_providers = tracked_price_instruments()
                        if (
                            name == "CurrentPrice"
                            and provider_names is None
                            and _default_providers
                        ):
                            provider_names = _default_providers
                    elif not uses_shared_symbols:
                        symbols = None
                    else:
                        symbols = sym.symbols

                eff_filename = filename if filename is not None else config["filename"]
                if name == "CurrentPrice":
                    instance = manager_class(
                        sync_ts=eff_sync_ts,
                        symbols=symbols,
                        filename=eff_filename,
                        ws_manager=ws_manager,
                        provider_names=provider_names,
                        read_only=eff_read_only,
                        data_sync_mode=resolved_mode,
                    )
                    eff_sync_ts = instance.sync_ts
                elif name == "ShortTrend":
                    instance = manager_class(
                        symbols,
                        eff_filename,
                        data_sync_mode=resolved_mode,
                        cache24_getter=lambda: get_cache_manager("24Price"),
                        current_price_getter=get_current_price_manager,
                    )
                elif "prefix" in config:
                    prefix, ext = config["prefix"], config["ext"]
                    instance = {
                        s: manager_class(
                            sync_ts=eff_sync_ts,
                            filename=f"{prefix}{s}.{ext}",
                            symbols=[s],
                            read_only=eff_read_only,
                            data_sync_mode=resolved_mode,
                            **extra_kwargs,
                        )
                        for s in symbols
                    }
                else:
                    instance = manager_class(
                        sync_ts=eff_sync_ts,
                        filename=eff_filename,
                        symbols=symbols,
                        read_only=eff_read_only,
                        data_sync_mode=resolved_mode,
                        **extra_kwargs,
                    )

                if name != "ShortTrend":
                    for manager in cls._iter_managers(instance):
                        manager.data_sync_mode = resolved_mode
                        if save_state_flag:
                            manager.save_state = True
                            manager.read_only = False

                cls._instances[name] = instance
            else:
                cls._apply_mode_to_instance(
                    cls._instances[name],
                    resolved_mode,
                    save_state_flag,
                    cfg_read_only=cfg_read_only,
                )

            if (
                not created
                and resolved_mode == CacheManagerInterface.MODE_DISK_READER
                and not start_periodic
                and name not in cls._sync_started
            ):
                cls._resync_readonly_consumer_instance(
                    name, cls._instances[name], symbols=symbols
                )

            if (
                start_periodic
                and not cfg_read_only
                and (name not in cls._sync_started or save_state_flag or sync_ts is not None)
            ):
                if eff_sync_ts is None:
                    eff_sync_ts = (
                        cls._instances[name].sync_ts
                        if name == "CurrentPrice"
                        and hasattr(cls._instances[name], "sync_ts")
                        else cls._CONFIG[name]["sync_ts"]()
                    )
                for manager in cls._iter_managers(cls._instances[name]):
                    manager.periodic_sync(eff_sync_ts, save_state_flag)
                cls._sync_started.add(name)

            return cls._instances[name]

    @classmethod
    def shutdown_all(cls, timeout=5.0):
        """Stop every factory-owned synchronization loop and clear the registry."""
        with cls._lock:
            managers = []
            seen_ids = set()
            for value in cls._instances.values():
                for manager in cls._iter_managers(value):
                    if id(manager) not in seen_ids:
                        seen_ids.add(id(manager))
                        managers.append(manager)
            cls._instances = {}
            cls._sync_started = set()
            _READONLY_TREND_READERS.clear()
            _READONLY_PRICE_READERS.clear()
            _READONLY_ACCOUNT_READERS.clear()
            for manager in managers:
                req_stop = getattr(manager, "request_stop", None)
                if callable(req_stop):
                    req_stop()
            eff_timeout = safe_nonneg_float(timeout, default=5.0)
            deadline = _monotonic_time() + eff_timeout
            results = [
                manager.shutdown(timeout=max(0.0, deadline - _monotonic_time()))
                for manager in managers
                if hasattr(manager, "shutdown")
            ]
            return all(results) if results else True

    @classmethod
    def remove(cls, name, timeout=5.0):
        """Remove a singleton cleanly after stopping all threads it owns."""
        with cls._lock:
            value = cls._instances.get(name)
            cls._sync_started.discard(name)
            _READONLY_TREND_READERS.pop(name, None)
            _READONLY_PRICE_READERS.pop(name, None)
            _READONLY_ACCOUNT_READERS.pop(name, None)
            if value is None:
                return True
            results = [
                manager.shutdown(timeout=timeout)
                for manager in cls._iter_managers(value)
                if hasattr(manager, "shutdown")
            ]
            stopped = all(results) if results else True
            if stopped and cls._instances.get(name) is value:
                cls._instances.pop(name, None)
            return stopped


# Register all canonical domain cache implementations in one place.
CacheFactory.register(
    "BinanceTrade",
    CacheTradeManager,
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "BinanceOrder",
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
    sync_ts=lambda: CM_LONG_ARCHIVE_FLUSH_SEC,
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
    "ShortTrend",
    CacheShortTrendManager,
    sync_ts=0,
    read_only=True,
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


def get_cache_manager(
    name,
    symbols=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
):
    return CacheFactory.get(
        name,
        symbols,
        data_sync_mode=data_sync_mode,
    )


get_cache_manager._is_ssot_cache_manager_getter = True  # type: ignore[attr-defined]
with CacheFactory._lock:
    for _reader_map in (_READONLY_TREND_READERS, _READONLY_PRICE_READERS, _READONLY_ACCOUNT_READERS):
        for _reader_key, _reader_inst in list(_reader_map.items()):
            CacheFactory._instances.setdefault(_reader_key, _reader_inst)


# Inject readers and runtime services across the domain base classes and MarketDataProvider.
bind_account_cache_getter(get_cache_manager)
CachePriceManagerBase.bind_current_price_getter(get_current_price_manager)
CacheTrendManagerBase.bind_service_getters(
    cache24_getter=lambda: get_cache_manager("24Price"),
    current_price_getter=get_current_price_manager,
)
if hasattr(MarketDataProvider, "bind_ssot_cache_readers"):
    MarketDataProvider.bind_ssot_cache_readers(
        current_price_reader=lambda: CacheFactory._instances.get("CurrentPrice"),
        cache_manager_getter=get_cache_manager,
        short_trend_getter=get_short_trend_manager,
        ensure_account_readers_fn=ensure_account_cache_readers,
        current_price_cls=CacheCurrentPriceManager,
        price24_cls=Cache24PriceManager,
        trade_cache_cls=CacheTradeManager,
    )


if __name__ == "__main__":
    single_instance("cacheManager")
    start_account_cache_daemon(
        cache_getter=get_cache_manager,
        sync_started_set=CacheFactory._sync_started,
    )
    threads = []
    _nb_poller = None
    _trend_mgr = None

    _trend_syms, _nb_syms, _nb_provider_names = tracked_price_instruments()
    if _nb_syms:
        builtins.print(
            f"[cacheManager] tracking non-Binance instruments from instruments.conf: {_nb_syms}"
        )

    _trend_cpm = None
    try:
        _trend_cpm = attach_default_price_ws_stream(
            get_current_price_manager,
            sync_ts=0.8,
        )
        if _nb_syms and _trend_cpm is not None:
            _nb_poller = _start_nonbinance_trend_poller(
                _trend_cpm,
                _nb_syms,
                interval_sec=10,
                provider_names=_nb_provider_names,
            )
    except Exception as _ws_init_err:
        print(f"[cacheManager] Warning pre-attaching price WS: {_ws_init_err}")

    for name, config in CacheFactory._CONFIG.items():
        if config.get("read_only"):
            continue
        if name == "365Price":
            _arch_syms = tracked_price_instruments(role="archive")[0]
            cache = get_cache_manager(name, symbols=_arch_syms or None)
        else:
            cache = get_cache_manager(name)
        interval = (
            cache.sync_ts
            if name == "CurrentPrice" and hasattr(cache, "sync_ts")
            else config["sync_ts"]()
        )

        if isinstance(cache, dict):
            for idx, manager in enumerate(cache.values()):
                if name == "365Price":
                    manager._maint_stagger_sec = float(idx * 300)
                threads.append(manager.periodic_sync(interval, save_state=True))
        else:
            threads.append(cache.periodic_sync(interval, save_state=True))
        CacheFactory._sync_started.add(name)

    try:
        if _trend_cpm is None:
            _trend_cpm = attach_default_price_ws_stream(
                get_current_price_manager,
                sync_ts=0.8,
                data_sync_mode="producer",
            )
        _trend_cache24 = CacheFactory.get("24Price")
        _trend_cache365 = CacheFactory._instances.get("365Price") or {}
        if isinstance(_trend_cache365, dict):
            for _c365 in _trend_cache365.values():
                _trend_cpm.subscribe_price(_c365)
        _trend_mgr = get_short_trend_manager(data_sync_mode="producer")
        _trend_mgr.start_computation(
            _trend_cache24, _trend_cpm, run_full_eval=True
        )
        print("⚙️ cacheManager: full trend computation started (cache_short_trend.json).")
        if _nb_syms and _nb_poller is None:
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
        disable_account_cache_daemon_writer()
        print("Cleanup / releasing resources...")
        shutdown_deadline = time.monotonic() + 2.5
        request_stop_account_ws_streams()
        request_stop_price_ws_streams()
        if _nb_poller is not None and hasattr(_nb_poller, "request_stop"):
            _nb_poller.request_stop()
        if _trend_mgr is not None and hasattr(_trend_mgr, "request_stop"):
            _trend_mgr.request_stop()
        _c365_Shutdown_managers = list(
            CacheFactory._iter_managers(
                CacheFactory._instances.get("365Price") or {}
            )
        )
        for _inst in list(CacheFactory._instances.values()):
            for _mgr in CacheFactory._iter_managers(_inst):
                if hasattr(_mgr, "request_stop"):
                    _mgr.request_stop()
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
        stop_account_ws_streams(shutdown_deadline)
        stop_price_ws_streams(shutdown_deadline)

