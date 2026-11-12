"""Dependency-inversion runtime hook bindings and SSOT cache bridge mixin for MarketDataProvider."""

import contextlib
import importlib
import sys
import threading
from typing import Any, List, Optional

from botcore import (
    PRICE_FRESHNESS_MAX_AGE_SEC,
    _parse_bool_flag,
    is_shared_disk_cache_enabled,
    require_nonneg_float,
    resolve_patched_attr,
    resolve_positive_max_age,
    safe_nonneg_int,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    cache_path,
    canonical_symbol_key,
    clean_symbol_key,
    split_symbol_assets,
)

_LIVE_FETCH_LOCAL = threading.local()


@contextlib.contextmanager
def live_price_fetch_context():
    """Mark the current thread as performing a live venue price poll for cacheManager."""
    depth = safe_nonneg_int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0, default=0)
    _LIVE_FETCH_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _LIVE_FETCH_LOCAL.depth = max(0, depth)


def is_in_live_price_fetch() -> bool:
    """Return True when called inside a cacheManager remote price poll."""
    return safe_nonneg_int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0, default=0) > 0


class ProviderSsotBridgeMixin:
    """Level-2 mixin isolating runtime upward-layer hook bindings and SSOT cache operations."""

    _DEFAULT_FILLS_CACHE_FILENAME: Optional[str] = None
    _FILLS_CACHE_PATH_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_DEFAULT_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_MAX_STALE_S: float = 60.0
    _USES_SSOT_MARKET_CACHE: bool = True
    _fills_match_fn: Any = None
    _fills_normalize_fn: Any = None
    _ssot_current_price_reader: Any = None
    _ssot_cache_manager_getter: Any = None
    _ssot_short_trend_getter: Any = None
    _ssot_ensure_account_readers_fn: Any = None
    _ssot_current_price_cls: Any = None
    _ssot_price24_cls: Any = None
    _ssot_trade_cache_cls: Any = None
    _policy_weight_limit_fn: Any = None
    _policy_window_cfg_fn: Any = None
    _policy_window_reference_fn: Any = None
    _policy_resolve_trade_weight_fn: Any = None
    _policy_compute_weight_capped_qty_fn: Any = None
    _bound_instrument_cls: Any = None
    _bound_load_tradable_instruments_fn: Any = None
    _bound_tracked_order_lifecycle_cls: Any = None
    _retry_discard_refused_intent_fn: Any = None
    _retry_discard_superseded_symbol_fn: Any = None
    _retry_accepted_owner_map_fn: Any = None
    _retry_resolve_accepted_order_fn: Any = None
    _telemetry_intervals_resolver: Any = None

    @classmethod
    def bind_instrument_class(
        cls, instrument_cls: Any, *, load_tradable_instruments_fn: Any = None
    ) -> None:
        """Bind the ``Instrument`` gateway class and loader from the higher Order Gateway layer."""
        if instrument_cls is not None:
            cls._bound_instrument_cls = instrument_cls
        if callable(load_tradable_instruments_fn):
            cls._bound_load_tradable_instruments_fn = staticmethod(load_tradable_instruments_fn)

    @classmethod
    def load_tradable_instruments(cls, *args: Any, **kwargs: Any) -> dict:
        """Invoke the bound ``instrument.load_tradable_instruments`` loader without upward static imports."""
        if cls._bound_load_tradable_instruments_fn is None:
            importlib.import_module("instrument")
        fn = resolve_patched_attr(
            "load_tradable_instruments",
            cls._bound_load_tradable_instruments_fn,
            modules=("instrument",),
            require_callable=True,
        )
        if fn is None:
            raise RuntimeError("load_tradable_instruments is not bound")
        return fn(*args, **kwargs)

    @classmethod
    def resolve_instrument_class(cls) -> Any:
        """Return the bound ``Instrument`` class (respecting ``instrument`` monkeypatches)."""
        if cls._bound_instrument_cls is None:
            importlib.import_module("instrument")
        inst_cls = resolve_patched_attr(
            "Instrument",
            cls._bound_instrument_cls,
            modules=("instrument",),
            require_callable=True,
        )
        if inst_cls is None:
            raise RuntimeError("Instrument class is not bound")
        return inst_cls

    @classmethod
    def bind_ssot_cache_readers(
        cls,
        *,
        current_price_reader: Any = None,
        cache_manager_getter: Any = None,
        short_trend_getter: Any = None,
        ensure_account_readers_fn: Any = None,
        current_price_cls: Any = None,
        price24_cls: Any = None,
        trade_cache_cls: Any = None,
    ) -> None:
        """Bind read-only SSOT cache getters and snapshot classes from ``cacheManager`` without upward imports."""
        if current_price_reader is not None:
            cls._ssot_current_price_reader = staticmethod(current_price_reader)
        if cache_manager_getter is not None:
            cls._ssot_cache_manager_getter = staticmethod(cache_manager_getter)
        if short_trend_getter is not None:
            cls._ssot_short_trend_getter = staticmethod(short_trend_getter)
        if ensure_account_readers_fn is not None:
            cls._ssot_ensure_account_readers_fn = staticmethod(ensure_account_readers_fn)
        if current_price_cls is not None:
            cls._ssot_current_price_cls = current_price_cls
        if price24_cls is not None:
            cls._ssot_price24_cls = price24_cls
        if trade_cache_cls is not None:
            cls._ssot_trade_cache_cls = trade_cache_cls

    @classmethod
    def _ensure_ssot_cache_bindings(cls) -> None:
        if cls._ssot_cache_manager_getter is None:
            importlib.import_module("cacheManager")

    @classmethod
    def ensure_ssot_account_cache_readers(cls, status: Any) -> Any:
        """Synchronize account-cache reader versions via bound hook (respecting ``cacheManager`` monkeypatches)."""
        cls._ensure_ssot_cache_bindings()
        fn = resolve_patched_attr(
            "ensure_account_cache_readers",
            cls._ssot_ensure_account_readers_fn,
            modules=("cacheManager",),
            require_callable=True,
        )
        return fn(status) if callable(fn) else None

    @classmethod
    def get_ssot_cache_manager(cls, cache_name: str, *args: Any, **kwargs: Any) -> Any:
        """Return a named SSOT cache manager via bound hook (respecting ``cacheManager`` monkeypatches)."""
        cls._ensure_ssot_cache_bindings()
        fn = resolve_patched_attr(
            "get_cache_manager",
            cls._ssot_cache_manager_getter,
            modules=("cacheManager",),
            require_callable=True,
        )
        if not callable(fn):
            raise RuntimeError("SSOT cache manager getter is not bound")
        if not args and not kwargs:
            return fn(cache_name)
        try:
            return fn(cache_name, *args, **kwargs)
        except TypeError:
            if "symbols" in kwargs and not args:
                return fn(cache_name, kwargs["symbols"])
            raise

    @classmethod
    def get_ssot_short_trend_manager(cls) -> Any:
        """Return the ShortTrend cache manager via bound hook (respecting ``cacheManager`` / ``cachetrend`` monkeypatches)."""
        ct_mod = (
            importlib.import_module("cachetrend")
            if cls._ssot_short_trend_getter is None
            else sys.modules.get("cachetrend")
        )
        orig_fn = getattr(ct_mod, "_ORIG_GET_SHORT_TREND_MANAGER", None) if ct_mod is not None else None
        default_fn = (
            cls._ssot_short_trend_getter
            or orig_fn
            or (getattr(ct_mod, "get_short_trend_manager", None) if ct_mod is not None else None)
        )
        fn = resolve_patched_attr(
            "get_short_trend_manager",
            default_fn,
            defaults=(cls._ssot_short_trend_getter, orig_fn),
            modules=("cacheManager", "cachetrend"),
            require_callable=True,
        )
        return fn() if callable(fn) else None

    @classmethod
    def bind_order_guard_hooks(
        cls,
        *,
        weight_limit_fn: Any = None,
        window_cfg_fn: Any = None,
        window_reference_fn: Any = None,
        resolve_trade_weight_fn: Any = None,
        compute_weight_capped_qty_fn: Any = None,
    ) -> None:
        """Bind order-guard policy callbacks from the higher Order Gateway layer."""
        if weight_limit_fn is not None:
            cls._policy_weight_limit_fn = staticmethod(weight_limit_fn)
        if window_cfg_fn is not None:
            cls._policy_window_cfg_fn = staticmethod(window_cfg_fn)
        if window_reference_fn is not None:
            cls._policy_window_reference_fn = staticmethod(window_reference_fn)
        if resolve_trade_weight_fn is not None:
            cls._policy_resolve_trade_weight_fn = staticmethod(resolve_trade_weight_fn)
        if compute_weight_capped_qty_fn is not None:
            cls._policy_compute_weight_capped_qty_fn = staticmethod(compute_weight_capped_qty_fn)

    @staticmethod
    def _bind_order_guard_hook(method_name: str, attr_name: str, fallback_attr: str):
        def _method(cls, *args: Any, **kwargs: Any) -> Any:
            if cls._policy_weight_limit_fn is None or cls._policy_resolve_trade_weight_fn is None:
                importlib.import_module("order_guard")
            fn = resolve_patched_attr(
                attr_name,
                getattr(cls, fallback_attr, None),
                modules=("order_guard",),
                require_callable=True,
            )
            if not callable(fn):
                raise RuntimeError(f"order_guard.{attr_name} is not bound")
            return fn(*args, **kwargs)

        _method.__name__ = method_name
        _method.__qualname__ = f"MarketDataProvider.{method_name}"
        return classmethod(_method)

    resolve_order_guard_trade_weight = _bind_order_guard_hook(
        "resolve_order_guard_trade_weight",
        "resolve_trade_weight",
        "_policy_resolve_trade_weight_fn",
    )
    compute_order_guard_weight_capped_qty = _bind_order_guard_hook(
        "compute_order_guard_weight_capped_qty",
        "compute_weight_capped_qty",
        "_policy_compute_weight_capped_qty_fn",
    )

    @classmethod
    def resolve_tracked_order_lifecycle_class(cls) -> Any:
        """Return ``TrackedOrderLifecycle`` from ``order_lifecycle`` (respecting monkeypatches)."""
        if cls._bound_tracked_order_lifecycle_cls is None:
            mod = importlib.import_module("order_lifecycle")
            cls._bound_tracked_order_lifecycle_cls = getattr(mod, "TrackedOrderLifecycle", None)
        lifecycle_cls = resolve_patched_attr(
            "TrackedOrderLifecycle",
            cls._bound_tracked_order_lifecycle_cls,
            modules=("order_lifecycle",),
            require_callable=True,
        )
        if lifecycle_cls is None:
            raise RuntimeError("TrackedOrderLifecycle class is not bound")
        return lifecycle_cls

    @classmethod
    def bind_order_retry_hooks(
        cls,
        *,
        discard_refused_intent_fn: Any = None,
        discard_superseded_symbol_fn: Any = None,
        accepted_owner_map_fn: Any = None,
        resolve_accepted_order_fn: Any = None,
    ) -> None:
        """Bind persistent retry-outbox callbacks from ``order_retry`` without upward imports."""
        if discard_refused_intent_fn is not None:
            cls._retry_discard_refused_intent_fn = staticmethod(discard_refused_intent_fn)
        if discard_superseded_symbol_fn is not None:
            cls._retry_discard_superseded_symbol_fn = staticmethod(discard_superseded_symbol_fn)
        if accepted_owner_map_fn is not None:
            cls._retry_accepted_owner_map_fn = staticmethod(accepted_owner_map_fn)
        if resolve_accepted_order_fn is not None:
            cls._retry_resolve_accepted_order_fn = staticmethod(resolve_accepted_order_fn)

    @classmethod
    def _resolve_order_retry_hook(cls, attr_name: str, slot_name: str) -> Any:
        fallback = getattr(cls, slot_name, None)
        if fallback is None:
            importlib.import_module("order_retry")
            fallback = getattr(cls, slot_name, None)
        return resolve_patched_attr(
            attr_name,
            fallback,
            modules=("order_retry",),
            require_callable=True,
        )

    @classmethod
    def discard_refused_intent_retry_records(cls, intent_id: str) -> list[str]:
        """Discard ``refused`` retry-outbox records matching ``intent_id``."""
        fn = cls._resolve_order_retry_hook("discard_refused_intent_records", "_retry_discard_refused_intent_fn")
        return list(fn(intent_id)) if callable(fn) else []

    @classmethod
    def discard_superseded_symbol_retry_records(cls, *args: Any, **kwargs: Any) -> list[str]:
        """Discard superseded pending retry-outbox records for a symbol/side."""
        fn = cls._resolve_order_retry_hook("discard_superseded_symbol_records", "_retry_discard_superseded_symbol_fn")
        return list(fn(*args, **kwargs)) if callable(fn) else []

    @classmethod
    def get_accepted_retry_order_owner_map(cls) -> dict[str, str]:
        """Return accepted retry-outbox owner map via bound ``order_retry`` hook."""
        fn = cls._resolve_order_retry_hook("get_accepted_order_owner_map", "_retry_accepted_owner_map_fn")
        res = fn() if callable(fn) else {}
        return res if isinstance(res, dict) else {}

    @classmethod
    def resolve_accepted_retry_order(cls, symbol: str, order_id: Any) -> Any:
        """Resolve an accepted retry-outbox order via bound ``order_retry`` hook."""
        fn = cls._resolve_order_retry_hook("resolve_accepted_order", "_retry_resolve_accepted_order_fn")
        return fn(symbol, order_id) if callable(fn) else None

    @classmethod
    def resolve_unfilled_order_cancel_grace_sec(cls) -> float:
        """Return ``unfilled_order_cancel_grace_sec`` from telemetry SSOT (respecting monkeypatches)."""
        if cls._telemetry_intervals_resolver is None:
            mod = importlib.import_module("intelligence.macro_sentiment_ssot")
            fn_init = getattr(mod, "resolve_derived_telemetry_intervals", None)
            if callable(fn_init):
                cls._telemetry_intervals_resolver = staticmethod(fn_init)
        fn = resolve_patched_attr(
            "resolve_derived_telemetry_intervals",
            cls._telemetry_intervals_resolver,
            modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot"),
            require_callable=True,
        )
        if not callable(fn):
            raise RuntimeError("resolve_derived_telemetry_intervals is not bound")
        return require_nonneg_float(
            fn()["unfilled_order_cancel_grace_sec"],
            name="unfilled_order_cancel_grace_sec",
        )

    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        """Return venue-scoped symbol keys to look up in SSOT cacheManager stores."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return ()
        raw_u = symbol.strip().upper()
        cleaned = clean_symbol_key(raw_u)
        parts = split_symbol_assets(raw_u)
        canon = canonical_symbol_key(raw_u) if parts is not None else ""
        return tuple(dict.fromkeys(c for c in (cleaned, raw_u, canon) if c))

    def _allow_ssot_price_cache(self) -> bool:
        """Return True when this provider should consult cacheManager SSOT before venue HTTP."""
        in_live_fn = resolve_patched_attr(
            "is_in_live_price_fetch",
            is_in_live_price_fetch,
            modules=("providers.base", __name__),
            require_callable=True,
        ) or is_in_live_price_fetch
        if in_live_fn():
            return False
        ssot_flag = _parse_bool_flag(getattr(self, "_ssot_cache_enabled", None), None)
        if ssot_flag is False:
            return False
        has_explicit_ssot = bool(
            ssot_flag is True
            or _parse_bool_flag(getattr(self, "_force_ssot_cache", False), False)
            or getattr(self, "_ssot_price_cache_path", None)
            or getattr(self, "_ssot_trend_cache_path", None)
            or getattr(self, "_ssot_24price_cache_dir", None)
            or getattr(self, "_ssot_fills_cache_path", None)
        )
        inst_dict = getattr(self, "__dict__", {})
        inj_client = inst_dict.get("_cli")
        if inj_client is None:
            inj_client = inst_dict.get("_client")
        client_cls_name = self._resolve_provider_attr("_CLIENT_CLASS_NAME", None)
        is_real_client = bool(
            client_cls_name and type(inj_client).__name__ == client_cls_name
        )
        if inj_client is not None and not is_real_client and not has_explicit_ssot:
            return False
        if has_explicit_ssot:
            return True
        shared_fn = resolve_patched_attr(
            "is_shared_disk_cache_enabled",
            is_shared_disk_cache_enabled,
            modules=("providers.base", "botcore", __name__),
            require_callable=True,
        ) or is_shared_disk_cache_enabled
        return shared_fn()

    def _prepare_ssot_price_candidates(self, symbol: str) -> Optional[tuple[str, ...]]:
        if not self._USES_SSOT_MARKET_CACHE or not self._allow_ssot_price_cache():
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        self._ensure_ssot_cache_bindings()
        return candidates

    @staticmethod
    def _resolve_cache_path(filename: str) -> str:
        cp_fn = resolve_patched_attr(
            "cache_path",
            cache_path,
            modules=("providers.base", "cachepaths", __name__),
            require_callable=True,
        ) or cache_path
        return cp_fn(filename)

    @classmethod
    def read_passive_ssot_current_price(
        cls, symbol: str, max_age_sec: float = PRICE_FRESHNESS_MAX_AGE_SEC
    ) -> Optional[float]:
        """Read fresh SSOT current price without instantiating a provider or triggering network I/O."""
        in_live_fn = resolve_patched_attr(
            "is_in_live_price_fetch",
            is_in_live_price_fetch,
            modules=("providers.base", __name__),
            require_callable=True,
        ) or is_in_live_price_fetch
        shared_fn = resolve_patched_attr(
            "is_shared_disk_cache_enabled",
            is_shared_disk_cache_enabled,
            modules=("providers.base", "botcore", __name__),
            require_callable=True,
        ) or is_shared_disk_cache_enabled
        eff_age = resolve_positive_max_age(max_age_sec, PRICE_FRESHNESS_MAX_AGE_SEC)
        if (
            in_live_fn()
            or not shared_fn()
            or eff_age is None
            or isinstance(symbol, bool)
            or not isinstance(symbol, str)
            or not symbol.strip()
        ):
            return None
        raw_u = symbol.strip().upper()
        cleaned = clean_symbol_key(raw_u)
        parts = split_symbol_assets(raw_u)
        canon = canonical_symbol_key(raw_u) if parts is not None else (f"{cleaned}USDC" if cleaned else "")
        candidates = tuple(dict.fromkeys(c for c in (cleaned, raw_u, canon) if c))
        if not candidates:
            return None
        cls._ensure_ssot_cache_bindings()
        cp_cls = cls._ssot_current_price_cls
        if cp_cls is None:
            return None
        return cp_cls.read_ssot_current_price_candidates(
            candidates,
            cls._resolve_cache_path(CACHE_CURRENT_PRICE_FILENAME),
            eff_age,
            use_memory_manager=True,
            reader_fn=cls._ssot_current_price_reader,
        )

    def _cached_current_price(
        self, symbol: str, max_age_sec: float = PRICE_FRESHNESS_MAX_AGE_SEC
    ) -> Optional[float]:
        """Read fresh current price for ``symbol`` from CacheCurrentPriceManager / cache_current_price.json."""
        eff_age = resolve_positive_max_age(max_age_sec, PRICE_FRESHNESS_MAX_AGE_SEC)
        if (
            eff_age is None
            or not (candidates := self._prepare_ssot_price_candidates(symbol))
            or (cp_cls := self._ssot_current_price_cls) is None
        ):
            return None
        custom_path = getattr(self, "_ssot_price_cache_path", None)
        price_path = str(custom_path) if custom_path else self._resolve_cache_path(CACHE_CURRENT_PRICE_FILENAME)
        return cp_cls.read_ssot_current_price_candidates(
            candidates,
            price_path,
            eff_age,
            use_memory_manager=(custom_path is None),
            reader_fn=self._ssot_current_price_reader,
        )

    def _cached_price_history(
        self,
        symbol: str,
        lookback_hours: float,
        max_latest_age_sec: float = 120.0,
    ) -> Optional[list[dict]]:
        """Read 24h tick history from Cache24PriceManager when fresh and covering ``lookback_hours``."""
        if not (candidates := self._prepare_ssot_price_candidates(symbol)) or (p24_cls := self._ssot_price24_cls) is None:
            return None
        return p24_cls.read_ssot_price_history(
            candidates,
            lookback_hours,
            max_latest_age_sec=max_latest_age_sec,
            custom_dir=getattr(self, "_ssot_24price_cache_dir", None),
        )

    def _fills_cache_file(self) -> Optional[str]:
        custom = getattr(self, "_ssot_fills_cache_path", None)
        if custom:
            return str(custom)
        mod_name = type(self).__module__
        mod_path = (
            resolve_patched_attr(self._FILLS_CACHE_PATH_MODULE_ATTR, None, modules=(mod_name,))
            if self._FILLS_CACHE_PATH_MODULE_ATTR
            else None
        )
        def_path = (
            resolve_patched_attr(self._FILLS_CACHE_DEFAULT_MODULE_ATTR, None, modules=(mod_name,))
            if self._FILLS_CACHE_DEFAULT_MODULE_ATTR
            else None
        )
        if mod_path and def_path and mod_path != def_path:
            return str(mod_path)
        if self._allow_ssot_price_cache():
            if mod_path:
                return str(mod_path)
            if self._DEFAULT_FILLS_CACHE_FILENAME:
                return self._resolve_cache_path(self._DEFAULT_FILLS_CACHE_FILENAME)
        return None

    def _fills_from_cache(self, symbol: str) -> Optional[List[dict]]:
        """Read symbol fills from the shared SSOT trade cache, or return None for API fallback."""
        cache_file = self._fills_cache_file()
        if not cache_file:
            return None
        stale_s = resolve_patched_attr("_CACHE_MAX_STALE_S", None, modules=(type(self).__module__,))
        return self.read_ssot_fills_cache(
            symbol,
            cache_file,
            max_stale_sec=stale_s if stale_s is not None else self._FILLS_CACHE_MAX_STALE_S,
            match_fn=getattr(type(self), "_fills_match_fn", None),
            normalize_fn=getattr(type(self), "_fills_normalize_fn", None),
        )

    def _invalidate_fills_cache(self, symbol: str) -> None:
        """Expire ``fetchtime`` for ``symbol`` in this provider's configured SSOT trade cache."""
        cache_file = self._fills_cache_file()
        if cache_file:
            self.invalidate_ssot_fills_cache(
                symbol, cache_file, match_fn=getattr(type(self), "_fills_match_fn", None)
            )

    def _resolve_ssot_trade_cache_cls(self, symbol: str, cache_file: str) -> Any:
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return None
        self._ensure_ssot_cache_bindings()
        return self._ssot_trade_cache_cls

    def read_ssot_fills_cache(
        self,
        symbol: str,
        cache_file: str,
        *,
        max_stale_sec: float = 90.0,
        match_fn: Any = None,
        normalize_fn: Any = None,
    ) -> Optional[List[dict]]:
        """Read normalized fills for ``symbol`` from a shared SSOT trade cache file (`items` + `fetchtime`)."""
        tm_cls = self._resolve_ssot_trade_cache_cls(symbol, cache_file)
        if tm_cls is None:
            return None
        return tm_cls.read_venue_fills_snapshot(
            symbol,
            cache_file,
            max_stale_sec=max_stale_sec,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
            normalize_fn=normalize_fn,
        )

    def write_ssot_fills_cache(self, symbol: str, cache_file: str, fills: List[dict]) -> bool:
        """Persist ``fills`` for ``symbol`` into the shared SSOT trade cache file (`items` + `fetchtime`)."""
        tm_cls = self._resolve_ssot_trade_cache_cls(symbol, cache_file)
        if tm_cls is None:
            return False
        sym_key = clean_symbol_key(symbol) or str(symbol).strip().upper()
        return bool(
            tm_cls.merge_venue_fills_snapshot(
                cache_file, {sym_key: list(fills or ())}, is_snapshot=True
            )
        )

    @classmethod
    def merge_ssot_fills_cache(cls, cache_file: str, fills_by_symbol: dict) -> bool:
        """Atomically merge normalized multi-symbol fills into a shared SSOT trade cache snapshot."""
        if not cache_file or not isinstance(fills_by_symbol, dict):
            return False
        cls._ensure_ssot_cache_bindings()
        tm_cls = cls._ssot_trade_cache_cls
        return bool(tm_cls.merge_venue_fills_snapshot(cache_file, fills_by_symbol)) if tm_cls is not None else False

    def invalidate_ssot_fills_cache(
        self, symbol: str, cache_file: str, *, match_fn: Any = None
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in the shared SSOT trade cache after order execution."""
        tm_cls = self._resolve_ssot_trade_cache_cls(symbol, cache_file)
        if tm_cls is not None:
            tm_cls.invalidate_venue_fills_snapshot(
                symbol,
                cache_file,
                match_fn=match_fn,
                candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
            )
