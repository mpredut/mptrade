"""Provider primitives with no registry or venue-module dependencies."""

import contextlib
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    bool_env,
    exact_nonneg_int,
    extract_current_price_item,
    is_shared_disk_cache_enabled,
    resolve_now_ts,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_SHORT_TREND_FILENAME,
    cache_24_price_filename,
    cache_path,
    cache_symbol_variants,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    get_cachedb_dir,
    normalize_venue_name,
    resolve_cache_dir_path,
    resolve_cachedb_filepath,
    resolve_kraken_lookup_symbol,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from state_io import read_json_dict_recovering, save_json_state


_LIVE_FETCH_LOCAL = threading.local()


@contextlib.contextmanager
def live_price_fetch_context():
    """Mark the current thread as performing a live venue price poll for cacheManager."""
    depth = int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0)
    _LIVE_FETCH_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _LIVE_FETCH_LOCAL.depth = max(0, depth)


def is_in_live_price_fetch() -> bool:
    """Return True when called inside a cacheManager remote price poll."""
    return int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0) > 0


from .order_ohlc_utils import (  # noqa: E402, F401
    _is_fresh_obs_ts,
    _ts_ms_to_s,
    aggregate_fills_by_order_id,
    extract_order_id,
    extract_order_price,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_side,
    normalize_timestamp_ms,
)
from .quantity import (  # noqa: E402, F401
    compute_fills_cost_basis,
    decide_quantity,
    venue_fee_cap_quantity,
)
from .strategy_executor import (  # noqa: E402, F401
    OrderReconciliationCapabilities,
    PairPrecision,
    ProviderError,
)


class MarketDataProvider(ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        ...

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    _DEFAULT_FILLS_CACHE_FILENAME: Optional[str] = None
    _FILLS_CACHE_PATH_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_DEFAULT_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_MAX_STALE_S: float = 60.0
    _fills_match_fn: Any = None
    _fills_normalize_fn: Any = None
    _ssot_current_price_reader: Any = None
    _ssot_short_trend_reader: Any = None

    @classmethod
    def bind_ssot_cache_readers(
        cls,
        *,
        current_price_reader: Any = None,
        short_trend_reader: Any = None,
    ) -> None:
        """Bind read-only SSOT memory cache getters for current price and short trend."""
        if current_price_reader is not None:
            cls._ssot_current_price_reader = current_price_reader
        if short_trend_reader is not None:
            cls._ssot_short_trend_reader = short_trend_reader

    def resolve_cached_fee_and_limits(self, symbol: str, *, price: Optional[float] = None) -> dict:
        """Return non-blocking cached fee tier and precision/limit overrides for ``symbol``."""
        return {}

    @staticmethod
    def resolve_venue_provider_class(venue: Any) -> Any:
        """Resolve the concrete ``MarketDataProvider`` subclass for ``venue``."""
        v_norm = normalize_venue_name(venue)
        if v_norm == "kraken":
            from .kraken_provider import KrakenProvider

            return KrakenProvider
        if v_norm == "hyperliquid":
            from .hyperliquid_provider import HyperliquidProvider

            return HyperliquidProvider
        if v_norm == "t212":
            from .t212_provider import T212Provider

            return T212Provider
        return None

    @staticmethod
    def resolve_provider_cached_fee_and_limits(
        venue: str,
        symbol: str,
        provider: Any,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Resolve cached fee/limit overrides from ``provider`` via public hook or venue adapter."""
        if provider is None:
            return {}
        method = getattr(type(provider), "resolve_cached_fee_and_limits", None)
        if callable(method):
            try:
                res = provider.resolve_cached_fee_and_limits(symbol, price=price)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        prov_cls = MarketDataProvider.resolve_venue_provider_class(
            venue or getattr(provider, "name", "")
        )
        extract_fn = getattr(prov_cls, "extract_cached_fee_and_limits", None) if prov_cls is not None else None
        if callable(extract_fn):
            try:
                res = extract_fn(provider, symbol, price=price)
            except TypeError:
                res = extract_fn(provider, symbol)
            if isinstance(res, dict):
                return res
        return {}

    @classmethod
    def fetch_venue_remote_trades(
        cls,
        venue: Any,
        symbol: str,
        start_time_ms: int = 0,
        *,
        api_client: Any = None,
    ) -> List[dict]:
        """Venue-agnostic remote trade fetcher for non-append ``CacheTradeManager`` instances."""
        if api_client is None:
            return []
        if isinstance(api_client, cls) or "fetch_remote_trades" in getattr(api_client, "__dict__", {}):
            return list(api_client.fetch_remote_trades(symbol, start_time_ms) or [])
        prov_cls = cls.resolve_venue_provider_class(venue)
        if prov_cls is not None and callable(getattr(prov_cls, "fetch_remote_trades_with_client", None)):
            res = prov_cls.fetch_remote_trades_with_client(symbol, start_time_ms, api_client)
            if res is not None:
                return list(res)
        fetch_fn = getattr(api_client, "fetch_remote_trades", None)
        if callable(fetch_fn):
            return list(fetch_fn(symbol, start_time_ms) or [])
        return []

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol incompatible with this configured provider instance."""
        return None

    def _ensure_client_lock(self):
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free balance: ``0.0`` is real zero; ``None`` means unavailable."""
        return None

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
                return cache_path(self._DEFAULT_FILLS_CACHE_FILENAME)
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

    def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0) -> List[dict]:
        """Fetch and normalize remote fills for ``symbol`` without mutating disk caches."""
        self.validate_symbol(symbol)
        api_fn = getattr(self, "_fills_from_api", None)
        if not callable(api_fn):
            return []
        try:
            rows = api_fn(symbol, publish_cache=False)
        except TypeError:
            rows = api_fn(symbol)
        return list(rows) if isinstance(rows, list) else []

    def refresh_ssot_fills_cache(self, symbol: str, fills: Optional[List[dict]] = None) -> List[dict]:
        """Authoritatively fetch (if ``fills`` not provided) and explicitly publish ``symbol`` fills to the SSOT trade cache."""
        self.validate_symbol(symbol)
        if fills is None:
            api_fn = getattr(self, "_fills_from_api", None)
            if not callable(api_fn):
                return []
            try:
                rows = api_fn(symbol, publish_cache=True)
            except TypeError:
                rows = api_fn(symbol)
                cache_file = self._fills_cache_file()
                if cache_file and rows is not None:
                    self.write_ssot_fills_cache(symbol, cache_file, rows)
            return list(rows) if isinstance(rows, list) else []
        cache_file = self._fills_cache_file()
        if cache_file:
            self.write_ssot_fills_cache(symbol, cache_file, fills)
        return list(fills)

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        """Return normalized recent orders (fills aggregated by orderId), reading SSOT cache first unless ``authoritative=True``."""
        api_fn = getattr(self, "_fills_from_api", None)
        if not callable(api_fn):
            return []
        self.validate_symbol(symbol)
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return []
        try:
            cache_fn = getattr(self, "_fills_from_cache", None)
            rows = cache_fn(symbol) if (callable(cache_fn) and not authoritative and not publish_cache) else None
            if rows is None:
                try:
                    rows = api_fn(symbol, publish_cache=publish_cache)
                except TypeError:
                    rows = api_fn(symbol)
                    if publish_cache and rows is not None:
                        cache_file = self._fills_cache_file()
                        if cache_file:
                            self.write_ssot_fills_cache(symbol, cache_file, rows)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            return aggregate_fills_by_order_id(rows, cutoff_ms=(time.time() - since_val) * 1000.0, side=side)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] get_orders({symbol},{side}) failed: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

    def get_trades(
        self,
        symbol: str,
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        if not authoritative and not publish_cache:
            try:
                return self.get_orders(
                    symbol,
                    None,
                    since_s,
                    authoritative=False,
                    publish_cache=False,
                )
            except TypeError:
                return self.get_orders(symbol, None, since_s)
        return self.get_orders(
            symbol,
            None,
            since_s,
            authoritative=authoritative,
            publish_cache=publish_cache,
        )

    @staticmethod
    def resolve_venue_instrument_spec(venue: Any, symbol: str) -> Any:
        """Resolve ``InstrumentSpec`` for ``symbol`` on ``venue`` from ``instrument_registry``."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_u = symbol.strip().upper()
        sym_clean = clean_symbol_key(sym_u)
        sym_canon = canonical_symbol_key(sym_u)
        try:
            import instrument_registry as _default_ir

            ir = resolve_patched_attr("instrument_registry", _default_ir, modules=("providers.market_api",)) or _default_ir
            specs = list(ir.select_instruments(normalize_venue_name(getattr(venue, "name", venue) or "")).values())
            for item in specs:
                item_sym = str(getattr(item, "symbol", "") or "").strip().upper()
                if item_sym == sym_u or (sym_clean and clean_symbol_key(item_sym) == sym_clean):
                    return item
            for item in specs:
                item_sym = str(getattr(item, "symbol", "") or "").strip().upper()
                if sym_canon and canonical_symbol_key(item_sym) == sym_canon:
                    return item
            return None
        except Exception:
            return None

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost from fresh SSOT fills, or None when unknown."""
        since_val = safe_pos_float(since_s, default=None)
        if since_val is None:
            raise ValueError("since_s must be finite and positive")
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        fills_fn = getattr(self, "_fills_from_cache", None)
        rows = fills_fn(symbol) if callable(fills_fn) else None
        if rows is None:
            return None
        sym_u = symbol.strip().upper()
        spec = self.resolve_venue_instrument_spec(getattr(self, "name", ""), sym_u)
        base_asset = spec.base if spec is not None else ""
        quote_asset = spec.quote if spec is not None else ""
        if not base_asset and (parts := split_symbol_assets(sym_u)) is not None:
            base_asset, quote_asset = parts
        return compute_fills_cost_basis(
            rows, held_qty, since_val, base_asset=base_asset, quote_asset=quote_asset
        )

    def open_orders(self, symbol: str) -> List[dict]:
        return []

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        """Return explicitly supported strict lifecycle operations.

        The default is deliberately conservative. An inherited empty
        ``open_orders`` implementation must never be mistaken for a confirmed
        empty venue snapshot.
        """
        return OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    def _is_live_order_enabled(self, env_key: str, folder: Optional[str] = None) -> bool:
        """Resolve live-order gate from explicit ``self._live_enabled`` override or ``bool_env``."""
        live_override = getattr(self, "_live_enabled", None)
        if live_override is None:
            return bool_env(env_key, folder)
        return _parse_bool_flag(live_override, False)

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    @staticmethod
    def resolve_strategy_dry_run(
        force_paper: bool, strategy_execute: bool, venue_live_enabled: bool
    ) -> bool:
        """Combine CLI `--paper`, strategy gate (`STRAT_EXECUTE`), and venue live gate (`*_LIVE_ORDERS`)."""
        return bool(
            _parse_bool_flag(force_paper, True)
            or not _parse_bool_flag(strategy_execute, False)
            or not _parse_bool_flag(venue_live_enabled, False)
        )

    def _round_required_qty(self, symbol: str, qty: float, err_prefix: str = "submit_order") -> float:
        """Round ``qty`` using ``round_quantity`` and raise ``ProviderError`` if non-positive."""
        rounded = self.round_quantity(symbol, qty)
        if rounded <= 0:
            raise ProviderError(f"{err_prefix}({symbol}): invalid non-positive quantity qty={rounded}")
        return rounded

    @staticmethod
    def _execute_confirmed_cancel(order_id: Any, cancel_fn: Any) -> None:
        """Execute ``cancel_fn()`` and raise ``ProviderError`` unless cancellation is confirmed."""
        try:
            if not cancel_fn():
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e

    def prepare_order_state(self) -> Any:
        """Synchronize provider state used by shared pre-submit policy checks."""
        return None

    def validate_order_state(self, expected_state: Any) -> Any:
        """Require that policy checks still describe the prepared provider state."""
        return expected_state

    def place_order(self, symbol: str, side: str, price: float, qty: float, force: bool = False, **kwargs):
        return None

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    def guards_internally(self) -> bool:
        return False

    def _is_binance(self) -> bool:
        return normalize_venue_name(getattr(self, "name", "")) == "binance"

    def requires_cache_permit(self) -> bool:
        """Return whether this provider enforces account-cache submit permits."""
        return self._is_binance()

    def defers_smart_cancellation(self) -> bool:
        """Return whether opposing order cancellation is deferred until dispatch."""
        return self._is_binance()

    def supports_business_minimum(self) -> bool:
        """Return whether this provider enforces venue business minimum rules."""
        return self._is_binance()

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        """Return normalized pair precision and limits, or None if unavailable."""
        return None

    def _pair_precision_attr(self, symbol: str, attr: str, default: Any = None) -> Any:
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                return getattr(pp, attr, default)
        except Exception:
            pass
        return default

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round or floor quantity to venue volume precision."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "volume_decimals"), default=None)
        if dec is not None:
            factor = 10 ** min(dec, 18)
            return math.floor(qty_f * factor + 1e-12) / factor
        return qty_f

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "price_decimals"), default=None)
        return round(price_f, min(dec, 18)) if dec is not None else price_f

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        return safe_pos_float(self._pair_precision_attr(symbol, "order_min", 0.0), default=0.0)

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return "qty_zero_or_negative"
        try:
            min_qty = self.min_order_qty(symbol)
            if min_qty > 0 and qty_f < min_qty:
                return f"filter_lot_size_min:{qty_f}<{min_qty}"
        except Exception:
            pass
        return None

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        import order_guard
        return order_guard.weight_limit(self, symbol, side, price, qty, available_qty=available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        return venue_fee_cap_quantity(self.name, side, price, available_qty)

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None,
        *, prefer_safeback: bool = False, default_safeback_sec: float = 0.0,
    ):
        import order_guard
        sb_f = safe_pos_float(safeback_sec, default=None)
        if prefer_safeback and sb_f is not None:
            window_s = sb_f
        else:
            win_cfg = order_guard.get_window_cfg(
                self.name,
                symbol=symbol,
                order_type=side,
                regime_context=regime_context,
                module=module,
                reference_mode=reference_mode,
            )
            win_f = safe_pos_float(win_cfg, default=None)
            window_s = win_f if win_f is not None else (sb_f if sb_f is not None else default_safeback_sec)
        wr_kwargs = {k: v for k, v in (("reference_mode", reference_mode), ("module", module)) if v is not None}
        return order_guard.window_reference(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        ot_norm = normalize_side(order_type)
        if ot_norm not in ("BUY", "SELL"):
            return None
        opposite = "SELL" if ot_norm == "BUY" else "BUY"
        eff_since_s = safe_nonneg_float(since_s, default=90.0 * 24.0 * 3600.0)
        fills = self.get_orders(symbol, opposite, eff_since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        valid_fills = [o for o in fills if isinstance(o, dict) and extract_order_price(o, default=0.0) > 0.0]
        if not valid_fills:
            return None
        latest = max(valid_fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        return extract_order_price(latest) or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            series = self.ohlc_series(symbol, interval_min)
            closes = getattr(series, "closes", None) if series is not None else None
            return list(closes) if closes is not None else []
        return []

    # ── Agnostic SSOT cacheManager coupling (price, 24h history, short trend) ─

    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        """Return venue-scoped symbol keys to look up in SSOT cacheManager stores."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return ()
        raw_u = symbol.strip().upper()
        cleaned = clean_symbol_key(raw_u)
        parts = split_symbol_assets(raw_u)
        canon = canonical_symbol_key(raw_u) if parts is not None else ""
        kr_lookup = resolve_kraken_lookup_symbol(raw_u) if (parts is not None and parts[1] == "USD") else ""
        return tuple(dict.fromkeys(c for c in (cleaned, raw_u, canon, kr_lookup) if c))

    _REAL_CLIENT_CLASS_NAMES = frozenset({"KrakenClient", "HLClient", "T212Client"})
    _DEFINITIVE_REFUSAL_TOKENS = (
        "insufficient",
        "minimum value",
        "order minimum",
        "volume minimum",
        "cost minimum",
        "invalid price",
        "invalid size",
        "invalid order",
        "invalid arguments",
        "positions limit exceeded",
        "tick size",
        "could not immediately match",
        "price must be",
        "too far from",
    )

    @classmethod
    def _is_definitive_refusal(cls, error: object) -> bool:
        """Return True when a venue error message indicates a permanent order rejection."""
        text = str(error or "").lower()
        return any(token in text for token in cls._DEFINITIVE_REFUSAL_TOKENS)

    @staticmethod
    def clamp_price_to_bbo_mid(
        price: Optional[float],
        bid: Any,
        ask: Any,
        *,
        max_spread_ratio: float = 0.05,
    ) -> Optional[float]:
        """Return ``price`` clamped to the BBO mid ``(bid + ask) / 2`` when ``price`` falls outside a tight ``[bid, ask]`` spread."""
        px_f = safe_pos_float(price, default=None) if not isinstance(price, bool) else None
        bid_f = safe_pos_float(bid, default=None) if not isinstance(bid, bool) else None
        ask_f = safe_pos_float(ask, default=None) if not isinstance(ask, bool) else None
        if (
            bid_f is not None
            and ask_f is not None
            and bid_f <= ask_f
            and (ask_f - bid_f) / bid_f <= max_spread_ratio
            and (px_f is None or px_f < bid_f or px_f > ask_f)
        ):
            return round((bid_f + ask_f) * 0.5, 8)
        return px_f

    def _allow_ssot_price_cache(self) -> bool:
        """Return True when this provider should consult cacheManager SSOT before venue HTTP."""
        if is_in_live_price_fetch():
            return False
        if getattr(self, "_ssot_cache_enabled", None) is False:
            return False
        has_explicit_ssot = bool(
            getattr(self, "_ssot_cache_enabled", None) is True
            or getattr(self, "_force_ssot_cache", False)
            or getattr(self, "_ssot_price_cache_path", None)
            or getattr(self, "_ssot_trend_cache_path", None)
            or getattr(self, "_ssot_24price_cache_dir", None)
            or getattr(self, "_ssot_fills_cache_path", None)
        )
        inj_client = getattr(self, "_cli", None) or getattr(self, "_client", None)
        if (
            inj_client is not None
            and type(inj_client).__name__ not in self._REAL_CLIENT_CLASS_NAMES
            and not has_explicit_ssot
        ):
            return False
        if has_explicit_ssot:
            return True
        return is_shared_disk_cache_enabled()

    @staticmethod
    def _parse_ssot_price_point(entry: Any, fallback_ts: Optional[float] = None) -> Optional[tuple[int, float]]:
        if isinstance(entry, dict):
            ts_ms = extract_timestamp_ms(entry, default=None)
            if ts_ms is None and fallback_ts is not None:
                ts_ms = normalize_timestamp_ms(fallback_ts, default=None)
            px_f = extract_current_price_item(entry)
            if px_f is None:
                raw_px = entry.get("price") if entry.get("price") is not None else entry.get("px")
                px_f = safe_pos_float(raw_px, default=None)
            if ts_ms is not None and ts_ms > 0 and px_f is not None:
                return ts_ms, px_f
            return None
        if isinstance(entry, (list, tuple)):
            if len(entry) >= 2 and not isinstance(entry[0], (list, tuple, dict)):
                ts_ms = normalize_timestamp_ms(entry[0], default=None)
                px_f = safe_pos_float(entry[1], default=None)
                if ts_ms is not None and ts_ms > 0 and px_f is not None:
                    return ts_ms, px_f
                return None
            for sub in reversed(entry):
                if not isinstance(sub, (list, tuple, dict)):
                    continue
                parsed = MarketDataProvider._parse_ssot_price_point(sub, fallback_ts=fallback_ts)
                if parsed is not None:
                    return parsed
            return None
        px_f = extract_current_price_item(entry)
        if px_f is not None and fallback_ts is not None:
            ts_ms = normalize_timestamp_ms(fallback_ts, default=None)
            if ts_ms is not None and ts_ms > 0:
                return ts_ms, px_f
        return None

    @classmethod
    def _resolve_ssot_memory_manager(cls, reader_fn: Any, factory_key: str) -> Any:
        if callable(reader_fn):
            with contextlib.suppress(Exception):
                mgr = reader_fn()
                if mgr is not None:
                    return mgr
        cf = resolve_patched_attr("CacheFactory", None, modules=("cacheManager",))
        instances = getattr(cf, "_instances", None)
        return instances.get(factory_key) if isinstance(instances, dict) else None

    @staticmethod
    def _read_ssot_current_price_candidates(
        candidates: tuple[str, ...] | List[str],
        price_path: str,
        max_age_sec: float,
        *,
        now_s: Optional[float] = None,
        use_memory_manager: bool = True,
    ) -> Optional[float]:
        max_age = safe_pos_float(max_age_sec, default=12.0) or 12.0
        if use_memory_manager:
            cpm = MarketDataProvider._resolve_ssot_memory_manager(
                MarketDataProvider._ssot_current_price_reader, "CurrentPrice"
            )
            cpm_file = getattr(cpm, "filename", None) if cpm is not None else None
            cpm_matches_path = (
                not isinstance(cpm_file, str)
                or not cpm_file
                or os.path.abspath(str(resolve_cachedb_filepath(cpm_file))) == os.path.abspath(str(resolve_cachedb_filepath(price_path)))
            )
            if cpm is not None and cpm_matches_path and hasattr(cpm, "get_cached_price_observation"):
                for cand in candidates:
                    try:
                        obs = cpm.get_cached_price_observation(cand, max_age)
                        if obs is not None:
                            px = safe_pos_float(obs[1], default=None)
                            if px is not None:
                                return px
                    except Exception:
                        continue
        if not price_path or not os.path.exists(price_path):
            return None
        try:
            data = read_json_dict_recovering(price_path)
            if not isinstance(data, dict):
                return None
            items = data.get("items") if isinstance(data.get("items"), dict) else data
            fetchtime_map = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
            file_ts = extract_timestamp_s(data, default=None)
            eff_now = resolve_now_ts(now_s)
            for cand in candidates:
                raw_entry = casefold_get(items, cand)
                if raw_entry is None:
                    continue
                fb_ts = casefold_get(fetchtime_map, cand)
                if fb_ts is None:
                    fb_ts = file_ts
                parsed = MarketDataProvider._parse_ssot_price_point(raw_entry, fallback_ts=fb_ts)
                if parsed is None:
                    continue
                ts_ms, px = parsed
                if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), eff_now, max_age):
                    return px
        except Exception:
            pass
        return None

    def get_ssot_cached_price(self, symbol: str, max_age_sec: float = 12.0) -> Optional[float]:
        """Read fresh current price for ``symbol`` from CacheCurrentPriceManager / cache_current_price.json."""
        if not self._allow_ssot_price_cache():
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        custom_path = getattr(self, "_ssot_price_cache_path", None)
        price_path = str(custom_path) if custom_path else cache_path(CACHE_CURRENT_PRICE_FILENAME)
        return self._read_ssot_current_price_candidates(
            candidates,
            price_path,
            safe_pos_float(max_age_sec, default=12.0) or 12.0,
            use_memory_manager=(custom_path is None),
        )

    @staticmethod
    def _read_ssot_24price_entries(c24_path: str, cand: str) -> tuple[Optional[dict], Optional[list]]:
        if not c24_path or not os.path.exists(c24_path):
            return None, None
        raw = read_json_dict_recovering(c24_path)
        if not isinstance(raw, dict):
            return None, None
        items = raw.get("items")
        entries = items if isinstance(items, list) else casefold_get(items if isinstance(items, dict) else raw, cand)
        return (raw, entries) if isinstance(entries, list) and entries else (raw, None)

    def get_ssot_price_history(
        self,
        symbol: str,
        lookback_hours: float,
        max_latest_age_sec: float = 120.0,
    ) -> Optional[list[dict]]:
        """Read 24h tick history from Cache24PriceManager when fresh and covering ``lookback_hours``."""
        if not self._allow_ssot_price_cache():
            return None
        lh_f = safe_pos_float(lookback_hours, default=None)
        if lh_f is None or lh_f > 24.0:
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        max_latest_age = safe_pos_float(max_latest_age_sec, default=120.0) or 120.0
        now_s = time.time()
        cutoff_ms = int((now_s - lh_f * 3600.0) * 1000)
        custom_dir = getattr(self, "_ssot_24price_cache_dir", None)
        for cand in candidates:
            c24_fname = cache_24_price_filename(cand)
            c24_path = (
                str(resolve_cachedb_filepath(os.path.join(str(custom_dir), c24_fname)))
                if custom_dir
                else cache_path(c24_fname)
            )
            try:
                _, entries = self._read_ssot_24price_entries(c24_path, cand)
                if not entries or len(entries) < 2:
                    continue
                valid = [p for e in entries if (p := MarketDataProvider._parse_ssot_price_point(e)) is not None]
                if len(valid) < 2:
                    continue
                valid.sort(key=lambda pt: pt[0])
                if not _is_fresh_obs_ts(valid[-1][0] / 1000.0, now_s, max_latest_age):
                    continue
                out = [{"timestamp": ts, "price": px} for ts, px in valid if ts >= cutoff_ms]
                if len(out) < 2:
                    continue
                if custom_dir is None and lh_f > 0.25 and (out[-1]["timestamp"] - out[0]["timestamp"]) < (lh_f * 3600.0 * 1000.0 * 0.75):
                    continue
                return out
            except Exception:
                continue
        return None

    @staticmethod
    def read_ssot_short_trend_file(
        trend_path: str,
        candidates: tuple[str, ...] | List[str],
        max_age_sec: float = 15.0,
        *,
        now_s: Optional[float] = None,
    ) -> Optional[dict]:
        """Read a fresh short-trend snapshot for ``candidates`` from ``trend_path``."""
        if not trend_path:
            return None
        max_age = safe_pos_float(max_age_sec, default=15.0) or 15.0
        try:
            data = read_json_dict_recovering(trend_path)
            if not isinstance(data, dict):
                return None
            items = data.get("items") if isinstance(data.get("items"), dict) else data
            eff_now = resolve_now_ts(now_s)
            for cand in candidates:
                if not cand:
                    continue
                snap = casefold_get(items, cand)
                if not isinstance(snap, dict):
                    continue
                raw_ts = extract_timestamp_s(snap, default=None)
                if raw_ts is not None and raw_ts > 0 and _is_fresh_obs_ts(raw_ts, eff_now, max_age):
                    return dict(snap)
        except Exception:
            pass
        return None

    def trend_snapshot(self, symbol: str, max_age_sec: float = 15.0) -> Optional[dict]:
        """Return a fresh short-trend snapshot for ``symbol`` from CacheShortTrendManager / cache_short_trend.json."""
        if not self._allow_ssot_price_cache():
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        max_age = safe_pos_float(max_age_sec, default=15.0) or 15.0
        custom_path = getattr(self, "_ssot_trend_cache_path", None)
        now_s = time.time()
        if custom_path is None:
            stm = self._resolve_ssot_memory_manager(
                self._ssot_short_trend_reader, "ShortTrend"
            )
            if stm is not None:
                if callable(getattr(type(stm), "get_cached_memory_snapshot", None)):
                    for cand in candidates:
                        with contextlib.suppress(Exception):
                            snap = stm.get_cached_memory_snapshot(cand, max_age_sec=max_age, now_s=now_s)
                            if isinstance(snap, dict):
                                return snap
                else:
                    stm_mem = getattr(stm, "_mem", None)
                    if isinstance(stm_mem, dict):
                        lock_ctx = getattr(stm, "_lock", None) or getattr(stm, "lock", None) or contextlib.nullcontext()
                        with contextlib.suppress(Exception), lock_ctx:
                            for cand in candidates:
                                snap = casefold_get(stm_mem, cand)
                                if isinstance(snap, dict) and _is_fresh_obs_ts(extract_timestamp_s(snap, default=None), now_s, max_age):
                                    return dict(snap)
                    elif hasattr(stm, "get_fresh_snapshot"):
                        for cand in candidates:
                            with contextlib.suppress(Exception):
                                snap = stm.get_fresh_snapshot(cand)
                                if isinstance(snap, dict):
                                    raw_ts = extract_timestamp_s(snap, default=None)
                                    if raw_ts is None or _is_fresh_obs_ts(raw_ts, now_s, max_age):
                                        return dict(snap)
        trend_path = str(custom_path) if custom_path else cache_path(CACHE_SHORT_TREND_FILENAME)
        return self.read_ssot_short_trend_file(trend_path, candidates, max_age_sec=max_age, now_s=now_s)

    def trend(self, symbol: str, max_age_sec: float = 15.0) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0.0 neutral/unknown) from cacheManager SSOT."""
        snap = self.trend_snapshot(symbol, max_age_sec=max_age_sec)
        if not isinstance(snap, dict):
            return 0.0
        try:
            from pricewindow import extract_short_trend_slope

            slope = safe_float(extract_short_trend_slope(snap, symbol), default=0.0)
            if slope != 0.0:
                return slope
        except Exception:
            pass
        for key in ("trend", "final_trend", "growth_coefficient", "gradient_recent", "slope_small"):
            val = safe_float(snap.get(key), default=None)
            if val is not None and val != 0.0:
                return val
        return 0.0

    def trend_direction(self, symbol: str, max_age_sec: float = 15.0) -> Optional[str]:
        """Return short-trend direction ('up', 'down', or 'flat') from cacheManager SSOT, or None if unavailable."""
        snap = self.trend_snapshot(symbol, max_age_sec=max_age_sec)
        if not isinstance(snap, dict):
            return None
        try:
            from cachetrend import CacheShortTrendManager

            if CacheShortTrendManager._eval_upward_formula(snap):
                return "up"
            inv_snap = dict(snap)
            for key in ("growth_coefficient", "gradient_recent", "slope_small", "final_trend"):
                val = safe_float(inv_snap.get(key), default=None)
                if val is not None:
                    inv_snap[key] = -val
            if CacheShortTrendManager._eval_upward_formula(inv_snap):
                return "down"
            return "flat"
        except Exception:
            return None

    @staticmethod
    def normalize_orderbook_payload(raw_data: Any) -> Optional[dict]:
        """Normalize Binance, Kraken, or Hyperliquid L2 orderbook payloads into ``{'bids': [(px, qty), ...], 'asks': [(px, qty), ...]}``."""
        if not isinstance(raw_data, dict):
            return None
        data = raw_data
        if "bids" not in data and isinstance(data.get("levels"), list) and len(data["levels"]) >= 2:
            hl_bids, hl_asks = data["levels"][0], data["levels"][1]
            if isinstance(hl_bids, list) and isinstance(hl_asks, list):
                data = {
                    "bids": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_bids],
                    "asks": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_asks],
                }
        elif "bids" not in data and isinstance(data.get("result"), dict):
            krk_book = next(
                (v for v in data["result"].values() if isinstance(v, dict) and "bids" in v),
                None,
            )
            if isinstance(krk_book, dict):
                data = krk_book
            else:
                return None

        def _parse_side(rows: Any) -> list[tuple[float, float]]:
            if not isinstance(rows, list):
                return []
            out_side: list[tuple[float, float]] = []
            for item in rows:
                if isinstance(item, dict):
                    raw_p = item.get("px") if item.get("px") is not None else item.get("price")
                    raw_q = next((item.get(k) for k in ("sz", "qty", "volume") if item.get(k) is not None), None)
                    pf = safe_pos_float(raw_p, default=None)
                    qf = safe_pos_float(raw_q, default=None)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pf = safe_pos_float(item[0], default=None)
                    qf = safe_pos_float(item[1], default=None)
                else:
                    continue
                if pf is not None and qf is not None:
                    out_side.append((pf, qf))
            return out_side

        bids = _parse_side(data.get("bids"))
        asks = _parse_side(data.get("asks"))
        if not bids or not asks:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return {"bids": bids, "asks": asks}

    def _matching_ssot_cache_keys(self, mapping: dict, symbol: str, match_fn: Any = None) -> list[str]:
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.matching_snapshot_keys(
            mapping,
            symbol,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
        )

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
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return None
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.read_venue_fills_snapshot(
            symbol,
            cache_file,
            max_stale_sec=max_stale_sec,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
            normalize_fn=normalize_fn,
        )

    def write_ssot_fills_cache(self, symbol: str, cache_file: str, fills: List[dict]) -> bool:
        """Persist ``fills`` for ``symbol`` into the shared SSOT trade cache file (`items` + `fetchtime`)."""
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.write_venue_fills_snapshot(symbol, cache_file, fills)

    @staticmethod
    def merge_ssot_fills_cache(cache_file: str, fills_by_symbol: dict) -> bool:
        """Merge ``{symbol: [fill_dict, ...]}`` into a shared SSOT trade cache file (`items` + `fetchtime`) deduplicating by `orderId`."""
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.merge_venue_fills_snapshot(cache_file, fills_by_symbol)

    def invalidate_ssot_fills_cache(
        self, symbol: str, cache_file: str, *, match_fn: Any = None
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in the shared SSOT trade cache after order execution."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return
        from cacheaccount import CacheTradeManager

        CacheTradeManager.invalidate_venue_fills_snapshot(
            symbol,
            cache_file,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
        )


def resolve_ssot_price_from_cache(
    symbol: str,
    cache_dir: Optional[str] = None,
    max_age_sec: float = 60.0,
    *,
    quote_isolated: bool = True,
    now_ts: Optional[float] = None,
) -> Optional[float]:
    """Resolve a fresh positive spot price for ``symbol`` from SSOT cacheManager stores (0 ms local)."""
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    sym_u = symbol.strip().upper()
    sym_clean = clean_symbol_key(sym_u)
    if not sym_clean:
        return None
    max_age = safe_pos_float(max_age_sec, default=60.0) or 60.0
    now_s = resolve_now_ts(now_ts)
    quotes = (
        compatible_quote_suffixes(sym_clean, default_usd=("USDC", "USDT", "USD", "FDUSD"))
        if quote_isolated
        else ("USDC", "USDT", "USD", "EUR")
    )
    cands = list(
        dict.fromkeys(
            c
            for c in (sym_clean, sym_u, *cache_symbol_variants(sym_clean, quotes=quotes))
            if isinstance(c, str) and c and "/" not in c
        )
    )
    is_default_dir = not cache_dir or str(cache_dir).strip() in ("", "cachedb")
    eff_dir = resolve_cache_dir_path(cache_dir) or get_cachedb_dir()
    price_path = str(resolve_cachedb_filepath(os.path.join(eff_dir, CACHE_CURRENT_PRICE_FILENAME)))
    px = MarketDataProvider._read_ssot_current_price_candidates(
        cands,
        price_path,
        max_age,
        now_s=now_s,
        use_memory_manager=is_default_dir,
    )
    if px is not None:
        return px

    for cand in cands:
        c24_path = str(resolve_cachedb_filepath(os.path.join(eff_dir, cache_24_price_filename(cand))))
        try:
            raw, entries = MarketDataProvider._read_ssot_24price_entries(c24_path, cand)
            if not entries or raw is None:
                continue
            parsed = MarketDataProvider._parse_ssot_price_point(entries[-1], fallback_ts=extract_timestamp_s(raw, default=None))
            if parsed is not None:
                ts_ms, px = parsed
                if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), now_s, max_age):
                    return px
        except Exception:
            continue
    return None


def mount_connect_retry(session, *, attempts: int = 3, backoff: float = 0.4,
                        read: int = 0, status_forcelist=None,
                        raise_on_status: bool = False) -> bool:
    """Mount an order-safe Retry adapter on ``session`` for https:// and http:// (best-effort)."""
    try:
        if session is None or isinstance(status_forcelist, (bool, str, bytes)):
            return False
        att_i = safe_nonneg_int(attempts, default=None)
        read_i = safe_nonneg_int(read, default=None)
        bo_f = safe_nonneg_float(backoff, default=None)
        if att_i is None or read_i is None or bo_f is None:
            return False
        forcelist = None
        if status_forcelist:
            forcelist = []
            for code in status_forcelist:
                c_i = safe_pos_int(code, default=None)
                if c_i is None:
                    return False
                forcelist.append(c_i)
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=att_i,
            connect=att_i,
            read=read_i,
            status=att_i if forcelist else 0,
            redirect=0,
            backoff_factor=bo_f,
            status_forcelist=forcelist if forcelist else None,
            raise_on_status=_parse_bool_flag(raise_on_status, default=False),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return True
    except Exception:  # noqa: BLE001 -- retries are best-effort; the timeout still guards
        return False

