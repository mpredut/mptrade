"""Declarative venue registry metadata, fee/precision ladders, and public orderbook helpers."""

import inspect
from pathlib import Path
from typing import Any, Optional

from botcore import (
    clean_upper_token,
    load_env_file_defaults,
    resolve_patched_attr,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (
    canonical_symbol_key,
    clean_symbol_key,
    normalize_venue_name,
    symbol_quote_asset,
)
import instrument_registry as _default_ir

from .order_ohlc_utils import (
    VENUE_PROVIDER_SPECS,
    derive_venue_128bit_client_order_id,
    resolve_venue_provider_class as _resolve_venue_provider_class,
)


class ProviderVenueMetadataMixin:
    """Level-2 mixin isolating declarative venue capabilities, fee ladders, and public L2 book helpers."""

    _CLIENT_CLASS_NAME: Optional[str] = None
    _DEFAULT_QUOTE_ASSET: Optional[str] = None
    _DEFAULT_INTENT_EXECUTOR_TAG: str = "[auto_cryptotrade_executor]"
    _ASSET_SCOPE: str = "crypto"
    _RUNTIME_ENV_RELPATH: Optional[tuple[str, ...]] = None
    _STRATEGY_HEARTBEAT_NAME: str = ""
    _CLIENT_ID_128BIT_SPEC: Optional[tuple[str, bytes, Any, str]] = None
    _INTENT_CLIENT_ID_FORMAT: Optional[str] = None
    _BUY_SLIPPAGE_FEE_BUFFER: Optional[float] = None
    _SELL_DUST_SPEC: Optional[tuple[int, float]] = None
    _DEFAULT_FEE_SPEC: tuple[str, float, float, float] = ("STANDARD", 15.0, 15.0, 5.0)
    _DEFAULT_STEP_LADDER: tuple[tuple[float, float], ...] = ((1000.0, 0.0001), (50.0, 0.001))
    _DEFAULT_STEP_FALLBACK: float = 0.01
    _DEFAULT_TICK_LADDER: tuple[tuple[float, float], ...] = ((10.0, 0.01), (1.0, 0.001))
    _DEFAULT_TICK_FALLBACK: float = 0.0001
    _TCA_DECIMALS_LADDER: tuple[tuple[float, int], ...] = ((10.0, 2), (1.0, 3), (0.1, 4))
    _TCA_DECIMALS_FALLBACK: int = 6
    _PREFER_STATIC_TICK_SIZE: bool = False
    _USES_SHARED_REGISTRY_SYMBOLS: bool = False
    _HISTORICAL_OHLC_PRIORITY: Optional[int] = None
    _ORDERBOOK_L2_INFO_BOOK: bool = False
    _EXACT_TELEMETRY_CACHE_SYMBOL: bool = False
    _QUOTE_ISOLATED_TELEMETRY_VENUE: bool = True
    _HAS_NATIVE_L1_DERIVATIVES: bool = False
    _MERGES_AUDIT_TERMINAL_ORDERS: bool = False
    _USES_BARE_BASE_ORDER_SYMBOLS: bool = False
    _USES_SHARED_TRAILING_MAP: bool = False
    _POSITION_PRICE_DECIMALS: int = 6
    _VENUE_PROVIDER_SPECS: dict[str, tuple[str, str]] = VENUE_PROVIDER_SPECS

    @classmethod
    def derive_client_order_id(cls, client_order_id: Any) -> str:
        """Derive the venue-native 128-bit clientOrderId for ``cls``."""
        return derive_venue_128bit_client_order_id(cls, client_order_id)

    @staticmethod
    def _invoke_fee_and_limits_callable(
        fn: Any, *leading_args: Any, symbol: str, price: Optional[float] = None
    ) -> Optional[dict]:
        if not callable(fn):
            return None
        try:
            params = inspect.signature(fn).parameters
            has_price = "price" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            has_price = False
        res = fn(*leading_args, symbol, price=price) if has_price else fn(*leading_args, symbol)
        return res if isinstance(res, dict) else None

    def resolve_cached_fee_and_limits(self, symbol: str, *, price: Optional[float] = None) -> dict:
        """Return non-blocking cached fee tier and precision/limit overrides for ``symbol``."""
        extract_fn = getattr(type(self), "extract_cached_fee_and_limits", None)
        if not callable(extract_fn):
            prov_cls = self.resolve_venue_provider_class(getattr(self, "name", ""))
            extract_fn = getattr(prov_cls, "extract_cached_fee_and_limits", None) if prov_cls is not None else None
        res = self._invoke_fee_and_limits_callable(extract_fn, self, symbol=symbol, price=price)
        return res if res is not None else {}

    @classmethod
    def resolve_venue_provider_class(cls, venue: Any, *, provider: Any = None) -> Any:
        """Resolve the concrete ``MarketDataProvider`` subclass for ``venue`` or ``provider``."""
        return _resolve_venue_provider_class(
            venue,
            provider=provider,
            base_cls=ProviderVenueMetadataMixin,
            specs=cls._VENUE_PROVIDER_SPECS,
        )

    @staticmethod
    def _lookup_price_ladder(px: float, ladder: tuple[tuple[float, Any], ...], fallback: Any) -> Any:
        for threshold, val in ladder:
            if px >= threshold:
                return val
        return fallback

    @classmethod
    def default_step_and_tick(cls, price: float = 0.0) -> tuple[float, float]:
        """Return default ``(step_size, tick_size)`` fallback for TCA/fee limit evaluation."""
        px_f = safe_pos_float(price, default=0.0) or 0.0
        step = cls._lookup_price_ladder(px_f, cls._DEFAULT_STEP_LADDER, cls._DEFAULT_STEP_FALLBACK)
        tick = cls._lookup_price_ladder(px_f, cls._DEFAULT_TICK_LADDER, cls._DEFAULT_TICK_FALLBACK)
        fb_step = safe_pos_float(cls._DEFAULT_STEP_FALLBACK, default=0.01) or 0.01
        fb_tick = safe_pos_float(cls._DEFAULT_TICK_FALLBACK, default=0.01) or 0.01
        return safe_pos_float(step, default=fb_step) or fb_step, safe_pos_float(tick, default=fb_tick) or fb_tick

    @classmethod
    def tca_price_decimals(cls, limit_price: float = 0.0) -> int:
        """Return rounding precision decimals for limit-order queue TCA analysis."""
        px_f = safe_pos_float(limit_price, default=0.0) or 0.0
        fb_dec = safe_nonneg_int(cls._TCA_DECIMALS_FALLBACK, default=2)
        return safe_nonneg_int(
            cls._lookup_price_ladder(px_f, cls._TCA_DECIMALS_LADDER, cls._TCA_DECIMALS_FALLBACK),
            default=fb_dec,
        )

    @classmethod
    def resolve_venue_default_fee_spec(
        cls,
        venue: Any,
        *,
        fallback_venue: Optional[str] = None,
    ) -> tuple[str, float, float, float]:
        """Return declarative ``(fee_tier, maker_bps, taker_bps, min_notional_usd)`` for ``venue``."""
        prov_cls = cls.resolve_venue_provider_class(venue)
        if prov_cls is None and fallback_venue:
            prov_cls = cls.resolve_venue_provider_class(fallback_venue)
        return getattr(prov_cls, "_DEFAULT_FEE_SPEC", cls._DEFAULT_FEE_SPEC)

    @classmethod
    def _resolve_venue_target_cls(cls, venue: Any) -> type["ProviderVenueMetadataMixin"]:
        return cls.resolve_venue_provider_class(venue) or cls

    @classmethod
    def resolve_venue_default_fee_and_precision(
        cls,
        venue: Any,
        price: float = 0.0,
    ) -> tuple[str, float, float, float, float, float, bool]:
        """Return ``(fee_tier, maker_bps, taker_bps, min_notional, default_step, default_tick, prefer_static_tick)``."""
        target_cls = cls._resolve_venue_target_cls(venue)
        fee_tier, maker_bps, taker_bps, min_notional = getattr(
            target_cls, "_DEFAULT_FEE_SPEC", cls._DEFAULT_FEE_SPEC
        )
        default_step, default_tick = target_cls.default_step_and_tick(price)
        prefer_static_tick = bool(getattr(target_cls, "_PREFER_STATIC_TICK_SIZE", False))
        return (
            fee_tier,
            maker_bps,
            taker_bps,
            min_notional,
            default_step,
            default_tick,
            prefer_static_tick,
        )

    @classmethod
    def resolve_venue_tca_price_decimals(cls, venue: Any, limit_price: float = 0.0) -> int:
        """Return venue-specific TCA limit-price rounding decimals."""
        return safe_nonneg_int(cls._resolve_venue_target_cls(venue).tca_price_decimals(limit_price), default=2)

    @classmethod
    def uses_shared_registry_symbols(cls, venue: Any) -> bool:
        """Return whether ``venue`` defaults to ``instrument_registry.symbols`` when ``symbols=None``."""
        v_str = str(getattr(venue, "name", venue) or "").strip()
        if not v_str:
            return True
        prov_cls = cls.resolve_venue_provider_class(v_str)
        return bool(getattr(prov_cls, "_USES_SHARED_REGISTRY_SYMBOLS", False))

    @classmethod
    def is_exact_telemetry_symbol_venue(
        cls,
        venue: Any,
        *,
        quote_isolated: bool = False,
        multi_venue_base: bool = False,
    ) -> bool:
        """Return whether external telemetry disk caching requires exact symbol isolation for ``venue``."""
        target_cls = cls._resolve_venue_target_cls(venue)
        if getattr(target_cls, "_EXACT_TELEMETRY_CACHE_SYMBOL", False):
            return True
        return bool(
            quote_isolated
            and (
                getattr(target_cls, "_QUOTE_ISOLATED_TELEMETRY_VENUE", True)
                or multi_venue_base
            )
        )

    @staticmethod
    def _bind_venue_bool_capability(attr: str):
        return classmethod(
            lambda cls, venue, _attr=attr: bool(getattr(cls.resolve_venue_provider_class(venue), _attr, False))
        )

    has_native_l1_derivatives = _bind_venue_bool_capability("_HAS_NATIVE_L1_DERIVATIVES")

    @classmethod
    def canonical_venues(cls) -> tuple[str, ...]:
        """Return all registered canonical venue identifiers in deterministic order."""
        return tuple(cls._VENUE_PROVIDER_SPECS.keys())

    merges_audit_terminal_orders = _bind_venue_bool_capability("_MERGES_AUDIT_TERMINAL_ORDERS")
    uses_bare_base_order_symbols = _bind_venue_bool_capability("_USES_BARE_BASE_ORDER_SYMBOLS")
    uses_shared_trailing_map = _bind_venue_bool_capability("_USES_SHARED_TRAILING_MAP")

    @classmethod
    def resolve_venue_default_quote(cls, venue: Any, fallback: str = "USD") -> str:
        """Return the default quote currency for ``venue`` when unparsed from ``symbol``."""
        prov_cls = cls.resolve_venue_provider_class(venue)
        q = getattr(prov_cls, "_DEFAULT_QUOTE_ASSET", None) if prov_cls is not None else None
        return str(q).strip().upper() if isinstance(q, str) and q.strip() else str(fallback or "USD").strip().upper()

    @classmethod
    def resolve_venue_position_price_decimals(cls, venue: Any) -> int:
        """Return mark/position price rounding decimals for ``venue``."""
        dec = safe_nonneg_int(getattr(cls._resolve_venue_target_cls(venue), "_POSITION_PRICE_DECIMALS", 6), default=6)
        return 6 if dec is None else dec

    @classmethod
    def format_strategy_heartbeat_name(cls, symbol: str = "") -> str:
        """Return the daemon heartbeat module name for this provider's strategy engine."""
        if cls._STRATEGY_HEARTBEAT_NAME:
            return cls._STRATEGY_HEARTBEAT_NAME
        raw_name = getattr(cls, "name", "")
        label = raw_name if isinstance(raw_name, str) and raw_name else cls.__name__.removesuffix("Provider")
        return f"{label or 'Strategy'}-bot"

    @classmethod
    def resolve_venue_strategy_heartbeat(cls, venue: Any, symbol: str = "") -> str:
        """Resolve the strategy daemon heartbeat module name for ``venue`` and ``symbol``."""
        prov_cls = cls.resolve_venue_provider_class(venue)
        if prov_cls is not None:
            return prov_cls.format_strategy_heartbeat_name(symbol)
        venue_str = str(venue or "").strip()
        return f"{venue_str}-bot" if venue_str else "Strategy-bot"

    @classmethod
    def resolve_venue_intent_executor_tag(cls, venue: Any) -> str:
        """Resolve the canonical autonomous intent executor bracket tag for ``venue``."""
        prov_cls = cls.resolve_venue_provider_class(venue)
        tag = getattr(prov_cls, "_DEFAULT_INTENT_EXECUTOR_TAG", None) if prov_cls is not None else None
        return str(tag) if isinstance(tag, str) and tag.strip() else cls._DEFAULT_INTENT_EXECUTOR_TAG

    @classmethod
    def load_venue_env_defaults(cls, workspace_dir: Any, *, asset_scope: str = "all") -> None:
        """Load per-venue runtime/config env defaults matching ``asset_scope`` ('all', 'crypto', 'equities')."""
        if not workspace_dir:
            return
        scope = str(asset_scope or "all").strip().lower()
        ws_path = Path(workspace_dir)
        loader = resolve_patched_attr(
            "load_env_file_defaults",
            load_env_file_defaults,
            modules=("providers.base", "botcore", __name__),
            require_callable=True,
        ) or load_env_file_defaults
        for venue_key in cls._VENUE_PROVIDER_SPECS:
            prov_cls = cls.resolve_venue_provider_class(venue_key)
            rel = getattr(prov_cls, "_RUNTIME_ENV_RELPATH", None) if prov_cls is not None else None
            prov_scope = str(getattr(prov_cls, "_ASSET_SCOPE", "crypto") or "crypto").strip().lower()
            if rel and (scope == "all" or scope == prov_scope):
                loader(str(ws_path.joinpath(*rel)))

    @classmethod
    def resolve_provider_cached_fee_and_limits(
        cls,
        venue: str,
        symbol: str,
        provider: Any,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Resolve cached fee/limit overrides from ``provider`` via public hook or venue adapter."""
        if provider is None:
            return {}
        inst_hook = getattr(provider, "resolve_cached_fee_and_limits", None)
        if callable(inst_hook) and getattr(type(provider), "resolve_cached_fee_and_limits", None) is not cls.resolve_cached_fee_and_limits:
            try:
                res = cls._invoke_fee_and_limits_callable(inst_hook, symbol=symbol, price=price)
                if res is not None:
                    return res
            except Exception:
                pass
        extract_fn = getattr(type(provider), "extract_cached_fee_and_limits", None)
        if not callable(extract_fn):
            prov_cls = cls.resolve_venue_provider_class(venue or getattr(provider, "name", ""))
            extract_fn = getattr(prov_cls, "extract_cached_fee_and_limits", None) if prov_cls is not None else None
        try:
            res = cls._invoke_fee_and_limits_callable(extract_fn, provider, symbol=symbol, price=price)
            if res is not None:
                return res
        except Exception:
            pass
        return {}

    @classmethod
    def resolve_instrument_cached_fee_and_limits(
        cls,
        inst: Any,
        venue: str,
        symbol: str,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Resolve cached fee and limit metadata from an ``Instrument`` or compatible stub."""
        if inst is None:
            return {}
        inst_hook = getattr(type(inst), "cached_fee_and_limits", None)
        if callable(inst_hook):
            try:
                res = inst.cached_fee_and_limits(price=price)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        mock_children = getattr(inst, "_mock_children", None)
        if isinstance(mock_children, dict) and "provider" in mock_children and "_provider" not in mock_children:
            prov = getattr(inst, "provider", None)
        else:
            prov = getattr(inst, "_provider", None)
            if prov is None or isinstance(prov, str):
                prov = getattr(inst, "provider", None)
        if prov is None or isinstance(prov, str):
            return {}
        return cls.resolve_provider_cached_fee_and_limits(
            venue, symbol, prov, price=price
        )

    @staticmethod
    def resolve_venue_instrument_spec(venue: Any, symbol: str) -> Any:
        """Resolve ``InstrumentSpec`` for ``symbol`` on ``venue`` from ``instrument_registry``."""
        if (sym_u := clean_upper_token(symbol)) is None:
            return None
        sym_clean = clean_symbol_key(sym_u)
        sym_canon = canonical_symbol_key(sym_u)
        try:
            ir = (
                resolve_patched_attr(
                    "instrument_registry",
                    _default_ir,
                    modules=("providers.market_api", "providers.base"),
                )
                or _default_ir
            )
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

    @classmethod
    def fetch_public_market_price_usd(cls, symbol: str) -> Optional[float]:
        """Return public unauthenticated USD spot/equity price for ``symbol`` if supported."""
        return None

    @classmethod
    def resolve_venue_public_price_usd(cls, venue: object, symbol: str) -> Optional[float]:
        """Resolve public USD market price for ``symbol`` via the registered venue provider class."""
        prov_cls = cls.resolve_venue_provider_class(venue)
        fn = getattr(prov_cls, "fetch_public_market_price_usd", None) if prov_cls is not None else None
        return fn(symbol) if callable(fn) else None

    @classmethod
    def public_orderbook_depth_urls(cls, symbol: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(primary_urls, secondary_urls)`` public L2 orderbook endpoints for ``symbol``."""
        return ((), ())

    @classmethod
    def resolve_orderbook_depth_urls(
        cls,
        symbol: str,
        *,
        venue: str = "",
        multi_venue: bool = False,
    ) -> tuple[tuple[str, ...], str]:
        """Return ``(ordered_depth_urls, hl_book_url)`` across registered venue providers."""
        canon_sym = canonical_symbol_key(symbol)
        if symbol_quote_asset(canon_sym) == "EUR":
            eur_urls: list[str] = []
            for v_key in cls._VENUE_PROVIDER_SPECS:
                prov_cls = cls.resolve_venue_provider_class(v_key)
                if prov_cls is not None and getattr(prov_cls, "_ASSET_SCOPE", "crypto") == "crypto":
                    prim, _ = prov_cls.public_orderbook_depth_urls(canon_sym)
                    eur_urls.extend(prim)
            return tuple(eur_urls), ""

        crypto_venues: list[str] = []
        l2_info_venues: set[str] = set()
        primary_map: dict[str, tuple[str, ...]] = {}
        secondary_map: dict[str, tuple[str, ...]] = {}
        for v_key in cls._VENUE_PROVIDER_SPECS:
            prov_cls = cls.resolve_venue_provider_class(v_key)
            if prov_cls is None or getattr(prov_cls, "_ASSET_SCOPE", "crypto") != "crypto":
                continue
            crypto_venues.append(v_key)
            if getattr(prov_cls, "_ORDERBOOK_L2_INFO_BOOK", False):
                l2_info_venues.add(v_key)
            prim, sec = prov_cls.public_orderbook_depth_urls(canon_sym)
            primary_map[v_key] = prim
            secondary_map[v_key] = sec

        l2_info_urls = tuple(
            u for v_k in crypto_venues if v_k in l2_info_venues for u in primary_map.get(v_k, ())
        )
        hl_book_url = l2_info_urls[0] if l2_info_urls else ""
        default_venue = next(
            (v_k for v_k in crypto_venues if primary_map.get(v_k)),
            crypto_venues[0] if crypto_venues else "",
        )
        v_norm = normalize_venue_name(venue)
        eff_venue = v_norm if (v_norm in primary_map and primary_map[v_norm]) else default_venue
        if multi_venue:
            return primary_map.get(eff_venue, ()), hl_book_url

        ordered: list[str] = list(primary_map.get(eff_venue, ()))
        for peer in crypto_venues:
            if peer != eff_venue and peer not in l2_info_venues:
                ordered.extend(primary_map.get(peer, ()))
        for peer in crypto_venues:
            ordered.extend(secondary_map.get(peer, ()))
        if eff_venue not in l2_info_venues:
            ordered.extend(l2_info_urls)
        return tuple(ordered), hl_book_url

    @classmethod
    def normalize_orderbook_payload(cls, raw_data: Any) -> Optional[dict]:
        """Normalize L2 orderbook payloads into ``{'bids': [(px, qty), ...], 'asks': [(px, qty), ...]}``."""
        if not isinstance(raw_data, dict):
            return None
        data = raw_data
        if "bids" not in data:
            extract_self = getattr(cls, "extract_orderbook_sides", None)
            extracted = extract_self(data) if callable(extract_self) else None
            if not isinstance(extracted, dict):
                for venue_key in cls._VENUE_PROVIDER_SPECS:
                    prov_cls = cls.resolve_venue_provider_class(venue_key)
                    extract_fn = getattr(prov_cls, "extract_orderbook_sides", None) if prov_cls is not None else None
                    if callable(extract_fn):
                        extracted = extract_fn(data)
                        if isinstance(extracted, dict):
                            break
            if not isinstance(extracted, dict):
                return None
            data = extracted

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
