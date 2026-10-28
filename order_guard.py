# order_guard.py
"""Platform-agnostic profit guard.

Rule: do not BUY above the last SELL or SELL below the last BUY, subject to a minimum
margin. Decoupled from Binance, it accepts any `provider` implementing
`last_opposite_fill(symbol, order_type)` and an optional caller-computed `window_ref`
(for example, min(sell)/max(buy) from the Binance order cache). The SAME profit logic
therefore runs on every venue instead of being embedded only in bapi_placeorder.

Market-regime data sources and provider lookup are injected by callers, while the
shared market-regime types remain provider-neutral. Reference read failures from
provider.last_opposite_fill propagate so the caller can fail closed.
Returns True when placement is allowed and False when blocked.

The profit threshold is configured per venue in versioned, non-sensitive
`order_guard.conf`. A missing file or required key aborts startup immediately."""
import os
import time
import math
import json
from botcore import _parse_bool_flag, parse_dotenv
from intelligence.macro_sentiment_ssot import safe_pos_float as _clean_pos_float
from typing import Optional, Callable, Any, List, Tuple, Dict
import utils as u
from market_regime import MarketRegimeService, MarketRegimeDecision, MarketRegimeContext
from providers.base import (
    extract_order_price,
    extract_order_qty,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_side,
)
from providers.strategy_executor import ProviderError

_DEFAULT_REGIME_SERVICE = MarketRegimeService()
_MARGINS = None   # cache: {provider_lower: percentage, "default": 1.15}
_MARGINS_FILE_MTIME: float = 0.0
_MARGINS_LAST_CHECK: float = 0.0
from llm_order_guard import _GEO_SHADOW_NOTIFY_COOLDOWN as _SHADOW_NOTIFY_COOLDOWN


_REQUIRED_ORDER_GUARD_KEYS = frozenset({
    "default",
    "binance",
    "kraken",
    "default_window_h",
    "kraken_window_h",
    "default_weight_proxy",
    "binance_weight_proxy",
    "kraken_weight_proxy",
    "default_chop_weight",
    "binance_chop_weight",
    "kraken_chop_weight",
    "default_max_daily_trades",
    "default_antispam_sec",
    "default_safeback_sec",
    "default_buy_reference",
    "binance_buy_reference",
    "kraken_buy_reference",
    "default_sell_reference",
    "binance_sell_reference",
    "kraken_sell_reference",
    "assetguardian_buy_reference",
    "assetguardian_sell_reference",
    "monitortrades_buy_reference",
    "monitortrades_sell_reference",
    "tradeall_buy_reference",
    "tradeall_sell_reference",
    "rtrade_buy_reference",
    "rtrade_sell_reference",
    "dynamic_window_favorable_h",
    "dynamic_window_flat_h",
    "dynamic_window_adverse_h",
    "regime_context_max_age_sec",
    # Pillar 1: Internal Quantitative Mathematical Guards
    "internal_guard_mode",
    "intelligence_guards_mode",
    "parabolic_surge_pct",
    "parabolic_pullback_pct",
    "weibull_exhaustion_policy",
    "weibull_exhausted_scale",
    "weibull_p90_days",
    # Pillar 2: External Microstructure & Derivatives Flow Guards
    "external_guard_mode",
    "whale_guard_mode",
    "severe_cascade_oi_pct",
    "min_taker_ratio",
    "taker_downscale_threshold",
    "taker_downscaled_scale",
    "orderbook_wall_guard_mode",
    "whale_wall_usd_limit",
    "min_buy_imbalance",
    "max_sell_imbalance",
    "high_stake_min_notional_usd",
    "funding_guard_mode",
    "funding_max_long_rate",
    "funding_min_short_rate",
    "funding_crowding_policy",
    "funding_crowded_scale",
    # Pillar 2 Sub-Module: Smart Stop-Loss
    "smart_sl_mode",
    "smart_sl_max_threshold_pct",
    "smart_sl_grace_seconds",
    "smart_sl_disaster_buffer_pct",
    "smart_sl_min_whale_long_pct",
    "smart_sl_min_taker_ratio",
    # Pillar 3: AI Intelligence High-Stake Guard & Post-Stop Re-entry
    "llm_guard_mode",
    "llm_timeout_sec",
    "llm_fallback",
    "post_stop_reentry_mode",
    "post_stop_reentry_timeout_sec",
    "post_stop_reentry_fallback",
    "post_stop_reentry_notify",
    "post_stop_reentry_cache_ttl_sec",
    "macro_advisor_interval_sec",
    "macro_llm_interval_sec",
    # Pillar 4 & Global
    "geopolitical_guard_mode",
    "shadow_notify",
    # Autonomous AI Supervisor & Trading Gateway
    "supervisor_mode",
    "supervisor_notify",
    "autonomous_trading_mode",
    "autonomous_trading_max_notional_usd",
    "autonomous_trading_min_notional_usd",
    "autonomous_trading_min_confidence",
    "autonomous_trading_notify",
    # Guard bypass configuration
    "default_bypass_guards",
    "binance_bypass_guards",
    "kraken_bypass_guards",
})

VALID_PROFIT_GUARD_MODES = frozenset({"strict", "average", "last", "hybrid", "dynamic", "off"})
VALID_GUARD_MODES = frozenset({"off", "shadow", "enforce"})

_GUARD_MODE_KEYS = (
    "internal_guard_mode",
    "intelligence_guards_mode",
    "external_guard_mode",
    "whale_guard_mode",
    "orderbook_wall_guard_mode",
    "funding_guard_mode",
    "smart_sl_mode",
    "llm_guard_mode",
    "post_stop_reentry_mode",
    "geopolitical_guard_mode",
    "supervisor_mode",
    "autonomous_trading_mode",
)

_MODULE_REFERENCE_KEYS = frozenset({
    "assetguardian_buy_reference",
    "assetguardian_sell_reference",
    "monitortrades_buy_reference",
    "monitortrades_sell_reference",
    "tradeall_buy_reference",
    "tradeall_sell_reference",
    "rtrade_buy_reference",
    "rtrade_sell_reference",
})

_DISCRETE_POLICY_SETS: Dict[str, frozenset] = {
    "weibull_exhaustion_policy": frozenset({"block", "veto", "downscale"}),
    "funding_crowding_policy": frozenset({"defer", "downscale"}),
    "llm_fallback": frozenset({"allow", "block", "veto", "downscale"}),
    "post_stop_reentry_fallback": frozenset({"allow", "block", "veto", "downscale"}),
}


def validate_profit_guard_mode_cfg(val: Any) -> str:
    """Validate that a profit guard mode is strictly one of VALID_PROFIT_GUARD_MODES."""
    if not isinstance(val, str):
        raise ValueError(
            f"Invalid profit guard mode {val!r}; must be a string in {sorted(VALID_PROFIT_GUARD_MODES)}"
        )
    mode = val.strip().lower()
    if mode not in VALID_PROFIT_GUARD_MODES:
        raise ValueError(
            f"Invalid profit guard mode {val!r}; must be one of {sorted(VALID_PROFIT_GUARD_MODES)}"
        )
    return mode


def validate_guard_mode_cfg(val: Any, key_name: str = "guard_mode") -> str:
    """Validate that a guard / supervisor / gateway mode is strictly in VALID_GUARD_MODES."""
    if not isinstance(val, str):
        raise ValueError(
            f"Invalid {key_name} {val!r}; must be a string in {sorted(VALID_GUARD_MODES)}"
        )
    mode = val.strip().lower()
    if mode not in VALID_GUARD_MODES:
        raise ValueError(
            f"Invalid {key_name} {val!r}; must be one of {sorted(VALID_GUARD_MODES)}"
        )
    return mode


def _validate_margins_cfg(m: Dict[str, Any], path: str) -> None:
    """Validate discrete mode and policy values in order_guard.conf at load time."""
    for key in _GUARD_MODE_KEYS:
        validate_guard_mode_cfg(m[key], key_name=key)

    for key, val in m.items():
        if key.endswith("_buy_reference") or key.endswith("_sell_reference"):
            if key in _MODULE_REFERENCE_KEYS:
                sval = str(val).strip().lower() if isinstance(val, str) else None
                if sval != "venue":
                    validate_profit_guard_mode_cfg(val)
            else:
                validate_profit_guard_mode_cfg(val)

    for policy_key, valid_set in _DISCRETE_POLICY_SETS.items():
        raw_val = m[policy_key]
        if not isinstance(raw_val, str) or raw_val.strip().lower() not in valid_set:
            raise ValueError(
                f"Invalid {policy_key} {raw_val!r} in {path}; must be one of {sorted(valid_set)}"
            )

    min_auto_usd = float(m["autonomous_trading_min_notional_usd"])
    max_auto_usd = float(m["autonomous_trading_max_notional_usd"])
    if not (math.isfinite(min_auto_usd) and min_auto_usd > 0):
        raise ValueError(
            f"Invalid autonomous_trading_min_notional_usd {m['autonomous_trading_min_notional_usd']!r} in {path}"
        )
    if not (math.isfinite(max_auto_usd) and max_auto_usd >= min_auto_usd):
        raise ValueError(
            f"Invalid autonomous_trading_max_notional_usd {m['autonomous_trading_max_notional_usd']!r} in {path}; "
            f"must be >= autonomous_trading_min_notional_usd ({min_auto_usd})"
        )


def _load_margins():
    """Read and cache `key = value` lines strictly from order_guard.conf.
    The configuration file is the SINGLE source of truth without code fallbacks.
    Checks file mtime periodically (every 5 seconds) to hot-reload config edits."""
    global _MARGINS, _MARGINS_FILE_MTIME, _MARGINS_LAST_CHECK
    now_ts = time.time()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_guard.conf")
    if _MARGINS is not None and 0.0 <= (now_ts - _MARGINS_LAST_CHECK) < 5.0:
        return _MARGINS
    _MARGINS_LAST_CHECK = now_ts
    if not os.path.exists(path):
        if _MARGINS is not None:
            return _MARGINS
        raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
    mtime = os.path.getmtime(path)
    if _MARGINS is not None and (_MARGINS_FILE_MTIME == 0.0 or mtime == _MARGINS_FILE_MTIME):
        _MARGINS_FILE_MTIME = mtime
        return _MARGINS
    _MARGINS_FILE_MTIME = mtime
    raw = parse_dotenv(path)
    m: Dict[str, Any] = {}
    for k, v in raw.items():
        k_clean = k.strip().lower()
        try:
            m[k_clean] = float(v)
        except (TypeError, ValueError):
            m[k_clean] = v
    missing = sorted(_REQUIRED_ORDER_GUARD_KEYS - set(m.keys()))
    if missing:
        if _MARGINS is not None and _parse_bool_flag(os.environ.get("BACKTEST_MODE")):
            return _MARGINS
        raise KeyError(f"Missing required keys in {path}: {', '.join(missing)}")
    _validate_margins_cfg(m, path)
    _MARGINS = m
    return m


def resolve_high_stake_min_notional_usd(
    margins: Dict[str, Any],
    *,
    side: Optional[str] = None,
    require_key: bool = False,
) -> float:
    """Resolve the unified high-stake notional threshold (high_stake_min_notional_usd) from margins."""
    if isinstance(margins, dict) and "high_stake_min_notional_usd" in margins:
        raw_val = margins["high_stake_min_notional_usd"]
        if raw_val is not None and not isinstance(raw_val, bool):
            try:
                f = float(raw_val)
                if math.isfinite(f) and f > 0.0:
                    return f
            except (TypeError, ValueError):
                pass
        return 700.0
    if require_key:
        raise KeyError("high_stake_min_notional_usd")
    return 700.0


_load_margins()


def _provider_name(provider):
    if provider is None or isinstance(provider, bool):
        return ""
    raw_name = provider if isinstance(provider, str) else getattr(provider, "name", "")
    if not raw_name or isinstance(raw_name, bool):
        return ""
    from cachepaths import normalize_venue_name
    return normalize_venue_name(str(raw_name))


_PROCESS_TO_MODULE = {
    "AG": "assetguardian",
    "MT": "monitortrades",
    "TA": "tradeall",
    "RT": "rtrade",
}
_KNOWN_GUARD_MODULES = frozenset(_PROCESS_TO_MODULE.values())


def resolve_caller_module(
    module: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    client_order_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve canonical module identity from explicit module, order lineage prefix, or process name."""
    if module and not isinstance(module, bool):
        m_raw = str(module).strip()
        if m_raw:
            m_up = m_raw.upper()
            if m_up in _PROCESS_TO_MODULE:
                return _PROCESS_TO_MODULE[m_up]
            m_low = m_raw.lower()
            if m_low == "asset_guardian":
                return "assetguardian"
            return m_low
    if (client_order_id and not isinstance(client_order_id, bool)) or (prefix and not isinstance(prefix, bool)):
        try:
            from binance_api.order_id_context import resolve_order_owner
            for candidate in (client_order_id, prefix):
                if not candidate or isinstance(candidate, bool):
                    continue
                owner = resolve_order_owner(candidate)
                if owner in _KNOWN_GUARD_MODULES:
                    return owner
        except Exception:
            pass
    try:
        from binance_api.order_id_context import get_process_name
        return _PROCESS_TO_MODULE.get(get_process_name())
    except Exception:
        return None


def _normalize_side_token(order_type: Any) -> str:
    if not isinstance(order_type, str):
        return ""
    return normalize_side(order_type)


# _parse_bool_flag imported from botcore (SSOT)


def opposite_side(order_type: str) -> str:
    """Return the opposite order side ('SELL' for 'BUY', 'BUY' for 'SELL')."""
    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    return "SELL" if side == "BUY" else "BUY"


def profit_diff_pct(order_type: str, price: float, ref_price: float) -> float:
    """Compute signed favorable profit percentage of `price` relative to `ref_price` symmetrically."""
    side = _normalize_side_token(order_type)
    if side == "BUY":
        return u.value_diff_to_percent(ref_price, price)
    if side == "SELL":
        return u.value_diff_to_percent(price, ref_price)
    raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")


def select_extreme_reference(prices, order_type: str, *, strict: bool = True):
    """Select the directional extreme reference symmetrically for BUY and SELL."""
    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    if prices is None or isinstance(prices, (bool, str, bytes)):
        return None
    clean_prices: List[float] = []
    try:
        for p in prices:
            if p is None or isinstance(p, bool):
                continue
            try:
                pf = float(p)
                if math.isfinite(pf) and pf > 0.0:
                    clean_prices.append(pf)
            except (TypeError, ValueError, OverflowError):
                continue
    except TypeError:
        return None
    if not clean_prices:
        return None
    if _parse_bool_flag(strict, True):
        return min(clean_prices) if side == "BUY" else max(clean_prices)
    return max(clean_prices) if side == "BUY" else min(clean_prices)


def get_profit_guard_mode_cfg(
    provider_name,
    order_type: str,
    *,
    module: Optional[str] = None,
    reference_mode: Optional[str] = None,
) -> str:
    """Read the BUY or SELL profit guard difficulty mode strictly from order_guard.conf.

    Fails fast (ValueError / KeyError) if any key or value is missing or invalid.
    """
    if reference_mode is not None:
        return validate_profit_guard_mode_cfg(reference_mode)

    side_u = _normalize_side_token(order_type)
    if side_u not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    side_l = side_u.lower()

    venue = _provider_name(provider_name).strip().lower()
    if not venue:
        raise ValueError(f"Missing provider_name for get_profit_guard_mode_cfg({provider_name!r}, {order_type!r})")

    m = _load_margins()
    mod = resolve_caller_module(module)

    if mod:
        mod_key = f"{mod}_{side_l}_reference"
        if mod_key not in m:
            raise KeyError(f"Missing required module config key {mod_key!r} in order_guard.conf")
        mod_val = str(m[mod_key]).strip().lower()
        if mod_val != "venue":
            return validate_profit_guard_mode_cfg(mod_val)

    venue_key = f"{venue}_{side_l}_reference"
    if venue_key in m:
        return validate_profit_guard_mode_cfg(m[venue_key])

    default_key = f"default_{side_l}_reference"
    if default_key in m:
        return validate_profit_guard_mode_cfg(m[default_key])

    raise KeyError(
        f"Missing required profit guard config key {venue_key!r} (and {default_key!r}) in order_guard.conf"
    )


def is_profit_guard_enabled_cfg(
    provider_name,
    order_type: str,
    *,
    module: Optional[str] = None,
    reference_mode: Optional[str] = None,
) -> bool:
    """Return True if the configured profit guard mode in order_guard.conf is not 'off'."""
    return get_profit_guard_mode_cfg(
        provider_name, order_type, module=module, reference_mode=reference_mode
    ) != "off"


def symbol_regime(
    symbol: str,
    provider=None,
    *,
    regime_service: Optional[MarketRegimeService] = None,
    now: Optional[float] = None,
    allow_fallback: bool = True,
    snapshot_resolver: Optional[Callable[[str, Optional[float]], Optional[dict]]] = None,
    provider_resolver: Optional[Callable[[Any], Any]] = None,
) -> MarketRegimeDecision:
    """Resolve the authoritative market regime decision for symbol via MarketRegimeService.

    Verifies freshness, observation timestamps, candle continuity, and provider fallback.
    Resolvers can be injected to avoid hidden imports and global state dependencies.
    """
    svc = regime_service or _DEFAULT_REGIME_SERVICE
    if not symbol:
        return svc.evaluator.unknown("missing_symbol")

    if hasattr(provider, "market_regime"):
        try:
            return provider.market_regime(symbol, allow_fallback=allow_fallback)
        except Exception:
            pass

    snap = None
    if snapshot_resolver is not None:
        try:
            snap = snapshot_resolver(symbol, now=now)
        except Exception:
            snap = None
    elif allow_fallback:
        # Graceful fallback for direct callers that did not inject a snapshot resolver
        try:
            import cacheManager as _cm
            mgr = _cm.get_short_trend_manager()
            snap_fn = getattr(mgr, "get_fresh_snapshot", None)
            snap = snap_fn(symbol, now=now) if callable(snap_fn) else None
        except Exception:
            snap = None

    target_provider = provider
    if provider_resolver is not None:
        try:
            target_provider = provider_resolver(provider)
        except Exception:
            target_provider = None
    elif allow_fallback and isinstance(provider, str) and provider:
        try:
            from providers.market_api import api as _market_api
            target_provider = _market_api.provider_by_name(provider) or provider
        except Exception:
            target_provider = provider

    try:
        resolution = svc.resolve_with_evidence(
            target_provider,
            symbol,
            horizon="short",
            snapshot=snap,
            snapshot_max_age_seconds=svc.default_snapshot_max_age_seconds(),
            allow_fallback=allow_fallback,
            now=now,
        )
        return resolution.decision
    except Exception:
        return svc.evaluator.unknown("regime_resolution_failed")


def symbol_regime_context(
    symbol: str,
    provider=None,
    *,
    regime_service: Optional[MarketRegimeService] = None,
    now: Optional[float] = None,
    allow_fallback: bool = True,
    snapshot_resolver: Optional[Callable[[str, Optional[float]], Optional[dict]]] = None,
    provider_resolver: Optional[Callable[[Any], Any]] = None,
    trend_duration_seconds: float = 0.0,
    benchmark_symbol: Optional[str] = None,
) -> MarketRegimeContext:
    """Resolve and bundle MarketRegimeContext once for multi-guard placement pipelines."""
    now_ts: Optional[float] = None
    if now is not None and not isinstance(now, bool):
        try:
            nf = float(now)
            if math.isfinite(nf) and nf >= 0.0:
                now_ts = nf
        except (TypeError, ValueError):
            pass
    if now_ts is None:
        now_ts = time.time()
    decision = symbol_regime(
        symbol,
        provider=provider,
        regime_service=regime_service,
        now=now_ts,
        allow_fallback=allow_fallback,
        snapshot_resolver=snapshot_resolver,
        provider_resolver=provider_resolver,
    )
    provider_name = getattr(provider, "name", None) or (provider if isinstance(provider, str) else None)
    dur_sec = _clean_pos_float(trend_duration_seconds, 0.0)
    if dur_sec <= 0.0 and symbol:
        dur_sec = _resolve_trend_duration(symbol)
    return MarketRegimeContext.from_decision(
        decision,
        evaluated_at=now_ts,
        symbol=symbol,
        provider=provider_name,
        trend_duration_seconds=dur_sec,
        benchmark_symbol=benchmark_symbol,
    )


def _symbol_trend(
    symbol: str,
    provider=None,
    regime_service=None,
    now=None,
    *,
    regime_context: Optional[MarketRegimeContext] = None,
    snapshot_resolver=None,
    provider_resolver=None,
) -> str:
    """Detect current macro/instant trend for symbol ('bull', 'bear', 'flat', 'unknown').

    Consumes the unified MarketRegimeDecision from MarketRegimeService.
    """
    if regime_context is not None:
        max_age = float(_load_margins()["regime_context_max_age_sec"])
        if hasattr(regime_context, "is_valid_for"):
            if regime_context.is_valid_for(
                symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
                require_identity=True,
            ):
                return regime_context.resolved_trend
        else:
            return regime_context.resolved_trend
    try:
        decision = symbol_regime(
            symbol,
            provider=provider,
            regime_service=regime_service,
            now=now,
            snapshot_resolver=snapshot_resolver,
            provider_resolver=provider_resolver,
        )
        regime = decision.regime
        if regime == "sideways":
            return "flat"
        return regime
    except Exception:
        return "unknown"


def get_margin_cfg(provider_name):
    """Return the venue's configured minimum profit percentage from order_guard.conf."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    return m.get(venue, m["default"])


def _is_favorable_trend(side: str, trend: str) -> bool:
    """Return True when the market trend aligns favorably with the order side."""
    return (side == "BUY" and trend == "bull") or (side == "SELL" and trend == "bear")


def _is_adverse_trend(side: str, trend: str) -> bool:
    """Return True when the market trend opposes the order side."""
    return (side == "BUY" and trend == "bear") or (side == "SELL" and trend == "bull")


def get_dynamic_window_sec_cfg(
    symbol=None,
    provider=None,
    order_type: str = "BUY",
    regime_service=None,
    now=None,
    *,
    resolved_trend=None,
    regime_context=None,
    snapshot_resolver=None,
    provider_resolver=None,
) -> float:
    """Calculate the dynamic lookback window (in seconds) symmetrically for BUY and SELL:
    - Favorable trend (BUY in bull / SELL in bear): 4h - 12h (configured via dynamic_window_favorable_h)
    - Adverse trend (BUY in bear / SELL in bull): 48h - 168h (configured via dynamic_window_adverse_h)
    - Flat / unknown trend: 12h - 48h (configured via dynamic_window_flat_h)
    """
    side = "BUY" if (order_type is None or order_type == "") else _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")

    m = _load_margins()
    favorable_h = float(m["dynamic_window_favorable_h"])
    flat_h = float(m["dynamic_window_flat_h"])
    adverse_h = float(m["dynamic_window_adverse_h"])

    trend = resolved_trend
    if trend is None and regime_context is not None:
        max_age = float(m["regime_context_max_age_sec"])
        if not hasattr(regime_context, "is_valid_for") or regime_context.is_valid_for(
            symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
            require_identity=True,
        ):
            trend = getattr(regime_context, "resolved_trend", None)
    if trend not in {"bull", "bear", "flat", "sideways", "unknown"}:
        trend = (
            _symbol_trend(
                symbol,
                provider=provider,
                regime_service=regime_service,
                now=now,
                regime_context=regime_context,
                snapshot_resolver=snapshot_resolver,
                provider_resolver=provider_resolver,
            )
            if symbol
            else "unknown"
        )
    if _is_favorable_trend(side, trend):
        hours = max(4.0, min(12.0, favorable_h))
    elif _is_adverse_trend(side, trend):
        hours = max(48.0, min(168.0, adverse_h))
    else:  # flat, sideways, chop, or unknown
        hours = max(12.0, min(48.0, flat_h))
    return hours * 3600.0


def get_window_cfg(
    provider_name,
    symbol=None,
    order_type=None,
    *,
    regime_context=None,
    module=None,
    reference_mode=None,
) -> float:
    """Return the configured reference window in seconds from order_guard.conf symmetrically
    for the given venue and side.

    When dynamic windowing is active (mode == 'dynamic'), returns get_dynamic_window_sec_cfg(...)
    scaling lookback from 4h-12h in favorable trend to 48h-168h in adverse trend.
    Otherwise returns `<venue>_window_h` (or default_window_h) in seconds.
    """
    name = _provider_name(provider_name)
    m = _load_margins()
    side = _normalize_side_token(order_type)

    if side in ("BUY", "SELL"):
        mode = get_profit_guard_mode_cfg(
            name, side, module=module, reference_mode=reference_mode
        )
        if mode == "off":
            return 0.0
        if mode == "dynamic":
            return get_dynamic_window_sec_cfg(
                symbol,
                provider=name,
                order_type=side,
                regime_context=regime_context,
            )

    key = (name.lower() if name else "") + "_window_h"
    hours_val = float(m.get(key, m["default_window_h"]))
    return hours_val * 3600.0


def get_weight_proxy_cfg(provider_name):
    """Return the trend/Gaussian-weight proxy symbol from order_guard.conf when the current symbol lacks
    its own long trend (e.g. HYPE -> BTC until data exists). Uses `<venue>_weight_proxy`
    or `default_weight_proxy`."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    return m.get(venue + "_weight_proxy", m["default_weight_proxy"])


def _order_ts_qty_price(o):
    if not isinstance(o, dict):
        return 0.0, 0.0, 0.0
    ts = extract_timestamp_s(o, default=0.0) or 0.0
    qty = extract_order_qty(o, default=0.0)
    px = extract_order_price(o, default=0.0)
    if not math.isfinite(ts) or ts < 0.0:
        ts = 0.0
    if not math.isfinite(qty) or qty < 0.0:
        qty = 0.0
    if not math.isfinite(px) or px < 0.0:
        px = 0.0
    return ts, qty, px


def _aggregate_reference_prices(active_lots, valid_opp, side: str, mode: str):
    """Aggregate active lots into a reference price symmetrically for BUY and SELL."""
    active_prices = []
    valid_weighted = []
    for item in (active_lots or []):
        try:
            q_raw, px_raw = item
        except (TypeError, ValueError):
            continue
        px_f = _clean_pos_float(px_raw, 0.0)
        if px_f > 0.0:
            active_prices.append(px_f)
            q_f = _clean_pos_float(q_raw, 0.0)
            if q_f > 0.0:
                valid_weighted.append((q_f, px_f))
    if not active_prices:
        return None
    strict_ref = select_extreme_reference(active_prices, side, strict=True)
    if mode in ("on", "strict", "dynamic"):
        return strict_ref
    if mode == "off":
        return None
    weighted_qty = sum(q for q, _ in valid_weighted)
    if weighted_qty > 0:
        avg_ref = sum(q * px for q, px in valid_weighted) / weighted_qty
    else:
        avg_ref = sum(active_prices) / len(active_prices)
    finite_ts_opp = []
    for item in (valid_opp or []):
        try:
            ts_raw, _, px_raw = item
        except (TypeError, ValueError):
            continue
        ts_f = _clean_pos_float(ts_raw, -1.0)
        px_f = _clean_pos_float(px_raw, 0.0)
        if ts_f >= 0.0 and px_f > 0.0:
            finite_ts_opp.append((ts_f, px_f))
    if finite_ts_opp:
        max_ts = max(ts for ts, _ in finite_ts_opp)
        last_ref = [px for ts, px in finite_ts_opp if ts == max_ts][-1]
    else:
        last_ref = active_prices[-1]
    if mode == "average":
        return avg_ref
    if mode == "last":
        return last_ref
    if mode == "hybrid":
        return select_extreme_reference([avg_ref, last_ref], side, strict=False)
    return strict_ref


def active_cycle_window_ref(opp_orders, same_orders=None, order_type="SELL", *, reference_mode=None):
    """Compute the window reference price from opposite-side fills, excluding earlier
    cycles that were already closed by subsequent same-side fills within the window."""
    if not opp_orders:
        return None
    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        return None
    mode = validate_profit_guard_mode_cfg(reference_mode or "strict")
    if mode == "off":
        return None
    valid_opp = []
    for o in opp_orders:
        ts, qty, px = _order_ts_qty_price(o)
        if px > 0 and math.isfinite(px):
            valid_opp.append((ts, qty, px))
    if not valid_opp:
        return None
    default_lots = [(qty, px) for _, qty, px in valid_opp]
    default_ref = _aggregate_reference_prices(default_lots, valid_opp, side, mode)
    if len(valid_opp) <= 1 or not same_orders:
        return default_ref

    valid_same = []
    for o in same_orders:
        ts, qty, px = _order_ts_qty_price(o)
        if ts > 0 and qty > 0:
            valid_same.append((ts, qty, px))
    if not valid_same or not all(ts > 0 and qty > 0 for ts, qty, _ in valid_opp):
        return default_ref

    events = (
        [(ts, 0, qty, px) for ts, qty, px in valid_opp]
        + [(ts, 1, qty, px) for ts, qty, px in valid_same]
    )
    events.sort(key=lambda x: (x[0], x[1]))

    lots = []  # list of [remaining_qty, initial_qty, price, ts]
    for ts, is_same, qty, px in events:
        if is_same == 0:
            lots.append([qty, qty, px, ts])
        else:
            rem_to_consume = qty
            while lots and rem_to_consume > 1e-12:
                rem_lot, init_lot, _lot_px, _lot_ts = lots[0]
                if rem_lot <= rem_to_consume + 1e-12:
                    rem_to_consume -= rem_lot
                    lots.pop(0)
                else:
                    new_rem = rem_lot - rem_to_consume
                    rem_to_consume = 0.0
                    if new_rem <= init_lot * 0.02 + 1e-12:
                        lots.pop(0)
                    else:
                        lots[0][0] = new_rem

    active_lots = [(lot[0], lot[2]) for lot in lots if lot[0] > lot[1] * 0.02 + 1e-12 and lot[2] > 0]
    active_opp = [(lot[3], lot[0], lot[2]) for lot in lots if lot[0] > lot[1] * 0.02 + 1e-12 and lot[2] > 0]
    if not active_lots:
        return default_ref
    return _aggregate_reference_prices(active_lots, active_opp, side, mode)


def window_reference(
    provider,
    symbol,
    order_type,
    window_s,
    *,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
):
    """Return a time-windowed reference from opposite-side orders/fills symmetrically
    for BUY and SELL according to the resolved difficulty mode ('on'/'strict', 'average',
    'last', 'hybrid', 'dynamic', or 'off').
    Return None for an empty or disabled window. Read errors propagate so the caller fails closed."""
    if window_s is None or isinstance(window_s, bool):
        return None
    try:
        window_s_f = float(window_s)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(window_s_f) or window_s_f <= 0:
        return None
    side = _normalize_side_token(order_type)
    eff_mode = get_profit_guard_mode_cfg(
        provider, side, module=module, reference_mode=reference_mode
    )
    if eff_mode == "off":
        return None
    opp = opposite_side(side)
    recent = provider.get_orders(symbol, opp, window_s_f)
    if recent is None:
        raise ProviderError(f"window_reference({symbol}): failed to read order history")
    prices = [extract_order_price(o, default=0.0) for o in recent if extract_order_price(o, default=0.0) > 0]
    if not prices:
        return None
    if len(recent) > 1 and all(
        isinstance(o, dict) and (extract_timestamp_ms(o, default=0) or 0) > 0
        for o in recent
    ):
        try:
            same_orders = provider.get_orders(symbol, side, window_s_f) or []
        except Exception:
            same_orders = []
        if same_orders or eff_mode in ("average", "last", "hybrid"):
            return active_cycle_window_ref(recent, same_orders, side, reference_mode=eff_mode)
    if eff_mode in ("average", "last", "hybrid"):
        return active_cycle_window_ref(recent, None, side, reference_mode=eff_mode)
    return select_extreme_reference(prices, side, strict=True)


def get_chop_weight_cfg(provider_name):
    """Return the venue's conservative chop fallback weight from order_guard.conf."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    key = venue + "_chop_weight"
    val = float(m.get(key, m["default_chop_weight"]))
    if not math.isfinite(val) or val <= 0:
        raise ValueError(f"Invalid chop_weight in order_guard.conf: {val!r}")
    return val


def resolve_trade_weight(symbol, order_type, provider_name=None, *, default_weight=None, raise_on_error=False):
    """Resolve trade weight using priceAnalysis Gaussian curve, proxy, or conservative fallback.

    1. Gaussian weight for symbol when long trend exists.
    2. Proxy Gaussian weight (e.g. BTCUSDC) when symbol has no trend.
    3. Conservative chop fallback (default 0.03 or venue-configured chop weight).
    """
    import math

    def _gauss(sym):
        try:
            import priceAnalysis as pa
            val = pa.get_weight_for_cash_permission_at_quant_time(sym, order_type)
            if val is not None:
                fval = float(val)
                if not math.isfinite(fval) or not 0 < fval <= 1:
                    if raise_on_error and sym == symbol:
                        from providers.strategy_executor import SubmissionRefused
                        raise SubmissionRefused("invalid_weight_policy_weight")
                    return None
                return fval
            return None
        except Exception as e:
            if raise_on_error and sym == symbol:
                raise
            print(f"[WEIGHT] {sym}: cannot compute the gauss value ({e})")
            return None

    weight = _gauss(symbol)
    if weight is None:
        proxy = get_weight_proxy_cfg(provider_name)
        if proxy and proxy != symbol:
            try:
                pw = _gauss(proxy)
                if pw is not None:
                    weight = pw
                    print(f"[WEIGHT] {symbol}: no trend of its own -> proxy {proxy} (weight={weight})")
            except Exception as e:
                print(f"[WEIGHT] {symbol}: proxy {proxy} lookup failed ({e})")

    if weight is None:
        if default_weight is None:
            default_weight = get_chop_weight_cfg(provider_name)
        weight = default_weight
        print(f"[WEIGHT] {symbol}: no trend or proxy -> fallback weight={weight}")

    return float(weight)


def compute_weight_capped_qty(weight, price, required_qty, available_qty, traded_24h_value):
    """Cap per-order quantity using Gaussian/chop weight and 24h traded quote reference."""
    w = min(1.0, _clean_pos_float(weight, 0.0))
    px = _clean_pos_float(price, 0.0)
    if required_qty is None or isinstance(required_qty, bool):
        return 0.0
    try:
        req = float(required_qty)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(req) or req <= 0.0 or w <= 0.0 or px <= 0.0:
        return 0.0
    avail = _clean_pos_float(available_qty, 0.0)
    traded = _clean_pos_float(traded_24h_value, 0.0)

    total_ref = traded + avail * px
    max_trade_value = total_ref * w
    remaining_value = max(0.0, max_trade_value - traded)
    remaining_qty = remaining_value / px if px > 0 else 0.0
    return min(req, remaining_qty)


def weight_limit(provider, symbol, order_type, price, required_qty, *, available_qty):
    """Cap per-order quantity using the Gaussian curve, the platform-agnostic equivalent
    of bapi.apply_weight_limit. Allocate tradable value by trend position so the whole
    amount is not bought or sold at once. priceAnalysis supplies the Gaussian weight;
    provider.get_orders supplies 24-hour traded value. The shared quantity decision must
    supply the side-aware balance, which this guard does not re-read. Return the smaller
    of requested and permitted quantity. Errors propagate so the caller fails closed."""
    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    provider_name = _provider_name(provider)
    weight = resolve_trade_weight(symbol, side, provider_name)
    recent = provider.get_orders(symbol, side, 86400)
    if recent is None:
        raise ProviderError(f"weight_limit({symbol}): order history unavailable")
    traded_value = sum(
        qty * px
        for o in recent
        for _ts, qty, px in (_order_ts_qty_price(o),)
    )
    available = _clean_pos_float(available_qty, 0.0)
    px_clean = _clean_pos_float(price, 0.0)
    if required_qty is None or isinstance(required_qty, bool):
        req_clean = 0.0
    else:
        try:
            rf = float(required_qty)
            req_clean = rf if (not math.isnan(rf) and rf > 0.0) else 0.0
        except (TypeError, ValueError, OverflowError):
            req_clean = 0.0
    adjusted = compute_weight_capped_qty(weight, px_clean, req_clean, available, traded_value)
    total_ref = traded_value + available * px_clean
    max_trade_value = total_ref * weight
    remaining_value = max(0.0, max_trade_value - traded_value)
    print(f"[WEIGHT] {side} {symbol}: weight={weight} traded24h={traded_value:.2f} "
          f"avail={available:.6f} max={max_trade_value:.2f} remaining={remaining_value:.2f} "
          f"cerut={req_clean:.6f} -> {adjusted:.6f}")
    return adjusted


def get_max_daily_trades_cfg(provider_name):
    """Return the venue's 24h trade cap strictly from order_guard.conf."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    key = venue + "_max_daily_trades"
    return int(m.get(key, m["default_max_daily_trades"]))


def get_antispam_sec_cfg(provider_name):
    """Return the per-symbol anti-spam pause in seconds from order_guard.conf."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    key = venue + "_antispam_sec"
    return float(m.get(key, m["default_antispam_sec"]))


def get_safeback_sec_cfg(provider_name):
    """Return the per-venue historical lookback window in seconds for the profit guard reference."""
    m = _load_margins()
    venue = _provider_name(provider_name).strip().lower()
    key = venue + "_safeback_sec"
    return float(m.get(key, m["default_safeback_sec"]))


DAILY_LIMIT_WINDOW_SEC = 86400.0


def daily_limit_guard(provider, symbol, order_type, max_daily_trades=None, *, bypass_guards=None):
    """Enforce GO1 symmetrically per symbol and side:
    1. Max 25 filled orders per 24 hours (86400s).
    2. Minimum 5-second anti-spam pause between orders on the same symbol and side.

    Return (True, None) or (False, "daily_limit" | "history_unavailable").
    """
    if bypass_guards:
        from guard_hierarchy import normalize_bypassed_guards, GUARD_REGISTRY
        bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
        if GUARD_REGISTRY["GO1"].is_bypassed(bypassed):
            return True, None
    name = _provider_name(provider)
    order_type = _normalize_side_token(order_type)
    if order_type not in ("BUY", "SELL"):
        return False, "invalid_side"
    resolved_max_trades = None
    if max_daily_trades is not None and not isinstance(max_daily_trades, bool):
        try:
            mt_f = float(max_daily_trades)
            if math.isfinite(mt_f) and mt_f >= 0 and mt_f == int(mt_f):
                resolved_max_trades = int(mt_f)
        except (TypeError, ValueError, OverflowError):
            resolved_max_trades = None
    max_daily_trades = (
        resolved_max_trades
        if resolved_max_trades is not None
        else get_max_daily_trades_cfg(name)
    )
    antispam_sec = _clean_pos_float(get_antispam_sec_cfg(name), 0.0)
    try:
        trades = provider.get_orders(symbol, order_type, DAILY_LIMIT_WINDOW_SEC)
    except ProviderError as exc:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history unavailable ({exc}) -> BLOCKED")
        return False, "history_unavailable"
    except Exception as exc:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history read failed ({exc}) -> BLOCKED")
        return False, "history_unavailable"
    if trades is None:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history unavailable -> BLOCKED")
        return False, "history_unavailable"
    if len(trades) >= max_daily_trades:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: {len(trades)} trades in 24h, "
              f"cap of {max_daily_trades}/24h -> BLOCKED")
        return False, "daily_limit"
    if antispam_sec > 0 and trades:
        now_s = time.time()
        for t in trades:
            if not isinstance(t, dict):
                continue
            ts_s = extract_timestamp_s(t, default=0.0) or 0.0
            if math.isfinite(ts_s) and ts_s > 0:
                elapsed_s = max(0.0, now_s - ts_s)
                if elapsed_s < antispam_sec:
                    print(f"[DAILY-LIMIT] {order_type} {symbol}: last fill {elapsed_s:.1f}s ago "
                          f"< {antispam_sec:.0f}s anti-spam -> BLOCKED")
                    return False, "daily_limit"
    return True, None



_whale_collector = None
_orderbook_collector = None
_derivatives_collector = None


def get_whale_collector():
    global _whale_collector
    if _whale_collector is not None:
        return _whale_collector
    from external_order_guard import get_whale_collector as _gwc
    return _gwc()


def get_orderbook_collector():
    global _orderbook_collector
    if _orderbook_collector is not None:
        return _orderbook_collector
    from external_order_guard import get_orderbook_collector as _goc
    return _goc()


def get_derivatives_collector():
    global _derivatives_collector
    if _derivatives_collector is not None:
        return _derivatives_collector
    from external_order_guard import get_derivatives_collector as _gdc
    return _gdc()


def _normalize_base_asset(sym: str) -> str:
    from cachepaths import normalize_cache_base_asset
    return normalize_cache_base_asset(sym)


def _resolve_cachedb_dir() -> str:
    from cachepaths import get_cachedb_dir
    return get_cachedb_dir()


def _read_json_recovering(path: str) -> Optional[dict]:
    from cachepaths import resolve_cachedb_filepath
    from state_io import previous_generation_path, read_recovering_previous

    resolved = str(resolve_cachedb_filepath(path))
    backup_path = previous_generation_path(resolved)
    if not os.path.exists(resolved) and not os.path.exists(backup_path):
        return None

    def _loader(candidate: str) -> dict:
        with open(candidate, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("cache payload must be a dict")
        return payload

    return read_recovering_previous(resolved, _loader, backup_path=backup_path)


def _read_cached_long_trend_entries(symbol: str) -> Optional[list]:
    try:
        from cachepaths import cache_symbol_variants
        cachedb_dir = _resolve_cachedb_dir()
        p = os.path.join(cachedb_dir, "cache_long_trend.json")
        data = _read_json_recovering(p)
        if not isinstance(data, dict):
            return None
        all_items = data.get("items", {})
        if not isinstance(all_items, dict):
            return None
        for cand in (symbol, symbol.upper(), *cache_symbol_variants(symbol)):
            items = all_items.get(cand)
            if items and isinstance(items, list):
                return items
        su_base = _normalize_base_asset(symbol)
        for k, v in all_items.items():
            if v and isinstance(v, list) and _normalize_base_asset(k) == su_base:
                return v
    except Exception:
        pass
    return None


def _read_cached_trend_duration(symbol: str) -> float:
    try:
        items = _read_cached_long_trend_entries(symbol)
        if items:
            for entry in reversed(items):
                if entry is None:
                    break
                if isinstance(entry, dict) and entry.get("duration_seconds") is not None:
                    raw_dur = entry.get("duration_seconds", 0.0)
                    if isinstance(raw_dur, bool):
                        continue
                    dur = float(raw_dur or 0.0)
                    if math.isfinite(dur) and dur > 0:
                        return dur
    except Exception:
        pass
    return 0.0


def _resolve_trend_duration(symbol: str) -> float:
    """Resolve trend duration in seconds from memory/cache files once."""
    if not symbol or not isinstance(symbol, str):
        return 0.0
    try:
        import cacheManager as cm
        trend_meta = getattr(cm, "read_long_term_trend_file", lambda s: None)(symbol)
        if trend_meta and "duration_seconds" in trend_meta:
            raw_dur = trend_meta["duration_seconds"]
            if not isinstance(raw_dur, bool):
                dur = float(raw_dur or 0.0)
                if math.isfinite(dur) and dur > 0:
                    return dur
    except Exception:
        pass
    return _read_cached_trend_duration(symbol)


def _read_cached_trend_direction(symbol: str) -> Optional[str]:
    try:
        items = _read_cached_long_trend_entries(symbol)
        if items:
            for entry in reversed(items):
                if entry is None:
                    break
                if isinstance(entry, dict) and isinstance(entry.get("direction"), str):
                    d = entry.get("direction").strip().lower()
                    if d in ("up", "bull", "uptrend", "bullish"):
                        return "bull" if d in ("uptrend", "bullish") else d
                    if d in ("down", "bear", "downtrend", "bearish"):
                        return "bear" if d in ("downtrend", "bearish") else d
    except Exception:
        pass
    return None


def _resolve_trend_direction(symbol: str, regime_context=None) -> Optional[str]:
    """Resolve long-term trend direction ('up'/'down'/'bull'/'bear') from regime_context or cache."""
    if regime_context is not None:
        if isinstance(regime_context, dict):
            rt = regime_context.get("resolved_trend") or regime_context.get("regime")
        else:
            rt = getattr(regime_context, "resolved_trend", None) or getattr(regime_context, "regime", None)
        if isinstance(rt, str):
            rtl = rt.strip().lower()
            if rtl in ("up", "bull", "uptrend", "bullish"):
                return "bull" if rtl in ("uptrend", "bullish") else rtl
            if rtl in ("down", "bear", "downtrend", "bearish"):
                return "bear" if rtl in ("downtrend", "bearish") else rtl
    if not symbol:
        return None
    try:
        import cacheManager as cm
        trend_meta = getattr(cm, "read_long_term_trend_file", lambda s: None)(symbol)
        if trend_meta and trend_meta.get("direction"):
            d = str(trend_meta["direction"]).strip().lower()
            if d in ("up", "bull", "uptrend", "bullish"):
                return "bull" if d in ("uptrend", "bullish") else d
            if d in ("down", "bear", "downtrend", "bearish"):
                return "bear" if d in ("downtrend", "bearish") else d
    except Exception:
        pass
    return _read_cached_trend_direction(symbol)


def _read_cached_price_history(
    symbol: str,
    window_seconds: float = 7200.0,
    now: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Read rolling price history from local cache for parabolic surge evaluation.

    Prefers the authoritative 24h per-symbol cache (cachedb/cache_24_price_{symbol}.json)
    which captures every high-resolution tick, and falls back to cache_prices_multi.json
    with exact normalized base-symbol matching for cross-venue assets (e.g. HYPE on Kraken).
    Never uses substring matching so e.g. ETHFI does not accidentally match ETH.
    """
    if not symbol or not isinstance(symbol, str):
        return None
    su = symbol.strip().upper()
    if not su:
        return None
    win_sec = _clean_pos_float(window_seconds, 7200.0)
    now_ts = time.time()
    if now is not None and not isinstance(now, bool):
        try:
            nf = float(now)
            if math.isfinite(nf) and nf > 0.0:
                now_ts = nf
        except (TypeError, ValueError, OverflowError):
            pass
    cachedb_dir = _resolve_cachedb_dir()
    cutoff_ts = now_ts - win_sec
    history: List[Tuple[float, float]] = []

    # 1. Primary: dedicated 24h cache (cache_24_price_{symbol}.json)
    from cachepaths import cache_symbol_variants
    su_base = _normalize_base_asset(su)
    safe_su = su.replace("/", "").replace("-", "").replace("_", "")
    cand_symbols: List[str] = []
    for c in (safe_su, su, *cache_symbol_variants(symbol), f"{su_base}USDC", f"{su_base}USDT", f"{su_base}USD", su_base):
        if c and c not in cand_symbols:
            cand_symbols.append(c)

    def _extract_valid_ticks(raw_items: Any) -> None:
        for item in raw_items or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if isinstance(item[0], bool) or isinstance(item[1], bool):
                    continue
                try:
                    ts_raw = float(item[0])
                    px = float(item[1])
                    if not (math.isfinite(ts_raw) and math.isfinite(px)):
                        continue
                    ts_sec = ts_raw / 1000.0 if ts_raw > 1e11 else ts_raw
                    if cutoff_ts <= ts_sec <= (now_ts + 5.0) and px > 0:
                        history.append((ts_sec, px))
                except (ValueError, TypeError, OverflowError):
                    continue

    p24_paths: List[str] = []
    for cand in cand_symbols:
        if "/" in cand or "_" in cand:
            continue
        p_cand = os.path.join(cachedb_dir, f"cache_24_price_{cand}.json")
        if p_cand not in p24_paths:
            p24_paths.append(p_cand)
    for p24 in p24_paths:
        data = _read_json_recovering(p24)
        if not isinstance(data, dict):
            continue
        try:
            items_map = data.get("items", {})
            raw_items = None
            if isinstance(items_map, dict):
                for cand in cand_symbols:
                    raw_items = items_map.get(cand)
                    if raw_items:
                        break
                if not raw_items:
                    for k, v in items_map.items():
                        if v and _normalize_base_asset(str(k)) == su_base:
                            raw_items = v
                            break
            elif isinstance(items_map, list):
                raw_items = items_map
            _extract_valid_ticks(raw_items)
            if history:
                break
        except Exception:
            pass

    # 2. Fallback: shared multi-symbol cache (cache_prices_multi.json)
    if not history:
        p_multi = os.path.join(cachedb_dir, "cache_prices_multi.json")
        data = _read_json_recovering(p_multi)
        if isinstance(data, dict):
            try:
                all_items = data.get("items", {})
                raw_items = None
                for cand in cand_symbols:
                    raw_items = all_items.get(cand)
                    if raw_items:
                        break
                if not raw_items:
                    for k, v in all_items.items():
                        if k and _normalize_base_asset(k) == su_base:
                            raw_items = v
                            break
                _extract_valid_ticks(raw_items)
            except Exception:
                pass

    return history if history else None


def check_intelligence_guards(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    *,
    price_history=None,
    trend_duration_seconds: float = 0.0,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> tuple[bool, str, float]:
    """Evaluate intelligence guards across all active pillars.

    Supports staged rollouts via order_guard.conf:
    - 'intelligence_guards_mode': 'off' | 'shadow' | 'enforce' (Pillar 1)
    - 'gemini_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 3, purchases >= 1000 EUR)
    - 'geopolitical_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 4, Black Swan Shield)
    - 'whale_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Whale Flow & OI Divergence)
    - 'orderbook_wall_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Orderbook Depth & Ask Walls)
    - 'funding_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Derivatives Funding Crowding)

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    from guard_hierarchy import normalize_bypassed_guards, GUARD_REGISTRY
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    m = _load_margins()
    valid_context = False
    if regime_context is not None:
        max_age = float(m["regime_context_max_age_sec"])
        if hasattr(regime_context, "is_valid_for"):
            valid_context = regime_context.is_valid_for(
                symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
                require_identity=True,
            )
        else:
            valid_context = True

    mode = validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode")
    if mode == "off":
        return True, "intelligence_guards_off", 1.0

    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    px_clean = _clean_pos_float(price, 0.0)
    qty_clean = _clean_pos_float(qty, 0.0)
    notional_clean = _clean_pos_float(notional_eur, 0.0)
    computed_notional = (
        notional_clean
        if notional_clean > 0.0
        else ((px_clean * qty_clean) if (px_clean > 0.0 and qty_clean > 0.0) else None)
    )
    explicit_dur_sec = _clean_pos_float(trend_duration_seconds, 0.0)

    # Check if expensive static analysis is already cached on regime_context for this (side, intel_bypass) key
    _raw_intel_codes = ("GT_L", "G_QLIMIT1", "G_QLIMIT2", "GI_OBOOK", "GI_WHALE", "GI_FUNDING", "GLLM2", "GLLM3")
    cache_key = (
        side,
        frozenset(code for code in _raw_intel_codes if GUARD_REGISTRY[code].is_bypassed(bypassed)),
    )
    legacy_cache_key = (side, frozenset(bypassed))
    decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None) if valid_context else None
    if isinstance(decisions_map, dict) and (cache_key in decisions_map or legacy_cache_key in decisions_map):
        raw_entry = decisions_map.get(cache_key)
        if not isinstance(raw_entry, dict):
            raw_entry = decisions_map.get(legacy_cache_key)
        cached_entry = raw_entry if isinstance(raw_entry, dict) else {}
        cached_res = cached_entry.get("decision")
        cached_price = cached_entry.get("price")
        cached_notional = cached_entry.get("notional")
        cached_dur = _clean_pos_float(cached_entry.get("trend_duration_seconds"), 0.0)
    elif valid_context and getattr(regime_context, "_intelligence_decision_key", cache_key) in (cache_key, legacy_cache_key):
        cached_res = getattr(regime_context, "_intelligence_decision", None)
        cached_price = getattr(regime_context, "_intelligence_price", None)
        cached_notional = getattr(regime_context, "_intelligence_notional", None)
        cached_dur = _clean_pos_float(getattr(regime_context, "_intelligence_trend_duration_seconds", 0.0), 0.0)
    else:
        cached_res = None
        cached_price = None
        cached_notional = None
        cached_dur = 0.0

    if (
        cached_res is not None
        and (
            (
                not cached_res[0]
                and side in ("BUY", "SELL")
                and ("parabolic_surge" in str(cached_res[1]) or "parabolic_dump" in str(cached_res[1]))
                and (price != cached_price or now is not None)
            )
            or (
                ("weibull_exhaustion" in str(cached_res[1]) or "trend_exhausted" in str(cached_res[1]))
                and explicit_dur_sec != cached_dur
            )
        )
    ):
        cached_res = None

    was_cached = cached_res is not None
    if cached_res is None:
        raw_bypass = (bypassed | {"gllm1"}) if valid_context else bypassed
        cached_res = _evaluate_intelligence_guards_raw(
            provider,
            symbol,
            side,
            price,
            price_history=price_history,
            trend_duration_seconds=explicit_dur_sec,
            regime_context=regime_context if valid_context else None,
            qty=qty,
            notional_eur=computed_notional,
            now=now,
            bypass_guards=raw_bypass,
        )
        if valid_context:
            try:
                decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None)
                if not isinstance(decisions_map, dict):
                    decisions_map = {}
                existing_entry = decisions_map.get(cache_key, {})
                decisions_map[cache_key] = {
                    **existing_entry,
                    "decision": cached_res,
                    "price": price,
                    "notional": computed_notional,
                    "trend_duration_seconds": explicit_dur_sec,
                }
                object.__setattr__(regime_context, "_intelligence_decisions_by_key", decisions_map)
                object.__setattr__(regime_context, "_intelligence_decision_key", cache_key)
                object.__setattr__(regime_context, "_intelligence_decision", cached_res)
                object.__setattr__(regime_context, "_intelligence_price", price)
                object.__setattr__(regime_context, "_intelligence_notional", computed_notional)
                object.__setattr__(regime_context, "_intelligence_trend_duration_seconds", explicit_dur_sec)
            except Exception:
                pass

    static_allowed, static_reason, static_scale = cached_res
    if not static_allowed:
        # Rejected by baseline or macro guard (e.g. Black Swan, Whale divergence, Ask Wall)
        return False, static_reason, 0.0

    if not valid_context:
        return cached_res

    effective_scale = static_scale
    active_reason = static_reason

    skip_internal = GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(bypassed)
    internal_mode = validate_guard_mode_cfg(m["internal_guard_mode"], "internal_guard_mode")
    internal_active = was_cached and not skip_internal and internal_mode != "off"

    # Dynamic Re-evaluation for cached context:
    # 1. Dynamic Parabolic Surge / Dump Guard (Anti-FOMO on BUY, Anti-Capitulation on SELL)
    if internal_active and px_clean > 0:
        from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
        from intelligence.internal.guards.guard_decision import BrakeAction
        surge_pct = float(m["parabolic_surge_pct"])
        pullback_pct = float(m["parabolic_pullback_pct"])
        history = price_history
        if history is None:
            try:
                history = _read_cached_price_history(symbol, window_seconds=7200.0, now=now)
            except TypeError:
                history = _read_cached_price_history(symbol, window_seconds=7200.0)
        if history:
            p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
            p_dec = p_guard.check(symbol, side, px_clean, price_history=history, now=now)
            if not p_dec.allowed:
                prefix = "[INTELLIGENCE_GUARD_SHADOW]" if internal_mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol} @ {price}: {p_dec.reason} (brake={p_dec.brake_action})")
                if internal_mode == "enforce":
                    return False, p_dec.reason, 0.0

    # Resolve trend duration early so both LLM guard and Weibull guard have access
    raw_ctx_dur = (
        regime_context.get("trend_duration_seconds", 0.0)
        if isinstance(regime_context, dict)
        else getattr(regime_context, "trend_duration_seconds", 0.0)
    ) if valid_context else 0.0
    dur_sec = explicit_dur_sec or _clean_pos_float(raw_ctx_dur, 0.0)
    if dur_sec <= 0.0 and symbol:
        dur_sec = _resolve_trend_duration(symbol)

    # 2. Dynamic LLM Pre-Trade Guard if notional crossed the threshold
    from llm_order_guard import evaluate_high_stake_order_guard

    hs_ok, hs_reason, hs_scale = evaluate_high_stake_order_guard(
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        computed_notional=computed_notional,
        regime_context=regime_context,
        valid_context=valid_context,
        cache_key=cache_key,
        margins=m,
        bypassed=bypassed,
        now=now,
        dur_sec=dur_sec,
        fallback_notional=cached_notional,
        notify_on_eval=True,
    )
    if not hs_ok:
        return False, hs_reason, 0.0
    if hs_scale < 1.0 and hs_scale <= effective_scale:
        effective_scale = hs_scale
        active_reason = hs_reason

    # 3. Dynamic Weibull trend exhaustion check if duration provided or resolved
    if internal_active and dur_sec > 0.0:
        from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
        from intelligence.internal.guards.guard_decision import BrakeAction
        from qlimit_order_guard import resolve_cached_gaussian_horizon
        e_policy = str(m["weibull_exhaustion_policy"]).strip().lower()
        e_scale = float(m["weibull_exhausted_scale"])
        e_p90 = float(m["weibull_p90_days"])
        e_guard = WeibullExhaustionGuard(default_p90_days=e_p90, policy=e_policy, exhausted_scale=e_scale)
        explicit_dur = explicit_dur_sec > 0.0
        ctx_rt = (
            (regime_context.get("resolved_trend") or regime_context.get("regime"))
            if isinstance(regime_context, dict)
            else (getattr(regime_context, "resolved_trend", None) or getattr(regime_context, "regime", None))
        )
        if explicit_dur and ctx_rt is None:
            trend_dir = None
        else:
            trend_dir = _resolve_trend_direction(symbol, regime_context=regime_context if valid_context else None)
        gauss_horizon_d = None
        p90_d = None
        median_d = None
        if not explicit_dur:
            qlimit1_active = not GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(bypassed)
            p90_d, median_d, cached_gauss_h = resolve_cached_gaussian_horizon(symbol)
            if qlimit1_active and cached_gauss_h:
                gauss_horizon_d = cached_gauss_h
        e_dec = e_guard.check(
            symbol,
            side,
            trend_duration_seconds=dur_sec,
            p90_days=p90_d,
            median_days=median_d,
            trend_direction=trend_dir,
            gaussian_horizon_days=gauss_horizon_d,
        )
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if internal_mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if internal_mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                s_val = _clean_pos_float(getattr(e_dec, "suggested_scale", None), 0.0)
                if s_val <= 0.0:
                    return False, e_dec.reason, 0.0
                s_val = min(1.0, s_val)
                if s_val < 1.0 and s_val <= effective_scale:
                    effective_scale = s_val
                    active_reason = e_dec.reason

    final_res = (True, active_reason if effective_scale < 1.0 else "ok", effective_scale)
    if valid_context:
        try:
            decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None)
            if isinstance(decisions_map, dict) and cache_key in decisions_map:
                decisions_map[cache_key]["price"] = price
                if computed_notional is not None:
                    decisions_map[cache_key]["notional"] = computed_notional
            object.__setattr__(regime_context, "_intelligence_price", price)
            if computed_notional is not None:
                object.__setattr__(regime_context, "_intelligence_notional", computed_notional)
            scales_by_side = getattr(regime_context, "_suggested_scale_by_side", None)
            if not isinstance(scales_by_side, dict):
                scales_by_side = {}
            scales_by_side[side] = effective_scale
            object.__setattr__(regime_context, "_suggested_scale_by_side", scales_by_side)
            if side == "BUY":
                object.__setattr__(regime_context, "suggested_scale", effective_scale)
                object.__setattr__(regime_context, "intelligence_reason", active_reason)
        except Exception:
            pass
    return final_res


def _evaluate_intelligence_guards_raw(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    *,
    price_history=None,
    trend_duration_seconds: float = 0.0,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> tuple[bool, str, float]:
    from guard_hierarchy import normalize_bypassed_guards, GUARD_REGISTRY
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if "all" in bypassed or "*" in bypassed:
        return True, "stop_loss_exempt", 1.0

    m = _load_margins()
    mode = validate_guard_mode_cfg(m["intelligence_guards_mode"], "intelligence_guards_mode")
    if mode == "off":
        return True, "intelligence_guards_off", 1.0

    side = _normalize_side_token(order_type)
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    if any(fake in (symbol or "").upper() for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE")):
        return True, "test_symbol_exempt", 1.0

    px_clean = _clean_pos_float(price, 0.0)
    qty_clean = _clean_pos_float(qty, 0.0)
    notional_clean = _clean_pos_float(notional_eur, 0.0)
    computed_notional = (
        notional_clean
        if notional_clean > 0.0
        else ((px_clean * qty_clean) if (px_clean > 0.0 and qty_clean > 0.0) else None)
    )

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Internal Quantitative Mathematical Guards (G_QLIMIT2: Parabolic Surge, Weibull Exhaustion)
    if not GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(bypassed):
        from qlimit_order_guard import check_qlimit2_order_guards
        int_ok, int_reason, int_scale = check_qlimit2_order_guards(
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            notional_eur=computed_notional,
            price_history=price_history,
            trend_duration_seconds=trend_duration_seconds,
            regime_context=regime_context,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not int_ok:
            return False, int_reason, 0.0
        if int_scale < 1.0:
            effective_scale = min(effective_scale, int_scale)
            active_reason = int_reason

    # 2. External Microstructure & Derivatives Guards (GI_OBOOK, GI_WHALE, GI_FUNDING)
    skip_external = (
        GUARD_REGISTRY["GI_OBOOK"].is_bypassed(bypassed)
        and GUARD_REGISTRY["GI_WHALE"].is_bypassed(bypassed)
        and GUARD_REGISTRY["GI_FUNDING"].is_bypassed(bypassed)
    )
    if not skip_external:
        from external_order_guard import check_external_order_guards
        ext_ok, ext_reason, ext_scale = check_external_order_guards(
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            notional_eur=computed_notional,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not ext_ok:
            return False, ext_reason, 0.0
        if ext_scale < 1.0 and ext_scale <= effective_scale:
            effective_scale = ext_scale
            active_reason = ext_reason

    # 3. LLM Guards (GLLM1 High-Stake + GLLM2 Sentiment + GLLM3 Geopolitical)
    skip_intel = (
        GUARD_REGISTRY["GLLM1"].is_bypassed(bypassed)
        and GUARD_REGISTRY["GLLM2"].is_bypassed(bypassed)
        and GUARD_REGISTRY["GLLM3"].is_bypassed(bypassed)
    )
    if not skip_intel:
        from llm_order_guard import check_llm_order_guards
        intel_ok, intel_reason, intel_scale = check_llm_order_guards(
            provider=provider,
            symbol=symbol,
            order_type=side,
            price=price,
            qty=qty,
            notional_eur=computed_notional,
            regime_context=regime_context,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not intel_ok:
            return False, intel_reason, 0.0
        if intel_scale < 1.0 and intel_scale <= effective_scale:
            effective_scale = intel_scale
            active_reason = intel_reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


def check_math_profit_reference(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    profit_percentage: float,
    window_ref=None,
    *,
    regime_context=None,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
) -> bool:
    """Evaluate deterministic price reference and margin percentage thresholds symmetrically
    for BUY and SELL across all difficulty tiers ('off', 'hybrid', 'last', 'average', 'dynamic', 'on').

    Pure mathematical calculation (0 tokens, deterministic, no AI or macro push alerts).
    """
    side = _normalize_side_token(order_type)
    opp_side = opposite_side(order_type)
    provider_name = _provider_name(provider)
    mode = get_profit_guard_mode_cfg(
        provider_name, side, module=module, reference_mode=reference_mode
    )

    if mode == "off":
        print(f"[GUARD] {side} {symbol}: the historical {opp_side.lower()} reference is off "
              f"for this venue/module (order_guard.conf); price {price} is not compared with past {opp_side.lower()}s")
        return True

    def _parse_ref(val):
        if val is None or isinstance(val, bool):
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) and f > 0 else None
        except (TypeError, ValueError):
            return None

    px_val = _parse_ref(price)
    if px_val is None or profit_percentage is None or isinstance(profit_percentage, bool):
        return False
    try:
        prof_pct = float(profit_percentage)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(prof_pct):
        return False

    if mode == "dynamic":
        trend = None
        if regime_context is not None:
            max_age = float(_load_margins()["regime_context_max_age_sec"])
            if not hasattr(regime_context, "is_valid_for") or regime_context.is_valid_for(
                symbol=symbol, provider=provider_name, max_age_seconds=max_age,
                require_identity=True,
            ):
                trend = getattr(regime_context, "resolved_trend", None)
        if trend is None:
            trend = _symbol_trend(symbol, provider=provider)
        dyn_window_s = get_dynamic_window_sec_cfg(
            symbol,
            provider=provider,
            order_type=side,
            resolved_trend=trend,
            regime_context=regime_context,
        )
        dyn_hours = dyn_window_s / 3600.0
        pos_window_ref = _parse_ref(window_ref)
        if pos_window_ref is not None:
            diff = profit_diff_pct(side, px_val, pos_window_ref)
            if diff < prof_pct:
                if hasattr(provider, "get_orders"):
                    try:
                        recent = provider.get_orders(symbol, opp_side, dyn_window_s)
                    except (ProviderError, Exception) as exc:
                        print(f"[GUARD] {side} {symbol}: dynamic window history unavailable ({exc}) -> BLOCKED")
                        return False
                    if recent is None:
                        print(f"[GUARD] {side} {symbol}: dynamic window history unavailable -> BLOCKED")
                        return False
                    recent_prices = []
                    for o in recent:
                        if not isinstance(o, dict):
                            continue
                        px = extract_order_price(o, default=0.0)
                        if px > 0 and math.isfinite(px):
                            recent_prices.append(px)
                    if not recent_prices:
                        print(f"[GUARD] {side} {symbol}: dynamic window ({dyn_hours:.1f}h, trend='{trend}') has no fills; "
                              f"anchor {pos_window_ref} from older period bypassed")
                        return True
                    pos_window_ref = select_extreme_reference(recent_prices, side, strict=True)
                    if len(recent) > 1 and all(
                        isinstance(o, dict) and (extract_timestamp_ms(o, default=0) or 0) > 0
                        for o in recent
                    ):
                        try:
                            same_orders = provider.get_orders(symbol, side, dyn_window_s) or []
                        except Exception:
                            same_orders = []
                        if same_orders:
                            active_ref = active_cycle_window_ref(recent, same_orders, side, reference_mode="strict")
                            if active_ref is not None and active_ref > 0:
                                pos_window_ref = active_ref
                    diff = profit_diff_pct(side, px_val, pos_window_ref)
                    if diff >= prof_pct:
                        print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                              f"recent {opp_side.lower()} ref {pos_window_ref}, price {px_val}, diff {diff:.2f}% >= {prof_pct}%. Permitted.")
                        return True
                elif _is_favorable_trend(side, trend):
                    if not hasattr(provider, "get_orders"):
                        print(f"[GUARD] {side} {symbol}: dynamic mode bypassed historical {opp_side.lower()} reference "
                              f"during confirmed {trend.upper()} trend; price {px_val} permitted")
                        return True
                print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                      f"found recent {opp_side.lower()} ref {pos_window_ref}, price {px_val}, diff {diff:.2f}%, threshold {prof_pct}%")
                print(f"Percentage difference ({diff:.2f}%) below threshold {prof_pct}%. "
                      f"The {side} order is BLOCKED.")
                return False
            print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                  f"recent {opp_side.lower()} ref {pos_window_ref}, price {px_val}, diff {diff:.2f}% >= {prof_pct}%. Permitted.")
            return True
        else:
            print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                  f"has no recent {opp_side.lower()} reference; price {px_val} permitted")
            return True

    if (
        (reference_mode is not None or module is not None or mode in ("average", "last", "hybrid"))
        and window_ref is None
        and hasattr(provider, "get_orders")
    ):
        try:
            window_ref = window_reference(
                provider, symbol, side, get_safeback_sec_cfg(provider_name),
                reference_mode=mode, module=module,
            )
        except Exception:
            pass

    raw_ref = (
        window_ref
        if window_ref is not None
        else (provider.last_opposite_fill(symbol, side) if hasattr(provider, "last_opposite_fill") else None)
    )

    ref = _parse_ref(raw_ref)
    if ref is None:
        return True
    diff = profit_diff_pct(side, px_val, ref)
    src = "the window" if window_ref is not None else "the provider"
    print(f"[GUARD] {side} {symbol} (mode={mode}): ref {ref} ({src}), price {px_val}, "
          f"diff {diff:.2f}%, threshold {prof_pct}%")
    if diff < prof_pct:
        print(f"Percentage difference ({diff:.2f}%) below threshold {prof_pct}%. "
              f"The {side} order is BLOCKED.")
        return False
    return True


_check_math_profit_reference = check_math_profit_reference


def profit_guard(
    provider,
    symbol,
    order_type,
    price,
    profit_percentage,
    window_ref=None,
    *,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    evaluate_intelligence: bool = True,
    bypass_guards: Optional[set[str]] = None,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
):
    """Return whether the order is profitable relative to its reference.

    Execution Pipeline:
    - Emergency Stop-Loss exits or bypass_guards with 'all' bypass all guards.
    - Stage 1 (Mathematical Reference): Deterministic profit margin vs historical reference.
      Bypassed if bypass_guards contains 'math_reference' (or 'math_profit').
    - Stage 2 (Intelligence, Macro & Sentiment): Evaluates Parabolic, Weibull, Gemini LLM,
      and Geopolitical shock shield only on mathematically cleared orders when
      evaluate_intelligence=True.
    """
    from guard_hierarchy import normalize_bypassed_guards, GUARD_REGISTRY
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if "all" in bypassed or "*" in bypassed:
        return True

    # Stage 1: Deterministic Mathematical Reference Check (G_PROF)
    if not GUARD_REGISTRY["G_PROF"].is_bypassed(bypassed):
        cm_kwargs = dict(window_ref=window_ref, regime_context=regime_context)
        if reference_mode is not None:
            cm_kwargs["reference_mode"] = reference_mode
        if module is not None:
            cm_kwargs["module"] = module
        if not check_math_profit_reference(
            provider, symbol, order_type, price, profit_percentage,
            **cm_kwargs,
        ):
            return False

    has_valid_qty = _clean_pos_float(qty, 0.0) > 0.0
    has_valid_notional = _clean_pos_float(notional_eur, 0.0) > 0.0
    if not _parse_bool_flag(evaluate_intelligence, True) or (not has_valid_qty and not has_valid_notional):
        return True

    # Stage 2: Intelligence, Macro & Sentiment Guards
    intel_ok, intel_reason, suggested_scale = check_intelligence_guards(
        provider, symbol, order_type, price, regime_context=regime_context, qty=qty, notional_eur=notional_eur,
        bypass_guards=bypassed,
    )
    if not intel_ok:
        print(f"[GUARD] {order_type} {symbol}: blocked by intelligence guard ({intel_reason})")
        return False

    if regime_context is not None and suggested_scale < 1.0:
        try:
            side_norm = _normalize_side_token(order_type)
            scales_by_side = getattr(regime_context, "_suggested_scale_by_side", None)
            if not isinstance(scales_by_side, dict):
                scales_by_side = {}
            curr_side = scales_by_side.get(side_norm)
            if curr_side is None or suggested_scale < curr_side:
                scales_by_side[side_norm] = float(suggested_scale)
            object.__setattr__(regime_context, "_suggested_scale_by_side", scales_by_side)
            if side_norm == "BUY":
                curr = getattr(regime_context, "suggested_scale", None)
                if curr is None or suggested_scale < curr:
                    object.__setattr__(regime_context, "suggested_scale", float(suggested_scale))
                    object.__setattr__(regime_context, "intelligence_scale_reason", intel_reason)
        except Exception:
            pass

    return True
