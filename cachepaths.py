"""Centralized Single Source of Truth (SSOT) for cachedb file path and symbol resolution."""
from __future__ import annotations

import builtins
from dataclasses import dataclass
from enum import Enum
import glob
import math
import numbers
import os
import shutil
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))

CACHE_CURRENT_PRICE_FILENAME = "cache_current_price.json"
CACHE_CURRENT_PRICE_TICKS_SOCKET_FILENAME = "current_price_ticks.sock"
CACHE_WATCHER_ALL_PRICE_FILENAME = "cache_watcher_all_price.json"
CMC_LISTINGS_CACHE_FILENAME = "cmc_listings_cache.json"
CACHE_SHORT_TREND_FILENAME = "cache_short_trend.json"
CACHE_LONG_TREND_FILENAME = "cache_long_trend.json"
CACHE_KALMAN_TREND_FILENAME = "cache_kalman_trend.json"
CACHE_T_TREND_FILENAME = "cache_T_trend.json"
PRICE_ANALYSIS_FILENAME = "priceanalysis.json"
CACHE_BINANCE_ORDER_FILLED_FILENAME = "cache_order_filled_binance.json"
CACHE_KRAKEN_ORDER_FILLED_FILENAME = "cache_order_filled_kraken.json"
CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME = "cache_order_filled_hyperliquid.json"
CACHE_T212_ORDER_FILLED_FILENAME = "cache_order_filled_t212.json"
CACHE_BINANCE_ORDER_OPEN_FILENAME = "cache_order_open_binance.json"
CACHE_KRAKEN_ORDER_OPEN_FILENAME = "cache_order_open_kraken.json"
CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME = "cache_order_open_hyperliquid.json"
CACHE_T212_ORDER_OPEN_FILENAME = "cache_order_open_t212.json"
CACHE_BINANCE_BALANCE_FILENAME = "cache_balance_binance.json"
CACHE_KRAKEN_BALANCE_FILENAME = "cache_balance_kraken.json"
CACHE_HYPERLIQUID_BALANCE_FILENAME = "cache_balance_hyperliquid.json"
CACHE_T212_BALANCE_FILENAME = "cache_balance_t212.json"
CACHE_BINANCE_TRADE_FILENAME = "cache_trade_binance.jsonl"
CACHE_ACCOUNT_HEALTH_FILENAME = "cache_account_health.json"
CACHE_FEE_SCHEDULE_FILENAME = "cache_fee_schedule.json"
CACHE_ASSET_VALUE_FILENAME = "cache_asset_value.jsonl"
CACHE_KRAKEN_TRADE_FILENAME = "cache_trade_kraken.json"
CACHE_HYPERLIQUID_TRADE_FILENAME = "cache_trade_hyperliquid.json"
CACHE_T212_TRADE_FILENAME = "cache_trade_t212.json"
ORDER_RETRY_QUEUE_FILENAME = "order_retry_queue.jsonl"
ORDER_RETRY_QUARANTINE_ALERTS_FILENAME = "order_retry_quarantine_alerts.json"
ORDER_LEDGER_ACTIVE_FILENAME = "orders_active.json"
ORDER_LEDGER_EVENTS_PREFIX = "order_events_"
ORDER_LEDGER_CLOSED_PREFIX = "orders_closed_"
ASSETGUARDIAN_STATE_FILENAME = "assetguardian_state.json"
AUTO_ORDER_INTENTS_STATE_FILENAME = "auto_order_intents_state.json"
AUTONOMOUS_ORDER_INTENTS_STATE_FILENAME = AUTO_ORDER_INTENTS_STATE_FILENAME
AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME = AUTO_ORDER_INTENTS_STATE_FILENAME
MT_STATE_FILENAME = "mt_state.json"
RTRADE_PAIRS_STATE_FILENAME = "rtrade_pairs_state.json"
SMART_STOP_LOSS_STATE_FILENAME = "smart_stop_loss_state.json"
TRAILING_STATE_FILENAME = "trailing_state.json"
NAV_HISTORY_STATE_FILENAME = "nav_history_state.json"
PORTFOLIO_POSITIONS_STATE_FILENAME = "portfolio_positions_state.json"
VENUE_BALANCES_STATE_FILENAME = "venue_balances_state.json"
VENUE_FEE_MAINTENANCE_STATE_FILENAME = "venue_fee_maintenance_state.json"
KINEMATIC_IMPULSE_SIGNAL_FILENAME = "kinematic_impulse_signal.json"
AI_ATTRIBUTION_STATUS_FILENAME = "ai_attribution_status.json"
AI_ATTRIBUTION_30D_FILENAME = "ai_attribution_30d.json"
LLM_QUOTA_STATUS_FILENAME = "llm_quota_status.json"
LLM_MODEL_CATALOG_FILENAME = "llm_model_catalog.json"
LLM_TOPIC_THREADS_FILENAME = "llm_topic_threads.json"
NOTIFY_QUEUE_FILENAME = "notify_queue.json"
NOTIFICATION_DELIVERY_STATE_FILENAME = "notification_delivery_state.json"
OS_RESOURCES_WATCHDOG_STATE_FILENAME = "os_resources_watchdog_state.json"

CACHE_ORDERBOOK_DEPTH_PREFIX = "orderbook_depth_"
CACHE_DERIVATIVES_TELEMETRY_PREFIX = "derivatives_telemetry_"
CACHE_WHALE_SNAPSHOT_PREFIX = "whale_snapshot_"
MICROSTRUCTURE_CYCLE_HISTORY_FILENAME = "microstructure_cycle_history.json"
CACHE_REALIZED_LIQUIDATIONS_FILENAME = "realized_liquidations.json"
CACHE_FEAR_GREED_FILENAME = "fear_greed_collect.json"
CACHE_MARKET_BREADTH_FILENAME = "market_breadth_collect.json"
CACHE_MACRO_BENCHMARKS_FILENAME = "macro_benchmarks_collect.json"
CACHE_MACRO_CALENDAR_FILENAME = "macro_calendar_collect.json"
CACHE_ETF_FLOWS_FILENAME = "etf_flows_collect.json"
CACHE_LONG_HORIZON_FILENAME = "long_horizon_collect.json"
OPERATOR_REGIME_NOTE_FILENAME = "operator_regime_note.txt"
CACHE_MACRO_LEXICON_FILENAME = "macro_lexicon_collect.json"
CACHE_NEWS_FEED_FILENAME = "news_feed_collect.json"
CACHE_SENTIMENT_EVAL_FILENAME = "sentiment_eval.json"
CACHE_GEOPOLITICAL_EVAL_FILENAME = "geopolitical_eval.json"

CACHE_SPARSE_PRICE_PREFIX = "cache_sparse_price_"
CACHE_SPARSE_PRICE_EXT = ".jsonl"
CACHE_24_PRICE_PREFIX = "cache_24_price_"
CACHE_24_PRICE_EXT = ".jsonl"
CACHE_365_PRICE_PREFIX = "cache_365_price_"
CACHE_365_PRICE_EXT = ".jsonl"
CACHE_OHLC_24H_PREFIX = "cache_ohlc_24h_"
CACHE_OHLC_24H_EXT = ".json"
CACHE_OHLC_BARS_PREFIX = "cache_ohlc_bars_"
CACHE_OHLC_BARS_EXT = ".json"
CYCLE_STATE_PREFIX = ".state_"
CYCLE_STATE_EXT = ".json"


INVALID_SCALAR_STRINGS = frozenset((
    "nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity",
    "none", "null",
))


def clean_non_bool_str(val: object) -> str:
    """Return ``str(val).strip()`` unless ``val`` is ``None`` or a ``bool``, in which case return ``''``."""
    return "" if (val is None or isinstance(val, bool)) else str(val).strip()


def clean_nonempty_str(val: object) -> str | None:
    """Return ``val.strip()`` when ``val`` is a non-empty string, else ``None``."""
    return s if isinstance(val, str) and (s := val.strip()) else None


def clean_upper_token(val: object) -> str | None:
    """Return ``val.strip().upper()`` when ``val`` is a non-empty string, else ``None``."""
    return s.upper() if (s := clean_nonempty_str(val)) is not None else None


def clean_scalar_str(val: object) -> str | None:
    """Return ``val.strip()`` when ``val`` is a non-empty string not in ``INVALID_SCALAR_STRINGS``, else ``None``."""
    return s if (s := clean_nonempty_str(val)) is not None and s.lower() not in INVALID_SCALAR_STRINGS else None


def clean_upper_scalar(val: object) -> str | None:
    """Return ``val.strip().upper()`` when ``val`` is a valid scalar string, else ``None``."""
    return s.upper() if (s := clean_scalar_str(val)) is not None else None


_SYMBOL_CASE_RESOLVER = globals().get("_SYMBOL_CASE_RESOLVER")


def bind_symbol_case_resolver(resolver) -> None:
    """Bind a higher-layer symbol-case canonicalizer (e.g. ``instrument_registry.resolve_registered_symbol_case``)."""
    global _SYMBOL_CASE_RESOLVER
    _SYMBOL_CASE_RESOLVER = resolver if callable(resolver) else None


def _resolve_canonical_cache_symbol(raw_sym: str) -> str:
    resolver = _SYMBOL_CASE_RESOLVER
    if callable(resolver):
        try:
            resolved = resolver(raw_sym)
            if isinstance(resolved, str) and resolved:
                return resolved
        except Exception:
            pass
    return raw_sym


def _clean_cache_symbol(symbol: object, *, preserve_case: bool = False) -> str:
    if symbol is None or isinstance(symbol, (bool, bytes, bytearray, dict, list, tuple, set, frozenset)):
        return ""
    raw = symbol if isinstance(symbol, str) else str(symbol)
    s = clean_scalar_str(raw)
    if s is None:
        return ""
    if not preserve_case:
        s = _resolve_canonical_cache_symbol(s)
    return s if (preserve_case or is_kraken_tokenized_asset(s)) else s.upper()


def cache_sparse_price_filename(symbol: object, *, preserve_case: bool = False) -> str:
    """Return canonical on-disk filename for a sparse long-horizon price archive (`cache_sparse_price_<SYM>.jsonl`)."""
    return f"{CACHE_SPARSE_PRICE_PREFIX}{_clean_cache_symbol(symbol, preserve_case=preserve_case)}{CACHE_SPARSE_PRICE_EXT}"


def cache_24_price_filename(symbol: object, *, ext: str = CACHE_24_PRICE_EXT, preserve_case: bool = False) -> str:
    """Return canonical on-disk filename for a 24h dense price series (`cache_24_price_<SYM>.jsonl`)."""
    eff_ext = clean_scalar_str(ext) or CACHE_24_PRICE_EXT
    return f"{CACHE_24_PRICE_PREFIX}{_clean_cache_symbol(symbol, preserve_case=preserve_case)}{eff_ext}"


def cache_365_price_filename(symbol: object, *, ext: str = CACHE_365_PRICE_EXT, preserve_case: bool = False) -> str:
    """Return canonical on-disk filename for a 365d dense price archive (`cache_365_price_<SYM>.jsonl`)."""
    eff_ext = clean_scalar_str(ext) or CACHE_365_PRICE_EXT
    return f"{CACHE_365_PRICE_PREFIX}{_clean_cache_symbol(symbol, preserve_case=preserve_case)}{eff_ext}"


def cache_ohlc_24h_filename(symbol: object, *, preserve_case: bool = False) -> str:
    """Return canonical on-disk filename for a 24h 5m OHLCV series (`cache_ohlc_24h_<SYM>.json`)."""
    return f"{CACHE_OHLC_24H_PREFIX}{_clean_cache_symbol(symbol, preserve_case=preserve_case)}{CACHE_OHLC_24H_EXT}"


def cache_ohlc_bars_filename(
    symbol: object,
    timeframe: str = "1d",
    *,
    preserve_case: bool = False,
) -> str:
    """Return canonical on-disk filename for a persistent OHLCV bar archive (`cache_ohlc_bars_<TF>_<SYM>.json`)."""
    tf_raw = clean_scalar_str(timeframe) or "1d"
    tf_norm = tf_raw if tf_raw == "1M" else tf_raw.lower()
    return f"{CACHE_OHLC_BARS_PREFIX}{tf_norm}_{_clean_cache_symbol(symbol, preserve_case=preserve_case)}{CACHE_OHLC_BARS_EXT}"


def price_cache_filename_variants(symbol: object, builder_fn, *extra_symbols: object) -> tuple[str, ...]:
    """Return deduplicated canonical and legacy uppercase cache filenames for ``symbol`` and ``extra_symbols``."""
    out: list[str] = []
    for sym in (symbol, *extra_symbols):
        s_clean = _clean_cache_symbol(sym, preserve_case=True)
        if not s_clean:
            continue
        for cand in (s_clean, s_clean.upper()):
            for preserve in (True, False):
                fn = builder_fn(cand, preserve_case=preserve)
                if fn and fn not in out:
                    out.append(fn)
    return tuple(out)


EXACT_LEGACY_FILENAMES = {
    CACHE_CURRENT_PRICE_FILENAME: ("cache_currentprice.json",),
    CACHE_SHORT_TREND_FILENAME: ("cache_instant_trend.json",),
    CACHE_LONG_TREND_FILENAME: ("cache_price_long_trend.json",),
    CACHE_BINANCE_ORDER_FILLED_FILENAME: ("cache_order_binance.json", "cache_order.json"),
    CACHE_BINANCE_TRADE_FILENAME: ("cache_trade.jsonl",),
    CACHE_ACCOUNT_HEALTH_FILENAME: ("binance_account_cache_health.json", "account_cache_health.json"),
    RTRADE_PAIRS_STATE_FILENAME: ("rtrade_pairs.json",),
    AUTO_ORDER_INTENTS_STATE_FILENAME: (
        "autonomous_order_intents_state.json",
        "autonomous_trade_intents_state.json",
        "autonomous_trade_intents.json",
    ),
}
PREFIX_LEGACY_FILENAMES = (
    (CACHE_SPARSE_PRICE_PREFIX, ("cache_price_sparse_", "cache_price_")),
    (CACHE_24_PRICE_PREFIX, ("cache_24price_", "price_24h_", "24price_")),
    (CACHE_365_PRICE_PREFIX, ("cache_365price_", "cache_24price_long_")),
)

# Regenerable caches (can be deleted and re-synced from exchanges / external feeds).
CACHEDB_SUBDOMAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "trends": {
        "exact": (
            "cache_short_trend.json", "cache_instant_trend.json",
            "cache_long_trend.json", "cache_price_long_trend.json",
            "cache_T_trend.json", "cache_kalman_trend.json",
            "priceanalysis.json", "24trend",
        ),
        "prefixes": (
            "24trend/", "priceanalysis.", "cache_short_trend.", "cache_instant_trend.",
            "cache_long_trend.", "cache_price_long_trend.", "cache_T_trend.", "cache_kalman_trend.",
        ),
        "suffixes": (),
    },
    "prices": {
        "exact": (
            "cache_current_price.json",
            "cache_currentprice.json",
            "cache_watcher_all_price.json",
            "cmc_listings_cache.json",
            "current_price_ticks.sock",
        ),
        "prefixes": (
            "cache_current_price.", "cache_currentprice.", "cache_watcher_all_price.", "cmc_listings_cache.",
            "cache_24_price_", "cache_24price_", "price_24h_",
            "cache_365_price_", "cache_365price_",
            "cache_sparse_price_", "cache_price_", "cache_ohlc_",
        ),
        "suffixes": (".npz",),
    },
    "telemetry": {
        "exact": (
            MICROSTRUCTURE_CYCLE_HISTORY_FILENAME,
            CACHE_REALIZED_LIQUIDATIONS_FILENAME,
            KINEMATIC_IMPULSE_SIGNAL_FILENAME,
        ),
        "prefixes": (
            "orderbook_depth_",
            "derivatives_telemetry_",
            "whale_snapshot_",
            "microstructure_cycle_history.",
            "realized_liquidations.",
            "kinematic_impulse_signal.",
        ),
        "suffixes": (),
    },
    "macro": {
        "exact": (
            CACHE_MACRO_LEXICON_FILENAME,
            CACHE_MACRO_BENCHMARKS_FILENAME,
            CACHE_MACRO_CALENDAR_FILENAME,
            CACHE_ETF_FLOWS_FILENAME,
            CACHE_LONG_HORIZON_FILENAME,
            CACHE_SENTIMENT_EVAL_FILENAME,
            CACHE_GEOPOLITICAL_EVAL_FILENAME,
            CACHE_FEAR_GREED_FILENAME,
            CACHE_MARKET_BREADTH_FILENAME,
            CACHE_NEWS_FEED_FILENAME,
        ),
        "prefixes": (
            "fear_greed_", "market_breadth_", "macro_benchmarks_", "macro_calendar_",
            "etf_flows_", "long_horizon_", "news_feed_", "geopolitical_", "macro_lexicon_", "sentiment_eval.",
        ),
        "suffixes": (),
    },
    "account": {
        "exact": (
            CACHE_ACCOUNT_HEALTH_FILENAME,
            "account_cache_health.json",
            "binance_account_cache_health.json",
            CACHE_FEE_SCHEDULE_FILENAME,
        ),
        "prefixes": (
            "cache_order", "cache_balance", "cache_trade", "cache_asset_value",
            "cache_fee_schedule", "cache_t212_client_",
            "cache_account_health.", "account_cache_health.", "binance_account_cache_health.",
        ),
        "suffixes": (),
    },
    "llm": {
        "exact": (
            AI_ATTRIBUTION_STATUS_FILENAME,
            AI_ATTRIBUTION_30D_FILENAME,
            LLM_QUOTA_STATUS_FILENAME,
            LLM_MODEL_CATALOG_FILENAME,
        ),
        "prefixes": (
            "auto_", "ai_attribution_", "llm_quota_status.", "llm_model_catalog.",
        ),
        "suffixes": ("_eval.json",),
    },
    "system/heartbeat": {
        "exact": (),
        "prefixes": (),
        "suffixes": (".heartbeat",),
    },
    "system": {
        "exact": ("fleet_health_status.json", "dev_regression_status.json"),
        "prefixes": ("fleet_health_status.", "dev_regression_status."),
        "suffixes": (),
    },
}

# Irreplaceable local mptrade state (backup-mandatory, fsync=True, 1:1 state_stores/ owners).
STATEDB_SUBDOMAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "orders": {
        "exact": (
            ORDER_LEDGER_ACTIVE_FILENAME,
            ORDER_RETRY_QUEUE_FILENAME,
            ORDER_RETRY_QUARANTINE_ALERTS_FILENAME,
            AUTO_ORDER_INTENTS_STATE_FILENAME,
            "autonomous_order_intents_state.json",
            "autonomous_trade_intents_state.json",
            "autonomous_trade_intents.json",
        ),
        "prefixes": (
            "order_retry_",
            "orders_active.",
            ORDER_LEDGER_EVENTS_PREFIX,
            ORDER_LEDGER_CLOSED_PREFIX,
            "auto_order_intents",
            "autonomous_order_intents",
            "autonomous_trade_intents",
        ),
        "suffixes": (),
    },
    "portfolio": {
        "exact": (
            NAV_HISTORY_STATE_FILENAME,
            PORTFOLIO_POSITIONS_STATE_FILENAME,
            VENUE_BALANCES_STATE_FILENAME,
        ),
        "prefixes": (
            "nav_history_state",
            "portfolio_positions_state",
            "venue_balances_state",
        ),
        "suffixes": (),
    },
    "bots": {
        "exact": (
            ASSETGUARDIAN_STATE_FILENAME,
            MT_STATE_FILENAME,
            TRAILING_STATE_FILENAME,
            SMART_STOP_LOSS_STATE_FILENAME,
            RTRADE_PAIRS_STATE_FILENAME,
            VENUE_FEE_MAINTENANCE_STATE_FILENAME,
        ),
        "prefixes": (
            "assetguardian_state",
            "rtrade_pairs",
            "mt_state.",
            "trailing_state.",
            "trailing_state_",
            "smart_stop_loss_state.",
            "venue_fee_maintenance_state",
        ),
        "suffixes": ("_state.json",),
    },
    "llm": {
        "exact": (LLM_TOPIC_THREADS_FILENAME,),
        "prefixes": ("llm_topic_threads.",),
        "suffixes": (),
    },
    "system": {
        "exact": (
            NOTIFY_QUEUE_FILENAME,
            NOTIFICATION_DELIVERY_STATE_FILENAME,
            OS_RESOURCES_WATCHDOG_STATE_FILENAME,
        ),
        "prefixes": (
            "notify_queue.",
            "notification_delivery_state.",
            "os_resources_watchdog_state.",
        ),
        "suffixes": (),
    },
}

_LEGACY_STORAGE_SUBDOMAINS: frozenset[str] = frozenset({"state"})
_ALL_STORAGE_SUBDOMAINS: frozenset[str] = (
    frozenset(CACHEDB_SUBDOMAINS) | frozenset(STATEDB_SUBDOMAINS) | _LEGACY_STORAGE_SUBDOMAINS
)
_SIDECAR_SUFFIXES: tuple[str, ...] = (".generation.lock", ".previous", ".meta", ".lock", ".tmp")
_SORTED_SUBDOMAINS: tuple[str, ...] = tuple(sorted(_ALL_STORAGE_SUBDOMAINS, key=len, reverse=True))
_TOP_LEVEL_SUBDOMAINS: frozenset[str] = frozenset(sub.split("/", 1)[0] for sub in _ALL_STORAGE_SUBDOMAINS)
STALE_STAGING_MAX_AGE_SEC: float = 300.0


class DurabilityTier(str, Enum):
    """Declarative SSOT durability classification for cachedb files."""

    FINANCIAL_CRITICAL = "financial_critical"
    ARCHIVE_DURABLE = "archive_durable"
    EPHEMERAL_TELEMETRY = "ephemeral_telemetry"


@dataclass(frozen=True)
class DurabilityPolicy:
    """Resolved write durability & formatting policy for a single cachedb path."""

    tier: DurabilityTier
    fsync: bool
    keep_previous: bool
    compact_json: bool
    separators: tuple[str, str] | None = None


DURABILITY_POLICY_FINANCIAL_CRITICAL = DurabilityPolicy(
    tier=DurabilityTier.FINANCIAL_CRITICAL,
    fsync=True,
    keep_previous=True,
    compact_json=False,
    separators=None,
)
DURABILITY_POLICY_ARCHIVE_DURABLE = DurabilityPolicy(
    tier=DurabilityTier.ARCHIVE_DURABLE,
    fsync=True,
    keep_previous=True,
    compact_json=True,
    separators=(",", ":"),
)
DURABILITY_POLICY_EPHEMERAL_TELEMETRY = DurabilityPolicy(
    tier=DurabilityTier.EPHEMERAL_TELEMETRY,
    fsync=False,
    keep_previous=False,
    compact_json=True,
    separators=(",", ":"),
)

_ARCHIVE_DURABLE_PREFIXES: tuple[str, ...] = (
    CACHE_SPARSE_PRICE_PREFIX,
    CACHE_365_PRICE_PREFIX,
    CACHE_OHLC_BARS_PREFIX,
    "cache_price_sparse_",
    "cache_price_",
    "cache_365price_",
    "cache_24price_long_",
)
_EPHEMERAL_SUBDOMAINS: frozenset[str] = frozenset({"system/heartbeat", "telemetry", "trends"})
_EPHEMERAL_MACRO_PREFIXES: tuple[str, ...] = (
    "fear_greed_",
    "market_breadth_",
    "macro_benchmarks_",
    "macro_calendar_",
    "etf_flows_",
    "long_horizon_",
    "news_feed_",
)


def _strip_sidecar_basename(path: object) -> str:
    raw = _clean_path_str(path)
    if not raw:
        return ""
    base = os.path.basename(raw.replace("\\", "/").rstrip("/"))
    stripped = base[1:] if base.startswith(".") and not base.startswith("..") else base
    while True:
        matched = next(
            (sfx for sfx in _SIDECAR_SUFFIXES if stripped.endswith(sfx) and len(stripped) > len(sfx)),
            None,
        )
        if matched:
            stripped = stripped[: -len(matched)]
        elif ".previous." in stripped:
            stripped = stripped.split(".previous.", 1)[0]
        else:
            break
    return stripped


def is_financial_critical_path(path: object) -> bool:
    """Return True when ``path`` resolves to ``statedb/`` or ``cachedb/account/`` requiring durable ``fsync=True``."""
    dom = resolve_storage_domain(path)
    if dom is None:
        return False
    root_kind, sub = dom
    return root_kind == "statedb" or sub == "account"


def resolve_durability_policy(path: object) -> DurabilityPolicy:
    """Resolve the declarative ``DurabilityPolicy`` for ``path`` (fail-safe ``FINANCIAL_CRITICAL`` default)."""
    dom = resolve_storage_domain(path)
    if dom is None:
        return DURABILITY_POLICY_FINANCIAL_CRITICAL
    root_kind, sub = dom
    if root_kind == "statedb" or sub == "account":
        return DURABILITY_POLICY_FINANCIAL_CRITICAL
    if sub in _EPHEMERAL_SUBDOMAINS:
        return DURABILITY_POLICY_EPHEMERAL_TELEMETRY
    if sub == "prices":
        base = _strip_sidecar_basename(path)
        if base.startswith(_ARCHIVE_DURABLE_PREFIXES):
            return DURABILITY_POLICY_ARCHIVE_DURABLE
        return DURABILITY_POLICY_EPHEMERAL_TELEMETRY
    if sub == "macro":
        base = _strip_sidecar_basename(path)
        if base.startswith(_EPHEMERAL_MACRO_PREFIXES):
            return DURABILITY_POLICY_EPHEMERAL_TELEMETRY
        return DURABILITY_POLICY_FINANCIAL_CRITICAL
    return DURABILITY_POLICY_FINANCIAL_CRITICAL


def _clean_path_str(p: object) -> str:
    """Return stripped string representation of path, rejecting booleans, None, and non-path types."""
    if p is None or isinstance(p, bool) or not isinstance(p, (str, os.PathLike)):
        return ""
    return str(os.fspath(p)).strip()


def _norm_rel_cache_path(raw: str, *, strip_cachedb_prefix: bool = False) -> str:
    s = raw.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    if strip_cachedb_prefix and s.startswith("cachedb/"):
        s = s[len("cachedb/"):]
    return s


def cache_relpath(filename: object) -> str:
    """Return a workspace-relative ``cachedb/<filename>`` path string."""
    norm = _norm_rel_cache_path(_clean_path_str(filename), strip_cachedb_prefix=True)
    return f"cachedb/{norm}" if norm else ""


def is_cachedb_relpath(path: object) -> bool:
    """Return True if ``path`` is a workspace-relative ``cachedb/...`` path."""
    norm = _norm_rel_cache_path(_clean_path_str(path), strip_cachedb_prefix=False)
    return norm.startswith("cachedb/") and len(norm) > len("cachedb/")


def _exists_with_prev(path: str) -> bool:
    return bool(path and (os.path.exists(path) or os.path.exists(f"{path}.previous")))


def _move_sidecars(src: str, dst: str, *, include_archives: bool = False) -> None:
    for sfx in (".previous", ".meta", ".generation.lock"):
        if os.path.exists(src + sfx) and not os.path.exists(dst + sfx):
            os.replace(src + sfx, dst + sfx)
    if include_archives:
        for old_arch in glob.glob(glob.escape(src) + ".*.archive*"):
            new_arch = dst + old_arch[len(src):]
            if not os.path.exists(new_arch):
                os.replace(old_arch, new_arch)


def _strip_all_sidecars_from_name(raw_name: str) -> tuple[str, str]:
    """Split ``raw_name`` into ``(base_filename, trailing_sidecar_suffix)``."""
    stripped = raw_name[1:] if raw_name.startswith(".") and not raw_name.startswith("..") else raw_name
    while True:
        matched = next((sfx for sfx in _SIDECAR_SUFFIXES if stripped.endswith(sfx) and len(stripped) > len(sfx)), None)
        if matched:
            stripped = stripped[: -len(matched)]
        elif ".previous." in stripped:
            stripped = stripped.split(".previous.", 1)[0]
        else:
            break
    prefix_dot = "." if (raw_name.startswith(".") and not raw_name.startswith("..")) else ""
    sidecar = raw_name[len(prefix_dot) + len(stripped):]
    return stripped, sidecar


def _legacy_basenames_for_item(rel_item: str) -> tuple[str, ...]:
    """Return legacy exact filename variants for ``rel_item``, preserving any sidecar suffix."""
    if "/" in rel_item or "\\" in rel_item:
        return ()
    stripped, sidecar = _strip_all_sidecars_from_name(rel_item)
    legacy_bases = EXACT_LEGACY_FILENAMES.get(stripped)
    if not legacy_bases:
        return ()
    prefix_dot = "." if (rel_item.startswith(".") and not rel_item.startswith("..")) else ""
    return tuple(f"{prefix_dot}{lb}{sidecar}" for lb in legacy_bases)


def _legacy_cachedb_subdomains_for(root_kind: str, sub: str, stripped_name: str) -> tuple[str, ...]:
    """Return ordered pre-Variant-A ``cachedb/<sub_dir>/`` fallback subdomains for dual-layout resolution."""
    if root_kind == "cachedb":
        if sub == "account":
            return ("orders",)
        if sub == "llm":
            return ("system",)
        if sub == "telemetry" and stripped_name.startswith("kinematic_impulse_signal"):
            return ("system",)
        if sub == "system/heartbeat":
            return ("system",)
        return ()
    if sub == "orders":
        if stripped_name.startswith(("auto_order_intents", "autonomous_order_intents", "autonomous_trade_intents")):
            return ("orders", "state")
        return ("orders",)
    if sub in ("portfolio", "bots"):
        return (sub, "state")
    if sub == "llm":
        return ("llm", "system")
    if sub == "system":
        return ("system",)
    return (sub,)


def resolve_storage_domain(name: object) -> tuple[str, str] | None:
    """Classify ``name`` into ``("statedb" | "cachedb", subdomain)`` under Variant A taxonomy."""
    raw = _norm_rel_cache_path(_clean_path_str(name), strip_cachedb_prefix=True)
    if raw.startswith("statedb/"):
        raw = raw[len("statedb/"):]
    if not raw or raw in _ALL_STORAGE_SUBDOMAINS:
        return None
    explicit_sub: str | None = None
    if "/" in raw:
        for sub in _SORTED_SUBDOMAINS:
            if raw.startswith(f"{sub}/") and raw[len(sub) + 1:]:
                explicit_sub = sub
                raw = os.path.basename(raw)
                break
        else:
            if raw.startswith("24trend/"):
                return ("cachedb", "trends")
            raw = os.path.basename(raw)
    stripped, _ = _strip_all_sidecars_from_name(raw)
    cands = (stripped, raw)
    # 0. Heartbeat files always route to cachedb/system/heartbeat regardless of daemon name prefix (e.g. auto_*)
    for cand in cands:
        if cand.endswith(".heartbeat"):
            return ("cachedb", "system/heartbeat")
    # 1. Exact filename matches (statedb first so durable files never misclassify as cache)
    for cand in cands:
        for sub, rules in STATEDB_SUBDOMAINS.items():
            if cand in rules["exact"]:
                return ("statedb", sub)
    for cand in cands:
        for sub, rules in CACHEDB_SUBDOMAINS.items():
            if cand in rules["exact"]:
                return ("cachedb", sub)
    # 2. Prefix matches (statedb first so autonomous_*_intents matches orders before _state.json)
    for cand in cands:
        for sub, rules in STATEDB_SUBDOMAINS.items():
            if cand.startswith(rules["prefixes"]):
                return ("statedb", sub)
    for cand in cands:
        for sub, rules in CACHEDB_SUBDOMAINS.items():
            if cand.startswith(rules["prefixes"]):
                return ("cachedb", sub)
    # 3. Suffix matches
    for cand in cands:
        for sub, rules in STATEDB_SUBDOMAINS.items():
            if cand.endswith(rules["suffixes"]):
                return ("statedb", sub)
    for cand in cands:
        for sub, rules in CACHEDB_SUBDOMAINS.items():
            if cand.endswith(rules["suffixes"]):
                return ("cachedb", sub)
    if explicit_sub is not None:
        if explicit_sub in STATEDB_SUBDOMAINS and explicit_sub not in CACHEDB_SUBDOMAINS:
            return ("statedb", explicit_sub)
        return ("cachedb", explicit_sub)
    return None


def resolve_cache_subdomain(name: object) -> str | None:
    """Return the canonical thematic storage subdomain for ``name``, or ``None`` if unclassified."""
    dom = resolve_storage_domain(name)
    return dom[1] if dom is not None else None


def resolve_statedb_dir_from_cachedb(cache_dir: object) -> str:
    """Derive the sibling ``statedb`` root for ``cache_dir`` (preserving isolated temp test dirs)."""
    raw = _clean_path_str(cache_dir)
    if not raw:
        return get_statedb_dir()
    norm = os.path.normpath(os.path.abspath(raw.rstrip("/\\") or raw))
    base_lc = os.path.basename(norm).lower()
    if base_lc == "statedb":
        return norm
    if base_lc == "cachedb":
        return os.path.join(os.path.dirname(norm), "statedb")
    cand_nested = os.path.join(norm, "statedb")
    if os.path.isdir(cand_nested):
        return cand_nested
    return norm


def _resolve_storage_filepath(path: object, *, migrate: bool = False) -> object:
    """Dual-layout SSOT path resolver across ``cachedb/<sub_dir>/`` and ``statedb/<sub_dir>/``.

    Zero-split-brain guarantee:
      - Variant A target directories (``statedb/<sub_dir>/``, ``cachedb/account/``, ``cachedb/llm/``)
        activate automatically ONLY once created on disk (e.g. via ``organize_storage_domains``),
        preserving 100% compatibility with pre-cutover ``cachedb/state/`` & ``cachedb/orders/`` layouts
        and existing unit-test fixtures.
      - Passive by default (``migrate=False``); pass ``migrate=True`` from writers/migrations to move
        flat/legacy files and sidecars into their active subdomain directory.
    """
    raw = _clean_path_str(path)
    if not raw:
        return path
    parts = [p for p in raw.replace("\\", "/").split("/") if p and p != "."]
    if not parts:
        return raw

    # Determine whether ``raw`` already includes an explicit subdomain directory.
    explicit_sub: str | None = None
    if len(parts) >= 3 and f"{parts[-3]}/{parts[-2]}" in _ALL_STORAGE_SUBDOMAINS:
        explicit_sub = f"{parts[-3]}/{parts[-2]}"
        rel_item = parts[-1]
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(raw)))
    elif len(parts) >= 2 and parts[-2] in _ALL_STORAGE_SUBDOMAINS:
        explicit_sub = parts[-2]
        rel_item = parts[-1]
        base_dir = os.path.dirname(os.path.dirname(raw))
    elif len(parts) >= 3 and parts[-2] == "24trend" and parts[-3] in _ALL_STORAGE_SUBDOMAINS:
        return raw
    elif len(parts) >= 2 and parts[-2] == "24trend":
        rel_item = os.path.join("24trend", parts[-1])
        base_dir = os.path.dirname(os.path.dirname(raw))
    else:
        rel_item = parts[-1]
        base_dir = os.path.dirname(raw)

    base_dir = base_dir or "."
    dom = resolve_storage_domain(rel_item)
    if dom is None:
        return raw
    root_kind, sub = dom
    stripped, _ = _strip_all_sidecars_from_name(rel_item)
    legacy_subs = _legacy_cachedb_subdomains_for(root_kind, sub, stripped)

    base_leaf_lc = os.path.basename(os.path.normpath(base_dir)).lower()
    if root_kind == "statedb":
        if base_leaf_lc == "statedb":
            canonical_root = base_dir
            sibling_cachedb = os.path.join(os.path.dirname(base_dir), "cachedb")
            legacy_root = sibling_cachedb if os.path.isdir(sibling_cachedb) else base_dir
        elif base_leaf_lc == "cachedb":
            canonical_root = os.path.join(os.path.dirname(base_dir), "statedb")
            legacy_root = base_dir
        else:
            nested_state = os.path.join(base_dir, "statedb")
            canonical_root = nested_state if os.path.isdir(nested_state) else base_dir
            legacy_root = base_dir
    else:
        if base_leaf_lc == "statedb":
            sibling_cachedb = os.path.join(os.path.dirname(base_dir), "cachedb")
            canonical_root = sibling_cachedb if os.path.isdir(sibling_cachedb) else base_dir
            legacy_root = canonical_root
        else:
            canonical_root = base_dir
            legacy_root = base_dir

    # Preserve already-canonical explicit paths when no reclassification is needed.
    if explicit_sub is not None:
        canonical_sub_dir_chk = os.path.join(canonical_root, *sub.split("/"))
        if (
            explicit_sub == sub
            and (
                (root_kind == "cachedb" and base_leaf_lc != "statedb")
                or (
                    root_kind == "statedb"
                    and (
                        base_leaf_lc == "statedb"
                        or os.path.abspath(os.path.dirname(raw)) == os.path.abspath(canonical_sub_dir_chk)
                        or not os.path.isdir(canonical_sub_dir_chk)
                    )
                )
            )
        ):
            if _exists_with_prev(raw):
                return raw
            for leg_name in _legacy_basenames_for_item(rel_item):
                leg_same_dir = os.path.join(os.path.dirname(raw), leg_name)
                if _exists_with_prev(leg_same_dir):
                    return leg_same_dir
            return raw

    canonical_sub_dir = os.path.join(canonical_root, *sub.split("/"))
    canonical_sub_path = os.path.join(canonical_sub_dir, rel_item)
    canonical_active = os.path.isdir(canonical_sub_dir)

    legacy_sub_dirs = [
        os.path.join(legacy_root, *lsub.split("/"))
        for lsub in legacy_subs
        if os.path.abspath(os.path.join(legacy_root, *lsub.split("/"))) != os.path.abspath(canonical_sub_dir)
    ]
    legacy_sub_paths = [os.path.join(ldir, rel_item) for ldir in legacy_sub_dirs]
    flat_path = os.path.join(legacy_root, rel_item)

    if _parse_bool_flag(migrate, False):
        if canonical_active:
            for old_p in (*legacy_sub_paths, flat_path):
                if old_p and os.path.isfile(old_p) and not _exists_with_prev(canonical_sub_path):
                    try:
                        os.makedirs(os.path.dirname(canonical_sub_path), exist_ok=True)
                        os.replace(old_p, canonical_sub_path)
                        _move_sidecars(old_p, canonical_sub_path)
                    except OSError:
                        pass
        else:
            for ldir, lpath in zip(legacy_sub_dirs, legacy_sub_paths):
                if os.path.isdir(ldir) and os.path.isfile(flat_path) and not _exists_with_prev(lpath):
                    try:
                        os.makedirs(os.path.dirname(lpath), exist_ok=True)
                        os.replace(flat_path, lpath)
                        _move_sidecars(flat_path, lpath)
                    except OSError:
                        pass
                    break

    # 1. Exact rel_item check across canonical -> legacy subdomains -> flat root
    ordered_exact = (canonical_sub_path, *legacy_sub_paths, flat_path)
    for cand_p in dict.fromkeys(ordered_exact):
        if _exists_with_prev(cand_p):
            return cand_p

    # 2. Passive fallback to legacy pre-rename filenames (e.g. binance_account_cache_health.json,
    #    autonomous_trade_intents_state.json) when canonical filename does not yet exist on disk.
    for leg_name in _legacy_basenames_for_item(rel_item):
        for cand_dir in dict.fromkeys((canonical_sub_dir, *legacy_sub_dirs, legacy_root)):
            leg_p = os.path.join(cand_dir, leg_name)
            if _exists_with_prev(leg_p):
                return leg_p

    # 3. No file exists yet: choose active canonical subdomain, else active legacy subdomain, else raw.
    if canonical_active:
        return canonical_sub_path
    for ldir, lpath in zip(legacy_sub_dirs, legacy_sub_paths):
        if os.path.isdir(ldir):
            return lpath
    return raw


def resolve_cachedb_filepath(path: object, *, migrate: bool = False) -> object:
    """Resolve ``path`` to its active ``cachedb/<subdomain>/`` or ``statedb/<subdomain>/`` location."""
    return _resolve_storage_filepath(path, migrate=migrate)


def resolve_statedb_filepath(path: object, *, migrate: bool = False) -> object:
    """Resolve ``path`` to its active ``statedb/<subdomain>/`` location (with pre-cutover ``cachedb/`` fallback)."""
    return _resolve_storage_filepath(path, migrate=migrate)


def _clean_stale_staging_in_dir(
    target_dir: str,
    *,
    min_file_age_sec: float = STALE_STAGING_MAX_AGE_SEC,
) -> None:
    """Remove transient ``.<name>.previous.<rand>`` directories and stale ``.tmp`` files in ``target_dir``."""
    if not os.path.isdir(target_dir):
        return
    now = time.time()
    for item in os.listdir(target_dir):
        if (
            (item.startswith(".") and (".previous." in item or item.endswith(".tmp")))
            or item == "test-artifacts"
        ):
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isdir(item_path) and not os.path.islink(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                elif (now - os.path.getmtime(item_path)) >= min_file_age_sec:
                    os.unlink(item_path)
            except OSError:
                pass


def _replace_or_discard_file(
    src: str,
    dst: str,
    *,
    is_jsonl: bool = False,
    preserve_larger: bool = False,
) -> None:
    if not os.path.exists(dst):
        os.replace(src, dst)
        return
    src_sz = os.path.getsize(src)
    dst_sz = os.path.getsize(dst)
    if is_jsonl:
        should_replace = src_sz > dst_sz
    elif preserve_larger and src_sz > 0 and dst_sz > 0 and (src_sz > dst_sz * 1.25 or dst_sz > src_sz * 1.25):
        # Protect irreplaceable state files from being overwritten by smaller fresh stubs during migration.
        should_replace = src_sz > dst_sz
    else:
        should_replace = src_sz > 0 and os.path.getmtime(src) >= os.path.getmtime(dst)
    if should_replace:
        os.replace(src, dst)
    else:
        os.unlink(src)


def organize_cachedb_subdomains(
    cachedb_dir: str | None = None,
    *,
    remove_stale_staging: bool = True,
) -> dict[str, list[str]]:
    """Partition flat/legacy files in ``cachedb_dir`` into thematic ``cachedb/<subdomain>/`` subdirectories."""
    base = cachedb_dir or get_cachedb_dir()
    for sub in CACHEDB_SUBDOMAINS:
        os.makedirs(os.path.join(base, *sub.split("/")), exist_ok=True)
    moved: dict[str, list[str]] = {sub: [] for sub in _ALL_STORAGE_SUBDOMAINS}
    if not os.path.isdir(base):
        return moved
    top_level_subs = set(_TOP_LEVEL_SUBDOMAINS) | {"bots"}
    if remove_stale_staging:
        _clean_stale_staging_in_dir(base)
        for sub in _ALL_STORAGE_SUBDOMAINS:
            sub_dir_p = os.path.join(base, *sub.split("/"))
            if os.path.isdir(sub_dir_p):
                _clean_stale_staging_in_dir(sub_dir_p)
                trend24_p = os.path.join(sub_dir_p, "24trend")
                if os.path.isdir(trend24_p):
                    _clean_stale_staging_in_dir(trend24_p)
    scan_dirs: list[tuple[str, set[str]]] = [(base, top_level_subs)]
    for sub_k, sub_skips in (
        ("system", {"heartbeat"}),
        ("account", set()),
        ("orders", set()),
        ("prices", set()),
        ("trends", {"24trend"}),
        ("telemetry", set()),
        ("macro", set()),
        ("llm", set()),
        ("portfolio", set()),
        ("state", set()),
        ("bots", set()),
    ):
        sub_p = os.path.join(base, sub_k)
        if os.path.isdir(sub_p):
            scan_dirs.append((sub_p, sub_skips))
    legacy_state_dir = os.path.join(base, "state")
    sibling_statedb = resolve_statedb_dir_from_cachedb(base)
    statedb_active = os.path.abspath(sibling_statedb) != os.path.abspath(base) and os.path.isdir(sibling_statedb)
    for scan_dir, skip_names in scan_dirs:
        for entry in sorted(os.listdir(scan_dir)):
            if entry in skip_names:
                continue
            src = os.path.join(scan_dir, entry)
            dom = resolve_storage_domain(entry)
            if not dom:
                continue
            root_kind, sub = dom
            if root_kind == "statedb" and statedb_active:
                dst_dir = os.path.join(sibling_statedb, *sub.split("/"))
            else:
                dst_dir = os.path.join(base, *sub.split("/"))
            moved_key = sub
            os.makedirs(dst_dir, exist_ok=True)
            target_entry = entry
            prefix_dot = "." if (entry.startswith(".") and not entry.startswith("..")) else ""
            bare_entry = entry[len(prefix_dot):]
            for new_exact, old_exacts in EXACT_LEGACY_FILENAMES.items():
                for old_ex in old_exacts:
                    if bare_entry == old_ex or bare_entry.startswith(old_ex + "."):
                        target_entry = prefix_dot + new_exact + bare_entry[len(old_ex):]
                        break
                if target_entry != entry:
                    break
            if os.path.abspath(scan_dir) == os.path.abspath(dst_dir) and target_entry == entry:
                continue
            dst = os.path.join(dst_dir, target_entry)
            try:
                if os.path.isdir(src) and not os.path.islink(src):
                    if remove_stale_staging:
                        _clean_stale_staging_in_dir(src)
                    os.makedirs(dst, exist_ok=True)
                    for child in os.listdir(src):
                        c_src = os.path.join(src, child)
                        c_dst = os.path.join(dst, child)
                        if os.path.isdir(c_src) and not os.path.islink(c_src):
                            shutil.rmtree(c_src, ignore_errors=True)
                        else:
                            _replace_or_discard_file(c_src, c_dst)
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    _replace_or_discard_file(
                        src,
                        dst,
                        is_jsonl=(entry.endswith(".jsonl") or ".jsonl." in entry),
                        preserve_larger=(root_kind == "statedb" and not entry.endswith(".lock")),
                    )
                moved[moved_key].append(target_entry)
            except OSError:
                pass
    if os.path.isdir(legacy_state_dir):
        try:
            os.rmdir(legacy_state_dir)
        except OSError:
            pass
    return moved


def organize_storage_domains(
    workspace_dir: str | None = None,
    *,
    cachedb_dir: str | None = None,
    statedb_dir: str | None = None,
    remove_stale_staging: bool = True,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Atomically organize workspace storage into pure Variant A ``cachedb/`` (regenerable) and ``statedb/`` (durable).

    Moves durable state/orders/llm/system files into ``statedb/<sub_dir>/`` and exchange-mirror
    account caches into ``cachedb/account/`` while renaming legacy filenames via ``EXACT_LEGACY_FILENAMES``.
    """
    ws = _clean_path_str(workspace_dir) or _ROOT
    c_base = _clean_path_str(cachedb_dir) or (os.path.join(ws, "cachedb") if workspace_dir else get_cachedb_dir())
    s_base = _clean_path_str(statedb_dir) or (os.path.join(ws, "statedb") if workspace_dir else get_statedb_dir())
    is_dry = _parse_bool_flag(dry_run, False)

    moved: dict[str, list[str]] = {
        **{f"cachedb/{sub}": [] for sub in CACHEDB_SUBDOMAINS},
        **{f"statedb/{sub}": [] for sub in STATEDB_SUBDOMAINS},
    }
    if not is_dry:
        for sub in CACHEDB_SUBDOMAINS:
            os.makedirs(os.path.join(c_base, *sub.split("/")), exist_ok=True)
        for sub in STATEDB_SUBDOMAINS:
            os.makedirs(os.path.join(s_base, *sub.split("/")), exist_ok=True)
        if remove_stale_staging:
            for root_p, subs in ((c_base, CACHEDB_SUBDOMAINS), (s_base, STATEDB_SUBDOMAINS)):
                if os.path.isdir(root_p):
                    _clean_stale_staging_in_dir(root_p)
                    for sub in subs:
                        sub_p = os.path.join(root_p, *sub.split("/"))
                        if os.path.isdir(sub_p):
                            _clean_stale_staging_in_dir(sub_p)

    top_skips = set(_TOP_LEVEL_SUBDOMAINS) | {"bots", "cachedb", "statedb"}
    scan_dirs: list[tuple[str, set[str]]] = []
    for root_p in (c_base, s_base):
        if os.path.isdir(root_p):
            scan_dirs.append((root_p, top_skips))
            for sub_k, sub_skips in (
                ("system", {"heartbeat"}),
                ("account", set()),
                ("orders", set()),
                ("prices", set()),
                ("trends", {"24trend"}),
                ("telemetry", set()),
                ("macro", set()),
                ("llm", set()),
                ("portfolio", set()),
                ("state", set()),
                ("bots", set()),
            ):
                sub_p = os.path.join(root_p, sub_k)
                if os.path.isdir(sub_p):
                    scan_dirs.append((sub_p, sub_skips))

    for scan_dir, skip_names in scan_dirs:
        for entry in sorted(os.listdir(scan_dir)):
            if entry in skip_names:
                continue
            src = os.path.join(scan_dir, entry)
            dom = resolve_storage_domain(entry)
            if dom is None:
                continue
            root_kind, sub = dom
            target_root = s_base if root_kind == "statedb" else c_base
            dst_dir = os.path.join(target_root, *sub.split("/"))
            target_entry = entry
            prefix_dot = "." if (entry.startswith(".") and not entry.startswith("..")) else ""
            bare_entry = entry[len(prefix_dot):]
            for new_exact, old_exacts in EXACT_LEGACY_FILENAMES.items():
                for old_ex in old_exacts:
                    if bare_entry == old_ex or bare_entry.startswith(old_ex + "."):
                        target_entry = prefix_dot + new_exact + bare_entry[len(old_ex):]
                        break
                if target_entry != entry:
                    break
            if os.path.abspath(scan_dir) == os.path.abspath(dst_dir) and target_entry == entry:
                continue
            dst = os.path.join(dst_dir, target_entry)
            key = f"{root_kind}/{sub}"
            if is_dry:
                moved[key].append(target_entry)
                continue
            try:
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.isdir(src) and not os.path.islink(src):
                    if remove_stale_staging:
                        _clean_stale_staging_in_dir(src)
                    os.makedirs(dst, exist_ok=True)
                    for child in os.listdir(src):
                        c_src = os.path.join(src, child)
                        c_dst = os.path.join(dst, child)
                        if os.path.isdir(c_src) and not os.path.islink(c_src):
                            shutil.rmtree(c_src, ignore_errors=True)
                        else:
                            _replace_or_discard_file(c_src, c_dst)
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    _replace_or_discard_file(
                        src,
                        dst,
                        is_jsonl=(entry.endswith(".jsonl") or ".jsonl." in entry),
                        preserve_larger=(root_kind == "statedb" and not entry.endswith(".lock")),
                    )
                moved[key].append(target_entry)
            except OSError:
                pass

    if not is_dry:
        for legacy_dir in ("orders", "state", "bots", "portfolio"):
            cand = os.path.join(c_base, legacy_dir)
            if os.path.isdir(cand):
                try:
                    os.rmdir(cand)
                except OSError:
                    pass
    return moved


def legacy_filenames_for(filepath: str) -> tuple[str, ...]:
    """Return legacy pre-symmetry and pre-subdomain candidate paths for ``filepath``."""
    raw = _clean_path_str(filepath)
    if not raw:
        return ()
    dirname, basename = os.path.split(raw)
    candidates = EXACT_LEGACY_FILENAMES.get(basename)
    if candidates is None:
        for new_prefix, old_prefixes in PREFIX_LEGACY_FILENAMES:
            if basename.startswith(new_prefix):
                suffix = basename[len(new_prefix):]
                candidates = tuple(old_prefix + suffix for old_prefix in old_prefixes)
                break
    dirs = [dirname]
    norm_dir = dirname.replace("\\", "/")
    for sub in _SORTED_SUBDOMAINS:
        if norm_dir.endswith(f"/{sub}") or norm_dir == sub:
            parent_dir = norm_dir[: -len(sub)].rstrip("/") or "."
            if parent_dir and parent_dir not in dirs:
                dirs.append(parent_dir)
            if "/" in sub:
                mid_dir = os.path.join(parent_dir, sub.split("/", 1)[0])
                if mid_dir not in dirs:
                    dirs.append(mid_dir)
            break
    out: list[str] = []
    for d in dirs[1:]:
        cand_same = os.path.join(d, basename)
        if cand_same not in out:
            out.append(cand_same)
    for d in dirs:
        for c in candidates or ():
            cand_p = os.path.join(d, c)
            if cand_p not in out:
                out.append(cand_p)
    return tuple(out)


def migrate_legacy_filename_if_needed(filepath: str) -> None:
    """Rename legacy pre-symmetry cache files in place when the new path is absent."""
    filepath = _clean_path_str(filepath)
    if not filepath or _exists_with_prev(filepath):
        return
    for legacy_path in legacy_filenames_for(filepath):
        if not _exists_with_prev(legacy_path):
            continue
        try:
            if os.path.exists(legacy_path):
                os.replace(legacy_path, filepath)
            _move_sidecars(legacy_path, filepath, include_archives=True)
            builtins.print(f"[cacheManager][migration] renamed {legacy_path} -> {filepath}")
            return
        except OSError as exc:
            builtins.print(f"[cacheManager][migration] failed renaming {legacy_path} -> {filepath}: {exc}")


def get_cachedb_dir() -> str:
    """Return the canonical cachedb directory path honoring MPTRADE_CACHEDB_DIR and BINANCE_CACHE_DIR."""
    for env_key in ("MPTRADE_CACHEDB_DIR", "BINANCE_CACHE_DIR"):
        if clean_dir := clean_scalar_str(os.environ.get(env_key)):
            return clean_dir
    return os.path.join(_ROOT, "cachedb")


def get_statedb_dir() -> str:
    """Return the canonical statedb directory path honoring MPTRADE_STATEDB_DIR."""
    if clean_dir := clean_scalar_str(os.environ.get("MPTRADE_STATEDB_DIR")):
        return clean_dir
    for env_key in ("MPTRADE_CACHEDB_DIR", "BINANCE_CACHE_DIR"):
        if clean_cdir := clean_scalar_str(os.environ.get(env_key)):
            return resolve_statedb_dir_from_cachedb(clean_cdir)
    return os.path.join(_ROOT, "statedb")


def resolve_workspace_dir_from_cachedb(cache_dir: object) -> str:
    """Resolve the workspace root directory corresponding to ``cache_dir`` (or ``statedb``)."""
    raw = _clean_path_str(cache_dir)
    if not raw:
        return ""
    abs_dir = os.path.normpath(os.path.abspath(raw.rstrip("/\\") or raw))
    return os.path.dirname(abs_dir) if os.path.basename(abs_dir).lower() in ("cachedb", "statedb") else abs_dir


def state_path(name: str) -> str:
    """Return the canonical path for a durable ``statedb`` file (with automatic pre-cutover ``cachedb`` fallback)."""
    raw = _clean_path_str(name)
    if not raw:
        return name
    if os.path.isabs(raw):
        return str(resolve_statedb_filepath(raw))
    norm = _norm_rel_cache_path(raw, strip_cachedb_prefix=False)
    if norm.startswith("statedb/"):
        norm = norm[len("statedb/"):]
    elif norm.startswith("cachedb/"):
        norm = norm[len("cachedb/"):]
    elif "/" in norm:
        head = norm.split("/", 1)[0]
        if head not in _TOP_LEVEL_SUBDOMAINS:
            return str(resolve_statedb_filepath(raw))
    s_dir = get_statedb_dir()
    if os.path.isdir(s_dir):
        return str(resolve_statedb_filepath(os.path.join(s_dir, norm)))
    return cache_path(norm)


state_path._is_canonical_cache_path = True  # type: ignore[attr-defined]


def cache_path(name: str) -> str:
    """Return the path to a cache/state file inside the canonical storage directories.

    Preserves name when it is already absolute or None/empty, while resolving
    thematic ``cachedb/<subdomain>/`` and ``statedb/<subdomain>/`` paths when active on disk.
    """
    raw = _clean_path_str(name)
    if not raw:
        return name
    if os.path.isabs(raw):
        return str(resolve_cachedb_filepath(raw))
    norm = _norm_rel_cache_path(raw, strip_cachedb_prefix=False)
    if "/" in norm:
        head = norm.split("/", 1)[0]
        if head not in ("cachedb", "statedb", "24trend") and head not in _TOP_LEVEL_SUBDOMAINS:
            return str(resolve_cachedb_filepath(raw))
    name = _norm_rel_cache_path(raw, strip_cachedb_prefix=True)
    if name.startswith("statedb/"):
        name = name[len("statedb/"):]
    d = get_cachedb_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return str(resolve_cachedb_filepath(os.path.join(d, name)))


cache_path._is_canonical_cache_path = True  # type: ignore[attr-defined]


def _resolve_order_ledger_file(filename: str, cachedb_dir: str | None = None) -> str:
    """Resolve an order-ledger filename inside ``cachedb_dir`` (or canonical ``cachedb/orders/``)."""
    base = _clean_path_str(cachedb_dir) if cachedb_dir else ""
    if base:
        return str(resolve_cachedb_filepath(os.path.join(base, filename)))
    return cache_path(filename)


def order_ledger_active_path(cachedb_dir: str | None = None) -> str:
    """Return the canonical path for the materialized active order ledger snapshot."""
    return _resolve_order_ledger_file(ORDER_LEDGER_ACTIVE_FILENAME, cachedb_dir=cachedb_dir)


def order_ledger_events_filename(day_utc: str) -> str:
    """Return the canonical daily append-only order-events JSONL filename for ``day_utc`` (YYYY-MM-DD)."""
    return f"{ORDER_LEDGER_EVENTS_PREFIX}{str(day_utc).strip()}.jsonl"


def order_ledger_closed_filename(month_utc: str) -> str:
    """Return the canonical monthly closed-orders archive JSONL filename for ``month_utc`` (YYYY-MM)."""
    return f"{ORDER_LEDGER_CLOSED_PREFIX}{str(month_utc).strip()}.jsonl"


def order_ledger_events_path(day_utc: str, cachedb_dir: str | None = None) -> str:
    """Return the canonical path for one daily order-events JSONL file."""
    return _resolve_order_ledger_file(order_ledger_events_filename(day_utc), cachedb_dir=cachedb_dir)


def order_ledger_closed_path(month_utc: str, cachedb_dir: str | None = None) -> str:
    """Return the canonical path for one monthly closed-orders archive JSONL file."""
    return _resolve_order_ledger_file(order_ledger_closed_filename(month_utc), cachedb_dir=cachedb_dir)


CACHE_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "USD", "EUR", "RON", "GBP",
)
_BARE_STABLECOIN_BASES: frozenset[str] = frozenset({
    "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "USDP", "DAI",
    "USDE", "USDS", "USD1", "PYUSD", "RLUSD", "BFUSD", "AEUR", "EURI",
})
_LEGACY_KRAKEN_X_BASES: frozenset[str] = frozenset({
    "XXBT", "XETH", "XXDG", "XXRP", "XLTC", "XXLM", "XZEC", "XXMR", "XETC", "XREP", "XMLN",
})
_KNOWN_T_OR_B_BASES: frozenset[str] = frozenset({
    "SHIB", "ABNB", "NFTB", "LAMB", "SCHB", "PNUT", "GOAT", "BRETT",
    "BLAST", "DRIFT", "POPCAT", "PROMPT", "IOST", "SCRT", "DENT", "ZENT",
    "FORT", "VIDT", "XPRT", "MSFT", "RDDT", "RIOT", "IBIT", "MSTRT",
})
_CANONICAL_BASE_ALIASES: dict[str, str] = {
    "XBT": "BTC", "XXBT": "BTC", "XXBTZ": "BTC",
    "XETH": "ETH", "XETHZ": "ETH",
    "XDG": "DOGE", "XXDG": "DOGE", "XXDGZ": "DOGE",
}
_CROSS_VENUE_BASE_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "BTC": ("XBT",), "XBT": ("BTC",),
    "ETH": ("XETH",), "XETH": ("ETH",),
    "DOGE": ("XDG",), "XDG": ("DOGE",),
}
_KNOWN_USD_BASES_ENDING_IN_T_OR_B: frozenset[str] = frozenset({
    "XBT", "XXBT", "XXBTZ", "XAUT", "USDT", "ARB", "DOT", "BAT",
    "HOT", "NOT", "ACT", "FET", "APT", "CRB", "SUB", "GRT", "LPT",
})
_VENUE_ALIASES: dict[str, str] = {
    "bin": "binance", "binance_spot": "binance", "binancespot": "binance",
    "krk": "kraken", "kraken_spot": "kraken", "krakenspot": "kraken",
    "hl": "hyperliquid", "hyper_liquid": "hyperliquid",
    "hyperliquid_spot": "hyperliquid", "hyperliquidspot": "hyperliquid",
    "trading212": "t212", "trading_212": "t212", "212trading": "t212", "212": "t212",
}
EQUITY_VENUES: frozenset[str] = frozenset({"t212"})
PRIMARY_STREAM_VENUES: frozenset[str] = frozenset({"binance"})
CYCLE_STATE_VENUES: frozenset[str] = frozenset({"kraken", "hyperliquid", "t212"})
USDC_DEFAULT_VENUES: frozenset[str] = frozenset({"binance", "hyperliquid"})
DEFAULT_USD_QUOTES: tuple[str, ...] = ("USDC", "USDT", "USD", "FDUSD")
_USD_FAMILY_QUOTES: tuple[str, ...] = ("USDC", "USDT", "USD", "FDUSD", "TUSD", "BUSD")
_USD_STABLECOIN_QUOTES: tuple[str, ...] = ("USDC", "USDT", "FDUSD", "TUSD", "BUSD")


def _has_xstock_lowercase_x(token: str) -> bool:
    return len(token) > 2 and token.endswith("x") and token[:-1].isalnum() and token[:-1].isupper()


READONLY_XSTOCK_SUFFIXES: tuple[str, ...] = ("X.T", ".T", "X.T/USD", "X.TUSD")
_CRYPTO_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USD", "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "EUR", "BTC", "ETH",
)
_EXPLICIT_CRYPTO_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "EUR",
)


def has_crypto_quote_suffix(symbol: object) -> bool:
    """Return True if ``symbol`` ends with a known crypto quote suffix."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return False
    return symbol.strip().upper().endswith(_CRYPTO_QUOTE_SUFFIXES)


def _extract_raw_symbol_base(raw: str, *, min_stem_len: int = 1) -> str:
    base = next((raw.split(sep, 1)[0].strip() for sep in ("/", "-", "_") if sep in raw), raw)
    base_u = base.upper()
    for quote in CACHE_QUOTE_SUFFIXES:
        if not _is_false_tusd_or_busd_match(base_u, quote) and base_u.endswith(quote) and len(base) >= len(quote) + min_stem_len:
            return base[: -len(quote)]
    return base


def is_readonly_xstock_symbol(symbol: object) -> bool:
    """Return True if ``symbol`` is a read-only Kraken xStock token or pair."""
    if (raw := clean_scalar_str(symbol)) is None:
        return False
    if raw.upper().endswith(READONLY_XSTOCK_SUFFIXES):
        return True
    stem = _extract_raw_symbol_base(raw, min_stem_len=3)
    return stem.upper().endswith(("X.T", ".T")) or _has_xstock_lowercase_x(stem)


def is_kraken_tokenized_asset(symbol: object) -> bool:
    """Return True if ``symbol`` requires Kraken ``tokenized_asset`` API routing (xStocks / SPV pairs)."""
    if is_readonly_xstock_symbol(symbol):
        return True
    if (raw := clean_scalar_str(symbol)) is None:
        return False
    raw_base = _extract_raw_symbol_base(raw, min_stem_len=1).upper()
    return len(raw_base) >= 6 and raw_base.endswith("SPV")


def has_equity_suffix(symbol: object, *, include_compact: bool = False) -> bool:
    """Return True if ``symbol`` is a string ending in ``_EQ`` or ``_US_EQ`` (case-insensitive)."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return False
    s = symbol.strip().upper()
    if s.endswith(("_EQ", "_US_EQ")):
        return True
    return bool(include_compact and len(s) > 4 and s.endswith("USEQ"))


def ensure_equity_suffix(symbol: object, *, is_equity: bool = True, suffix: str = "_US_EQ") -> str:
    """Append ``suffix`` when ``is_equity`` is True and ``symbol`` lacks an equity suffix."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return ""
    raw = symbol.strip()
    if not _parse_bool_flag(is_equity, True) or has_equity_suffix(raw):
        return raw
    s = raw.upper()
    return f"{s}{suffix}" if s else raw


def strip_equity_suffix(symbol: object, *, base_only: bool = False) -> str:
    """Return uppercase ``symbol`` stripped of trailing ``_US_EQ`` or ``_EQ`` equity suffixes."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return ""
    s = symbol.strip().upper()
    if s.endswith("_US_EQ"):
        s = s[:-6]
    elif s.endswith("_EQ"):
        s = s[:-3]
    return s.split("_", 1)[0] if base_only else s


def ensure_xstock_asset(symbol: object) -> str:
    """Return the canonical mixed-case Kraken xStock asset token ``<STEM>x`` (e.g. ``SPCX`` / ``SPCXx.T`` / ``SPCXxUSD`` -> ``SPCXx``)."""
    raw = clean_scalar_str(symbol)
    if raw is None:
        return ""
    stem = canonical_base_asset(raw)
    return f"{stem}x" if stem else ""


def ensure_xstock_pair(symbol: object, quote: str = "USD") -> str:
    """Return the canonical mixed-case Kraken xStock spot pair ``<STEM>x<QUOTE>`` (e.g. ``SPCX`` -> ``SPCXxUSD``)."""
    asset = ensure_xstock_asset(symbol)
    if not asset:
        return ""
    q = clean_symbol_key(quote) if isinstance(quote, str) else ""
    return f"{asset}{q or 'USD'}"


def filter_asset_class_variants(
    symbol: object,
    variants: object,
    *,
    is_equity: bool | None = None,
    include_compact_equity: bool = False,
) -> tuple[str, ...]:
    """Filter ``variants`` so US equities (``_EQ``/``_US_EQ``) and crypto pairs never cross-contaminate."""
    eq_flag = has_equity_suffix(symbol) if is_equity is None else bool(is_equity)
    if eq_flag:
        return tuple(
            c
            for c in (variants or ())
            if isinstance(c, str) and not has_crypto_quote_suffix(c)
        )
    return tuple(
        c
        for c in (variants or ())
        if isinstance(c, str) and not has_equity_suffix(c, include_compact=include_compact_equity)
    )


def _is_false_tusd_or_busd_match(token: str, quote: str) -> bool:
    """Return True if a symbol ending in TUSD or BUSD is actually a USD quote pair
    whose base ends in 'T' (e.g. XBTUSD, XAUTUSD, SPCXx.TUSD, DOTUSD, BATUSD, HOTUSD, NOTUSD, ACTUSD, USDTUSD, XXBTUSD)
    or 'B' (e.g. ARBUSD, CRBUSD, SUBUSD)."""
    if quote not in ("TUSD", "BUSD"):
        return False
    if len(token) - len(quote) < 3:
        return True
    if token.endswith("USD"):
        stem_usd = token[:-3]
        if (
            stem_usd in _KNOWN_USD_BASES_ENDING_IN_T_OR_B
            or stem_usd in _CANONICAL_BASE_ALIASES
            or stem_usd in _LEGACY_KRAKEN_X_BASES
            or stem_usd in _BARE_STABLECOIN_BASES
            or stem_usd in _KNOWN_T_OR_B_BASES
            or stem_usd.endswith((".T", "X.T", "XBT"))
            or stem_usd.startswith("XAU")
        ):
            return True
    return False


def _strip_kraken_asset_suffixes(token: str) -> str:
    """Strip Kraken xStock/SPV/futures/staking suffixes (``x.T``, ``.T``, ``.F``, ``.S``, ``SPV``)."""
    if token.endswith("X.T") and len(token) > 3:
        return token[:-3]
    if token.endswith((".T", ".F", ".S")) and len(token) > 2:
        return token[:-2]
    if token.endswith("SPV") and len(token) >= 6:
        return token[:-3]
    return token


def clean_symbol_key(symbol: object) -> str:
    """Return uppercase alphanumeric symbol key stripped of '/', '-', '_'."""
    raw_u = _clean_cache_symbol(symbol, preserve_case=True)
    if not raw_u:
        return ""
    stripped = raw_u.replace("/", "").replace("-", "").replace("_", "")
    return clean_upper_scalar(stripped) or ""


def normalize_cache_base_asset(symbol: object) -> str:
    """Extract the canonical uppercase base asset from a spot/perp/venue symbol.

    Handles ``/``, ``-``, and ``_`` separators (e.g. ``TAO/USD``, ``SOL-USD``,
    ``HYPE_USDC``, ``SPCXx/USD``) as well as concatenated quote suffixes while
    preserving the stablecoin pair ``USDCUSDT`` intact. Returns ``""`` for
    non-string or blank inputs.
    """
    raw = clean_scalar_str(symbol)
    if raw is None:
        return ""
    if clean_symbol_key(raw) == "USDCUSDT":
        return "USDCUSDT"
    raw_base = next((raw.split(sep, 1)[0].strip() for sep in ("/", "-", "_") if sep in raw), raw)
    for quote in CACHE_QUOTE_SUFFIXES:
        if _is_false_tusd_or_busd_match(raw_base.upper(), quote):
            continue
        if raw_base.upper().endswith(quote) and len(raw_base) > len(quote) + 2:
            stem = raw_base[: -len(quote)]
            if _has_xstock_lowercase_x(stem):
                raw_base = stem[:-1] + quote
            break
    if _has_xstock_lowercase_x(raw_base):
        raw_base = raw_base[:-1]
    s = _strip_kraken_asset_suffixes(raw_base.upper())
    if s.endswith("USEQ") and len(s) > 4:
        s = s[:-4]
    elif s not in _BARE_STABLECOIN_BASES:
        for quote in CACHE_QUOTE_SUFFIXES:
            if not _is_false_tusd_or_busd_match(s, quote) and s.endswith(quote) and len(s) > len(quote):
                s = s[: -len(quote)]
                break
    s = _strip_kraken_asset_suffixes(s)
    if (len(s) == 5 and s.startswith("X") and s.endswith("Z") and s[:-1] in _LEGACY_KRAKEN_X_BASES) or s in {"XBTZ", "XDGZ"}:
        s = s[:-1]
    if s in _LEGACY_KRAKEN_X_BASES:
        s = s[1:]
    return clean_upper_scalar(s) or ""


def normalize_venue_name(venue: object, default: str | None = "") -> str | None:
    """Return the canonical lowercase venue identifier ('binance' | 'kraken' | 'hyperliquid' | 't212').

    Accepts either a venue string or a provider object exposing a string ``.name`` attribute.
    """
    if venue is None or isinstance(venue, bool):
        return default
    raw_venue = venue if isinstance(venue, str) else getattr(venue, "name", None)
    clean_v = clean_scalar_str(raw_venue)
    if clean_v is None:
        return default
    v = clean_v.lower().replace("-", "_").replace(" ", "_")
    return _VENUE_ALIASES.get(v, v) if v else default


def is_equity_venue(venue: object) -> bool:
    """Return True if ``venue`` resolves to an equities venue (e.g. ``t212``)."""
    return normalize_venue_name(venue) in EQUITY_VENUES


def is_equity_venue_or_suffix(symbol: object = None, venue: object = None) -> bool:
    """Return True if ``venue`` is an equities venue or ``symbol`` carries an equity suffix."""
    return is_equity_venue(venue) or has_equity_suffix(symbol)


def matches_asset_scope(symbol: object, venue: object = None, scope: object = "all") -> bool:
    """Return True if ``(symbol, venue)`` belongs to ``scope`` (``'crypto'``, ``'equities'``, or ``'all'``)."""
    scope_norm = str(scope or "all").strip().lower() if not isinstance(scope, bool) else "all"
    if scope_norm not in ("equities", "crypto"):
        return True
    v_norm = normalize_venue_name(venue)
    sym_u = symbol.strip().upper() if isinstance(symbol, str) and not isinstance(symbol, bool) else ""
    is_eq_sym = has_equity_suffix(sym_u)
    if scope_norm == "equities":
        return (
            (not v_norm or is_equity_venue(v_norm))
            and not sym_u.endswith(_EXPLICIT_CRYPTO_QUOTE_SUFFIXES)
            and (is_equity_venue(v_norm) or is_eq_sym)
        )
    return not is_equity_venue(v_norm) and not is_eq_sym


def is_primary_stream_venue(venue: object, *, default_primary: bool = False) -> bool:
    """Return True if ``venue`` resolves to the primary shared-registry/multiplex venue (``binance``)."""
    v_norm = normalize_venue_name(venue, default=None)
    if not v_norm:
        return bool(default_primary)
    return v_norm in PRIMARY_STREAM_VENUES


def uses_cycle_state_files(venue: object) -> bool:
    """Return True if ``venue`` persists per-symbol ``.state_*.json`` DCA cycle state files."""
    return normalize_venue_name(venue) in CYCLE_STATE_VENUES


def venue_default_quote_asset(venue: object) -> str:
    """Return the default settlement quote asset ('USDC' or 'USD') for ``venue`` when unquoted."""
    return "USDC" if normalize_venue_name(venue) in USDC_DEFAULT_VENUES else "USD"


def canonical_base_asset(symbol: object) -> str:
    """Return the venue-agnostic canonical ticker for any symbol (e.g. XBTUSD -> BTC, SPCXx.T -> SPCX)."""
    base = normalize_cache_base_asset(symbol)
    return _CANONICAL_BASE_ALIASES.get(base, base)


def same_canonical_base_asset(sym_a: object, sym_b: object) -> bool:
    """Return True when both symbols resolve to the same non-empty canonical base asset."""
    base_a = canonical_base_asset(sym_a)
    base_b = canonical_base_asset(sym_b)
    return bool(base_a and base_a == base_b)


def symbol_quote_asset(symbol: object) -> str:
    """Return the uppercase quote asset of ``symbol`` (e.g. ``BTCUSDC`` -> ``USDC``, ``XBTEUR`` -> ``EUR``), or ``''``."""
    parts = split_symbol_assets(symbol)
    return parts[1] if parts is not None else ""


def _usd_base_pair(base_src: str) -> tuple[str, str] | None:
    base = normalize_cache_base_asset(base_src)
    return (base, "USD") if base else None


def split_symbol_assets(symbol: object) -> tuple[str, str] | None:
    """Split a spot/equity pair symbol (e.g. TAOUSD, BTCUSDC, TAO/USD, SPCX_US_EQ, SPCXx.T) into ``(base, quote)``."""
    raw = clean_scalar_str(symbol)
    if raw is None:
        return None
    sym_u = clean_symbol_key(raw)
    if not sym_u:
        return None
    if (sym_u.endswith("USEQ") and len(sym_u) > 4) or (raw.upper().endswith("_EQ") and len(sym_u) > 2):
        return _usd_base_pair(raw)
    for quote in (*CACHE_QUOTE_SUFFIXES, "BTC", "ETH"):
        if _is_false_tusd_or_busd_match(sym_u, quote):
            continue
        if sym_u.endswith(quote) and len(sym_u) > len(quote):
            if "/" in raw:
                base_raw = raw.split("/", 1)[0].strip()
            elif raw.upper().endswith(quote):
                base_raw = raw[: -len(quote)].strip(" -_/")
            else:
                base_raw = sym_u[: -len(quote)]
            base = normalize_cache_base_asset(base_raw)
            return (base, quote) if base else None
    if sym_u.endswith(("X.T", ".T", ".F", ".S")) and len(sym_u) > 3:
        return _usd_base_pair(raw)
    if _has_xstock_lowercase_x(raw):
        return _usd_base_pair(raw[:-1])
    return None


def is_fiat_usd_pair(symbol: object) -> bool:
    """Return True if ``symbol`` is a crypto fiat-USD pair (e.g. ``XBTUSD``, ``DOTUSD``, ``ARBUSD``, ``TAO/USD``), excluding stablecoins and equities."""
    raw_u = clean_upper_scalar(symbol)
    return bool(
        raw_u
        and raw_u.endswith("USD")
        and not has_equity_suffix(raw_u, include_compact=True)
        and symbol_quote_asset(symbol) == "USD"
    )


def is_usd_stablecoin_pair(symbol: object) -> bool:
    """Return True if ``symbol`` is quoted in a USD stablecoin (``USDC``, ``USDT``, ``FDUSD``, ``TUSD``, ``BUSD``), excluding false ``TUSD``/``BUSD`` stems like ``XBTUSD``/``DOTUSD``/``ARBUSD``."""
    return symbol_quote_asset(symbol) in _USD_STABLECOIN_QUOTES


def is_usdc_pair(symbol: object) -> bool:
    """Return True if ``symbol`` is a non-equity pair explicitly quoted in ``USDC``."""
    return symbol_quote_asset(symbol) == "USDC"


def is_eur_pair(symbol: object) -> bool:
    """Return True if ``symbol`` is a non-equity pair explicitly quoted in ``EUR``."""
    return symbol_quote_asset(symbol) == "EUR"


def is_fiat_quote_pair(symbol: object) -> bool:
    """Return True if ``symbol`` is a non-equity crypto pair quoted in fiat ``USD`` or ``EUR``."""
    return is_fiat_usd_pair(symbol) or is_eur_pair(symbol)


def is_usd_compatible_pair(symbol: object) -> bool:
    """Return True if ``symbol`` carries a USD-compatible quote family (``USD``, ``USDC``, ``USDT``, etc.)."""
    return "USD" in compatible_quote_suffixes(symbol)


def ensure_quote_suffix(symbol: object, default_quote: str = "USDC") -> str:
    """Append ``default_quote`` when ``symbol`` is a bare token lacking a quote suffix, preserving xStock mixed case."""
    raw = clean_scalar_str(symbol)
    if raw is None:
        return ""
    raw_u = raw.upper()
    if has_equity_suffix(raw_u, include_compact=True):
        return raw_u
    if is_readonly_xstock_symbol(raw):
        q_default = clean_symbol_key(default_quote) if isinstance(default_quote, str) else ""
        q_default = q_default or "USD"
        existing_q = symbol_quote_asset(raw) if _extract_raw_symbol_base(raw, min_stem_len=3) != raw else ""
        return ensure_xstock_pair(raw, quote=existing_q or q_default)
    if not (sym := clean_symbol_key(raw_u)):
        return ""
    q = clean_symbol_key(default_quote) if isinstance(default_quote, str) else ""
    q = q or "USDC"
    return sym if (sym == q or symbol_quote_asset(sym)) else f"{sym}{q}"


def crypto_quote_priority_rank(
    symbol: object,
    quote_order: tuple[str, ...] = ("USDC", "USDT", "USD", "EUR"),
) -> int:
    """Return 0-based priority rank of ``symbol``'s resolved crypto quote asset within ``quote_order``."""
    order = (
        quote_order
        if isinstance(quote_order, tuple) and all(isinstance(x, str) and x == x.strip().upper() for x in quote_order)
        else tuple(
            x.strip().upper()
            for x in (quote_order if isinstance(quote_order, (list, tuple, set, frozenset)) else ())
            if isinstance(x, str) and x.strip()
        )
    )
    if has_equity_suffix(symbol, include_compact=True):
        return len(order)
    q = symbol_quote_asset(symbol)
    return order.index(q) if q in order else len(order)


def canonical_symbol_key(symbol: object) -> str:
    """Return a delimiter-stripped uppercase symbol key with canonical base asset aliases
    (e.g. ``XBT/USDC`` -> ``BTCUSDC``, ``XXBTZUSD`` -> ``BTCUSD``, ``XDG-USD`` -> ``DOGEUSD``,
    ``SPCX_US_EQ`` -> ``SPCXUSD``, ``XBT`` -> ``BTC``) while preserving quote currency."""
    if isinstance(symbol, bool) or symbol is None:
        return ""
    raw_str = str(symbol).strip()
    cleaned = clean_symbol_key(raw_str)
    if not cleaned:
        return ""
    if cleaned == "USDCUSDT":
        return "USDCUSDT"
    parts = split_symbol_assets(raw_str)
    if parts is not None:
        base, quote = parts
        return f"{_CANONICAL_BASE_ALIASES.get(base, base)}{quote}"
    return canonical_base_asset(raw_str) or cleaned


def resolve_binance_lookup_symbol(symbol: object) -> str:
    """Resolve a cross-venue symbol to its canonical Binance ``<BASE>USDT`` lookup symbol."""
    if not isinstance(symbol, str):
        return ""
    s = symbol.strip().upper()
    if not s:
        return ""
    if clean_symbol_key(s) == "USDCUSDT":
        return "USDCUSDT"
    base = canonical_base_asset(s)
    return f"{base}USDT" if base else (s.replace("USDC", "USDT") if "USDC" in s else s)


def resolve_kraken_lookup_symbol(symbol: object, quote: str = "USD") -> str:
    """Resolve any venue symbol to its canonical Kraken public ``<BASE><QUOTE>`` / ``XBT<QUOTE>`` lookup pair."""
    base = canonical_base_asset(symbol)
    if not base:
        return ""
    q = quote.strip().upper() if isinstance(quote, str) and not isinstance(quote, bool) and quote.strip() else "USD"
    if base == "BTC":
        return f"XBT{q}"
    if base == "DOGE":
        return f"XDG{q}"
    return f"{base}{q}"


_KRAKEN_QUOTES = (
    "USDC",
    "USDT",
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "JPY",
    "AUD",
    "BTC",
    "XBT",
    "ETH",
)


def split_kraken_pair(value: object) -> tuple[str, str]:
    """Split a Kraken pair or bare asset into ``(canonical_base, canonical_quote)``."""
    raw = clean_scalar_str(value)
    if raw is None:
        return "", ""
    key = "".join(ch for ch in raw.upper() if ch.isalnum())
    if not key or clean_upper_scalar(key) is None:
        return "", ""
    quote = next(
        (cand for cand in _KRAKEN_QUOTES if key.endswith(cand) and len(key) > len(cand)),
        "",
    )
    if not quote:
        base_raw = raw
    elif "/" in raw:
        base_raw = raw.split("/", 1)[0].strip()
    elif raw.upper().endswith(quote):
        base_raw = raw[: -len(quote)].strip(" -_/")
    else:
        base_raw = key[: -len(quote)]
    base = canonical_base_asset(base_raw) or (clean_upper_scalar("".join(ch for ch in base_raw.upper() if ch.isalnum())) or "")
    if not base:
        return "", ""
    if quote == "XBT":
        quote = "BTC"
    return base, quote


def kraken_pair_key(value: object) -> str:
    """Canonicalize common Kraken legacy pair wrappers for strict comparison."""
    base, quote = split_kraken_pair(value)
    return base + quote


def kraken_pair_matches(expected: object, actual: object) -> bool:
    """Return True when ``expected`` and ``actual`` refer to the same canonical Kraken pair."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    b_exp, q_exp = split_kraken_pair(expected)
    b_act, q_act = split_kraken_pair(actual)
    if not b_exp or not b_act:
        return False
    is_tok = is_kraken_tokenized_asset(expected) or is_kraken_tokenized_asset(actual)
    base_ok = (
        b_exp == b_act
        or (
            is_tok
            and (
                (len(b_exp) >= 3 and b_exp.endswith("X") and b_exp[:-1] == b_act)
                or (len(b_act) >= 3 and b_act.endswith("X") and b_act[:-1] == b_exp)
            )
        )
    )
    if not base_ok:
        return False
    if q_exp and q_act:
        return q_exp == q_act
    return True


def resolve_cache_dir_path(cache_dir: object, *, fallback_default: bool = False) -> str | None:
    """Resolve a collector ``cache_dir`` argument (e.g. ``'cachedb'``) to its effective directory path."""
    raw = _clean_path_str(cache_dir)
    if not raw:
        return get_cachedb_dir() if fallback_default else None
    if raw == "cachedb" or os.path.abspath(raw) == os.path.abspath(_ROOT):
        return get_cachedb_dir()
    return raw


def _strip_cachedb_subdomain_dir(dir_path: str) -> str:
    """Strip any trailing thematic cachedb subdomain directory (including ``system/heartbeat`` and ``trends/24trend``)."""
    cur = os.path.normpath(dir_path) if dir_path else ""
    if not cur:
        return ""
    parent = os.path.dirname(cur)
    base = os.path.basename(cur)
    pbase = os.path.basename(parent) if parent else ""
    if parent and (f"{pbase}/{base}" in _ALL_STORAGE_SUBDOMAINS or (pbase == "trends" and base == "24trend")):
        gp = os.path.dirname(parent)
        return gp if gp and gp != parent else cur
    if base in _ALL_STORAGE_SUBDOMAINS and parent and parent != cur:
        return parent
    return cur


def cachedb_root_for_file(filepath: object) -> str:
    """Return the root ``cachedb`` directory containing ``filepath``, stripping any thematic subdomain directory."""
    raw = _clean_path_str(filepath)
    if not raw:
        return ""
    stripped = _strip_cachedb_subdomain_dir(os.path.dirname(raw))
    if os.path.basename(stripped).lower() == "statedb":
        sibling_c = os.path.join(os.path.dirname(stripped), "cachedb")
        if os.path.isdir(sibling_c):
            return sibling_c
    return stripped


@dataclass(frozen=True)
class CachedbRoot:
    """Immutable value-object representing the root ``cachedb/`` directory (never a subdomain leaf)."""

    path: str

    @classmethod
    def from_path(cls, any_path: object) -> "CachedbRoot":
        """Normalize any file or directory path inside ``cachedb/`` or ``statedb/`` to its root ``CachedbRoot``."""
        raw = _clean_path_str(any_path)
        if not raw:
            return cls(get_cachedb_dir())
        norm = os.path.normpath(raw)
        if os.path.isdir(norm):
            root_dir = _strip_cachedb_subdomain_dir(norm)
        else:
            base = os.path.basename(norm)
            parent = os.path.dirname(norm)
            pbase = os.path.basename(parent) if parent else ""
            if (
                base in _ALL_STORAGE_SUBDOMAINS
                or (parent and f"{pbase}/{base}" in _ALL_STORAGE_SUBDOMAINS)
                or (pbase == "trends" and base == "24trend")
                or base.lower() in ("cachedb", "statedb")
            ):
                root_dir = _strip_cachedb_subdomain_dir(norm)
            else:
                root_dir = _strip_cachedb_subdomain_dir(parent)
        if root_dir and os.path.basename(root_dir).lower() == "statedb":
            sibling_c = os.path.join(os.path.dirname(root_dir), "cachedb")
            if os.path.isdir(sibling_c):
                root_dir = sibling_c
        return cls(root_dir or get_cachedb_dir())

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def resolve(self, filename: str, *, migrate: bool = False) -> str:
        """Resolve ``filename`` inside this ``cachedb`` root into its canonical thematic subdomain."""
        clean_name = _norm_rel_cache_path(_clean_path_str(filename), strip_cachedb_prefix=True)
        if not clean_name:
            return self.path
        return str(resolve_cachedb_filepath(os.path.join(self.path, clean_name), migrate=migrate))

    def subdomain_dir(self, subdomain: str) -> str:
        """Return the absolute/joined directory path for ``subdomain`` under this ``cachedb`` root."""
        parts = [p for p in _clean_path_str(subdomain).replace("\\", "/").split("/") if p]
        return os.path.join(self.path, *parts) if parts else self.path


def telemetry_cache_file_path(cache_dir: object, prefix: str, symbol: object) -> str | None:
    """Build the disk snapshot path ``<cache_dir>/<prefix>_<SAFE_SYMBOL>.json`` for a symbol collector."""
    eff_dir = resolve_cache_dir_path(cache_dir)
    raw_sym = clean_scalar_str(symbol)
    clean_pfx = clean_scalar_str(prefix)
    if not eff_dir or raw_sym is None or clean_pfx is None:
        return None
    raw_sym = _resolve_canonical_cache_symbol(raw_sym)
    raw_u = raw_sym.upper()
    if is_kraken_tokenized_asset(raw_sym) and raw_sym.replace("_", "").isalnum():
        safe_sym = raw_sym
    elif has_equity_suffix(raw_u) and raw_u.replace("_", "").isalnum():
        safe_sym = raw_u
    else:
        safe_sym = clean_symbol_key(raw_sym)
    if not safe_sym:
        return None
    return str(resolve_cachedb_filepath(os.path.join(eff_dir, f"{clean_pfx}_{safe_sym}.json")))


VENUE_RUNTIME_DIR_MAP: dict[str, str] = {
    "binance": "cachedb",
    "kraken": "kraken",
    "hyperliquid": "hyperliquid",
    "t212": "212trading",
}


def resolve_venue_runtime_dir(
    venue: object = "binance",
    *,
    workspace_dir: object = None,
    cache_dir: object = None,
) -> str:
    """Resolve the runtime state directory for a canonical venue ('binance', 'kraken', 'hyperliquid', 't212')."""
    v_norm = normalize_venue_name(venue, default="binance") or "binance"
    c_dir = _clean_path_str(cache_dir)
    ws_dir = _clean_path_str(workspace_dir)
    if is_primary_stream_venue(v_norm):
        if c_dir:
            return c_dir
        if ws_dir:
            return os.path.join(ws_dir, "cachedb")
        return get_cachedb_dir()
    base_ws = ws_dir or _ROOT
    rel_dir = VENUE_RUNTIME_DIR_MAP.get(v_norm, v_norm)
    return os.path.join(base_ws, rel_dir)


def resolve_venue_trailing_state_path(
    venue: object = "binance",
    *,
    workspace_dir: object = None,
    cache_dir: object = None,
    filename: str | None = None,
    resolve_subdir: bool = True,
) -> str:
    """Resolve the canonical per-venue trailing state path (``statedb/bots/trailing_state_<venue>.json``)."""
    v_norm = normalize_venue_name(venue, default="binance") or "binance"
    fname = _clean_path_str(filename) if filename is not None else f"trailing_state_{v_norm}.json"
    if not fname:
        fname = f"trailing_state_{v_norm}.json"
    if cache_dir is not None and (c_dir := _clean_path_str(cache_dir)):
        statedb_dir = resolve_statedb_dir_from_cachedb(c_dir)
        return os.path.join(statedb_dir, "bots", fname)
    if workspace_dir is not None and (w_dir := _clean_path_str(workspace_dir)):
        return os.path.join(w_dir, "statedb", "bots", fname)
    return state_path(fname)


_SHADOW_CYCLE_STATE_TOKENS: frozenset[str] = frozenset({"PAPER", "REPLAY", "SHADOW", "BACKTEST", "MOCK"})
_SHADOW_OID_PREFIXES: tuple[str, ...] = ("PAPER-", "REPLAY-", "SHADOW-", "BACKTEST-", "MOCK-")


def is_shadow_or_paper_order_id(val: object) -> bool:
    """Return True if ``val`` (scalar ID or order dict) carries a paper/replay/shadow/backtest/mock prefix."""
    if isinstance(val, str):
        return val.strip().upper().startswith(_SHADOW_OID_PREFIXES)
    if isinstance(val, dict):
        return any(
            str(val.get(k) or "").strip().upper().startswith(_SHADOW_OID_PREFIXES)
            for k in ("oid", "order_id", "id", "txid", "intent_id", "client_order_id", "clientOrderId")
        )
    return False


def is_shadow_or_paper_state_payload(payload: object) -> bool:
    """Return True if a parsed ``.state_*.json`` dict represents a paper, replay, shadow, or mock session."""
    if not isinstance(payload, dict):
        return False
    if any(_parse_bool_flag(payload.get(fk), False) is True for fk in ("paper", "dry_run", "shadow", "replay")):
        return True
    if str(payload.get("mode") or payload.get("execution_mode") or "").strip().upper() in _SHADOW_CYCLE_STATE_TOKENS:
        return True
    if any(
        is_shadow_or_paper_order_id(payload.get(pk))
        for pk in ("last_hist_fill_id", "pending_intent", "pending_submit", "pending_exit", "tp_pending_exit", "sl_pending_exit")
    ):
        return True
    raw_orders = payload.get("orders")
    if isinstance(raw_orders, list) and raw_orders:
        oids = [
            oid
            for o in raw_orders
            if isinstance(o, dict)
            and (oid := str(o.get("oid") or o.get("order_id") or o.get("id") or o.get("txid") or "").strip().upper())
        ]
        if oids and all(oid.startswith(_SHADOW_OID_PREFIXES) for oid in oids):
            return True
    return False


def cycle_state_filename(token: object, *, preserve_case: bool = False) -> str:
    """Return the canonical venue DCA cycle state filename ``.state_<TOKEN>.json`` preserving mixed-case xStocks."""
    tok = _clean_cache_symbol(token, preserve_case=preserve_case)
    if not tok:
        return ""
    return f"{CYCLE_STATE_PREFIX}{tok}{CYCLE_STATE_EXT}"


def parse_cycle_state_symbol(path: object) -> str:
    """Extract the symbol/token stem from a ``.state_<TOKEN>.json`` filename or path, or ``''``."""
    bname = os.path.basename(str(path or "")).strip()
    if not (bname.startswith(CYCLE_STATE_PREFIX) and bname.lower().endswith(CYCLE_STATE_EXT)):
        return ""
    return bname[len(CYCLE_STATE_PREFIX) : -len(CYCLE_STATE_EXT)].strip()


def is_shadow_or_paper_state_filename(path: object) -> bool:
    """Return True if a ``.state_*.json`` filename stem denotes a paper, replay, shadow, or mock cycle state."""
    bname = os.path.basename(str(path or "")).strip()
    if not (bname.startswith(CYCLE_STATE_PREFIX) and bname.lower().endswith(CYCLE_STATE_EXT)):
        return False
    stem = parse_cycle_state_symbol(bname).upper()
    if not stem or stem.startswith(_SHADOW_OID_PREFIXES):
        return True
    parts = [p for p in stem.replace("-", "_").split("_") if p]
    return bool(parts) and (parts[0] in _SHADOW_CYCLE_STATE_TOKENS or parts[-1] in _SHADOW_CYCLE_STATE_TOKENS)


def iter_venue_cycle_state_files(
    venue: object,
    *,
    workspace_dir: object = None,
    cache_dir: object = None,
) -> list[str]:
    """Return sorted ``.state_*.json`` cycle state paths for ``venue``, excluding paper/replay/shadow filenames."""
    v_norm = normalize_venue_name(venue, default="binance") or "binance"
    rt_dir = resolve_venue_runtime_dir(v_norm, workspace_dir=workspace_dir, cache_dir=cache_dir)
    if not os.path.isdir(rt_dir):
        return []
    paths = [p for p in glob.glob(os.path.join(rt_dir, ".state_*.json")) if not is_shadow_or_paper_state_filename(p)]
    return sorted(
        paths,
        key=lambda p: (
            0 if os.path.basename(p).upper().endswith(("_LONG.JSON", "_SHORT.JSON")) else 1,
            os.path.basename(p),
        ),
    )


def compatible_quote_suffixes(
    symbol: object,
    quote: object = None,
    *,
    default_usd: tuple[str, ...] = ("USDC", "USDT", "USD"),
) -> tuple[str, ...]:
    """Return quote suffixes that share the same currency valuation family as ``symbol`` / ``quote``."""
    q_str = quote.strip().upper() if isinstance(quote, str) and quote.strip().isalnum() else ""
    if not q_str and isinstance(symbol, str) and symbol.strip():
        parts = split_symbol_assets(symbol.strip().upper())
        if parts is not None and parts[1]:
            q_str = parts[1]
    return default_usd if (not q_str or q_str in _USD_FAMILY_QUOTES) else (q_str,)


def cache_symbol_variants(
    symbol: object,
    *,
    include_original: bool = False,
    include_base: bool = False,
    quotes: tuple[str, ...] = ("USDC", "USDT", "USD", "EUR"),
) -> list:
    """Return ordered cross-venue symbol candidates for cache lookup."""
    if not isinstance(symbol, str) or not symbol.strip():
        return [symbol] if include_original else []
    raw = symbol.strip().upper()
    cleaned = clean_symbol_key(raw)
    variants: list = []

    def _add(cand: object) -> None:
        if cand and cand not in variants:
            variants.append(cand)

    if include_original:
        variants.append(symbol)
        _add(raw)
        _add(cleaned)
    else:
        _add(cleaned)
        _add(raw)
    if cleaned != "USDCUSDT":
        base = normalize_cache_base_asset(symbol)
        bases = list(dict.fromkeys([base, *_CROSS_VENUE_BASE_EQUIVALENTS.get(base, ())])) if base else []
        sym_parts = split_symbol_assets(cleaned)
        sym_quote = sym_parts[1] if sym_parts is not None else ""
        if sym_quote and sym_quote in quotes:
            for b in bases:
                _add(f"{b}{sym_quote}")
        for suffix in quotes:
            for b in bases:
                _add(f"{b}{suffix}")
        include_usd_equities = any(q in _USD_FAMILY_QUOTES for q in quotes)
        for b in bases:
            if include_base:
                _add(b)
            if include_usd_equities:
                for xst_cand in (
                    f"{b}_US_EQ", f"{b}x.T", f"{b}X.T", f"{b}X.T/USD",
                    f"{b}X.TUSD", f"{b}x/USD", f"{b}xUSD", f"{b}/USDC", f"{b}/USD",
                ):
                    _add(xst_cand)
    return variants


def filter_candidates_by_asset_class(candidates: object, *, is_equity: bool) -> list[str]:
    """Filter symbol candidates to match either equity or crypto asset-class conventions."""
    if candidates is None or isinstance(candidates, (str, bytes, bool)):
        return []
    if is_equity:
        return [
            c
            for c in candidates
            if isinstance(c, str) and (has_equity_suffix(c) or split_symbol_assets(c) is None)
        ]
    return [c for c in candidates if isinstance(c, str) and not has_equity_suffix(c)]


def safe_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to finite float, rejecting booleans, None, non-finite values, and non-numeric objects."""
    if val is None or isinstance(val, bool):
        return default
    if not isinstance(val, (int, float, str, bytes, bytearray, numbers.Real)) and type(val).__name__ != "Decimal":
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def clean_finite_floats(values: object) -> list[float]:
    """Filter an iterable into a list of finite floats, rejecting None, booleans, strings, and non-iterables."""
    if values is None or isinstance(values, (bool, str, bytes, bytearray)):
        return []
    try:
        return [vf for x in values if (vf := safe_float(x, default=None)) is not None]
    except TypeError:
        return []


def safe_pos_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to a strictly positive finite float, otherwise return ``default``."""
    f = safe_float(val, default=None)
    return f if (f is not None and f > 0.0) else default


def safe_nonneg_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to a non-negative finite float (>= 0.0), otherwise return ``default``."""
    f = safe_float(val, default=None)
    return f if (f is not None and f >= 0.0) else default


def _coerce_int(val: object, default: int | None, *, exact: bool) -> int | None:
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, (str, bytes, bytearray)):
        try:
            return int(val, 10)
        except (TypeError, ValueError):
            pass
    f = safe_float(val, default=None)
    if f is None or (exact and not f.is_integer()):
        return default
    try:
        return int(f)
    except (ValueError, OverflowError):
        return default


def _bind_bounded_int(name: str, min_val: int, default_val: int, *, exact: bool):
    def _coercer(val: object, default: int | None = default_val) -> int | None:
        i = _coerce_int(val, None, exact=exact)
        return i if (i is not None and i >= min_val) else default

    _coercer.__name__ = _coercer.__qualname__ = name
    return _coercer


def safe_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to an integer, rejecting booleans, None, non-finite values, and non-numeric objects."""
    return _coerce_int(val, default, exact=False)


safe_pos_int = _bind_bounded_int("safe_pos_int", 1, 1, exact=False)
safe_nonneg_int = _bind_bounded_int("safe_nonneg_int", 0, 0, exact=False)


def exact_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to an exact integer without truncating fractional floats or strings."""
    return _coerce_int(val, default, exact=True)


exact_pos_int = _bind_bounded_int("exact_pos_int", 1, 1, exact=True)
exact_nonneg_int = _bind_bounded_int("exact_nonneg_int", 0, 0, exact=True)


def _parse_bool_flag(val, default: bool | None = False):
    fb = None if default is None else (default if isinstance(default, bool) else _parse_bool_flag(default, False))
    if isinstance(val, bool):
        return val
    if val is None or isinstance(val, (dict, list, tuple, set, frozenset)):
        return fb
    if isinstance(val, (int, float)):
        return fb if (isinstance(val, float) and not math.isfinite(val)) else (val != 0)
    if isinstance(val, str):
        s = val.split("#", 1)[0].split(";", 1)[0].strip().strip('"').strip("'").strip().lower()
        if not s or s in {"none", "null"}:
            return fb
        if s == "nan":
            return False if fb is not None else None
        if s in {"1", "1.0", "true", "yes", "on", "y", "t", "da"}:
            return True
        if s in {"0", "0.0", "false", "no", "off", "n", "f"}:
            return False
        fv = safe_float(s, default=None)
        return (fv != 0.0) if fv is not None else fb
    return fb


def sync_module_reexports(
    source_globals: dict[str, object],
    target_module_names: tuple[str, ...],
    allowed_origins: tuple[str, ...],
) -> None:
    """Keep loaded re-export modules bound to active callables in ``source_globals`` across reload."""
    for mod_name in target_module_names:
        reexport_mod = sys.modules.get(mod_name)
        if reexport_mod is None:
            continue
        for attr_name, attr_val in list(reexport_mod.__dict__.items()):
            if (
                not attr_name.startswith("__")
                and attr_name in source_globals
                and callable(attr_val)
                and getattr(attr_val, "__module__", None) in allowed_origins
            ):
                setattr(reexport_mod, attr_name, source_globals[attr_name])


def _sync_cachepaths_reexports() -> None:
    """Keep loaded re-export modules bound to active ``cachepaths`` callables across reload."""
    sync_module_reexports(
        globals(),
        (
            "botcore",
            "core.math_coercion",
            "core.config",
            "core.http",
            "core.time_market",
            "core.heartbeat_runtime",
            "state_io",
            "intelligence.macro_sentiment_ssot",
        ),
        (__name__,),
    )


_sync_cachepaths_reexports()


if __name__ == "__main__":
    organize_cachedb_subdomains()
