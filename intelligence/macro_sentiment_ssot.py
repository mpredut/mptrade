"""Single Source of Truth (SSOT) for Macro Sentiment, Geopolitics, Specialized Feeds, Dynamic Lexicon & Shared Primitives.

Eliminates hardcoded URLs, keyword tuples, cache filenames, session schedules, and duplicated coercion/daemon
boilerplate across ``intelligence/`` and ``llm_order_guard.py`` by:
1. Loading declarative settings from ``macro_intelligence.conf`` (with hot-reload on mtime change).
2. Providing shared coercion, cache-freshness, object/dict attribute extraction, symbol detection, bounded thread
   execution, and heartbeat inspection primitives.
3. Fetching from curated specialized editorial RSS feeds + dynamically discovering emerging crisis terms in
   ``cachedb/macro_lexicon_collect.json``.
4. Resolving global trading sessions (Europe / US / Asia) and context-adaptive dynamic evaluation intervals.
5. Providing unified cached telemetry & assessment loaders across all consumers.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
import re
import threading
import time
import urllib.parse
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from botcore import (
    _parse_bool_flag,
    fetch_public_json as _fetch_public_json,
    parse_dotenv,
    safe_float,
    safe_int,
    safe_nonneg_float,  # re-exported for intelligence.sentiment.guards
    safe_nonneg_float as safe_nonneg_ts,
    safe_pos_float,
)

fetch_public_json = _fetch_public_json
from cachepaths import cache_path, resolve_cachedb_filepath
from providers.base import normalize_side
from state_io import (
    atomic_write_json,
    discard_generation_file,
    read_json_dict,
    read_json_dict_cached,
    save_json_state,
)

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MACRO_CONFIG_PATH = os.path.join(_ROOT_DIR, "macro_intelligence.conf")
_CONFIG_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "parser": None}


# ============================================================================
# 0. Universal Coercion, Cache Freshness, Attribute & Daemon Primitives
# ============================================================================

def safe_thresh(val: Any, default: int) -> int:
    """Safely coerce sentiment index/threshold to an integer in [0, 100]."""
    f = safe_float(val, -1.0)
    return int(round(f)) if 0.0 <= f <= 100.0 else default


def clean_scale(
    val: Any,
    default: float = 1.0,
    *,
    allow_percent: bool = False,
    allow_percentage: bool = False,
    min_positive: bool = False,
) -> float:
    """Normalize a scale factor into [0.0, 1.0], optionally converting (1.0, 100.0] or '%' percentages."""
    allow_pct = allow_percent or allow_percentage
    if allow_pct and isinstance(val, str) and val.strip().endswith("%"):
        pct = safe_nonneg_float(val.strip()[:-1].strip(), default=None)
        f = round(pct / 100.0, 4) if (pct is not None and pct <= 100.0) else -1.0
    else:
        f = safe_float(val, -1.0)
        if allow_pct and 1.0 < f <= 100.0:
            f = round(f / 100.0, 4)
    if f < 0.0 or (min_positive and f <= 0.0):
        return default
    return f if 0.0 <= f <= 1.0 else default


def resolve_now_ts(now_ts: Any = None) -> float:
    """Return ``now_ts`` if it is a valid positive finite timestamp, otherwise ``time.time()``."""
    f = safe_pos_float(now_ts, -1.0)
    return f if f > 0.0 else time.time()


def is_cache_fresh(
    last_ts: Any,
    now_ts: float,
    ttl_sec: Any,
    *,
    force_refresh: Any = False,
) -> bool:
    """Return True if cached state at ``last_ts`` is still within ``ttl_sec`` at ``now_ts``."""
    if _parse_bool_flag(force_refresh, False):
        return False
    lt = safe_nonneg_ts(last_ts, 0.0)
    ttl = safe_nonneg_ts(ttl_sec, 0.0)
    return lt > 0.0 and 0.0 <= (now_ts - lt) < ttl


def is_interval_due(
    current_ts: float,
    last_ts: Any,
    interval_sec: Any,
    *,
    force: Any = False,
) -> bool:
    """Return True if ``force`` is truthy or ``interval_sec`` has elapsed since ``last_ts``."""
    if _parse_bool_flag(force, False):
        return True
    lt = safe_nonneg_ts(last_ts, 0.0)
    itv = safe_nonneg_float(interval_sec, 0.0)
    return not (0.0 <= (current_ts - lt) < itv)


def resolve_order_notional(
    notional_eur: Any,
    price: Any,
    qty: Any,
    *,
    default: Optional[float] = None,
    fallback_on_nonfinite: bool = True,
) -> Optional[float]:
    """Resolve positive order notional from explicit ``notional_eur`` or ``price * qty`` fallback."""
    n_clean = safe_pos_float(notional_eur, 0.0)
    if n_clean > 0.0:
        return n_clean
    if not fallback_on_nonfinite and notional_eur is not None and safe_float(notional_eur, None) is None:
        return default
    px_clean = safe_pos_float(price, 0.0)
    qty_clean = safe_pos_float(qty, 0.0)
    if px_clean > 0.0 and qty_clean > 0.0:
        return px_clean * qty_clean
    return default


def extract_high_severity_headlines_and_digest(
    news_snapshot: Any,
    max_digest_items: int = 12,
) -> Tuple[List[Any], str]:
    """Extract high-severity headlines list and canonical digest string from a NewsFeedSnapshot."""
    raw_hl = obj_attr(news_snapshot, "headlines", None) if news_snapshot is not None else None
    high_sev = (
        [h for h in raw_hl if _parse_bool_flag(obj_attr(h, "is_high_severity", False), False)]
        if isinstance(raw_hl, (list, tuple))
        else []
    )
    digest = " || ".join(str(obj_attr(h, "title", None) or "") for h in high_sev[:max_digest_items])
    return high_sev, digest


def obj_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None:
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if val is None else val


def obj_float(obj: Any, key: str, default: float = 0.0) -> float:
    """Extract ``key`` from ``obj`` (dict or object) and coerce to finite float."""
    return safe_float(obj_attr(obj, key, None), default)


def normalize_order_side(side: Any) -> str:
    """Normalize order side aliases ('BUY'/'B'/'BID'/'LONG' -> 'BUY', 'SELL'/'S'/'ASK'/'SHORT' -> 'SELL')."""
    return normalize_side(side)


def clean_symbol_key(symbol: Any) -> str:
    """Return uppercase alphanumeric symbol key stripped of '/', '-', '_'."""
    if symbol is None or isinstance(symbol, bool):
        return ""
    return str(symbol).strip().upper().replace("/", "").replace("-", "").replace("_", "")


def parse_symbol_list(symbols: Any) -> List[str]:
    """Normalize comma-separated string or iterable of symbols into uppercase list."""
    raw = (
        symbols.split(",")
        if isinstance(symbols, str)
        else (symbols if isinstance(symbols, (list, tuple, set, frozenset)) else [])
    )
    return [
        s.strip().upper()
        for s in raw
        if isinstance(s, str) and s.strip()
    ]


def detect_active_binance_symbols(
    cache_dir: str = "cachedb",
    *,
    include_kraken_cache: bool = False,
    default: Tuple[str, ...] = ("BTCUSDC", "TAOUSDC"),
) -> List[str]:
    """Detect enabled symbols from ``instruments.conf`` plus optional Kraken reference pairs."""
    out: List[str] = []
    seen_bases: Set[str] = set()
    try:
        from cachepaths import canonical_base_asset
    except Exception:
        canonical_base_asset = lambda x: str(x).strip().upper()
    if include_kraken_cache:
        try:
            from instrument_registry import tracked_price_instruments
            all_syms, _, _ = tracked_price_instruments()
            for s in all_syms:
                s_clean = str(s).strip().upper()
                if s_clean and not s_clean.endswith(("_EQ", "_US_EQ")) and s_clean not in out:
                    out.append(s_clean)
                    seen_bases.add(canonical_base_asset(s_clean))
        except Exception:
            pass
    if not out:
        try:
            from instruments_config import binance_symbols
            for s in binance_symbols() or []:
                s_clean = str(s).strip().upper()
                if s_clean and s_clean not in out:
                    out.append(s_clean)
                    seen_bases.add(canonical_base_asset(s_clean))
        except Exception:
            pass
    if include_kraken_cache:
        try:
            krk_cfg = os.path.join(_ROOT_DIR, "kraken", "config.env")
            if os.path.exists(krk_cfg):
                parsed = parse_dotenv(krk_cfg)
                val = parsed.get("KRAKEN_CACHE_PAIRS", "")
                if val:
                    for item in val.split(","):
                        p = item.strip().strip("'\"").upper()
                        base = canonical_base_asset(p)
                        if p and base and base not in seen_bases and p not in out:
                            out.append(p)
                            seen_bases.add(base)
        except Exception:
            pass
    return out if out else list(default)


def check_cooldown_gate(
    cooldown_map: Dict[Any, float],
    key: Any,
    now_ts: float,
    cooldown_sec: float = 1800.0,
    max_entries: int = 1000,
) -> bool:
    """Return True and record ``now_ts`` if ``key`` is not within ``cooldown_sec``."""
    last = cooldown_map.get(key, -1e18)
    if 0.0 <= (now_ts - last) < cooldown_sec:
        return False
    cooldown_map[key] = now_ts
    while len(cooldown_map) > max_entries:
        cooldown_map.pop(next(iter(cooldown_map)), None)
    return True


def run_step_bounded(
    fn: Callable[..., Any],
    *args: Any,
    timeout_sec: float = 15.0,
    label: str = "telemetry_step",
    logger_inst: Optional[logging.Logger] = None,
) -> bool:
    """Execute ``fn(*args)`` in a daemon thread so a hung socket/DNS read cannot freeze the daemon."""
    log = logger_inst or logger
    timeout_f = safe_pos_float(timeout_sec, 15.0)
    outcome: List[bool] = [False]
    err_holder: List[BaseException] = []

    def _target() -> None:
        try:
            outcome[0] = bool(fn(*args))
        except BaseException as exc:
            err_holder.append(exc)

    t = threading.Thread(target=_target, name=f"step-{label}", daemon=True)
    t.start()
    t.join(timeout=timeout_f)
    if t.is_alive():
        log.warning("Step '%s' timed out after %.1fs; skipping to preserve heartbeat.", label, timeout_f)
        return False
    if err_holder:
        log.warning("Step '%s' raised error: %s", label, err_holder[0])
        return False
    return outcome[0]


def sanitize_heartbeat_status(status: Any) -> Tuple[float, Dict[str, Any]]:
    """Sanitize a daemon cycle status dict, replacing non-finite floats with None."""
    raw = status if isinstance(status, dict) else {}
    hb_ts = safe_pos_float(raw.get("ts"), time.time())
    safe_status = {
        k: (hb_ts if k == "ts" else (None if isinstance(v, float) and safe_float(v, default=None) is None else v))
        for k, v in raw.items()
    }
    return hb_ts, safe_status


def inspect_daemon_heartbeat(
    cache_dir: str,
    filenames: Sequence[str],
    *,
    max_alive_age_sec: float = 120.0,
    include_symbols: bool = False,
) -> Dict[str, Any]:
    """Inspect a background daemon's heartbeat file(s) and check PID liveness."""
    data = None
    for fname in filenames:
        cand_data = read_json_dict(os.path.join(cache_dir, fname), default=None, recover_previous=True)
        if isinstance(cand_data, dict):
            data = cand_data
            break
    if data is None:
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        pid = safe_int(data.get("pid", 0), 0)
        ts = safe_float(data.get("ts", 0.0), 0.0)
        age = max(0.0, time.time() - ts) if ts > 0.0 else float("inf")
        is_alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                is_alive = True
            except PermissionError:
                is_alive = True
            except (OSError, OverflowError):
                is_alive = False

        status_str = "alive" if (is_alive and age < max_alive_age_sec) else ("stale" if is_alive else "dead")
        res: Dict[str, Any] = {
            "running": is_alive and age < max_alive_age_sec,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(safe_float(age, default=None), 1) if safe_float(age, default=None) is not None else None,
            "intervals": data.get("intervals") if isinstance(data.get("intervals"), dict) else {},
            "last_run": data.get("last_run") if isinstance(data.get("last_run"), dict) else {},
        }
        if include_symbols:
            raw_syms = data.get("symbols")
            res["symbols"] = (
                [str(s) for s in raw_syms if s is not None and not isinstance(s, bool)]
                if isinstance(raw_syms, list)
                else []
            )
        return res
    except Exception as exc:
        return {"running": False, "error": str(exc)}


def write_daemon_heartbeat_files(
    paths: Sequence[str],
    payload: Dict[str, Any],
    *,
    use_save_json_state: bool = False,
    logger_inst: Optional[logging.Logger] = None,
) -> None:
    """Atomically write daemon heartbeat JSON to ``paths`` with direct-write fallback."""
    log = logger_inst or logger
    if use_save_json_state:
        try:
            for raw_target in paths:
                save_json_state(raw_target, payload, indent=2)
        except Exception as exc:
            log.debug("Failed writing daemon heartbeat: %s", exc)
        return

    for raw_target in paths:
        hb_target = str(resolve_cachedb_filepath(raw_target))
        try:
            atomic_write_json(hb_target, payload, indent=2)
        except Exception as exc:
            log.warning("Primary atomic heartbeat write failed for %s: %s; attempting fallback", hb_target, exc)
            try:
                discard_generation_file(hb_target)
                atomic_write_json(hb_target, payload, indent=2)
            except Exception as fallback_err:
                log.error("Fallback heartbeat write failed for %s: %s", hb_target, fallback_err)


def prune_automated_cli_threads(logger_inst: Optional[logging.Logger] = None) -> int:
    """Prune automated CLI threads older than configured retention limit."""
    log = logger_inst or logger
    try:
        from botcore import required_float_env
        from orchestratorOS.admin.prune_llm_threads import prune_old_cli_threads
        from orchestratorOS.agy.paths import resolve_bot_cli_base

        retention_d = required_float_env("CLI_THREAD_RETENTION_DAYS")
        bot_cli_base = resolve_bot_cli_base()
        pruned_cnt, _ = prune_old_cli_threads(cli_base=bot_cli_base, retention_days=retention_d)
        if pruned_cnt > 0:
            log.info(
                "Pruned %d automated CLI threads older than %.1f days from %s",
                pruned_cnt,
                retention_d,
                bot_cli_base,
            )
        return pruned_cnt
    except Exception as exc:
        log.debug("Automatic CLI thread pruning error: %s", exc)
        return 0


class ExternalTelemetryMixin:
    """Shared mixin for Orderbook, Derivatives, and Whale collectors across telemetry daemons."""

    cache_dir: str
    orderbook_interval_sec: float
    derivatives_interval_sec: float
    whale_interval_sec: float
    orderbook_collector: Any
    derivatives_collector: Any
    whale_collector: Any
    _telemetry_logger: logging.Logger

    def _init_external_collectors(self, logger_inst: logging.Logger) -> None:
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector

        self._telemetry_logger = logger_inst
        self.orderbook_collector = OrderbookDepthCollector(
            cache_ttl_sec=max(5.0, self.orderbook_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=60.0,
        )
        self.derivatives_collector = DerivativesTelemetryCollector(
            cache_ttl_sec=max(10.0, self.derivatives_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=120.0,
        )
        self.whale_collector = WhalePositioningCollector(
            cache_ttl_sec=max(15.0, self.whale_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=300.0,
        )

    @staticmethod
    def _run_step_bounded(
        fn: Callable[..., Any],
        *args: Any,
        timeout_sec: float = 15.0,
        label: str = "telemetry_step",
    ) -> bool:
        return run_step_bounded(fn, *args, timeout_sec=timeout_sec, label=label)

    def _step_external_symbols(
        self,
        current_ts: float,
        force: bool,
        results: Dict[str, Any],
        *,
        bounded: bool = False,
        on_symbol_done: Optional[Callable[[], None]] = None,
    ) -> None:
        if getattr(self, "skip_external", False):
            return
        steps = (
            ("orderbook_updated", "_last_orderbook_ts", self.orderbook_interval_sec, self.update_orderbook, 10.0, "ob"),
            ("derivatives_updated", "_last_derivatives_ts", self.derivatives_interval_sec, self.update_derivatives, 12.0, "deriv"),
            ("whale_updated", "_last_whale_ts", self.whale_interval_sec, self.update_whale, 15.0, "whale"),
        )
        for sym in getattr(self, "symbols", ()):
            for res_key, ts_attr, interval_sec, fn, timeout_s, lbl_prefix in steps:
                ts_map: Dict[str, float] = getattr(self, ts_attr)
                if is_interval_due(current_ts, ts_map.get(sym, 0.0), interval_sec, force=force):
                    ok = (
                        self._run_step_bounded(fn, sym, timeout_sec=timeout_s, label=f"{lbl_prefix}_{sym}")
                        if bounded
                        else fn(sym)
                    )
                    if ok:
                        results[res_key] += 1
                        ts_map[sym] = current_ts
            if on_symbol_done is not None:
                on_symbol_done()

    def update_orderbook(self, symbol: str) -> bool:
        """Fetch spot orderbook depth and update disk snapshot."""
        log = getattr(self, "_telemetry_logger", logger)
        try:
            snap = self.orderbook_collector.fetch(symbol, allow_network=True)
            if snap:
                log.debug(
                    "[%s] Orderbook updated: mid=%.2f, bid_depth=$%.0f, ask_depth=$%.0f, imbalance=%.2f",
                    symbol,
                    obj_float(snap, "mid_price"),
                    obj_float(snap, "bid_depth_usd"),
                    obj_float(snap, "ask_depth_usd"),
                    obj_float(snap, "imbalance_ratio"),
                )
                return True
        except Exception as exc:
            log.warning("[%s] Failed updating orderbook: %s", symbol, exc)
        return False

    def update_derivatives(self, symbol: str) -> bool:
        """Fetch perpetual derivatives funding rate & OI and update disk snapshot."""
        log = getattr(self, "_telemetry_logger", logger)
        try:
            tel = self.derivatives_collector.fetch(symbol, allow_network=True)
            if tel:
                log.debug(
                    "[%s] Derivatives updated: funding=%.6f, OI=$%.0f",
                    symbol,
                    obj_float(tel, "funding_rate"),
                    obj_float(tel, "open_interest_usd"),
                )
                return True
        except Exception as exc:
            log.warning("[%s] Failed updating derivatives: %s", symbol, exc)
        return False

    def update_whale(self, symbol: str) -> bool:
        """Fetch whale positioning & OI divergence and update disk snapshot."""
        log = getattr(self, "_telemetry_logger", logger)
        try:
            snap = self.whale_collector.fetch(symbol, allow_network=True)
            if snap:
                log.debug(
                    "[%s] Whale positioning updated: top_ratio=%.2f, taker_ratio=%.2f, regime=%s",
                    symbol,
                    obj_float(snap, "top_traders_long_ratio"),
                    obj_float(snap, "taker_buy_sell_ratio"),
                    obj_attr(snap, "divergence_regime", "NEUTRAL"),
                )
                return True
        except Exception as exc:
            log.warning("[%s] Failed updating whale positioning: %s", symbol, exc)
        return False


# ============================================================================
# 1. Config-Driven Taxonomies, Filenames, Endpoints & Interval Bounds
# ============================================================================

def _split_csv(raw: str, *, lower: bool = False, upper: bool = False) -> Tuple[str, ...]:
    items: List[str] = []
    for part in re.split(r"[,\n]+", raw or ""):
        cleaned = part.strip()
        if not cleaned:
            continue
        cleaned = cleaned.lower() if lower else (cleaned.upper() if upper else cleaned)
        if cleaned not in items:
            items.append(cleaned)
    return tuple(items)


def _parse_kv_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in _split_csv(raw, upper=True):
        if ":" in item:
            k, v = item.split(":", 1)
            if k.strip() and v.strip():
                mapping[k.strip()] = v.strip()
    return mapping


def load_macro_intelligence_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    """Load ``macro_intelligence.conf`` with automatic mtime-based hot reload."""
    target = config_path or os.environ.get("MACRO_INTELLIGENCE_CONF") or DEFAULT_MACRO_CONFIG_PATH
    try:
        mtime = os.path.getmtime(target) if os.path.exists(target) else 0.0
    except OSError:
        mtime = 0.0

    cached_parser = _CONFIG_CACHE.get("parser")
    if cached_parser is not None and _CONFIG_CACHE.get("path") == target and _CONFIG_CACHE.get("mtime") == mtime:
        return cached_parser

    parser = configparser.ConfigParser(interpolation=None)
    if os.path.exists(target):
        try:
            parser.read(target, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed reading %s: %s", target, exc)
    _CONFIG_CACHE.update(path=target, mtime=mtime, parser=parser)
    return parser


def _cfg_str(section: str, key: str, fallback: str = "") -> str:
    cfg = load_macro_intelligence_config()
    return cfg.get(section, key).strip() if cfg.has_option(section, key) else fallback


def _cfg_float(section: str, key: str, fallback: float) -> float:
    return safe_float(_cfg_str(section, key, ""), fallback)


def _cfg_int(section: str, key: str, fallback: int) -> int:
    return safe_int(_cfg_str(section, key, ""), fallback)


def get_market_biases() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "market_biases"), upper=True)


def get_risk_levels() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "risk_levels"), upper=True)


def get_recommended_actions() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "recommended_actions"), upper=True)


def get_breadth_regimes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "breadth_regimes"), upper=True)


def get_threat_levels() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "threat_levels"), upper=True)


def get_recommended_brakes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "recommended_brakes"), upper=True)


MARKET_BIASES: Tuple[str, ...] = get_market_biases()
RISK_LEVELS: Tuple[str, ...] = get_risk_levels()
RECOMMENDED_ACTIONS: Tuple[str, ...] = get_recommended_actions()
BREADTH_REGIMES: Tuple[str, ...] = get_breadth_regimes()
THREAT_LEVELS: Tuple[str, ...] = get_threat_levels()
RECOMMENDED_BRAKES: Tuple[str, ...] = get_recommended_brakes()

DEFAULT_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "default_sentiment_interval_sec", 10800.0)
MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "min_dynamic_sentiment_interval_sec", 1800.0)
MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "max_dynamic_sentiment_interval_sec", 16200.0)
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = _cfg_float("intervals", "sentiment_checkpoint_cooldown_sec", 2700.0)

DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "default_geopolitical_interval_sec", 16200.0)
MIN_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "min_dynamic_geopolitical_interval_sec", 900.0)
MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "max_dynamic_geopolitical_interval_sec", 21600.0)
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = _cfg_float("intervals", "default_news_rolling_window_sec", 14400.0)
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = _cfg_float("intervals", "default_fallback_retry_ttl_sec", 120.0)
DYNAMIC_LEXICON_TTL_SEC: float = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)


def get_fear_greed_endpoint_url() -> str:
    return _cfg_str("feeds", "fear_greed_endpoint_url")


def get_market_breadth_endpoint_url() -> str:
    return _cfg_str("feeds", "market_breadth_endpoint_url")


def get_breadth_excluded_bases() -> Set[str]:
    return set(_split_csv(_cfg_str("taxonomies", "breadth_excluded_bases"), upper=True))


def get_breadth_quote_suffixes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "breadth_quote_suffixes"), upper=True)


def get_cache_filenames(group_key: str) -> Tuple[str, ...]:
    return _split_csv(_cfg_str("cache_files", group_key))


def _normalize_taxonomy(
    value: Any,
    allowed: Tuple[str, ...],
    alias_cfg_key: str,
    default_idx: int,
    fallback_str: str,
    default: Optional[str] = None,
) -> str:
    fallback = default if (default and default in allowed) else (allowed[default_idx] if len(allowed) > default_idx else fallback_str)
    cleaned = str(value or "").strip().upper()
    if cleaned in allowed:
        return cleaned
    return _parse_kv_map(_cfg_str("taxonomies", alias_cfg_key)).get(cleaned, fallback)


def normalize_market_bias(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, get_market_biases(), "bias_aliases", 1, "NEUTRAL", default)


def normalize_risk_level(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, get_risk_levels(), "risk_aliases", 1, "MODERATE", default)


def normalize_recommended_action(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, get_recommended_actions(), "action_aliases", 1, "HOLD", default)


def normalize_threat_level(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, get_threat_levels(), "threat_aliases", 0, "NORMAL", default)


def normalize_recommended_brake(value: Any, threat_level: Optional[str] = None) -> str:
    brakes = get_recommended_brakes()
    cleaned = str(value or "").strip().upper()
    if cleaned in brakes:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "brake_aliases"))
    if cleaned in aliases:
        return aliases[cleaned]
    levels = get_threat_levels()
    norm_threat = normalize_threat_level(threat_level)
    if len(levels) >= 3 and len(brakes) >= 3:
        if norm_threat == levels[2]:
            return brakes[2]
        if norm_threat == levels[1]:
            return brakes[1]
    return brakes[0] if brakes else "NONE"


# ============================================================================
# 2. Specialized Multi-Source Feeds & Dynamic Crisis Lexicon Discovery
# ============================================================================

_LEXICON_FILE_CACHE: Dict[str, Tuple[tuple, Dict[str, Any]]] = {}
_LEXICON_EVAL_CACHE: Dict[Tuple[Any, ...], Dict[str, Tuple[str, ...]]] = {}


def _dynamic_lexicon_path(cache_dir: Optional[str] = None) -> str:
    fname = _cfg_str("cache_files", "dynamic_lexicon_file") or "macro_lexicon_collect.json"
    return str(resolve_cachedb_filepath(os.path.join(cache_dir, fname) if cache_dir else cache_path(fname)))


def load_dynamic_lexicon(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Dict[str, Tuple[str, ...]]:
    """Load unexpired dynamically discovered high-severity and acute-shock terms from cachedb."""
    now = resolve_now_ts(now_ts)
    ttl = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)
    empty_res: Dict[str, Tuple[str, ...]] = {
        "high_severity_terms": (),
        "acute_shock_phrases": (),
        "dynamic_severity_keywords": (),
        "dynamic_acute_shock_phrases": (),
    }
    try:
        data = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            return empty_res
        file_sig = (_LEXICON_FILE_CACHE.get(path) or (None,))[0]
        eval_key = (path, file_sig, id(data), int(now), ttl)
        cached_eval = _LEXICON_EVAL_CACHE.get(eval_key)
        if cached_eval is not None:
            return cached_eval
        high_seen: Dict[str, None] = {}
        acute_seen: Dict[str, None] = {}
        for term, meta in entries.items():
            if not isinstance(term, str) or not term.strip():
                continue
            t_clean = term.strip().lower()
            ts = safe_float(meta.get("ts") if isinstance(meta, dict) else meta, 0.0)
            tier = str(meta.get("tier", "high")).lower() if isinstance(meta, dict) else "high"
            if ts > 0.0 and (now - ts) <= ttl:
                high_seen.setdefault(t_clean, None)
                if tier == "acute":
                    acute_seen.setdefault(t_clean, None)
        high_tup, acute_tup = tuple(high_seen), tuple(acute_seen)
        res = {
            "high_severity_terms": high_tup,
            "acute_shock_phrases": acute_tup,
            "dynamic_severity_keywords": high_tup,
            "dynamic_acute_shock_phrases": acute_tup,
        }
        if len(_LEXICON_EVAL_CACHE) >= 32:
            _LEXICON_EVAL_CACHE.clear()
        _LEXICON_EVAL_CACHE[eval_key] = res
        return res
    except Exception as exc:
        logger.debug("Could not load dynamic lexicon from %s: %s", path, exc)
        return empty_res


def update_dynamic_lexicon(
    new_high_terms: Iterable[str],
    new_acute_terms: Iterable[str] = (),
    *,
    cache_dir: Optional[str] = None,
    source: str = "feed_discovery",
    now_ts: Optional[float] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Persist newly discovered crisis terms into ``cachedb/macro_lexicon_collect.json``."""
    now = resolve_now_ts(now_ts)
    ttl = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)

    existing_entries: Dict[str, Dict[str, Any]] = {}
    raw = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
    if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
        for k, v in raw["entries"].items():
            if isinstance(k, str) and isinstance(v, dict):
                ts = safe_float(v.get("ts"), 0.0)
                if ts > 0.0 and (now - ts) <= ttl:
                    existing_entries[k.strip().lower()] = dict(v)

    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))
    changed = False
    for raw_t, tier_override in (
        *[(t, None) for t in new_high_terms],
        *[(a, "acute") for a in new_acute_terms],
    ):
        t = re.sub(r"\s+", " ", str(raw_t or "").strip().lower())
        if len(t) < 4 or t in stopwords:
            continue
        tier = tier_override or existing_entries.get(t, {}).get("tier", "high")
        existing_entries[t] = {"ts": now, "tier": tier, "source": source}
        changed = True

    if changed:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_json_state(path, {"updated_ts": now, "entries": existing_entries}, indent=2)
        except Exception as exc:
            logger.debug("Could not save dynamic lexicon to %s: %s", path, exc)
        _LEXICON_FILE_CACHE.clear()
        _LEXICON_EVAL_CACHE.clear()

    return load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now)


def get_active_severity_keywords(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    seed = _split_csv(_cfg_str("lexicon_seed", "high_severity_terms"), lower=True)
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts).get("high_severity_terms", ())
    return tuple(dict.fromkeys((*seed, *dyn)))


def get_active_acute_shock_phrases(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    seed = _split_csv(_cfg_str("lexicon_seed", "acute_shock_phrases"), lower=True)
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts).get("acute_shock_phrases", ())
    return tuple(dict.fromkeys((*seed, *dyn)))


def discover_dynamic_terms_from_headlines(
    headlines: Sequence[Any],
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
    persist: bool = True,
) -> Tuple[str, ...]:
    """Extract emerging crisis bigrams/topics that surge across specialized feeds."""
    crisis_stems = set(_split_csv(_cfg_str("lexicon_seed", "crisis_context_stems"), lower=True))
    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))
    if not crisis_stems or not headlines:
        return ()

    phrase_sources: Dict[str, Set[str]] = {}
    phrase_counts: Dict[str, int] = {}

    for h in headlines:
        title = obj_attr(h, "title", "")
        source = obj_attr(h, "source", "")
        if not isinstance(title, str) or not title.strip():
            continue
        tokens = [tok for tok in re.findall(r"[a-zA-Z]{3,}", title.lower()) if tok not in stopwords]
        if len(tokens) < 2:
            continue
        if not any(any(tok.startswith(stem) or stem in tok for stem in crisis_stems) for tok in tokens):
            continue
        src_key = str(source or "unknown").strip().lower()
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
            phrase_sources.setdefault(bigram, set()).add(src_key)

    discovered = [
        phrase
        for phrase, count in phrase_counts.items()
        if count >= 2 or len(phrase_sources.get(phrase, set())) >= 2
    ]
    if discovered and persist:
        update_dynamic_lexicon(discovered, cache_dir=cache_dir, source="specialized_feed_salience", now_ts=now_ts)
    return tuple(discovered)


def record_dynamic_crisis_terms_from_assessment(
    summary: str,
    key_risks: Sequence[str] = (),
    *,
    is_acute_shock: bool = False,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Extract salient crisis phrases from AI geopolitical/sentiment assessments into dynamic lexicon."""
    crisis_stems = set(_split_csv(_cfg_str("lexicon_seed", "crisis_context_stems"), lower=True))
    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))

    extracted: List[str] = []
    for text in [str(summary or "")] + [str(r or "") for r in (key_risks or ())]:
        tokens = [tok for tok in re.findall(r"[a-zA-Z]{3,}", text.lower()) if tok not in stopwords]
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if any(w.startswith(stem) or stem in w for stem in crisis_stems for w in (w1, w2)):
                bg = f"{w1} {w2}"
                if bg not in extracted:
                    extracted.append(bg)

    if extracted:
        update_dynamic_lexicon(
            extracted,
            new_acute_terms=extracted if is_acute_shock else (),
            cache_dir=cache_dir,
            source="llm_assessment_feedback",
            now_ts=now_ts,
        )
    return tuple(extracted)


def build_dynamic_rss_url(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> str:
    """Dynamically construct the supplementary RSS query URL from active dynamic lexicon + config."""
    template = _cfg_str("feeds", "dynamic_search_rss_template")
    max_terms = max(4, _cfg_int("feeds", "dynamic_search_max_terms", 16))
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts)
    prioritized: List[str] = []
    for t in dyn.get("acute_shock_phrases", ()) + dyn.get("high_severity_terms", ()) + get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts):
        if t and t not in prioritized:
            prioritized.append(t)
        if len(prioritized) >= max_terms:
            break

    encoded_terms = [urllib.parse.quote_plus(f'"{t}"' if " " in t else t) for t in prioritized[:max_terms]]
    query_str = "+OR+".join(encoded_terms) if encoded_terms else "macro"
    return template.format(query=query_str) if "{query}" in template else template


def get_specialized_rss_urls(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    """Return curated specialized editorial RSS feeds + dynamically built crisis query feed."""
    urls = list(_split_csv(_cfg_str("feeds", "specialized_rss_urls")))
    dyn_url = build_dynamic_rss_url(cache_dir=cache_dir, now_ts=now_ts)
    if dyn_url and dyn_url not in urls:
        urls.append(dyn_url)
    return tuple(urls)


HIGH_SEVERITY_KEYWORDS: Tuple[str, ...] = get_active_severity_keywords()
ACUTE_SHOCK_PHRASES: Tuple[str, ...] = get_active_acute_shock_phrases()
DEFAULT_RSS_URL: str = build_dynamic_rss_url()


def match_high_severity_keywords(
    text: str,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Return all matched high-severity keywords for ``text`` using the active lexicon."""
    lower = (text or "").lower()
    if not lower:
        return ()
    return tuple(
        kw for kw in get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts)
        if kw in lower and re.search(r"\b" + re.escape(kw) + r"\b", lower)
    )


def has_high_severity_keywords(
    text: str,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Return True if a headline matches any active high-severity keyword."""
    return bool(match_high_severity_keywords(text, cache_dir=cache_dir, now_ts=now_ts))


def has_acute_shock_keywords(
    texts: Sequence[str],
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Return True if any headline in ``texts`` contains an active acute systemic shock phrase."""
    phrases = get_active_acute_shock_phrases(cache_dir=cache_dir, now_ts=now_ts)
    return any(any(p in (t or "").lower() for p in phrases) for t in texts)


# ============================================================================
# 3. Global Trading Session & Intraday Flow Resolver (Europe / US / Asia)
# ============================================================================

@dataclass(frozen=True)
class GlobalSessionContext:
    """Represents the active global trading session and trader behavioral flow."""
    session_key: str
    session_label: str
    eet_time_str: str
    ny_time_str: str
    utc_time_str: str
    is_weekend: bool
    is_active_western_session: bool
    is_peak_transition_window: bool
    trader_flow_profile: str
    llm_session_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "session_label": self.session_label,
            "eet_time_str": self.eet_time_str,
            "ny_time_str": self.ny_time_str,
            "utc_time_str": self.utc_time_str,
            "is_weekend": self.is_weekend,
            "is_active_western_session": self.is_active_western_session,
            "is_peak_transition_window": self.is_peak_transition_window,
            "trader_flow_profile": self.trader_flow_profile,
            "llm_session_guidance": self.llm_session_guidance,
        }


# Declarative intraday weekday session windows in EET minutes:
# (start_min, end_min, session_key, label, is_western, is_peak_transition, flow_profile, llm_guidance)
_WEEKDAY_SESSION_WINDOWS_EET: Tuple[Tuple[int, int, str, str, bool, bool, str, str], ...] = (
    (
        8 * 60,
        11 * 60,
        "EU_OPEN_ACCUMULATION",
        "European Morning Open & Accumulation (08:00-11:00 EET)",
        True,
        True,
        "European traders and desks (Frankfurt/London) enter the market and typically accumulate risk positions in the morning when breadth is constructive.",
        "European Morning Accumulation window (08:00-11:00 EET): European traders typically open and build long positions during this phase. If breadth is stabilizing or advancing and geopolitical risk is contained, favor constructive positioning (ACCUMULATE/HOLD) rather than premature profit-taking.",
    ),
    (
        11 * 60,
        15 * 60 + 30,
        "EU_MIDDAY_CONTINUATION",
        "European Midday & US Pre-Market (11:00-15:30 EET)",
        True,
        False,
        "European midday consolidation ahead of 08:30 EST (15:30 EET) US macro economic releases.",
        "European Midday & US Pre-Market (11:00-15:30 EET): Morning European accumulation is established; market awaits US macro prints and Wall Street pre-market flows.",
    ),
    (
        15 * 60 + 30,
        18 * 60,
        "US_OPEN_OVERLAP",
        "EU/US Overlap & Wall Street Open (15:30-18:00 EET)",
        True,
        True,
        "Peak global liquidity overlap between European afternoon desks and US Wall Street open (09:30 NY). Highest intraday volume, macro data reactions, and trend acceleration.",
        "EU/US Overlap & Wall Street Open (15:30-18:00 EET): Peak global volume and volatility. Weigh cross-asset dispersion and institutional whale net flows heavily, as US open either validates or reverses the European morning trend.",
    ),
    (
        18 * 60,
        20 * 60 + 30,
        "EU_EVENING_CLOSE",
        "European Evening Close & Profit-Taking (18:00-20:30 EET)",
        True,
        True,
        "European traders and funds close intraday positions, lock in profits, and reduce overnight exposure between 18:00 and 20:30 EET, frequently causing evening pullbacks or momentum exhaustion.",
        "European Evening Close window (18:00-20:30 EET): European traders typically unwind intraday positions and take profits toward the European close. Be vigilant against buying late-day tops; if sentiment is greedy or breadth is losing steam, favor TRIM_PROFITS/CAUTION or tighter risk.",
    ),
    (
        20 * 60 + 30,
        23 * 60 + 15,
        "US_AFTERNOON_CLOSE",
        "US Afternoon & Wall Street Close (20:30-23:15 EET)",
        True,
        False,
        "US institutional afternoon session leading into the 16:00 EST (23:00 EET) Wall Street close.",
        "US Afternoon & Close (20:30-23:15 EET): European desks have closed; US institutional flows and ETF settlement drive direction into 23:00 EET before overnight Asian handoff.",
    ),
)


def get_session_transition_checkpoints_eet() -> Tuple[Tuple[int, int, str], ...]:
    """Load intraday session transition checkpoints from ``macro_intelligence.conf``."""
    parsed: List[Tuple[int, int, str]] = []
    for item in _split_csv(_cfg_str("sessions", "checkpoints_eet")):
        parts = item.split(":")
        if len(parts) >= 3:
            try:
                parsed.append((int(parts[0]), int(parts[1]), parts[2].strip()))
            except ValueError:
                continue
    return tuple(parsed)


SESSION_TRANSITION_CHECKPOINTS_EET: Tuple[Tuple[int, int, str], ...] = get_session_transition_checkpoints_eet()


def _to_tz(now_ts: float, tz_name: str, fallback_offset_hours: int) -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.fromtimestamp(now_ts, tz=timezone(timedelta(hours=fallback_offset_hours)))


def resolve_global_trading_session(now_ts: Optional[float] = None) -> GlobalSessionContext:
    """Resolve current global market session and European/US/Asian trader flow dynamics."""
    ts = resolve_now_ts(now_ts)
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _to_tz(ts, _cfg_str("sessions", "primary_timezone", "Europe/Bucharest"), _cfg_int("sessions", "primary_fallback_offset_hours", 2))
    dt_ny = _to_tz(ts, _cfg_str("sessions", "secondary_timezone", "America/New_York"), _cfg_int("sessions", "secondary_fallback_offset_hours", -5))

    eet_str, ny_str, utc_str = dt_eet.strftime("%Y-%m-%d %H:%M %Z"), dt_ny.strftime("%Y-%m-%d %H:%M %Z"), dt_utc.strftime("%Y-%m-%d %H:%M UTC")
    if dt_eet.weekday() >= 5:
        return GlobalSessionContext(
            "WEEKEND_GLOBAL_CALM",
            "Weekend Crypto-Only Liquidity",
            eet_str,
            ny_str,
            utc_str,
            True,
            False,
            False,
            "Thin institutional liquidity; retail-driven crypto range trading unless geopolitical shock occurs.",
            "Weekend session: institutional desks in Europe and US are closed. Avoid chasing low-volume breakouts; require stronger confirmation for BULLISH/ACCUMULATE unless a genuine washout bounce occurs.",
        )

    minutes_eet = dt_eet.hour * 60 + dt_eet.minute
    for start_m, end_m, key, label, is_west, is_peak, flow, guidance in _WEEKDAY_SESSION_WINDOWS_EET:
        if start_m <= minutes_eet < end_m:
            return GlobalSessionContext(key, label, eet_str, ny_str, utc_str, False, is_west, is_peak, flow, guidance)

    return GlobalSessionContext(
        "ASIA_OVERNIGHT_CALM",
        "Asian Session & European Overnight Quiet (23:15-08:00 EET)",
        eet_str,
        ny_str,
        utc_str,
        False,
        False,
        False,
        "Overnight European/US quiet and Asian trading hours (Tokyo/Singapore/Hong Kong). Lower Western spot liquidity prior to the 08:00 EET European morning open.",
        "Asian Overnight Session (23:15-08:00 EET): Western desks are inactive until the 08:00 EET European morning open. Maintain steady regime discipline and avoid overreacting to low-volume overnight noise.",
    )


def did_cross_session_checkpoint(
    last_eval_ts: float,
    now_ts: float,
    min_cooldown_sec: float = SENTIMENT_CHECKPOINT_COOLDOWN_SEC,
) -> Optional[str]:
    """Return the checkpoint label if a canonical EET session transition was crossed in (last_eval_ts, now_ts]."""
    last_f = safe_pos_float(last_eval_ts, 0.0)
    now_f = safe_pos_float(now_ts, 0.0)
    cooldown_f = safe_nonneg_float(min_cooldown_sec, SENTIMENT_CHECKPOINT_COOLDOWN_SEC)
    if last_f <= 0.0 or now_f <= last_f or (now_f - last_f) < cooldown_f:
        return None

    primary_tz = _cfg_str("sessions", "primary_timezone", "Europe/Bucharest")
    p_off = _cfg_int("sessions", "primary_fallback_offset_hours", 2)
    dt_last = _to_tz(last_f, primary_tz, p_off)
    dt_now = _to_tz(now_f, primary_tz, p_off)

    candidate_days = [dt_last.date()] + ([dt_now.date()] if dt_now.date() != dt_last.date() else [])
    crossed_label: Optional[str] = None
    for day in candidate_days:
        is_weekend_day = day.weekday() >= 5
        for hour, minute, label in get_session_transition_checkpoints_eet():
            if is_weekend_day and label not in ("EU_MORNING_BUY_OPEN", "EU_EVENING_POSITION_CLOSE"):
                continue
            cp_ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=dt_now.tzinfo).timestamp()
            if last_f < cp_ts <= now_f:
                crossed_label = label
    return crossed_label


# ============================================================================
# 4. Context-Adaptive Dynamic Interval Calculators
# ============================================================================

def compute_dynamic_sentiment_interval(
    base_interval_sec: float = DEFAULT_SENTIMENT_INTERVAL_SEC,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    fear_greed: Optional[FearGreedSnapshot] = None,
    breadth: Optional[MarketBreadthSnapshot] = None,
    prev_fear_greed_val: Optional[int] = None,
    prev_advance_ratio: Optional[float] = None,
    geopolitical_threat_level: str = "NORMAL",
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_sentiment`` evaluation."""
    base = max(60.0, safe_pos_float(base_interval_sec, DEFAULT_SENTIMENT_INTERVAL_SEC))
    ts = resolve_now_ts(now_ts)

    scale = base / DEFAULT_SENTIMENT_INTERVAL_SEC
    min_interval = max(60.0, MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    max_interval = max(base, MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    checkpoint_cooldown = max(60.0, SENTIMENT_CHECKPOINT_COOLDOWN_SEC * scale)

    norm_geo = normalize_threat_level(geopolitical_threat_level)
    cached_risk = normalize_risk_level(obj_attr(cached_assessment, "risk_level", None))
    breadth_regime = str(obj_attr(breadth, "regime", "NEUTRAL") or "NEUTRAL") if breadth is not None else "NEUTRAL"
    median_abs = abs(obj_float(breadth, "median_change_pct", 0.0)) if breadth is not None else 0.0

    if norm_geo == "CRITICAL_SHOCK":
        return min_interval, "critical_geopolitical_shock_30m"
    if breadth_regime in ("PANIC_WASHOUT", "EUPHORIC_BLOWOFF") or median_abs >= 5.5 or cached_risk == "EXTREME":
        return min_interval, f"extreme_market_regime_{breadth_regime.lower()}_{cached_risk.lower()}"
    cur_fg_val = safe_int(obj_attr(fear_greed, "value", None), None)
    prev_fg_val = safe_int(prev_fear_greed_val, None)
    if cur_fg_val is not None and prev_fg_val is not None and abs(cur_fg_val - prev_fg_val) >= 8:
        return min_interval, f"fear_greed_jump_{cur_fg_val - prev_fg_val:+d}pts"
    cur_adv = safe_float(obj_attr(breadth, "advance_ratio", None), None)
    prev_adv = safe_float(prev_advance_ratio, None)
    if cur_adv is not None and prev_adv is not None and abs(cur_adv - prev_adv) >= 0.18:
        return min_interval, f"breadth_shift_{cur_adv - prev_adv:+.2f}"

    crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=checkpoint_cooldown)
    if crossed_cp is not None:
        return max(min_interval, min(base, checkpoint_cooldown)), f"session_checkpoint_{crossed_cp}"

    session_ctx = resolve_global_trading_session(ts)
    is_extreme_fg = bool(
        fear_greed is not None
        and (
            _parse_bool_flag(obj_attr(fear_greed, "is_extreme_fear", False))
            or _parse_bool_flag(obj_attr(fear_greed, "is_extreme_greed", False))
        )
    )
    if cached_risk == "HIGH" or norm_geo == "ELEVATED" or median_abs >= 3.0 or is_extreme_fg:
        return max(min_interval, base * 0.5), f"elevated_volatility_risk_{cached_risk.lower()}"
    if session_ctx.is_peak_transition_window:
        return max(min_interval, base * 0.5), f"active_flow_window_{session_ctx.session_key}"
    if session_ctx.is_active_western_session:
        return base, f"standard_western_session_{session_ctx.session_key}"
    if cached_risk in ("LOW", "MODERATE") and norm_geo == "NORMAL" and breadth_regime == "NEUTRAL" and median_abs < 2.0 and not is_extreme_fg:
        return min(max_interval, base * (4.0 / 3.0)), f"global_calm_quiet_{session_ctx.session_key}"
    return base, f"baseline_{session_ctx.session_key}"


def compute_dynamic_geopolitical_interval(
    base_interval_sec: float = DEFAULT_GEOPOLITICAL_INTERVAL_SEC,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    news_snapshot: Optional[Any] = None,
    headlines_changed: bool = False,
    fallback_retry_ttl_sec: float = DEFAULT_FALLBACK_RETRY_TTL_SEC,
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_geopolitical`` evaluation."""
    base = max(60.0, safe_pos_float(base_interval_sec, DEFAULT_GEOPOLITICAL_INTERVAL_SEC))
    ts = resolve_now_ts(now_ts)
    changed_b = _parse_bool_flag(headlines_changed)

    if cached_assessment is not None and _parse_bool_flag(obj_attr(cached_assessment, "is_fallback", False)):
        return max(30.0, safe_pos_float(fallback_retry_ttl_sec, DEFAULT_FALLBACK_RETRY_TTL_SEC)), "fallback_short_retry"

    threat_level = normalize_threat_level(obj_attr(cached_assessment, "threat_level", None))
    has_shock = _parse_bool_flag(obj_attr(news_snapshot, "has_critical_shock_keywords", False)) if news_snapshot is not None else False
    high_sev_count = safe_int(obj_attr(news_snapshot, "high_severity_count", 0), 0) if news_snapshot is not None else 0

    if has_shock and (threat_level == "NORMAL" or (threat_level == "ELEVATED" and changed_b)):
        return 0.0, "acute_shock_keyword_alert"

    if changed_b and high_sev_count > 0:
        if threat_level == "CRITICAL_SHOCK":
            return min(base, 1800.0), "critical_shock_active_monitoring_30m"
        if threat_level == "ELEVATED":
            return min(base, 3600.0), "elevated_threat_new_headlines_1h"
        if high_sev_count >= 3:
            return min(base, 2700.0), f"high_severity_headline_surge_{high_sev_count}"
        crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=2700.0)
        if crossed_cp is not None:
            return min(base, 2700.0), f"geopolitical_session_checkpoint_{crossed_cp}"

    if high_sev_count == 0 and threat_level == "NORMAL":
        session_ctx = resolve_global_trading_session(ts)
        if not session_ctx.is_active_western_session:
            return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.3333), "global_overnight_calm_6h"
        return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.15), "global_daytime_calm"

    return base, f"standard_geopolitical_ttl_{threat_level.lower()}"


# ============================================================================
# 5. Unified Cached Telemetry & Assessment Loaders (SSOT)
# ============================================================================

def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached Fear & Greed snapshot using configured candidate filenames."""
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

    eval_now = resolve_now_ts(now_ts)
    for fg_fname in get_cache_filenames("fear_greed_files"):
        fg_path = os.path.join(cache_dir, fg_fname)
        try:
            fg_data = read_json_dict(fg_path, default=None, recover_previous=True)
            if not isinstance(fg_data, dict):
                continue
            fg_val = max(0, min(100, int(round(safe_float(fg_data.get("value"), 50.0)))))
            raw_cls = fg_data.get("classification")
            if raw_cls is None or isinstance(raw_cls, bool) or not str(raw_cls).strip():
                raw_cls = fg_data.get("sentiment")
            fg_cls = str(raw_cls).strip() if (raw_cls is not None and not isinstance(raw_cls, bool) and str(raw_cls).strip()) else "Neutral"
            raw_fg_ts = fg_data.get("ts") if fg_data.get("ts") is not None else fg_data.get("timestamp")
            fg_ts = safe_pos_float(raw_fg_ts, eval_now)
            fg_t7d = int(round(safe_float(fg_data.get("trend_7d_change"), 0.0)))
            raw_hist = fg_data.get("historical_values") or ()
            hist_vals = [
                max(0, min(100, int(round(safe_float(v, -1.0)))))
                for v in (raw_hist if isinstance(raw_hist, (list, tuple)) else ())
                if safe_float(v, -1.0) >= 0.0
            ]
            return FearGreedSnapshot(
                value=fg_val,
                classification=fg_cls,
                timestamp=fg_ts,
                historical_values=tuple(hist_vals),
                trend_7d_change=fg_t7d,
                is_extreme_fear=_parse_bool_flag(fg_data.get("is_extreme_fear"), fg_val <= 25),
                is_extreme_greed=_parse_bool_flag(fg_data.get("is_extreme_greed"), fg_val >= 75),
            )
        except Exception as fg_err:
            logger.debug("Failed loading Fear & Greed from %s: %s", fg_path, fg_err)
    return None


def load_cached_market_breadth_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[MarketBreadthSnapshot]:
    """Load cached Market Breadth snapshot using configured candidate filenames."""
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

    eval_now = resolve_now_ts(now_ts)
    for mb_fname in get_cache_filenames("market_breadth_files"):
        mb_path = os.path.join(cache_dir, mb_fname)
        try:
            mb_data = read_json_dict(mb_path, default=None, recover_previous=True)
            if not isinstance(mb_data, dict):
                continue
            raw_gainers = mb_data.get("top_gainers") or ()
            raw_losers = mb_data.get("top_losers") or ()
            raw_adv = mb_data.get("advancing_count") if mb_data.get("advancing_count") is not None else mb_data.get("advancing")
            raw_dec = mb_data.get("declining_count") if mb_data.get("declining_count") is not None else mb_data.get("declining")
            raw_tot = mb_data.get("total_count") if mb_data.get("total_count") is not None else mb_data.get("total_symbols")
            raw_reg = mb_data.get("regime")
            mb_regime = str(raw_reg).strip() if (raw_reg is not None and not isinstance(raw_reg, bool) and str(raw_reg).strip()) else "NEUTRAL"
            raw_ts = mb_data.get("ts") if mb_data.get("ts") is not None else mb_data.get("timestamp")
            return MarketBreadthSnapshot(
                advance_ratio=safe_float(mb_data.get("advance_ratio"), 0.5),
                advancing_count=safe_int(raw_adv, 0),
                declining_count=safe_int(raw_dec, 0),
                total_count=safe_int(raw_tot, 0),
                median_change_pct=safe_float(mb_data.get("median_change_pct"), 0.0),
                mean_change_pct=safe_float(mb_data.get("mean_change_pct"), 0.0),
                dispersion_std=safe_float(mb_data.get("dispersion_std"), 0.0),
                regime=mb_regime,
                top_gainers=tuple(
                    (str(x[0]), safe_float(x[1], 0.0))
                    for x in (raw_gainers if isinstance(raw_gainers, (list, tuple)) else ())
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                ),
                top_losers=tuple(
                    (str(x[0]), safe_float(x[1], 0.0))
                    for x in (raw_losers if isinstance(raw_losers, (list, tuple)) else ())
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                ),
                ts=safe_float(raw_ts, eval_now),
            )
        except Exception as mb_err:
            logger.debug("Failed loading market breadth from %s: %s", mb_path, mb_err)
    return None


def load_cached_whale_telemetry(
    cache_dir: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached whale positioning snapshot from ``cache_dir`` without network I/O."""
    from intelligence.external.collectors.orderbook_depth import _symbol_variants

    prefix = _cfg_str("cache_files", "whale_snapshot_prefix", "whale_snapshot_")
    for sym in (list(symbols) if symbols else ["BTCUSDC"]):
        for cand_sym in ([sym] + [v for v in _symbol_variants(sym) if v != sym]):
            safe_sym = clean_symbol_key(cand_sym)
            if not safe_sym:
                continue
            w_path = os.path.join(cache_dir, f"{prefix}{safe_sym}.json")
            cand_w = read_json_dict(w_path, default=None, recover_previous=True)
            if isinstance(cand_w, dict):
                return cand_w
    return None


def load_cached_sentiment_advisor_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached SentimentAdvisor evaluation dictionary from ``cache_dir``."""
    for fname in get_cache_filenames("sentiment_eval_files"):
        cand_path = os.path.join(cache_dir, fname)
        data = read_json_dict(cand_path, default=None, recover_previous=True)
        if isinstance(data, dict) and data:
            norm = dict(data)
            norm["market_bias"] = normalize_market_bias(data.get("market_bias"))
            norm["risk_level"] = normalize_risk_level(data.get("risk_level"))
            norm["recommended_action"] = normalize_recommended_action(data.get("recommended_action"))
            return norm
    return None


def load_cached_geopolitical_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached GeopoliticalThreatAnalyzer evaluation dictionary from ``cache_dir``."""
    fname = _cfg_str("cache_files", "geopolitical_eval_file", "geopolitical_eval.json")
    cand_path = os.path.join(cache_dir, fname)
    data = read_json_dict(cand_path, default=None, recover_previous=True)
    if isinstance(data, dict) and data:
        norm = dict(data)
        tl = normalize_threat_level(data.get("threat_level"))
        norm["threat_level"] = tl
        norm["recommended_brake"] = normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl)
        return norm
    return None
