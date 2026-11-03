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
from dataclasses import asdict, dataclass
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
    exact_pos_int,
    fetch_public_json as _fetch_public_json,
    obj_attr,
    parse_dotenv,
    required_float_env,
    safe_float,
    safe_int,
    safe_nonneg_float,  # re-exported for intelligence.sentiment.guards
    safe_nonneg_float as safe_nonneg_ts,
    safe_pos_float,
)

fetch_public_json = _fetch_public_json
from cachepaths import (
    cache_path,
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    resolve_cachedb_filepath,
)
from instrument_registry import casefold_get
from providers.base import normalize_side
from state_io import (
    atomic_write_json,
    discard_generation_file,
    read_json_dict,
    read_json_dict_cached,
    read_json_dict_recovering,
    read_text_lines,
    save_json_state,
)

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MACRO_CONFIG_PATH = os.path.join(_ROOT_DIR, "macro_intelligence.conf")
DEFAULT_CRONTAB_PATH = os.path.join(_ROOT_DIR, "systemd", "crontab.prod.txt")
_CONFIG_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "parser": None}
_CRONTAB_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "interval_sec": None}


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


def obj_float(obj: Any, key: str, default: float = 0.0) -> float:
    """Extract ``key`` from ``obj`` (dict or object) and coerce to finite float."""
    return safe_float(obj_attr(obj, key, None), default)


normalize_order_side = normalize_side


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

    def _add_sym(raw_s: Any, *, skip_eq: bool = False, unique_base: bool = False) -> None:
        s_clean = str(raw_s or "").strip().strip("'\"").upper()
        if not s_clean or (skip_eq and s_clean.endswith(("_EQ", "_US_EQ"))) or s_clean in out:
            return
        base = canonical_base_asset(s_clean)
        if unique_base and (not base or base in seen_bases):
            return
        out.append(s_clean)
        seen_bases.add(base)

    if include_kraken_cache:
        try:
            from instrument_registry import tracked_price_instruments
            all_syms, _, _ = tracked_price_instruments()
            for s in all_syms:
                _add_sym(s, skip_eq=True)
        except Exception:
            pass
    if not out:
        try:
            from instruments_config import binance_symbols
            for s in binance_symbols() or []:
                _add_sym(s)
        except Exception:
            pass
    if include_kraken_cache:
        try:
            krk_cfg = os.path.join(_ROOT_DIR, "kraken", "config.env")
            if os.path.exists(krk_cfg):
                val = parse_dotenv(krk_cfg).get("KRAKEN_CACHE_PAIRS", "")
                for item in (val.split(",") if val else ()):
                    _add_sym(item, unique_base=True)
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


_ACTIVE_BOUNDED_STEPS: Dict[str, threading.Thread] = {}
_ACTIVE_BOUNDED_LOCK = threading.Lock()


def compute_step_retry_ts(last_ts: float, current_ts: float, interval_sec: float) -> float:
    """Compute a bounded retry timestamp after a failed/timed-out step to prevent 1s poll hammering."""
    itv = safe_pos_float(interval_sec, 0.0)
    if itv <= 0.0:
        return last_ts
    retry_delay = min(itv, max(5.0, itv * 0.25))
    return max(safe_nonneg_ts(last_ts, 0.0), current_ts - itv + retry_delay)


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
    with _ACTIVE_BOUNDED_LOCK:
        existing = _ACTIVE_BOUNDED_STEPS.get(label)
        if existing is not None and existing.is_alive():
            log.warning("Step '%s' still running from previous timeout; skipping duplicate execution.", label)
            return False

    outcome: List[bool] = [False]
    err_holder: List[BaseException] = []

    def _target() -> None:
        try:
            outcome[0] = bool(fn(*args))
        except BaseException as exc:
            err_holder.append(exc)
        finally:
            with _ACTIVE_BOUNDED_LOCK:
                if _ACTIVE_BOUNDED_STEPS.get(label) is threading.current_thread():
                    _ACTIVE_BOUNDED_STEPS.pop(label, None)

    t = threading.Thread(target=_target, name=f"step-{label}", daemon=True)
    with _ACTIVE_BOUNDED_LOCK:
        _ACTIVE_BOUNDED_STEPS[label] = t
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
        cand_data = read_json_dict_recovering(os.path.join(cache_dir, fname))
        if isinstance(cand_data, dict):
            data = cand_data
            break
    if data is None:
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        pid = exact_pos_int(data.get("pid", 0), 0)
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
        age_f = safe_float(age, default=None)
        res: Dict[str, Any] = {
            "running": is_alive and age < max_alive_age_sec,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(age_f, 1) if age_f is not None else None,
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


def run_daemon_loop(
    daemon_obj: Any,
    *,
    poll_interval_sec: float,
    default_poll_sec: float,
    daemon_name: str,
    stop_label: str = "daemon",
    cycle_label: str = "daemon",
    start_msg: Optional[str] = None,
    logger_inst: Optional[logging.Logger] = None,
    initial_force: bool = True,
    initial_heartbeat: Optional[Dict[str, Any]] = None,
    reraise_initial_exit: bool = False,
) -> None:
    """Run a standard signal-handled intelligence daemon poll loop."""
    import signal as _signal

    log = logger_inst or logger
    poll_sec = safe_pos_float(poll_interval_sec, default_poll_sec)
    daemon_obj.running = True

    def _handle_exit(signum, frame):
        log.info("Received termination signal (%d), stopping %s...", signum, stop_label)
        daemon_obj.running = False
        raise SystemExit(0)

    _signal.signal(_signal.SIGINT, _handle_exit)
    _signal.signal(_signal.SIGTERM, _handle_exit)

    if start_msg:
        log.info("%s", start_msg)
    if initial_heartbeat is not None:
        daemon_obj.write_heartbeat(initial_heartbeat)

    try:
        daemon_obj.run_cycle(force=initial_force)
    except SystemExit:
        if reraise_initial_exit:
            raise
        log.info("%s stopped gracefully.", daemon_name)
        return
    except Exception as exc:
        log.warning("Error during initial %s update cycle: %s", cycle_label, exc)

    while daemon_obj.running:
        try:
            daemon_obj.run_cycle(force=False)
            if daemon_obj.running:
                time.sleep(poll_sec)
        except SystemExit:
            break
        except Exception as exc:
            log.error("Unexpected error in %s loop: %s", cycle_label, exc)
            if daemon_obj.running:
                time.sleep(poll_sec)

    log.info("%s stopped gracefully.", daemon_name)


def prune_automated_cli_threads(logger_inst: Optional[logging.Logger] = None) -> int:
    """Prune automated CLI threads older than configured retention limit."""
    log = logger_inst or logger
    try:
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
        ob_ttl = max(5.0, self.orderbook_interval_sec * 0.8)
        deriv_ttl = max(10.0, self.derivatives_interval_sec * 0.8)
        whale_ttl = max(15.0, self.whale_interval_sec * 0.8)
        self.orderbook_collector = OrderbookDepthCollector(
            cache_ttl_sec=ob_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=ob_ttl,
        )
        self.derivatives_collector = DerivativesTelemetryCollector(
            cache_ttl_sec=deriv_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=deriv_ttl,
        )
        self.whale_collector = WhalePositioningCollector(
            cache_ttl_sec=whale_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=whale_ttl,
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
            ("orderbook_updated", "_last_orderbook_ts", self.orderbook_interval_sec, self.update_orderbook, 15.0, "ob"),
            ("derivatives_updated", "_last_derivatives_ts", self.derivatives_interval_sec, self.update_derivatives, 15.0, "deriv"),
            ("whale_updated", "_last_whale_ts", self.whale_interval_sec, self.update_whale, 18.0, "whale"),
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
                    elif bounded:
                        ts_map[sym] = compute_step_retry_ts(ts_map.get(sym, 0.0), current_ts, interval_sec)
            if on_symbol_done is not None:
                on_symbol_done()

    def _fetch_and_log(self, collector: Any, symbol: str, err_label: str, fmt_msg: str, *attr_specs: Any) -> bool:
        log = getattr(self, "_telemetry_logger", logger)
        try:
            snap = collector.fetch(symbol, allow_network=True)
            if snap:
                vals = [
                    obj_attr(snap, spec[0], spec[1]) if isinstance(spec, tuple) else obj_float(snap, spec)
                    for spec in attr_specs
                ]
                log.debug(fmt_msg, symbol, *vals)
                return True
        except Exception as exc:
            log.warning("[%s] Failed updating %s: %s", symbol, err_label, exc)
        return False

    def update_orderbook(self, symbol: str) -> bool:
        """Fetch spot orderbook depth and update disk snapshot."""
        return self._fetch_and_log(
            self.orderbook_collector, symbol, "orderbook",
            "[%s] Orderbook updated: mid=%.2f, bid_depth=$%.0f, ask_depth=$%.0f, imbalance=%.2f",
            "mid_price", "bid_depth_usd", "ask_depth_usd", "imbalance_ratio",
        )

    def update_derivatives(self, symbol: str) -> bool:
        """Fetch perpetual derivatives funding rate & OI and update disk snapshot."""
        return self._fetch_and_log(
            self.derivatives_collector, symbol, "derivatives",
            "[%s] Derivatives updated: funding=%.6f, OI=$%.0f",
            "funding_rate", "open_interest_usd",
        )

    def update_whale(self, symbol: str) -> bool:
        """Fetch whale positioning & OI divergence and update disk snapshot."""
        return self._fetch_and_log(
            self.whale_collector, symbol, "whale positioning",
            "[%s] Whale positioning updated: top_ratio=%.2f, taker_ratio=%.2f, regime=%s",
            "top_traders_long_ratio", "taker_buy_sell_ratio", ("divergence_regime", "NEUTRAL"),
        )


# ============================================================================
# 1. Config-Driven Taxonomies, Filenames, Endpoints & Interval Bounds
# ============================================================================

def _split_csv(raw: str, *, lower: bool = False, upper: bool = False) -> Tuple[str, ...]:
    items: List[str] = []
    for part in re.split(r"[,\n]+", raw or ""):
        if cleaned := part.strip():
            cleaned = cleaned.lower() if lower else (cleaned.upper() if upper else cleaned)
            if cleaned not in items:
                items.append(cleaned)
    return tuple(items)


def _parse_kv_map(raw: str) -> Dict[str, str]:
    return {
        k.strip(): v.strip()
        for item in _split_csv(raw, upper=True)
        if ":" in item and (k := item.split(":", 1)[0]).strip() and (v := item.split(":", 1)[1]).strip()
    }


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


def _cfg_str(section: str, key: str, fallback: str = "", config_path: Optional[str] = None) -> str:
    cfg = load_macro_intelligence_config(config_path)
    return cfg.get(section, key).strip() if cfg.has_option(section, key) else fallback


def _cfg_float(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    return safe_float(_cfg_str(section, key, "", config_path=config_path), fallback)


def _cfg_ratio(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    """Parse a float or rational multiplier expression (e.g. ``1/24``, ``5/6``, ``5/3``) from config."""
    raw = _cfg_str(section, key, "", config_path=config_path)
    if not raw:
        return fallback
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num = safe_float(num_s.strip(), None)
        den = safe_pos_float(den_s.strip(), 0.0)
        if num is not None and den > 0.0:
            return float(num) / float(den)
    return safe_float(raw, fallback)


def _cfg_int(section: str, key: str, fallback: int, config_path: Optional[str] = None) -> int:
    return safe_int(_cfg_str(section, key, "", config_path=config_path), fallback)


def _parse_crontab_executor_interval_sec(crontab_path: Optional[str] = None) -> Optional[float]:
    """Extract ``auto_cryptotrade_executor`` schedule cadence in seconds from crontab (e.g. ``*/6`` -> 360s, ``*/10`` -> 600s)."""
    target = crontab_path or os.environ.get("MPTRADE_CRONTAB_PATH") or DEFAULT_CRONTAB_PATH
    try:
        mtime = os.path.getmtime(target) if os.path.exists(target) else -1.0
    except OSError:
        mtime = -1.0
    if mtime >= 0.0 and _CRONTAB_CACHE.get("path") == target and _CRONTAB_CACHE.get("mtime") == mtime:
        return _CRONTAB_CACHE.get("interval_sec")
    resolved: Optional[float] = None
    if mtime >= 0.0:
        for raw_line in read_text_lines(target):
            line = (raw_line or "").strip()
            if not line or line.startswith("#") or "auto_cryptotrade_executor" not in line:
                continue
            m = re.match(r"^\*/(\d+)\s+\*\s+\*\s+\*\s+\*\s+", line)
            if m:
                mins = safe_pos_float(m.group(1), 0.0)
                if mins > 0.0:
                    resolved = mins * 60.0
                    break
    _CRONTAB_CACHE.update(path=target, mtime=mtime, interval_sec=resolved)
    return resolved


def resolve_crypto_executor_interval_sec(
    config_path: Optional[str] = None,
    crontab_path: Optional[str] = None,
) -> float:
    """Resolve primary ``auto_cryptotrade_executor`` cycle interval in seconds (default 360s = 6m)."""
    env_sec = safe_pos_float(os.environ.get("AUTO_CRYPTOTRADE_INTERVAL_SEC"), 0.0)
    if env_sec > 0.0:
        return max(30.0, env_sec)
    env_min = safe_pos_float(os.environ.get("AUTO_CRYPTOTRADE_INTERVAL_MIN"), 0.0)
    if env_min > 0.0:
        return max(30.0, env_min * 60.0)

    cfg_sec = safe_pos_float(_cfg_float("intervals", "crypto_executor_interval_sec", 360.0, config_path=config_path), 360.0)
    cron_sec = _parse_crontab_executor_interval_sec(crontab_path=crontab_path)
    if cron_sec is not None and cron_sec > 0.0:
        if crontab_path is not None or os.environ.get("MPTRADE_CRONTAB_PATH") or (cfg_sec == 360.0 and cron_sec != 360.0):
            return max(30.0, cron_sec)
    return max(30.0, cfg_sec)


def resolve_derived_telemetry_intervals(
    executor_interval_sec: Optional[float] = None,
    config_path: Optional[str] = None,
    crontab_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute all dependent collector, freshness, grace, and LLM evaluator intervals as mathematical functions of ``T_exec``."""
    exec_sec = safe_pos_float(executor_interval_sec, 0.0) or resolve_crypto_executor_interval_sec(
        config_path=config_path,
        crontab_path=crontab_path,
    )
    exec_min = exec_sec / 60.0

    def _sec(mult_key: str, default_mult: float, legacy_sec_key: str = "") -> float:
        if legacy_sec_key:
            explicit = safe_pos_float(_cfg_str("intervals", legacy_sec_key, "", config_path=config_path), 0.0)
            if explicit > 0.0:
                return explicit
        return round(exec_sec * _cfg_ratio("intervals", mult_key, default_mult, config_path=config_path), 6)

    return {
        "crypto_executor_interval_sec": round(exec_sec, 6),
        "share_executor_interval_sec": _sec("share_executor_interval_mult", 5.0),
        "ui_sync_interval_sec": _sec("ui_sync_interval_mult", 2.0),
        "ui_sync_idle_sec": _sec("ui_sync_idle_mult", 1.0 / 3.0),
        "orderbook_interval_sec": _sec("orderbook_interval_mult", 1.0 / 24.0),
        "derivatives_interval_sec": _sec("derivatives_interval_mult", 1.0 / 12.0),
        "whale_interval_sec": _sec("whale_interval_mult", 1.0 / 6.0),
        "news_interval_sec": _sec("news_interval_mult", 1.0 / 3.0),
        "breadth_interval_sec": _sec("breadth_interval_mult", 5.0 / 6.0),
        "macro_benchmarks_interval_sec": _sec("macro_benchmarks_interval_mult", 5.0 / 6.0),
        "fear_greed_interval_sec": _sec("fear_greed_interval_mult", 5.0),
        "sensor_stale_age_min": round(
            exec_min * _cfg_ratio("intervals", "sensor_stale_age_mult", 11.0 / 12.0, config_path=config_path),
            6,
        ),
        "unfilled_order_cancel_grace_sec": _sec("unfilled_order_cancel_grace_mult", 5.0 / 3.0),
        "default_sentiment_interval_sec": _sec("default_sentiment_interval_mult", 30.0, "default_sentiment_interval_sec"),
        "min_dynamic_sentiment_interval_sec": _sec("min_dynamic_sentiment_interval_mult", 5.0, "min_dynamic_sentiment_interval_sec"),
        "peak_overlap_sentiment_interval_sec": _sec("peak_overlap_sentiment_interval_mult", 3.0, "peak_overlap_sentiment_interval_sec"),
        "max_dynamic_sentiment_interval_sec": _sec("max_dynamic_sentiment_interval_mult", 45.0, "max_dynamic_sentiment_interval_sec"),
        "sentiment_checkpoint_cooldown_sec": _sec("sentiment_checkpoint_cooldown_mult", 7.5, "sentiment_checkpoint_cooldown_sec"),
        "default_geopolitical_interval_sec": _sec("default_geopolitical_interval_mult", 45.0, "default_geopolitical_interval_sec"),
        "min_dynamic_geopolitical_interval_sec": _sec("min_dynamic_geopolitical_interval_mult", 2.5, "min_dynamic_geopolitical_interval_sec"),
        "max_dynamic_geopolitical_interval_sec": _sec("max_dynamic_geopolitical_interval_mult", 60.0, "max_dynamic_geopolitical_interval_sec"),
        "default_news_rolling_window_sec": _sec("default_news_rolling_window_mult", 40.0, "default_news_rolling_window_sec"),
        "default_fallback_retry_ttl_sec": _sec("default_fallback_retry_ttl_mult", 1.0 / 3.0, "default_fallback_retry_ttl_sec"),
        "dynamic_lexicon_ttl_sec": _sec("dynamic_lexicon_ttl_mult", 720.0, "dynamic_lexicon_ttl_sec"),
    }


def _tax_tuple(key: str, *, lower: bool = False) -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", key), lower=lower, upper=not lower)


def get_market_biases() -> Tuple[str, ...]:
    return _tax_tuple("market_biases")


def get_risk_levels() -> Tuple[str, ...]:
    return _tax_tuple("risk_levels")


def get_recommended_actions() -> Tuple[str, ...]:
    return _tax_tuple("recommended_actions")


def get_breadth_regimes() -> Tuple[str, ...]:
    return _tax_tuple("breadth_regimes")


def get_threat_levels() -> Tuple[str, ...]:
    return _tax_tuple("threat_levels")


def get_recommended_brakes() -> Tuple[str, ...]:
    return _tax_tuple("recommended_brakes")


MARKET_BIASES: Tuple[str, ...] = get_market_biases()
RISK_LEVELS: Tuple[str, ...] = get_risk_levels()
RECOMMENDED_ACTIONS: Tuple[str, ...] = get_recommended_actions()
BREADTH_REGIMES: Tuple[str, ...] = get_breadth_regimes()
THREAT_LEVELS: Tuple[str, ...] = get_threat_levels()
RECOMMENDED_BRAKES: Tuple[str, ...] = get_recommended_brakes()

_INIT_DERIVED_INTERVALS: Dict[str, float] = resolve_derived_telemetry_intervals()
DEFAULT_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_sentiment_interval_sec"]
MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["min_dynamic_sentiment_interval_sec"]
PEAK_OVERLAP_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["peak_overlap_sentiment_interval_sec"]
MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["max_dynamic_sentiment_interval_sec"]
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = _INIT_DERIVED_INTERVALS["sentiment_checkpoint_cooldown_sec"]

DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_geopolitical_interval_sec"]
MIN_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["min_dynamic_geopolitical_interval_sec"]
MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["max_dynamic_geopolitical_interval_sec"]
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = _INIT_DERIVED_INTERVALS["default_news_rolling_window_sec"]
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = _INIT_DERIVED_INTERVALS["default_fallback_retry_ttl_sec"]
DYNAMIC_LEXICON_TTL_SEC: float = _INIT_DERIVED_INTERVALS["dynamic_lexicon_ttl_sec"]


def get_fear_greed_endpoint_url() -> str:
    return _cfg_str("feeds", "fear_greed_endpoint_url")


def get_market_breadth_endpoint_url() -> str:
    return _cfg_str("feeds", "market_breadth_endpoint_url")


def get_breadth_excluded_bases() -> Set[str]:
    return set(_tax_tuple("breadth_excluded_bases"))


def get_breadth_quote_suffixes() -> Tuple[str, ...]:
    return _tax_tuple("breadth_quote_suffixes")


def get_breadth_min_volume_usd() -> float:
    return _cfg_float("taxonomies", "breadth_min_volume_usd", 5_000_000.0)


def get_breadth_top_mover_min_volume_usd() -> float:
    return _cfg_float("taxonomies", "breadth_top_mover_min_volume_usd", 10_000_000.0)


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
    return cleaned if cleaned in allowed else _parse_kv_map(_cfg_str("taxonomies", alias_cfg_key)).get(cleaned, fallback)


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
    if (aliased := _parse_kv_map(_cfg_str("taxonomies", "brake_aliases")).get(cleaned)) is not None:
        return aliased
    levels = get_threat_levels()
    norm_threat = normalize_threat_level(threat_level)
    if len(levels) >= 3 and len(brakes) >= 3 and norm_threat in (levels[1], levels[2]):
        return brakes[2 if norm_threat == levels[2] else 1]
    return brakes[0] if brakes else "NONE"


# ============================================================================
# 2. Specialized Multi-Source Feeds & Dynamic Crisis Lexicon Discovery
# ============================================================================

_LEXICON_FILE_CACHE: Dict[str, Tuple[tuple, Dict[str, Any]]] = {}
_LEXICON_EVAL_CACHE: Dict[Tuple[Any, ...], Dict[str, Tuple[str, ...]]] = {}

_NOISE_LEXICON_PHRASES: frozenset[str] = frozenset({
    "more than", "first time", "yahoo finance", "financial times", "new york",
    "york times", "wall street", "street journal", "jerusalem post", "military times",
    "associated press", "was heartbroken", "leave everything", "united states",
    "middle east", "according pentagon", "pentagon estimate", "cuba wins",
    "shocks however", "profit thanks", "thanks gold", "insane plan", "focus live",
    "doordash strikes", "billion deal", "rival instacart", "grocery delivery",
    "gillibrand presses", "civilian schools", "hacked emergency", "shock emergency",
    "emergency sanction",
})

_NON_CRISIS_STRIKE_CONTEXT_WORDS: frozenset[str] = frozenset({
    "deal", "deals", "agreement", "partnership", "pact", "contract", "bargain",
    "bet", "chord", "balance", "gold", "rich", "pose", "lightning",
})

_NON_CRISIS_ATTACK_CONTEXT_WORDS: frozenset[str] = frozenset({
    "allegedly", "arrest", "arrested", "baseball", "bat", "boy", "burglary",
    "charged", "dog", "heart", "mugger", "panic", "robbery", "shark",
    "stabbing", "suspect", "teenager", "volunteer",
})

_TRACKING_QUERY_PARAMS: frozenset[str] = frozenset({
    "at_campaign", "at_medium", "mod", "oc", "rss", "traffic_source",
    "utm_campaign", "utm_content", "utm_medium", "utm_source", "utm_term",
})


def canonical_article_url(url: Any) -> str:
    """Normalize an article URL by stripping RSS tracking query parameters and trailing slashes."""
    if not isinstance(url, str) or not url.strip():
        return ""
    raw = url.strip()
    try:
        parsed = urllib.parse.urlsplit(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw.rstrip("/")
        kept_q = [
            (k, v)
            for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_QUERY_PARAMS
        ]
        clean_path = parsed.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            urllib.parse.urlencode(kept_q),
            "",
        ))
    except Exception:
        return raw.rstrip("/")

_STEM_EXACT_PREFIX_WHITELIST: Dict[str, frozenset[str]] = {
    "ban": frozenset({"ban", "bans", "banned", "banning"}),
    "freeze": frozenset({"freeze", "freezes", "freezing", "frozen"}),
    "war": frozenset({
        "war", "wars", "warfare", "wartime", "warship", "warships",
        "warhead", "warheads", "warplane", "warplanes", "warzone", "warring",
    }),
    "hack": frozenset({"hack", "hacks", "hacked", "hacker", "hackers", "hacking"}),
    "halt": frozenset({"halt", "halts", "halted", "halting"}),
}


def _get_stopwords_set() -> Set[str]:
    return set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))


def _get_crisis_stems_set() -> Set[str]:
    return set(_split_csv(_cfg_str("lexicon_seed", "crisis_context_stems"), lower=True))


def _get_seed_terms_set() -> Set[str]:
    return set(_split_csv(_cfg_str("lexicon_seed", "high_severity_terms"), lower=True)) | set(
        _split_csv(_cfg_str("lexicon_seed", "acute_shock_phrases"), lower=True)
    )


def _token_matches_crisis_stem(tok: str, crisis_stems: Set[str]) -> bool:
    """Return True if ``tok`` is a genuine derivative of a configured crisis stem (avoiding substring false positives)."""
    w = (tok or "").strip().lower()
    if len(w) < 3:
        return False
    return any(
        (w in allowed if (allowed := _STEM_EXACT_PREFIX_WHITELIST.get(stem)) is not None else w.startswith(stem))
        for stem in crisis_stems
    )


_NON_CRISIS_DYNAMIC_BIGRAM_WORDS: frozenset[str] = frozenset({
    "ahead", "allegedly", "already", "answered", "apps", "arrested", "article",
    "articles", "baseball", "bat", "boy", "brake", "briefing", "bus", "calls",
    "car", "carried", "carries", "chip", "commentary", "conduct", "conducted",
    "conducts", "continue", "continues", "criticize", "criticized", "criticizes",
    "debunked", "defensive", "digested", "downscale", "eight", "elevated", "eve",
    "exceptionally", "explained", "exposes", "false", "features", "five", "four",
    "history", "horrific", "introduces", "involvement", "island", "jet", "join",
    "keyword", "landing", "limited", "links", "localized", "media", "moderate",
    "nine", "non", "normal", "noting", "official", "officials", "one", "orderly",
    "plane", "potential", "prices", "processed", "progress", "qeshm", "questions",
    "readiness", "record", "reported", "reportedly", "researchers", "retaining",
    "say", "seven", "six", "somalia", "suspected", "targets", "ten", "three",
    "timeline", "trace", "traced", "train", "trains", "truck", "two", "uganda",
    "unpriced", "veto", "volunteer", "withdrawing", "worse", "wrap",
})


def _extract_contiguous_token_runs(text: str, stopwords: Set[str]) -> List[List[str]]:
    """Split ``text`` into contiguous runs of >=3-char non-stopword/non-noise alphabetic tokens."""
    runs: List[List[str]] = []
    cur: List[str] = []
    for raw_tok in re.findall(r"[a-z]+|[^\s\w]+|\d+", (text or "").lower()):
        if raw_tok.isalpha() and len(raw_tok) >= 3 and raw_tok not in stopwords and raw_tok not in _NON_CRISIS_DYNAMIC_BIGRAM_WORDS:
            cur.append(raw_tok)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _is_noise_dynamic_keyword(term: Any, stopwords: Optional[Set[str]] = None) -> bool:
    """Return True if ``term`` is a generic stopword/publisher/lifestyle phrase that must not trigger [HIGH-SEV]."""
    if not isinstance(term, str):
        return True
    t = re.sub(r"\s+", " ", term.strip().lower())
    sw = stopwords if stopwords is not None else _get_stopwords_set()
    if len(t) < 4 or t in _NOISE_LEXICON_PHRASES or t in sw:
        return True
    words = re.findall(r"[a-zA-Z]{2,}", t)
    return not words or any(w in sw or w in _NON_CRISIS_DYNAMIC_BIGRAM_WORDS for w in words)


def _dynamic_keyword_has_crisis_stem(term: str, crisis_stems: Optional[Set[str]] = None) -> bool:
    """Return True if at least one token in ``term`` matches a configured crisis stem."""
    stems = crisis_stems if crisis_stems is not None else _get_crisis_stems_set()
    return any(_token_matches_crisis_stem(w, stems) for w in re.findall(r"[a-zA-Z]{3,}", (term or "").lower()))


def _is_excluded_non_crisis_attack_keyword(
    kw_clean: str,
    text_words: Set[str],
    crisis_stems: Set[str],
) -> bool:
    """Return True if an 'attack'-based keyword appears solely in a local crime/medical/animal attack context."""
    kw_words = set(re.findall(r"[a-zA-Z]{3,}", (kw_clean or "").lower()))
    combined_words = text_words | kw_words
    if not (combined_words & _NON_CRISIS_ATTACK_CONTEXT_WORDS):
        return False
    other_stems = crisis_stems - {"attack"}
    if any(_token_matches_crisis_stem(w, other_stems) for w in kw_words):
        return False
    if text_words and any(_token_matches_crisis_stem(w, other_stems) for w in text_words):
        return False
    return any(w.startswith("attack") for w in kw_words)


def filter_valid_severity_keywords(
    raw_keywords: Any,
    *,
    title: Optional[str] = None,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Filter a headline's matched keywords, stripping generic stopword/publisher noise while keeping seed terms."""
    seed_set = _get_seed_terms_set()
    crisis_stems = _get_crisis_stems_set()
    sw = _get_stopwords_set()
    title_words = set(re.findall(r"[a-zA-Z]{3,}", title.lower())) if isinstance(title, str) and title.strip() else set()
    out: List[str] = []
    if isinstance(raw_keywords, (list, tuple)):
        for k in raw_keywords:
            if not isinstance(k, str) or not k.strip():
                continue
            k_clean = re.sub(r"\s+", " ", k.strip()).lower()
            if _is_excluded_non_crisis_attack_keyword(k_clean, title_words, crisis_stems):
                continue
            if k_clean in seed_set or (
                not _is_noise_dynamic_keyword(k_clean, sw)
                and _dynamic_keyword_has_crisis_stem(k_clean, crisis_stems)
            ):
                if k.strip() not in out:
                    out.append(k.strip())
    if not out and isinstance(title, str) and title.strip():
        for kw in match_high_severity_keywords(title, cache_dir=cache_dir, now_ts=now_ts):
            if kw not in out:
                out.append(kw)
    return tuple(out)


def _strip_headline_publisher_suffix(title: str) -> str:
    """Strip trailing ' - Publisher Name' suffix from RSS headline titles before dynamic n-gram discovery."""
    cleaned = (title or "").strip()
    parts = re.split(r"\s+[-–—|]\s+", cleaned)
    if len(parts) >= 2 and len(parts[-1].split()) <= 5 and len(parts[0]) >= 12:
        return " - ".join(parts[:-1]).strip()
    return cleaned


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
        stopwords = _get_stopwords_set()
        valid_rows: List[Tuple[float, int, str, str]] = []
        for idx, (term, meta) in enumerate(entries.items()):
            if not isinstance(term, str) or not term.strip():
                continue
            t_clean = re.sub(r"\s+", " ", term.strip().lower())
            if _is_noise_dynamic_keyword(t_clean, stopwords):
                continue
            ts = safe_float(meta.get("ts") if isinstance(meta, dict) else meta, 0.0)
            tier = str(meta.get("tier", "high")).lower() if isinstance(meta, dict) else "high"
            if ts > 0.0 and (now - ts) <= ttl:
                valid_rows.append((ts, idx, t_clean, tier))
        valid_rows.sort(key=lambda r: (-r[0], r[1]))
        high_seen: Dict[str, None] = {}
        acute_seen: Dict[str, None] = {}
        for _, _, t_clean, tier in valid_rows:
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

    stopwords = _get_stopwords_set()
    existing_entries: Dict[str, Dict[str, Any]] = {}
    pruned_existing = False
    raw = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
    if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
        for k, v in raw["entries"].items():
            if isinstance(k, str) and isinstance(v, dict):
                k_clean = re.sub(r"\s+", " ", k.strip().lower())
                if _is_noise_dynamic_keyword(k_clean, stopwords):
                    pruned_existing = True
                    continue
                ts = safe_float(v.get("ts"), 0.0)
                if ts > 0.0 and (now - ts) <= ttl:
                    existing_entries[k_clean] = dict(v)
                else:
                    pruned_existing = True
            else:
                pruned_existing = True

    changed = pruned_existing
    for raw_t, tier_override in (
        *[(t, None) for t in new_high_terms],
        *[(a, "acute") for a in new_acute_terms],
    ):
        t = re.sub(r"\s+", " ", str(raw_t or "").strip().lower())
        if _is_noise_dynamic_keyword(t, stopwords):
            continue
        prev_entry = existing_entries.get(t, {})
        prev_tier = str(prev_entry.get("tier", "high")).lower()
        prev_ts = safe_float(prev_entry.get("ts"), 0.0)
        if tier_override:
            tier = tier_override
        elif prev_tier == "acute" and prev_ts > 0.0 and (now - prev_ts) <= 3600.0:
            tier = "acute"
        else:
            tier = "high"
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


def demote_dynamic_acute_terms(
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Demote dynamically discovered ``acute`` terms to ``high`` once threat is not CRITICAL_SHOCK."""
    now = resolve_now_ts(now_ts)
    ttl = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)
    raw = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), dict):
        return False

    stopwords = _get_stopwords_set()
    cleaned_entries: Dict[str, Dict[str, Any]] = {}
    changed = False
    for k, v in raw["entries"].items():
        if not isinstance(k, str) or not isinstance(v, dict):
            changed = True
            continue
        k_clean = re.sub(r"\s+", " ", k.strip().lower())
        if _is_noise_dynamic_keyword(k_clean, stopwords):
            changed = True
            continue
        ts = safe_float(v.get("ts"), 0.0)
        if ts <= 0.0 or (now - ts) > ttl:
            changed = True
            continue
        entry = dict(v)
        if str(entry.get("tier", "high")).lower() == "acute":
            entry["tier"] = "high"
            changed = True
        cleaned_entries[k_clean] = entry

    if changed:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_json_state(path, {"updated_ts": now, "entries": cleaned_entries}, indent=2)
        except Exception as exc:
            logger.debug("Could not save demoted dynamic lexicon to %s: %s", path, exc)
        _LEXICON_FILE_CACHE.clear()
        _LEXICON_EVAL_CACHE.clear()
    return changed


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
    crisis_stems, stopwords = _get_crisis_stems_set(), _get_stopwords_set()
    if not crisis_stems or not headlines:
        return ()

    phrase_sources: Dict[str, Set[str]] = {}
    phrase_headlines: Dict[str, Set[str]] = {}
    phrase_counts: Dict[str, int] = {}

    phrase_has_direct_stem: Dict[str, bool] = {}
    seen_norm_titles: Set[str] = set()
    seen_canon_urls: Set[str] = set()

    for h in headlines:
        raw_title = obj_attr(h, "title", "")
        if not isinstance(raw_title, str) or not raw_title.strip():
            continue
        norm_title = re.sub(r"\s+", " ", _strip_headline_publisher_suffix(raw_title).lower()).strip()
        if not norm_title:
            continue
        canon_url = canonical_article_url(obj_attr(h, "url", ""))
        if norm_title in seen_norm_titles or (canon_url and canon_url in seen_canon_urls):
            continue
        seen_norm_titles.add(norm_title)
        if canon_url:
            seen_canon_urls.add(canon_url)
        all_words = re.findall(r"[a-zA-Z]{3,}", norm_title)
        word_set = set(all_words)
        eff_stems = set(crisis_stems)
        if word_set & _NON_CRISIS_STRIKE_CONTEXT_WORDS:
            eff_stems.discard("strike")
        if word_set & _NON_CRISIS_ATTACK_CONTEXT_WORDS:
            eff_stems.discard("attack")
        src_key = str(obj_attr(h, "source", "") or "unknown").strip().lower()
        article_key = canon_url or norm_title
        for tokens in _extract_contiguous_token_runs(norm_title, stopwords):
            if len(tokens) < 2:
                continue
            stem_flags = [_token_matches_crisis_stem(tok, eff_stems) for tok in tokens]
            if not any(stem_flags):
                continue
            for i in range(len(tokens) - 1):
                if not any(stem_flags[max(0, i - 1) : min(len(tokens), i + 3)]):
                    continue
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                if _is_noise_dynamic_keyword(bigram, stopwords):
                    continue
                if stem_flags[i] or stem_flags[i + 1]:
                    phrase_has_direct_stem[bigram] = True
                seen_articles = phrase_headlines.setdefault(bigram, set())
                if article_key not in seen_articles:
                    seen_articles.add(article_key)
                    phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
                phrase_sources.setdefault(bigram, set()).add(src_key)

    discovered = [
        phrase
        for phrase, count in phrase_counts.items()
        if len(phrase_headlines.get(phrase, set())) >= 2
        and (
            len(phrase_sources.get(phrase, set())) >= 2
            or (count >= 2 and phrase_has_direct_stem.get(phrase, False))
        )
    ]
    if persist:
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
    crisis_stems, stopwords = _get_crisis_stems_set(), _get_stopwords_set()
    extracted: List[str] = []
    for text in (str(summary or ""), *(str(r or "") for r in (key_risks or ()))):
        for tokens in _extract_contiguous_token_runs(text, stopwords):
            for w1, w2 in zip(tokens, tokens[1:]):
                if any(_token_matches_crisis_stem(w, crisis_stems) for w in (w1, w2)):
                    bg = f"{w1} {w2}"
                    if not _is_noise_dynamic_keyword(bg, stopwords) and bg not in extracted:
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
    candidates = (*dyn.get("acute_shock_phrases", ()), *dyn.get("high_severity_terms", ()), *get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts))
    prioritized = list(dict.fromkeys(t for t in candidates if t))[:max_terms]
    encoded_terms = [urllib.parse.quote_plus(f'"{t}"' if " " in t else t) for t in prioritized]
    query_str = "+OR+".join(encoded_terms) if encoded_terms else "macro"
    return template.format(query=query_str) if "{query}" in template else template


def get_specialized_rss_urls(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    """Return curated specialized editorial RSS feeds + dynamically built crisis query feed."""
    dyn_url = build_dynamic_rss_url(cache_dir=cache_dir, now_ts=now_ts)
    return tuple(dict.fromkeys((*_split_csv(_cfg_str("feeds", "specialized_rss_urls")), *([dyn_url] if dyn_url else []))))


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
    seed_set, crisis_stems = _get_seed_terms_set(), _get_crisis_stems_set()
    text_words = set(re.findall(r"[a-zA-Z]{3,}", lower))
    return tuple(
        kw
        for kw in get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts)
        if kw in lower
        and not _is_excluded_non_crisis_attack_keyword(kw, text_words, crisis_stems)
        and (kw in seed_set or _dynamic_keyword_has_crisis_stem(kw, crisis_stems))
        and re.search(r"\b" + re.escape(kw) + r"\b", lower)
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
        return asdict(self)


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


def _primary_eet_dt(ts: float) -> datetime:
    return _to_tz(ts, _cfg_str("sessions", "primary_timezone", "Europe/Bucharest"), _cfg_int("sessions", "primary_fallback_offset_hours", 2))


def resolve_global_trading_session(now_ts: Optional[float] = None) -> GlobalSessionContext:
    """Resolve current global market session and European/US/Asian trader flow dynamics."""
    ts = resolve_now_ts(now_ts)
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _primary_eet_dt(ts)
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

    dt_last, dt_now = _primary_eet_dt(last_f), _primary_eet_dt(now_f)
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


def get_macro_calendar_windows_eet() -> Tuple[Tuple[int, int, str, str], ...]:
    """Load weekday macro economic & liquidity calendar windows from ``macro_intelligence.conf``."""
    parsed: List[Tuple[int, int, str, str]] = []
    for item in _split_csv(_cfg_str("sessions", "macro_calendar_windows_eet")):
        parts = [p.strip() for p in item.split("|")]
        if len(parts) >= 3 and ":" in parts[0]:
            try:
                hh_s, mm_s = parts[0].split(":", 1)
                parsed.append((int(hh_s), int(mm_s), parts[1], parts[2]))
            except ValueError:
                continue
    return tuple(parsed)


def get_macro_calendar_headline_keywords() -> Tuple[str, ...]:
    """Load macro economic calendar keywords from ``macro_intelligence.conf``."""
    return _split_csv(_cfg_str("sessions", "macro_calendar_headline_keywords"), lower=True)


def _fmt_countdown_min(delta_min: float) -> str:
    m = int(round(abs(delta_min)))
    if m < 60:
        return f"{m}m"
    h, rem = divmod(m, 60)
    return f"{h}h{rem:02d}m" if rem else f"{h}h"


def resolve_macro_economic_calendar(
    now_ts: Optional[float] = None,
    breaking_news: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Resolve current trading session, next high-impact macro/liquidity window, and live macro headlines."""
    ts = resolve_now_ts(now_ts)
    session_ctx = resolve_global_trading_session(ts)
    dt_eet = _primary_eet_dt(ts)
    tz_abbr = dt_eet.strftime("%Z") or "EET"
    windows = get_macro_calendar_windows_eet()

    active_window: Optional[str] = None
    upcoming: List[Tuple[float, str, str, str]] = []
    for day_offset in range(0, 4):
        cand_date = (dt_eet + timedelta(days=day_offset)).date()
        if cand_date.weekday() >= 5:
            continue
        for hh, mm, code, desc in windows:
            ev_dt = datetime(cand_date.year, cand_date.month, cand_date.day, hh, mm, tzinfo=dt_eet.tzinfo)
            delta_min = (ev_dt.timestamp() - ts) / 60.0
            time_lbl = f"{ev_dt.strftime('%a')} {hh:02d}:{mm:02d} {tz_abbr}" if day_offset > 0 else f"{hh:02d}:{mm:02d} {tz_abbr}"
            if -20.0 <= delta_min <= 0.0 and active_window is None:
                active_window = f"{desc} at {time_lbl} ({abs(int(round(delta_min)))}m ago — ACTIVE VOLATILITY WINDOW)"
            elif delta_min > 0.0:
                upcoming.append((delta_min, code, desc, f"{desc} at {time_lbl} in {_fmt_countdown_min(delta_min)}"))

    upcoming.sort(key=lambda x: x[0])
    next_event = upcoming[0][3] if upcoming else "None scheduled"
    next_code = upcoming[0][1] if upcoming else "NONE"
    next_in_min = round(upcoming[0][0], 1) if upcoming else None

    macro_kws = get_macro_calendar_headline_keywords()
    detected_macro_headlines: List[str] = []
    if breaking_news and macro_kws:
        for item in breaking_news:
            title = str(obj_attr(item, "title", "") or "").strip()
            if title and any(re.search(r"\b" + re.escape(kw) + r"\b", title.lower()) for kw in macro_kws):
                detected_macro_headlines.append(title)

    return {
        "session_key": session_ctx.session_key,
        "session_label": session_ctx.session_label,
        "eet_time": session_ctx.eet_time_str,
        "ny_time": session_ctx.ny_time_str,
        "is_weekend": session_ctx.is_weekend,
        "active_release_window": active_window,
        "next_high_impact_event": active_window or next_event,
        "next_scheduled_event": next_event,
        "next_event_code": next_code,
        "next_event_in_min": next_in_min,
        "upcoming_windows": [x[3] for x in upcoming[:3]],
        "detected_macro_headlines": detected_macro_headlines[:3],
        "ts": ts,
    }


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
    derived = resolve_derived_telemetry_intervals()
    dyn_default_sent = derived["default_sentiment_interval_sec"]
    raw_base = safe_pos_float(base_interval_sec, dyn_default_sent)
    if raw_base == DEFAULT_SENTIMENT_INTERVAL_SEC and dyn_default_sent != DEFAULT_SENTIMENT_INTERVAL_SEC:
        raw_base = dyn_default_sent
    base = max(60.0, raw_base)
    ts = resolve_now_ts(now_ts)

    scale = base / dyn_default_sent
    min_interval = max(60.0, derived["min_dynamic_sentiment_interval_sec"] * scale)
    peak_overlap_interval = max(60.0, derived["peak_overlap_sentiment_interval_sec"] * scale)
    max_interval = max(base, derived["max_dynamic_sentiment_interval_sec"] * scale)
    checkpoint_cooldown = max(60.0, derived["sentiment_checkpoint_cooldown_sec"] * scale)

    session_ctx = resolve_global_trading_session(ts)
    if not session_ctx.is_weekend:
        dt_eet = _to_tz(
            ts,
            _cfg_str("sessions", "primary_timezone", "Europe/Bucharest"),
            _cfg_int("sessions", "primary_fallback_offset_hours", 2),
        )
        minutes_eet = dt_eet.hour * 60 + dt_eet.minute
        if (15 * 60 + 30) <= minutes_eet <= (18 * 60 + 30):
            return peak_overlap_interval, f"critical_us_eu_overlap_window_{session_ctx.session_key}"

    norm_geo = normalize_threat_level(geopolitical_threat_level)
    cached_risk = normalize_risk_level(obj_attr(cached_assessment, "risk_level", None))
    breadth_regime = str(obj_attr(breadth, "regime", "NEUTRAL") or "NEUTRAL") if breadth is not None else "NEUTRAL"
    median_abs = abs(obj_float(breadth, "median_change_pct", 0.0)) if breadth is not None else 0.0
    cur_adv = safe_float(obj_attr(breadth, "advance_ratio", None), None)

    if norm_geo == "CRITICAL_SHOCK":
        return min_interval, "critical_geopolitical_shock_30m"
    if (
        breadth_regime in ("PANIC_WASHOUT", "EUPHORIC_BLOWOFF")
        or median_abs >= 5.5
        or cached_risk == "EXTREME"
        or (cur_adv is not None and (cur_adv <= 0.15 or cur_adv >= 0.85))
    ):
        return min_interval, f"extreme_market_regime_{breadth_regime.lower()}_{cached_risk.lower()}"
    cur_fg_val = safe_int(obj_attr(fear_greed, "value", None), None)
    prev_fg_val = safe_int(prev_fear_greed_val, None)
    if cur_fg_val is not None and prev_fg_val is not None and abs(cur_fg_val - prev_fg_val) >= 8:
        return min_interval, f"fear_greed_jump_{cur_fg_val - prev_fg_val:+d}pts"
    prev_adv = safe_float(prev_advance_ratio, None)
    if cur_adv is not None and prev_adv is not None and abs(cur_adv - prev_adv) >= 0.18:
        return min_interval, f"breadth_shift_{cur_adv - prev_adv:+.2f}"

    crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=checkpoint_cooldown)
    if crossed_cp is not None:
        return max(min_interval, min(base, checkpoint_cooldown)), f"session_checkpoint_{crossed_cp}"

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
    last_headlines_digest: str = "",
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_geopolitical`` evaluation."""
    derived = resolve_derived_telemetry_intervals()
    dyn_default_geo = derived["default_geopolitical_interval_sec"]
    raw_base = safe_pos_float(base_interval_sec, dyn_default_geo)
    if raw_base == DEFAULT_GEOPOLITICAL_INTERVAL_SEC and dyn_default_geo != DEFAULT_GEOPOLITICAL_INTERVAL_SEC:
        raw_base = dyn_default_geo
    base = max(60.0, raw_base)
    geo_scale = base / dyn_default_geo
    min_geo_itv = max(60.0, derived["min_dynamic_geopolitical_interval_sec"] * geo_scale)
    max_geo_itv = max(base, derived["max_dynamic_geopolitical_interval_sec"] * geo_scale)
    cp_cooldown_itv = max(60.0, derived["sentiment_checkpoint_cooldown_sec"] * geo_scale)
    ts = resolve_now_ts(now_ts)
    changed_b = _parse_bool_flag(headlines_changed)

    if cached_assessment is not None and _parse_bool_flag(obj_attr(cached_assessment, "is_fallback", False)):
        raw_retry = safe_pos_float(fallback_retry_ttl_sec, derived["default_fallback_retry_ttl_sec"])
        if raw_retry == DEFAULT_FALLBACK_RETRY_TTL_SEC and derived["default_fallback_retry_ttl_sec"] != DEFAULT_FALLBACK_RETRY_TTL_SEC:
            raw_retry = derived["default_fallback_retry_ttl_sec"]
        return max(30.0, raw_retry), "fallback_short_retry"

    threat_level = normalize_threat_level(obj_attr(cached_assessment, "threat_level", None))
    has_shock = _parse_bool_flag(obj_attr(news_snapshot, "has_critical_shock_keywords", False)) if news_snapshot is not None else False
    high_sev_count = safe_int(obj_attr(news_snapshot, "high_severity_count", 0), 0) if news_snapshot is not None else 0

    has_new_acute = not bool(last_headlines_digest)
    new_high_sev_count = high_sev_count
    if changed_b and last_headlines_digest and news_snapshot is not None:
        high_sev, _ = extract_high_severity_headlines_and_digest(news_snapshot)
        prev_lower = last_headlines_digest.lower()
        if high_sev:
            new_high_sev_count = sum(
                1
                for h in high_sev[:12]
                if (str(obj_attr(h, "title", None) or "").strip().lower() not in prev_lower)
            )
        if threat_level in ("NORMAL", "ELEVATED"):
            active_acute_phrases = get_active_acute_shock_phrases(now_ts=ts)
            has_new_acute = any(
                phrase in str(obj_attr(h, "title", None) or "").lower() and phrase not in prev_lower
                for h in high_sev[:12]
                for phrase in active_acute_phrases
            )

    if has_shock and threat_level in ("NORMAL", "ELEVATED") and has_new_acute:
        return 0.0, "acute_shock_keyword_alert"

    if changed_b and new_high_sev_count > 0:
        if threat_level == "CRITICAL_SHOCK":
            return min(base, min_geo_itv * 2.0), "critical_shock_active_monitoring_30m"
        if threat_level == "ELEVATED":
            return min(base, min_geo_itv * 4.0), "elevated_threat_new_headlines_1h"
        if new_high_sev_count >= 3:
            return min(base, cp_cooldown_itv), f"high_severity_headline_surge_{new_high_sev_count}"
        crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=cp_cooldown_itv)
        if crossed_cp is not None:
            return min(base, cp_cooldown_itv), f"geopolitical_session_checkpoint_{crossed_cp}"

    if high_sev_count == 0 and threat_level == "NORMAL":
        session_ctx = resolve_global_trading_session(ts)
        if not session_ctx.is_active_western_session:
            return min(max_geo_itv, base * 1.3333), "global_overnight_calm_6h"
        return min(max_geo_itv, base * 1.15), "global_daytime_calm"

    return base, f"standard_geopolitical_ttl_{threat_level.lower()}"


# ============================================================================
# 5. Unified Cached Telemetry & Assessment Loaders (SSOT)
# ============================================================================

def read_first_cached_dict(
    cache_dir: str,
    group_key: str = "",
    *extra_filenames: str,
    read_json_fn: Any = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read and return the first non-empty cached dict and its path from candidate filenames."""
    reader = read_json_fn if callable(read_json_fn) else read_json_dict_recovering
    candidates = (*(get_cache_filenames(group_key) if group_key else ()), *extra_filenames)
    for fname in dict.fromkeys(f for f in candidates if f):
        path = os.path.join(cache_dir, fname)
        try:
            data = reader(path)
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            return data, path
    return None, ""


def _clean_nonbool_str(val: Any, default: str = "") -> str:
    return str(val).strip() if (val is not None and not isinstance(val, bool) and str(val).strip()) else default


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
            fg_data = read_json_dict_recovering(fg_path)
            if not isinstance(fg_data, dict) or not fg_data:
                continue
            fg_val = max(0, min(100, int(round(safe_float(fg_data.get("value"), 50.0)))))
            fg_cls = _clean_nonbool_str(fg_data.get("classification"), "") or _clean_nonbool_str(fg_data.get("sentiment"), "Neutral")
            raw_fg_ts = fg_data.get("ts") if fg_data.get("ts") is not None else fg_data.get("timestamp")
            raw_hist = fg_data.get("historical_values") or ()
            hist_vals = tuple(
                max(0, min(100, int(round(v_f))))
                for v in (raw_hist if isinstance(raw_hist, (list, tuple)) else ())
                if (v_f := safe_float(v, -1.0)) >= 0.0
            )
            return FearGreedSnapshot(
                value=fg_val,
                classification=fg_cls,
                timestamp=safe_pos_float(raw_fg_ts, eval_now),
                historical_values=hist_vals,
                trend_7d_change=int(round(safe_float(fg_data.get("trend_7d_change"), 0.0))),
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

    def _norm_pairs(items: Any) -> Tuple[Tuple[str, float], ...]:
        return tuple(
            (s_clean, safe_float(x[1], 0.0))
            for x in (items if isinstance(items, (list, tuple)) else ())
            if isinstance(x, (list, tuple))
            and len(x) >= 2
            and x[0] is not None
            and (s_clean := str(x[0]).strip())
            and s_clean.isascii()
        )

    eval_now = resolve_now_ts(now_ts)
    for mb_fname in get_cache_filenames("market_breadth_files"):
        mb_path = os.path.join(cache_dir, mb_fname)
        try:
            mb_data = read_json_dict_recovering(mb_path)
            if not isinstance(mb_data, dict) or not mb_data:
                continue
            raw_adv = mb_data.get("advancing_count") if mb_data.get("advancing_count") is not None else mb_data.get("advancing")
            raw_dec = mb_data.get("declining_count") if mb_data.get("declining_count") is not None else mb_data.get("declining")
            raw_tot = mb_data.get("total_count") if mb_data.get("total_count") is not None else mb_data.get("total_symbols")
            raw_ts = mb_data.get("ts") if mb_data.get("ts") is not None else mb_data.get("timestamp")
            return MarketBreadthSnapshot(
                advance_ratio=safe_float(mb_data.get("advance_ratio"), 0.5),
                advancing_count=safe_int(raw_adv, 0),
                declining_count=safe_int(raw_dec, 0),
                total_count=safe_int(raw_tot, 0),
                median_change_pct=safe_float(mb_data.get("median_change_pct"), 0.0),
                mean_change_pct=safe_float(mb_data.get("mean_change_pct"), 0.0),
                dispersion_std=safe_float(mb_data.get("dispersion_std"), 0.0),
                regime=_clean_nonbool_str(mb_data.get("regime"), "NEUTRAL"),
                top_gainers=_norm_pairs(mb_data.get("top_gainers") or ()),
                top_losers=_norm_pairs(mb_data.get("top_losers") or ()),
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
    prefix = _cfg_str("cache_files", "whale_snapshot_prefix", "whale_snapshot_")
    for sym in (list(symbols) if symbols else ["BTCUSDC"]):
        for cand_sym in dict.fromkeys((sym, *cache_symbol_variants(sym))):
            safe_sym = clean_symbol_key(cand_sym)
            if safe_sym:
                cand_w = read_json_dict_recovering(os.path.join(cache_dir, f"{prefix}{safe_sym}.json"))
                if isinstance(cand_w, dict) and cand_w:
                    return cand_w
    return None


def load_cached_sentiment_advisor_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached SentimentAdvisor evaluation dictionary from ``cache_dir``."""
    data, _ = read_first_cached_dict(cache_dir, "sentiment_eval_files")
    if data:
        return {
            **data,
            "market_bias": normalize_market_bias(data.get("market_bias")),
            "risk_level": normalize_risk_level(data.get("risk_level")),
            "recommended_action": normalize_recommended_action(data.get("recommended_action")),
        }
    return None


def load_cached_geopolitical_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached GeopoliticalThreatAnalyzer evaluation dictionary from ``cache_dir``."""
    fname = _cfg_str("cache_files", "geopolitical_eval_file", "geopolitical_eval.json")
    data, _ = read_first_cached_dict(cache_dir, "", fname)
    if data:
        tl = normalize_threat_level(data.get("threat_level"))
        return {
            **data,
            "threat_level": tl,
            "recommended_brake": normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl),
        }
    return None


def lookup_symbol_cache_items(
    mapping: Any,
    symbol: Any,
    *,
    cand_keys: Optional[Sequence[Any]] = None,
    allow_base_fallback: bool = True,
    require_list: bool = False,
) -> Any:
    """Resolve symbol entries from a cache dictionary across symbol variants and optional base-asset fallback."""
    if not isinstance(mapping, dict) or not isinstance(symbol, str) or not symbol.strip():
        return None
    sym_u = symbol.strip().upper()
    sym_clean = clean_symbol_key(sym_u)
    sym_canon = canonical_symbol_key(sym_u)
    cands: Sequence[Any] = (
        cand_keys
        if cand_keys is not None
        else (symbol, sym_u, sym_clean, *( [sym_canon] if sym_canon else [] ), *cache_symbol_variants(symbol))
    )
    for cand in cands:
        if not cand:
            continue
        items = casefold_get(mapping, cand)
        if items and (not require_list or isinstance(items, list)):
            return items
    if allow_base_fallback:
        s_base = canonical_base_asset(symbol)
        if s_base:
            for k, v in mapping.items():
                if v and (not require_list or isinstance(v, list)) and canonical_base_asset(str(k)) == s_base:
                    return v
    return None


def scan_latest_long_trend_value(
    entries: Any,
    parser: Callable[[Dict[str, Any]], Any],
    *,
    stop_on_none: bool = True,
) -> Any:
    """Scan a symbol's long-trend entries list (or dict) from newest to oldest and return the first parsed value."""
    if isinstance(entries, dict):
        return parser(entries)
    if isinstance(entries, list):
        for entry in reversed(entries):
            if entry is None:
                if stop_on_none:
                    break
                continue
            if isinstance(entry, dict):
                val = parser(entry)
                if val is not None:
                    return val
    return None


class CachedTopicAssessmentAgent:
    """Unified base class for stateful topic-backed macro/sentiment AI assessment agents."""

    TOPIC_KEY: str = ""
    DEFAULT_STATE_FILE: str = ""
    DEFAULT_CACHE_FILENAME: str = ""
    STATE_FILE_ATTR: str = "state_file"
    AGENT_LABEL: str = "assessment"
    _logger: logging.Logger = logger

    gemini_client: Any
    cache_ttl_sec: float
    _cached_assessment: Optional[Any]
    _last_eval_ts: float

    def _effective_state_file(self) -> str:
        raw_f = getattr(self, self.STATE_FILE_ATTR, None) or self.DEFAULT_STATE_FILE
        if raw_f == self.DEFAULT_STATE_FILE and self.DEFAULT_CACHE_FILENAME:
            return cache_path(self.DEFAULT_CACHE_FILENAME)
        return str(raw_f)

    _effective_cache_file = _effective_state_file

    def _deserialize_from_disk(self, data: Dict[str, Any]) -> Any:
        return self._build_assessment(data)  # type: ignore[attr-defined]

    def _on_loaded_from_disk(self, data: Dict[str, Any], assessment: Any) -> None:
        pass

    def _serialize_for_disk(self, assessment: Any) -> Dict[str, Any]:
        return asdict(assessment)

    def _load_from_disk(self) -> None:
        path = self._effective_state_file()
        if not path:
            return
        try:
            data = read_json_dict(path, default=None, recover_previous=True)
            if not isinstance(data, dict):
                return
            assessment = self._deserialize_from_disk(data)
            self._cached_assessment = assessment
            self._last_eval_ts = float(getattr(assessment, "ts", 0.0) or 0.0)
            self._on_loaded_from_disk(data, assessment)
        except Exception as e:
            self._logger.debug("Could not load cached %s state %s: %s", self.AGENT_LABEL, path, e)

    def _save_to_disk(self, assessment: Any) -> None:
        path = self._effective_state_file()
        if not path:
            return
        try:
            save_json_state(path, self._serialize_for_disk(assessment), indent=2)
        except Exception as e:
            self._logger.warning("Could not persist %s state to %s: %s", self.AGENT_LABEL, path, e)

    def _query_topic(
        self,
        prompt: str,
        *,
        subtitle: Optional[str] = None,
        timeout_sec: float = 120.0,
    ) -> Optional[Dict[str, Any]]:
        from intelligence.sentiment.gemini_client import query_topic_json

        return query_topic_json(
            self.gemini_client,
            self.TOPIC_KEY,
            prompt,
            subtitle=subtitle,
            timeout_sec=timeout_sec,
        )

    def _commit_topic_assessment(
        self,
        assessment: Any,
        now: float,
        *,
        record_crisis_terms: bool = False,
        crisis_summary: str = "",
        crisis_items: Sequence[str] = (),
        is_acute_shock: bool = False,
    ) -> Any:
        if record_crisis_terms and (crisis_summary or crisis_items):
            record_dynamic_crisis_terms_from_assessment(
                crisis_summary,
                crisis_items,
                is_acute_shock=is_acute_shock,
                cache_dir=os.path.dirname(os.path.abspath(self._effective_state_file())),
                now_ts=now,
            )
        self._cached_assessment = assessment
        self._last_eval_ts = now
        self._save_to_disk(assessment)
        return assessment
