"""Shared Daemon Runtime, Heartbeat, Bounded Step Execution, and Collector/Agent Base Classes (SSOT)."""
from __future__ import annotations

from dataclasses import asdict
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from botcore import (
    build_heartbeat_payload,
    exact_pos_int,
    fetch_public_json as _fetch_public_json,
    is_cache_fresh,
    is_interval_due,
    obj_attr,
    obj_float,
    parse_dotenv,
    required_float_env,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float as safe_nonneg_ts,
    safe_pos_float,
)
from cachepaths import (
    cache_path,
    canonical_base_asset,
    resolve_cachedb_filepath,
)
from state_io import (
    atomic_write_json as _default_atomic_write_json,
    discard_generation_file as _default_discard_generation_file,
    read_json_dict as _default_read_json_dict,
    read_json_dict_recovering as _default_read_json_dict_recovering,
    save_json_state as _default_save_json_state,
)

fetch_public_json = _fetch_public_json
atomic_write_json = _default_atomic_write_json
discard_generation_file = _default_discard_generation_file
read_json_dict = _default_read_json_dict
read_json_dict_recovering = _default_read_json_dict_recovering
save_json_state = _default_save_json_state
logger = logging.getLogger("intelligence.macro_sentiment_ssot")
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ACTIVE_BOUNDED_STEPS: Dict[str, threading.Thread] = {}
_ACTIVE_BOUNDED_LOCK = threading.Lock()


def _resolve_daemon_helper(
    name: str,
    local_fn: Any,
    default_fn: Any,
    owner_mod_name: Optional[str] = None,
) -> Any:
    if owner_mod_name:
        owner_mod = sys.modules.get(owner_mod_name)
        cand_owner = getattr(owner_mod, name, None) if owner_mod is not None else None
        if callable(cand_owner) and cand_owner is not default_fn:
            return cand_owner
    ssot_mod = sys.modules.get("intelligence.macro_sentiment_ssot")
    cand_ssot = getattr(ssot_mod, name, None) if ssot_mod is not None else None
    if callable(cand_ssot) and cand_ssot is not default_fn:
        return cand_ssot
    if callable(local_fn) and local_fn is not default_fn:
        return local_fn
    return local_fn


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
        try:
            from instruments_config import discover_hyperliquid_spot
            for hl_sym, _, _ in discover_hyperliquid_spot(_ROOT_DIR):
                _add_sym(hl_sym, skip_eq=True)
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
                env_map = parse_dotenv(krk_cfg)
                for cfg_k in ("KRAKEN_TRADABLE_PAIRS", "KRAKEN_CACHE_PAIRS"):
                    val = env_map.get(cfg_k, "")
                    for item in (val.split(",") if val else ()):
                        _add_sym(item, skip_eq=True, unique_base=False)
        except Exception:
            pass
    return out if out else list(default)


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


def inspect_daemon_heartbeat(
    cache_dir: str,
    filenames: Sequence[str],
    *,
    max_alive_age_sec: float = 120.0,
    include_symbols: bool = False,
) -> Dict[str, Any]:
    """Inspect a background daemon's heartbeat file(s) and check PID liveness."""
    _read_dict = _resolve_daemon_helper(
        "read_json_dict_recovering", read_json_dict_recovering, _default_read_json_dict_recovering
    )
    data = None
    for fname in filenames:
        cand_data = _read_dict(os.path.join(cache_dir, fname))
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
        _save_state = _resolve_daemon_helper("save_json_state", save_json_state, _default_save_json_state)
        try:
            for raw_target in paths:
                _save_state(raw_target, payload, indent=2)
        except Exception as exc:
            log.debug("Failed writing daemon heartbeat: %s", exc)
        return

    _atomic_write = _resolve_daemon_helper("atomic_write_json", atomic_write_json, _default_atomic_write_json)
    _discard_gen = _resolve_daemon_helper(
        "discard_generation_file", discard_generation_file, _default_discard_generation_file
    )
    for raw_target in paths:
        hb_target = str(resolve_cachedb_filepath(raw_target))
        try:
            _atomic_write(hb_target, payload, indent=2)
        except Exception as exc:
            log.warning("Primary atomic heartbeat write failed for %s: %s; attempting fallback", hb_target, exc)
            try:
                _discard_gen(hb_target)
                _atomic_write(hb_target, payload, indent=2)
            except Exception as fallback_err:
                log.error("Fallback heartbeat write failed for %s: %s", hb_target, fallback_err)


def write_daemon_heartbeat(
    heartbeat_path: str,
    status: Any,
    *,
    symbols: Optional[Sequence[str]] = None,
    intervals: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    use_save_json_state: bool = False,
    logger_inst: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Build canonical heartbeat payload via ``botcore.build_heartbeat_payload`` and persist atomically."""
    payload = build_heartbeat_payload(
        status,
        symbols=list(symbols) if symbols is not None else None,
        intervals=intervals,
        extra=extra,
    )
    write_daemon_heartbeat_files(
        (heartbeat_path,),
        payload,
        use_save_json_state=use_save_json_state,
        logger_inst=logger_inst,
    )
    return payload


def run_scheduled_step(
    obj: Any,
    *,
    current_ts: float,
    ts_attr: str,
    interval_sec: float,
    fn: Callable[..., Any],
    fn_args: Tuple[Any, ...] = (),
    force: bool = False,
    bounded: bool = False,
    timeout_sec: float = 15.0,
    label: str = "telemetry_step",
    results: Optional[Dict[str, Any]] = None,
    result_key: Optional[str] = None,
    on_done: Optional[Callable[[], None]] = None,
) -> bool:
    """Execute ``fn(*fn_args)`` when ``interval_sec`` has elapsed since ``getattr(obj, ts_attr)``, updating timestamp or retry backoff."""
    last_ts = safe_nonneg_ts(getattr(obj, ts_attr, 0.0), 0.0)
    if not is_interval_due(current_ts, last_ts, interval_sec, force=force):
        return False
    runner = getattr(obj, "_run_step_bounded", None)
    ok = (
        runner(fn, *fn_args, timeout_sec=timeout_sec, label=label)
        if (bounded and callable(runner))
        else bool(fn(*fn_args))
    )
    if ok:
        if isinstance(results, dict) and result_key:
            results[result_key] = True
        setattr(obj, ts_attr, current_ts)
    else:
        setattr(obj, ts_attr, compute_step_retry_ts(last_ts, current_ts, interval_sec))
    if on_done is not None:
        on_done()
    return ok


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


def configure_daemon_cli_logging(
    verbose: bool = False,
    fmt: str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
) -> None:
    """Configure standard CLI logging format and verbosity for intelligence daemons."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
        from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
        from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector

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


class CachedHttpJsonCollector:
    """Shared base class for HTTP JSON collectors with in-memory TTL caching and stale fallback."""

    _USER_AGENT: str = "mptrade-sentiment/1.0"
    _DEFAULT_TIMEOUT_SEC: float = 5.0
    _COLLECTOR_LABEL: str = "HTTP JSON snapshot"
    _LOGGER: logging.Logger = logger

    def __init__(
        self,
        cache_ttl_sec: float,
        default_ttl_sec: float,
        endpoint_url: Optional[str],
        default_endpoint_fn: Callable[[], str],
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, default_ttl_sec)
        self.endpoint_url = (endpoint_url.strip() if isinstance(endpoint_url, str) else "") or default_endpoint_fn()
        self._cached_snapshot: Any = None
        self._last_fetch_ts: float = 0.0

    def _parse_fetched_payload(self, data: Any, now: float) -> Any:
        raise NotImplementedError

    def fetch(self, force_refresh: bool = False, now_ts: Optional[float] = None) -> Any:
        """Fetch the latest snapshot from ``self.endpoint_url``, respecting cache TTL with stale fallback."""
        now = resolve_now_ts(now_ts)
        if self._cached_snapshot and is_cache_fresh(
            self._last_fetch_ts, now, self.cache_ttl_sec, force_refresh=force_refresh
        ):
            return self._cached_snapshot
        try:
            fetch_fn = _resolve_daemon_helper(
                "fetch_public_json", fetch_public_json, _fetch_public_json, self.__class__.__module__
            )
            data = fetch_fn(
                self.endpoint_url,
                timeout=self._DEFAULT_TIMEOUT_SEC,
                user_agent=self._USER_AGENT,
            )
            snapshot = self._parse_fetched_payload(data, now)
            if snapshot:
                self._cached_snapshot, self._last_fetch_ts = snapshot, now
                return snapshot
        except Exception as exc:
            self._LOGGER.warning(
                "Failed to fetch %s: %s. Using cached fallback if available.",
                self._COLLECTOR_LABEL,
                exc,
            )
        return self._cached_snapshot


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
            _read_fn = _resolve_daemon_helper(
                "read_json_dict", read_json_dict, _default_read_json_dict, self.__class__.__module__
            )
            data = _read_fn(path, default=None, recover_previous=True)
            if not isinstance(data, dict):
                return
            assessment = self._deserialize_from_disk(data)
            self._cached_assessment = assessment
            self._last_eval_ts = safe_nonneg_ts(getattr(assessment, "ts", 0.0), 0.0)
            self._on_loaded_from_disk(data, assessment)
        except Exception as e:
            self._logger.debug("Could not load cached %s state %s: %s", self.AGENT_LABEL, path, e)

    def _save_to_disk(self, assessment: Any) -> None:
        path = self._effective_state_file()
        if not path:
            return
        try:
            _save_fn = _resolve_daemon_helper(
                "save_json_state", save_json_state, _default_save_json_state, self.__class__.__module__
            )
            _save_fn(path, self._serialize_for_disk(assessment), indent=2)
        except Exception as e:
            self._logger.warning("Could not persist %s state to %s: %s", self.AGENT_LABEL, path, e)

    def _query_topic(
        self,
        prompt: str,
        *,
        subtitle: Optional[str] = None,
        timeout_sec: float = 120.0,
    ) -> Optional[Dict[str, Any]]:
        from intelligence.sentiment.gemini_client import query_topic_json as _default_query_topic_json

        _query_fn = _resolve_daemon_helper(
            "query_topic_json", _default_query_topic_json, _default_query_topic_json, self.__class__.__module__
        )
        return _query_fn(
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
            from intelligence.macro.lexicon_ssot import (
                record_dynamic_crisis_terms_from_assessment as _default_record_crisis_terms,
            )

            _record_fn = _resolve_daemon_helper(
                "record_dynamic_crisis_terms_from_assessment",
                _default_record_crisis_terms,
                _default_record_crisis_terms,
                self.__class__.__module__,
            )
            _record_fn(
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


__all__ = [
    "CachedHttpJsonCollector",
    "CachedTopicAssessmentAgent",
    "ExternalTelemetryMixin",
    "_ACTIVE_BOUNDED_LOCK",
    "_ACTIVE_BOUNDED_STEPS",
    "compute_step_retry_ts",
    "configure_daemon_cli_logging",
    "detect_active_binance_symbols",
    "inspect_daemon_heartbeat",
    "prune_automated_cli_threads",
    "run_daemon_loop",
    "run_scheduled_step",
    "run_step_bounded",
    "write_daemon_heartbeat",
    "write_daemon_heartbeat_files",
]
