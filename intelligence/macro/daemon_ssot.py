"""Shared Daemon Runtime, Heartbeat, Bounded Step Execution, and Collector/Agent Base Classes (SSOT)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
import importlib
import inspect
import logging
import math
import os
import signal as _signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from botcore import (
    _parse_bool_flag,
    add_store_true_cli_args,
    build_heartbeat_payload,
    exact_pos_int,
    fetch_public_json as _fetch_public_json,
    is_cache_fresh,
    is_interval_due,
    load_dotenv,
    normalize_epoch_sec,
    obj_attr,
    obj_float,
    parse_dotenv,
    required_float_env,
    resolve_now_ts,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
    single_instance,
    split_csv,
)
from cachepaths import (
    cache_path,
    cachedb_root_for_file,
    canonical_base_asset,
    resolve_cachedb_filepath,
)
from instrument_registry import (
    binance_symbols,
    discover_hyperliquid_spot,
    is_equity_symbol,
    tracked_price_instruments,
)
from intelligence.external.collectors.base_collector import SyncModeMixin
from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetryCollector
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
from intelligence.external.collectors.whale_positioning_collector import WhalePositioningCollector
from intelligence.macro.lexicon_ssot import (
    record_dynamic_crisis_terms_from_assessment as _default_record_crisis_terms,
)
from state_io import (
    atomic_write_json as _default_atomic_write_json,
    discard_generation_file as _default_discard_generation_file,
    encode_json_payload,
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
    return resolve_patched_attr(
        name,
        local_fn,
        defaults=(default_fn,),
        modules=(owner_mod_name or "", "intelligence.macro_sentiment_ssot"),
        require_callable=True,
    )


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
        if not s_clean or (skip_eq and is_equity_symbol(s_clean)) or s_clean in out:
            return
        base = canonical_base_asset(s_clean)
        if unique_base and (not base or base in seen_bases):
            return
        out.append(s_clean)
        seen_bases.add(base)

    if include_kraken_cache:
        try:
            all_syms, _, _ = tracked_price_instruments()
            for s in all_syms:
                _add_sym(s, skip_eq=False)
        except Exception:
            pass
        try:
            for hl_sym, _, _ in discover_hyperliquid_spot(_ROOT_DIR):
                _add_sym(hl_sym, skip_eq=True)
        except Exception:
            pass
    if not out:
        try:
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
                    for item in split_csv(env_map.get(cfg_k, ""), upper=True):
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
        ts = safe_nonneg_ts(data.get("ts", 0.0), 0.0)
        age = max(0.0, time.time() - ts) if ts > 0.0 else math.inf
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


def normalize_cycle_args(force: Any = False, now: Optional[float] = None) -> Tuple[bool, float]:
    """Normalize ``(force, now)`` arguments for daemon ``run_cycle`` entrypoints."""
    return _parse_bool_flag(force, default=False), resolve_now_ts(now)


def finalize_daemon_cycle(
    daemon_obj: Any,
    results: Dict[str, Any],
    *,
    current_ts: float,
    force: bool = False,
    hb_interval_sec: float = 60.0,
    logger_inst: Optional[logging.Logger] = None,
    hb_msg: Optional[str] = None,
    hb_args: Tuple[Any, ...] = (),
) -> Dict[str, Any]:
    """Emit periodic stdout heartbeat log, write daemon heartbeat file, and return ``results``."""
    last_hb_ts = safe_nonneg_ts(getattr(daemon_obj, "_last_stdout_hb_ts", 0.0), 0.0)
    if hb_msg and is_interval_due(current_ts, last_hb_ts, hb_interval_sec, force=force):
        (logger_inst or logger).info(hb_msg, *hb_args)
        daemon_obj._last_stdout_hb_ts = current_ts
    daemon_obj.write_heartbeat(results)
    return results


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
    log = logger_inst or logger
    poll_sec = safe_pos_float(poll_interval_sec, default_poll_sec)
    daemon_obj.running = True

    def _handle_exit(signum, frame):
        if not getattr(daemon_obj, "running", False):
            return
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
        level=logging.DEBUG if _parse_bool_flag(verbose, default=False) else logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_and_configure_daemon_cli(
    parser: Any,
    *,
    fmt: str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
) -> Any:
    """Parse CLI arguments, load environment defaults, and configure standard daemon logging."""
    args = parser.parse_args()
    try:
        load_dotenv()
    except Exception:
        pass
    configure_daemon_cli_logging(getattr(args, "verbose", False), fmt=fmt)
    return args


def add_external_telemetry_cli_args(
    parser: Any,
    *,
    ob_default: Optional[float] = None,
    deriv_default: Optional[float] = None,
    whale_default: Optional[float] = None,
    skip_flag: Optional[str] = None,
    skip_help: str = "",
    extra_float_flags: tuple[tuple[str, str], ...] = (),
) -> Any:
    """Register shared CLI arguments for external telemetry collector daemons."""
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to monitor.")
    add_store_true_cli_args(
        parser,
        ("--once", "Run a single update cycle and exit."),
        ("--skip-external", "Skip external orderbook and derivatives telemetry."),
        *(((skip_flag, skip_help),) if skip_flag else ()),
        (("--verbose", "-v"), "Enable verbose DEBUG logging."),
    )
    for flag_name, default_val, flag_help in (
        ("--ob-interval", ob_default, "Orderbook refresh interval in seconds."),
        ("--deriv-interval", deriv_default, "Derivatives refresh interval in seconds."),
        ("--whale-interval", whale_default, "Whale positioning refresh interval in seconds."),
        *((name, None, hlp) for name, hlp in extra_float_flags),
    ):
        parser.add_argument(flag_name, type=float, default=default_val, help=flag_help)
    return parser


def add_macro_analyzer_cli_args(parser: Any) -> Any:
    """Register CLI arguments for ``MacroAnalyzer`` daemon."""
    add_store_true_cli_args(
        parser,
        ("--once", "Run a single evaluation cycle and exit."),
        ("--force", "Force execution of LLM assessment regardless of cache TTL."),
        (("--verbose", "-v"), "Enable verbose DEBUG logging."),
    )
    for flag_name, flag_help in (
        ("--macro-llm-interval", "Geopolitical LLM evaluation interval in seconds."),
        ("--advisor-interval", "Market advisor synthesis interval in seconds."),
    ):
        parser.add_argument(flag_name, type=float, default=None, help=flag_help)
    return parser


def run_daemon_cli_once_or_start(
    instance: Any,
    *,
    once: bool,
    force: bool = True,
    once_msg: str = "Running single update cycle...",
    logger_inst: Optional[logging.Logger] = None,
    print_json_on_once: bool = False,
    exit_on_once: bool = False,
    single_instance_name: Optional[str] = None,
) -> None:
    """Execute a single daemon cycle when ``--once`` is set, otherwise start the daemon loop."""
    log = logger_inst or logger
    if _parse_bool_flag(once, default=False):
        log.info(once_msg)
        res = instance.run_cycle(force=_parse_bool_flag(force, default=True))
        if _parse_bool_flag(print_json_on_once, default=False):
            print(encode_json_payload(res, indent=2))
        else:
            log.info("Cycle completed.")
        if _parse_bool_flag(exit_on_once, default=False):
            sys.exit(0)
    else:
        if single_instance_name:
            single_instance(single_instance_name)
        instance.start()


def prune_automated_cli_threads(logger_inst: Optional[logging.Logger] = None) -> int:
    """Prune automated CLI threads older than configured retention limit."""
    log = logger_inst or logger
    try:
        prune_old_cli_threads = importlib.import_module("orchestratorOS.admin.prune_llm_threads").prune_old_cli_threads
        resolve_bot_cli_base = importlib.import_module("orchestratorOS.agy.paths").resolve_bot_cli_base

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

    def _init_external_collectors(
        self, logger_inst: logging.Logger, *, skip_external: Any = None
    ) -> None:
        if skip_external is not None:
            self.skip_external = _parse_bool_flag(skip_external, default=False)
        self.running = False
        self._telemetry_logger = logger_inst
        self._last_orderbook_ts: Dict[str, float] = {}
        self._last_derivatives_ts: Dict[str, float] = {}
        self._last_whale_ts: Dict[str, float] = {}
        self._last_stdout_hb_ts: float = 0.0
        ob_ttl = max(5.0, self.orderbook_interval_sec * 0.8)
        deriv_ttl = max(10.0, self.derivatives_interval_sec * 0.8)
        whale_ttl = max(15.0, self.whale_interval_sec * 0.8)
        self.orderbook_collector = OrderbookDepthCollector(
            cache_ttl_sec=ob_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=ob_ttl,
            start_sync=True,
        )
        self.derivatives_collector = DerivativesTelemetryCollector(
            cache_ttl_sec=deriv_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=deriv_ttl,
            start_sync=True,
        )
        self.whale_collector = WhalePositioningCollector(
            cache_ttl_sec=whale_ttl,
            cache_dir=self.cache_dir,
            disk_ttl_sec=whale_ttl,
            start_sync=True,
        )

    @staticmethod
    def _run_step_bounded(
        fn: Callable[..., Any],
        *args: Any,
        timeout_sec: float = 15.0,
        label: str = "telemetry_step",
    ) -> bool:
        return run_step_bounded(fn, *args, timeout_sec=timeout_sec, label=label)

    @staticmethod
    def _call_collector_fetch(
        collector: Any, *, force_refresh: bool, now_ts: Optional[float] = None
    ) -> Any:
        force_b = _parse_bool_flag(force_refresh, default=False)
        eff_now = normalize_epoch_sec(now_ts, default=None)
        fetch_fn = collector.fetch
        if eff_now is not None:
            accepts_now_ts = False
            try:
                params = inspect.signature(fetch_fn).parameters
                accepts_now_ts = "now_ts" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                accepts_now_ts = False
            if accepts_now_ts:
                return fetch_fn(force_refresh=force_b, now_ts=eff_now)
        return fetch_fn(force_refresh=force_b)

    def _new_external_cycle_results(
        self, current_ts: float, **extra_fields: Any
    ) -> Dict[str, Any]:
        return {
            "ts": current_ts,
            "symbols": list(getattr(self, "symbols", ())),
            "orderbook_updated": 0,
            "derivatives_updated": 0,
            "whale_updated": 0,
            **extra_fields,
        }

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
            cycle_now = getattr(self, "_active_cycle_ts", None)
            if cycle_now is not None:
                try:
                    snap = collector.fetch(symbol, allow_network=True, now=cycle_now)
                except TypeError:
                    snap = collector.fetch(symbol, allow_network=True)
            else:
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


class CachedHttpJsonCollector(SyncModeMixin, ABC):
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
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_float(cache_ttl_sec, default=default_ttl_sec)
        self.endpoint_url = (endpoint_url.strip() if isinstance(endpoint_url, str) else "") or default_endpoint_fn()
        self._cached_snapshot: Any = None
        self._last_fetch_ts: float = 0.0
        self._init_sync_mode(start_sync)

    @abstractmethod
    def _parse_fetched_payload(self, data: Any, now: float) -> Any:
        """Parse the fetched HTTP JSON payload into a domain snapshot."""

    def fetch(
        self,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
        *,
        allow_network: Optional[bool] = None,
    ) -> Any:
        """Fetch the latest snapshot from ``self.endpoint_url``, respecting cache TTL with stale fallback."""
        now, should_fetch, ready = self._check_in_memory_fetch_cache(
            force_refresh=force_refresh,
            now_ts=now_ts,
            allow_network=allow_network,
        )
        if not should_fetch:
            return ready
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

    def read_snapshot(self, *, now_ts: Optional[float] = None) -> Any:
        """Read-only consumer entrypoint: return cached in-memory snapshot with 0 network I/O."""
        return self.fetch(force_refresh=False, now_ts=now_ts, allow_network=False)

    def collect(
        self,
        cachedb_dir: str = "",
        symbol: str = "*",
        *,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
        allow_network: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Polymorphic ``TelemetryCacheCollectorInterface`` adapter delegating to ``fetch()``."""
        return self.fetch(force_refresh=force_refresh, now_ts=now_ts, allow_network=allow_network)


class CachedTopicAssessmentAgent(SyncModeMixin):
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

    def _init_topic_agent_state(
        self,
        gemini_client: Any,
        cache_ttl_sec: Optional[float],
        default_ttl_sec: float,
        state_file: str,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        self._init_sync_mode(start_sync)
        self.gemini_client = gemini_client
        self.cache_ttl_sec = safe_nonneg_float(cache_ttl_sec, default=default_ttl_sec)
        setattr(self, self.STATE_FILE_ATTR, state_file)
        self._cached_assessment = None
        self._last_eval_ts = 0.0
        self._load_from_disk()

    _resolve_allow_llm = SyncModeMixin._resolve_allow_sync_flag

    def _effective_state_file(self) -> str:
        raw_f = getattr(self, self.STATE_FILE_ATTR, None) or self.DEFAULT_STATE_FILE
        if raw_f == self.DEFAULT_STATE_FILE and self.DEFAULT_CACHE_FILENAME:
            return cache_path(self.DEFAULT_CACHE_FILENAME)
        return str(raw_f)

    def _effective_cache_dir(self) -> str:
        return cachedb_root_for_file(os.path.abspath(self._effective_state_file()))

    def _load_from_disk(self) -> None:
        if not (path := self._effective_state_file()):
            return
        try:
            _read_fn = _resolve_daemon_helper(
                "read_json_dict", read_json_dict, _default_read_json_dict, self.__class__.__module__
            )
            data = _read_fn(path, default=None, recover_previous=True)
            if not isinstance(data, dict):
                return
            assessment = self._build_assessment(data)  # type: ignore[attr-defined]
            self._cached_assessment = assessment
            self._last_eval_ts = safe_nonneg_ts(getattr(assessment, "ts", 0.0), 0.0)
            if hasattr(self, "_last_headlines_digest"):
                raw_digest = data.get("headlines_digest")
                self._last_headlines_digest = raw_digest if isinstance(raw_digest, str) else ""
        except Exception as e:
            self._logger.debug("Could not load cached %s state %s: %s", self.AGENT_LABEL, path, e)

    def _save_to_disk(self, assessment: Any) -> None:
        if not (path := self._effective_state_file()):
            return
        try:
            _save_fn = _resolve_daemon_helper(
                "save_json_state", save_json_state, _default_save_json_state, self.__class__.__module__
            )
            payload = asdict(assessment)
            if hasattr(self, "_last_headlines_digest"):
                payload["headlines_digest"] = getattr(self, "_last_headlines_digest", "")
            _save_fn(path, payload, indent=2)
        except Exception as e:
            self._logger.warning("Could not persist %s state to %s: %s", self.AGENT_LABEL, path, e)

    def _query_topic(
        self,
        prompt: str,
        *,
        subtitle: Optional[str] = None,
        timeout_sec: float = 120.0,
    ) -> Optional[Dict[str, Any]]:
        _default_query_topic_json = importlib.import_module("intelligence.sentiment.gemini_client").query_topic_json

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
                cache_dir=self._effective_cache_dir(),
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
    "add_external_telemetry_cli_args",
    "compute_step_retry_ts",
    "configure_daemon_cli_logging",
    "detect_active_binance_symbols",
    "inspect_daemon_heartbeat",
    "normalize_cycle_args",
    "parse_and_configure_daemon_cli",
    "prune_automated_cli_threads",
    "run_daemon_cli_once_or_start",
    "run_daemon_loop",
    "run_scheduled_step",
    "run_step_bounded",
    "write_daemon_heartbeat",
    "write_daemon_heartbeat_files",
]
