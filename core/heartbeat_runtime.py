"""core/heartbeat_runtime.py -- Heartbeats, procs.conf manifest, RuntimeServices, and module loaders."""
from __future__ import annotations

import asyncio
import atexit
import contextlib
from enum import Enum
import importlib
import importlib.machinery
import importlib.util
import math
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time as _time
import types

from cachepaths import (
    _parse_bool_flag,
    cache_path,
    exact_int,
    resolve_cachedb_filepath,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from core.config import _REPO_ROOT, any_bool_env, bool_env, int_env
from core.math_coercion import _UNSET, clean_json_value, split_csv
from core.time_market import is_us_rth_open
import state_io as _state_io


def sanitize_heartbeat_status(status: object) -> tuple[float, dict]:
    """Sanitize a daemon/module cycle status dict, replacing non-finite floats with None."""
    raw = status if isinstance(status, dict) else {}
    hb_ts = safe_pos_float(raw.get("ts"), _time.time())
    safe_status = {
        k: (hb_ts if k == "ts" else (None if isinstance(v, float) and safe_float(v, default=None) is None else v))
        for k, v in raw.items()
    }
    return hb_ts, safe_status


def build_heartbeat_payload(
    status: object = None,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
    pid: int | None = None,
) -> dict[str, object]:
    """Build the canonical module heartbeat payload used across all bots, daemons, and supervisors."""
    hb_ts, safe_status = sanitize_heartbeat_status(status)
    payload: dict[str, object] = {
        "pid": os.getpid() if pid is None else (exact_int(pid, default=os.getpid()) or os.getpid()),
        "status": "alive",
        "ts": hb_ts,
    }
    if symbols is not None:
        payload["symbols"] = list(symbols)
    payload["intervals"] = clean_json_value(intervals) if isinstance(intervals, dict) else {}
    payload["last_run"] = safe_status
    if isinstance(extra, dict):
        cleaned_extra = clean_json_value(extra)
        if isinstance(cleaned_extra, dict):
            payload.update(cleaned_extra)
    return payload


def touch_heartbeat_file(
    path: str | os.PathLike,
    *,
    last_ts: float = -math.inf,
    interval_sec: float = 30.0,
    force: object = False,
    now: float | None = None,
    status: object = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
) -> float:
    """Write canonical heartbeat JSON and touch ``path`` if ``force`` is truthy or ``interval_sec`` has elapsed."""
    force_flag = _parse_bool_flag(force, default=False)
    now_val = safe_float(now, default=None)
    now_m = _time.monotonic() if now_val is None else now_val
    if not force_flag and 0.0 <= (now_m - last_ts) < interval_sec:
        return last_ts
    try:
        hb_path = str(path)
        payload = build_heartbeat_payload(
            status,
            symbols=symbols,
            intervals=intervals if intervals is not None else {"heartbeat": interval_sec},
            extra=extra,
        )
        write_fn = _state_io.atomic_write_json
        if (
            getattr(write_fn, "__module__", None) == "state_io"
            and getattr(write_fn, "__name__", None) == "atomic_write_json"
        ):
            write_fn(hb_path, payload, separators=(",", ":"), fsync=False)
        else:
            write_fn(hb_path, payload)
        os.utime(hb_path, None)
    except OSError:
        return last_ts
    return now_m


_MODULE_HEARTBEAT_LOCK = globals().get("_MODULE_HEARTBEAT_LOCK") or threading.Lock()
_MODULE_HEARTBEAT_LAST_TS: dict[str, float] = globals().get("_MODULE_HEARTBEAT_LAST_TS") or {}
_MODULE_HEARTBEAT_META: dict[str, dict[str, object]] = globals().get("_MODULE_HEARTBEAT_META") or {}


def touch_module_heartbeat(
    module_name: str,
    *,
    interval_sec: float = 30.0,
    force: object = False,
    now: float | None = None,
    status: object = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
    path: str | os.PathLike | None = None,
) -> float:
    """Thread-safe shared heartbeat writer for any bot or fleet daemon."""
    clean_name = str(module_name or "").strip() if not isinstance(module_name, bool) else ""
    if not clean_name and path is None:
        return -math.inf
    if path is not None:
        hb_path = str(path)
    else:
        fname = clean_name if clean_name.endswith(".heartbeat") else f"{clean_name}.heartbeat"
        try:
            bc = sys.modules.get("botcore")
            cp_fn = bc.__dict__.get("cache_path", cache_path) if bc is not None else cache_path
            hb_path = str(resolve_cachedb_filepath(cp_fn(fname)))
        except Exception:
            hb_path = os.path.join(_REPO_ROOT, "cachedb", "system", "heartbeat", fname)
    with _MODULE_HEARTBEAT_LOCK:
        meta = _MODULE_HEARTBEAT_META.setdefault(hb_path, {})
        if status is not None:
            meta["status"] = status
        if symbols is not None:
            prev_syms = meta.get("symbols")
            if not (isinstance(prev_syms, (list, tuple)) and len(prev_syms) > 1 and len(symbols) == 1):
                meta["symbols"] = list(symbols)
        if intervals is not None:
            meta["intervals"] = intervals
        if extra is not None:
            meta["extra"] = extra
        prev_ts = _MODULE_HEARTBEAT_LAST_TS.get(hb_path, -math.inf)
        updated_ts = touch_heartbeat_file(
            hb_path,
            last_ts=prev_ts,
            interval_sec=interval_sec,
            force=force,
            now=now,
            status=status if status is not None else meta.get("status"),
            symbols=meta.get("symbols") if symbols is not None or "symbols" in meta else None,  # type: ignore[arg-type]
            intervals=intervals if intervals is not None else meta.get("intervals"),  # type: ignore[arg-type]
            extra=extra if extra is not None else meta.get("extra"),  # type: ignore[arg-type]
        )
        _MODULE_HEARTBEAT_LAST_TS[hb_path] = updated_ts
        return updated_ts


def evaluate_heartbeat_files_freshness(
    hb_paths: list[str] | tuple[str, ...],
    stale_threshold_s: float,
    now_ts: float,
    *,
    fallback_start_ts: float | None = None,
) -> tuple[bool, float, float, str]:
    """Evaluate heartbeat freshness across ``hb_paths``."""
    if not hb_paths:
        return True, 0.0, 0.0, ""
    hb_only = [p for p in hb_paths if str(p).strip().endswith(".heartbeat")]
    effective_paths = hb_only if hb_only else hb_paths
    mtimes: list[float] = []
    missing_any, stale_file, max_stale_duration = False, "", 0.0
    start_ts = fallback_start_ts if fallback_start_ts is not None else now_ts

    def _mark_missing_or_invalid(hbp: str) -> None:
        nonlocal missing_any, max_stale_duration, stale_file
        missing_any = True
        dur = max(0.0, now_ts - start_ts)
        if stale_threshold_s > 0 and (not stale_file or dur >= max_stale_duration):
            max_stale_duration, stale_file = dur, hbp

    for raw_hbp in effective_paths:
        hbp = str(resolve_cachedb_filepath(raw_hbp))
        if os.path.exists(hbp):
            try:
                mt = safe_pos_float(os.path.getmtime(hbp), default=None)
                if mt is None or mt > now_ts + 60.0:
                    _mark_missing_or_invalid(hbp)
                    continue
                mtimes.append(mt)
                dur = now_ts - mt
                if stale_threshold_s > 0 and dur > stale_threshold_s and dur > max_stale_duration:
                    max_stale_duration, stale_file = dur, hbp
            except Exception:
                _mark_missing_or_invalid(hbp)
        else:
            _mark_missing_or_invalid(hbp)

    age_sec = max(0.0, now_ts - min(mtimes)) if mtimes else 0.0
    fresh = not (stale_threshold_s > 0 and (missing_any or not mtimes or age_sec > stale_threshold_s))
    return fresh, age_sec, max_stale_duration, stale_file


def resolve_heartbeat_paths(
    workspace_dir: str,
    proc_dir: str,
    name: str,
    pattern: str,
    hb_log: str,
    hb_stale_s: float,
    *,
    include_stdout_with_heartbeat_file: bool = False,
) -> tuple[list[str], str]:
    """Resolve canonical heartbeat file paths and stdout log file path for a procs.conf entry."""
    raw_stem = os.path.basename((name or pattern).rstrip("$"))
    raw_stem = raw_stem[:-3] if raw_stem.endswith(".py") else raw_stem
    clean_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw_stem)
    stdout_log_file = os.path.join(workspace_dir, "logs", f"{clean_stem}.log")

    hb_paths: list[str] = []
    if hb_log:
        for part in split_csv(hb_log):
            if os.path.isabs(part):
                hb_paths.append(part)
            elif part.startswith("cachedb/"):
                hb_paths.append(str(resolve_cachedb_filepath(os.path.join(workspace_dir, part))))
            elif part.startswith("logs/"):
                hb_paths.append(os.path.join(workspace_dir, part))
            else:
                hb_paths.append(os.path.join(proc_dir, part))
        if (
            include_stdout_with_heartbeat_file
            and hb_paths
            and stdout_log_file not in hb_paths
            and any(p.endswith(".heartbeat") for p in hb_paths)
        ):
            hb_paths.append(stdout_log_file)
    elif hb_stale_s > 0:
        hb_paths = [stdout_log_file]

    return hb_paths, stdout_log_file


def parse_procs_manifest(
    workspace_dir: str,
    *,
    venv: str = "myenv",
    roles: tuple[str, ...] = ("bot", "fleet"),
    include_stdout_with_heartbeat_file: bool = False,
    require_cmd: bool = False,
) -> list[dict[str, object]]:
    """Parse ``procs.conf`` from ``workspace_dir`` as the single source of truth for bots and fleet modules."""
    procs: list[dict[str, object]] = []
    procs_conf = os.path.join(workspace_dir, "procs.conf")
    if not os.path.exists(procs_conf):
        return procs

    try:
        raw_lines = _state_io.read_text_lines(procs_conf, errors="replace")
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or len(parts := line.split("|")) < 7:
                continue
            pat, dr, cmd, label, hblog, hbstale, role = [p.strip() for p in parts[:7]]
            if role not in roles or (require_cmd and not cmd):
                continue
            hb_stale_s = max(0.0, safe_float(hbstale, default=0.0) or 0.0)
            resolved_dir = dr.replace("$ROOT", workspace_dir)
            resolved_cmd = cmd.replace("$ROOT", workspace_dir).replace("$VENV", venv)
            resolved_hblog = hblog.replace("$ROOT", workspace_dir) if hblog else ""
            name = label or pat
            hb_paths, log_file = resolve_heartbeat_paths(
                workspace_dir,
                resolved_dir,
                name,
                pat,
                resolved_hblog,
                hb_stale_s,
                include_stdout_with_heartbeat_file=include_stdout_with_heartbeat_file,
            )
            tokens: set[str] = {name.lower(), *([name[:-3].lower()] if name.lower().endswith(".py") else [])}
            if clean_pat := pat.rstrip("$").strip():
                pat_first = clean_pat.split()[0]
                base = os.path.basename(pat_first)
                tokens.update({clean_pat.lower(), pat_first.lower(), base.lower(), *([base[:-3].lower()] if base.endswith(".py") else [])})
            for tok in (resolved_cmd or "").split():
                if (cleaned := tok.strip("\"'&;")).endswith(".py"):
                    b_tok = os.path.basename(cleaned)
                    tokens.update({cleaned.lower(), b_tok.lower(), b_tok[:-3].lower()})
            aliases = {t for t in tokens if t}
            procs.append({
                "name": name,
                "pattern": pat,
                "clean_pattern": pat.rstrip("$"),
                "dir": resolved_dir,
                "cmd": resolved_cmd,
                "hb_log": resolved_hblog,
                "hb_paths": hb_paths,
                "hb_files": hb_paths,
                "hb_file": hb_paths[0] if hb_paths else "",
                "log_file": log_file,
                "hb_stale_s": hb_stale_s,
                "role": role,
                "aliases": sorted(aliases),
            })
    except Exception:
        pass

    return procs


def resolve_mem_bounded_worker_cap(
    max_workers: int = 4,
    *,
    default_cpus: int = 2,
    env_var: str = "WORKERS",
    read_text_fn: object = None,
) -> int:
    """Resolve worker pool size from ``env_var`` or CPU count bounded to 2 on <=4.5 GiB hosts."""
    raw_w = int_env(env_var)
    if raw_w is not None and raw_w > 0:
        return raw_w
    cpu_cap = min(max(1, safe_pos_int(max_workers, 4)), os.cpu_count() or max(1, safe_pos_int(default_cpus, 2)))
    reader = read_text_fn if callable(read_text_fn) else _state_io.read_text_file
    meminfo = reader("/proc/meminfo", default="")
    if isinstance(meminfo, str):
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_kb = safe_nonneg_int(parts[1], default=0)
                    if 0 < mem_kb <= 4608 * 1024:
                        return min(2, cpu_cap)
                break
    return cpu_cap


class RuntimeServices:
    """Explicit dependency-injection container for clock, sleep, retry, snapshot, and external transport policies."""

    def __init__(
        self,
        *,
        clock_fn: object = None,
        sleep_fn: object = None,
        stale_cache_retry_delay_sec: float = 1.5,
        enforce_rth_gate: bool = True,
        allow_live_network_backfill: bool = True,
        allow_unmocked_llm_queries: bool = True,
        allow_shared_disk_cache: bool = True,
    ) -> None:
        self.clock_fn = clock_fn if callable(clock_fn) else None
        self.sleep_fn = sleep_fn if callable(sleep_fn) else None
        self.stale_cache_retry_delay_sec = safe_nonneg_float(stale_cache_retry_delay_sec, default=1.5)
        self.enforce_rth_gate = _parse_bool_flag(enforce_rth_gate, True)
        self.allow_live_network_backfill = _parse_bool_flag(allow_live_network_backfill, True)
        self.allow_unmocked_llm_queries = _parse_bool_flag(allow_unmocked_llm_queries, True)
        self.allow_shared_disk_cache = _parse_bool_flag(allow_shared_disk_cache, True)

    def configure(self, **kwargs: object) -> "RuntimeServices":
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown RuntimeServices attribute: {k!r}")
            if k in ("clock_fn", "sleep_fn"):
                setattr(self, k, v if callable(v) else None)
            elif k == "stale_cache_retry_delay_sec":
                setattr(self, k, safe_nonneg_float(v, default=1.5))
            else:
                setattr(self, k, _parse_bool_flag(v, getattr(self, k, True)))
        return self

    def rth_gate_enabled(self, override: bool | None = None, *, rth_fn: object = None) -> bool:
        if override is not None:
            return _parse_bool_flag(override, False)
        if rth_fn is not None and not (
            rth_fn is is_us_rth_open
            or (
                getattr(rth_fn, "__module__", None) in ("botcore", "core.time_market")
                and getattr(rth_fn, "__qualname__", None) == "is_us_rth_open"
            )
        ):
            return True
        if bool_env("MPTRADE_FORCE_RTH_GATE", default=False):
            return True
        return bool(self.enforce_rth_gate)

    def live_network_backfill_enabled(self, override: bool | None = None) -> bool:
        if override is not None:
            return bool(override)
        env_flag = os.environ.get("MPTRADE_ALLOW_LIVE_BACKFILL")
        if env_flag is not None and str(env_flag).strip() != "":
            return bool(_parse_bool_flag(env_flag, default=True))
        return bool(self.allow_live_network_backfill)

    def unmocked_llm_query_allowed(self, override: bool | None = None) -> bool:
        return bool(override) if override is not None else bool(self.allow_unmocked_llm_queries)

    def shared_disk_cache_enabled(self, override: bool | None = None) -> bool:
        return bool(override) if override is not None else bool(self.allow_shared_disk_cache)

    def stale_cache_retry_delay(self, override: float | None = None) -> float:
        if override is not None:
            val = safe_nonneg_float(override, default=None)
            if val is not None:
                return val
        return safe_nonneg_float(self.stale_cache_retry_delay_sec, default=0.0) or 0.0


_bc_mod = sys.modules.get("botcore")
_prev_rs = (_bc_mod.__dict__.get("_RUNTIME_SERVICES") if _bc_mod is not None else None) or globals().get("_RUNTIME_SERVICES")
if _prev_rs is not None and hasattr(_prev_rs, "enforce_rth_gate"):
    with contextlib.suppress(TypeError):
        _prev_rs.__class__ = RuntimeServices
    _RUNTIME_SERVICES: RuntimeServices = _prev_rs  # type: ignore[assignment]
else:
    _RUNTIME_SERVICES = RuntimeServices()


def _eff_runtime_services() -> RuntimeServices:
    bc = sys.modules.get("botcore")
    rs = bc.__dict__.get("_RUNTIME_SERVICES") if bc is not None and "_RUNTIME_SERVICES" in bc.__dict__ else None
    return rs if rs is not None and hasattr(rs, "enforce_rth_gate") else _RUNTIME_SERVICES


def configure_runtime_services(**kwargs: object) -> RuntimeServices:
    """Update fields on the active ``RuntimeServices`` container and return it."""
    return _eff_runtime_services().configure(**kwargs)


SIMULATED_PROVIDER_NAMES: frozenset[str] = frozenset(
    {"replay", "backtest", "sim", "mock", "paper_sim", "test"}
)


def is_simulated_provider_name(provider: object) -> bool:
    """Return True when ``provider`` identifies a simulated/replay/backtest/mock execution adapter."""
    return bool(
        provider
        and not isinstance(provider, bool)
        and str(provider).strip().lower() in SIMULATED_PROVIDER_NAMES
    )


def is_replay_or_backtest_env(*extra_keys: str, env: dict | None = None) -> bool:
    """Return True when ``BACKTEST_MODE``, ``REPLAY_MODE``, or any ``extra_keys`` flag is enabled."""
    return any_bool_env("BACKTEST_MODE", "REPLAY_MODE", *extra_keys, env=env)


def is_gemini_llm_disabled(env: dict | None = None) -> bool:
    """Return True when Gemini LLM queries are suppressed via ``DISABLE_GEMINI_LLM``, ``BACKTEST_MODE``, or ``REPLAY_MODE``."""
    return is_replay_or_backtest_env("DISABLE_GEMINI_LLM", env=env)


def is_offline_disk_cache_enabled(env: dict | None = None) -> bool:
    """Return True when ``MPTRADE_OFFLINE_DISK_CACHE`` is enabled for offline OHLC disk reads."""
    return bool_env("MPTRADE_OFFLINE_DISK_CACHE", env=env, default=False)


def is_auto_start_websockets_enabled(env: dict | None = None) -> bool:
    """Return True when ``BINANCE_AUTO_START_WEBSOCKETS`` permits background WS startup (default True)."""
    return bool_env("BINANCE_AUTO_START_WEBSOCKETS", env=env, default=True)


def disable_auto_start_websockets_default() -> None:
    """Default ``BINANCE_AUTO_START_WEBSOCKETS=0`` in ``os.environ`` for read-only CLI/supervisor entrypoints."""
    os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")


def is_rth_gate_enabled(override: bool | None = None, *, rth_fn: object = None) -> bool:
    """Return whether US equities RTH session gating is active under the current runtime policy."""
    return _eff_runtime_services().rth_gate_enabled(override, rth_fn=rth_fn)


def is_live_network_backfill_enabled(override: bool | None = None) -> bool:
    """Return whether opportunistic on-demand live network backfills are enabled."""
    return _eff_runtime_services().live_network_backfill_enabled(override)


def is_unmocked_llm_query_allowed(override: bool | None = None) -> bool:
    """Return whether unmocked external LLM CLI/REST queries are permitted."""
    return _eff_runtime_services().unmocked_llm_query_allowed(override)


def is_shared_disk_cache_enabled(override: bool | None = None) -> bool:
    """Return whether default cross-process disk caches are enabled when no explicit path is injected."""
    return _eff_runtime_services().shared_disk_cache_enabled(override)


def get_stale_cache_retry_delay_sec(override: float | None = None) -> float:
    """Return the retry delay in seconds for stale account cache order retries."""
    return _eff_runtime_services().stale_cache_retry_delay(override)


async def async_interruptible_sleep(
    stop_event: object, delay: object, step: object = 0.2, *, sleep_fn: object = None
) -> None:
    """Sleep asynchronously up to ``delay`` seconds in ``step`` increments while ``stop_event`` is not set."""
    eff_delay = safe_pos_float(delay, default=None)
    if eff_delay is None:
        return
    eff_step = safe_pos_float(step, default=0.2) or 0.2
    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    is_stopped = getattr(stop_event, "is_set", None)
    elapsed = 0.0
    while elapsed < eff_delay and not (callable(is_stopped) and is_stopped()):
        await sleep_fn(min(eff_step, eff_delay - elapsed))
        elapsed += eff_step


def is_worker_thread_alive(instance: object, attr: str = "_thread") -> bool:
    """Return True when the thread stored on ``getattr(instance, attr, None)`` is alive."""
    thread = getattr(instance, attr, None)
    return thread is not None and bool(getattr(thread, "is_alive", lambda: False)())


def set_stop_event(instance: object, attr: str = "_stop_event") -> None:
    """Signal the ``threading.Event`` stored on ``getattr(instance, attr, None)`` to stop."""
    ev = getattr(instance, attr, None)
    if ev is not None and hasattr(ev, "set"):
        ev.set()


def sleep_monitoring_interval(
    instance: object,
    interval_sec: object,
    *,
    max_sleep_step: float = 1.0,
    sleep_fn: object = None,
) -> None:
    """Sleep in interruptible steps summing to ``interval_sec`` while ``instance._running`` remains truthy."""
    remaining = safe_pos_float(interval_sec, default=60.0) or 60.0
    step_cap = safe_pos_float(max_sleep_step, default=1.0) or 1.0
    sleeper = sleep_fn if callable(sleep_fn) else _time.sleep
    while remaining > 1e-9 and _parse_bool_flag(getattr(instance, "_running", False), default=False):
        chunk = min(step_cap, remaining)
        sleeper(chunk)
        remaining -= chunk


def start_monitoring_daemon_thread(
    instance: object, target: object, name: str
) -> threading.Thread:
    """Create, attach to ``instance._thread``, and start a daemon monitoring thread."""
    thread = threading.Thread(target=target, name=name, daemon=True)
    setattr(instance, "_thread", thread)
    thread.start()
    return thread


def stop_monitoring_thread(
    instance: object, label: str = "", *, timeout: float = 5.0
) -> None:
    """Stop a background monitoring loop backed by ``_running`` and ``_thread``."""
    setattr(instance, "_running", False)
    thread = getattr(instance, "_thread", None)
    if thread and thread is not threading.current_thread():
        thread.join(timeout=safe_nonneg_float(timeout, default=5.0))
    eff_label = label or getattr(instance, "_MONITOR_LABEL", None) or type(instance).__name__
    print(f"[{eff_label}] Monitoring stopped")


class StringEnum(str, Enum):
    """String-backed ``Enum`` whose ``__str__`` returns the raw string ``value`` across Python versions."""

    def __str__(self) -> str:
        return str(self.value)


class _BoundAttrCall:
    """Non-data method descriptor dispatching ``getattr(self, method_name)`` with owner fallback for duck stubs."""

    def __init__(
        self,
        method_name: str,
        bound_args: tuple[object, ...],
        bound_kwargs: dict[str, object],
        *,
        owner: type | None = None,
        name: str = "",
    ) -> None:
        self._method_name = method_name
        self._bound_args = bound_args
        self._bound_kwargs = bound_kwargs
        self._owner = owner
        self.__name__ = name or method_name
        self.__qualname__ = f"{owner.__qualname__}.{self.__name__}" if owner is not None else self.__name__

    def __set_name__(self, owner: type, name: str) -> None:
        self._owner = owner
        self.__name__ = name
        self.__qualname__ = f"{owner.__qualname__}.{name}"

    def __call__(self, target: object, *args: object, **kwargs: object):
        merged_kw = {**self._bound_kwargs, **kwargs} if self._bound_kwargs else kwargs
        fn = getattr(target, self._method_name, None)
        if callable(fn):
            return fn(*self._bound_args, *args, **merged_kw)
        if self._owner is not None:
            owner_fn = getattr(self._owner, self._method_name, None)
            if callable(owner_fn):
                return owner_fn(target, *self._bound_args, *args, **merged_kw)
        return getattr(target, self._method_name)(*self._bound_args, *args, **merged_kw)

    def __get__(self, instance: object, owner: type | None = None):
        eff_owner = owner if owner is not None else self._owner
        bound_target = (
            self
            if eff_owner is self._owner
            else _BoundAttrCall(
                self._method_name,
                self._bound_args,
                self._bound_kwargs,
                owner=eff_owner,
                name=self.__name__,
            )
        )
        return bound_target if instance is None else types.MethodType(bound_target, instance)


def bind_attr_call(method_name: str, *bound_args: object, **bound_kwargs: object) -> _BoundAttrCall:
    """Bind a method dispatching ``getattr(self, method_name)`` at call time so overrides, monkeypatches, and duck stubs apply."""
    return _BoundAttrCall(method_name, bound_args, dict(bound_kwargs))


def bind_module_importer(module_name: str):
    """Return a zero-arg callable resolving ``importlib.import_module(module_name)`` dynamically at call time."""
    return lambda _m=module_name: importlib.import_module(_m)


class LazyPluginRegistryFactory:
    """Shared lazy-loading plugin class registry and singleton factory base."""

    _instances: dict[str, object] = {}
    _registry: dict[str, type] = {}
    _defaults_loaded: bool = False
    _unknown_id_label: str = "plugin"
    _default_modules: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "_instances" not in cls.__dict__:
            cls._instances = {}
        if "_registry" not in cls.__dict__:
            cls._registry = {}
        if "_defaults_loaded" not in cls.__dict__:
            cls._defaults_loaded = False

    @classmethod
    def register(cls, plugin_id: str, plugin_cls: type) -> None:
        cls._registry[str(plugin_id)] = plugin_cls

    @classmethod
    def _resolve_registered_class(cls, plugin_id: str) -> type:
        key = str(plugin_id)
        plugin_cls = cls._registry.get(key)
        if plugin_cls is None and not cls._defaults_loaded:
            cls._defaults_loaded = True
            for mod_name in getattr(cls, "_default_evaluator_modules", None) or cls._default_modules:
                try:
                    __import__(mod_name)
                except Exception:
                    pass
            plugin_cls = cls._registry.get(key)
        if plugin_cls is None:
            raise KeyError(f"Unknown {cls._unknown_id_label} id: {key}")
        return plugin_cls

    @staticmethod
    def bind_creator(plugin_id: str):
        return classmethod(
            lambda sub_cls, *, start_sync=False, _id=plugin_id: sub_cls.create(_id, start_sync=start_sync)
        )


_LOCAL_MODULE_LOCK = globals().get("_LOCAL_MODULE_LOCK") or threading.RLock()


def ensure_local_module_priority(directory: str, *module_names: str) -> None:
    """Ensure ``directory`` is first on ``sys.path`` (with repo root available) and evict foreign cached modules."""
    with _LOCAL_MODULE_LOCK:
        abs_dir = os.path.abspath(directory)
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        if sys.path[:1] != [abs_dir]:
            if abs_dir in sys.path:
                sys.path.remove(abs_dir)
            sys.path.insert(0, abs_dir)
        for mod_name in module_names:
            mod_obj = sys.modules.get(mod_name)
            if mod_obj is None:
                continue
            mod_file = getattr(mod_obj, "__file__", None)
            if type(mod_obj) is object or (
                isinstance(mod_file, str) and mod_file and os.path.dirname(os.path.abspath(mod_file)) != abs_dir
            ):
                sys.modules.pop(mod_name, None)


_AMBIGUOUS_LOCAL_MODULES = (
    "strategy", "market_data", "replay", "common", "hl_client", "t212_client", "ipo_common",
    "notify", "ipo_notify", "signals",
)

_LEGACY_ARCHIVE_DIR: str | None = None


def _resolve_archived_module_dir(abs_dir: str, stem: str) -> str:
    global _LEGACY_ARCHIVE_DIR
    if os.path.isfile(os.path.join(abs_dir, f"{stem}.py")) or os.path.isdir(os.path.join(abs_dir, stem)):
        return abs_dir
    archive_tar = os.path.join(_REPO_ROOT, "archive", "legacy_archive.tar.gz")
    if not os.path.isfile(archive_tar):
        return abs_dir
    rel = os.path.relpath(abs_dir, _REPO_ROOT)
    if not (rel == "archive" or rel.startswith("archive" + os.sep)):
        return abs_dir
    if _LEGACY_ARCHIVE_DIR is None or not os.path.isdir(_LEGACY_ARCHIVE_DIR):
        _LEGACY_ARCHIVE_DIR = tempfile.mkdtemp(prefix="mptrade_legacy_archive_")
        atexit.register(shutil.rmtree, _LEGACY_ARCHIVE_DIR, True)
        with tarfile.open(archive_tar, "r:gz") as tf:
            tf.extractall(_LEGACY_ARCHIVE_DIR, filter="data")
    extracted_dir = os.path.join(_LEGACY_ARCHIVE_DIR, rel)
    return extracted_dir if (os.path.isfile(os.path.join(extracted_dir, f"{stem}.py")) or os.path.isdir(os.path.join(extracted_dir, stem))) else abs_dir


def load_local_module(
    directory: str, name: str, *, module_key: str | None = None, register: bool = True, reload: bool = False
):
    """Load a sibling module ``<name>.py`` from ``directory`` without colliding with other venue modules."""
    with _LOCAL_MODULE_LOCK:
        stem = name[:-3] if name.endswith(".py") else name
        abs_dir = _resolve_archived_module_dir(os.path.abspath(directory), stem)
        key = module_key or stem
        if register and not reload:
            mod = sys.modules.get(key)
            mod_file = getattr(mod, "__file__", None) if mod is not None else None
            if (
                mod is not None
                and getattr(mod, "__mptrade_exec_done__", True)
                and (not isinstance(mod_file, str) or not mod_file or os.path.dirname(os.path.abspath(mod_file)) == abs_dir)
            ):
                return mod

        orig_sys_path = list(sys.path)
        saved_ambiguous: dict[str, object] = {}
        for amb_name in _AMBIGUOUS_LOCAL_MODULES:
            if amb_name in sys.modules:
                amb_file = getattr(sys.modules[amb_name], "__file__", None)
                if isinstance(amb_file, str) and amb_file and os.path.dirname(os.path.abspath(amb_file)) != abs_dir:
                    saved_ambiguous[amb_name] = sys.modules.pop(amb_name)
        succeeded = False
        try:
            while abs_dir in sys.path:
                sys.path.remove(abs_dir)
            while _REPO_ROOT in sys.path:
                sys.path.remove(_REPO_ROOT)
            sys.path.insert(0, _REPO_ROOT)
            repo_base = os.path.basename(_REPO_ROOT)
            for seg in abs_dir.split(os.sep):
                if not seg or seg == repo_base:
                    continue
                cand_dir = os.path.join(_REPO_ROOT, seg)
                if cand_dir != abs_dir and os.path.isdir(cand_dir) and cand_dir not in sys.path:
                    sys.path.insert(0, cand_dir)
            sys.path.insert(0, abs_dir)
            spec = importlib.util.spec_from_file_location(key, os.path.join(abs_dir, f"{stem}.py"))
            mod = importlib.util.module_from_spec(spec)
            setattr(mod, "__mptrade_exec_done__", False)
            prev_key_mod = sys.modules.get(key)
            sys.modules[key] = mod
            try:
                spec.loader.exec_module(mod)
                setattr(mod, "__mptrade_exec_done__", True)
            except Exception:
                if prev_key_mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prev_key_mod
                raise
            if not register:
                if prev_key_mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prev_key_mod
            succeeded = True
            return mod
        finally:
            sys.path[:] = orig_sys_path
            for amb_name, saved_mod in saved_ambiguous.items():
                if not (succeeded and register and amb_name == key):
                    sys.modules[amb_name] = saved_mod


def prepare_shadow_runtime(root: str, *module_dirs: str) -> None:
    """Make repository modules importable and disable live Binance WebSockets."""
    ensure_local_module_priority(root)
    for path in module_dirs:
        absolute = os.path.abspath(path)
        if absolute not in sys.path:
            sys.path.append(absolute)
    os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")


def resolve_patched_attr(
    attr_name: str,
    default_val: object = None,
    *,
    local_val: object = _UNSET,
    defaults: tuple[object, ...] = (),
    modules: tuple[str, ...] | list[str] = (),
    require_callable: bool = False,
) -> object:
    """Resolve ``attr_name`` across ``local_val`` and ``sys.modules[modules]`` when monkeypatched away from ``defaults``."""
    raw_unpatched = list(defaults if defaults else (default_val,))
    if any(
        str(getattr(d, "__module__", "") or "").startswith(("botcore", "core."))
        and getattr(d, "__name__", None) == attr_name
        for d in raw_unpatched
        if d is not None
    ):
        for mod_key in ("botcore", __name__):
            mod_ref = sys.modules.get(mod_key)
            cur_fn = mod_ref.__dict__.get(attr_name) if mod_ref is not None else None
            if cur_fn is not None:
                raw_unpatched.append(cur_fn)
    unpatched = tuple(d for d in raw_unpatched if d is not None)

    def _is_valid(cand: object) -> bool:
        return False if (cand is _UNSET or cand is None) else (callable(cand) if require_callable else True)

    def _is_patched(cand: object) -> bool:
        return _is_valid(cand) and all(cand is not d for d in unpatched)

    if local_val is not _UNSET and _is_patched(local_val):
        return local_val
    for mod_name in modules:
        if not mod_name:
            continue
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr_name):
            cand = getattr(mod, attr_name, _UNSET)
            if _is_patched(cand):
                return cand
            if not unpatched and cand is not _UNSET and not require_callable:
                return cand
    if local_val is not _UNSET and _is_valid(local_val):
        return local_val
    return default_val


def bind_lazy_module_exports(
    module_globals: dict[str, object],
    lazy_imports: dict[str, str],
    submodules: tuple[str, ...] | list[str] | set[str] | frozenset[str] = (),
):
    """Bind PEP 562 ``__getattr__``, ``__dir__``, and ``__all__`` into ``module_globals``."""
    pkg_name = str(module_globals.get("__name__", ""))
    sub_set = frozenset(submodules)
    all_names = list(module_globals.get("__all__") or list(lazy_imports))
    module_globals["__all__"] = all_names

    def __getattr__(name: str):
        mod_name = lazy_imports.get(name)
        if mod_name is not None:
            val = getattr(importlib.import_module(mod_name), name)
            module_globals[name] = val
            return val
        if name in sub_set:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            module_globals[name] = mod
            return mod
        raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(set(module_globals) | set(all_names))

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    return __getattr__, __dir__


def register_module_aliases(module_name: str, *aliases: str) -> None:
    """Bind ``aliases`` in ``sys.modules`` (and parent package attributes) to ``sys.modules[module_name]``."""
    self_mod = sys.modules.get(module_name)
    if self_mod is None:
        return
    self_file = getattr(self_mod, "__file__", None)
    for alias in aliases:
        prev = sys.modules.get(alias)
        prev_loader = getattr(prev, "__loader__", None)
        if (
            prev is None
            or prev is self_mod
            or getattr(prev, "__file__", None) == self_file
            or getattr(prev_loader, "__class__", None).__name__ == "_MergedModuleAliasLoader"
        ):
            sys.modules[alias] = self_mod
            if "." in alias:
                parent_name, child_attr = alias.rsplit(".", 1)
                parent_pkg = sys.modules.get(parent_name)
                if parent_pkg is not None:
                    setattr(parent_pkg, child_attr, self_mod)


_MERGED_MODULE_TARGETS = {
    "hyperliquid.signals": "hyperliquid.signals",
    "symbols": "instrument_registry",
}


class _MergedModuleAliasLoader:
    def __init__(self, alias_name: str, target_name: str) -> None:
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        if self.target_name == "hyperliquid.signals":
            hl_dir = os.path.join(_REPO_ROOT, "hyperliquid")
            is_reload = module is sys.modules.get("signals") and getattr(module, "__mptrade_exec_done__", False)
            target_mod = load_local_module(hl_dir, "signals", reload=is_reload)
            for alias_k in ("signals", "hyperliquid.signals"):
                sys.modules[alias_k] = target_mod
            hl_pkg = sys.modules.get("hyperliquid")
            if hl_pkg is not None:
                setattr(hl_pkg, "signals", target_mod)
            sys.modules[self.alias_name] = target_mod
            return

        target_mod = importlib.import_module(self.target_name)
        if "." in self.alias_name:
            parent_name, child_attr = self.alias_name.rsplit(".", 1)
            parent_pkg = sys.modules.get(parent_name)
            if parent_pkg is not None:
                setattr(parent_pkg, child_attr, target_mod)
        sys.modules[self.alias_name] = target_mod


class _MergedModuleAliasFinder:
    """Resolve merged module names even before host import or after sys.modules.pop."""

    @staticmethod
    def find_spec(fullname: str, path=None, target=None):
        target_name = _MERGED_MODULE_TARGETS.get(fullname)
        if target_name is None and fullname == "signals" and target is not None:
            target_file = getattr(target, "__file__", None)
            hl_signals_file = os.path.join(_REPO_ROOT, "hyperliquid", "signals.py")
            if isinstance(target_file, str) and os.path.abspath(target_file) == os.path.abspath(hl_signals_file):
                target_name = "hyperliquid.signals"
        if target_name is None and fullname.startswith("archive."):
            rel_parts = fullname.split(".")
            sub_dir = os.path.join(_REPO_ROOT, *rel_parts[:-1])
            stem = rel_parts[-1]
            resolved_dir = _resolve_archived_module_dir(sub_dir, stem)
            py_file = os.path.join(resolved_dir, f"{stem}.py")
            if os.path.isfile(py_file):
                return importlib.util.spec_from_file_location(fullname, py_file)
            pkg_dir = os.path.join(resolved_dir, stem)
            if os.path.isdir(pkg_dir):
                init_py = os.path.join(pkg_dir, "__init__.py")
                if os.path.isfile(init_py):
                    return importlib.util.spec_from_file_location(
                        fullname, init_py, submodule_search_locations=[pkg_dir]
                    )
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = [pkg_dir]
                return spec
        if target_name is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _MergedModuleAliasLoader(fullname, target_name),
        )


if not any(
    getattr(f, "__name__", "") == "_MergedModuleAliasFinder" or isinstance(f, _MergedModuleAliasFinder)
    for f in sys.meta_path
):
    sys.meta_path.append(_MergedModuleAliasFinder())
