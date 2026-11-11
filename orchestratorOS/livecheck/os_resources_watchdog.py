#!/usr/bin/env python3
"""Alert when any process sustains excessive CPU or resident memory usage."""
import os
import signal
import time
from pathlib import Path
import sys
_LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
import watchdog_common as wc
from botcore import (
    _parse_bool_flag,
    exact_pos_int,
    load_env_stack,
    require_int,
    required_bool_env,
    required_float_env,
    required_int_env,
    safe_float,
    safe_nonneg_int,
    safe_pos_int,
)
from state_io import parse_proc_stat_fields, read_bytes_file, read_text_file
from technical_repair_ssot import fleet_script_basenames

ROOT = wc.ROOT
STATE_FILE = ROOT / ".resource_watchdog_state.json"
load_env_stack()

CPU_THRESHOLD = required_float_env("RESOURCE_CPU_THRESHOLD_PCT")
MEMORY_THRESHOLD_MB = required_float_env("RESOURCE_MEMORY_THRESHOLD_MB")
CONSECUTIVE = required_int_env("RESOURCE_CONSECUTIVE_CHECKS")
RECOVERY_CHECKS = required_int_env("RESOURCE_RECOVERY_CHECKS")
COOLDOWN_MIN = required_float_env("RESOURCE_ALERT_COOLDOWN_MINUTES")
CHECK_INTERVAL_MIN = required_float_env("RESOURCE_CHECK_INTERVAL_MINUTES")
AUTO_RESTART = required_bool_env("RESOURCE_AUTO_RESTART")
RESTART_COOLDOWN_MIN = required_float_env("RESOURCE_RESTART_COOLDOWN_MINUTES")
RESTART_MAX = required_int_env("RESOURCE_RESTART_MAX")
RESTART_WINDOW_H = required_float_env("RESOURCE_RESTART_WINDOW_HOURS")

if not all(
    v > 0
    for v in (
        CPU_THRESHOLD, MEMORY_THRESHOLD_MB, CONSECUTIVE, RECOVERY_CHECKS,
        COOLDOWN_MIN, CHECK_INTERVAL_MIN, RESTART_COOLDOWN_MIN, RESTART_MAX,
        RESTART_WINDOW_H,
    )
):
    raise ValueError("resource watchdog configuration values must be positive")


def _managed_script(proc):
    if not isinstance(proc, dict):
        return None
    parts = str(proc.get("command") or "").split()
    scripts = fleet_script_basenames(str(ROOT))
    pid = exact_pos_int(proc.get("pid"), default=None)
    if pid is None or pid <= 1:
        return None
    for part in parts[1:]:
        name = os.path.basename(part)
        if name in scripts:
            try:
                if Path(f"/proc/{pid}/cwd").resolve() == ROOT.resolve():
                    return name
            except (FileNotFoundError, OSError):
                return None
    return None


def _restart_allowed(state, script, now):
    rh = state.get("restart_history")
    if not isinstance(rh, dict):
        rh = {}
        state["restart_history"] = rh
    history = rh.get(script)
    if not isinstance(history, list):
        history = []
        rh[script] = history
    cutoff = now - RESTART_WINDOW_H * 3600
    history[:] = [
        stamp for stamp in history
        if isinstance(stamp, (int, float))
        and not isinstance(stamp, bool)
        and safe_float(stamp, default=None) is not None
        and cutoff <= stamp <= now
    ]
    if history and now - history[-1] < RESTART_COOLDOWN_MIN * 60:
        return False
    return len(history) < RESTART_MAX


def _restart_managed(proc, state, now):
    script = _managed_script(proc)
    if not AUTO_RESTART or not script or not _restart_allowed(state, script, now):
        return None
    pid = exact_pos_int(proc.get("pid") if isinstance(proc, dict) else None, default=None)
    if pid is None or pid <= 1:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
        return False
    state.setdefault("restart_history", {}).setdefault(script, []).append(now)
    return True


def _parse_proc_stat(stat_text: str) -> tuple[str, int, int, int]:
    """Parse (comm, start_time, total_ticks, rss_pages) from /proc/<pid>/stat safely."""
    comm, after = parse_proc_stat_fields(stat_text, min_fields=22)
    return (
        comm,
        require_int(after[19], name="start_time", min_value=0),
        require_int(after[11], name="utime", min_value=0) + require_int(after[12], name="stime", min_value=0),
        require_int(after[21], name="rss_pages"),
    )


def _usage(previous, current, previous_total, current_total, cpu_count):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return None
    cpu_cnt = safe_float(cpu_count, 0.0)
    if cpu_cnt <= 0.0:
        return None
    prev_tot = safe_float(previous_total, -1.0)
    curr_tot = safe_float(current_total, -1.0)
    if prev_tot < 0.0 or curr_tot < 0.0:
        return None
    total_delta = curr_tot - prev_tot
    if total_delta <= 0:
        return None
    prev_ticks = safe_float(previous.get("ticks"), -1.0)
    curr_ticks = safe_float(current.get("ticks"), -1.0)
    if prev_ticks < 0.0 or curr_ticks < 0.0:
        return None
    tick_delta = curr_ticks - prev_ticks
    if tick_delta < 0:
        return None
    return tick_delta * cpu_cnt * 100.0 / total_delta


def check_once(now=None, *, total_ticks=None, processes=None, cpu_count=None):
    now = safe_float(now, 0.0)
    if now <= 0.0:
        now = time.time()
    if total_ticks is None:
        total_ticks = sum(require_int(v, "proc_stat_tick") for v in read_text_file("/proc/stat").splitlines()[0].split()[1:])
    if processes is None:
        processes = {}
        page_size = os.sysconf("SC_PAGE_SIZE")
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                comm, start_time, ticks, rss_pages = _parse_proc_stat(read_text_file(entry / "stat"))
                key = f"{entry.name}:{start_time}"
                rss_mb = rss_pages * page_size / 1024 / 1024
                command = read_bytes_file(entry / "cmdline").replace(b"\0", b" ").decode(errors="replace").strip() or comm
                processes[key] = {"pid": exact_pos_int(entry.name, default=0) or 0, "ticks": ticks, "rss_mb": rss_mb, "command": command[:180]}
            except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, OSError, ValueError):
                continue
    cpu_count = (os.cpu_count() or 1) if cpu_count is None else safe_pos_int(cpu_count, default=1)
    state = wc.load_state(STATE_FILE)
    if not isinstance(state, dict):
        state = {}
    previous_total = state.get("total_ticks")
    previous = state.get("processes")
    if not isinstance(previous, dict):
        previous = {}
    tracked = state.get("tracked")
    if not isinstance(tracked, dict):
        tracked = {}
    offenders = []
    recovered = []

    for key, proc in processes.items():
        if not isinstance(proc, dict):
            continue
        old = previous.get(key)
        cpu = (_usage(old, proc, previous_total, total_ticks, cpu_count)
               if old is not None and previous_total is not None else None)
        reasons = []
        if cpu is not None and cpu >= CPU_THRESHOLD:
            reasons.append(f"CPU {cpu:.1f}%")
        rss_mb = safe_float(proc.get("rss_mb"), 0.0)
        if rss_mb >= MEMORY_THRESHOLD_MB:
            reasons.append(f"RSS {rss_mb:.0f} MB")
        item = tracked.get(key, {"high": 0, "normal": 0, "alerted": False, "last_alert": 0})
        if not isinstance(item, dict):
            item = {"high": 0, "normal": 0, "alerted": False, "last_alert": 0}
        alerted = _parse_bool_flag(item.get("alerted"), False)
        item["alerted"] = alerted
        last_alert = safe_float(item.get("last_alert", 0), 0.0)
        if last_alert > now or last_alert < 0.0:
            last_alert = 0.0
            item["last_alert"] = 0.0
        if reasons:
            item["high"] = safe_nonneg_int(item.get("high", 0), default=0) + 1
            item["normal"] = 0
            item["last_reasons"] = reasons
            if item["high"] >= CONSECUTIVE and (not alerted or
                    now - last_alert >= COOLDOWN_MIN * 60):
                offenders.append((proc, reasons, item["high"]))
                item["alerted"] = True
                item["last_alert"] = now
        else:
            item["high"] = 0
            item["normal"] = safe_nonneg_int(item.get("normal", 0), default=0) + 1
            if alerted and item["normal"] >= RECOVERY_CHECKS:
                recovered.append(proc)
                item["alerted"] = False
                item["last_alert"] = now
        tracked[key] = item

    # Drop exited processes; their disappearance is already covered by the fleet watchdog.
    tracked = {key: value for key, value in tracked.items() if key in processes}
    state.update({"total_ticks": total_ticks, "processes": processes,
                  "tracked": tracked, "last_run": now})
    wc.save_state(STATE_FILE, state)

    if offenders:
        lines = []
        for proc, reasons, checks in offenders:
            restarted = _restart_managed(proc, state, now)
            action = ("automatic restart requested" if restarted is True else
                      "automatic restart failed" if restarted is False else
                      "alert only (unmanaged process or restart safety limit)")
            lines.append(f"PID {proc.get('pid', '?')} — {', '.join(reasons)} for at least "
                         f"{checks * CHECK_INTERVAL_MIN:.0f} min — {action} — {proc.get('command', '?')}")
        wc.save_state(STATE_FILE, state)
        wc.send_ntfy("High sustained process resource usage", "\n".join(lines))
    if recovered:
        lines = [f"PID {proc.get('pid', '?')} — {proc.get('command', '?')}" for proc in recovered]
        wc.send_ntfy("Process resource usage recovered", "\n".join(lines))
    if not offenders and not recovered:
        print(f"[resources] OK — sampled {len(processes)} processes")
    return bool(offenders or recovered)


if __name__ == "__main__":
    check_once()
