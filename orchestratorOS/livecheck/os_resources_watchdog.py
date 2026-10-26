#!/usr/bin/env python3
"""Alert when any process sustains excessive CPU or resident memory usage."""
import math
import os
import signal
import time
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import watchdog_common as wc

ROOT = wc.ROOT
STATE_FILE = ROOT / ".resource_watchdog_state.json"
wc.load_env()

CPU_THRESHOLD = wc.required_float_env("RESOURCE_CPU_THRESHOLD_PCT")
MEMORY_THRESHOLD_MB = wc.required_float_env("RESOURCE_MEMORY_THRESHOLD_MB")
CONSECUTIVE = wc.required_int_env("RESOURCE_CONSECUTIVE_CHECKS")
RECOVERY_CHECKS = wc.required_int_env("RESOURCE_RECOVERY_CHECKS")
COOLDOWN_MIN = wc.required_float_env("RESOURCE_ALERT_COOLDOWN_MINUTES")
CHECK_INTERVAL_MIN = wc.required_float_env("RESOURCE_CHECK_INTERVAL_MINUTES")
AUTO_RESTART = wc.required_bool_env("RESOURCE_AUTO_RESTART")
RESTART_COOLDOWN_MIN = wc.required_float_env("RESOURCE_RESTART_COOLDOWN_MINUTES")
RESTART_MAX = wc.required_int_env("RESOURCE_RESTART_MAX")
RESTART_WINDOW_H = wc.required_float_env("RESOURCE_RESTART_WINDOW_HOURS")

if not all(
    math.isfinite(v) and v > 0
    for v in (
        CPU_THRESHOLD, MEMORY_THRESHOLD_MB, CONSECUTIVE, RECOVERY_CHECKS,
        COOLDOWN_MIN, CHECK_INTERVAL_MIN, RESTART_COOLDOWN_MIN, RESTART_MAX,
        RESTART_WINDOW_H,
    )
):
    raise ValueError("resource watchdog configuration values must be positive")


def _fleet_scripts():
    scripts = set()
    try:
        for line in (ROOT / "procs.conf").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) >= 7 and fields[6].strip() == "fleet":
                scripts.add(fields[0].strip())
    except OSError:
        pass
    return scripts


def _managed_script(proc):
    parts = proc["command"].split()
    scripts = _fleet_scripts()
    for part in parts[1:]:
        name = os.path.basename(part)
        if name in scripts:
            try:
                if Path(f"/proc/{proc['pid']}/cwd").resolve() == ROOT.resolve():
                    return name
            except (FileNotFoundError, OSError):
                return None
    return None


def _safe_float(val, default=0.0):
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return int(f) if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


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
        and math.isfinite(stamp)
        and cutoff <= stamp <= now
    ]
    if history and now - history[-1] < RESTART_COOLDOWN_MIN * 60:
        return False
    return len(history) < RESTART_MAX


def _restart_managed(proc, state, now):
    script = _managed_script(proc)
    if not AUTO_RESTART or not script or not _restart_allowed(state, script, now):
        return None
    try:
        os.kill(proc["pid"], signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    state.setdefault("restart_history", {}).setdefault(script, []).append(now)
    return True


def _system_ticks():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    return sum(int(value) for value in fields)


def _parse_proc_stat(stat_text: str) -> tuple[str, int, int, int]:
    """Parse (comm, start_time, total_ticks, rss_pages) from /proc/<pid>/stat safely.

    The comm field is enclosed in parentheses and may contain spaces or nested
    parentheses. Splitting after the last ')' keeps all numeric field indices fixed:
    after[0] is state (field 3), after[11] is utime (field 14), after[12] is stime
    (field 15), after[19] is starttime (field 22), and after[21] is rss (field 24).
    """
    lparen = stat_text.find("(")
    rparen = stat_text.rfind(")")
    if lparen < 0 or rparen <= lparen:
        raise ValueError("malformed /proc/stat line")
    comm = stat_text[lparen + 1:rparen]
    after = stat_text[rparen + 1:].split()
    start_time = int(after[19])
    ticks = int(after[11]) + int(after[12])
    rss_pages = int(after[21])
    return comm, start_time, ticks, rss_pages


def _processes():
    result = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm, start_time, ticks, rss_pages = _parse_proc_stat(
                (entry / "stat").read_text()
            )
            key = f"{entry.name}:{start_time}"
            rss_mb = rss_pages * page_size / 1024 / 1024
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
            if not command:
                command = comm
            result[key] = {
                "pid": int(entry.name), "ticks": ticks, "rss_mb": rss_mb,
                "command": command[:180],
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError,
                OSError, ValueError):
            continue
    return result


def _usage(previous, current, previous_total, current_total, cpu_count):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return None
    cpu_cnt = _safe_float(cpu_count, 0.0)
    if cpu_cnt <= 0.0:
        return None
    prev_tot = _safe_float(previous_total, -1.0)
    curr_tot = _safe_float(current_total, -1.0)
    if prev_tot < 0.0 or curr_tot < 0.0:
        return None
    total_delta = curr_tot - prev_tot
    if total_delta <= 0:
        return None
    prev_ticks = _safe_float(previous.get("ticks"), -1.0)
    curr_ticks = _safe_float(current.get("ticks"), -1.0)
    if prev_ticks < 0.0 or curr_ticks < 0.0:
        return None
    tick_delta = curr_ticks - prev_ticks
    if tick_delta < 0:
        return None
    return tick_delta * cpu_cnt * 100.0 / total_delta


def check_once(now=None, *, total_ticks=None, processes=None, cpu_count=None):
    now = _safe_float(now, 0.0)
    if now <= 0.0:
        now = time.time()
    total_ticks = _system_ticks() if total_ticks is None else total_ticks
    processes = _processes() if processes is None else processes
    cpu_count = (os.cpu_count() or 1) if cpu_count is None else _safe_int(cpu_count, 1)
    if cpu_count <= 0:
        cpu_count = 1
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
        rss_mb = _safe_float(proc.get("rss_mb"), 0.0)
        if rss_mb >= MEMORY_THRESHOLD_MB:
            reasons.append(f"RSS {rss_mb:.0f} MB")
        item = tracked.get(key, {"high": 0, "normal": 0, "alerted": False, "last_alert": 0})
        if not isinstance(item, dict):
            item = {"high": 0, "normal": 0, "alerted": False, "last_alert": 0}
        last_alert = _safe_float(item.get("last_alert", 0), 0.0)
        if last_alert > now or last_alert < 0.0:
            last_alert = 0.0
            item["last_alert"] = 0.0
        if reasons:
            item["high"] = max(0, _safe_int(item.get("high", 0), 0)) + 1
            item["normal"] = 0
            item["last_reasons"] = reasons
            if item["high"] >= CONSECUTIVE and (not item.get("alerted") or
                    now - last_alert >= COOLDOWN_MIN * 60):
                offenders.append((proc, reasons, item["high"]))
                item["alerted"] = True
                item["last_alert"] = now
        else:
            item["high"] = 0
            item["normal"] = max(0, _safe_int(item.get("normal", 0), 0)) + 1
            if item.get("alerted") and item["normal"] >= RECOVERY_CHECKS:
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
            lines.append(f"PID {proc['pid']} — {', '.join(reasons)} for at least "
                         f"{checks * CHECK_INTERVAL_MIN:.0f} min — {action} — {proc['command']}")
        wc.save_state(STATE_FILE, state)
        wc.alert("High sustained process resource usage", "\n".join(lines))
    if recovered:
        lines = [f"PID {proc['pid']} — {proc['command']}" for proc in recovered]
        wc.alert("Process resource usage recovered", "\n".join(lines))
    if not offenders and not recovered:
        print(f"[resources] OK — sampled {len(processes)} processes")
    return bool(offenders or recovered)


if __name__ == "__main__":
    check_once()
