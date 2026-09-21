#!/usr/bin/env python3
"""Alert when any process sustains excessive CPU or resident memory usage."""
import os
import signal
import time
from pathlib import Path
import sys
import os
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

if min(CPU_THRESHOLD, MEMORY_THRESHOLD_MB, CONSECUTIVE, RECOVERY_CHECKS,
       COOLDOWN_MIN, CHECK_INTERVAL_MIN, RESTART_COOLDOWN_MIN, RESTART_MAX,
       RESTART_WINDOW_H) <= 0:
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
    for part in parts[1:]:
        name = os.path.basename(part)
        if name in _fleet_scripts():
            try:
                if Path(f"/proc/{proc['pid']}/cwd").resolve() == ROOT.resolve():
                    return name
            except (FileNotFoundError, OSError):
                return None
    return None


def _restart_allowed(state, script, now):
    history = state.setdefault("restart_history", {}).setdefault(script, [])
    cutoff = now - RESTART_WINDOW_H * 3600
    history[:] = [stamp for stamp in history if stamp >= cutoff]
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


def _processes():
    result = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            start_time = int(fields[21])
            key = f"{entry.name}:{start_time}"
            ticks = int(fields[13]) + int(fields[14])
            rss_mb = int(fields[23]) * page_size / 1024 / 1024
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
            if not command:
                command = fields[1].strip("()")
            result[key] = {
                "pid": int(entry.name), "ticks": ticks, "rss_mb": rss_mb,
                "command": command[:180],
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError,
                OSError, ValueError):
            continue
    return result


def _usage(previous, current, previous_total, current_total, cpu_count):
    total_delta = current_total - previous_total
    if total_delta <= 0:
        return None
    tick_delta = current["ticks"] - previous["ticks"]
    if tick_delta < 0:
        return None
    return tick_delta * cpu_count * 100.0 / total_delta


def check_once(now=None, *, total_ticks=None, processes=None, cpu_count=None):
    now = time.time() if now is None else now
    total_ticks = _system_ticks() if total_ticks is None else total_ticks
    processes = _processes() if processes is None else processes
    cpu_count = (os.cpu_count() or 1) if cpu_count is None else cpu_count
    state = wc.load_state(STATE_FILE)
    previous_total = state.get("total_ticks")
    previous = state.get("processes", {})
    tracked = state.get("tracked", {})
    offenders = []
    recovered = []

    for key, proc in processes.items():
        old = previous.get(key)
        cpu = (_usage(old, proc, previous_total, total_ticks, cpu_count)
               if old is not None and previous_total is not None else None)
        reasons = []
        if cpu is not None and cpu >= CPU_THRESHOLD:
            reasons.append(f"CPU {cpu:.1f}%")
        if proc["rss_mb"] >= MEMORY_THRESHOLD_MB:
            reasons.append(f"RSS {proc['rss_mb']:.0f} MB")
        item = tracked.get(key, {"high": 0, "normal": 0, "alerted": False, "last_alert": 0})
        if reasons:
            item["high"] = int(item.get("high", 0)) + 1
            item["normal"] = 0
            item["last_reasons"] = reasons
            if item["high"] >= CONSECUTIVE and (not item.get("alerted") or
                    now - item.get("last_alert", 0) >= COOLDOWN_MIN * 60):
                offenders.append((proc, reasons, item["high"]))
                item["alerted"] = True
                item["last_alert"] = now
        else:
            item["high"] = 0
            item["normal"] = int(item.get("normal", 0)) + 1
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
