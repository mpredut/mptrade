"""Single Source of Truth (SSOT) for Technical Repair (`auto_technical_repair`), Fleet Manifest, and System Telemetry.

Centralizes:
1. `procs.conf` process manifest parsing, heartbeat file resolution, and bot/module alias matching.
2. Canonical taxonomy of technical repair actions, system/fleet/VPN/DNS targets, and diagnosed issue categories.
3. Normalization of LLM repair actions (`normalize_repair_action`) across bots, modules, fleet, VPN/DNS, and system reboot.
4. Workspace configuration auditing (`audit_workspace_configs`) and patchable file boundary/secret validation (`resolve_patchable_target`).
5. Linux OS, memory, disk, internet default route, DNS resolvers, and WireGuard VPN (`wgpia0`) telemetry collection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import urllib.request

from botcore import (
    _parse_bool_flag,
    parse_notify_enabled_flag,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import normalize_venue_name, resolve_cachedb_filepath
from state_io import atomic_write_text, read_json_recovering, save_json_state

parse_finite_float = safe_float

# ---------------------------------------------------------------------------
# 1. Canonical Taxonomy: Repair Actions, Targets & Issue Categories (SSOT)
# ---------------------------------------------------------------------------

BOT_MODULE_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_BOT", "RESTART_MODULE")
FLEET_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_FLEET",)
SYSTEM_INFRA_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_VPN", "PIN_DNS", "REBOOT_SYSTEM")
VENUE_ORDER_REPAIR_ACTIONS: Tuple[str, ...] = ("CANCEL_ORDER", "SYNC_POSITION")

ALL_REPAIR_ACTIONS: Tuple[str, ...] = (
    *BOT_MODULE_REPAIR_ACTIONS,
    *FLEET_REPAIR_ACTIONS,
    *SYSTEM_INFRA_REPAIR_ACTIONS,
    *VENUE_ORDER_REPAIR_ACTIONS,
    "NONE",
)

BOT_OR_FLEET_ERROR_ACTIONS: FrozenSet[str] = frozenset({
    *BOT_MODULE_REPAIR_ACTIONS,
    *FLEET_REPAIR_ACTIONS,
    *VENUE_ORDER_REPAIR_ACTIONS,
})
SYSTEM_OS_ERROR_ACTIONS: FrozenSet[str] = frozenset(SYSTEM_INFRA_REPAIR_ACTIONS)

BOT_AND_ORDER_ERROR_CATEGORIES: FrozenSet[str] = frozenset({
    "BOT_EXCEPTION",
    "ORPHAN_ORDER",
    "FLEET_STALE",
    "CODE_BUG",
    "CONFIG_ERROR",
})
SYSTEM_OS_ERROR_CATEGORIES: FrozenSet[str] = frozenset({
    "SYSTEM_ERROR",
    "SERVER_ERROR",
    "OS_RESOURCE",
    "VPN_ERROR",
    "NETWORK_ERROR",
})

ALL_DIAGNOSED_CATEGORIES: Tuple[str, ...] = (
    "BOT_EXCEPTION",
    "FLEET_STALE",
    "SYSTEM_ERROR",
    "OS_RESOURCE",
    "VPN_ERROR",
    "NETWORK_ERROR",
    "CONFIG_ERROR",
    "CODE_BUG",
    "ORPHAN_ORDER",
    "GUARD_VETO",
    "HEALTHY",
)

DIAGNOSED_SEVERITIES: Tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
ACTIONABLE_ERROR_SEVERITIES: FrozenSet[str] = frozenset({"CRITICAL", "HIGH", "MEDIUM"})

AUDITED_CONFIG_FILES: Tuple[str, ...] = (
    "procs.conf",
    "order_guard.conf",
    "config.env",
    "instruments.conf",
    "macro_intelligence.conf",
)
PROTECTED_SECRET_FILES: FrozenSet[str] = frozenset({".env"})

TRADING_ADMIN_BIN: str = "/usr/local/sbin/trading-admin"

# Maps canonical system/fleet actions to (trading-admin subcommand, systemctl fallback command, log description)
SYSTEM_ADMIN_ACTION_SPECS: Dict[str, Tuple[str, List[str], str]] = {
    "RESTART_FLEET": (
        "restart",
        ["/bin/systemctl", "restart", "python_orchestrator.service"],
        "Restarting entire trading fleet + VPN",
    ),
    "RESTART_VPN": (
        "restart",
        ["/bin/systemctl", "restart", "pia.service"],
        "Restarting VPN service (wgpia0) + fleet",
    ),
    "PIN_DNS": (
        "pin-dns",
        ["/bin/systemctl", "restart", "systemd-resolved"],
        "Pinning DNS and restarting systemd-resolved",
    ),
    "REBOOT_SYSTEM": (
        "reboot",
        ["/bin/systemctl", "reboot"],
        "Rebooting entire Linux system",
    ),
}

ACTION_SYNONYM_MAP: Dict[str, str] = {
    "REBOOT": "REBOOT_SYSTEM",
    "SYSTEM_REBOOT": "REBOOT_SYSTEM",
    "REBOOT_HOST": "REBOOT_SYSTEM",
    "REBOOT_MACHINE": "REBOOT_SYSTEM",
    "REBOOT_SERVER": "REBOOT_SYSTEM",
    "REBOOT_FLEET": "RESTART_FLEET",
    "RESTART_ORCHESTRATOR": "RESTART_FLEET",
    "RESTART_ALL": "RESTART_FLEET",
    "REBOOT_VPN": "RESTART_VPN",
    "RESTART_PIA": "RESTART_VPN",
    "RECONNECT_VPN": "RESTART_VPN",
    "FIX_DNS": "PIN_DNS",
    "RESTART_DNS": "PIN_DNS",
    "RESET_DNS": "PIN_DNS",
    "REBOOT_BOT": "RESTART_BOT",
    "REBOOT_MODULE": "RESTART_MODULE",
    "RESTART_PROCESS": "RESTART_MODULE",
    "RESTART_SERVICE": "RESTART_MODULE",
}

FLEET_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "fleet",
    "flota",
    "all",
    "all_bots",
    "all_processes",
    "orchestrator",
    "python_orchestrator",
    "python_orchestrator.service",
})
VPN_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "vpn",
    "wgpia0",
    "pia",
    "pia.service",
    "piavpn",
    "piavpn.service",
    "pia-vpn",
    "pia-vpn.service",
    "wireguard",
    "tun0",
})
DNS_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "dns",
    "resolved",
    "systemd-resolved",
    "systemd-resolved.service",
    "resolv.conf",
    "pin-dns",
})
SYSTEM_REBOOT_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "system",
    "host",
    "machine",
    "linux",
    "os",
    "server",
    "kernel",
    "intreg_sistemul",
    "full_system",
})


# ---------------------------------------------------------------------------
# 2. Shared Primitive Coercion & JSON Sanitization Helpers
# ---------------------------------------------------------------------------

as_bool_flag = _parse_bool_flag


def clean_json_value(val: Any) -> Any:
    """Recursively sanitize non-finite floats (NaN/Inf -> None) and normalize dict keys to strings."""
    if isinstance(val, float):
        return safe_float(val, default=None)
    if isinstance(val, dict):
        return {str(k): clean_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json_value(v) for v in val]
    return val


# ---------------------------------------------------------------------------
# 3. `procs.conf` Manifest & Process Target Resolution SSOT
# ---------------------------------------------------------------------------

def _extract_script_tokens(pattern: str, cmd: str) -> Set[str]:
    """Extract normalized script paths, basenames, and stems from a procs.conf pattern and start_cmd."""
    tokens: Set[str] = set()
    clean_pat = pattern.rstrip("$").strip()
    if clean_pat:
        tokens.add(clean_pat.lower())
        pat_first = clean_pat.split()[0]
        tokens.add(pat_first.lower())
        base = os.path.basename(pat_first)
        tokens.add(base.lower())
        if base.endswith(".py"):
            tokens.add(base[:-3].lower())
    for tok in (cmd or "").split():
        cleaned = tok.strip("\"'&;")
        if cleaned.endswith(".py"):
            tokens.add(cleaned.lower())
            b_tok = os.path.basename(cleaned)
            tokens.add(b_tok.lower())
            tokens.add(b_tok[:-3].lower())
    return {t for t in tokens if t}


def resolve_heartbeat_paths(
    workspace_dir: str,
    proc_dir: str,
    name: str,
    pattern: str,
    hb_log: str,
    hb_stale_s: float,
    *,
    include_stdout_with_heartbeat_file: bool = False,
) -> Tuple[List[str], str]:
    """Resolve canonical heartbeat file paths and stdout log file path for a procs.conf entry."""
    raw_stem = os.path.basename((name or pattern).rstrip("$"))
    if raw_stem.endswith(".py"):
        raw_stem = raw_stem[:-3]
    clean_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw_stem)
    stdout_log_file = os.path.join(workspace_dir, "logs", f"{clean_stem}.log")

    hb_paths: List[str] = []
    if hb_log:
        for raw_part in hb_log.split(","):
            part = raw_part.strip()
            if not part:
                continue
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
    roles: Tuple[str, ...] = ("bot", "fleet"),
    include_stdout_with_heartbeat_file: bool = False,
    require_cmd: bool = False,
) -> List[Dict[str, Any]]:
    """Parse ``procs.conf`` from ``workspace_dir`` as the single source of truth for bots and fleet modules."""
    procs: List[Dict[str, Any]] = []
    procs_conf = os.path.join(workspace_dir, "procs.conf")
    if not os.path.exists(procs_conf):
        return procs

    try:
        with open(procs_conf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 7:
                    continue
                pat, dr, cmd, label, hblog, hbstale, role = [p.strip() for p in parts[:7]]
                if role not in roles or (require_cmd and not cmd):
                    continue
                hb_val = safe_float(hbstale, default=0.0) or 0.0
                hb_stale_s = hb_val if hb_val > 0 else 0.0

                resolved_dir = dr.replace("$ROOT", workspace_dir)
                resolved_cmd = cmd.replace("$ROOT", workspace_dir).replace("$VENV", venv)
                resolved_hblog = hblog.replace("$ROOT", workspace_dir) if hblog else ""
                name = label or pat
                clean_pat = pat.rstrip("$")

                hb_paths, log_file = resolve_heartbeat_paths(
                    workspace_dir,
                    resolved_dir,
                    name,
                    pat,
                    resolved_hblog,
                    hb_stale_s,
                    include_stdout_with_heartbeat_file=include_stdout_with_heartbeat_file,
                )

                aliases = _extract_script_tokens(pat, resolved_cmd)
                aliases.add(name.lower())
                if name.lower().endswith(".py"):
                    aliases.add(name[:-3].lower())

                procs.append({
                    "name": name,
                    "pattern": pat,
                    "clean_pattern": clean_pat,
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


def fleet_script_basenames(workspace_dir: str) -> Set[str]:
    """Return the set of Python script basenames for ``role=fleet`` processes in ``procs.conf``."""
    scripts: Set[str] = set()
    for p in parse_procs_manifest(workspace_dir, roles=("fleet",)):
        raw_pat = str(p.get("clean_pattern") or "").strip()
        token = raw_pat.split()[0] if raw_pat else ""
        if token:
            scripts.add(os.path.basename(token))
    return scripts


def resolve_manifest_process(target: Any, workspace_dir: str) -> Optional[Dict[str, Any]]:
    """Resolve a bot or module name/pattern/script stem against ``procs.conf``."""
    if not isinstance(target, str) or not target.strip():
        return None
    norm = target.strip().rstrip("$").lower()
    norm_base = os.path.basename(norm)
    norm_stem = norm_base[:-3] if norm_base.endswith(".py") else norm_base

    manifest = parse_procs_manifest(workspace_dir)
    # Pass 1: exact name, pattern, or basename match
    for p in manifest:
        p_name = p["name"].lower()
        p_pat = p["clean_pattern"].lower()
        p_base = os.path.basename(p_pat).lower()
        if norm in (p_name, p_pat, p_base) or norm_base in (p_name, p_base):
            return p
    # Pass 2: alias or stem match (e.g. 'hl_bot' -> 'HL-bot', 'price_watcher.py' -> 'price_watcher')
    for p in manifest:
        alias_set = set(p.get("aliases") or [])
        if norm in alias_set or norm_base in alias_set or norm_stem in alias_set:
            return p
    return None


# ---------------------------------------------------------------------------
# 4. Normalization of Technical Repair Actions & Targets (SSOT)
# ---------------------------------------------------------------------------

def build_system_admin_command(
    action_type: str,
    *,
    admin_bin: str = TRADING_ADMIN_BIN,
) -> Optional[Tuple[List[str], str]]:
    """Build the privileged ``sudo -n`` command and log description for a system/fleet repair action."""
    spec = SYSTEM_ADMIN_ACTION_SPECS.get(str(action_type or "").strip().upper())
    if spec is None:
        return None
    admin_subcmd, systemctl_fallback, desc = spec
    if os.path.exists(admin_bin):
        return (["sudo", "-n", admin_bin, admin_subcmd], desc)
    return (["sudo", "-n", *systemctl_fallback], desc)


def normalize_repair_action(act: Any, workspace_dir: str) -> Dict[str, Any]:
    """Normalize a raw repair action dict into canonical action, venue, symbol, and resolved process metadata."""
    if not isinstance(act, dict):
        return {"action": "NONE", "venue": "fleet", "symbol": "", "order_id": "", "reason": "", "matched_process": None}

    raw_action = act.get("action")
    action_type = "" if isinstance(raw_action, bool) or raw_action is None else str(raw_action).strip().upper()
    action_type = ACTION_SYNONYM_MAP.get(action_type, action_type) or "NONE"

    raw_venue = act.get("venue") if act.get("venue") is not None else act.get("provider")
    venue = normalize_venue_name(raw_venue, default="binance")

    raw_sym = (
        act.get("symbol")
        or act.get("bot")
        or act.get("module")
        or act.get("target")
        or act.get("service")
    )
    symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip()

    raw_oid = (
        act.get("order_id")
        if act.get("order_id") is not None
        else (act.get("orderId") if act.get("orderId") is not None else act.get("oid"))
    )
    from providers.base import extract_order_id
    order_id = extract_order_id(raw_oid, default="") or ""

    raw_reason = act.get("reason")
    reason = "" if isinstance(raw_reason, bool) or raw_reason is None else str(raw_reason)

    sym_lower = symbol.lower()
    venue_lower = venue.lower()

    # Upgrade generic restart/reboot actions targeting fleet, VPN, DNS, or full system to canonical actions
    if action_type in (*BOT_MODULE_REPAIR_ACTIONS, "RESTART_FLEET", "RESTART_VPN", "PIN_DNS", "REBOOT_SYSTEM"):
        if sym_lower in SYSTEM_REBOOT_TARGET_ALIASES and action_type == "REBOOT_SYSTEM":
            venue = "system"
            symbol = symbol or "SYSTEM"
        elif sym_lower in FLEET_TARGET_ALIASES and action_type in BOT_MODULE_REPAIR_ACTIONS:
            action_type = "RESTART_FLEET"
            venue = "fleet"
            symbol = symbol or "FLEET"
        elif (sym_lower in VPN_TARGET_ALIASES or venue_lower in VPN_TARGET_ALIASES) and action_type in BOT_MODULE_REPAIR_ACTIONS:
            action_type = "RESTART_VPN"
            venue = "system"
            symbol = symbol or "wgpia0"
        elif (sym_lower in DNS_TARGET_ALIASES or venue_lower in DNS_TARGET_ALIASES) and action_type in BOT_MODULE_REPAIR_ACTIONS:
            action_type = "PIN_DNS"
            venue = "system"
            symbol = symbol or "dns"
        elif sym_lower in SYSTEM_REBOOT_TARGET_ALIASES and action_type in BOT_MODULE_REPAIR_ACTIONS:
            action_type = "REBOOT_SYSTEM"
            venue = "system"
            symbol = symbol or "SYSTEM"

    matched_proc: Optional[Dict[str, Any]] = None
    if action_type in BOT_MODULE_REPAIR_ACTIONS and symbol:
        matched_proc = resolve_manifest_process(symbol, workspace_dir)
        if matched_proc is not None:
            symbol = matched_proc["name"]
            venue = "fleet"

    return {
        "action": action_type,
        "venue": venue,
        "symbol": symbol,
        "order_id": order_id,
        "reason": reason,
        "matched_process": matched_proc,
    }


# ---------------------------------------------------------------------------
# 5. Workspace Config Audit & Patch Target Validation SSOT
# ---------------------------------------------------------------------------

def resolve_patchable_target(target_file: str, workspace_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate that ``target_file`` is an existing file inside ``workspace_dir`` and not a protected secret file."""
    if not isinstance(target_file, str) or not target_file.strip():
        return None, "empty target_file"
    real_ws = os.path.realpath(workspace_dir)
    real_target = os.path.realpath(
        target_file if os.path.isabs(target_file) else os.path.join(workspace_dir, target_file.strip())
    )
    try:
        in_workspace = os.path.commonpath([real_ws, real_target]) == real_ws
    except ValueError:
        in_workspace = False
    if not in_workspace:
        return None, f"Refusing autonomous patch outside workspace_dir: {target_file}"
    if not os.path.isfile(real_target):
        return None, f"Target patch file does not exist: {target_file}"
    base_name = os.path.basename(real_target)
    if base_name in PROTECTED_SECRET_FILES:
        return None, f"Refusing autonomous patch on protected secret credentials file: {target_file}"
    return real_target, None


def audit_workspace_configs(workspace_dir: str) -> Dict[str, Any]:
    """Audit core workspace configuration files for syntax errors, invalid modes, or broken definitions."""
    checked: List[str] = []
    issues: List[str] = []
    for cfg_name in AUDITED_CONFIG_FILES:
        cfg_path = os.path.join(workspace_dir, cfg_name)
        if not os.path.isfile(cfg_path):
            continue
        checked.append(cfg_name)
        try:
            with open(cfg_path, "r", encoding="utf-8", errors="replace") as cf:
                raw_lines = cf.readlines()
            if cfg_name == "procs.conf":
                for idx, raw_l in enumerate(raw_lines, start=1):
                    line = raw_l.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 7:
                        issues.append(f"procs.conf:{idx} has {len(parts)} fields (expected >=7)")
                        continue
                    pat, _, _, label, _, _, role = parts[:7]
                    if role not in ("bot", "fleet"):
                        issues.append(f"procs.conf:{idx} ({label or pat}) has invalid role {role!r}")
            elif cfg_name in ("order_guard.conf", "config.env"):
                seen_keys: Dict[str, str] = {}
                for idx, raw_l in enumerate(raw_lines, start=1):
                    line = raw_l.split("#", 1)[0].strip()
                    if not line:
                        continue
                    if "=" not in line:
                        issues.append(f"{cfg_name}:{idx} malformed line missing '=': {line[:60]!r}")
                        continue
                    k, v = line.split("=", 1)
                    k_clean = k.strip().lower()
                    v_clean = v.strip().strip("\"'")
                    if not k_clean:
                        issues.append(f"{cfg_name}:{idx} empty key name")
                    else:
                        seen_keys[k_clean] = v_clean
                if cfg_name == "order_guard.conf":
                    for mode_key in ("supervisor_mode", "llm_guard_mode", "autonomous_trading_mode"):
                        m_val = seen_keys.get(mode_key, "").lower()
                        if m_val and m_val not in ("enforce", "shadow", "off"):
                            issues.append(f"order_guard.conf invalid {mode_key}={m_val!r}")
            elif cfg_name == "instruments.conf":
                for idx, raw_l in enumerate(raw_lines, start=1):
                    line = raw_l.split("#", 1)[0].strip()
                    if not line:
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        if not line[1:-1].strip():
                            issues.append(f"instruments.conf:{idx} empty section header")
                        continue
                    if "|" in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) < 6:
                            issues.append(f"instruments.conf:{idx} has {len(parts)} fields (expected >=6)")
                        continue
                    if "=" in line:
                        k, _ = line.split("=", 1)
                        if not k.strip():
                            issues.append(f"instruments.conf:{idx} empty key name")
                        continue
                    issues.append(f"instruments.conf:{idx} malformed line: {line[:60]!r}")
        except Exception as exc:
            issues.append(f"{cfg_name} read error: {exc}")
    return {"checked_files": checked, "issues": issues}


# ---------------------------------------------------------------------------
# 6. Unified Linux OS, Memory, Disk, Internet, DNS & VPN Telemetry Collector
# ---------------------------------------------------------------------------

def gather_basic_os_resources(workspace_dir: str) -> Dict[str, Any]:
    """Collect CPU load averages and workspace disk usage."""
    stats: Dict[str, Any] = {}
    try:
        load1, load5, load15 = os.getloadavg()
        stats["load_avg"] = [round(load1, 2), round(load5, 2), round(load15, 2)]
        disk = shutil.disk_usage(workspace_dir)
        stats["disk_used_pct"] = round((disk.used / disk.total) * 100, 1) if disk.total > 0 else 0.0
        stats["disk_free_gb"] = round(disk.free / (1024 ** 3), 1)
    except Exception:
        pass
    return stats


def gather_system_and_vpn_telemetry(
    workspace_dir: str,
    *,
    base_resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect complete Linux OS, RAM, disk, WireGuard VPN (`wgpia0`), internet/DNS, watchdog, and config audit state."""
    sys_vpn: Dict[str, Any] = dict(base_resources) if isinstance(base_resources, dict) else gather_basic_os_resources(workspace_dir)

    try:
        if os.path.exists("/proc/meminfo"):
            mem_kv: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as mf:
                for mline in mf:
                    parts = mline.split(":")
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v_parts = parts[1].strip().split()
                        if v_parts and v_parts[0].isdigit():
                            mem_kv[k] = int(v_parts[0])
            if mem_kv.get("MemTotal", 0) > 0 and "MemAvailable" in mem_kv:
                total_kb = mem_kv["MemTotal"]
                avail_kb = mem_kv["MemAvailable"]
                sys_vpn["mem_used_pct"] = round(((total_kb - avail_kb) / total_kb) * 100.0, 1)
                sys_vpn["mem_avail_mb"] = round(avail_kb / 1024.0, 0)
    except Exception:
        pass

    vpn_if = os.environ.get("PIA_VPN_IF", "wgpia0").strip() or "wgpia0"
    vpn_net_dir = f"/sys/class/net/{vpn_if}"
    if os.path.exists(vpn_net_dir):
        try:
            with open(f"{vpn_net_dir}/operstate", "r", encoding="utf-8") as vf:
                oper = vf.read().strip().lower()
        except Exception:
            oper = "unknown"
        sys_vpn["vpn_interface"] = vpn_if
        sys_vpn["vpn_tunnel_up"] = oper in ("up", "unknown")
    else:
        sys_vpn["vpn_interface"] = vpn_if
        sys_vpn["vpn_tunnel_up"] = False

    try:
        default_route_present = False
        if os.path.exists("/proc/net/route"):
            with open("/proc/net/route", "r", encoding="utf-8", errors="replace") as rf:
                for rline in rf.readlines()[1:]:
                    rcols = rline.strip().split()
                    if len(rcols) >= 2 and rcols[1] == "00000000":
                        default_route_present = True
                        break
        sys_vpn["default_route_present"] = default_route_present
    except Exception:
        pass

    try:
        dns_servers: List[str] = []
        if os.path.exists("/etc/resolv.conf"):
            with open("/etc/resolv.conf", "r", encoding="utf-8", errors="replace") as df:
                for dline in df:
                    dline = dline.split("#", 1)[0].strip()
                    if dline.startswith("nameserver"):
                        dparts = dline.split()
                        if len(dparts) >= 2:
                            dns_servers.append(dparts[1])
        if dns_servers:
            sys_vpn["dns_servers"] = dns_servers
    except Exception:
        pass

    sys_vpn["vpn_outage_active"] = os.path.exists("/var/lib/pia_selfheal/outage_since")
    sys_vpn["internet_healthy"] = bool(
        sys_vpn.get("default_route_present", True)
        and not sys_vpn.get("vpn_outage_active", False)
    )

    rw_state = read_json_recovering(os.path.join(workspace_dir, ".resource_watchdog_state.json"))
    if isinstance(rw_state, dict) and isinstance(rw_state.get("tracked"), dict):
        offenders = [
            {"key": k, "reasons": v.get("last_reasons"), "consecutive_high": v.get("high")}
            for k, v in rw_state["tracked"].items()
            if isinstance(v, dict) and (safe_float(v.get("high"), default=0.0) or 0.0) > 0
        ]
        if offenders:
            sys_vpn["resource_watchdog_offenders"] = offenders

    sys_vpn["config_audit"] = audit_workspace_configs(workspace_dir)
    return sys_vpn


# ---------------------------------------------------------------------------
# 7. Log Anomaly Harvesting & Audit Error Classification SSOT
# ---------------------------------------------------------------------------

_FULL_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_GLOG_TS_RE = re.compile(r"^[IWEF](\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_CLOCK_TS_RE = re.compile(r"^\.*\[?(\d{2}):(\d{2}):(\d{2})\b")


@dataclass
class LogEventSummary:
    """Summary of log anomalies and guard interventions collected within the lookback window."""
    total_scanned_files: int = 0
    error_count: int = 0
    warning_count: int = 0
    critical_events: List[str] = field(default_factory=list)
    guard_interventions: List[str] = field(default_factory=list)
    persistent_errors: List[Dict[str, Any]] = field(default_factory=list)


def recent_log_lines(lines: List[str], file_mtime: float, cutoff_ts: float) -> List[str]:
    """Return the tail of ``lines`` whose own timestamps are >= cutoff_ts (chronological order)."""
    if not isinstance(lines, (list, tuple)) or not lines:
        return []
    fm = safe_pos_float(file_mtime, default=None) or time.time()
    cutoff_ts = safe_nonneg_float(cutoff_ts, default=0.0) or 0.0
    try:
        ref = datetime.fromtimestamp(fm)
    except (ValueError, OverflowError, OSError):
        fm = time.time()
        ref = datetime.fromtimestamp(fm)
    day = ref.date()
    prev_ts = float(fm)
    picked: List[str] = []

    def _resolve_fold_ts(dt_naive: datetime, upper_ts: float) -> float:
        ts0 = dt_naive.replace(fold=0).timestamp()
        ts1 = dt_naive.replace(fold=1).timestamp()
        if ts1 > ts0 and ts1 <= upper_ts + 300.0:
            return ts1
        return ts0

    pending_unstamped: List[str] = []
    for line in reversed(lines):
        if not isinstance(line, str):
            continue
        ts: Optional[float] = None
        has_stamp_token = False
        m = _FULL_TS_RE.search(line)
        if m:
            has_stamp_token = True
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")
                day = dt.date()
                ts = _resolve_fold_ts(dt, prev_ts)
                prev_ts = ts
            except ValueError:
                pass
        else:
            g = _GLOG_TS_RE.match(line)
            c = None if g else _CLOCK_TS_RE.match(line)
            if g:
                has_stamp_token = True
                try:
                    dt = datetime(
                        ref.year, int(g.group(1)), int(g.group(2)),
                        int(g.group(3)), int(g.group(4)), int(g.group(5)),
                    )
                    if (dt - ref).total_seconds() > 86400:
                        dt = datetime(
                            ref.year - 1, int(g.group(1)), int(g.group(2)),
                            int(g.group(3)), int(g.group(4)), int(g.group(5)),
                        )
                    day = dt.date()
                    ts = _resolve_fold_ts(dt, prev_ts)
                    prev_ts = ts
                except ValueError:
                    pass
            elif c and int(c.group(1)) < 24 and int(c.group(2)) < 60 and int(c.group(3)) < 60:
                has_stamp_token = True
                secs = int(c.group(1)) * 3600 + int(c.group(2)) * 60 + int(c.group(3))
                dt_same = datetime.combine(day, datetime.min.time()) + timedelta(seconds=secs)
                ts0 = dt_same.replace(fold=0).timestamp()
                ts1 = dt_same.replace(fold=1).timestamp()
                if ts1 > ts0 and ts1 <= prev_ts + 300.0:
                    ts = ts1
                elif ts0 <= prev_ts + 300.0:
                    ts = ts0
                else:
                    day -= timedelta(days=1)
                    dt_prev = datetime.combine(day, datetime.min.time()) + timedelta(seconds=secs)
                    ts = _resolve_fold_ts(dt_prev, prev_ts)
                prev_ts = ts
        if ts is not None and ts < cutoff_ts:
            break
        if has_stamp_token:
            if pending_unstamped:
                picked.extend(pending_unstamped)
                pending_unstamped.clear()
            picked.append(line)
        else:
            pending_unstamped.append(line)
    else:
        if pending_unstamped:
            picked.extend(pending_unstamped)
    picked.reverse()
    return picked


def normalize_error_signature(line: str) -> str:
    """Strip timestamps, PIDs, and hex addresses so recurring log errors group into one signature."""
    s = _FULL_TS_RE.sub("", line)
    s = _GLOG_TS_RE.sub("", s)
    s = _CLOCK_TS_RE.sub("", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"\b\d{4,}\b", "NUM", s)
    return " ".join(s.strip().split())[:180]


def harvest_log_anomalies_from_dirs(logs_dirs: List[str], lookback_hours: float) -> LogEventSummary:
    """Parse recent log files across ``logs_dirs`` for errors, persistent unresolved exceptions, and guard vetoes."""
    cutoff_ts = time.time() - (lookback_hours * 3600.0)
    critical_events: List[str] = []
    guard_interventions: List[str] = []
    persistent_errors: List[Dict[str, Any]] = []
    error_count = 0
    warning_count = 0
    scanned_count = 0

    err_pattern = re.compile(
        r"(\bERROR\b|\bCRITICAL\b(?!_SHOCK)|Traceback|fail-closed|BUY BLOCKED|\bException\b)",
        re.IGNORECASE,
    )
    benign_warn_pattern = re.compile(r"(\[CMCPlatform\]\s+Warning|\[WARNING\])", re.IGNORECASE)
    supervisor_summary_pattern = re.compile(
        r"(Autonomous AI Supervisor cycle complete:|\[(ENFORCE|SHADOW)\]\s+auto_|"
        r"mailer:\s+Email delivery skipped|Email gated by common delivery policy|"
        r"ntfy delivery gated by common policy|Dispatched ntfy alert to|"
        r"<!--\s*a padding to disable MSIE|<(?:head|center)>.*500 Internal Server Error|"
        r"Exception ignored in:.*BaseSubprocessTransport\.__del__|RuntimeError:\s+Event loop is closed|"
        r"curl:\s*\(22\).*returned error:\s*429|deadman:\s*(?:curl ERROR after retries|ntfy rate-limited))",
        re.IGNORECASE,
    )
    guard_pattern = re.compile(r"(\[.*GUARD.*\]|vetoed|downscaled|HARD_VETO|DOWNSCALE_QTY)", re.IGNORECASE)

    for l_dir in logs_dirs:
        if not os.path.isdir(l_dir):
            continue
        try:
            for fname in os.listdir(l_dir):
                if not (fname.endswith(".log") or fname.endswith(".txt")):
                    continue
                fpath = os.path.join(l_dir, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    if mtime < cutoff_ts:
                        continue
                    scanned_count += 1
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        lines = recent_log_lines(f.readlines()[-1000:], mtime, cutoff_ts)
                        total_lines = len(lines)
                        tail_start_idx = max(0, total_lines - max(10, total_lines // 4))
                        file_sig_stats: Dict[str, Dict[str, Any]] = {}
                        for idx, line in enumerate(lines):
                            if supervisor_summary_pattern.search(line):
                                continue
                            if (
                                ("Exception ignored in:" in line or "Traceback (most recent call last):" in line)
                                and any(
                                    "BaseSubprocessTransport.__del__" in lines[j]
                                    for j in range(max(0, idx - 1), min(total_lines, idx + 3))
                                )
                            ):
                                continue
                            if err_pattern.search(line) and not (
                                benign_warn_pattern.search(line)
                                and not re.search(
                                    r"(Traceback|CRITICAL(?!_SHOCK)|fail-closed|BUY BLOCKED)",
                                    line,
                                    re.IGNORECASE,
                                )
                            ):
                                error_count += 1
                                cleaned = line.strip()
                                if cleaned not in critical_events and len(critical_events) < 50:
                                    critical_events.append(cleaned)
                                sig = normalize_error_signature(cleaned)
                                if sig:
                                    st = file_sig_stats.setdefault(
                                        sig,
                                        {"file": fname, "count": 0, "unresolved_at_tail": False, "sample": cleaned},
                                    )
                                    st["count"] += 1
                                    st["sample"] = cleaned
                                    if idx >= tail_start_idx:
                                        st["unresolved_at_tail"] = True
                            elif guard_pattern.search(line):
                                warning_count += 1
                                cleaned = line.strip()
                                if cleaned not in guard_interventions and len(guard_interventions) < 50:
                                    guard_interventions.append(cleaned)
                        for st in file_sig_stats.values():
                            if st["count"] >= 2 or st["unresolved_at_tail"]:
                                persistent_errors.append(st)
                except Exception:
                    pass
        except Exception:
            pass

    persistent_errors.sort(
        key=lambda item: (1 if item.get("unresolved_at_tail") else 0, int(item.get("count", 0))),
        reverse=True,
    )
    return LogEventSummary(
        total_scanned_files=scanned_count,
        error_count=error_count,
        warning_count=warning_count,
        critical_events=critical_events,
        guard_interventions=guard_interventions,
        persistent_errors=persistent_errors[:25],
    )


def is_bot_or_code_error_report(report: Any) -> bool:
    """Return True strictly for trading bot exceptions, fleet/config anomalies, or code/config patches."""
    for issue in getattr(report, "diagnosed_issues", None) or []:
        if not isinstance(issue, dict):
            continue
        cat = str(issue.get("category") or "").strip().upper()
        sev = str(issue.get("severity") or "").strip().upper()
        if cat in BOT_AND_ORDER_ERROR_CATEGORIES and sev in ACTIONABLE_ERROR_SEVERITIES:
            return True
    code_rem = getattr(report, "code_remediation", None)
    if isinstance(code_rem, dict) and _parse_bool_flag(code_rem.get("has_fix"), False):
        return True
    if getattr(report, "tests_passed", None) is False:
        return True
    for act in getattr(report, "venue_actions", None) or []:
        if isinstance(act, dict):
            norm_act = normalize_repair_action(act, "").get("action")
            if norm_act in BOT_OR_FLEET_ERROR_ACTIONS:
                return True
    return False


def is_system_os_error_report(report: Any) -> bool:
    """Return True for Linux OS, internet/VPN, memory, disk space, or server infrastructure anomalies."""
    for issue in getattr(report, "diagnosed_issues", None) or []:
        if not isinstance(issue, dict):
            continue
        cat = str(issue.get("category") or "").strip().upper()
        sev = str(issue.get("severity") or "").strip().upper()
        if cat in SYSTEM_OS_ERROR_CATEGORIES and sev in ACTIONABLE_ERROR_SEVERITIES:
            return True
        if sev in ACTIONABLE_ERROR_SEVERITIES and cat not in BOT_AND_ORDER_ERROR_CATEGORIES:
            return True
    for act in getattr(report, "venue_actions", None) or []:
        if isinstance(act, dict):
            norm_act = normalize_repair_action(act, "").get("action")
            if norm_act in SYSTEM_OS_ERROR_ACTIONS:
                return True
    return False


def is_error_report(report: Any) -> bool:
    """Return True if the audit report diagnoses any bot/order/code error or system/OS error."""
    return is_bot_or_code_error_report(report) or is_system_os_error_report(report)


def apply_code_patch_safely(
    target_file: str,
    search_block: str,
    replace_block: str,
    rationale: str,
    workspace_dir: str,
    log: Optional[Any] = None,
) -> bool:
    """Apply code or configuration patch, run pytest test suite, commit on 100% green or roll back immediately."""
    real_target, err_msg = resolve_patchable_target(target_file, workspace_dir)
    if real_target is None:
        if log:
            log.warning("%s", err_msg)
        return False

    if not search_block or search_block == replace_block:
        if log:
            log.warning("Refusing empty or no-op autonomous patch on %s.", target_file)
        return False

    content: Optional[str] = None
    try:
        with open(real_target, "r", encoding="utf-8") as f:
            content = f.read()

        if search_block not in content:
            if log:
                log.warning("Search block not found in %s. Skipping patch.", target_file)
            return False

        new_content = content.replace(search_block, replace_block, 1)
        atomic_write_text(real_target, new_content)

        pytest_bin = "pytest"
        for cand in (".venv/bin/pytest", "myenv/bin/pytest"):
            cand_abs = os.path.join(workspace_dir, cand)
            if os.path.exists(cand_abs):
                pytest_bin = cand_abs
                break
        test_cmd = [pytest_bin, "tests/"]
        res = subprocess.run(test_cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=240)

        if res.returncode == 0:
            if log:
                log.info("Self-healing patch verified with 100%% green tests. Committing to git.")
            subprocess.run(["git", "add", real_target], cwd=workspace_dir, check=False)
            commit_msg = f"[AI-SUPERVISOR] Autonomous self-heal: {rationale}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=workspace_dir, check=False)
            subprocess.run(["git", "push", "origin", "main"], cwd=workspace_dir, check=False)
            return True
        if log:
            log.warning("Self-healing patch broke tests (exit code %d). Performing immediate rollback.", res.returncode)
        atomic_write_text(real_target, content)
        subprocess.run(["git", "checkout", "--", real_target], cwd=workspace_dir, check=False)
        return False
    except Exception as e:
        if log:
            log.error("Error during safe code patching: %s. Performing rollback.", e)
        if content is not None:
            try:
                atomic_write_text(real_target, content)
            except OSError:
                pass
        subprocess.run(["git", "checkout", "--", real_target], cwd=workspace_dir, check=False)
        return False


def execute_venue_or_fleet_repair_action(
    act: Dict[str, Any],
    workspace_dir: str,
    resolve_instrument_fn: Optional[Any] = None,
    log: Optional[Any] = None,
) -> bool:
    """Execute a single repair action (bot/module/fleet/VPN/DNS restart, system reboot, or order sync/cancel)."""
    if not isinstance(act, dict):
        return False
    norm = normalize_repair_action(act, workspace_dir)
    action_type = norm["action"]
    venue = norm["venue"]
    symbol = norm["symbol"]
    order_id = norm["order_id"]
    reason = norm["reason"]
    matched = norm["matched_process"]

    if action_type in BOT_MODULE_REPAIR_ACTIONS:
        if not symbol:
            if log:
                log.warning("[ENFORCE] %s missing bot/module name in action: %s", action_type, act)
            return False
        try:
            if matched is None:
                if log:
                    log.warning("[ENFORCE] Refusing %s for unknown process %r (not in procs.conf)", action_type, symbol)
                return False
            pattern = matched.get("clean_pattern") or matched["pattern"].rstrip("$")
            if log:
                log.warning("[ENFORCE EXECUTING] Restarting fleet bot/module %s (pattern=%s): %s", matched["name"], pattern, reason)
            subprocess.run(["pkill", "-f", pattern], check=False)
            return True
        except Exception as e:
            if log:
                log.error("[ENFORCE] Failed restarting fleet bot/module %s: %s", symbol, e)
            return False

    if action_type in SYSTEM_ADMIN_ACTION_SPECS:
        try:
            built = build_system_admin_command(action_type)
            if built is None:
                return False
            cmd, desc = built
            if log:
                log.warning("[ENFORCE EXECUTING] %s: %s", desc, reason)
            res = subprocess.run(cmd, check=False)
            return getattr(res, "returncode", 0) == 0
        except Exception as e:
            if log:
                log.error("[ENFORCE] Failed %s: %s", action_type, e)
            return False

    if action_type in {"CANCEL_ORDER", "SYNC_POSITION"}:
        is_cancel = action_type == "CANCEL_ORDER"
        if not symbol or (is_cancel and not order_id) or not callable(resolve_instrument_fn):
            if is_cancel and log:
                log.warning("[ENFORCE] CANCEL_ORDER missing symbol or order_id: %s", act)
            return False
        try:
            inst = resolve_instrument_fn(symbol.upper(), venue=venue)
            if inst is None:
                if is_cancel and log:
                    log.warning("[ENFORCE] CANCEL_ORDER could not resolve instrument %s on %s", symbol, venue)
                return False
            api, prov = getattr(inst, "_api", None), getattr(inst, "_provider", None)
            if is_cancel:
                if log:
                    log.warning("[ENFORCE EXECUTING] Canceling order %s for %s on %s: %s", order_id, inst.symbol, inst.provider_name, reason)
                if api is not None and hasattr(api, "cancel_order"):
                    api.cancel_order(inst.symbol, order_id, provider_name=inst.provider_name)
                elif prov is not None and hasattr(prov, "cancel_order"):
                    prov.cancel_order(inst.symbol, order_id)
                else:
                    if log:
                        log.warning("[ENFORCE] Provider for %s does not expose cancel_order", inst.symbol)
                    return False
            else:
                if log:
                    log.info("[ENFORCE EXECUTING] Syncing open orders/state for %s on %s: %s", inst.symbol, inst.provider_name, reason)
                if api is not None and hasattr(api, "open_orders"):
                    api.open_orders(inst.symbol, provider_name=inst.provider_name)
                elif prov is not None and hasattr(prov, "open_orders"):
                    prov.open_orders(inst.symbol)
            return True
        except Exception as e:
            if log:
                if is_cancel:
                    log.error("[ENFORCE] Failed CANCEL_ORDER %s on %s (%s): %s", order_id, symbol, venue, e)
                else:
                    log.error("[ENFORCE] Failed SYNC_POSITION for %s on %s: %s", symbol, venue, e)
            return False

    if log:
        log.warning("[ENFORCE] Unsupported venue/fleet action %r: %s", action_type, act)
    return False


def harvest_heartbeats_and_state_files(
    cache_dir: str,
    workspace_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Harvest active heartbeat JSON files and watchdog state file ages."""
    _rj = read_json_fn or read_json_recovering
    active_heartbeats: Dict[str, Any] = {}
    for hb_name in (
        "macro_analyzer.heartbeat",
        "globaltelemetry_collector.heartbeat",
        "rtrade.heartbeat",
    ):
        data = _rj(os.path.join(cache_dir, hb_name))
        if isinstance(data, dict) and data:
            active_heartbeats[hb_name] = data

    state_files: Dict[str, Any] = {}
    for s_file in (".watchdog_state.json", ".anomaly_watchdog_state.json"):
        s_path = os.path.join(workspace_dir, s_file)
        if os.path.exists(s_path):
            try:
                mtime = os.path.getmtime(s_path)
                state_files[s_file] = {"age_hours": max(0.0, (now_ts - mtime) / 3600.0)}
            except OSError:
                pass
    return active_heartbeats, state_files


def format_supervisor_ntfy_notification(
    report: Any,
    topic_key: str,
    ntfy_topic_error: str,
    ntfy_topic_server: str,
    ntfy_topic: str,
    *,
    is_bot_error: Optional[bool] = None,
    is_sys_error: Optional[bool] = None,
) -> Tuple[str, str, str, str, bool]:
    """Build (title, body, priority, target_topic, is_any_error) for a SupervisorAuditReport ntfy push."""
    from orchestratorOS.agy.topics import TOPIC_ALIASES, TOPIC_SPECS, topic_spec

    issues_cnt = sum(
        1 for i in (getattr(report, "diagnosed_issues", None) or [])
        if isinstance(i, dict) and str(i.get("category") or "").strip().upper() != "HEALTHY"
    )
    actions_cnt = sum(
        1 for a in (getattr(report, "venue_actions", None) or [])
        if isinstance(a, dict) and str(a.get("action") or "NONE").strip().upper() != "NONE"
    )
    code_rem = getattr(report, "code_remediation", None)
    has_fix = bool(
        isinstance(code_rem, dict)
        and _parse_bool_flag(code_rem.get("has_fix", False), False)
    )
    if has_fix:
        actions_cnt += 1
    mode_label = str(getattr(report, "mode", "shadow")).upper()
    if is_bot_error is None:
        is_bot_error = is_bot_or_code_error_report(report)
    if is_sys_error is None:
        is_sys_error = is_system_os_error_report(report)
    is_any_error = bool(is_bot_error or is_sys_error)

    spec = (
        topic_spec(topic_key)
        if (topic_key in TOPIC_SPECS or topic_key in TOPIC_ALIASES)
        else topic_spec("auto_technical_repair")
    )
    if is_bot_error:
        title = f"🚨 {spec.bracket_tag} ERROR Issues: {issues_cnt} | Actions: {actions_cnt}"
        target_topic = ntfy_topic_error or ntfy_topic or "ntfy-error-1978"
        priority = "high"
    elif is_sys_error:
        title = f"🚨 {spec.bracket_tag} SYSTEM Issues: {issues_cnt} | Actions: {actions_cnt}"
        target_topic = ntfy_topic_server or ntfy_topic or "ntfy-server-1978"
        priority = "high"
    else:
        title = f"{spec.ntfy_prefix} Issues: {issues_cnt} | Actions: {actions_cnt}"
        target_topic = ntfy_topic_server or ntfy_topic or "ntfy-server-1978"
        priority = "default"

    body_lines = [
        f"Mode: {mode_label} · Status: {'Executed' if getattr(report, 'actions_executed', False) else 'Simulated/Observed'}",
        f"{getattr(report, 'summary', '')}",
    ]
    diag_issues = getattr(report, "diagnosed_issues", None)
    if diag_issues:
        top_issue = diag_issues[0]
        if isinstance(top_issue, dict):
            body_lines.append(f"Primary Issue: [{top_issue.get('severity', 'INFO')}] {top_issue.get('description', '')}")

    if has_fix and isinstance(code_rem, dict):
        patch_file = code_rem.get("target_file", "unknown")
        body_lines.append(f"Code Patch Proposed: {patch_file} (tests={getattr(report, 'tests_passed', None)})")

    m_decs = getattr(report, "market_trading_decisions", None)
    m_single = getattr(report, "market_trading_decision", None)
    decisions_to_show = m_decs or ([m_single] if m_single else [])
    for m_dec in decisions_to_show:
        if not isinstance(m_dec, dict):
            continue
        m_decision = str(m_dec.get("decision", "HOLD")).upper()
        if m_decision in ("BUY", "SELL"):
            m_sym = m_dec.get("symbol", "")
            m_venue = m_dec.get("venue", "")
            m_conf = safe_nonneg_float(m_dec.get("confidence"), default=0.0) or 0.0
            m_usd = safe_nonneg_float(m_dec.get("suggested_notional_usd"), default=0.0) or 0.0
            venue_sfx = f" ({m_venue})" if m_venue else ""
            body_lines.append(f"Trading Intent: {m_decision} {m_sym}{venue_sfx} ${m_usd:.0f} (conf={m_conf:.2f})")

    return title, "\n".join(body_lines), priority, target_topic, is_any_error


def persist_supervisor_audit_artifact(
    report_dict: Dict[str, Any],
    cache_dir: str,
    state_file: str,
    heartbeat_file: str,
    log: Optional[Any] = None,
) -> None:
    """Persist full SupervisorAuditReport dict and heartbeat JSON to disk in cachedb/."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cleaned_report = clean_json_value(report_dict)
        save_json_state(state_file, cleaned_report, indent=2)

        hb = clean_json_value({
            "status": "alive",
            "ts": report_dict.get("ts"),
            "mode": report_dict.get("mode"),
            "summary": report_dict.get("summary"),
            "diagnosed_issues_count": len(report_dict.get("diagnosed_issues") or []),
            "venue_actions_count": len(report_dict.get("venue_actions") or []),
            "market_trading_decision": report_dict.get("market_trading_decision") or {},
            "market_trading_decisions": report_dict.get("market_trading_decisions") or [],
        })
        save_json_state(heartbeat_file, hb, indent=2)
    except Exception as e:
        if log:
            log.warning("Could not persist audit artifact: %s", e)


def apply_reconciliation_plan(
    plan: Dict[str, Any],
    workspace_dir: str,
    *,
    is_enforce: bool,
    allow_code_remediation: bool = True,
    execute_action_fn: Optional[Any] = None,
    apply_patch_fn: Optional[Any] = None,
    log: Optional[Any] = None,
) -> Tuple[bool, Optional[bool]]:
    """Apply or simulate venue/fleet repair actions and code remediation based on operational mode."""
    if not isinstance(plan, dict):
        return False, None
    allow_remediation = _parse_bool_flag(allow_code_remediation, True)
    raw_actions = plan.get("venue_actions")
    venue_actions = raw_actions if isinstance(raw_actions, list) else []
    raw_fix = plan.get("code_remediation") if allow_remediation else None
    code_fix = raw_fix if isinstance(raw_fix, dict) else {}

    venue_executed = False
    for act in venue_actions:
        if not isinstance(act, dict):
            continue
        norm = normalize_repair_action(act, workspace_dir)
        action_type = norm["action"]
        if action_type == "NONE":
            continue
        if not is_enforce or not allow_remediation:
            if log:
                log.info(
                    "[SHADOW WOULD EXECUTE] Venue/fleet action %s for %s (%s): %s",
                    action_type,
                    norm["symbol"] or act.get("symbol"),
                    norm["venue"],
                    norm["reason"],
                )
        elif callable(execute_action_fn) and execute_action_fn(act):
            venue_executed = True

    tests_passed: Optional[bool] = None
    has_fix = _parse_bool_flag(code_fix.get("has_fix"), False)
    raw_target_file = code_fix.get("target_file")
    raw_search_block = code_fix.get("search_block")
    if (
        has_fix
        and isinstance(raw_target_file, str)
        and raw_target_file.strip()
        and isinstance(raw_search_block, str)
        and raw_search_block
    ):
        target_file = os.path.join(workspace_dir, raw_target_file.strip())
        raw_rationale = code_fix.get("rationale")
        rationale = (
            str(raw_rationale).strip()
            if raw_rationale is not None and not isinstance(raw_rationale, bool) and str(raw_rationale).strip()
            else "Autonomous self-heal patch"
        )
        raw_replace = code_fix.get("replace_block", "")
        replace_block = raw_replace if isinstance(raw_replace, str) else ""
        if not is_enforce:
            if log:
                log.info("[SHADOW WOULD APPLY] Code patch on %s: %s", target_file, rationale)
        elif callable(apply_patch_fn):
            if log:
                log.info("[ENFORCE ATTEMPTING PATCH] Applying self-healing patch on %s: %s", target_file, rationale)
            tests_passed = apply_patch_fn(
                target_file=target_file,
                search_block=raw_search_block,
                replace_block=replace_block,
                rationale=rationale,
            )

    executed = is_enforce and bool(venue_executed or (has_fix and tests_passed))
    return executed, tests_passed


def send_supervisor_ntfy_http(
    title: str,
    body: str,
    priority: str,
    target_topic: str,
    is_any_error: bool,
    *,
    ntfy_url: str = "",
    ntfy_token: str = "",
    urlopen_fn: Optional[Any] = None,
    log: Optional[Any] = None,
) -> bool:
    """Gate and send a supervisor ntfy push notification via HTTP POST."""
    try:
        from notify_engine.alertnotifiers import pass_common_gate
        allowed, reason, _, out_payload = pass_common_gate(
            "ntfy",
            title,
            body,
            priority,
            target_topic,
            [{"type": "bot_event", "name": title, "body": body, "source": "ai_supervisor", "symbol": "SYSTEM"}],
            urgent=is_any_error,
        )
        if not allowed:
            if log:
                log.info("Supervisor ntfy gated by common policy (%s): %s", reason, title)
            return reason == "buffered_for_bulk"
        if out_payload:
            title, body, priority, target_topic = out_payload
    except Exception as e:
        if log:
            log.debug("Common gate fallback in reconciler: %s", e)

    try:
        from notify_engine.alertnotifiers import AlertNotifier, format_ntfy_url
    except Exception:
        format_ntfy_url = None  # type: ignore[assignment]
        AlertNotifier = None  # type: ignore[assignment]

    url_raw = (ntfy_url or "").strip()
    if format_ntfy_url is not None:
        url = format_ntfy_url(url_raw or target_topic)
    else:
        url = url_raw if url_raw.startswith(("http://", "https://")) else f"https://ntfy.sh/{target_topic}"
    if not url:
        return False
    safe_title = (
        (AlertNotifier.utf8_header(title or "Alert") if AlertNotifier is not None else "")
        or str(title or "Alert").replace("\r", " ").replace("\n", " ").strip().encode("utf-8").decode("latin-1", "ignore")
        or "Alert"
    )
    headers = {
        "Title": safe_title,
        "Priority": priority,
        "Tags": "rotating_light,robot" if is_any_error else "shield,robot",
    }
    if ntfy_token:
        headers["Authorization"] = f"Bearer {ntfy_token}"

    _urlopen = urlopen_fn or urllib.request.urlopen
    try:
        req = urllib.request.Request(
            url,
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", None) or getattr(resp, "code", None)
            if status is None and hasattr(resp, "getcode") and callable(resp.getcode):
                status = resp.getcode()
            return status == 200
    except Exception as exc:
        if log:
            log.error("Failed sending standalone ntfy notification: %s", exc)
        return False


def default_supervisor_fallback_plan(spec: Any, logs: Any) -> Dict[str, Any]:
    """Return baseline fallback plan when Gemini query returns empty or fails."""
    if not getattr(spec, "can_self_heal", False):
        return {
            "summary": "Autonomous trade evaluation completed: holding posture.",
            "market_trading_decisions": [],
        }
    return {
        "summary": "Autonomous audit completed: telemetry sensors active and bots healthy.",
        "diagnosed_issues": [
            {
                "severity": "INFO",
                "category": "HEALTHY",
                "description": (
                    f"Audited {getattr(logs, 'total_scanned_files', 0)} log files. "
                    f"{getattr(logs, 'error_count', 0)} errors, {getattr(logs, 'warning_count', 0)} warnings."
                ),
                "root_cause": "System operational within expected tolerances.",
            }
        ],
        "venue_actions": [],
    }


def parse_supervisor_config(
    conf: Dict[str, str],
    *,
    mode: Optional[str] = None,
    notify_enabled: Optional[bool] = None,
) -> Tuple[str, bool, str, str, str, str, str]:
    """Validate and parse (mode, notify_enabled, ntfy_url, ntfy_topic_guard, ntfy_topic_server, ntfy_topic_error, ntfy_token)."""
    if mode is not None:
        if isinstance(mode, bool):
            raise ValueError(f"Invalid supervisor_mode {mode!r}; must be one of ['enforce', 'off', 'shadow']")
        raw_mode = mode
    else:
        if "supervisor_mode" not in conf:
            raise KeyError("Missing required 'supervisor_mode' in order_guard.conf")
        raw_mode = conf["supervisor_mode"]
    norm_mode = str(raw_mode).strip().lower()
    if norm_mode not in ("off", "shadow", "enforce"):
        raise ValueError(f"Invalid supervisor_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']")

    notify_flag = parse_notify_enabled_flag(notify_enabled, conf, "supervisor_notify")
    ntfy_url = conf.get("phone_alert_url", "")
    if "ntfy_topic_guard" not in conf:
        raise KeyError("Missing required 'ntfy_topic_guard' in config.env")
    ntfy_topic_guard = conf["ntfy_topic_guard"]
    ntfy_topic_server = conf.get("ntfy_topic_server") or ntfy_topic_guard
    ntfy_topic_error = conf.get("ntfy_topic_error") or ntfy_topic_server
    ntfy_token = conf.get("ntfy_token", "")
    return (
        norm_mode,
        notify_flag,
        ntfy_url,
        ntfy_topic_guard,
        ntfy_topic_server,
        ntfy_topic_error,
        ntfy_token,
    )


def harvest_fleet_and_system_telemetry(workspace_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect fleet process status via FleetHealthMonitor and Linux OS / VPN / DNS / config audit telemetry."""
    fleet_processes: List[Dict[str, Any]] = []
    base_res: Dict[str, Any] = {}
    try:
        from orchestratorOS.livecheck.fleet_health_monitor import FleetHealthMonitor

        monitor = FleetHealthMonitor(workspace_dir=workspace_dir)
        fleet_processes = [asdict(p) for p in monitor.check_processes()]
        base_res = dict(monitor.gather_system_resources() or {})
    except Exception:
        base_res = {}
    system_and_vpn = gather_system_and_vpn_telemetry(workspace_dir, base_resources=base_res)
    return fleet_processes, system_and_vpn


def submit_supervisor_trading_decisions(
    plan: Dict[str, Any],
    spec: Any,
    *,
    topic_key: str,
    mode: str,
    now_ts: float,
    intent_gateway: Any,
    is_rth_open_fn: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """Normalize and submit LLM trading decisions to AutonomousIntentGateway."""
    from providers.execution_audit import normalize_trading_decisions

    if not getattr(spec, "can_trade", False):
        return [], {}, [], None, False
    decisions_list = normalize_trading_decisions(plan)
    scope = getattr(spec, "asset_scope", "all")
    if scope in ("equities", "crypto") and decisions_list:
        want_t212 = scope == "equities"
        decisions_list = [
            d
            for d in decisions_list
            if str(d.get("decision") or "HOLD").upper() not in ("BUY", "SELL")
            or (
                normalize_venue_name(d.get("venue")) == "t212"
                or str(d.get("symbol") or "").strip().upper().endswith(("_EQ", "_US_EQ"))
            )
            == want_t212
        ]
    market_decision = decisions_list[0] if decisions_list else {}
    intents_emitted: List[Dict[str, Any]] = []
    actionable_decs = [
        d for d in decisions_list
        if str(d.get("decision") or "HOLD").upper() in ("BUY", "SELL")
    ]
    submit_decs = actionable_decs if actionable_decs else decisions_list[:1]
    scope = getattr(spec, "asset_scope", "all")
    effective_max_mode = "shadow" if (scope == "equities" and not is_rth_open_fn(now_ts)) else mode
    any_executed = False
    for dec in submit_decs:
        emitted = intent_gateway.submit(dec, source=topic_key, max_mode=effective_max_mode)
        if isinstance(emitted, dict):
            intents_emitted.append(emitted)
            if mode == "enforce" and emitted.get("status") == "EXECUTED":
                any_executed = True
    primary_intent = intents_emitted[0] if intents_emitted else None
    return decisions_list, market_decision, intents_emitted, primary_intent, any_executed


def build_sensor_health_summary(sensors: Any) -> Dict[str, Any]:
    """Build sensor_health dictionary for SupervisorAuditReport."""
    return {
        "fear_greed_present": getattr(sensors, "fear_greed", None) is not None,
        "market_breadth_present": getattr(sensors, "market_breadth", None) is not None,
        "sentiment_advisor_present": getattr(sensors, "sentiment_advisor", None) is not None,
        "geopolitical_threat_present": getattr(sensors, "geopolitical_threat", None) is not None,
        "prices_and_trends_count": len(getattr(sensors, "prices_and_trends", None) or {}),
        "active_heartbeats_count": len(getattr(sensors, "active_heartbeats", None) or {}),
    }


def evaluate_heartbeat_files_freshness(
    hb_paths: List[str],
    stale_threshold_s: float,
    now_ts: float,
    *,
    fallback_start_ts: Optional[float] = None,
) -> Tuple[bool, float, float, str]:
    """Evaluate heartbeat freshness across ``hb_paths``.

    Returns ``(is_fresh, max_existing_age_sec, max_stale_duration_sec, stale_or_missing_path)``.
    """
    if not hb_paths:
        return True, 0.0, 0.0, ""
    mtimes: List[float] = []
    missing_any = False
    stale_file = ""
    max_stale_duration = 0.0
    start_ts = fallback_start_ts if fallback_start_ts is not None else now_ts

    for raw_hbp in hb_paths:
        hbp = str(resolve_cachedb_filepath(raw_hbp))
        if os.path.exists(hbp):
            try:
                mt = os.path.getmtime(hbp)
                mtimes.append(mt)
                dur = now_ts - mt
                if stale_threshold_s > 0 and dur > stale_threshold_s and dur > max_stale_duration:
                    max_stale_duration = dur
                    stale_file = hbp
            except Exception:
                missing_any = True
        else:
            missing_any = True
            dur = max(0.0, now_ts - start_ts)
            if stale_threshold_s > 0 and dur >= max_stale_duration:
                max_stale_duration = dur
                stale_file = hbp

    age_sec = max(0.0, now_ts - min(mtimes)) if mtimes else 0.0
    fresh = True
    if stale_threshold_s > 0 and (missing_any or not mtimes or age_sec > stale_threshold_s):
        fresh = False
    return fresh, age_sec, max_stale_duration, stale_file


def summarize_fleet_sensor_status(
    cachedb_dir: str,
    read_json_fn: Any = None,
) -> Dict[str, Any]:
    """Load and summarize real-time risk guards and telemetry files for FleetHealthMonitor."""
    _rj = read_json_fn or read_json_recovering
    sensors: Dict[str, Any] = {}

    def _read(*fnames: str) -> Optional[Dict[str, Any]]:
        for fname in fnames:
            data = _rj(os.path.join(cachedb_dir, fname))
            if isinstance(data, dict) and data:
                return data
        return None

    geo = _read("geopolitical_eval.json")
    if geo:
        sensors["geopolitical_threat"] = {
            "threat_level": str(geo.get("threat_level") or "NORMAL"),
            "risk_score": safe_float(geo.get("risk_score"), default=0.0),
            "summary": str(geo.get("summary") or ""),
        }

    macro = _read("sentiment_eval.json", "sentiment_state.json")
    if macro:
        sensors["sentiment_advisor"] = {
            "market_bias": str(macro.get("market_bias") or "NEUTRAL"),
            "risk_level": str(macro.get("risk_level") or "MODERATE"),
            "action": str(macro.get("recommended_action") or "HOLD"),
        }

    sup = _read(
        "auto_cryptotrade_executor_eval.json",
        "auto_trade_executor_eval.json",
        "auto_sharetrade_executor_eval.json",
    )
    if sup:
        raw_dec = sup.get("market_trading_decision")
        dec = raw_dec if isinstance(raw_dec, dict) else {}
        sensors["supervisor_eval"] = {
            "mode": str(sup.get("mode") or "enforce"),
            "decision": str(dec.get("decision") or "HOLD"),
            "confidence": safe_float(dec.get("confidence"), default=0.0),
            "summary": str(sup.get("summary") or ""),
        }

    fg = _read("fear_greed_collect.json", "fear_greed_state.json")
    if fg:
        sensors["fear_greed"] = safe_float(fg.get("value"), default=None)

    mb = _read("market_breadth_collect.json", "market_breadth_state.json")
    if mb:
        sensors["market_breadth"] = safe_float(mb.get("advance_ratio"), default=None)

    return sensors




