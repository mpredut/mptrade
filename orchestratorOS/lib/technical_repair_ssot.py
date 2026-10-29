"""Single Source of Truth (SSOT) for Technical Repair (`auto_technical_repair`), Fleet Manifest, and System Telemetry.

Centralizes:
1. `procs.conf` process manifest parsing, heartbeat file resolution, and bot/module alias matching.
2. Canonical taxonomy of technical repair actions, system/fleet/VPN/DNS targets, and diagnosed issue categories.
3. Normalization of LLM repair actions (`normalize_repair_action`) across bots, modules, fleet, VPN/DNS, and system reboot.
4. Workspace configuration auditing (`audit_workspace_configs`) and patchable file boundary/secret validation (`resolve_patchable_target`).
5. Linux OS, memory, disk, internet default route, DNS resolvers, and WireGuard VPN (`wgpia0`) telemetry collection.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from botcore import _parse_bool_flag, safe_float as parse_finite_float

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
        return parse_finite_float(val, default=None)
    if isinstance(val, dict):
        return {str(k): clean_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json_value(v) for v in val]
    return val


def read_json_recovering(fpath: str) -> Optional[Any]:
    """Read JSON from ``fpath`` with automatic fallback to ``fpath.previous`` and thematic cachedb subdomains."""
    from cachepaths import resolve_cachedb_filepath

    resolved = str(resolve_cachedb_filepath(fpath))
    for cand in (resolved, f"{resolved}.previous", fpath, f"{fpath}.previous"):
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8", errors="replace") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


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
                from cachepaths import resolve_cachedb_filepath
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
                hb_val = parse_finite_float(hbstale, 0.0) or 0.0
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
    from cachepaths import normalize_venue_name
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
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 6:
                        issues.append(f"instruments.conf:{idx} has {len(parts)} fields (expected >=6)")
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
            if isinstance(v, dict) and (parse_finite_float(v.get("high"), 0.0) or 0.0) > 0
        ]
        if offenders:
            sys_vpn["resource_watchdog_offenders"] = offenders

    sys_vpn["config_audit"] = audit_workspace_configs(workspace_dir)
    return sys_vpn
