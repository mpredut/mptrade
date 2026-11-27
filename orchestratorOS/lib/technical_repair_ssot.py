"""Single Source of Truth (SSOT) for Technical Repair (`auto_technical_repair`), Fleet Manifest, and System Telemetry.

Centralizes:
1. `procs.conf` process manifest parsing, heartbeat file resolution, and bot/module alias matching.
2. Canonical taxonomy of technical repair actions, system/fleet/VPN/DNS targets, and diagnosed issue categories.
3. Normalization of LLM repair actions (`normalize_repair_action`) across bots, modules, fleet, VPN/DNS, and system reboot.
4. Workspace configuration auditing (`audit_workspace_configs`) and patchable file boundary/secret validation (`resolve_patchable_target`).
5. Linux OS, memory, disk, internet default route, DNS resolvers, and WireGuard VPN (`wgpia0`) telemetry collection.
"""
from __future__ import annotations

import configparser
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from botcore import (
    PRICE_FRESHNESS_MAX_AGE_SEC,
    TOPIC_ALIASES,
    TOPIC_SPECS,
    _parse_bool_flag,
    build_heartbeat_payload,
    clean_json_value,
    exact_nonneg_int,
    extract_order_id,
    first_epoch_sec,
    http_post_status,
    is_unmocked_llm_query_allowed,
    load_ini_config,
    normalize_epoch_sec,
    obj_attr,
    optional_env,
    parse_dotenv_line,
    parse_notify_enabled_flag,
    parse_procs_manifest,
    parse_symbol_list,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    summarize_current_price_cache_freshness,
    topic_spec,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    cache_path,
    is_readonly_xstock_symbol,
    matches_asset_scope,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from core.config import parse_dotenv
from core.llm_topics import resolve_configured_llm_models
from instrument_registry import canonical_display_symbol, is_xstock_trading_enabled
from state_stores.os_resources_watchdog_state import OsResourcesWatchdogStateStore
from intelligence.macro_sentiment_ssot import (
    MACRO_SENSOR_HEALTH_KEYS,
    normalize_market_bias,
    normalize_recommended_action,
    normalize_risk_level,
    normalize_threat_level,
    read_first_cached_dict,
)
from intelligence.sentiment.llm_quota_router import is_degraded_trading_model
import notify_engine.ntfyapi as _ntfyapi
import notify_engine.ntfypolicy as _ntfypolicy
from orchestratorOS.agy.paths import resolve_agy_bot_bin
from providers.intent_state import decision_has_cancel_directive, normalize_trading_decisions
from state_io import (
    atomic_write_text,
    generation_file_lock,
    read_json_dict,
    read_json_recovering,
    read_text_file,
    read_text_lines,
    save_json_state,
)

_default_read_json_recovering = read_json_recovering
_default_save_json_state = save_json_state


def _resolve_repair_json_reader(read_json_fn: Any = None) -> Any:
    if callable(read_json_fn) and read_json_fn is not _default_read_json_recovering:
        return read_json_fn
    return resolve_patched_attr(
        "read_json_recovering",
        read_json_fn if callable(read_json_fn) else read_json_recovering,
        local_val=read_json_recovering,
        defaults=(_default_read_json_recovering,),
        modules=("orchestratorOS.supervisor.autonomous_ai_reconciler",),
        require_callable=True,
    )


# ---------------------------------------------------------------------------
# 1. Canonical Taxonomy: Repair Actions, Targets & Issue Categories (SSOT)
# ---------------------------------------------------------------------------

BOT_MODULE_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_BOT", "RESTART_MODULE")
FLEET_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_FLEET",)
SYSTEM_INFRA_REPAIR_ACTIONS: Tuple[str, ...] = ("RESTART_VPN", "PIN_DNS", "REBOOT_SYSTEM")
VENUE_ORDER_REPAIR_ACTIONS: Tuple[str, ...] = ("CANCEL_ORDER", "SYNC_POSITION")

ALL_REPAIR_ACTIONS: Tuple[str, ...] = (
    *BOT_MODULE_REPAIR_ACTIONS, *FLEET_REPAIR_ACTIONS,
    *SYSTEM_INFRA_REPAIR_ACTIONS, *VENUE_ORDER_REPAIR_ACTIONS, "NONE",
)
BOT_OR_FLEET_ERROR_ACTIONS: FrozenSet[str] = frozenset({
    *BOT_MODULE_REPAIR_ACTIONS, *FLEET_REPAIR_ACTIONS, *VENUE_ORDER_REPAIR_ACTIONS,
})
SYSTEM_OS_ERROR_ACTIONS: FrozenSet[str] = frozenset(SYSTEM_INFRA_REPAIR_ACTIONS)

BOT_AND_ORDER_ERROR_CATEGORIES: FrozenSet[str] = frozenset({
    "BOT_EXCEPTION", "ORPHAN_ORDER", "FLEET_STALE", "CODE_BUG", "CONFIG_ERROR",
})
SYSTEM_OS_ERROR_CATEGORIES: FrozenSet[str] = frozenset({
    "SYSTEM_ERROR", "SERVER_ERROR", "OS_RESOURCE", "VPN_ERROR", "NETWORK_ERROR",
})
ALL_DIAGNOSED_CATEGORIES: Tuple[str, ...] = (
    "BOT_EXCEPTION", "FLEET_STALE", "SYSTEM_ERROR", "OS_RESOURCE", "VPN_ERROR",
    "NETWORK_ERROR", "CONFIG_ERROR", "CODE_BUG", "ORPHAN_ORDER", "GUARD_VETO", "HEALTHY",
)

DIAGNOSED_SEVERITIES: Tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
ACTIONABLE_ERROR_SEVERITIES: FrozenSet[str] = frozenset({"CRITICAL", "HIGH", "MEDIUM"})

AUDITED_CONFIG_FILES: Tuple[str, ...] = (
    "procs.conf", "order_guard.conf", "config.env", "instruments.conf", "macro_intelligence.conf",
)
PROTECTED_SECRET_FILES: FrozenSet[str] = frozenset({".env"})

TRADING_ADMIN_BIN: str = "/usr/local/sbin/trading-admin"

# Maps canonical system/fleet actions to (trading-admin subcommand, systemctl fallback command, log description)
SYSTEM_ADMIN_ACTION_SPECS: Dict[str, Tuple[str, List[str], str]] = {
    "RESTART_FLEET": ("restart", ["/bin/systemctl", "restart", "python_orchestrator.service"], "Restarting entire trading fleet + VPN"),
    "RESTART_VPN": ("restart", ["/bin/systemctl", "restart", "pia.service"], "Restarting VPN service (wgpia0) + fleet"),
    "PIN_DNS": ("pin-dns", ["/bin/systemctl", "restart", "systemd-resolved"], "Pinning DNS and restarting systemd-resolved"),
    "REBOOT_SYSTEM": ("reboot", ["/bin/systemctl", "reboot"], "Rebooting entire Linux system"),
}

ACTION_SYNONYM_MAP: Dict[str, str] = {
    "REBOOT": "REBOOT_SYSTEM", "SYSTEM_REBOOT": "REBOOT_SYSTEM", "REBOOT_HOST": "REBOOT_SYSTEM",
    "REBOOT_MACHINE": "REBOOT_SYSTEM", "REBOOT_SERVER": "REBOOT_SYSTEM",
    "REBOOT_FLEET": "RESTART_FLEET", "RESTART_ORCHESTRATOR": "RESTART_FLEET", "RESTART_ALL": "RESTART_FLEET",
    "REBOOT_VPN": "RESTART_VPN", "RESTART_PIA": "RESTART_VPN", "RECONNECT_VPN": "RESTART_VPN",
    "FIX_DNS": "PIN_DNS", "RESTART_DNS": "PIN_DNS", "RESET_DNS": "PIN_DNS",
    "REBOOT_BOT": "RESTART_BOT", "REBOOT_MODULE": "RESTART_MODULE",
    "RESTART_PROCESS": "RESTART_MODULE", "RESTART_SERVICE": "RESTART_MODULE",
}

FLEET_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "fleet", "all", "all_bots", "all_processes", "orchestrator", "python_orchestrator", "python_orchestrator.service",
})
VPN_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "vpn", "wgpia0", "pia", "pia.service", "piavpn", "piavpn.service", "pia-vpn", "pia-vpn.service", "wireguard", "tun0",
})
DNS_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "dns", "resolved", "systemd-resolved", "systemd-resolved.service", "resolv.conf", "pin-dns",
})
SYSTEM_REBOOT_TARGET_ALIASES: FrozenSet[str] = frozenset({
    "system", "host", "machine", "linux", "os", "server", "kernel", "full_system",
})


# ---------------------------------------------------------------------------
# 2. `procs.conf` Manifest & Process Target Resolution SSOT
# ---------------------------------------------------------------------------

def fleet_script_basenames(workspace_dir: str) -> Set[str]:
    """Return the set of Python script basenames for ``role=fleet`` processes in ``procs.conf``."""
    return {
        os.path.basename(raw_pat.split()[0])
        for p in parse_procs_manifest(workspace_dir, roles=("fleet",))
        if (raw_pat := str(p.get("clean_pattern") or "").strip())
    }


def resolve_manifest_process(target: Any, workspace_dir: str) -> Optional[Dict[str, Any]]:
    """Resolve a bot or module name/pattern/script stem against ``procs.conf``."""
    if not isinstance(target, str) or not target.strip():
        return None
    norm = target.strip().rstrip("$").lower()
    norm_base = os.path.basename(norm)
    norm_stem = norm_base[:-3] if norm_base.endswith(".py") else norm_base

    manifest = parse_procs_manifest(workspace_dir)
    for p in manifest:
        p_name, p_pat = p["name"].lower(), p["clean_pattern"].lower()
        p_base = os.path.basename(p_pat).lower()
        if norm in (p_name, p_pat, p_base) or norm_base in (p_name, p_base):
            return p
    return next((p for p in manifest if {norm, norm_base, norm_stem} & set(p.get("aliases") or [])), None)


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
    return (["sudo", "-n", admin_bin, admin_subcmd] if os.path.exists(admin_bin) else ["sudo", "-n", *systemctl_fallback], desc)


def _nonbool_str(val: Any) -> str:
    return "" if isinstance(val, bool) or val is None else str(val)


def normalize_repair_action(act: Any, workspace_dir: str) -> Dict[str, Any]:
    """Normalize a raw repair action dict into canonical action, venue, symbol, and resolved process metadata."""
    if not isinstance(act, dict):
        return {"action": "NONE", "venue": "fleet", "symbol": "", "order_id": "", "reason": "", "matched_process": None}

    action_type = _nonbool_str(act.get("action")).strip().upper()
    action_type = ACTION_SYNONYM_MAP.get(action_type, action_type) or "NONE"
    venue = normalize_venue_name(act.get("venue") if act.get("venue") is not None else act.get("provider"), default="binance")
    raw_sym = act.get("symbol") or act.get("bot") or act.get("module") or act.get("target") or act.get("service")
    symbol = _nonbool_str(raw_sym).strip()
    raw_oid = act.get("order_id") if act.get("order_id") is not None else (act.get("orderId") if act.get("orderId") is not None else act.get("oid"))
    order_id = extract_order_id(raw_oid, default="") or ""
    reason = _nonbool_str(act.get("reason"))
    sym_lower, venue_lower = symbol.lower(), venue.lower()

    if action_type in (*BOT_MODULE_REPAIR_ACTIONS, "RESTART_FLEET", "RESTART_VPN", "PIN_DNS", "REBOOT_SYSTEM"):
        if sym_lower in SYSTEM_REBOOT_TARGET_ALIASES and action_type in (*BOT_MODULE_REPAIR_ACTIONS, "REBOOT_SYSTEM"):
            action_type, venue, symbol = "REBOOT_SYSTEM", "system", symbol or "SYSTEM"
        elif action_type in BOT_MODULE_REPAIR_ACTIONS:
            if sym_lower in FLEET_TARGET_ALIASES:
                action_type, venue, symbol = "RESTART_FLEET", "fleet", symbol or "FLEET"
            elif sym_lower in VPN_TARGET_ALIASES or venue_lower in VPN_TARGET_ALIASES:
                action_type, venue, symbol = "RESTART_VPN", "system", symbol or "wgpia0"
            elif sym_lower in DNS_TARGET_ALIASES or venue_lower in DNS_TARGET_ALIASES:
                action_type, venue, symbol = "PIN_DNS", "system", symbol or "dns"

    matched_proc: Optional[Dict[str, Any]] = None
    if action_type in BOT_MODULE_REPAIR_ACTIONS and symbol:
        if (matched_proc := resolve_manifest_process(symbol, workspace_dir)) is not None:
            symbol, venue = matched_proc["name"], "fleet"

    return {
        "action": action_type, "venue": venue, "symbol": symbol,
        "order_id": order_id, "reason": reason, "matched_process": matched_proc,
    }


# ---------------------------------------------------------------------------
# 5. Workspace Config Audit & Patch Target Validation SSOT
# ---------------------------------------------------------------------------

def resolve_patchable_target(target_file: str, workspace_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate that ``target_file`` is an existing file inside ``workspace_dir`` and not a protected secret file."""
    if not isinstance(target_file, str) or not target_file.strip():
        return None, "empty target_file"
    real_ws = os.path.realpath(workspace_dir)
    real_target = os.path.realpath(target_file if os.path.isabs(target_file) else os.path.join(workspace_dir, target_file.strip()))
    try:
        in_workspace = os.path.commonpath([real_ws, real_target]) == real_ws
    except ValueError:
        in_workspace = False
    if not in_workspace:
        return None, f"Refusing autonomous patch outside workspace_dir: {target_file}"
    if not os.path.isfile(real_target):
        return None, f"Target patch file does not exist: {target_file}"
    if os.path.basename(real_target) in PROTECTED_SECRET_FILES:
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
            raw_lines = read_text_file(cfg_path, errors="replace").splitlines(keepends=True)
            if cfg_name == "procs.conf":
                for idx, raw_l in enumerate(raw_lines, start=1):
                    line = raw_l.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 7:
                        issues.append(f"procs.conf:{idx} has {len(parts)} fields (expected >=7)")
                    elif parts[6] not in ("bot", "fleet"):
                        issues.append(f"procs.conf:{idx} ({parts[3] or parts[0]}) has invalid role {parts[6]!r}")
            elif cfg_name in ("order_guard.conf", "config.env"):
                seen_keys: Dict[str, str] = {}
                for idx, raw_l in enumerate(raw_lines, start=1):
                    try:
                        parsed_kv = parse_dotenv_line(raw_l, strict=True)
                    except ValueError:
                        line = raw_l.split("#", 1)[0].strip()
                        if "=" not in line:
                            issues.append(f"{cfg_name}:{idx} malformed line missing '=': {line[:60]!r}")
                        else:
                            issues.append(f"{cfg_name}:{idx} empty key name")
                        continue
                    if parsed_kv is not None:
                        k_raw, v_clean = parsed_kv
                        seen_keys[k_raw.lower()] = v_clean
                if cfg_name == "order_guard.conf":
                    for mode_key in ("supervisor_mode", "llm_guard_mode", "autonomous_trading_mode"):
                        if (m_val := seen_keys.get(mode_key, "").lower()) and m_val not in ("enforce", "shadow", "off"):
                            issues.append(f"order_guard.conf invalid {mode_key}={m_val!r}")
            elif cfg_name == "instruments.conf":
                for idx, raw_l in enumerate(raw_lines, start=1):
                    if not (line := raw_l.split("#", 1)[0].strip()):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        if not line[1:-1].strip():
                            issues.append(f"instruments.conf:{idx} empty section header")
                    elif "|" in line:
                        if len(parts := [p.strip() for p in line.split("|")]) < 6:
                            issues.append(f"instruments.conf:{idx} has {len(parts)} fields (expected >=6)")
                    elif "=" in line:
                        if not line.split("=", 1)[0].strip():
                            issues.append(f"instruments.conf:{idx} empty key name")
                    else:
                        issues.append(f"instruments.conf:{idx} malformed line: {line[:60]!r}")
            elif cfg_name == "macro_intelligence.conf":
                try:
                    cp = load_ini_config(cfg_path, must_exist=True)
                    for req_sec in ("feeds", "cache_files", "intervals", "taxonomies", "sessions"):
                        if not cp.has_section(req_sec):
                            issues.append(f"macro_intelligence.conf missing required [{req_sec}] section")
                except (configparser.Error, OSError, ValueError) as cp_exc:
                    issues.append(f"macro_intelligence.conf parse error: {cp_exc}")
        except Exception as exc:
            issues.append(f"{cfg_name} read error: {exc}")
    return {"checked_files": checked, "issues": issues}


# ---------------------------------------------------------------------------
# 5B. Periodic LLM Model Catalog & Standalone Binary Version Audit (Notify-Only)
# ---------------------------------------------------------------------------

_MODEL_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,64}$")
_EFFORT_ORDER: Tuple[str, ...] = ("low", "medium", "high", "xhigh")


def parse_agy_models_output(raw_stdout: str) -> List[str]:
    """Extract ordered, deduplicated model slugs from ``agy models`` stdout (handling spinner CR frames)."""
    if not isinstance(raw_stdout, str) or not raw_stdout.strip():
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for raw_line in raw_stdout.splitlines():
        seg = raw_line.split("\r")[-1].strip()
        if not seg or seg.startswith(("Usage:", "Flags:", "Fetching", "-", "=")):
            continue
        tok = seg.split()[0].strip().lower()
        if "-" in tok and _MODEL_SLUG_RE.fullmatch(tok) and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def group_model_families(slugs: Iterable[str]) -> Dict[str, List[str]]:
    """Group model slugs by family base (stripping -xhigh/-high/-medium/-low), preserving discovery order."""
    families: Dict[str, List[str]] = {}
    for raw in slugs:
        slug = raw.strip().lower() if isinstance(raw, str) else ""
        if not slug:
            continue
        base = slug
        for suf in ("-xhigh", "-high", "-medium", "-low"):
            if slug.endswith(suf):
                base = slug[: -len(suf)]
                break
        families.setdefault(base, []).append(slug)
    return families


def format_model_family_summary(family: str, slugs: Iterable[str]) -> str:
    """Format a model family and its effort tiers compactly (e.g. ``gemini-5-nova (low/medium/high/xhigh)``)."""
    tiers: List[str] = []
    for raw in slugs:
        slug = raw.strip().lower() if isinstance(raw, str) else ""
        if slug.startswith(f"{family}-"):
            tier = slug[len(family) + 1 :]
            if tier in _EFFORT_ORDER and tier not in tiers:
                tiers.append(tier)
    if tiers:
        tiers.sort(key=lambda t: _EFFORT_ORDER.index(t))
        return f"{family} ({'/'.join(tiers)})"
    return family


def _probe_binary_version(cmd: List[str], *, runner: Any, timeout_sec: float = 3.0) -> str:
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        if getattr(proc, "returncode", 1) != 0:
            return ""
        raw = (getattr(proc, "stdout", "") or getattr(proc, "stderr", "") or "").strip()
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        return first[:64]
    except Exception:
        return ""


def audit_llm_model_catalog_and_binaries(
    workspace_dir: str,
    *,
    catalog_path: Optional[str] = None,
    runner: Optional[Any] = None,
    allow_subprocess: Optional[bool] = None,
    min_interval_sec: float = 43200.0,
    force_refresh: bool = False,
    persist: bool = True,
) -> Dict[str, Any]:
    """Read-only periodic check of ``agy models`` (vs config.env & baseline) and key binary versions.

    Notify-only by design: detects missing/removed configured models (critical CONFIG_ERROR) and newly
    available model families (one-time advisory) without mutating config.env or binaries.
    """
    can_run = (
        runner is not None
        or allow_subprocess is True
        or (allow_subprocess is None and is_unmocked_llm_query_allowed())
    )
    if not can_run:
        return {}

    cfg_path = os.path.join(workspace_dir, "config.env")
    configured_models = list(
        resolve_configured_llm_models(config_path=cfg_path if os.path.isfile(cfg_path) else None)
    )
    state_path = catalog_path or cache_path("llm_model_catalog.json")
    now = time.time()
    prev = read_json_dict(state_path, default={}) if state_path else {}
    if not isinstance(prev, dict):
        prev = {}
    prev_known = [
        str(m).strip().lower()
        for m in (prev.get("known_models") or [])
        if isinstance(m, str) and m.strip()
    ]
    prev_ts = normalize_epoch_sec(prev.get("checked_ts"), default=0.0) or 0.0
    if (
        not force_refresh
        and prev_known
        and prev_ts > 0.0
        and 0.0 <= (now - prev_ts) < max(60.0, safe_pos_float(min_interval_sec, 43200.0))
    ):
        avail_set = set(prev_known)
        missing_cached = [m for m in configured_models if m.lower() not in avail_set]
        rec_cached = (
            compute_recommended_llm_config_updates(
                prev_known,
                parse_dotenv(cfg_path) if os.path.isfile(cfg_path) else {},
            )
            if missing_cached
            else {}
        )
        return {
            "checked_ts": prev_ts,
            "agy_models_ok": True,
            "cached": True,
            "baseline_initialized": False,
            "binary_versions": dict(prev.get("binary_versions") or {}),
            "available_count": len(prev_known),
            "available_models": prev_known,
            "configured_models": configured_models,
            "missing_configured_models": missing_cached,
            "new_models": [],
            "new_families": [],
            "removed_models": [],
            "recommended_config_updates": rec_cached,
            "catalog_path": state_path,
        }

    run_fn = runner or subprocess.run
    bot_bin = ""
    try:
        cand = resolve_agy_bot_bin()
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            bot_bin = cand
    except Exception:
        bot_bin = ""
    agy_bin = bot_bin or shutil.which("agy-bot") or shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
    rclone_home = os.path.expanduser("~/bin/rclone")
    rclone_bin = rclone_home if os.path.isfile(rclone_home) and os.access(rclone_home, os.X_OK) else shutil.which("rclone")
    piactl_bin = shutil.which("piactl")

    binary_versions: Dict[str, str] = {}
    if runner is not None or (agy_bin and os.path.isfile(agy_bin) and os.access(agy_bin, os.X_OK)):
        if v := _probe_binary_version([agy_bin, "--version"], runner=run_fn):
            binary_versions["agy"] = v
    if runner is not None or rclone_bin:
        if v := _probe_binary_version([rclone_bin or "rclone", "--version"], runner=run_fn):
            binary_versions["rclone"] = v
    if runner is not None or piactl_bin:
        if v := _probe_binary_version([piactl_bin or "piactl", "--version"], runner=run_fn):
            binary_versions["piactl"] = v

    available_models: List[str] = []
    err_msg = ""
    if runner is not None or (agy_bin and os.path.isfile(agy_bin) and os.access(agy_bin, os.X_OK)):
        try:
            proc = run_fn([agy_bin, "models"], capture_output=True, text=True, timeout=15.0, check=False)
            if getattr(proc, "returncode", 1) == 0:
                available_models = parse_agy_models_output(getattr(proc, "stdout", "") or "")
            else:
                err_msg = (getattr(proc, "stderr", "") or f"rc={getattr(proc, 'returncode', 'err')}").strip()[:120]
        except Exception as exc:
            err_msg = str(exc)[:120]
    else:
        err_msg = "agy binary not found"

    if not available_models:
        return {
            "checked_ts": now,
            "agy_models_ok": False,
            "cached": False,
            "baseline_initialized": False,
            "binary_versions": binary_versions,
            "available_count": 0,
            "configured_models": configured_models,
            "missing_configured_models": [],
            "new_models": [],
            "new_families": [],
            "removed_models": [],
            "error": err_msg or "empty agy models output",
        }

    avail_set = set(available_models)
    missing_configured = [m for m in configured_models if m.lower() not in avail_set]
    curr_families = group_model_families(available_models)
    baseline_initialized = not prev_known
    new_family_bases: List[str] = []
    if baseline_initialized:
        new_models: List[str] = []
        new_families: List[str] = []
        removed_models: List[str] = []
    else:
        prev_set = set(prev_known)
        prev_families = set(group_model_families(prev_known).keys())
        new_models = [m for m in available_models if m not in prev_set]
        new_family_bases = [fam for fam in curr_families if fam not in prev_families]
        new_families = [
            format_model_family_summary(fam, curr_families[fam])
            for fam in new_family_bases
        ]
        removed_models = [m for m in prev_known if m not in avail_set]

    current_cfg_kv = parse_dotenv(cfg_path) if os.path.isfile(cfg_path) else {}
    recommended_updates = compute_recommended_llm_config_updates(
        available_models,
        current_cfg_kv,
        new_family_names=new_family_bases,
    )

    result: Dict[str, Any] = {
        "checked_ts": now,
        "agy_models_ok": True,
        "cached": False,
        "baseline_initialized": baseline_initialized,
        "binary_versions": binary_versions,
        "available_count": len(available_models),
        "available_models": available_models,
        "configured_models": configured_models,
        "missing_configured_models": missing_configured,
        "new_models": new_models,
        "new_families": new_families,
        "new_family_bases": new_family_bases,
        "removed_models": removed_models,
        "recommended_config_updates": recommended_updates,
        "catalog_path": state_path,
    }
    if persist and state_path and not recommended_updates:
        persist_llm_model_catalog_state(state_path, result)
    return result


_EXCLUDED_AUTO_MODEL_TOKENS: Tuple[str, ...] = ("orchestrator", "image", "lite", "gpt-oss", "tab_")


def _extract_effort_suffix(model_name: str) -> str:
    low = model_name.strip().lower() if isinstance(model_name, str) else ""
    for suf in ("-xhigh", "-high", "-medium", "-low"):
        if low.endswith(suf):
            return suf
    return ""


def _pick_family_tier(family: str, preferred_suffix: str, available_set: Set[str]) -> Optional[str]:
    order = (
        (preferred_suffix, "-high", "-medium", "-low")
        if preferred_suffix == "-xhigh"
        else (preferred_suffix, "-medium", "-high", "-low")
        if preferred_suffix
        else ("-medium", "-high", "-low")
    )
    for suf in order:
        cand = f"{family}{suf}"
        if cand in available_set:
            return cand
    return None


def compute_recommended_llm_config_updates(
    available_models: Sequence[str],
    current_config: Mapping[str, str],
    *,
    new_family_names: Iterable[str] = (),
) -> Dict[str, str]:
    """Compute minimal config.env LLM_* updates strictly when configured models vanish from ``agy models``.

    Never silently migrates away from a valid configured model family merely because a new family
    (e.g. a weaker lightweight or slower flagship tier) appeared in the catalog -- new families stay
    notify-only advisories. When a configured model is retired, preserves each key's chosen effort tier
    (-low/-medium/-high/-xhigh), updates configured ``LLM_TOPIC_MODEL_*`` overrides, and preserves
    the user's fallback slot composition without re-injecting removed ``-low`` or Flash models.
    """
    avail = [m.strip().lower() for m in available_models if isinstance(m, str) and m.strip()]
    if not avail:
        return {}
    avail_set = set(avail)
    families = group_model_families(avail)
    new_fams = {str(f).strip().lower() for f in new_family_names if isinstance(f, str) and f.strip()}

    def _eligible(fam: str) -> bool:
        return not any(tok in fam for tok in _EXCLUDED_AUTO_MODEL_TOKENS)

    paid_gemini_fams = [
        fam
        for fam, slugs in families.items()
        if fam.startswith("gemini-")
        and "flash" not in fam
        and _eligible(fam)
        and {f"{fam}-low", f"{fam}-medium", f"{fam}-high"}.issubset(set(slugs))
    ]
    flash_fams = [
        fam
        for fam, slugs in families.items()
        if fam.startswith("gemini-")
        and "flash" in fam
        and _eligible(fam)
        and {f"{fam}-low", f"{fam}-medium"}.issubset(set(slugs))
    ]
    claude_sonnet_fams = [
        fam
        for fam, slugs in families.items()
        if fam.startswith("claude-sonnet-")
        and _eligible(fam)
        and {f"{fam}-low", f"{fam}-medium"}.issubset(set(slugs))
    ]

    cur_default = str(current_config.get("LLM_DEFAULT_MODEL") or "").strip().lower()
    cur_high = str(current_config.get("LLM_HIGH_STAKE_MODEL") or "").strip().lower()
    cur_low = str(current_config.get("LLM_LOW_STAKE_MODEL") or "").strip().lower()
    cur_compact = str(current_config.get("LLM_COMPACT_MODEL") or "").strip().lower()
    cur_fallbacks = [
        m.strip().lower()
        for m in str(current_config.get("LLM_FALLBACK_MODELS") or "").split(",")
        if m.strip()
    ]
    topic_override_keys = [
        spec.override_env_key
        for spec in TOPIC_SPECS.values()
        if spec.override_env_key and str(current_config.get(spec.override_env_key) or "").strip()
    ]
    cur_overrides = {
        k: str(current_config.get(k) or "").strip().lower()
        for k in topic_override_keys
    }

    def _fam_of(slug: str) -> str:
        suf = _extract_effort_suffix(slug)
        return slug[: -len(suf)] if suf else slug

    def _is_paid_gemini(slug: str) -> bool:
        return slug.startswith("gemini-") and "flash" not in slug

    cur_primary_fam = _fam_of(cur_default)
    cur_flash_fam = _fam_of(cur_compact)
    cur_claude_fams = [_fam_of(m) for m in cur_fallbacks if m.startswith("claude-")]
    cur_claude_fam = cur_claude_fams[0] if cur_claude_fams else ""

    primary_missing = (
        any(m and _is_paid_gemini(m) and m not in avail_set for m in (cur_default, cur_high, cur_low))
        or any(_is_paid_gemini(m) and m not in avail_set for m in cur_fallbacks)
        or any(_is_paid_gemini(m) and m not in avail_set for m in cur_overrides.values())
    )
    target_primary_fam = cur_primary_fam
    if primary_missing and paid_gemini_fams:
        newest_new_primary = next((f for f in paid_gemini_fams if f in new_fams), None)
        target_primary_fam = newest_new_primary or paid_gemini_fams[0]

    flash_missing = (
        bool(cur_compact and cur_compact not in avail_set)
        or bool(cur_low and "flash" in cur_low and cur_low not in avail_set)
        or any("flash" in m and m not in avail_set for m in cur_fallbacks)
        or any("flash" in m and m not in avail_set for m in cur_overrides.values())
    )
    target_flash_fam = cur_flash_fam
    if flash_missing and flash_fams:
        newest_new_flash = next((f for f in flash_fams if f in new_fams), None)
        target_flash_fam = newest_new_flash or flash_fams[0]

    claude_missing = (
        any(m.startswith("claude-") and m not in avail_set for m in cur_fallbacks)
        or any(m.startswith("claude-") and m not in avail_set for m in cur_overrides.values())
    )
    target_claude_fam = cur_claude_fam
    if claude_missing and claude_sonnet_fams:
        newest_new_claude = next((f for f in claude_sonnet_fams if f in new_fams), None)
        target_claude_fam = newest_new_claude or claude_sonnet_fams[0]

    def _remap_slug(slug: str, default_suf: str) -> Optional[str]:
        if not slug:
            return None
        if slug in avail_set:
            return slug
        suf = _extract_effort_suffix(slug) or default_suf
        if "flash" in slug and target_flash_fam:
            return _pick_family_tier(target_flash_fam, suf, avail_set)
        if slug.startswith("claude-") and target_claude_fam:
            return _pick_family_tier(target_claude_fam, suf, avail_set)
        if _is_paid_gemini(slug) and target_primary_fam:
            return _pick_family_tier(target_primary_fam, suf, avail_set)
        return None

    updates: Dict[str, str] = {}
    if primary_missing and target_primary_fam:
        if cur_default and (new_def := _pick_family_tier(target_primary_fam, _extract_effort_suffix(cur_default) or "-medium", avail_set)):
            if new_def != cur_default:
                updates["LLM_DEFAULT_MODEL"] = new_def
        if cur_high and (new_hi := _pick_family_tier(target_primary_fam, _extract_effort_suffix(cur_high) or "-high", avail_set)):
            if new_hi != cur_high:
                updates["LLM_HIGH_STAKE_MODEL"] = new_hi
        if cur_low and not ("flash" in cur_low):
            if new_lo := _pick_family_tier(target_primary_fam, _extract_effort_suffix(cur_low) or "-low", avail_set):
                if new_lo != cur_low:
                    updates["LLM_LOW_STAKE_MODEL"] = new_lo

    if flash_missing and target_flash_fam:
        if cur_low and "flash" in cur_low:
            if new_lo := _pick_family_tier(target_flash_fam, _extract_effort_suffix(cur_low) or "-low", avail_set):
                if new_lo != cur_low:
                    updates["LLM_LOW_STAKE_MODEL"] = new_lo
        if cur_compact:
            if new_cmp := _pick_family_tier(target_flash_fam, _extract_effort_suffix(cur_compact) or "-low", avail_set):
                if new_cmp != cur_compact:
                    updates["LLM_COMPACT_MODEL"] = new_cmp

    for ov_key, cur_ov in cur_overrides.items():
        if cur_ov not in avail_set:
            if (new_ov := _remap_slug(cur_ov, "-high")) and new_ov != cur_ov:
                updates[ov_key] = new_ov

    if cur_fallbacks and any(m not in avail_set for m in cur_fallbacks):
        fb_candidates: List[str] = []
        for m in cur_fallbacks:
            remapped = _remap_slug(m, "-medium")
            if remapped and remapped in avail_set and remapped not in fb_candidates:
                fb_candidates.append(remapped)
        new_fb_str = ",".join(fb_candidates)
        if new_fb_str and new_fb_str != ",".join(cur_fallbacks):
            updates["LLM_FALLBACK_MODELS"] = new_fb_str

    return updates


def build_llm_config_env_remediation(
    workspace_dir: str,
    updates: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a unique, comment-preserving ``code_remediation`` patch for ``config.env`` LLM model keys."""
    if not isinstance(updates, Mapping) or not updates:
        return None
    cfg_path = os.path.join(workspace_dir, "config.env")
    if not os.path.isfile(cfg_path):
        return None
    raw_lines = read_text_file(cfg_path, errors="replace").splitlines(keepends=True)
    matched_indices: List[int] = []
    updated_lines = list(raw_lines)
    applied_keys: List[str] = []
    for idx, line in enumerate(raw_lines):
        parsed = parse_dotenv_line(line)
        if parsed is None:
            continue
        key, _ = parsed
        if key not in updates:
            continue
        new_val = str(updates[key]).strip()
        if not new_val:
            continue
        eol = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
        body = line[: -len(eol)] if eol else line
        eq_pos = body.find("=")
        if eq_pos < 0:
            continue
        prefix = body[: eq_pos + 1]
        rhs = body[eq_pos + 1 :]
        comment_idx = rhs.find("#")
        if comment_idx >= 0:
            val_part = rhs[:comment_idx]
            pad_len = len(val_part) - len(val_part.rstrip(" \t"))
            sep = val_part[-pad_len:] if pad_len > 0 else " "
            comment_suffix = f"{sep}{rhs[comment_idx:]}"
        else:
            comment_suffix = ""
        new_line = f"{prefix}{new_val}{comment_suffix}{eol}"
        if new_line != line:
            updated_lines[idx] = new_line
            matched_indices.append(idx)
            applied_keys.append(f"{key}={new_val}")
    if not matched_indices:
        return None
    lo, hi = min(matched_indices), max(matched_indices)
    search_block = "".join(raw_lines[lo : hi + 1])
    replace_block = "".join(updated_lines[lo : hi + 1])
    if not search_block or search_block == replace_block:
        return None
    return {
        "has_fix": True,
        "target_file": "config.env",
        "rationale": f"Auto-update config.env LLM models from agy catalog ({', '.join(applied_keys[:4])})",
        "search_block": search_block,
        "replace_block": replace_block,
    }


def persist_llm_model_catalog_state(state_path: str, catalog_audit: Mapping[str, Any]) -> None:
    """Atomically save the latest known ``agy models`` baseline snapshot to ``cachedb/system/``."""
    if not state_path or not isinstance(catalog_audit, Mapping):
        return
    available_models = [
        str(m).strip().lower()
        for m in (catalog_audit.get("available_models") or [])
        if isinstance(m, str) and m.strip()
    ]
    if not available_models:
        return
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with generation_file_lock(state_path):
            now_cat = time.time()
            raw_cat_ts = normalize_epoch_sec(catalog_audit.get("checked_ts"), default=None)
            eff_cat_ts = raw_cat_ts if (raw_cat_ts is not None and raw_cat_ts <= now_cat + 60.0) else now_cat
            save_json_state(
                state_path,
                {
                    "checked_ts": eff_cat_ts,
                    "binary_versions": dict(catalog_audit.get("binary_versions") or {}),
                    "known_models": available_models,
                    "known_families": list(group_model_families(available_models).keys()),
                    "configured_models": list(catalog_audit.get("configured_models") or []),
                    "missing_configured_models": list(catalog_audit.get("missing_configured_models") or []),
                },
                indent=2,
            )
    except Exception:
        pass


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
            mem_kv = {
                parts[0].strip(): exact_nonneg_int(v_parts[0], default=0)
                for mline in read_text_lines("/proc/meminfo", errors="replace")
                if len(parts := mline.split(":")) == 2 and (v_parts := parts[1].strip().split()) and v_parts[0].isdigit()
            }
            if (total_kb := mem_kv.get("MemTotal", 0)) > 0 and "MemAvailable" in mem_kv:
                avail_kb = mem_kv["MemAvailable"]
                sys_vpn["mem_used_pct"] = round(((total_kb - avail_kb) / total_kb) * 100.0, 1)
                sys_vpn["mem_avail_mb"] = round(avail_kb / 1024.0, 0)
    except Exception:
        pass

    vpn_if = optional_env("PIA_VPN_IF", "wgpia0") or "wgpia0"
    vpn_net_dir = f"/sys/class/net/{vpn_if}"
    oper = str(read_text_file(f"{vpn_net_dir}/operstate", default="unknown") or "unknown").strip().lower() if os.path.exists(vpn_net_dir) else "down"
    sys_vpn["vpn_interface"] = vpn_if
    sys_vpn["vpn_tunnel_up"] = os.path.exists(vpn_net_dir) and (oper or "unknown") in ("up", "unknown")

    try:
        sys_vpn["default_route_present"] = bool(
            os.path.exists("/proc/net/route")
            and any(len(rcols := rline.split()) >= 2 and rcols[1] == "00000000" for rline in read_text_lines("/proc/net/route", strip=True, errors="replace")[1:])
        )
    except Exception:
        pass

    try:
        if os.path.exists("/etc/resolv.conf"):
            dns_servers = [
                dparts[1]
                for raw_dline in read_text_lines("/etc/resolv.conf", errors="replace")
                if (dline := raw_dline.split("#", 1)[0].strip()).startswith("nameserver") and len(dparts := dline.split()) >= 2
            ]
            if dns_servers:
                sys_vpn["dns_servers"] = dns_servers
    except Exception:
        pass

    sys_vpn["vpn_outage_active"] = os.path.exists("/var/lib/pia_selfheal/outage_since")
    sys_vpn["internet_healthy"] = bool(
        _parse_bool_flag(sys_vpn.get("default_route_present", True), True)
        and not _parse_bool_flag(sys_vpn.get("vpn_outage_active", False), False)
    )

    rw_path, rw_state, _ = OsResourcesWatchdogStateStore.inspect_workspace(
        workspace_dir,
        read_json_fn=_resolve_repair_json_reader(),
    )
    if isinstance(rw_state, dict) and isinstance(rw_state.get("tracked"), dict):
        now_rw = time.time()
        has_explicit_rw_ts = rw_state.get("last_run") is not None
        rw_last_run = normalize_epoch_sec(rw_state.get("last_run"), default=None)
        if rw_last_run is None and not has_explicit_rw_ts and rw_path is not None and os.path.exists(rw_path):
            try:
                rw_last_run = os.path.getmtime(rw_path)
            except OSError:
                rw_last_run = None
        if has_explicit_rw_ts and rw_last_run is None:
            rw_fresh = False
        elif rw_last_run is None:
            rw_fresh = True
        else:
            rw_fresh = -60.0 <= (now_rw - rw_last_run) <= 3600.0
        min_consec = max(2, safe_pos_int(optional_env("RESOURCE_CONSECUTIVE_CHECKS", ""), default=3) or 3)
        procs_map = rw_state.get("processes") if isinstance(rw_state.get("processes"), dict) else {}
        offenders = [
            {
                "key": k,
                **({"command": cmd_s} if isinstance(p_info := procs_map.get(k), dict) and (cmd_s := str(p_info.get("command") or "").strip()) else {}),
                "reasons": v.get("last_reasons"),
                "consecutive_high": v.get("high"),
            }
            for k, v in rw_state["tracked"].items()
            if rw_fresh
            and isinstance(v, dict)
            and (safe_float(v.get("high"), default=0.0) or 0.0) >= 1.0
            and (
                (safe_float(v.get("high"), default=0.0) or 0.0) >= min_consec
                or _parse_bool_flag(v.get("alerted"), False)
            )
        ]
        if offenders:
            sys_vpn["resource_watchdog_offenders"] = offenders

    sys_vpn["config_audit"] = audit_workspace_configs(workspace_dir)
    if llm_cat := audit_llm_model_catalog_and_binaries(workspace_dir):
        sys_vpn["llm_catalog_audit"] = llm_cat
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
    day, prev_ts = ref.date(), fm * 1.0
    picked: List[str] = []

    def _resolve_fold_ts(dt_naive: datetime, upper_ts: float) -> float:
        ts0, ts1 = dt_naive.replace(fold=0).timestamp(), dt_naive.replace(fold=1).timestamp()
        return ts1 if (ts1 > ts0 and ts1 <= upper_ts + 300.0) else ts0

    pending_unstamped: List[str] = []
    for line in reversed(lines):
        if not isinstance(line, str):
            continue
        ts: Optional[float] = None
        has_stamp_token = False
        if m := _FULL_TS_RE.search(line):
            has_stamp_token = True
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")
                day, ts = dt.date(), (prev_ts := _resolve_fold_ts(dt, prev_ts))
            except ValueError:
                pass
        elif g := _GLOG_TS_RE.match(line):
            has_stamp_token = True
            try:
                mm, dd, hh, mi, ss = (exact_nonneg_int(g.group(i), default=0) for i in range(1, 6))
                dt = datetime(ref.year, mm, dd, hh, mi, ss)
                if (dt - ref).total_seconds() > 86400:
                    dt = datetime(ref.year - 1, mm, dd, hh, mi, ss)
                day, ts = dt.date(), (prev_ts := _resolve_fold_ts(dt, prev_ts))
            except ValueError:
                pass
        elif (
            (c := _CLOCK_TS_RE.match(line))
            and (c_hh := exact_nonneg_int(c.group(1), default=24)) < 24
            and (c_mm := exact_nonneg_int(c.group(2), default=60)) < 60
            and (c_ss := exact_nonneg_int(c.group(3), default=60)) < 60
        ):
            has_stamp_token = True
            secs = c_hh * 3600 + c_mm * 60 + c_ss
            ts_same = _resolve_fold_ts(datetime.combine(day, datetime.min.time()) + timedelta(seconds=secs), prev_ts)
            if ts_same <= prev_ts + 300.0:
                ts = ts_same
            else:
                day -= timedelta(days=1)
                ts = _resolve_fold_ts(datetime.combine(day, datetime.min.time()) + timedelta(seconds=secs), prev_ts)
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


_SIG_SRC_LOC_RE = re.compile(r"\[(?:[A-Za-z0-9_.-]+:\d+|unknown)\]\s*", re.IGNORECASE)
_SIG_STDIO_RE = re.compile(r"\[(?:STDERR|STDOUT)\]\s*", re.IGNORECASE)
_SIG_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_SIG_NUM_RE = re.compile(r"\b\d{4,}\b")


def normalize_error_signature(line: str) -> str:
    """Strip timestamps, PIDs, and hex addresses so recurring log errors group into one signature."""
    s = _CLOCK_TS_RE.sub("", _GLOG_TS_RE.sub("", _FULL_TS_RE.sub("", line)))
    s = _SIG_NUM_RE.sub("NUM", _SIG_HEX_RE.sub("0xADDR", _SIG_STDIO_RE.sub("", _SIG_SRC_LOC_RE.sub("", s))))
    return " ".join(s.strip().split())[:180]


_LOG_ERR_RE = re.compile(
    r"(\bERROR\b|\bCRITICAL\b(?!_SHOCK)|Traceback|fail-closed|BUY BLOCKED|\bException\b|"
    r"(?:^|\s)[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Timeout)\s*:)",
    re.IGNORECASE,
)
_LOG_PY_EXC_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.]*\.)?[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Interrupt|Timeout)\s*:"
)
_LOG_TB_BOILERPLATE_RE = re.compile(
    r"^(?:The above exception was the direct cause of the following exception:|"
    r"During handling of the above exception, another exception occurred:|"
    r"File \".*\", line (?:NUM|\d+))",
    re.IGNORECASE,
)
_LOG_BENIGN_WARN_RE = re.compile(r"(\[CMCPlatform\]\s+Warning|\[WARNING\])", re.IGNORECASE)
_LOG_SERIOUS_ERR_RE = re.compile(r"(Traceback|CRITICAL(?!_SHOCK)|fail-closed|BUY BLOCKED)", re.IGNORECASE)
_LOG_STDIO_INDENT_RE = re.compile(r"\[(?:STDERR|STDOUT)\]\s{4,}", re.IGNORECASE)
_LOG_T212_SYM_RE = re.compile(r"\b([A-Z0-9]+_(?:US_)?EQ)\b")
_LOG_SUPERVISOR_SUMMARY_RE = re.compile(
    r"(Autonomous AI Supervisor cycle complete:|\[(ENFORCE|SHADOW)\]\s+auto_|"
    r"\[SHADOW WOULD (?:EXECUTE|APPLY)\]|\[ENFORCE (?:EXECUTING|ATTEMPTING PATCH)\]|"
    r"\"__orchestrator_intent__\"\s*:|"
    r"mailer:\s+Email delivery skipped|Email gated by common delivery policy|"
    r"(?:Supervisor\s+)?ntfy(?: delivery)? gated (?:by common policy|after due-batch flush)|Dispatched ntfy alert to|"
    r"Retry queue full .* dropping oldest notification|"
    r"<!--\s*a padding to disable MSIE|<(?:head|center)>.*500 Internal Server Error|"
    r"Exception ignored in:.*BaseSubprocessTransport\.__del__|RuntimeError:\s+Event loop is closed|"
    r"curl:\s*\(22\).*returned error:\s*429|deadman:\s*(?:curl ERROR after retries|ntfy rate-limited)|"
    r"In-place /compact for topic .* failed:|"
    r"the historical price reference(?: and quantity/weight guard)? (?:was|were) explicitly bypassed\b|"
    r"BLOCKED \(pre-submit\):\s*(?:replace_not_attempted|replace_cancel_only)\b|"
    r"\[<stdin>:\d+\])",
    re.IGNORECASE,
)
_LOG_GUARD_RE = re.compile(
    r"(\[[^\[\]\r\n]*GUARD[^\[\]\r\n]*\]|vetoed|downscaled|HARD_VETO|DOWNSCALE_QTY)", re.IGNORECASE
)
_LOG_GUARD_PERMIT_RE = re.compile(
    r"(\bPermitted\.|\bprice\s+\S+\s+permitted\b|\bfrom older period bypassed\b|"
    r"\breference is off for this venue/module\b|\bbrake=(?:BrakeAction\.)?ALLOW\b)",
    re.IGNORECASE,
)
_LOG_GUARD_DIFF_THRESHOLD_RE = re.compile(
    r"\bdiff\s+([-+]?\d+(?:\.\d+)?)%,\s*threshold\s+([-+]?\d+(?:\.\d+)?)%",
    re.IGNORECASE,
)


def _is_permitted_profit_guard_line(line: str) -> bool:
    """Return True when an order_guard diff/threshold check actually passed (diff >= threshold)."""
    if "BLOCKED" in line.upper():
        return False
    m = _LOG_GUARD_DIFF_THRESHOLD_RE.search(line)
    if not m:
        return False
    diff_v = safe_float(m.group(1), default=None)
    thr_v = safe_float(m.group(2), default=None)
    return diff_v is not None and thr_v is not None and diff_v >= thr_v
_LOG_GUARD_EURO_RE = re.compile(r"(\b(?:BUY|SELL)\s+[A-Z0-9_]*?(?:USD|USDC|USDT|_EQ)\s+)\u20ac(?=\d)")
_LOG_GUARD_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?%?(?:[hsm]\b)?")
_LOG_DATED_FNAME_RE = re.compile(r"^(.+?)_\d{4}-\d{2}-\d{2}(\.(?:log|txt))$", re.IGNORECASE)
_LOG_TRANSIENT_IP_AUTH_RE = re.compile(r"Invalid API-key,\s*IP,\s*or permissions for action", re.IGNORECASE)
_LOG_BOT_CYCLE_RECOVERED_RE = re.compile(
    r"(Order \d+ .* is protected from monitororder repricing|"
    r"For [A-Z0-9_]+ there are no open (?:BUY|SELL) orders to begin with\.|"
    r"get_trade_orders:\s+Total found \d+ orders|"
    r"\[[A-Z0-9]+\]\s+Existing SELL order blocks AssetGuardian tier:|"
    r"\[[A-Z0-9_]+\]\s+price=\d+(?:\.\d+)?,\s+window_min/max=|"
    r"\[[A-Z0-9_]+\]\s+Current price:\s+\d+(?:\.\d+)?|"
    r"apply_weight_limit|Started\.\s+check_interval=|"
    r"\[Cache(?:CurrentPrice|24Price|365Price|Order|Trade)Manager\]\[info\]\s+(?:New order fills:|Save cache to file|[A-Z0-9_]+:\s+Added \d+ new items)|"
    r"\[TRAIL\]\s+[A-Z0-9]+:\s+\d+(?:\.\d+)?\s+(?:peak|warming up)|\[TRAIL-K\]\s+heartbeat|"
    r"\[FORWARD \(from the anchor\)\]|\[STRAT\]\s+price=\d+(?:\.\d+)?|"
    r"===\s*xStock watcher started\s*===|price \d+(?:\.\d+)? vs allocation \d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOG_DAEMON_RECOVERED_RE = re.compile(
    r"(Daemon status:\s*active|Remote-control daemon started|Daemon is active and healthy|ui-sync:\s*check healthy|"
    r"===\s*(?:STRATEGIE\s+.*?\s+PORNITA|(?:[A-Z0-9_]+\s+)?STRATEGY\s+STARTED)\s*===|===\s*TRAILING STOP(?:\s+KRAKEN)?\s+started|\[STRAT\]\s+state loaded|"
    r"\[kraken_cache\]\s+start:|\[kraken_cache\]\[ws\]\s+subscribed ownTrades|"
    r"\[Pricefetcher\]\s+Checking symbol support:|\[price_?watcher\]|\[resources\]\s+OK\b|"
    r"\[verify\].*is LISTED and tradable|total size is [\d,]+\s+speedup is|"
    r"Starting GlobalTelemetryCollector|Global telemetry cycle active:|"
    r"Starting MacroAnalyzer|Macro analyzer cycle active:|"
    r"Starting Autonomous AI Supervisor cycle|\[INTENT GATEWAY\]|"
    r"\[TRADEALL\]\s+[A-Z0-9_]+\s+resolved trend:|"
    r"OK \(tun0 \+ HTTPS through the tunnel\)|PIA healthy \(tun0 \+ DNS \+ HTTPS\)|\[order_retry\]\s+(?:start|ACCEPTED\b|FILLED\b|\{'attempted':)|\.env loaded from .*hyperliquid)",
    re.IGNORECASE,
)
_LOG_STARTUP_FATAL_EXC_RE = re.compile(
    r"(?:^(?:ImportError|ModuleNotFoundError|SyntaxError|IndentationError|NameError|UnboundLocalError):|"
    r"ValueError:\s+Unknown cache type:|TypeError:.*got an unexpected keyword argument|"
    r"TypeError:\s+Can't instantiate abstract class\b|error:\s+the following arguments are required:|"
    r"\bname '[^']+' is not defined\b|\bcannot access local variable '[^']+' where it is not associated with a value\b)",
    re.IGNORECASE,
)
_LOG_IGNORED_TEST_PREFIXES = (
    ".",
    "pytest_",
    "__main___",
    "py.test_",
    "c_",
    "app_",
    "inspect_",
    "dump_crypto_prompt_",
    "deep_audit_crypto_prompt_",
    "check_crypto_prompt_quality_",
    "check_cache_coherence_",
    "audit_prompt_invariants_",
    "account_cache_coherence_",
    "audit_live_prompts_",
    "audit_cycle_",
    "post_fix_audit_",
    "run_15m_audit_cycle_",
    "probe_order_guards_",
    "ai_attribution_report_",
    "pre_commit_gate_",
    "chart_view_",
)
_ACCOUNT_CACHE_GUARD_TOKENS = (
    "account_cache_not_fresh", "account_cache_initializing",
    "account cache is not fresh", "account_cache_writer_not_running",
)
_ACCOUNT_CACHE_ERR_TOKENS = (*_ACCOUNT_CACHE_GUARD_TOKENS, "ERROR reading AssetValue cache via cacheManager")
_RECOVERABLE_TRANSIENT_ERR_TOKENS = (
    "Failed to recover daemon", "Error: not signed in:", "Error: failed to start service:",
    "antigravity-cli-daemon.service failed", "ui_sync_idle_sec", "[manage_agy] ERROR:",
    "[ws] error: ping/pong timed out", "Connection to remote host was lost", "WS error: ",
    "Temporary failure in name resolution", "NameResolutionError", "Network is unreachable",
    "Connection timed out", "Read timed out", "The read operation timed out", "ConnectTimeoutError",
    "cycle failed (", "EGeneral:Internal error", "[shadow_live] error:", "rsync error:",
    "[rsync_cachedb] ERROR:", "agy CLI with model", "authentication failed or timed out",
    "! network error ", "network error/empty response", "T212 order history: timeout/network error",
    "FAILURE: every rung exhausted, the VPN is still down",
)


def _is_log_line_recovered(line: str) -> bool:
    return bool(_LOG_BOT_CYCLE_RECOVERED_RE.search(line) or _LOG_DAEMON_RECOVERED_RE.search(line))


def _has_subsequent_recovery(lines: List[str], start_idx: int, *, allow_weight_limit: bool = False) -> bool:
    return any(
        (allow_weight_limit and "apply_weight_limit" in lines[j]) or _is_log_line_recovered(lines[j])
        for j in range(start_idx, len(lines))
    )


def _upsert_capped_event(events: List[str], sig_to_idx: Dict[str, int], sig: str, text: str) -> None:
    if sig in sig_to_idx:
        events[sig_to_idx[sig]] = text
    elif len(events) < 50:
        sig_to_idx[sig] = len(events)
        events.append(text)


def harvest_log_anomalies_from_dirs(logs_dirs: List[str], lookback_hours: float) -> LogEventSummary:
    """Parse recent log files across ``logs_dirs`` for errors, persistent unresolved exceptions, and guard vetoes."""
    now_harvest = time.time()
    cutoff_ts = now_harvest - (lookback_hours * 3600.0)
    critical_events: List[str] = []
    guard_interventions: List[str] = []
    persistent_errors: List[Dict[str, Any]] = []
    error_count = warning_count = scanned_count = 0
    crit_sig_to_idx: Dict[str, int] = {}
    guard_sig_to_idx: Dict[str, int] = {}
    persistent_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_count_by_key: Dict[Tuple[str, str], int] = {}

    for l_dir in logs_dirs:
        if not os.path.isdir(l_dir):
            continue
        try:
            for fname in os.listdir(l_dir):
                if not (fname.endswith(".log") or fname.endswith(".txt")):
                    continue
                if fname.lower().startswith(_LOG_IGNORED_TEST_PREFIXES):
                    continue
                fpath = os.path.join(l_dir, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    if mtime < cutoff_ts:
                        continue
                    eff_mtime = now_harvest if mtime > now_harvest + 60.0 else mtime
                    scanned_count += 1
                    lines = recent_log_lines(
                        read_text_lines(fpath, keepends=True, errors="replace")[-1000:],
                        eff_mtime,
                        cutoff_ts,
                    )
                    total_lines = len(lines)
                    tail_start_idx = max(0, total_lines - min(10, max(3, total_lines // 4)))
                    last_ip_err_idx = max(
                        (i for i, ln in enumerate(lines) if _LOG_TRANSIENT_IP_AUTH_RE.search(ln)),
                        default=-1,
                    )
                    ip_auth_burst_recovered = (
                        last_ip_err_idx >= 0
                        and (total_lines - 1 - last_ip_err_idx) >= 10
                        and _has_subsequent_recovery(lines, last_ip_err_idx + 1)
                    )
                    file_sig_stats: Dict[str, Dict[str, Any]] = {}
                    last_exc_sig: str = ""
                    last_exc_idx: int = -100
                    for idx, line in enumerate(lines):
                        if _LOG_SUPERVISOR_SUMMARY_RE.search(line):
                            continue
                        if (
                            ("Exception ignored in:" in line or "Traceback (most recent call last):" in line)
                            and any(
                                "BaseSubprocessTransport.__del__" in lines[j]
                                or "<module 'threading'" in lines[j]
                                or "SystemExit: 0" in lines[j]
                                for j in range(max(0, idx - 1), min(total_lines, idx + 8))
                            )
                        ):
                            continue
                        if _LOG_ERR_RE.search(line) and not (
                            _LOG_BENIGN_WARN_RE.search(line) and not _LOG_SERIOUS_ERR_RE.search(line)
                        ):
                            cleaned = line.strip()
                            norm_cleaned = normalize_error_signature(cleaned)
                            if (
                                not cleaned
                                or _LOG_TB_BOILERPLATE_RE.match(cleaned)
                                or _LOG_TB_BOILERPLATE_RE.match(norm_cleaned)
                            ):
                                continue
                            if (
                                (line.startswith("    ") or bool(_LOG_STDIO_INDENT_RE.search(line)))
                                and idx > 0
                                and (
                                    lines[idx - 1].lstrip().startswith('File "')
                                    or normalize_error_signature(lines[idx - 1]).startswith('File "')
                                )
                            ):
                                continue
                            if norm_cleaned == "Traceback (most recent call last):" and any(
                                _LOG_PY_EXC_LINE_RE.match(n_ln := normalize_error_signature(lines[j]))
                                or _LOG_TB_BOILERPLATE_RE.match(n_ln)
                                for j in range(idx + 1, min(total_lines, idx + 45))
                            ):
                                continue
                            if not (_FULL_TS_RE.search(line) or _GLOG_TS_RE.search(line) or _CLOCK_TS_RE.search(line)):
                                if any(
                                    cleaned in lines[j] and _LOG_ERR_RE.search(lines[j])
                                    for j in range(idx + 1, min(total_lines, idx + 3))
                                ):
                                    continue
                            if "insufficient-free-for-stocks-buy" in cleaned:
                                m_t212_sym = _LOG_T212_SYM_RE.search(cleaned)
                                t212_sym_tok = m_t212_sym.group(1) if m_t212_sym else ""
                                lookahead_end = total_lines if t212_sym_tok else min(total_lines, idx + 15)
                                if any(
                                    "Successfully placed BUY" in lines[j]
                                    and (not t212_sym_tok or t212_sym_tok in lines[j])
                                    for j in range(idx + 1, lookahead_end)
                                ):
                                    continue
                            if ip_auth_burst_recovered and _LOG_TRANSIENT_IP_AUTH_RE.search(cleaned):
                                continue
                            if any(tok in cleaned for tok in _ACCOUNT_CACHE_ERR_TOKENS) and _has_subsequent_recovery(
                                lines, idx + 1, allow_weight_limit=True
                            ):
                                continue
                            if any(tok in cleaned for tok in _RECOVERABLE_TRANSIENT_ERR_TOKENS) and _has_subsequent_recovery(
                                lines, idx + 1
                            ):
                                continue
                            sig = norm_cleaned
                            if (
                                sig
                                and sig == last_exc_sig
                                and 0 < (idx - last_exc_idx) <= 25
                                and any(
                                    "During handling of the above exception" in lines[j]
                                    or "The above exception was the direct cause" in lines[j]
                                    for j in range(last_exc_idx + 1, idx)
                                )
                            ):
                                last_exc_idx = idx
                                if idx >= tail_start_idx and sig in file_sig_stats:
                                    file_sig_stats[sig]["unresolved_at_tail"] = True
                                continue
                            last_exc_sig, last_exc_idx = sig, idx
                            error_count += 1
                            _upsert_capped_event(critical_events, crit_sig_to_idx, sig or cleaned, cleaned)
                            if sig:
                                st = file_sig_stats.setdefault(
                                    sig,
                                    {"file": fname, "count": 0, "unresolved_at_tail": False, "sample": cleaned},
                                )
                                st["count"] += 1
                                st["sample"] = cleaned
                                if idx >= tail_start_idx:
                                    st["unresolved_at_tail"] = True
                        elif _LOG_GUARD_RE.search(line):
                            if _LOG_GUARD_PERMIT_RE.search(line) or _is_permitted_profit_guard_line(line):
                                continue
                            if any(tok in line for tok in _ACCOUNT_CACHE_GUARD_TOKENS) and _has_subsequent_recovery(
                                lines, idx + 1, allow_weight_limit=True
                            ):
                                continue
                            warning_count += 1
                            cleaned = _LOG_GUARD_EURO_RE.sub(r"\1$", line.strip())
                            g_sig = _LOG_GUARD_NUM_RE.sub("NUM", normalize_error_signature(cleaned)) or cleaned
                            _upsert_capped_event(guard_interventions, guard_sig_to_idx, g_sig, cleaned)
                        elif _is_log_line_recovered(line):
                            for sig_k, st_prev in list(file_sig_stats.items()):
                                st_prev["unresolved_at_tail"] = False
                                sample_prev = str(st_prev.get("sample") or "").strip()
                                if _LOG_STARTUP_FATAL_EXC_RE.search(sample_prev) or _LOG_STARTUP_FATAL_EXC_RE.search(
                                    normalize_error_signature(sample_prev)
                                ):
                                    error_count = max(0, error_count - safe_nonneg_int(st_prev.get("count"), default=0))
                                    file_sig_stats.pop(sig_k, None)
                                    if not any(k[1] == sig_k and v > 0 for k, v in seen_count_by_key.items()):
                                        c_sig_prev = sig_k or sample_prev
                                        removed_ev = False
                                        if c_sig_prev in crit_sig_to_idx:
                                            del_idx = crit_sig_to_idx[c_sig_prev]
                                            if 0 <= del_idx < len(critical_events):
                                                critical_events.pop(del_idx)
                                            removed_ev = True
                                        elif sample_prev in critical_events:
                                            critical_events.remove(sample_prev)
                                            removed_ev = True
                                        if removed_ev:
                                            crit_sig_to_idx = {
                                                (normalize_error_signature(ev) or ev): i
                                                for i, ev in enumerate(critical_events)
                                            }
                    m_dated = _LOG_DATED_FNAME_RE.match(fname)
                    canon_fname = f"{m_dated.group(1)}{m_dated.group(2)}" if m_dated else fname
                    is_dated_fname = m_dated is not None
                    for sig_k, st in file_sig_stats.items():
                        pkey = (canon_fname.lower(), sig_k)
                        cur_cnt = safe_nonneg_int(st.get("count"), default=0)
                        prev_seen_cnt = seen_count_by_key.get(pkey, 0)
                        if prev_seen_cnt > 0:
                            error_count = max(0, error_count - min(prev_seen_cnt, cur_cnt))
                        seen_count_by_key[pkey] = max(prev_seen_cnt, cur_cnt)
                        if cur_cnt >= 2 or st["unresolved_at_tail"]:
                            prev_st = persistent_by_key.get(pkey)
                            if prev_st is None:
                                persistent_by_key[pkey] = st
                                persistent_errors.append(st)
                            else:
                                if cur_cnt > safe_nonneg_int(prev_st.get("count"), default=0):
                                    prev_st["count"] = st["count"]
                                    prev_st["sample"] = st["sample"]
                                if not is_dated_fname:
                                    prev_st["file"] = fname
                                prev_st["unresolved_at_tail"] = bool(
                                    prev_st.get("unresolved_at_tail") or st.get("unresolved_at_tail")
                                )
                except Exception:
                    pass
        except Exception:
            pass

    persistent_errors.sort(
        key=lambda item: (1 if item.get("unresolved_at_tail") else 0, safe_nonneg_int(item.get("count"), default=0)),
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
    if any(
        isinstance(issue, dict)
        and str(issue.get("severity") or "").strip().upper() in ACTIONABLE_ERROR_SEVERITIES
        and (cat := str(issue.get("category") or "").strip().upper()) not in ("", "HEALTHY", "GUARD_VETO")
        and cat in BOT_AND_ORDER_ERROR_CATEGORIES
        for issue in (obj_attr(report, "diagnosed_issues") or [])
    ):
        return True
    code_rem = obj_attr(report, "code_remediation")
    if (isinstance(code_rem, dict) and _parse_bool_flag(code_rem.get("has_fix"), False)) or obj_attr(report, "tests_passed") is False:
        return True
    return any(
        isinstance(act, dict) and normalize_repair_action(act, "").get("action") in BOT_OR_FLEET_ERROR_ACTIONS
        for act in (obj_attr(report, "venue_actions") or [])
    )


def is_system_os_error_report(report: Any) -> bool:
    """Return True for Linux OS, internet/VPN, memory, disk space, or server infrastructure anomalies."""
    if any(
        isinstance(issue, dict)
        and str(issue.get("severity") or "").strip().upper() in ACTIONABLE_ERROR_SEVERITIES
        and (cat := str(issue.get("category") or "").strip().upper()) not in ("", "HEALTHY", "GUARD_VETO")
        and (cat in SYSTEM_OS_ERROR_CATEGORIES or cat not in BOT_AND_ORDER_ERROR_CATEGORIES)
        for issue in (obj_attr(report, "diagnosed_issues") or [])
    ):
        return True
    return any(
        isinstance(act, dict) and normalize_repair_action(act, "").get("action") in SYSTEM_OS_ERROR_ACTIONS
        for act in (obj_attr(report, "venue_actions") or [])
    )


def select_patch_pytest_targets(real_target: str, workspace_dir: str) -> List[str]:
    """Select targeted pytest files covering ``real_target`` so live-VM patch verification completes quickly."""
    tests_dir = os.path.join(workspace_dir, "tests")
    if not os.path.isdir(tests_dir):
        return ["tests/"]
    try:
        test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py") and os.path.isfile(os.path.join(tests_dir, f))]
    except OSError:
        return ["tests/"]
    if not test_files:
        return ["tests/"]

    base_name = os.path.basename(real_target)
    stem = os.path.splitext(base_name)[0].strip().lower()
    candidates: List[Tuple[int, int, str]] = []
    for fname in test_files:
        fpath = os.path.join(tests_dir, fname)
        try:
            fsize = os.path.getsize(fpath)
        except OSError:
            fsize = 999_999
        if stem and len(stem) >= 3 and stem in fname.lower():
            candidates.append((0, fsize, f"tests/{fname}"))
            continue
        try:
            txt = read_text_file(fpath)
        except OSError:
            continue
        if (base_name and base_name in txt) or (stem and len(stem) >= 4 and stem in txt):
            candidates.append((1, fsize, f"tests/{fname}"))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[2] for item in candidates[:5]]

    return [f"tests/{fname}" for fname in ("test_runtime_configuration_policy.py", "test_config_env_shell_safe.py") if fname in test_files] or ["tests/"]


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
    if real_target is None or not (
        isinstance(search_block, str) and isinstance(replace_block, str) and search_block and search_block != replace_block
    ):
        if log:
            if real_target is None:
                log.warning("%s", err_msg)
            else:
                log.warning("Refusing empty or no-op autonomous patch on %s.", target_file)
        return False

    content: Optional[str] = None
    try:
        content = read_text_file(real_target)
        match_count = content.count(search_block)
        if match_count != 1:
            if log:
                if match_count == 0:
                    log.warning("Search block not found in %s. Skipping patch.", target_file)
                else:
                    log.warning(
                        "Search block matched %d locations in %s (must be unique). Skipping ambiguous patch.",
                        match_count, target_file,
                    )
            return False

        atomic_write_text(real_target, content.replace(search_block, replace_block, 1))
        pytest_bin = next((c_abs for c in (".venv/bin/pytest", "myenv/bin/pytest") if os.path.exists(c_abs := os.path.join(workspace_dir, c))), "pytest")
        res = subprocess.run([pytest_bin, *select_patch_pytest_targets(real_target, workspace_dir)], cwd=workspace_dir, capture_output=True, text=True, timeout=240)

        if res.returncode == 0:
            if log:
                log.info("Self-healing patch verified with 100%% green tests. Committing to git.")
            for git_cmd in (["git", "add", real_target], ["git", "commit", "-m", f"[AI-SUPERVISOR] Autonomous self-heal: {rationale}"], ["git", "push", "origin", "main"]):
                subprocess.run(git_cmd, cwd=workspace_dir, check=False)
            return True
        if log:
            log.warning("Self-healing patch broke tests (exit code %d). Performing immediate rollback.", res.returncode)
    except Exception as e:
        if log:
            log.error("Error during safe code patching: %s. Performing rollback.", e)
    rolled_back = False
    if content is not None:
        try:
            atomic_write_text(real_target, content)
            rolled_back = True
        except OSError:
            pass
    if not rolled_back:
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
    action_type, venue, symbol, order_id, reason, matched = (
        norm["action"], norm["venue"], norm["symbol"], norm["order_id"], norm["reason"], norm["matched_process"]
    )

    if action_type in BOT_MODULE_REPAIR_ACTIONS:
        if not symbol or matched is None:
            if log:
                log.warning(
                    "[ENFORCE] %s missing bot/module name in action: %s" if not symbol else "[ENFORCE] Refusing %s for unknown process %r (not in procs.conf)",
                    action_type, act if not symbol else symbol,
                )
            return False
        try:
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
            if (built := build_system_admin_command(action_type)) is None:
                return False
            cmd, desc = built
            if log:
                log.warning("[ENFORCE EXECUTING] %s: %s", desc, reason)
            return getattr(subprocess.run(cmd, check=False), "returncode", 0) == 0
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
            eff_sym = canonical_display_symbol(symbol, venue=venue) or symbol
            if (inst := resolve_instrument_fn(eff_sym, venue=venue)) is None:
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
                log.error("[ENFORCE] Failed %s on %s (%s): %s", f"CANCEL_ORDER {order_id}" if is_cancel else "SYNC_POSITION", symbol, venue, e)
            return False

    if log:
        log.warning("[ENFORCE] Unsupported venue/fleet action %r: %s", action_type, act)
    return False


_CORE_HEARTBEAT_NAMES = ("macro_analyzer.heartbeat", "globaltelemetry_collector.heartbeat", "rtrade.heartbeat")


def _resolve_heartbeat_entry(
    hb_path: str,
    data: Any,
    now_ts: float,
    max_age: float,
    *,
    allow_mtime_fallback: bool = True,
    prefer_mtime_when_stale: bool = False,
) -> Optional[Dict[str, Any]]:
    has_dict = isinstance(data, dict) and bool(data)
    has_explicit_ts = has_dict and (
        data.get("ts") is not None or data.get("timestamp") is not None
    )
    hb_ts = first_epoch_sec(data.get("ts"), data.get("timestamp"), allow_iso=True) if has_dict else None
    valid_hb_ts = hb_ts if (hb_ts is not None and 0.0 < hb_ts <= now_ts + 60.0) else None
    can_use_mtime = allow_mtime_fallback or (has_dict and not has_explicit_ts)
    mtime: Optional[float] = None
    if can_use_mtime and os.path.exists(hb_path):
        try:
            mtime = os.path.getmtime(hb_path)
        except OSError:
            mtime = None
    valid_mtime = mtime if (mtime is not None and 0.0 < mtime <= now_ts + 60.0) else None
    if has_dict:
        if prefer_mtime_when_stale and valid_mtime is not None and (valid_hb_ts is None or valid_hb_ts < valid_mtime - 60.0):
            age_sec = max(0.0, now_ts - valid_mtime)
        elif valid_hb_ts is not None:
            age_sec = max(0.0, now_ts - valid_hb_ts)
        elif valid_mtime is not None:
            age_sec = max(0.0, now_ts - valid_mtime)
        else:
            return None
        return {**data, "age_seconds": round(age_sec, 1)} if age_sec <= max_age else None
    if allow_mtime_fallback and valid_mtime is not None:
        age_sec = max(0.0, now_ts - valid_mtime)
        if age_sec <= max_age:
            return build_heartbeat_payload({"ts": valid_mtime}, extra={"age_seconds": round(age_sec, 1)})
    return None


def harvest_heartbeats_and_state_files(
    cache_dir: str,
    workspace_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Harvest active heartbeat JSON/mtime files and watchdog state file ages."""
    _rj = _resolve_repair_json_reader(read_json_fn)
    _is_default_rj = _rj is _default_read_json_recovering
    manifest_stale_by_hb: Dict[str, float] = {}
    if _is_default_rj and workspace_dir and os.path.exists(os.path.join(workspace_dir, "procs.conf")):
        for proc in parse_procs_manifest(workspace_dir):
            if (hb_s := safe_pos_float(proc.get("hb_stale_s"), default=None)) is not None:
                for hp in proc.get("hb_paths") or ():
                    if isinstance(hp, str) and hp.endswith(".heartbeat"):
                        manifest_stale_by_hb[os.path.basename(hp)] = hb_s
                if p_name := str(proc.get("name") or "").strip():
                    manifest_stale_by_hb.setdefault(f"{p_name}.heartbeat", hb_s)
    active_heartbeats: Dict[str, Any] = {}
    for hb_name in _CORE_HEARTBEAT_NAMES:
        hb_raw_path = os.path.join(cache_dir, hb_name)
        data = _rj(hb_raw_path)
        is_rtrade = hb_name == "rtrade.heartbeat"
        if not _is_default_rj:
            if isinstance(data, dict) and data:
                active_heartbeats[hb_name] = data
            continue
        resolved_entry = _resolve_heartbeat_entry(
            str(resolve_cachedb_filepath(hb_raw_path)),
            data,
            now_ts,
            manifest_stale_by_hb.get(hb_name, 300.0 if is_rtrade else 1800.0),
            allow_mtime_fallback=is_rtrade,
        )
        if resolved_entry is not None:
            active_heartbeats[hb_name] = resolved_entry
    if _is_default_rj:
        for scan_dir in (os.path.join(cache_dir, "system", "heartbeat"), cache_dir):
            if not os.path.isdir(scan_dir):
                continue
            try:
                for entry in sorted(os.listdir(scan_dir)):
                    if not entry.endswith(".heartbeat") or entry in active_heartbeats or entry in _CORE_HEARTBEAT_NAMES:
                        continue
                    hb_full = os.path.join(scan_dir, entry)
                    if not os.path.isfile(hb_full):
                        continue
                    resolved_entry = _resolve_heartbeat_entry(
                        hb_full, _rj(hb_full), now_ts, manifest_stale_by_hb.get(entry, 1800.0),
                        allow_mtime_fallback=True, prefer_mtime_when_stale=True,
                    )
                    if resolved_entry is not None:
                        active_heartbeats[entry] = resolved_entry
            except OSError:
                pass
    state_files: Dict[str, Any] = {}
    rw_found, _, rw_age_h = OsResourcesWatchdogStateStore.inspect_workspace(
        workspace_dir,
        now_ts=now_ts,
    )
    if rw_found is not None and rw_age_h is not None:
        state_files[OsResourcesWatchdogStateStore.FILENAME] = {"age_hours": round(rw_age_h, 4)}
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
    non_healthy_issues = [
        i
        for i in (obj_attr(report, "diagnosed_issues") or [])
        if isinstance(i, dict)
        and str(i.get("category") or "").strip().upper() not in ("", "HEALTHY", "GUARD_VETO")
        and str(i.get("severity") or "").strip().upper() != "INFO"
    ]
    issues_cnt = len(non_healthy_issues)
    actions_cnt = sum(1 for a in (obj_attr(report, "venue_actions") or []) if isinstance(a, dict) and str(a.get("action") or "NONE").strip().upper() != "NONE")
    code_rem = obj_attr(report, "code_remediation")
    has_fix = bool(isinstance(code_rem, dict) and _parse_bool_flag(code_rem.get("has_fix", False), False))
    actions_cnt += 1 if has_fix else 0
    mode_label = str(obj_attr(report, "mode", "shadow")).upper()
    is_bot_error = is_bot_or_code_error_report(report) if is_bot_error is None else is_bot_error
    is_sys_error = is_system_os_error_report(report) if is_sys_error is None else is_sys_error
    is_any_error = bool(is_bot_error or is_sys_error)

    spec = topic_spec(topic_key) if (topic_key in TOPIC_SPECS or topic_key in TOPIC_ALIASES) else topic_spec("auto_technical_repair")
    if is_bot_error:
        title, target_topic, priority = f"\U0001f6a8 {spec.bracket_tag} ERROR Issues: {issues_cnt} | Actions: {actions_cnt}", ntfy_topic_error or ntfy_topic or "ntfy-error-1978", "high"
    elif is_sys_error:
        title, target_topic, priority = f"\U0001f6a8 {spec.bracket_tag} SYSTEM Issues: {issues_cnt} | Actions: {actions_cnt}", ntfy_topic_server or ntfy_topic or "ntfy-server-1978", "high"
    else:
        title, target_topic, priority = f"{spec.ntfy_prefix} Issues: {issues_cnt} | Actions: {actions_cnt}", ntfy_topic_server or ntfy_topic or "ntfy-server-1978", "default"

    body_lines: List[str] = []
    if actions_cnt > 0 or obj_attr(report, "actions_executed", False):
        status_label = "Executed" if obj_attr(report, "actions_executed", False) else "Simulated/Observed"
        body_lines.append(f"Mode: {mode_label} \u00b7 Status: {status_label}")
    if summary_str := str(obj_attr(report, "summary", "") or "").strip():
        body_lines.append(summary_str)
    if non_healthy_issues:
        top_issue = non_healthy_issues[0]
        body_lines.append(f"Primary Issue: [{top_issue.get('severity', 'INFO')}] {top_issue.get('description', '')}")
    if has_fix and isinstance(code_rem, dict):
        body_lines.append(f"Code Patch Proposed: {code_rem.get('target_file', 'unknown')} (tests={obj_attr(report, 'tests_passed')})")
    sensor_health = obj_attr(report, "sensor_health")
    if isinstance(sensor_health, dict):
        if new_fams := [str(x).strip() for x in (sensor_health.get("new_llm_model_families") or []) if str(x).strip()]:
            body_lines.append(f"New agy CLI models detected: {', '.join(new_fams)}")

    decisions_to_show = obj_attr(report, "market_trading_decisions") or ([m_single] if (m_single := obj_attr(report, "market_trading_decision")) else [])
    for m_dec in decisions_to_show:
        if isinstance(m_dec, dict) and (m_decision := str(m_dec.get("decision", "HOLD")).upper()) in ("BUY", "SELL"):
            m_sym, m_venue = m_dec.get("symbol", ""), m_dec.get("venue", "")
            m_conf = safe_nonneg_float(m_dec.get("confidence"), default=0.0) or 0.0
            m_usd = safe_nonneg_float(m_dec.get("suggested_notional_usd"), default=0.0) or 0.0
            body_lines.append(f"Trading Intent: {m_decision} {m_sym}{f' ({m_venue})' if m_venue else ''} ${m_usd:.0f} (conf={m_conf:.2f})")

    return title, "\n".join(body_lines), priority, target_topic, is_any_error


def persist_supervisor_audit_artifact(
    report_dict: Dict[str, Any],
    cache_dir: str,
    state_file: str,
    heartbeat_file: str,
    log: Optional[Any] = None,
    *,
    persist_eval: bool = True,
) -> None:
    """Persist full SupervisorAuditReport dict and heartbeat JSON to disk in cachedb/.

    When ``persist_eval=False`` (used by Delta-Gate no-op skips), only ``heartbeat_file`` is
    refreshed so ``state_file`` retains the exact ``ts``/``completed_ts``/``mtime`` and price baseline
    of the last genuine LLM evaluation (preventing AI_IMPULSE_MIN_EVAL_GAP_SEC impulse-trigger suppression).
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        _save_json = resolve_patched_attr(
            "save_json_state",
            save_json_state,
            local_val=save_json_state,
            defaults=(_default_save_json_state,),
            modules=("orchestratorOS.supervisor.autonomous_ai_reconciler",),
            require_callable=True,
        )
        write_eval = _parse_bool_flag(persist_eval, True)
        if write_eval:
            _save_json(state_file, clean_json_value(report_dict), indent=2)
        hb_extra: Dict[str, Any] = {
            "mode": report_dict.get("mode"),
            "summary": report_dict.get("summary"),
            "diagnosed_issues_count": len(report_dict.get("diagnosed_issues") or []),
            "venue_actions_count": len(report_dict.get("venue_actions") or []),
            "market_trading_decision": report_dict.get("market_trading_decision") or {},
            "market_trading_decisions": report_dict.get("market_trading_decisions") or [],
        }
        if model_used := str(report_dict.get("model_used") or "").strip():
            hb_extra["model_used"] = model_used
        if not write_eval:
            hb_extra["delta_gate_skipped"] = True
        hb = build_heartbeat_payload(
            {"ts": report_dict.get("ts")},
            extra=hb_extra,
        )
        _save_json(heartbeat_file, hb, indent=2)
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
    venue_actions = plan.get("venue_actions") if isinstance(plan.get("venue_actions"), list) else []
    code_fix = plan.get("code_remediation") if (allow_remediation and isinstance(plan.get("code_remediation"), dict)) else {}

    venue_executed = False
    for act in venue_actions:
        if not isinstance(act, dict):
            continue
        norm = normalize_repair_action(act, workspace_dir)
        if (action_type := norm["action"]) == "NONE":
            continue
        if not is_enforce or not allow_remediation:
            if log:
                log.info("[SHADOW WOULD EXECUTE] Venue/fleet action %s for %s (%s): %s", action_type, norm["symbol"] or act.get("symbol"), norm["venue"], norm["reason"])
        elif callable(execute_action_fn) and execute_action_fn(act):
            venue_executed = True

    tests_passed: Optional[bool] = None
    has_fix = _parse_bool_flag(code_fix.get("has_fix"), False)
    raw_target_file, raw_search_block, raw_replace_block = (
        code_fix.get("target_file"),
        code_fix.get("search_block"),
        code_fix.get("replace_block"),
    )
    if (
        has_fix
        and isinstance(raw_target_file, str)
        and raw_target_file.strip()
        and isinstance(raw_search_block, str)
        and raw_search_block
        and isinstance(raw_replace_block, str)
    ):
        target_file = raw_target_file.strip()
        full_target_file = os.path.join(workspace_dir, target_file)
        rationale = _nonbool_str(code_fix.get("rationale")).strip() or "Autonomous self-heal patch"
        replace_block = raw_replace_block
        if not is_enforce:
            if log:
                log.info("[SHADOW WOULD APPLY] Code patch on %s: %s", full_target_file, rationale)
        elif callable(apply_patch_fn):
            if log:
                log.info("[ENFORCE ATTEMPTING PATCH] Applying self-healing patch on %s: %s", full_target_file, rationale)
            tests_passed = apply_patch_fn(target_file=target_file, search_block=raw_search_block, replace_block=replace_block, rationale=rationale)

    return is_enforce and bool(venue_executed or (has_fix and tests_passed)), tests_passed


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
        allowed, reason, _, out_payload = _ntfypolicy.pass_common_gate(
            "ntfy", title, body, priority, target_topic,
            [_ntfypolicy.make_bot_event_alert(title, body, "ai_supervisor")],
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

    format_ntfy_url = getattr(_ntfyapi, "format_ntfy_url", None)
    AlertNotifier = getattr(_ntfyapi, "AlertNotifier", None)
    topic_raw = (target_topic or "").strip()
    url_raw = (ntfy_url or "").strip()
    chosen_target = topic_raw or url_raw
    url = format_ntfy_url(chosen_target) if format_ntfy_url is not None else (chosen_target if chosen_target.startswith(("http://", "https://")) else f"https://ntfy.sh/{chosen_target}")
    if not url:
        return False
    safe_title = (
        (AlertNotifier.utf8_header(title, "Alert") if AlertNotifier is not None else "")
        or "Alert"
    )
    headers = {"Title": safe_title, "Priority": priority, "Tags": "rotating_light,robot" if is_any_error else "shield,robot"}
    if ntfy_token:
        headers["Authorization"] = f"Bearer {ntfy_token}"

    try:
        return http_post_status(url, body, headers=headers, timeout=10, urlopen_fn=urlopen_fn) == 200
    except Exception as exc:
        if log:
            log.error("Failed sending standalone ntfy notification: %s", exc)
        return False


def default_supervisor_fallback_plan(spec: Any, logs: Any) -> Dict[str, Any]:
    """Return baseline fallback plan when Gemini query returns empty or fails."""
    if not getattr(spec, "can_self_heal", False):
        return {"summary": "Autonomous trade evaluation completed: holding posture.", "market_trading_decisions": []}
    return {
        "summary": "Autonomous audit completed: telemetry sensors active and bots healthy.",
        "diagnosed_issues": [{
            "severity": "INFO",
            "category": "HEALTHY",
            "description": f"Audited {getattr(logs, 'total_scanned_files', 0)} log files. {getattr(logs, 'error_count', 0)} errors, {getattr(logs, 'warning_count', 0)} warnings.",
            "root_cause": "System operational within expected tolerances.",
        }],
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
    elif "supervisor_mode" in conf:
        raw_mode = conf["supervisor_mode"]
    else:
        raise KeyError("Missing required 'supervisor_mode' in order_guard.conf")
    norm_mode = str(raw_mode).strip().lower()
    if norm_mode not in ("off", "shadow", "enforce"):
        raise ValueError(f"Invalid supervisor_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']")
    if "ntfy_topic_guard" not in conf:
        raise KeyError("Missing required 'ntfy_topic_guard' in config.env")
    ntfy_topic_guard = conf["ntfy_topic_guard"]
    ntfy_topic_server = conf.get("ntfy_topic_server") or ntfy_topic_guard
    return (
        norm_mode,
        parse_notify_enabled_flag(notify_enabled, conf, "supervisor_notify"),
        conf.get("phone_alert_url", ""),
        ntfy_topic_guard,
        ntfy_topic_server,
        conf.get("ntfy_topic_error") or ntfy_topic_server,
        conf.get("ntfy_token", ""),
    )


def _is_decision_in_scope(dec: Dict[str, Any], scope: str) -> bool:
    if str(dec.get("decision") or "HOLD").upper() not in ("BUY", "SELL") and not decision_has_cancel_directive(dec):
        return True
    sym_raw = dec.get("symbol")
    if is_readonly_xstock_symbol(sym_raw) and not is_xstock_trading_enabled(sym_raw):
        return False
    return matches_asset_scope(sym_raw, dec.get("venue"), scope)


def submit_supervisor_trading_decisions(
    plan: Dict[str, Any],
    spec: Any,
    *,
    topic_key: str,
    mode: str,
    now_ts: float,
    intent_gateway: Any,
    is_rth_open_fn: Any,
    model_used: str = "",
    log: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """Normalize and submit LLM trading decisions to AutonomousIntentGateway."""
    if not getattr(spec, "can_trade", False):
        return [], {}, [], None, False
    decisions_list = normalize_trading_decisions(plan)
    scope = str(getattr(spec, "asset_scope", "all") or "all").strip().lower()
    if decisions_list:
        decisions_list = [d for d in decisions_list if _is_decision_in_scope(d, scope)]
    market_decision = decisions_list[0] if decisions_list else {}
    intents_emitted: List[Dict[str, Any]] = []
    actionable_decs = [
        d
        for d in decisions_list
        if str(d.get("decision") or "HOLD").upper() in ("BUY", "SELL")
        or decision_has_cancel_directive(d)
    ]
    submit_decs = actionable_decs if actionable_decs else decisions_list[:1]
    effective_max_mode = "shadow" if (scope == "equities" and not is_rth_open_fn(now_ts)) else mode
    clean_model_used = model_used.strip() if isinstance(model_used, str) else ""
    if effective_max_mode == "enforce" and is_degraded_trading_model(clean_model_used):
        if log:
            log.warning(
                "[%s] Downgrading execution mode from enforce to shadow because degraded fallback model '%s' answered.",
                topic_key,
                clean_model_used,
            )
        effective_max_mode = "shadow"
    any_executed = False
    # Execute standalone CANCEL directives before BUY/SELL placements so freed quote/base liquidity
    # is unlocked on the exchange before new same-cycle orders size against wallet balance.
    exec_decs = sorted(
        submit_decs,
        key=lambda d: (
            0
            if str(d.get("decision") or "HOLD").upper() not in ("BUY", "SELL")
            and decision_has_cancel_directive(d)
            else 1
        ),
    )
    scope_fn = getattr(intent_gateway, "cycle_shared_cash_scope", None)
    with (scope_fn() if callable(scope_fn) else contextlib.nullcontext()):
        for dec in exec_decs:
            dec_payload = {**dec, "model_used": clean_model_used} if clean_model_used else dec
            if isinstance(emitted := intent_gateway.submit(dec_payload, source=topic_key, max_mode=effective_max_mode), dict):
                intents_emitted.append(emitted)
                if effective_max_mode == "enforce" and emitted.get("status") in ("EXECUTED", "CANCELED", "CANCEL_PARTIAL"):
                    any_executed = True
    return decisions_list, market_decision, intents_emitted, (intents_emitted[0] if intents_emitted else None), any_executed


def build_sensor_health_summary(sensors: Any) -> Dict[str, Any]:
    """Build sensor_health dictionary for SupervisorAuditReport."""
    summary: Dict[str, Any] = {
        **{f"{k}_present": getattr(sensors, k, None) is not None for k in MACRO_SENSOR_HEALTH_KEYS},
        **{f"{k}_count": len(getattr(sensors, k, None) or {}) for k in ("prices_and_trends", "microstructure", "active_heartbeats")},
    }
    sys_vpn = getattr(sensors, "system_and_vpn", None)
    cat = sys_vpn.get("llm_catalog_audit") if isinstance(sys_vpn, dict) else None
    if isinstance(cat, dict) and cat:
        if cat.get("missing_configured_models"):
            summary["missing_llm_models"] = list(cat["missing_configured_models"])
        if cat.get("new_families"):
            summary["new_llm_model_families"] = list(cat["new_families"])
        if cat.get("binary_versions"):
            summary["binary_versions"] = dict(cat["binary_versions"])
    return summary


def summarize_fleet_sensor_status(
    cachedb_dir: str,
    read_json_fn: Any = None,
) -> Dict[str, Any]:
    """Load and summarize real-time risk guards and telemetry files for FleetHealthMonitor."""
    _rj = _resolve_repair_json_reader(read_json_fn)
    sensors: Dict[str, Any] = {}

    if geo := read_first_cached_dict(cachedb_dir, "geopolitical_eval_files", "geopolitical_eval.json", read_json_fn=_rj)[0]:
        sensors["geopolitical_threat"] = {
            "threat_level": normalize_threat_level(geo.get("threat_level")),
            "risk_score": safe_float(geo.get("risk_score"), default=0.0),
            "summary": str(geo.get("summary") or ""),
        }
    if macro := read_first_cached_dict(
        cachedb_dir, "sentiment_eval_files", "sentiment_eval.json", "sentiment_state.json", "macro_sentiment_state.json", read_json_fn=_rj
    )[0]:
        sensors["sentiment_advisor"] = {
            "market_bias": normalize_market_bias(macro.get("market_bias")),
            "risk_level": normalize_risk_level(macro.get("risk_level")),
            "action": normalize_recommended_action(macro.get("recommended_action") or macro.get("action")),
        }
    if sup := read_first_cached_dict(
        cachedb_dir, "", "auto_cryptotrade_executor_eval.json", "auto_trade_executor_eval.json", "auto_sharetrade_executor_eval.json", read_json_fn=_rj
    )[0]:
        dec = sup.get("market_trading_decision") if isinstance(sup.get("market_trading_decision"), dict) else {}
        sensors["supervisor_eval"] = {
            "mode": str(sup.get("mode") or "enforce"),
            "decision": str(dec.get("decision") or "HOLD"),
            "confidence": safe_float(dec.get("confidence"), default=0.0),
            "summary": str(sup.get("summary") or ""),
        }
    if fg := read_first_cached_dict(cachedb_dir, "fear_greed_files", "fear_greed_collect.json", "fear_greed_state.json", read_json_fn=_rj)[0]:
        sensors["fear_greed"] = safe_float(fg.get("value"), default=None)
    if mb := read_first_cached_dict(cachedb_dir, "market_breadth_files", "market_breadth_collect.json", "market_breadth_state.json", read_json_fn=_rj)[0]:
        sensors["market_breadth"] = safe_float(mb.get("advance_ratio"), default=None)
    if cp := read_first_cached_dict(cachedb_dir, "", CACHE_CURRENT_PRICE_FILENAME, read_json_fn=_rj)[0]:
        sensors["current_price_cache"] = summarize_current_price_cache_freshness(
            cp,
            now_ts=time.time(),
            max_age_sec=PRICE_FRESHNESS_MAX_AGE_SEC,
        )

    return sensors


def __getattr__(name: str):
    if name == "harvest_fleet_and_system_telemetry":
        mod = sys.modules.get("orchestratorOS.livecheck.fleet_health_monitor") or getattr(
            __import__("orchestratorOS.livecheck.fleet_health_monitor"), "livecheck"
        ).fleet_health_monitor
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")





