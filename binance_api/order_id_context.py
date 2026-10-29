import itertools
import os
import re
import sys
import threading
import time


from typing import Optional, Union, Iterable
from botcore import safe_nonneg_int

CLIENT_ORDER_ID_MAX_LEN = 36

_PROCESS_NAME_OVERRIDES = {
    "tradeall": "TA",
    "trade": "T",
    "trade2": "T2",
    "trade3": "T3",
    "trade4": "T4",
    "trade5": "T5",
    "rtrade": "RT",
    "monitortrades": "MT",
    "monitororder": "MO",
    "market_alerts": "MA",
    "watchdogfor_cacheandconfig": "CW",
    "server": "SRV",
    "spot_dca": "SD",
    "trailing_stop": "SD",
    "assetguardian": "AG",
}

# Canonical prefix -> bot name mapping.
BOT_PREFIX_MAP: dict[str, str] = {}
for _pfx, _bot in (
    ("MO_TA", "tradeall"), ("MO_MN", "manual"), ("MO_RT", "rtrade"),
    ("MO_MT", "monitortrades"), ("MO_SD", "spot_dca"),
    ("TA_OR", "tradeall"), ("MN_OR", "manual"), ("MO_OR", "monitororder"),
    ("RT_OR", "rtrade"), ("MT_OR", "monitortrades"), ("SD_OR", "spot_dca"),
    ("AG_OR", "assetguardian"), ("AI_OR", "ai_executor"),
    ("RT", "rtrade"), ("TA", "tradeall"), ("MO", "monitororder"),
    ("MT", "monitortrades"), ("SD", "spot_dca"), ("AG", "assetguardian"),
    ("MN", "manual"), ("OR", "order_retry"), ("SRV", "server"),
    ("AI", "ai_executor"),
):
    BOT_PREFIX_MAP[f"{_pfx}_"] = _bot
    BOT_PREFIX_MAP[_pfx] = _bot
BOT_PREFIX_MAP.update({
    "AGS": "assetguardian", "AGB": "assetguardian",
    "CW_": "watchdogfor_cacheandconfig", "MA_": "market_alerts",
    "T_": "trade", "T2_": "trade2", "T3_": "trade3", "T4_": "trade4", "T5_": "trade5",
})
_SORTED_BOT_PREFIXES = tuple(sorted(BOT_PREFIX_MAP.items(), key=lambda kv: len(kv[0]), reverse=True))

MANUAL_PREFIXES = ("and_", "ios_", "web_", "x-", "electron_")

REPRICEABLE_BOTS = frozenset({"tradeall", "monitororder", "manual"})

_CLIENT_ORDER_COUNTER = itertools.count()
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _script_stem():
    if not sys.argv:
        return "python"
    script = sys.argv[0] or "python"
    stem, _ = os.path.splitext(os.path.basename(script))
    return stem or "python"


def _derive_process_name(stem):
    if stem is None or isinstance(stem, bool):
        stem = "python"
    else:
        stem = str(stem) or "python"
    key = stem.lower()
    if key in _PROCESS_NAME_OVERRIDES:
        return _PROCESS_NAME_OVERRIDES[key]

    parts = [part for part in re.split(r"[_\-.]+", stem) if part]
    if len(parts) > 1:
        return "".join(part[0].upper() for part in parts if part)

    alpha = re.sub(r"[^A-Za-z0-9]", "", stem)
    if not alpha:
        return "PY"
    return alpha[:6].upper()


def _sanitize_name(value, default="unknown"):
    if isinstance(value, bool):
        value = ""
    safe = _SAFE_NAME_RE.sub("_", str(value or "")).strip("_")
    return safe or default


def _base36(value):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = safe_nonneg_int(value, default=0)
    if value == 0:
        return "0"

    chars = []
    while value:
        value, rem = divmod(value, 36)
        chars.append(alphabet[rem])
    return "".join(reversed(chars))


PROCESS_NAME = _sanitize_name(_derive_process_name(_script_stem()), default="PY")


def set_process_name(name):
    global PROCESS_NAME
    PROCESS_NAME = _sanitize_name(name, default="PY")
    return PROCESS_NAME


def get_process_name():
    return PROCESS_NAME


def get_thread_name():
    return _sanitize_name(threading.current_thread().name, default="MainThread")


def create_client_order_id(process_name=None):
    pname = _sanitize_name(process_name, default=get_process_name()) if process_name else get_process_name()
    prefix = _sanitize_name(f"{pname}_{get_thread_name()}", default=pname)
    counter = next(_CLIENT_ORDER_COUNTER) % (36 * 36)
    suffix = f"_{_base36(int(time.time() * 1000))[-8:]}{_base36(counter).zfill(2)}"
    max_prefix_len = CLIENT_ORDER_ID_MAX_LEN - len(suffix)

    if len(prefix) > max_prefix_len:
        prefix = (prefix[:max_prefix_len].rstrip("_-") or pname)[:max_prefix_len].rstrip("_-") or "PY"

    return f"{prefix}{suffix}"


def resolve_order_owner(client_order_id: Optional[str]) -> str:
    """Resolve the responsible bot or origin from a Binance clientOrderId.

    Single Source of Truth (SSOT) across all fleet bots and reporting tools.
    Returns:
        One of 'rtrade', 'tradeall', 'monitororder', 'monitortrades',
        'spot_dca', 'assetguardian', 'server', 'manual', 'unspecified', or 'alt:...'.
    """
    if not isinstance(client_order_id, str):
        return "unspecified"
    cid = client_order_id.strip()
    if not cid or cid.lower() in {"none", "null", "nan", "inf", "-inf", "+inf", "0", "0.0", "-0", "+0"}:
        return "unspecified"
    for prefix, bot in _SORTED_BOT_PREFIXES:
        if cid.startswith(prefix):
            return bot
    if cid.lower().startswith(MANUAL_PREFIXES):
        return "manual"
    return f"alt:{cid[:4]}" if len(cid) >= 4 else "unknown"


def is_order_repriceable_by_monitororder(client_order_id: Optional[str]) -> bool:
    """Determine whether an open Binance order is eligible for monitororder repricing.

    Safety invariant:
    - Only orders owned by 'tradeall' or 'monitororder' may be adjusted toward market price.
    - Grid orders ('rtrade'), protective stops ('spot_dca'), stop-loss ('monitortrades'),
      emergency guard ('assetguardian'), manual app/web orders, and unknown orders are protected
      and must NEVER be cancelled or modified by monitororder.
    - Unspecified/empty clientOrderId (e.g. legacy test mocks) is allowed for test compatibility.
    """
    owner = resolve_order_owner(client_order_id)
    if owner == "unspecified":
        return True
    return owner in REPRICEABLE_BOTS


def _normalize_allowed_owners(allowed_owners: Union[str, Iterable[str], None]) -> set[str]:
    if allowed_owners is None or isinstance(allowed_owners, bool):
        return set()
    if isinstance(allowed_owners, str):
        clean_owner = allowed_owners.strip().lower()
        return {clean_owner} if clean_owner else set()
    try:
        return {
            str(o).strip().lower()
            for o in allowed_owners
            if o is not None and not isinstance(o, bool) and str(o).strip()
        }
    except TypeError:
        return set()


def order_matches_owner(client_order_id: Optional[str], allowed_owners: Union[str, Iterable[str], None]) -> bool:
    """Check if an order's clientOrderId matches any of the allowed owners."""
    if allowed_owners is None:
        return True
    allowed_set = _normalize_allowed_owners(allowed_owners)
    if not allowed_set:
        return False
    owner = resolve_order_owner(client_order_id).lower()
    if owner in allowed_set:
        return True
    cid_str = "" if client_order_id is None or isinstance(client_order_id, bool) else str(client_order_id).strip()
    if "monitororder" in allowed_set and cid_str.startswith("MO_"):
        return True
    return False
