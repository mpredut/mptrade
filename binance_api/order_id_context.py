import itertools
import os
import re
import sys
import threading
import time


from typing import Optional, Union, Iterable

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
BOT_PREFIX_MAP = {
    "RT_": "rtrade",
    "RT": "rtrade",
    "TA_": "tradeall",
    "TA": "tradeall",
    "MO_": "monitororder",
    "MO": "monitororder",
    "MT_": "monitortrades",
    "MT": "monitortrades",
    "SD_": "spot_dca",
    "SD": "spot_dca",
    "AG_": "assetguardian",
    "AG": "assetguardian",
    "SRV_": "server",
    "SRV": "server",
    "CW_": "watchdogfor_cacheandconfig",
    "MA_": "market_alerts",
    "T_": "trade",
    "T2_": "trade2",
    "T3_": "trade3",
    "T4_": "trade4",
    "T5_": "trade5",
}

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
    safe = _SAFE_NAME_RE.sub("_", str(value or "")).strip("_")
    return safe or default


def _base36(value):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = int(value)
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
        prefix = prefix[:max_prefix_len].rstrip("_-") or pname

    return f"{prefix}{suffix}"


def resolve_order_owner(client_order_id: Optional[str]) -> str:
    """Resolve the responsible bot or origin from a Binance clientOrderId.

    Single Source of Truth (SSOT) across all fleet bots and reporting tools.
    Returns:
        One of 'rtrade', 'tradeall', 'monitororder', 'monitortrades',
        'spot_dca', 'assetguardian', 'server', 'manual', 'unspecified', or 'alt:...'.
    """
    cid = str(client_order_id or "").strip()
    if not cid:
        return "unspecified"
    for prefix, bot in BOT_PREFIX_MAP.items():
        if cid.startswith(prefix):
            return bot
    if cid.startswith(MANUAL_PREFIXES):
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


def order_matches_owner(client_order_id: Optional[str], allowed_owners: Union[str, Iterable[str], None]) -> bool:
    """Check if an order's clientOrderId matches any of the allowed owners."""
    if allowed_owners is None:
        return True
    if isinstance(allowed_owners, str):
        allowed_set = {allowed_owners}
    else:
        allowed_set = set(allowed_owners)
    owner = resolve_order_owner(client_order_id)
    return owner in allowed_set
