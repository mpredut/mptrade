"""Cross-venue clientOrderId process-lineage, bot-prefix, actor-attribution, and venue-CID SSOT."""

import hashlib
import importlib
import itertools
import math
import os
import re
import sys
import threading
import time
from typing import Any, Iterable, Optional, Union

from botcore import (
    _first_mapped,
    clean_nonempty_str,
    clean_order_id,
    extract_order_id,
    safe_nonneg_int,
    sanitize_identifier,
)
from cachepaths import normalize_venue_name
from .ohlc_utils import ProviderError

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

MANUAL_CID_PREFIXES: tuple[str, ...] = ("and_", "ios_", "web_", "x-", "electron_")
_GENERIC_CID_TOKENS = frozenset({"OR", "MAINTHREAD", "WORKER", "BOT", "ORDER", "RETRY", "OUTBOX"})
_CLIENT_ORDER_ID_KEYS = ("clientOrderId", "client_order_id", "cloid", "cl_ord_id", "userref")

# Declarative SSOT table: (prefix, owner_name, telemetry_module, match_mode).
# - "token": registers both "<pfx>_" and "<pfx>" in BOT_PREFIX_MAP and matches token-bounded CIDs.
# - "stem": registers "<pfx>" in BOT_PREFIX_MAP and matches stem-prefixed CIDs (e.g. AGS/AGB).
# - "prefix_only": registers "<pfx>_" in BOT_PREFIX_MAP.
BOT_CID_SPEC_TABLE: tuple[tuple[str, Optional[str], Optional[str], str], ...] = (
    ("MO_MN", "manual", "manual", "token"),
    ("MO_TA", "tradeall", "tradeall", "token"),
    ("MO_MT", "monitortrades", "monitortrades", "token"),
    ("MO_RT", "rtrade", "rtrade", "token"),
    ("MO_SD", "spot_dca", "trailing", "token"),
    ("TA_OR", "tradeall", "tradeall", "token"),
    ("MT_OR", "monitortrades", "monitortrades", "token"),
    ("AG_OR", "assetguardian", "assetguardian", "token"),
    ("RT_OR", "rtrade", "rtrade", "token"),
    ("SD_OR", "spot_dca", "trailing", "token"),
    ("MO_OR", "monitororder", "monitororder", "token"),
    ("MN_OR", "manual", "manual", "token"),
    ("AI_OR", "ai_executor", None, "token"),
    ("AGS", "assetguardian", "assetguardian", "stem"),
    ("AGB", "assetguardian", "assetguardian", "stem"),
    ("TRL", None, "trailing", "token"),
    ("SRV", "server", None, "token"),
    ("TA", "tradeall", "tradeall", "token"),
    ("TO", None, "tradeall", "token"),
    ("MT", "monitortrades", "monitortrades", "token"),
    ("AG", "assetguardian", "assetguardian", "token"),
    ("RT", "rtrade", "rtrade", "token"),
    ("MO", "monitororder", "monitororder", "token"),
    ("TS", None, "trailing", "token"),
    ("SD", "spot_dca", "trailing", "token"),
    ("MN", "manual", "manual", "token"),
    ("OR", "order_retry", None, "token"),
    ("AI", "ai_executor", None, "token"),
    ("CW", "watchdogfor_cacheandconfig", None, "prefix_only"),
    ("MA", "market_alerts", None, "prefix_only"),
    ("T2", "trade2", None, "prefix_only"),
    ("T3", "trade3", None, "prefix_only"),
    ("T4", "trade4", None, "prefix_only"),
    ("T5", "trade5", None, "prefix_only"),
    ("T", "trade", None, "prefix_only"),
)

BOT_PREFIX_MAP: dict[str, str] = {}
for _pfx, _owner, _tel_mod, _mode in BOT_CID_SPEC_TABLE:
    if _owner is None:
        continue
    if _mode == "token":
        BOT_PREFIX_MAP[f"{_pfx}_"] = _owner
        BOT_PREFIX_MAP[_pfx] = _owner
    elif _mode == "stem":
        BOT_PREFIX_MAP[_pfx] = _owner
    elif _mode == "prefix_only":
        BOT_PREFIX_MAP[f"{_pfx}_"] = _owner

_SORTED_BOT_PREFIXES = tuple(sorted(BOT_PREFIX_MAP.items(), key=lambda kv: len(kv[0]), reverse=True))
_BOT_CID_PREFIX_MODULES: tuple[tuple[str, str], ...] = tuple(
    (_pfx, _tel_mod)
    for _pfx, _owner, _tel_mod, _mode in BOT_CID_SPEC_TABLE
    if _tel_mod is not None
)
KNOWN_BOT_PROCESS_CODES = frozenset(_PROCESS_NAME_OVERRIDES.values()) | frozenset({"MN", "AI"})

REPRICEABLE_BOTS = frozenset({"tradeall", "monitororder", "manual"})

_CLIENT_ORDER_COUNTER = itertools.count()

VENUE_PROVIDER_SPECS: dict[str, tuple[str, str]] = {
    "binance": ("providers.binance_provider", "BinanceProvider"),
    "hyperliquid": ("providers.hyperliquid_provider", "HyperliquidProvider"),
    "kraken": ("providers.kraken_provider", "KrakenProvider"),
    "t212": ("providers.t212_provider", "T212Provider"),
}


def _derive_process_name(stem):
    stem = "python" if (stem is None or isinstance(stem, bool)) else (str(stem) or "python")
    key = stem.lower()
    if key in _PROCESS_NAME_OVERRIDES:
        return _PROCESS_NAME_OVERRIDES[key]

    parts = [part for part in re.split(r"[_\-.]+", stem) if part]
    if len(parts) > 1:
        return "".join(part[0].upper() for part in parts if part)

    alpha = re.sub(r"[^A-Za-z0-9]", "", stem)
    return alpha[:6].upper() if alpha else "PY"


def _sanitize_name(value, default="unknown"):
    return sanitize_identifier(value, default=default, sep="_", allow_dots_hyphens=False, strip_chars="_")


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


PROCESS_NAME = _sanitize_name(
    _derive_process_name(
        os.path.splitext(os.path.basename((sys.argv[0] if sys.argv else "") or "python"))[0] or "python"
    ),
    default="PY",
)


def get_process_name():
    return PROCESS_NAME


def create_client_order_id(process_name=None):
    pname = _sanitize_name(process_name, default=get_process_name()) if process_name else get_process_name()
    tname = _sanitize_name(threading.current_thread().name, default="MainThread")
    prefix = _sanitize_name(f"{pname}_{tname}", default=pname)
    counter = next(_CLIENT_ORDER_COUNTER) % (36 * 36)
    suffix = f"_{_base36(math.trunc(time.time() * 1000))[-8:]}{_base36(counter).zfill(2)}"
    max_prefix_len = CLIENT_ORDER_ID_MAX_LEN - len(suffix)

    if len(prefix) > max_prefix_len:
        prefix = (prefix[:max_prefix_len].rstrip("_-") or pname)[:max_prefix_len].rstrip("_-") or "PY"

    return f"{prefix}{suffix}"


def extract_client_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize client order ID or user reference from any venue shape."""
    return _first_mapped(order, _CLIENT_ORDER_ID_KEYS, clean_order_id, default)


def resolve_order_owner(client_order_id: Optional[str]) -> str:
    """Resolve the responsible bot or origin from a venue clientOrderId.

    Single Source of Truth (SSOT) across all fleet bots and reporting tools.
    Returns:
        One of 'rtrade', 'tradeall', 'monitororder', 'monitortrades',
        'spot_dca', 'assetguardian', 'server', 'manual', 'unspecified', or 'alt:...'.
    """
    if not isinstance(client_order_id, str):
        return "unspecified"
    cid = clean_order_id(client_order_id)
    if not cid:
        return "unspecified"
    for prefix, bot in _SORTED_BOT_PREFIXES:
        if cid.startswith(prefix):
            return bot
    if cid.lower().startswith(MANUAL_CID_PREFIXES):
        return "manual"
    return f"alt:{cid[:4]}" if len(cid) >= 4 else "unknown"


def is_order_repriceable_by_monitororder(client_order_id: Optional[str]) -> bool:
    """Determine whether an open Binance order is eligible for monitororder repricing.

    Safety invariant:
    - Only orders owned by bots in REPRICEABLE_BOTS ('tradeall', 'monitororder', 'manual')
      may be adjusted toward market price.
    - Grid orders ('rtrade'), protective stops ('spot_dca'), stop-loss ('monitortrades'),
      manual app/web orders, and unknown orders are protected
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


def _normalize_id_lookup_sets(ids: Any) -> tuple[set[str], set[str]]:
    if ids is None or isinstance(ids, (bool, str, bytes, dict)) or not hasattr(ids, "__iter__"):
        return set(), set()
    exact_ids: set[str] = set()
    lower_ids: set[str] = set()
    for item in ids:
        if item is not None and not isinstance(item, (bool, dict, list, tuple, set, frozenset)):
            s = str(item).strip()
            if s:
                exact_ids.add(s)
                lower_ids.add(s.lower())
    return exact_ids, lower_ids


def resolve_order_actor(order: Any, known_ai_order_ids: Optional[set[str]] = None) -> str:
    """Classify an order or fill record as placed by ``'AI'`` (autonomous executor) or ``'BOT'`` (deterministic fleet)."""
    if not isinstance(order, dict):
        return "BOT"
    explicit = (clean_nonempty_str(order.get("actor")) or "").upper()
    if explicit in ("AI", "BOT"):
        return explicit
    exact_ids, lower_ids = _normalize_id_lookup_sets(known_ai_order_ids)
    oid = extract_order_id(order, default="")
    if oid and exact_ids and (str(oid) in exact_ids or str(oid).lower() in lower_ids):
        return "AI"
    raw_cid = (extract_client_order_id(order, default="") or "").upper()
    if raw_cid.startswith("AI_") or (raw_cid and lower_ids and raw_cid.lower() in lower_ids):
        return "AI"
    return "BOT"


def resolve_order_origin(
    order: Any,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[dict[str, tuple[Optional[str], Optional[str]]]] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve ``(actor, module, rule)`` for an order or fill record."""
    actor = resolve_order_actor(order, known_ai_order_ids)
    if actor == "AI" or not isinstance(order, dict):
        return actor, None, None

    raw_mod = _first_mapped(order, ("module", "origin_module", "bot", "caller"), clean_nonempty_str)
    mod = raw_mod.lower() if raw_mod else None
    rule = _first_mapped(
        order, ("rule", "decision_rule", "trigger_rule", "kind", "role", "motivation"), clean_nonempty_str
    )
    oid = extract_order_id(order, default="")
    raw_cid = extract_client_order_id(order, default="") or ""

    if isinstance(audit_origin_map, dict):
        for lookup_key in (str(oid) if oid else "", raw_cid, raw_cid.lower()):
            if lookup_key and isinstance(entry := audit_origin_map.get(lookup_key), (tuple, list)) and len(entry) >= 2:
                a_mod, a_rule = entry[0], entry[1]
                mod = mod or clean_nonempty_str(a_mod)
                rule = rule or clean_nonempty_str(a_rule)

    if raw_cid:
        cid_up = raw_cid.upper()
        for pfx, bot_name in _BOT_CID_PREFIX_MODULES:
            if cid_up.startswith(f"{pfx}_") or cid_up == pfx or (pfx in ("AGS", "AGB") and cid_up.startswith(pfx)):
                mod = mod or bot_name
                if not rule:
                    if pfx in ("AGS", "AGB"):
                        rule = "SL_PROTECT" if pfx == "AGS" else "BUY_TIER"
                    else:
                        rem = raw_cid[len(pfx):].lstrip("_-")
                        tokens = [t for t in re.split(r"[_-]+", rem) if t and t.upper() not in _GENERIC_CID_TOKENS]
                        if tokens:
                            cand = (
                                "_".join(tokens[:-1])
                                if len(tokens) > 1 and (tokens[-1].isdigit() or len(tokens[-1]) >= 8)
                                else tokens[0]
                            )
                            if not (len(cand) >= 8 and all(c in "0123456789abcdefABCDEF" for c in cand)) and not cand.isdigit():
                                rule = cand.upper()
                break
        if mod is None and raw_cid.lower().startswith(MANUAL_CID_PREFIXES):
            mod = "manual"
    return "BOT", mod, rule


def format_actor_tag(actor: Optional[str], module: Optional[str] = None, rule: Optional[str] = None) -> str:
    """Format compact actor attribution string (e.g. ``'AI'`` or ``'BOT(module=assetguardian, rule=SL_PANIC)'``)."""
    act_u = str(actor or "").strip().upper()
    if act_u != "BOT":
        return act_u or "BOT"
    parts = [f"{k}={v}" for k, v in (("module", module), ("rule", rule)) if v]
    return f"BOT({', '.join(parts)})" if parts else "BOT"


def resolve_venue_provider_class(
    venue: Any,
    *,
    provider: Any = None,
    base_cls: Any = None,
    specs: Optional[dict[str, tuple[str, str]]] = None,
) -> Any:
    """Resolve the concrete ``MarketDataProvider`` subclass for ``venue`` or ``provider``."""
    spec_map = specs if isinstance(specs, dict) else VENUE_PROVIDER_SPECS
    if not isinstance(venue, bool) and venue is not None:
        if isinstance(venue, type) and (
            (base_cls is not None and issubclass(venue, base_cls)) or hasattr(venue, "supports_symbol")
        ):
            return venue
        if not isinstance(venue, (str, bytes)) and (
            (base_cls is not None and isinstance(venue, base_cls)) or hasattr(type(venue), "supports_symbol")
        ):
            return type(venue)
        spec = spec_map.get(normalize_venue_name(venue))
        if spec is not None:
            mod_path, cls_name = spec
            resolved = getattr(importlib.import_module(mod_path), cls_name, None)
            if resolved is not None:
                return resolved
    if provider is not None and not isinstance(provider, (str, bytes, bool)):
        if isinstance(provider, type) and (
            (base_cls is not None and issubclass(provider, base_cls)) or hasattr(provider, "supports_symbol")
        ):
            return provider
        if (base_cls is not None and isinstance(provider, base_cls)) or hasattr(type(provider), "supports_symbol"):
            return type(provider)
    return None


def derive_venue_128bit_client_order_id(venue: Any, client_order_id: Any) -> str:
    """Map one durable client ID to the declarative 128-bit clientOrderId form for ``venue``."""
    prov_cls = resolve_venue_provider_class(venue)
    spec = getattr(prov_cls, "_CLIENT_ID_128BIT_SPEC", None) if prov_cls is not None else None
    if spec is None:
        raise ProviderError(f"Unsupported 128-bit client_order_id venue: {venue!r}")
    venue_label, domain_prefix, exact_re, hex_prefix = spec
    if isinstance(client_order_id, (bool, list, tuple, dict, set, frozenset)):
        raise ProviderError(f"{venue_label} client_order_id must contain 1-128 characters")
    raw = clean_order_id(client_order_id) or ""
    if not raw or len(raw) > 128:
        raise ProviderError(f"{venue_label} client_order_id must contain 1-128 characters")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProviderError(f"{venue_label} client_order_id must be ASCII") from exc
    if exact_re.fullmatch(raw):
        return raw.lower()
    return hex_prefix + hashlib.sha256(domain_prefix + encoded).hexdigest()[:32]


def all_venue_native_client_order_ids(client_order_id: Any) -> set[str]:
    """Return the lowercase venue-native forms of one durable client ID across all registered venues."""
    raw = clean_order_id(client_order_id)
    if not raw:
        return set()
    out = {raw.lower()}
    for venue_key in VENUE_PROVIDER_SPECS:
        prov_cls = resolve_venue_provider_class(venue_key)
        if getattr(prov_cls, "_CLIENT_ID_128BIT_SPEC", None) is None:
            continue
        try:
            out.add(derive_venue_128bit_client_order_id(prov_cls, raw).lower())
        except Exception:
            pass
    return out


def format_venue_intent_client_order_id(venue: Any, token: str) -> Optional[str]:
    """Format a 32-hex intent token into the venue's clientOrderId format (or ``None`` when unsupported)."""
    if isinstance(token, bool) or not isinstance(token, str) or not token.strip():
        return None
    prov_cls = resolve_venue_provider_class(venue)
    fmt = getattr(prov_cls, "_INTENT_CLIENT_ID_FORMAT", None) if prov_cls is not None else None
    return fmt.format(token=token.strip()) if isinstance(fmt, str) and fmt else None
