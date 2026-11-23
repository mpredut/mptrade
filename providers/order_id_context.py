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
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
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
    ("MO_AG", "assetguardian", "assetguardian", "token"),
    ("MO_AI", "ai_executor", None, "token"),
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
_SORTED_BOT_CID_SPECS = tuple(sorted(BOT_CID_SPEC_TABLE, key=lambda row: len(row[0]), reverse=True))
_BOT_CID_PREFIX_MODULES: tuple[tuple[str, str], ...] = tuple(
    (_pfx, _tel_mod)
    for _pfx, _owner, _tel_mod, _mode in BOT_CID_SPEC_TABLE
    if _tel_mod is not None
)


KNOWN_BOT_PROCESS_CODES = frozenset(_PROCESS_NAME_OVERRIDES.values()) | frozenset({"MN", "AI"})

REPRICEABLE_BOTS = frozenset({"tradeall", "monitororder", "manual"})

BOT_PRIORITY_RANKS: dict[str, int] = {
    "rtrade": 10,
    "tradeall": 20,
    "assetguardian": 30,
    "monitortrades": 40,
    "spot_dca": 50,
    "trailing_stop": 50,
    "manual": 60,
    "ai_executor": 70,
}

_CLIENT_ORDER_COUNTER = itertools.count()

VENUE_PROVIDER_SPECS: dict[str, tuple[str, str]] = {
    "binance": ("providers.binance_provider", "BinanceProvider"),
    "hyperliquid": ("providers.hyperliquid_provider", "HyperliquidProvider"),
    "kraken": ("providers.kraken_provider", "KrakenProvider"),
    "t212": ("providers.t212_provider", "T212Provider"),
}


_PROCESS_STEM_SPLIT_RE = re.compile(r"[_\-.]+")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")
_CID_TOKEN_SPLIT_RE = re.compile(r"[_-]+")


def _derive_process_name(stem):
    stem = "python" if (stem is None or isinstance(stem, bool)) else (str(stem) or "python")
    key = stem.lower()
    if key in _PROCESS_NAME_OVERRIDES:
        return _PROCESS_NAME_OVERRIDES[key]

    parts = [part for part in _PROCESS_STEM_SPLIT_RE.split(stem) if part]
    if len(parts) > 1:
        return "".join(part[0].upper() for part in parts if part)

    alpha = _NON_ALNUM_RE.sub("", stem)
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


def record_client_order_id(record: Any) -> Optional[str]:
    """Strictly extract normalized ``record['place_kwargs']['client_order_id']`` from an outbox/claim record."""
    if not isinstance(record, dict):
        return None
    place_kw = record.get("place_kwargs")
    return clean_order_id(place_kw.get("client_order_id")) if isinstance(place_kw, dict) else None


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
    cid_up = cid.upper()
    for pfx, bot, _tel_mod, mode in _SORTED_BOT_CID_SPECS:
        if bot is not None and (
            cid_up.startswith(pfx)
            if mode == "stem"
            else (
                cid_up.startswith(f"{pfx}_")
                if mode == "prefix_only"
                else (cid_up == pfx or cid_up.startswith((f"{pfx}_", f"{pfx}-")))
            )
        ):
            return bot
    if cid.lower().startswith(MANUAL_CID_PREFIXES):
        return "manual"
    return f"alt:{cid[:4]}" if len(cid) >= 4 else "unknown"


def is_order_repriceable_by_monitororder(
    client_order_id: Optional[str], *, resolved_owner: Optional[str] = None
) -> bool:
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
    eff_owner = (
        resolved_owner.strip().lower()
        if isinstance(resolved_owner, str)
        and resolved_owner.strip()
        and resolved_owner.strip().lower() not in ("unspecified", "unknown", "monitororder")
        else owner
    )
    if eff_owner == "unspecified":
        return True
    return eff_owner in REPRICEABLE_BOTS


def resolve_priority_bot_owner(order_or_cid: Any, *, owner_map: Optional[dict[str, str]] = None) -> str:
    """Resolve the canonical bot owner of an order or clientOrderId for priority preemption."""
    if isinstance(order_or_cid, str):
        clean_s = order_or_cid.strip().lower()
        if clean_s in BOT_PRIORITY_RANKS:
            return clean_s
        if clean_s == "asset_guardian":
            return "assetguardian"
    cid = (
        extract_client_order_id(order_or_cid, default="")
        if isinstance(order_or_cid, dict)
        else (clean_order_id(order_or_cid) or "")
    )
    oid = extract_order_id(order_or_cid, default="") if isinstance(order_or_cid, dict) else ""
    if cid:
        cid_u = cid.upper()
        if cid_u == "TO" or cid_u.startswith(("TO_", "TO-")):
            return "tradeall"
        if cid_u in ("TS", "TRL") or cid_u.startswith(("TS_", "TS-", "TRL_", "TRL-")):
            return "trailing_stop"
        owner = resolve_order_owner(cid).lower()
        if owner in BOT_PRIORITY_RANKS:
            return owner
    if isinstance(owner_map, dict):
        for k in (oid, cid, cid.lower() if cid else ""):
            if k and k in owner_map:
                mapped = (clean_scalar_str(owner_map.get(k)) or "").lower()
                if mapped in BOT_PRIORITY_RANKS:
                    return mapped
    return "unspecified"


def resolve_owner_cid_prefix(owner: Optional[str]) -> Optional[str]:
    """Return the canonical ``create_client_order_id`` process prefix for a bot or actor owner name."""
    if not isinstance(owner, str) or not (key := owner.strip().lower()):
        return None
    if key == "ai_executor":
        return "AI"
    if key == "manual":
        return "MN"
    return _PROCESS_NAME_OVERRIDES.get(key)


def resolve_monitororder_replace_prefix(owner: Optional[str]) -> str:
    """Return the clientOrderId prefix for a monitororder replacement preserving ``owner``."""
    pfx = resolve_owner_cid_prefix(owner)
    return f"MO_{pfx}_OR" if pfx and pfx != "MO" else "MO_OR"


def bot_priority_rank(order_or_cid: Any, *, owner_map: Optional[dict[str, str]] = None) -> int:
    """Return numeric priority rank for a bot owner name, order dict, or clientOrderId (0 if unknown)."""
    owner = resolve_priority_bot_owner(order_or_cid, owner_map=owner_map)
    return BOT_PRIORITY_RANKS.get(owner, 0)


def preemptible_owners_for(caller_bot: Optional[str]) -> frozenset[str]:
    """Return the set of strictly lower-priority bot owners that ``caller_bot`` may preempt for balance."""
    caller_rank = bot_priority_rank(caller_bot)
    if caller_rank <= 0:
        return frozenset()
    return frozenset(
        bot for bot, rank in BOT_PRIORITY_RANKS.items() if 0 < rank < caller_rank
    )


def can_bot_preempt_order(
    caller_bot: Optional[str],
    order_or_cid: Any,
    *,
    owner_map: Optional[dict[str, str]] = None,
) -> bool:
    """Return True if ``caller_bot`` has strictly higher priority than the owner of ``order_or_cid``."""
    caller_rank = bot_priority_rank(caller_bot)
    if caller_rank <= 0:
        return False
    target_rank = bot_priority_rank(order_or_cid, owner_map=owner_map)
    return 0 < target_rank < caller_rank


def _normalize_allowed_owners(allowed_owners: Union[str, Iterable[str], None]) -> set[str]:
    if allowed_owners is None or isinstance(allowed_owners, (bool, dict)):
        return set()
    if isinstance(allowed_owners, str):
        clean_owner = clean_scalar_str(allowed_owners)
        return {clean_owner.lower()} if clean_owner else set()
    if not isinstance(allowed_owners, Iterable):
        return set()
    return {
        s.lower()
        for o in allowed_owners
        if o is not None
        and not isinstance(o, (bool, dict, list, tuple, set, frozenset))
        and (s := clean_scalar_str(str(o))) is not None
    }


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
    if "monitororder" in allowed_set and cid_str.upper().startswith("MO_"):
        return True
    return False


_KNOWN_AI_ID_SETS_CACHE: dict[frozenset[Any], tuple[frozenset[str], frozenset[str]]] = {}


def _resolve_known_ai_id_sets(
    known_ai_order_ids: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    if known_ai_order_ids is None or isinstance(known_ai_order_ids, (bool, str, bytes, dict)) or not hasattr(known_ai_order_ids, "__iter__"):
        return frozenset(), frozenset()
    if isinstance(known_ai_order_ids, frozenset):
        cached = _KNOWN_AI_ID_SETS_CACHE.get(known_ai_order_ids)
        if cached is not None:
            return cached
    exact_ids: set[str] = set()
    lower_ids: set[str] = set()
    for item in known_ai_order_ids:
        if (cid := clean_order_id(item)) is not None:
            exact_ids.add(cid)
            lower_ids.add(cid.lower())
    pair = (frozenset(exact_ids), frozenset(lower_ids))
    if isinstance(known_ai_order_ids, frozenset):
        if len(_KNOWN_AI_ID_SETS_CACHE) >= 64:
            _KNOWN_AI_ID_SETS_CACHE.clear()
        _KNOWN_AI_ID_SETS_CACHE[known_ai_order_ids] = pair
    return pair


def resolve_order_actor(order: Any, known_ai_order_ids: Optional[set[str]] = None) -> str:
    """Classify an order or fill record as placed by ``'AI'`` (autonomous executor) or ``'BOT'`` (deterministic fleet)."""
    if not isinstance(order, dict):
        return "BOT"
    explicit = clean_upper_scalar(order.get("actor")) or ""
    if explicit in ("AI", "BOT"):
        return explicit
    raw_cid = (extract_client_order_id(order, default="") or "").upper()
    if raw_cid.startswith(("AI_", "MO_AI_")):
        return "AI"
    oid = extract_order_id(order, default="")
    if not oid and not raw_cid:
        return "BOT"
    if isinstance(known_ai_order_ids, (set, frozenset)) and oid and str(oid) in known_ai_order_ids:
        return "AI"
    exact_ids, lower_ids = _resolve_known_ai_id_sets(known_ai_order_ids)
    if oid and exact_ids and (str(oid) in exact_ids or str(oid).lower() in lower_ids):
        return "AI"
    if raw_cid and lower_ids and raw_cid.lower() in lower_ids:
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

    raw_mod = _first_mapped(order, ("module", "origin_module", "bot", "caller"), clean_scalar_str)
    mod = raw_mod.lower() if raw_mod else None
    rule = _first_mapped(
        order, ("rule", "decision_rule", "trigger_rule", "kind", "role", "motivation"), clean_scalar_str
    )
    oid = extract_order_id(order, default="")
    raw_cid = extract_client_order_id(order, default="") or ""

    if isinstance(audit_origin_map, dict):
        for lookup_key in (str(oid) if oid else "", raw_cid, raw_cid.lower()):
            if lookup_key and isinstance(entry := audit_origin_map.get(lookup_key), (tuple, list)) and len(entry) >= 2:
                a_mod, a_rule = entry[0], entry[1]
                mod = mod or clean_scalar_str(a_mod)
                rule = rule or clean_scalar_str(a_rule)

    if raw_cid:
        cid_up = raw_cid.upper()
        for pfx, _owner, bot_name, mode in _SORTED_BOT_CID_SPECS:
            if bot_name is not None and (
                cid_up.startswith(pfx)
                if mode == "stem"
                else (
                    cid_up.startswith(f"{pfx}_")
                    if mode == "prefix_only"
                    else (cid_up == pfx or cid_up.startswith((f"{pfx}_", f"{pfx}-")))
                )
            ):
                mod = mod or bot_name
                if not rule:
                    if mode == "stem":
                        rule = "SL_PROTECT" if pfx == "AGS" else "BUY_TIER"
                    else:
                        rem = raw_cid[len(pfx):].lstrip("_-")
                        tokens = [t for t in _CID_TOKEN_SPLIT_RE.split(rem) if t and t.upper() not in _GENERIC_CID_TOKENS]
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
    act_u = (
        ""
        if actor is None or isinstance(actor, (bool, dict, list, tuple, set, frozenset))
        else (clean_upper_scalar(str(actor)) or "")
    )
    if act_u != "BOT":
        return act_u or "BOT"
    parts = [f"{k}={s}" for k, v in (("module", module), ("rule", rule)) if (s := clean_scalar_str(v))]
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


_UUID_HEX_SUFFIX_RE = re.compile(r"([0-9a-fA-F]{32})$")


def format_venue_intent_client_order_id(venue: Any, token: str) -> Optional[str]:
    """Format a 32-hex intent token into the venue's clientOrderId format (or ``None`` when unsupported)."""
    if isinstance(token, bool) or not isinstance(token, str) or not token.strip():
        return None
    prov_cls = resolve_venue_provider_class(venue)
    fmt = getattr(prov_cls, "_INTENT_CLIENT_ID_FORMAT", None) if prov_cls is not None else None
    return fmt.format(token=token.strip()) if isinstance(fmt, str) and fmt else None


def derive_ai_intent_client_order_id(intent_id: Any) -> Optional[str]:
    """Durable ``AI_`` client ID of an autonomous intent (<= 36 chars, Binance newClientOrderId compatible)."""
    raw = clean_order_id(intent_id) or ""
    if not raw:
        return None
    cid_body = (
        f"{raw[:18]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}_{raw[-7:]}"
        if len(raw) > 33
        else raw
    )
    return f"AI_{cid_body}"


def derive_executor_intent_client_order_id(venue: Any, intent_id: Any) -> Optional[str]:
    """Encode a strategy-executor intent ID into the venue's deterministic 128-bit clientOrderId format."""
    if isinstance(intent_id, (bool, list, tuple, dict, set, frozenset)):
        return None
    raw = clean_order_id(intent_id)
    if not raw:
        return None
    match = _UUID_HEX_SUFFIX_RE.search(raw)
    token = (
        match.group(1).lower()
        if match
        else hashlib.blake2s(raw.encode("utf-8"), digest_size=16).hexdigest()
    )
    v_slug = sanitize_identifier(venue, sep="-", allow_dots_hyphens=True, strip_chars="-").lower()
    return format_venue_intent_client_order_id(normalize_venue_name(v_slug) or v_slug, token)


def all_correlatable_client_order_ids(identifier: Any) -> set[str]:
    """Return all lowercase clientOrderId forms (raw, AI_, executor 128-bit, venue-native) for ``identifier``."""
    raw = clean_order_id(identifier)
    if not raw:
        return set()
    out = set(all_venue_native_client_order_ids(raw))
    if ai_cid := derive_ai_intent_client_order_id(raw):
        out.update(all_venue_native_client_order_ids(ai_cid))
    for venue_key in VENUE_PROVIDER_SPECS:
        if exec_cid := derive_executor_intent_client_order_id(venue_key, raw):
            out.add(exec_cid.lower())
    return out
