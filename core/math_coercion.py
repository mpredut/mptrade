"""core/math_coercion.py -- Canonical scalar coercion, order extraction, and deterministic math primitives."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import math
import re

from cachepaths import (
    INVALID_SCALAR_STRINGS,
    _parse_bool_flag,
    clean_scalar_str,
    exact_int,
    exact_pos_int,
    is_kraken_tokenized_asset,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)

_UNSET = globals().get("_UNSET") or object()


def strict_ascii_int(
    val: object, default: int | None = None, *, min_value: int | None = None, allow_float_str: bool = False
) -> int | None:
    """Coerce ``val`` (int, integral float, or ASCII digit string) to int, rejecting booleans and non-ASCII digits."""
    if isinstance(val, bool) or not isinstance(val, (int, float, str)):
        return default
    if isinstance(val, str):
        s = val.strip()
        if allow_float_str and s.endswith(".0"):
            s = s[:-2]
        cleaned = s.lstrip("+-")
        if not cleaned or not cleaned.isascii() or not cleaned.isdigit():
            return default
        parsed = exact_int(s, default=None)
    else:
        parsed = exact_int(val, default=None)
    if parsed is None or (min_value is not None and parsed < min_value):
        return default
    return parsed


def _bind_required_float(fn_name: str, coercer: object, default_desc: str):
    def _fn(
        val: object, name: str = "value", *, min_value: float | None = None, error_msg: str | None = None
    ) -> float:
        f = coercer(val, default=None)  # type: ignore[operator]
        if f is None or (min_value is not None and f < min_value):
            msg = (
                f"{name} must be finite and >= {min_value}"
                if min_value is not None
                else f"{name} must be {default_desc}"
            )
            raise ValueError(error_msg or msg)
        return f

    _fn.__name__ = _fn.__qualname__ = fn_name
    return _fn


require_finite_float = _bind_required_float("require_finite_float", safe_float, "finite")
require_pos_float = _bind_required_float("require_pos_float", safe_pos_float, "finite and positive")
require_nonneg_float = _bind_required_float("require_nonneg_float", safe_nonneg_float, "finite and non-negative")


def first_pos_float(*values: object, default: float | None = None) -> float | None:
    """Return the first strictly positive finite float among ``values``, skipping None/0/'0'/'0.0'/NaN/Inf/bool."""
    for val in values:
        if (parsed := safe_pos_float(val, default=None)) is not None:
            return parsed
    return default


def require_int(
    val: object, name: str = "value", *, min_value: int | None = None, error_msg: str | None = None
) -> int:
    """Coerce ``val`` to an exact integer (optionally ``>= min_value``) or raise ``ValueError``."""
    parsed = exact_int(val, default=None)
    if parsed is None or (min_value is not None and parsed < min_value):
        raise ValueError(error_msg or f"invalid {name}")
    return parsed


def obj_attr(obj: object, key: str, default: object = None) -> object:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None or isinstance(obj, (str, bytes, bytearray)):
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if val is None else val


def obj_float(obj: object, key: str, default: float = 0.0) -> float:
    """Extract ``key`` from ``obj`` (dict or object) and coerce to finite float."""
    return safe_float(obj_attr(obj, key, None), default)


def coerce_dict_maps(*maps: object) -> tuple[dict, ...]:
    """Return a tuple where each item is preserved if ``isinstance(item, dict)`` else ``{}``."""
    return tuple(m if isinstance(m, dict) else {} for m in maps)


def clean_json_value(val: object, *, stringify_keys: bool = True) -> object:
    """Recursively sanitize non-finite floats (NaN/Inf -> None) and optionally normalize dict keys to strings."""
    if isinstance(val, float):
        return safe_float(val, default=None)
    if isinstance(val, dict):
        return {(str(k) if stringify_keys else k): clean_json_value(v, stringify_keys=stringify_keys) for k, v in val.items()}
    if isinstance(val, (list, tuple, set, frozenset)):
        return [clean_json_value(v, stringify_keys=stringify_keys) for v in val]
    return val


def lru_dict_put(cache: dict, key: object, value: object, max_entries: int) -> None:
    """Insert/refresh ``(key, value)`` at the MRU end of ``cache`` and evict oldest entries beyond ``max_entries``."""
    if hasattr(cache, "move_to_end"):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
    else:
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        for old_key in list(cache)[: max(0, len(cache) - max_entries)]:
            cache.pop(old_key, None)


class OrderedIdSet:
    """Insertion-ordered set of IDs with deterministic FIFO eviction."""

    def __init__(self):
        self._d = OrderedDict()

    def add(self, item: str) -> None:
        self._d[item] = None
        self._d.move_to_end(item)

    def clear(self) -> None:
        self._d.clear()

    def trim(self, keep: int = 5000) -> None:
        while len(self._d) > keep:
            self._d.popitem(last=False)

    def difference_update(self, items) -> None:
        for item in items:
            self._d.pop(item, None)

    def __contains__(self, item) -> bool:
        return item in self._d

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self):
        return iter(self._d)


def format_price_log(price: object) -> str:
    """Format a price with 2 decimals when >= 1.0 and up to 6 decimals for sub-dollar values."""
    px = safe_float(price, default=0.0) or 0.0
    return f"{px:.2f}" if abs(px) >= 1.0 else (f"{px:.6f}".rstrip("0").rstrip(".") or "0")


def format_notify_qty(qty: object) -> str:
    """Format a trade/order quantity without trailing zeros."""
    qf = safe_pos_float(qty, default=0.0) or 0.0
    return f"{qf:.8f}".rstrip("0").rstrip(".") or "0"


def format_notify_price(price: object, *, comma: bool = True) -> str:
    """Format a price with optional comma grouping and full sub-dollar precision."""
    pf = safe_pos_float(price, default=0.0) or 0.0
    sep = "," if comma else ""
    if pf >= 1.0 and round(pf, 2) == round(pf, 6):
        return f"{pf:{sep}.2f}"
    raw = f"{pf:{sep}.8f}".rstrip("0")
    if raw.endswith("."):
        return f"{raw}00"
    if len(raw.rsplit(".", 1)[-1]) == 1:
        return f"{raw}0"
    return raw


def fmt_signed(
    val: object,
    decimals: int = 2,
    *,
    default: float = 0.0,
    comma: bool = False,
    strip_zeros: bool = False,
) -> str:
    """Format a signed finite float after normalizing negative zero at the target rounding precision."""
    num = safe_float(val, default=default)
    if num is None:
        num = default
    rounded = round(num, decimals) + 0.0
    sep = "," if comma else ""
    out = f"{rounded:+{sep}.{decimals}f}"
    if strip_zeros and decimals > 0:
        out = out.rstrip("0").rstrip(".")
        if out in ("-0", "+0", "0"):
            out = "+0"
    return out


def parse_ratio_or_pct(raw: object, pct_cutoff: float = 0.05) -> float:
    """Parse a numeric config value that may be expressed either as a ratio or as a percentage."""
    if isinstance(raw, str) and raw.strip().endswith("%"):
        return require_nonneg_float(raw.strip()[:-1].strip(), name="ratio_or_pct") / 100.0
    val = require_nonneg_float(raw, name="ratio_or_pct")
    cutoff = require_nonneg_float(pct_cutoff, name="pct_cutoff")
    return val / 100.0 if val > cutoff else val


def safe_thresh(val: object, default: int) -> int:
    """Safely coerce sentiment index/threshold to an integer in [0, 100]."""
    f = safe_float(val, -1.0)
    return round(f) if 0.0 <= f <= 100.0 else default


def obj_attr_first(obj: object, *names: str, default: object = None) -> object:
    """Return the first non-``None`` attribute or mapping value among ``names`` on ``obj``."""
    for name in names:
        val = obj_attr(obj, name, None)
        if val is not None:
            return val
    return default


def coerce_finite_float(
    raw: object, *, positive: bool = False, default: float | None = None
) -> float | None:
    """Coerce ``raw`` to a finite float (or strictly positive finite float when ``positive=True``)."""
    return safe_pos_float(raw, default=default) if positive else safe_float(raw, default=default)


def resolve_positive_max_age(max_age_sec: object, default: float | None) -> float | None:
    """Return sanitized positive ``default`` when ``max_age_sec is None``, else strictly positive finite float or ``None``."""
    return (
        safe_pos_float(default, default=None)
        if max_age_sec is None
        else safe_pos_float(max_age_sec, default=None)
    )


def resolve_order_notional(
    notional_eur: object,
    price: object,
    qty: object,
    *,
    default: float | None = None,
    fallback_on_nonfinite: bool = True,
) -> float | None:
    """Resolve positive order notional from explicit ``notional_eur`` or ``price * qty`` fallback."""
    n_clean = safe_pos_float(notional_eur, 0.0)
    if n_clean > 0.0:
        return n_clean
    if not fallback_on_nonfinite and notional_eur is not None and safe_float(notional_eur, None) is None:
        return default
    px_clean = safe_pos_float(price, 0.0)
    qty_clean = safe_pos_float(qty, 0.0)
    return (px_clean * qty_clean) if (px_clean > 0.0 and qty_clean > 0.0) else default


def clean_scale(
    val: object,
    default: float = 1.0,
    *,
    allow_percent: bool = False,
    min_positive: bool = False,
    clamp: bool = False,
) -> float:
    """Normalize a scale factor into [0.0, 1.0], optionally converting [5.0, 100.0] or '%' percentages."""
    if allow_percent and isinstance(val, str) and val.strip().endswith("%"):
        pct = safe_nonneg_float(val.strip()[:-1].strip(), default=None)
        f = round(pct / 100.0, 4) if (pct is not None and pct <= 100.0) else -1.0
    else:
        f = safe_float(val, -1.0)
        if allow_percent and 5.0 <= f <= 100.0:
            f = round(f / 100.0, 4)
    if f < 0.0 or (min_positive and f <= 0.0):
        return default
    if clamp and f > 1.0:
        return 1.0
    return f if 0.0 <= f <= 1.0 else default


def _first_mapped(mapping: object, keys: tuple[str, ...], parser: object, default: object = None) -> object:
    """Return the first non-None parsed value from ``mapping`` across ``keys``."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            val = parser(mapping[key])
            if val is not None:
                return val
    return default


_SAFE_ID_DOT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_ID_NODOT_RE = re.compile(r"[^A-Za-z0-9_-]+")
_TREND_UP_TOKENS = frozenset(("up", "bull", "uptrend", "bullish"))
_TREND_DOWN_TOKENS = frozenset(("down", "bear", "downtrend", "bearish"))
_BUY_SIDE_TOKENS = frozenset(("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"))
_SELL_SIDE_TOKENS = frozenset(("A", "S", "SELL", "SOLD", "ASK", "SHORT", "CLOSE LONG", "OPEN SHORT"))
_EXACT_SIDE_MAP = {
    "BUY": "BUY",
    "SELL": "SELL",
    "buy": "BUY",
    "sell": "SELL",
    "B": "BUY",
    "b": "BUY",
    "S": "SELL",
    "s": "SELL",
    "A": "SELL",
    "a": "SELL",
}
_INVALID_OID_STRINGS = INVALID_SCALAR_STRINGS | frozenset(("0", "0.0", "-0", "+0", "+0.0", "-0.0"))


def normalize_side(side: object, *, require_str: bool = False) -> str:
    """Normalize a scalar order side token to ``'BUY'`` or ``'SELL'`` (or ``''`` when unrecognized)."""
    if type(side) is str:
        fast = _EXACT_SIDE_MAP.get(side)
        if fast is not None:
            return fast
    elif side is None or isinstance(side, bool) or require_str:
        return ""
    s = " ".join(str(side).strip().upper().replace("_", " ").replace("-", " ").split())
    if s in _BUY_SIDE_TOKENS:
        return "BUY"
    if s in _SELL_SIDE_TOKENS:
        return "SELL"
    return ""


def require_buy_sell_side(side: object, *, name: str = "side") -> str:
    """Normalize ``side`` to ``'BUY'`` or ``'SELL'``, raising ``ValueError`` otherwise."""
    norm = normalize_side(side)
    if norm not in ("BUY", "SELL"):
        raise ValueError(f"Invalid {name} {side!r}: {name} must be BUY or SELL (expected 'BUY' or 'SELL')")
    return norm


def opposite_side(side: object) -> str:
    """Return the opposite order side (``'SELL'`` for ``'BUY'``, ``'BUY'`` for ``'SELL'``)."""
    return "SELL" if require_buy_sell_side(side, name="order_type") == "BUY" else "BUY"


def clean_order_id(val: object) -> str | None:
    """Validate and normalize a scalar venue order ID to a non-empty string (or ``None``)."""
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, (int, float)):
        val = exact_pos_int(val, default=None)
        if val is None:
            return None
    if not isinstance(val, (str, int)):
        return None
    text = str(val).strip()
    if not text or text.startswith("-") or text.lower() in _INVALID_OID_STRINGS:
        return None
    if text.endswith(".0") and text[:-2].lstrip("+").isdigit():
        text = text[:-2]
    if text in {"0", "+0"}:
        return None
    return text[1:] if text.startswith("+") and text[1:].isdigit() else text


def clean_order_id_str(val: object) -> str:
    """Return ``clean_order_id(val)`` or ``''`` when empty/invalid."""
    return clean_order_id(val) or ""


def normalize_scalar_order_id(val: object) -> int | str | None:
    """Normalize a scalar order ID, preserving positive integers as ``int`` and string IDs as ``str``."""
    cleaned = clean_order_id(val)
    if cleaned is None:
        return None
    return exact_pos_int(val, default=cleaned) if isinstance(val, (int, float)) else cleaned


def normalize_trend_direction(
    raw_dir: object,
    default: str | None = None,
    *,
    preserve_bull_bear: bool = False,
    allow_long_short: bool = False,
) -> str | None:
    """Normalize a raw trend token (``up``/``bull``/``uptrend``/``bullish``/``long``, etc.) into a canonical direction."""
    if not isinstance(raw_dir, str):
        return default
    d = raw_dir.strip().lower()
    if d in _TREND_UP_TOKENS or (allow_long_short and d == "long"):
        return ("up" if d == "up" else "bull") if preserve_bull_bear else "up"
    if d in _TREND_DOWN_TOKENS or (allow_long_short and d == "short"):
        return ("down" if d == "down" else "bear") if preserve_bull_bear else "down"
    return default


def sanitize_pipe_field(value: object) -> str:
    """Sanitize a field value for pipe-delimited logs by replacing ``|``, ``\\r``, and ``\\n``."""
    return "" if value is None else str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def sanitize_identifier(
    value: object,
    default: str = "unknown",
    *,
    sep: str = "_",
    allow_dots_hyphens: bool = True,
    strip_chars: str | None = None,
) -> str:
    """Sanitize an identifier/slug by replacing unsafe characters with ``sep`` and stripping edge separators."""
    if value is None or isinstance(value, bool):
        return default
    pattern = _SAFE_ID_DOT_RE if allow_dots_hyphens else _SAFE_ID_NODOT_RE
    cleaned = pattern.sub(sep, str(value).strip()).strip(strip_chars if strip_chars is not None else sep)
    return cleaned or default


def normalize_class_name(cls_or_name: object) -> str:
    """Return the class name for a ``type`` object or stripped string, else ``''``."""
    if isinstance(cls_or_name, type):
        return cls_or_name.__name__
    return cls_or_name.strip() if isinstance(cls_or_name, str) else ""


_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "txid", "oid", "i", "id")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "L", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "executed_qty", "filledQuantity", "filledQty", "filled_qty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "requested_qty", "orderedQuantity", "amount_shares", "l", "z", "q",
)
_COST_EXEC_KEY_PAIRS = (
    ("cummulativeQuoteQty", "executedQty"),
    ("cummulativeQuoteQty", "qty"),
    ("cummulativeQuoteQty", "quantity"),
    ("quoteQty", "qty"),
    ("quoteQty", "quantity"),
    ("cost", "vol_exec"),
    ("cost", "vol"),
    ("filledValue", "filledQuantity"),
    ("Y", "l"),
    ("Z", "z"),
)


def _pos_abs_float(value: object) -> float | None:
    val = abs(safe_float(value, default=0.0))
    return val if val > 0.0 else None


def _nonneg_abs_float(value: object) -> float | None:
    val = safe_float(value, default=None)
    return abs(val) if val is not None else None


def extract_order_id(order: object, default: str | None = None) -> str | None:
    """Extract and normalize a venue order ID from any response shape."""
    if order is None or isinstance(order, bool):
        return default
    if isinstance(order, (list, tuple)):
        for item in order:
            if (cleaned := extract_order_id(item)) is not None:
                return cleaned
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        cleaned = clean_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        return _first_mapped(order, _ORDER_ID_KEYS, extract_order_id, default)
    cleaned = clean_order_id(order)
    return cleaned if cleaned is not None else default


def extract_order_side(order: object) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if type(order) is not dict and not isinstance(order, dict):
        return ""
    for k in ("side", "S", "dir"):
        raw_s = order.get(k)
        if raw_s is not None:
            if type(raw_s) is str:
                fast = _EXACT_SIDE_MAP.get(raw_s)
                if fast is not None:
                    return fast
            if (s_norm := normalize_side(raw_s)):
                return s_norm
    descr = order.get("descr")
    for raw_type in (order.get("type"), descr.get("type") if isinstance(descr, dict) else None):
        if (t_norm := normalize_side(raw_type)) in ("BUY", "SELL"):
            return t_norm
    raw_buyer = order.get("isBuyer")
    if raw_buyer is not None:
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str) and (b_norm := raw_buyer.strip().lower()) in ("b", "buy", "s", "sell"):
            return "BUY" if b_norm.startswith("b") else "SELL"
    inst = order.get("instrument")
    has_broker_hint = "ticker" in order or "limitPrice" in order or (isinstance(inst, dict) and "ticker" in inst)
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        sq = safe_float(order.get(sq_key), default=0.0)
        if sq < 0.0:
            return "SELL"
        if sq > 0.0 and (sq_key in ("signed_qty", "orderedQuantity") or has_broker_hint):
            return "BUY"
    return ""


def extract_order_price(order: object, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, _PRICE_KEYS, lambda v: safe_pos_float(v, default=None))
    if val is not None:
        return val
    descr = order.get("descr")
    if isinstance(descr, dict) and (val := safe_pos_float(descr.get("price"), default=None)) is not None:
        return val
    for cost_key, exec_key in _COST_EXEC_KEY_PAIRS:
        if cost_key in order and exec_key in order:
            cost_val = _pos_abs_float(order.get(cost_key))
            exec_val = _pos_abs_float(order.get(exec_key))
            if cost_val and exec_val and (res := safe_pos_float(cost_val / exec_val, default=None)) is not None:
                return res
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote, total_qty = 0.0, 0.0
        for f in fills:
            if isinstance(f, dict):
                f_px = safe_pos_float(f.get("price", f.get("px", 0.0)), default=None)
                f_qty = _pos_abs_float(f.get("qty", f.get("quantity", f.get("sz", 0.0))))
                if f_px is not None and f_qty is not None:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
        if total_qty > 0 and (res := safe_pos_float(total_quote / total_qty, default=None)) is not None:
            return res
    msg_str = order.get("msg")
    if isinstance(msg_str, str) and "@" in msg_str and msg_str.strip().lower().startswith("filled "):
        if (msg_px := safe_pos_float(msg_str.rsplit("@", 1)[1].strip(), default=None)) is not None:
            return msg_px
    return default


def extract_order_qty(order: object, default: float = 0.0) -> float:
    """Extract order/fill quantity from any venue shape."""
    return _first_mapped(order, _QTY_KEYS, _pos_abs_float, default)


def is_market_order(order: object) -> bool:
    """Return whether ``order`` represents a market order (safe against string flags like ``'false'``)."""
    if not isinstance(order, Mapping):
        return False
    if "market" in order and order.get("market") is not None:
        return bool(_parse_bool_flag(order.get("market"), False))
    descr = order.get("descr")
    for raw_ot in (
        order.get("ordertype"),
        order.get("type"),
        descr.get("ordertype") if isinstance(descr, Mapping) else None,
    ):
        if isinstance(raw_ot, str) and raw_ot.strip().upper() == "MARKET":
            return True
    return False


def clamp_price_to_bbo_mid(
    price: float | None,
    bid: object,
    ask: object,
    *,
    max_spread_ratio: float = 0.05,
) -> float | None:
    """Return ``price`` clamped to the BBO mid ``(bid + ask) / 2`` when ``price`` falls outside a tight ``[bid, ask]`` spread."""
    px_f = safe_pos_float(price, default=None) if not isinstance(price, bool) else None
    bid_f = safe_pos_float(bid, default=None) if not isinstance(bid, bool) else None
    ask_f = safe_pos_float(ask, default=None) if not isinstance(ask, bool) else None
    max_ratio = safe_nonneg_float(max_spread_ratio, default=0.05)
    if (
        bid_f is not None
        and ask_f is not None
        and bid_f <= ask_f
        and (ask_f - bid_f) / bid_f <= max_ratio
        and (px_f is None or px_f < bid_f or px_f > ask_f)
    ):
        return round((bid_f + ask_f) * 0.5, 8)
    return px_f


_CSV_SPLIT_RE = re.compile(r"[,\n]+")


def split_csv(raw: str | None, *, lower: bool = False, upper: bool = False) -> tuple[str, ...]:
    """Split comma- or newline-delimited strings into a deduplicated tuple of stripped tokens."""
    if not isinstance(raw, str) or not raw:
        return ()
    seen: dict[str, None] = {}
    for part in _CSV_SPLIT_RE.split(raw):
        if cleaned := part.strip():
            cleaned = cleaned.lower() if lower else (cleaned.upper() if upper else cleaned)
            seen.setdefault(cleaned, None)
    return tuple(seen)


def _norm_symbol_list_token(raw_val: object) -> str | None:
    s = clean_scalar_str(raw_val)
    if s is None:
        return None
    return s if is_kraken_tokenized_asset(s) else s.upper()


def parse_symbol_list(symbols: object) -> list[str]:
    """Normalize comma-separated string or iterable of symbols into deduplicated canonical symbol list."""
    if isinstance(symbols, str):
        raw_items: object = split_csv(symbols, upper=False)
    elif symbols is None or isinstance(symbols, (bool, bytes, bytearray, dict)) or not hasattr(symbols, "__iter__"):
        return []
    else:
        raw_items = symbols
    seen: dict[str, str] = {}
    for s in raw_items:
        if (tok := _norm_symbol_list_token(s)) is not None:
            key_u = tok.upper()
            if key_u not in seen or is_kraken_tokenized_asset(tok):
                seen[key_u] = tok
    return list(seen.values())


def diff_percent(value1: float, value2: float) -> float:
    """Return symmetric percentage difference relative to the values' absolute mean."""
    v1, v2 = safe_float(value1, default=None), safe_float(value2, default=None)
    if v1 is None or v2 is None:
        return math.nan
    if v1 == v2:
        return 0.0
    denom = (abs(v1) + abs(v2)) / 2.0
    return 0.0 if denom == 0.0 else abs(v1 - v2) / denom * 100.0


def are_close(value1: float, value2: float, tolerance_percent: float = 1.0) -> bool:
    """Return deterministically whether values differ by at most tolerance_percent."""
    tol = safe_nonneg_float(tolerance_percent, default=None)
    if tol is None:
        return False
    diff_f = safe_nonneg_float(diff_percent(value1, value2), default=None)
    return diff_f is not None and diff_f <= tol


def clamp_float(
    val: object, minimum: float = 0.0, maximum: float = 1.0, *, default: float = 0.0
) -> float:
    """Safely coerce ``val`` to a float and clamp it to ``[minimum, maximum]``."""
    v = safe_float(val, default=default)
    return max(minimum, min(maximum, default if v is None else v))


def value_diff_to_percent(value1, value2) -> float:
    v1, v2 = safe_float(value1, default=None), safe_float(value2, default=None)
    if v1 is None or v2 is None or v1 == 0.0:
        return 0.0
    return safe_float(((v1 - v2) / v1) * 100.0, default=0.0)


def profit_diff_pct(order_type: str, price: float, ref_price: float) -> float:
    """Compute signed favorable profit percentage of ``price`` relative to ``ref_price`` symmetrically."""
    side = require_buy_sell_side(order_type, name="order_type")
    buy_pct = value_diff_to_percent(ref_price, price)
    return buy_pct if side == "BUY" else (-buy_pct + 0.0)


def slope(val1, idx1, val2, idx2) -> float:
    v1 = safe_float(val1, default=None)
    i1 = safe_float(idx1, default=None)
    v2 = safe_float(val2, default=None)
    i2 = safe_float(idx2, default=None)
    if v1 is None or i1 is None or v2 is None or i2 is None or i1 == i2:
        return 0.0
    return safe_float((v2 - v1) / (i2 - i1), default=0.0)


def gaussian_weights_from_idx(T, idx: int):
    """Return ``t`` as the position axis ``[idx..T-1]`` and ``w`` as its Gaussian weights."""
    import numpy as np

    t_f = safe_pos_float(T, default=None)
    idx_f = safe_float(idx, default=None)
    if t_f is None or idx_f is None:
        return np.array([]), np.array([])
    t_steps = min(100000, max(1, round(t_f)))
    idx_int = max(0, math.trunc(idx_f))
    if idx_int >= t_steps:
        return np.array([]), np.array([])

    t_full = np.linspace(0, t_steps - 1, t_steps)
    mu = (t_steps - 1) / 2
    sigma = t_steps / 4
    w_full = np.exp(-0.5 * ((t_full - mu) / sigma) ** 2)
    w_full = w_full / w_full.sum()
    return t_full[idx_int:], w_full[idx_int:]
