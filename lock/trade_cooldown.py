"""Trade-specific rapid-fire order gate built on ``lock.Cooldown``.

The generic ``FileLock`` and ``Cooldown`` mechanism lives in ``lock`` and can protect
any operation. This module retains the historical trade-specific API so the
``bapi_placeorder.__place_order`` choke point remains unchanged.

Prevent two orders on the same symbol within ``cooldown_sec`` for any side
combination. The default comes from ``[cooldown] default_sec`` in
``trade_cooldown.conf``. An explicit ``pair_id`` permits exactly one BUY and one SELL in the same
pair without allowing duplicates from another group or process. It is process- and
thread-safe through ``fcntl.flock``.

Flow at the ``__place_order`` choke point:
    with trade_slot(side, symbol) as slot:
        if not slot.allowed: return None     # Blocked by cooldown.
        order = ...submit...
        if order: slot.commit(order_id)       # Success keeps the cooldown active.
        # Failure, exception, or omission releases automatically on exit.
"""
import os
import math
import time
import contextlib
import configparser

from .cooldown import Cooldown
from providers.base import normalize_side as _norm_side

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # = lock/
STATE_FILE = os.path.join(BASE_DIR, "trade_cooldown.json")     # runtime state (gitignored)
LOCK_FILE = os.path.join(BASE_DIR, "trade_cooldown.lock")      # lock fcntl (gitignored)
CONF_FILE = os.path.join(BASE_DIR, "trade_cooldown.conf")      # tracked text/INI configuration


def _load_cooldown_sec() -> int:
    """Strictly read mandatory ``[cooldown] default_sec`` from ``trade_cooldown.conf``."""
    if not os.path.exists(CONF_FILE):
        raise FileNotFoundError(f"Missing mandatory configuration file: {CONF_FILE}")
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    if not cp.read(CONF_FILE, encoding="utf-8"):
        raise ValueError(f"Failed to read mandatory configuration file: {CONF_FILE}")
    if not cp.has_option("cooldown", "default_sec"):
        raise KeyError(f"Missing required [cooldown] default_sec in {CONF_FILE}")
    val = cp.getint("cooldown", "default_sec")
    if val <= 0:
        raise ValueError(f"Invalid [cooldown] default_sec={val!r} in {CONF_FILE}; must be > 0")
    return val


DEFAULT_COOLDOWN_SEC = _load_cooldown_sec()

# Lazy singleton respects test overrides of state and lock paths, rebuilding only when
# those paths change.
_cd = None


def _cooldown():
    global _cd
    if _cd is None or _cd.state_path != STATE_FILE or _cd.lock_path != LOCK_FILE:
        _cd = Cooldown("trade", state_path=STATE_FILE, lock_path=LOCK_FILE)
    return _cd


_INVALID_COOLDOWN_ID_STRINGS = {"", "0", "0.0", "-0", "+0", "-0.0", "+0.0", "none", "null", "nan", "inf", "-inf", "+inf"}


def _clean_pair_id(pair_id):
    if pair_id is None or isinstance(pair_id, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(pair_id, int):
        return str(pair_id) if pair_id > 0 else None
    if isinstance(pair_id, float):
        if not math.isfinite(pair_id) or pair_id <= 0 or not pair_id.is_integer():
            return None
        return str(int(pair_id))
    s = str(pair_id).strip()
    if s.lower() in _INVALID_COOLDOWN_ID_STRINGS or s.startswith("-"):
        return None
    if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
        s = s[:-2]
    if s.startswith("+") and s[1:].isdigit():
        s = s[1:]
    if s.isdigit():
        return str(int(s)) if int(s) > 0 else None
    try:
        f = float(s)
        if not math.isfinite(f) or f <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    return s if s and s.lower() not in _INVALID_COOLDOWN_ID_STRINGS else None


def _valid_pair_id(pair_id) -> bool:
    return _clean_pair_id(pair_id) is not None


def _clean_order_id_meta(val):
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    if isinstance(val, float):
        if not math.isfinite(val) or val <= 0.0 or not val.is_integer():
            return None
        return int(val)
    s = str(val).strip()
    if s.lower() in _INVALID_COOLDOWN_ID_STRINGS or s.startswith("-"):
        return None
    if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
        s = s[:-2]
    if s.startswith("+") and s[1:].isdigit():
        s = s[1:]
    if s.isdigit():
        return str(int(s)) if int(s) > 0 else None
    try:
        f = float(s)
        if not math.isfinite(f) or f <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    return s if s and s.lower() not in _INVALID_COOLDOWN_ID_STRINGS else None





def reserve_trade(side, symbol, cooldown_sec=DEFAULT_COOLDOWN_SEC, client_order_id=None,
                  pair_id=None):
    """Atomically check and reserve placement rights for ``symbol``."""
    clean_cid = _clean_order_id_meta(client_order_id)
    norm_side = _norm_side(side)
    clean_pid = _clean_pair_id(pair_id)
    if clean_pid is not None:
        return _cooldown().reserve_group_member(
            symbol, cooldown_sec, clean_pid, norm_side,
            side="PAIR", symbol=symbol, client_order_id=clean_cid)
    return _cooldown().reserve(symbol, cooldown_sec, side=norm_side, symbol=symbol,
                               client_order_id=clean_cid, binance_order_id=None)


def release_trade(symbol):
    """Release a failed symbol reservation so it no longer blocks cooldown."""
    _cooldown().release(symbol)


def release_pair_leg(symbol, pair_id, side):
    """Release a committed leg only after it is canceled successfully."""
    clean_pid = _clean_pair_id(pair_id)
    if clean_pid is None:
        return False
    return _cooldown().release_group_member(
        symbol, clean_pid, _norm_side(side), keep_group=True)


def update_binance_order_id(symbol, order_id):
    """Attach the Binance order ID after successful placement."""
    _cooldown().update(symbol, binance_order_id=_clean_order_id_meta(order_id))


class _TradeReservation:
    """Adapt positional ``commit(order_id)`` to the reservation's Binance ID field."""

    def __init__(self, res):
        self._res = res

    @property
    def allowed(self):
        return self._res.allowed

    @property
    def info(self):
        return self._res.info

    def commit(self, binance_order_id=None):
        self._res.commit(binance_order_id=_clean_order_id_meta(binance_order_id))


@contextlib.contextmanager
def trade_slot(side, symbol, cooldown_sec=DEFAULT_COOLDOWN_SEC, client_order_id=None,
               pair_id=None):
    """Provide RAII for an exclusive cooldown or one leg of a pair."""
    clean_cid = _clean_order_id_meta(client_order_id)
    norm_side = _norm_side(side)
    clean_pid = _clean_pair_id(pair_id)
    if clean_pid is not None:
        with _cooldown().group_slot(
                symbol, cooldown_sec, clean_pid, norm_side,
                side="PAIR", symbol=symbol, client_order_id=clean_cid) as res:
            yield _TradeReservation(res)
    else:
        with _cooldown().slot(symbol, cooldown_sec, side=norm_side, symbol=symbol,
                              client_order_id=clean_cid, binance_order_id=None) as res:
            yield _TradeReservation(res)


def get_last_trade_age(symbol):
    """Return the last symbol order's age in seconds, or None when absent."""
    return _cooldown().last_age(symbol)


def describe_last_trade(symbol):
    last = _cooldown().get(symbol)
    if not isinstance(last, dict):
        return f"{symbol}: no order recorded"
    raw_ts = last.get("timestamp")
    if raw_ts is None or isinstance(raw_ts, bool):
        return f"{symbol}: no order recorded"
    try:
        ts = float(raw_ts)
    except (TypeError, ValueError):
        return f"{symbol}: no order recorded"
    if not math.isfinite(ts) or ts <= 0:
        return f"{symbol}: no order recorded"
    age = max(0.0, time.time() - ts)
    return (f"{symbol}: ultim={last.get('side')} | age={age:.1f}s | "
            f"proc={last.get('process_name')} | tid={last.get('thread_id')} | "
            f"clientId={last.get('client_order_id')} | binId={last.get('binance_order_id')}")
