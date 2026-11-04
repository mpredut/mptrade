"""Generic cooldown gate: at most one operation per key within its TTL.

Atomic check-and-reserve uses ``FileLock``, disk state, and an RAII slot. The lock is
held only around reserve/release, not the protected operation, avoiding long I/O or
deadlocks under the lock. State persists; the lock does not.

    cd = Cooldown("trade", state_path=..., lock_path=...)
    with cd.slot("BTCUSDC", 180, side="BUY") as s:
        if not s.allowed:                 # Blocked by cooldown.
            return
        ...perform operation...
        if ok:
            s.commit(order_id=123)        # Success keeps the cooldown active.
        # Otherwise exit automatically releases the reservation.
"""
import configparser
import contextlib
import multiprocessing
import os
import socket
import sys
import threading
import time

from botcore import _parse_bool_flag, clean_json_value, exact_pos_int, load_ini_config, register_module_aliases, safe_float, safe_pos_float
from cachepaths import resolve_cachedb_filepath
from state_io import read_json_dict, save_json_state

try:
    import fcntl                            # Unix (Linux/WSL)
    _HAVE_FCNTL = True
except ImportError:                         # Windows: gate disabled (no-op)
    _HAVE_FCNTL = False


class FileLock:
    """Cross-process/thread lock using fcntl.flock(LOCK_EX) on an owned fd."""

    def __init__(self, path):
        self.path = resolve_cachedb_filepath(path)
        self._fd = None

    def __enter__(self):
        parent = os.path.dirname(os.fspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fd = open(self.path, "a+")
        if _HAVE_FCNTL:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            except BaseException:
                self._fd.close()
                self._fd = None
                raise
        return self

    def __exit__(self, *exc):
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None



def _is_active_entry(last, now: float, ttl: float) -> bool:
    if not isinstance(last, dict):
        return False
    ts = safe_float(last.get("timestamp"), None)
    ttl_f = safe_float(ttl, None)
    now_f = safe_float(now, None)
    if ts is None or ttl_f is None or now_f is None:
        return False
    age = now_f - ts
    return 0.0 <= age < ttl_f


def _sanitize_json_payload(value):
    return clean_json_value(value, stringify_keys=False)


_as_bool_flag = _parse_bool_flag


def _safe_list(val) -> list:
    if isinstance(val, (list, tuple, set, frozenset)):
        return list(val)
    return []


def _safe_dict(val) -> dict:
    if isinstance(val, dict):
        return dict(val)
    return {}


def _clean_cooldown_key(key) -> str:
    if key is None or isinstance(key, (bool, dict, list, tuple, set, frozenset)):
        return ""
    s = str(key).strip()
    if s.lower() in {"", "none", "null", "nan", "inf", "-inf", "+inf"}:
        return ""
    return s


_RESERVED_ENTRY_KEYS = frozenset({
    "timestamp",
    "key",
    "group_id",
    "group_members",
    "group_pending",
    "group_committed",
    "group_results",
})


class Reservation:
    """Obiectul dat de `Cooldown.slot` (stil RAII / guard C++).
    ``allowed=False`` means cooldown rejected the operation. Commit only after success;
    it retains the reservation and stores extra fields. Exiting without commit rolls back.
    """

    def __init__(self, cooldown, allowed, info, key):
        self._cd = cooldown
        self.allowed = allowed
        self.info = info
        self.key = key
        self._committed = False

    def commit(self, **fields):
        self._committed = True
        if fields:
            self._cd.update(self.key, **fields)


class GroupReservation:
    """Reserve one member of an atomic group.

    For example, this permits exactly one BUY and one SELL from the same quote pair
    through a symbol cooldown while rejecting duplicates and other processes/groups.
    """

    def __init__(self, cooldown, allowed, info, key, group_id, member):
        self._cd = cooldown
        self.allowed = allowed
        self.info = info
        self.key = key
        self.group_id = group_id
        self.member = member
        self._committed = False

    def commit(self, **fields):
        self._committed = True
        self._cd.commit_group_member(
            self.key, self.group_id, self.member, **fields)


class Cooldown:
    """Generic disk-backed rapid-fire gate shared across processes and threads."""

    def __init__(self, name, state_path=None, lock_path=None, base_dir=None):
        self.name = name
        base = base_dir or os.getcwd()
        self.state_path = state_path or os.path.join(base, f"cooldown_{name}.json")
        self.lock_path = lock_path or os.path.join(base, f"cooldown_{name}.lock")

    # ── Storage ──────────────────────────────────────────────────────────────
    def _read(self):
        return read_json_dict(self.state_path, default={}, recover_previous=True)

    def _write(self, state):
        save_json_state(self.state_path, _sanitize_json_payload(state), indent=2)

    @staticmethod
    def _clean_group_args(key, group_id, member):
        return (
            _clean_cooldown_key(key),
            _clean_cooldown_key(group_id),
            _clean_cooldown_key(member).upper(),
        )

    @staticmethod
    def _build_reservation_entry(now, key, meta, **extra):
        entry = {
            "timestamp": now,
            "key": key,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "process_name": multiprocessing.current_process().name,
            "hostname": socket.gethostname(),
            **extra,
        }
        for mk, mv in meta.items():
            if mk not in _RESERVED_ENTRY_KEYS:
                entry[mk] = mv
        return entry

    # ── API ──────────────────────────────────────────────────────────────────
    def reserve(self, key, ttl, **meta):
        """Atomically check and reserve the right to operate on ``key``.

        ``(True, entry)`` means newly reserved; ``(False, last)`` means blocked
        because the prior reservation is younger than the TTL.
        """
        key = _clean_cooldown_key(key)
        if not key:
            return False, {}
        with FileLock(self.lock_path):
            state = self._read()
            last = state.get(key)
            now = time.time()
            if _is_active_entry(last, now, ttl):
                return False, last
            entry = self._build_reservation_entry(now, key, meta)
            state[key] = entry
            self._write(state)
            return True, entry

    def reserve_group_member(self, key, ttl, group_id, member, **meta):
        """Atomically reserve a unique group member for the same key.

        While cooldown is active, only other members of the same group are allowed.
        Repeating a member or using another group is rejected. Legacy state remains
        compatible and exclusive, so a new group cannot bypass it.
        """
        key, group_id, member = self._clean_group_args(key, group_id, member)
        if not key or not group_id or not member:
            return False, {}
        with FileLock(self.lock_path):
            state = self._read()
            last = state.get(key)
            now = time.time()
            active = _is_active_entry(last, now, ttl)
            if active:
                if last.get("group_id") != group_id:
                    return False, last
                members = _safe_list(last.get("group_members"))
                if member in members:
                    return False, last
                members.append(member)
                last["group_members"] = members
                pending = _safe_list(last.get("group_pending"))
                pending.append(member)
                last["group_pending"] = pending
                state[key] = last
                self._write(state)
                return True, last

            entry = self._build_reservation_entry(
                now,
                key,
                meta,
                group_id=group_id,
                group_members=[member],
                group_pending=[member],
                group_committed=[],
                group_results={},
            )
            state[key] = entry
            self._write(state)
            return True, entry

    def commit_group_member(self, key, group_id, member, **fields):
        """Confirm the reserved member and keep its result separately."""
        key, group_id, member = self._clean_group_args(key, group_id, member)
        if not key or not group_id or not member:
            return
        with FileLock(self.lock_path):
            state = self._read()
            entry = state.get(key)
            if not isinstance(entry, dict) or entry.get("group_id") != group_id:
                return
            pending = _safe_list(entry.get("group_pending"))
            if member in pending:
                pending.remove(member)
            committed = _safe_list(entry.get("group_committed"))
            if member not in committed:
                committed.append(member)
            results = _safe_dict(entry.get("group_results"))
            results[member] = dict(fields)
            entry.update({
                "group_pending": pending,
                "group_committed": committed,
                "group_results": results,
                "side": "PAIR",
            })
            state[key] = entry
            self._write(state)

    def rollback_group_member(self, key, group_id, member):
        """Withdraw only the uncommitted member without removing the other leg."""
        key, group_id, member = self._clean_group_args(key, group_id, member)
        if not key or not group_id or not member:
            return
        with FileLock(self.lock_path):
            state = self._read()
            entry = state.get(key)
            if not isinstance(entry, dict) or entry.get("group_id") != group_id:
                return
            if member in _safe_list(entry.get("group_committed")):
                return
            members = [m for m in _safe_list(entry.get("group_members")) if m != member]
            pending = [m for m in _safe_list(entry.get("group_pending")) if m != member]
            if not members:
                del state[key]
            else:
                entry["group_members"] = members
                entry["group_pending"] = pending
                state[key] = entry
            self._write(state)

    def release_group_member(self, key, group_id, member, *, keep_group=True):
        """Explicitly release a committed member after its operation is canceled.

        With ``keep_group=True``, retain the group marker until TTL expiry. The same
        coordinator may replace the leg, but another process/group cannot exploit the
        short interval between cancellation and replacement.
        """
        key, group_id, member = self._clean_group_args(key, group_id, member)
        if not key or not group_id or not member:
            return False
        keep_group = _as_bool_flag(keep_group, True)
        with FileLock(self.lock_path):
            state = self._read()
            entry = state.get(key)
            if not isinstance(entry, dict) or entry.get("group_id") != group_id:
                return False
            entry["group_members"] = [
                m for m in _safe_list(entry.get("group_members")) if m != member]
            entry["group_pending"] = [
                m for m in _safe_list(entry.get("group_pending")) if m != member]
            entry["group_committed"] = [
                m for m in _safe_list(entry.get("group_committed")) if m != member]
            results = _safe_dict(entry.get("group_results"))
            results.pop(member, None)
            entry["group_results"] = results
            if keep_group or entry["group_members"]:
                state[key] = entry
            else:
                del state[key]
            self._write(state)
            return True

    def release(self, key, *, expected_timestamp=None):
        """Release a failed reservation so it no longer blocks the TTL."""
        key = _clean_cooldown_key(key)
        if not key:
            return
        with FileLock(self.lock_path):
            state = self._read()
            if key in state:
                if expected_timestamp is not None:
                    cur_ts = state[key].get("timestamp") if isinstance(state[key], dict) else None
                    if cur_ts != expected_timestamp:
                        return
                del state[key]
                self._write(state)

    def update(self, key, **fields):
        """Add fields, such as a resulting ID, to an existing reservation."""
        key = _clean_cooldown_key(key)
        if not key:
            return
        safe_fields = {k: v for k, v in fields.items() if k not in _RESERVED_ENTRY_KEYS}
        if not safe_fields:
            return
        with FileLock(self.lock_path):
            state = self._read()
            if isinstance(state.get(key), dict):
                state[key].update(safe_fields)
                self._write(state)

    def get(self, key):
        key = _clean_cooldown_key(key)
        if not key:
            return None
        return self._read().get(key)

    def last_age(self, key):
        """Return the latest reservation age in seconds, or None."""
        key = _clean_cooldown_key(key)
        if not key:
            return None
        last = self._read().get(key)
        if not isinstance(last, dict):
            return None
        ts = safe_pos_float(last.get("timestamp"), default=None)
        if ts is None:
            return None
        return max(0.0, time.time() - ts)

    @contextlib.contextmanager
    def slot(self, key, ttl, **meta):
        """Reserve on entry and roll back automatically when exiting without commit.

        The ``fcntl`` lock is held only during reserve/release, not across the body.
        """
        allowed, info = self.reserve(key, ttl, **meta)
        res = Reservation(self, allowed, info, key)
        try:
            yield res
        finally:
            if allowed and not res._committed:
                exp_ts = info.get("timestamp") if isinstance(info, dict) else None
                self.release(key, expected_timestamp=exp_ts)  # No completed operation, so roll back.

    @contextlib.contextmanager
    def group_slot(self, key, ttl, group_id, member, **meta):
        """Provide per-member RAII and rollback for one group member."""
        allowed, info = self.reserve_group_member(
            key, ttl, group_id, member, **meta)
        res = GroupReservation(self, allowed, info, key, group_id, member)
        try:
            yield res
        finally:
            if allowed and not res._committed:
                self.rollback_group_member(key, group_id, member)


# ---------------------------------------------------------------------------
# Trade-specific rapid-fire order gate built on Cooldown
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # = lock/
STATE_FILE = os.path.join(BASE_DIR, "trade_cooldown.json")     # runtime state (gitignored)
LOCK_FILE = os.path.join(BASE_DIR, "trade_cooldown.lock")      # lock fcntl (gitignored)
CONF_FILE = os.path.join(BASE_DIR, "trade_cooldown.conf")      # tracked text/INI configuration


def _load_cooldown_sec() -> int:
    """Strictly read mandatory ``[cooldown] default_sec`` from ``trade_cooldown.conf``."""
    cp = load_ini_config(CONF_FILE, must_exist=True, inline_comments=True)
    if not cp.sections():
        raise ValueError(f"Failed to read mandatory configuration file: {CONF_FILE}")
    if not cp.has_option("cooldown", "default_sec"):
        raise KeyError(f"Missing required [cooldown] default_sec in {CONF_FILE}")
    val = cp.getint("cooldown", "default_sec")
    if val <= 0:
        raise ValueError(f"Invalid [cooldown] default_sec={val!r} in {CONF_FILE}; must be > 0")
    return val


DEFAULT_COOLDOWN_SEC = _load_cooldown_sec()

_cd = None


def _cooldown():
    global _cd
    if _cd is None or _cd.state_path != STATE_FILE or _cd.lock_path != LOCK_FILE:
        _cd = Cooldown("trade", state_path=STATE_FILE, lock_path=LOCK_FILE)
    return _cd


def _clean_pair_id(val):
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        iv = exact_pos_int(val, default=None)
        return str(iv) if iv is not None else None
    s = _clean_cooldown_key(val)
    if not s or s.startswith("-") or s.lower() in {"0", "0.0", "-0", "+0", "+0.0", "-0.0", "infinity", "-infinity", "+infinity"}:
        return None
    if s.endswith(".0") and s[:-2].lstrip("+").isdigit():
        s = s[:-2]
    if s in {"0", "+0"}:
        return None
    return s[1:] if s.startswith("+") and s[1:].isdigit() else s


def _valid_pair_id(pair_id) -> bool:
    return _clean_pair_id(pair_id) is not None


def _clean_order_id_meta(val):
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return exact_pos_int(val, default=None)
    return _clean_pair_id(val)


def _norm_side(side) -> str:
    if not side or isinstance(side, bool):
        return ""
    s = str(side).strip().upper()
    if s in ("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"):
        return "BUY"
    if s in ("A", "S", "SELL", "SOLD", "ASK", "SHORT", "CLOSE LONG", "OPEN SHORT"):
        return "SELL"
    return ""


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
    ts = safe_pos_float(last.get("timestamp"), default=None)
    if ts is None:
        return f"{symbol}: no order recorded"
    age = max(0.0, time.time() - ts)
    return (f"{symbol}: ultim={last.get('side')} | age={age:.1f}s | "
            f"proc={last.get('process_name')} | tid={last.get('thread_id')} | "
            f"clientId={last.get('client_order_id')} | binId={last.get('binance_order_id')}")


register_module_aliases(__name__, "lock.file_lock", "lock.trade_cooldown")

