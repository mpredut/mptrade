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
import os
import json
from state_io import atomic_write_json, read_recovering_previous, save_json_state
import time
import socket
import threading
import multiprocessing
import contextlib

from .file_lock import FileLock
from botcore import _parse_bool_flag, safe_float, safe_pos_float


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
    if isinstance(value, float):
        return safe_float(value, default=None)
    if isinstance(value, dict):
        return {k: _sanitize_json_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_json_payload(v) for v in value]
    return value


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
        loaded = read_recovering_previous(self.state_path, json.loads, default=None)
        return loaded if isinstance(loaded, dict) else {}

    def _write(self, state):
        save_json_state(self.state_path, _sanitize_json_payload(state), indent=2)

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
            entry = {
                "timestamp": now,
                "key": key,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "process_name": multiprocessing.current_process().name,
                "hostname": socket.gethostname(),
            }
            for mk, mv in meta.items():
                if mk not in _RESERVED_ENTRY_KEYS:
                    entry[mk] = mv
            state[key] = entry
            self._write(state)
            return True, entry

    def reserve_group_member(self, key, ttl, group_id, member, **meta):
        """Atomically reserve a unique group member for the same key.

        While cooldown is active, only other members of the same group are allowed.
        Repeating a member or using another group is rejected. Legacy state remains
        compatible and exclusive, so a new group cannot bypass it.
        """
        key = _clean_cooldown_key(key)
        group_id = _clean_cooldown_key(group_id)
        member = _clean_cooldown_key(member).upper()
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

            entry = {
                "timestamp": now,
                "key": key,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "process_name": multiprocessing.current_process().name,
                "hostname": socket.gethostname(),
                "group_id": group_id,
                "group_members": [member],
                "group_pending": [member],
                "group_committed": [],
                "group_results": {},
            }
            for mk, mv in meta.items():
                if mk not in _RESERVED_ENTRY_KEYS:
                    entry[mk] = mv
            state[key] = entry
            self._write(state)
            return True, entry

    def commit_group_member(self, key, group_id, member, **fields):
        """Confirm the reserved member and keep its result separately."""
        key = _clean_cooldown_key(key)
        group_id = _clean_cooldown_key(group_id)
        member = _clean_cooldown_key(member).upper()
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
        key = _clean_cooldown_key(key)
        group_id = _clean_cooldown_key(group_id)
        member = _clean_cooldown_key(member).upper()
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
        key = _clean_cooldown_key(key)
        group_id = _clean_cooldown_key(group_id)
        member = _clean_cooldown_key(member).upper()
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
