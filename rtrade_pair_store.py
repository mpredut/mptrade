"""Atomic rtrade round store, separate from the generic order outbox."""
from __future__ import annotations

import json
import hashlib
import os
import threading
import time

from botcore import _parse_bool_flag, safe_nonneg_float, safe_nonneg_int, safe_pos_float
from cachepaths import canonical_symbol_key
from providers.base import _clean_single_order_id, normalize_side
from lock import FileLock
from state_io import atomic_write_json


def rtrade_client_order_id(pair_id, side, kind="limit"):
    raw_pair = str(pair_id if pair_id is not None else "").strip()
    raw_side = normalize_side(side) if side is not None else ""
    raw_kind = str(kind if kind is not None else "limit").strip()
    raw = f"{raw_pair}:{raw_side}:{raw_kind}".encode("utf-8")
    return "RT_" + hashlib.blake2s(raw, digest_size=16).hexdigest()


_INVALID_STORE_STRINGS = {"", "0", "0.0", "-0", "+0", "nan", "null", "none", "inf", "-inf", "+inf"}


class RTradePairStore:
    def __init__(self, path=None, terminal_retention=200):
        self.path = str(path) if path else None
        self.lock_path = (self.path + ".lock") if self.path else None
        self._mem_lock = threading.Lock()
        self._mem_data = {"version": 1, "pairs": {}}
        if isinstance(terminal_retention, (bool, dict, list, tuple, set)) or terminal_retention is None:
            self.terminal_retention = 200
        else:
            self.terminal_retention = safe_nonneg_int(terminal_retention, default=200)

    @staticmethod
    def _clean_store_id(value) -> str:
        res = _clean_single_order_id(value)
        return res if res is not None else ""

    _parse_bool_flag = staticmethod(_parse_bool_flag)

    @staticmethod
    def _finite_positive(value, field):
        parsed = safe_pos_float(value, default=None)
        if parsed is None:
            raise ValueError(f"invalid rtrade pair-store {field}")
        return parsed

    @staticmethod
    def _new_pair_record(symbol, pair_id: str, start_side: str, qty: float, now: float) -> dict:
        return {
            "symbol": str(symbol).strip() if isinstance(symbol, str) else symbol,
            "pair_id": pair_id,
            "start_side": normalize_side(start_side),
            "qty": qty,
            "phase": "reserved",
            "terminal": False,
            "intents": {},
            "state": None,
            "created_ts": now,
            "updated_ts": now,
        }

    @classmethod
    def _validate_intent(cls, pair_id, symbol, key, intent):
        if not isinstance(intent, dict):
            raise ValueError(f"invalid rtrade intent {key!r}: expected an object")
        side = normalize_side(intent.get("side"))
        kind = str(intent.get("kind") or "")
        if side not in {"BUY", "SELL"} or not kind or key != f"{kind}:{side}":
            raise ValueError(f"invalid rtrade intent identity {key!r}")
        raw_coid = intent.get("client_order_id")
        if not cls._clean_store_id(raw_coid):
            raise ValueError(f"invalid rtrade intent client ID {key!r}")
        if intent.get("pair_id") not in (None, pair_id):
            raise ValueError(f"rtrade intent {key!r} belongs to another pair")
        if intent.get("symbol") not in (None, symbol):
            raise ValueError(f"rtrade intent {key!r} belongs to another symbol")
        requested_qty = intent.get("requested_qty", intent.get("qty"))
        cls._finite_positive(requested_qty, f"intent {key!r} quantity")
        requested_price = intent.get("requested_price", intent.get("price"))
        if requested_price is not None:
            cls._finite_positive(requested_price, f"intent {key!r} price")
        order_id = intent.get("order_id")
        if order_id is not None and not cls._clean_store_id(order_id):
            raise ValueError(f"invalid rtrade intent order ID {key!r}")
        intent_id = intent.get("intent_id")
        if intent_id is not None and not cls._clean_store_id(intent_id):
            raise ValueError(f"invalid rtrade tracked intent ID {key!r}")

    @classmethod
    def _validate(cls, data):
        if not isinstance(data, dict):
            raise ValueError("invalid rtrade pair store: root must be an object")
        if (set(data) != {"version", "pairs"}
                or type(data.get("version")) is not int
                or data["version"] != 1):
            raise ValueError("invalid rtrade pair store: expected version 1")
        pairs = data.get("pairs")
        if not isinstance(pairs, dict):
            raise ValueError("invalid rtrade pair store: pairs must be an object")
        for pair_id, record in pairs.items():
            if not isinstance(pair_id, str) or not cls._clean_store_id(pair_id):
                raise ValueError("invalid rtrade pair key")
            if not isinstance(record, dict):
                raise ValueError(f"invalid rtrade pair record {pair_id!r}")
            required = {
                "symbol", "pair_id", "start_side", "qty", "phase",
                "terminal", "intents", "state", "created_ts", "updated_ts",
            }
            if not required.issubset(record):
                raise ValueError(f"incomplete rtrade pair record {pair_id!r}")
            if record.get("pair_id") != pair_id:
                raise ValueError(f"rtrade pair identity mismatch {pair_id!r}")
            symbol = record.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip() or symbol.strip().lower() in _INVALID_STORE_STRINGS:
                raise ValueError(f"invalid rtrade symbol for {pair_id!r}")
            if str(record.get("start_side") or "").upper() not in {"BUY", "SELL"}:
                raise ValueError(f"invalid rtrade start side for {pair_id!r}")
            cls._finite_positive(record.get("qty"), f"pair {pair_id!r} quantity")
            if not isinstance(record.get("phase"), str) or not record["phase"].strip() or record["phase"].strip().lower() in _INVALID_STORE_STRINGS:
                raise ValueError(f"invalid rtrade phase for {pair_id!r}")
            if not isinstance(record.get("terminal"), bool):
                raise ValueError(f"invalid rtrade terminal flag for {pair_id!r}")
            state = record.get("state")
            if state is not None and not isinstance(state, dict):
                raise ValueError(f"invalid rtrade checkpoint for {pair_id!r}")
            if isinstance(state, dict):
                if not state:
                    raise ValueError(f"empty rtrade checkpoint for {pair_id!r}")
                if (not isinstance(state.get("phase"), str)
                        or not state["phase"].strip()
                        or state["phase"].strip().lower() in _INVALID_STORE_STRINGS):
                    raise ValueError(
                        f"invalid rtrade checkpoint phase for {pair_id!r}")
                if state["phase"] != record["phase"]:
                    raise ValueError(
                        f"rtrade checkpoint phase mismatch for {pair_id!r}")
                state_pair_id = state.get("pair_id")
                if state_pair_id is not None and state_pair_id != pair_id:
                    raise ValueError(f"rtrade checkpoint identity mismatch {pair_id!r}")
                if not record["terminal"]:
                    required_state = {
                        "pair_id", "qty", "start_side", "phase", "tickets",
                    }
                    if not required_state.issubset(state):
                        raise ValueError(
                            f"incomplete active rtrade checkpoint {pair_id!r}")
                    cls._finite_positive(
                        state["qty"],
                        f"checkpoint {pair_id!r} quantity")
                    if str(state["start_side"]).upper() not in {"BUY", "SELL"}:
                        raise ValueError(
                            f"invalid checkpoint start side for {pair_id!r}")
                if ("tickets" in state
                        and not isinstance(state["tickets"], list)):
                    raise ValueError(
                        f"invalid rtrade checkpoint tickets for {pair_id!r}")
                ticket_ids = set()
                for ticket in state.get("tickets", []):
                    if not isinstance(ticket, dict):
                        raise ValueError(
                            f"invalid rtrade checkpoint ticket for {pair_id!r}")
                    raw_ticket_oid = ticket.get("order_id")
                    order_id = cls._clean_store_id(raw_ticket_oid)
                    side = str(ticket.get("side") or "").upper()
                    if (not order_id or order_id in ticket_ids
                            or side not in {"BUY", "SELL"}):
                        raise ValueError(
                            f"invalid rtrade checkpoint ticket identity "
                            f"for {pair_id!r}")
                    ticket_ids.add(order_id)
                    cls._finite_positive(
                        ticket.get("price"),
                        f"checkpoint ticket {order_id!r} price")
                    cls._finite_positive(
                        ticket.get("qty"),
                        f"checkpoint ticket {order_id!r} quantity")
                    if ("active" in ticket
                            and not isinstance(ticket["active"], bool)):
                        raise ValueError(
                            f"invalid rtrade checkpoint ticket state "
                            f"for {pair_id!r}")
                    if ticket.get("pair_id") not in (None, pair_id):
                        raise ValueError(
                            f"rtrade checkpoint ticket belongs to another pair")
                if ("snapshots" in state
                        and not isinstance(state["snapshots"], dict)):
                    raise ValueError(
                        f"invalid rtrade checkpoint snapshots for {pair_id!r}")
                for order_id, snapshot in dict(
                        state.get("snapshots") or {}).items():
                    if (not isinstance(order_id, str)
                            or not cls._clean_store_id(order_id)
                            or not isinstance(snapshot, dict)
                            or snapshot.get("status") not in {
                                "open", "closed", "canceled", "expired"}):
                        raise ValueError(
                            f"invalid rtrade checkpoint snapshot "
                            f"for {pair_id!r}")
                    for field in ("filled_qty", "cost", "fee"):
                        if safe_nonneg_float(snapshot.get(field, 0.0), default=None) is None:
                            raise ValueError(
                                f"invalid rtrade checkpoint {field} "
                                f"for {pair_id!r}")
            cls._finite_positive(
                record.get("created_ts"), f"pair {pair_id!r} created_ts")
            cls._finite_positive(
                record.get("updated_ts"), f"pair {pair_id!r} updated_ts")
            intents = record.get("intents")
            if not isinstance(intents, dict):
                raise ValueError(f"invalid rtrade intents for {pair_id!r}")
            for key, intent in intents.items():
                if not isinstance(key, str):
                    raise ValueError(f"invalid rtrade intent key for {pair_id!r}")
                cls._validate_intent(pair_id, symbol, key, intent)
        return data

    def _read(self):
        if not self.path:
            return self._validate(json.loads(json.dumps(self._mem_data)))
        from cachepaths import resolve_cachedb_filepath
        eff_path = str(resolve_cachedb_filepath(self.path))
        try:
            with open(eff_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return self._validate(data)
        except FileNotFoundError:
            return {"version": 1, "pairs": {}}

    def _write(self, data):
        if not self.path:
            self._mem_data = json.loads(json.dumps(data))
            return
        from cachepaths import resolve_cachedb_filepath
        eff_path = str(resolve_cachedb_filepath(self.path))
        atomic_write_json(
            eff_path, data, sort_keys=True, separators=(",", ":"),
        )

    def mutate(self, fn):
        lock_ctx = FileLock(self.lock_path) if self.lock_path else self._mem_lock
        with lock_ctx:
            data = self._read()
            result = fn(data["pairs"])
            if result is not False:
                self._prune_terminal(data["pairs"])
                self._write(self._validate(data))
            return result

    def _prune_terminal(self, pairs):
        terminal = sorted(
            (rec for rec in pairs.values() if isinstance(rec, dict) and rec.get("terminal")),
            key=lambda rec: float(rec.get("updated_ts") or 0.0), reverse=True)
        keep = {rec.get("pair_id") for rec in terminal[:self.terminal_retention]}
        for pair_id, rec in list(pairs.items()):
            if isinstance(rec, dict) and rec.get("terminal") and pair_id not in keep:
                del pairs[pair_id]

    def begin(self, symbol, pair_id, start_side, qty):
        qty_f = self._finite_positive(qty, f"pair {pair_id!r} quantity")
        pid_clean = str(pair_id).strip() if isinstance(pair_id, str) else pair_id
        now = time.time()
        def op(pairs):
            pairs[pid_clean] = self._new_pair_record(
                symbol, pid_clean, start_side, qty_f, now
            )
        self.mutate(op)

    def intent(self, pair_id, side, price, qty, client_order_id, kind="limit",
               symbol=None, start_side=None):
        qty_f = self._finite_positive(qty, f"pair {pair_id!r} quantity")
        price_f = None if price is None else self._finite_positive(price, f"pair {pair_id!r} price")
        coid = self._clean_store_id(client_order_id)
        if not coid:
            raise ValueError(f"invalid rtrade intent client ID for {pair_id!r}")
        side_u = normalize_side(side)
        kind_s = str(kind or "limit").strip()
        pid_clean = str(pair_id).strip() if isinstance(pair_id, str) else pair_id
        def op(pairs):
            now = time.time()
            rec = pairs.get(pid_clean)
            if rec is None:
                if not symbol:
                    raise KeyError(f"unknown pair: {pid_clean}")
                rec = pairs[pid_clean] = self._new_pair_record(
                    symbol, pid_clean, start_side or side_u, qty_f, now
                )
            key = f"{kind_s}:{side_u}"
            value = {
                "side": side_u, "price": price_f, "qty": qty_f,
                "client_order_id": coid, "kind": kind_s,
                "order_id": None,
            }
            if rec["intents"].get(key) == value:
                return False
            rec["intents"][key] = value
            rec["updated_ts"] = now
            return True
        self.mutate(op)

    def persist_intent(self, pair_id, side, kind, pending, *, symbol=None,
                       start_side=None, qty=None):
        """Durably replace one canonical tracked intent, or remove it.

        ``TrackedOrderLifecycle`` calls this function before and after every
        external observation.  The legacy ``price``/``qty`` aliases are retained
        in the JSON so an older checkout can still recover records written by the
        new coordinator.
        """
        side = normalize_side(side)
        kind = str(kind or "").strip()
        pid_clean = str(pair_id).strip() if isinstance(pair_id, str) else pair_id
        key = f"{kind}:{side}"

        def op(pairs):
            now = time.time()
            rec = pairs.get(pid_clean)
            if pending is None:
                if rec is None or key not in rec.get("intents", {}):
                    return False
                del rec["intents"][key]
                rec["updated_ts"] = now
                return True

            if not isinstance(pending, dict):
                raise ValueError("tracked rtrade intent must be a dict")
            value = dict(pending)
            requested_qty = value.get("requested_qty", value.get("qty", qty))
            if requested_qty is None:
                raise ValueError("tracked rtrade intent missing requested_qty")
            requested_qty = self._finite_positive(requested_qty, f"intent {key!r} quantity")
            requested_price = value.get(
                "requested_price", value.get("price"))
            if requested_price is not None:
                requested_price = self._finite_positive(requested_price, f"intent {key!r} price")

            if rec is None:
                if not symbol:
                    raise KeyError(f"unknown pair: {pid_clean}")
                rec = pairs[pid_clean] = self._new_pair_record(
                    symbol, pid_clean, start_side or side, requested_qty, now
                )

            value["side"] = side
            value["kind"] = kind
            value["requested_qty"] = requested_qty
            value["requested_price"] = requested_price
            # Backward-compatible aliases for the old startup recovery code.
            value["qty"] = requested_qty
            value["price"] = requested_price
            if rec.setdefault("intents", {}).get(key) == value:
                return False
            rec["intents"][key] = value
            rec["updated_ts"] = now
            return True

        self.mutate(op)

    def accepted(self, pair_id, side, order_id, kind="limit"):
        clean_oid = self._clean_store_id(order_id)
        if not clean_oid:
            raise ValueError(f"invalid rtrade accepted order_id: {order_id!r}")
        side_u = normalize_side(side)
        kind_s = str(kind or "limit").strip()
        pid_clean = str(pair_id).strip() if isinstance(pair_id, str) else pair_id
        def op(pairs):
            key = f"{kind_s}:{side_u}"
            pairs[pid_clean]["intents"][key]["order_id"] = clean_oid
            pairs[pid_clean]["updated_ts"] = time.time()
        self.mutate(op)

    def checkpoint(self, pair_id, state, terminal=False):
        self.checkpoint_many([(pair_id, state, terminal)])

    def checkpoint_many(self, checkpoints):
        """Use one lock and fsync for every round processed during the tick."""
        checkpoints = list(checkpoints)
        if not checkpoints:
            return
        for pair_id, state, _terminal in checkpoints:
            if not isinstance(state, dict):
                raise ValueError(f"invalid rtrade checkpoint for {pair_id!r}")
        def op(pairs):
            now = time.time()
            for pair_id, state, terminal in checkpoints:
                pid_clean = str(pair_id).strip() if isinstance(pair_id, str) else pair_id
                rec = pairs[pid_clean]
                rec["state"] = state
                rec["phase"] = state.get("phase")
                rec["terminal"] = self._parse_bool_flag(terminal, False)
                rec["updated_ts"] = now
            return True
        self.mutate(op)

    def active(self, symbol):
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return []
        sym_key = canonical_symbol_key(symbol)
        lock_ctx = FileLock(self.lock_path) if self.lock_path else self._mem_lock
        with lock_ctx:
            pairs = self._read().get("pairs", {})
        return [
            rec for rec in pairs.values()
            if canonical_symbol_key(rec.get("symbol")) == sym_key
            and not rec.get("terminal")
        ]
