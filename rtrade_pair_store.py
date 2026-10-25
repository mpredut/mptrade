"""Atomic rtrade round store, separate from the generic order outbox."""
from __future__ import annotations

import json
import hashlib
import math
import os
import time

from lock import FileLock
from state_io import atomic_write_json


def rtrade_client_order_id(pair_id, side, kind="limit"):
    raw = f"{pair_id}:{str(side).upper()}:{kind}".encode("utf-8")
    return "RT_" + hashlib.blake2s(raw, digest_size=16).hexdigest()


class RTradePairStore:
    def __init__(self, path=None, terminal_retention=200):
        root = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(root, "cachedb", "rtrade_pairs.json")
        self.lock_path = self.path + ".lock"
        self.terminal_retention = max(0, int(terminal_retention))

    @staticmethod
    def _finite_positive(value, field):
        if isinstance(value, bool):
            raise ValueError(f"invalid rtrade pair-store {field}")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid rtrade pair-store {field}") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"invalid rtrade pair-store {field}")
        return parsed

    @classmethod
    def _validate_intent(cls, pair_id, symbol, key, intent):
        if not isinstance(intent, dict):
            raise ValueError(f"invalid rtrade intent {key!r}: expected an object")
        side = str(intent.get("side") or "").upper()
        kind = str(intent.get("kind") or "")
        if side not in {"BUY", "SELL"} or not kind or key != f"{kind}:{side}":
            raise ValueError(f"invalid rtrade intent identity {key!r}")
        if not str(intent.get("client_order_id") or "").strip():
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
        if order_id is not None and not str(order_id).strip():
            raise ValueError(f"invalid rtrade intent order ID {key!r}")
        intent_id = intent.get("intent_id")
        if intent_id is not None and not str(intent_id).strip():
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
            if not isinstance(pair_id, str) or not pair_id.strip():
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
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"invalid rtrade symbol for {pair_id!r}")
            if str(record.get("start_side") or "").upper() not in {"BUY", "SELL"}:
                raise ValueError(f"invalid rtrade start side for {pair_id!r}")
            cls._finite_positive(record.get("qty"), f"pair {pair_id!r} quantity")
            if not isinstance(record.get("phase"), str) or not record["phase"]:
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
                        or not state["phase"]):
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
                    order_id = str(ticket.get("order_id") or "").strip()
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
                            or not order_id.strip()
                            or not isinstance(snapshot, dict)
                            or snapshot.get("status") not in {
                                "open", "closed", "canceled", "expired"}):
                        raise ValueError(
                            f"invalid rtrade checkpoint snapshot "
                            f"for {pair_id!r}")
                    for field in ("filled_qty", "cost", "fee"):
                        raw_val = snapshot.get(field, 0.0)
                        if isinstance(raw_val, bool):
                            raise ValueError(
                                f"invalid rtrade checkpoint {field} "
                                f"for {pair_id!r}")
                        try:
                            value = float(raw_val)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(
                                f"invalid rtrade checkpoint {field} "
                                f"for {pair_id!r}") from exc
                        if not math.isfinite(value) or value < 0:
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
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            return self._validate(data)
        except FileNotFoundError:
            return {"version": 1, "pairs": {}}

    def _write(self, data):
        atomic_write_json(
            self.path, data, sort_keys=True, separators=(",", ":"),
        )

    def mutate(self, fn):
        with FileLock(self.lock_path):
            data = self._read()
            result = fn(data["pairs"])
            if result is not False:
                self._prune_terminal(data["pairs"])
                self._write(self._validate(data))
            return result

    def _prune_terminal(self, pairs):
        terminal = sorted(
            (rec for rec in pairs.values() if rec.get("terminal")),
            key=lambda rec: float(rec.get("updated_ts", 0)), reverse=True)
        keep = {rec.get("pair_id") for rec in terminal[:self.terminal_retention]}
        for pair_id, rec in list(pairs.items()):
            if rec.get("terminal") and pair_id not in keep:
                del pairs[pair_id]

    def begin(self, symbol, pair_id, start_side, qty):
        now = time.time()
        def op(pairs):
            pairs[pair_id] = {
                "symbol": symbol, "pair_id": pair_id, "start_side": start_side,
                "qty": float(qty), "phase": "reserved", "terminal": False,
                "intents": {}, "state": None, "created_ts": now, "updated_ts": now,
            }
        self.mutate(op)

    def intent(self, pair_id, side, price, qty, client_order_id, kind="limit",
               symbol=None, start_side=None):
        def op(pairs):
            now = time.time()
            rec = pairs.get(pair_id)
            if rec is None:
                if not symbol:
                    raise KeyError(f"unknown pair: {pair_id}")
                rec = pairs[pair_id] = {
                    "symbol": symbol, "pair_id": pair_id,
                    "start_side": (start_side or side).upper(),
                    "qty": float(qty), "phase": "reserved", "terminal": False,
                    "intents": {}, "state": None,
                    "created_ts": now, "updated_ts": now,
                }
            key = f"{kind}:{side.upper()}"
            value = {
                "side": side.upper(), "price": price, "qty": float(qty),
                "client_order_id": client_order_id, "kind": kind,
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
        side = str(side).upper()
        kind = str(kind)
        key = f"{kind}:{side}"

        def op(pairs):
            now = time.time()
            rec = pairs.get(pair_id)
            if pending is None:
                if rec is None or key not in rec.get("intents", {}):
                    return False
                del rec["intents"][key]
                rec["updated_ts"] = now
                return True

            value = dict(pending)
            requested_qty = value.get("requested_qty", value.get("qty", qty))
            if requested_qty is None:
                raise ValueError("tracked rtrade intent missing requested_qty")
            requested_qty = float(requested_qty)
            requested_price = value.get(
                "requested_price", value.get("price"))
            if requested_price is not None:
                requested_price = float(requested_price)

            if rec is None:
                if not symbol:
                    raise KeyError(f"unknown pair: {pair_id}")
                rec = pairs[pair_id] = {
                    "symbol": symbol, "pair_id": pair_id,
                    "start_side": (start_side or side).upper(),
                    "qty": requested_qty, "phase": "reserved",
                    "terminal": False, "intents": {}, "state": None,
                    "created_ts": now, "updated_ts": now,
                }

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
        def op(pairs):
            key = f"{kind}:{side.upper()}"
            pairs[pair_id]["intents"][key]["order_id"] = str(order_id)
            pairs[pair_id]["updated_ts"] = time.time()
        self.mutate(op)

    def checkpoint(self, pair_id, state, terminal=False):
        def op(pairs):
            rec = pairs[pair_id]
            rec["state"] = state
            rec["phase"] = state.get("phase")
            rec["terminal"] = bool(terminal)
            rec["updated_ts"] = time.time()
        self.mutate(op)

    def checkpoint_many(self, checkpoints):
        """Use one lock and fsync for every round processed during the tick."""
        checkpoints = list(checkpoints)
        if not checkpoints:
            return
        def op(pairs):
            now = time.time()
            for pair_id, state, terminal in checkpoints:
                rec = pairs[pair_id]
                rec["state"] = state
                rec["phase"] = state.get("phase")
                rec["terminal"] = bool(terminal)
                rec["updated_ts"] = now
            return True
        self.mutate(op)

    def active(self, symbol):
        with FileLock(self.lock_path):
            pairs = self._read().get("pairs", {})
        return [rec for rec in pairs.values()
                if rec.get("symbol") == symbol and not rec.get("terminal")]
