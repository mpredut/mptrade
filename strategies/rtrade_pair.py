"""Deterministic coordinator for a pair of rtrade quotes.

The module imports no live APIs. Its venue is injected so the same state machine
can be characterized with synthetic fills and then used by the live entry point.
One instance owns both order IDs in a round; the outer scheduler may run multiple
independent instances concurrently.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import uuid
from typing import Any, Callable, Optional, Protocol

from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.base import normalize_side
from providers.order_ohlc_utils import clean_order_id


_MAX_LIMIT_REVISION = 1_000_000
_TERMINAL_PHASES = {"complete", "expired", "failed", "hard_stop"}
_VALID_PHASES = _TERMINAL_PHASES | {
    "idle", "startup_recovery", "quoting", "exposed", "closing", "stopping", "hard_stop_recovery",
}
_TERMINAL_ORDER_STATUSES = {"closed", "canceled", "expired"}


def _parse_nonneg_int(
    val: object,
    *,
    max_val: Optional[int] = None,
    error_msg: str = "must be a non-negative integer",
) -> int:
    iv = exact_nonneg_int(val, default=None)
    if iv is None or (max_val is not None and iv > max_val):
        raise ValueError(error_msg)
    return iv


def _clean_optional_reason(val: object, label: str) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        raise ValueError(f"invalid rtrade checkpoint {label}")
    s = str(val).strip()
    return s if s else None


@dataclass
class OrderTicket:
    order_id: str
    side: str
    price: float
    qty: float
    active: bool = True
    pair_id: Optional[str] = None
    revision: int = 0

    def __post_init__(self):
        clean_oid = clean_order_id(self.order_id)
        self.side = normalize_side(self.side)
        px = safe_pos_float(self.price, default=None)
        q = safe_pos_float(self.qty, default=None)
        if clean_oid is None or self.side not in {"BUY", "SELL"} or px is None or q is None:
            raise ValueError("invalid rtrade ticket")
        self.order_id, self.price, self.qty = clean_oid, px, q
        self.revision = _parse_nonneg_int(
            self.revision, max_val=_MAX_LIMIT_REVISION, error_msg="invalid rtrade ticket"
        )
        if self.pair_id is not None:
            clean_pid = clean_order_id(self.pair_id)
            if clean_pid is None:
                raise ValueError("invalid rtrade ticket")
            self.pair_id = clean_pid
        self.active = _parse_bool_flag(self.active, default=False)


def _coerce_dataclass_float_fields(
    instance: Any, fields: tuple[str, ...], coercer: Callable[..., Optional[float]], err_msg: str
) -> None:
    for name in fields:
        f = coercer(getattr(instance, name), default=None)
        if f is None:
            raise ValueError(err_msg)
        object.__setattr__(instance, name, f)


@dataclass(frozen=True)
class OrderSnapshot:
    status: str = "open"       # open|closed|canceled|expired
    filled_qty: float = 0.0
    cost: float = 0.0
    fee: float = 0.0

    def __post_init__(self):
        norm_status = "" if self.status is None or isinstance(self.status, bool) else str(self.status).strip().lower()
        if norm_status not in {"open", "closed", "canceled", "expired"}:
            raise ValueError(f"invalid rtrade order status: {self.status}")
        object.__setattr__(self, "status", norm_status)
        _coerce_dataclass_float_fields(
            self, ("filled_qty", "cost", "fee"), safe_nonneg_float, "invalid rtrade financial snapshot"
        )


@dataclass(frozen=True)
class PairPolicy:
    adjustment_fraction: float
    quote_ttl_sec: float = 32.0
    poll_sec: float = 1.0
    fast_fill_ratio: float = 0.25
    min_edge_fraction: float = 0.0115
    shock_hard_stop_fraction: float = 0.04
    hard_stop_fraction: float = 0.08
    price_decimals: int = 4

    def __post_init__(self):
        _coerce_dataclass_float_fields(
            self,
            (
                "adjustment_fraction", "quote_ttl_sec", "poll_sec", "fast_fill_ratio",
                "min_edge_fraction", "shock_hard_stop_fraction", "hard_stop_fraction",
            ),
            safe_float,
            "the rtrade policy must hold finite values",
        )
        if not 0 < self.adjustment_fraction < 1:
            raise ValueError("adjustment_fraction must be in (0, 1)")
        if self.quote_ttl_sec <= 0 or self.poll_sec <= 0:
            raise ValueError("quote_ttl_sec and poll_sec must be positive")
        if not 0 < self.fast_fill_ratio <= 1:
            raise ValueError("fast_fill_ratio must be in (0, 1]")
        if not 0 <= self.min_edge_fraction < 1:
            raise ValueError("min_edge_fraction must be in [0, 1)")
        if not 0 < self.shock_hard_stop_fraction < 1:
            raise ValueError("shock_hard_stop_fraction must be in (0, 1)")
        if not self.shock_hard_stop_fraction <= self.hard_stop_fraction < 1:
            raise ValueError("hard_stop_fraction must be >= shock_hard_stop_fraction and < 1")
        object.__setattr__(
            self,
            "price_decimals",
            _parse_nonneg_int(self.price_decimals, error_msg="price_decimals must be a non-negative integer"),
        )


@dataclass(frozen=True)
class PairOutcome:
    phase: str
    pair_id: str
    terminal: bool
    shock: bool = False
    first_fill_side: Optional[str] = None
    fill_latency_sec: Optional[float] = None
    net_qty: float = 0.0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_avg: Optional[float] = None
    sell_avg: Optional[float] = None
    gross_pnl: float = 0.0
    fees: float = 0.0
    reason: Optional[str] = None


class PairVenue(Protocol):
    def current_price(self) -> Optional[float]: ...
    def preflight_order(self, side: str, qty: float, price: Optional[float] = None,
                        *, market: bool = False,
                        kind: Optional[str] = None) -> object: ...
    def place_limit(self, side: str, price: float, qty: float,
                    pair_id: str, *,
                    cache_permit=None,
                    revision: int = 0) -> Optional[OrderTicket]: ...
    def place_market_exit(self, side: str, qty: float,
                          reason: str,
                          pair_id: Optional[str] = None, *,
                          cache_permit=None,
                          revision: int = 0) -> Optional[OrderTicket]: ...
    def market_exit_allowed(self, exposure_side: str, loss_fraction: float,
                            reason: str) -> bool: ...
    def order_status(self, order_id: str) -> OrderSnapshot: ...
    def cancel(self, order_id: str) -> bool: ...


def _is_recovery_pending(reason: Optional[str]) -> bool:
    return str(reason or "").split(":", 1)[0].endswith("_recovery_pending")


def _place_failure_reason(venue: PairVenue, side: str) -> str:
    detail = getattr(venue, "last_place_failure_reason", None)
    if callable(detail):
        reason = detail(side)
        if reason:
            return str(reason)
    return f"{side.lower()}_place_failed"


def _validate_adj_and_decimals(adjustment_fraction: float, decimals: int) -> tuple[float, int]:
    adj_f = safe_float(adjustment_fraction, default=None)
    if adj_f is None or not 0 < adj_f < 1:
        raise ValueError("adjustment_fraction must be in (0, 1)")
    return adj_f, _parse_nonneg_int(decimals, error_msg="decimals must be a non-negative integer")


def quote_prices(mid: float, adjustment_fraction: float,
                 decimals: int = 4) -> tuple[float, float]:
    """Return bid/ask quotes symmetric around the same market snapshot."""
    mid_f = safe_pos_float(mid, default=None)
    if mid_f is None:
        raise ValueError("mid must be positive")
    adj_f, dec_i = _validate_adj_and_decimals(adjustment_fraction, decimals)
    return round(mid_f * (1 - adj_f), dec_i), round(mid_f * (1 + adj_f), dec_i)


def anchored_exit_price(exit_side: str, fill_price: float, current_price: float,
                        adjustment_fraction: float, min_edge_fraction: float,
                        decimals: int = 4) -> float:
    """Anchor the target so it cannot fall below cost plus edge or chase a loss."""
    side = normalize_side(exit_side)
    fill_f, cur_f = safe_pos_float(fill_price, default=None), safe_pos_float(current_price, default=None)
    if fill_f is None or cur_f is None:
        raise ValueError("the prices must be positive")
    adj_f, dec_i = _validate_adj_and_decimals(adjustment_fraction, decimals)
    edge_f = safe_float(min_edge_fraction, default=None)
    if edge_f is None or not 0 <= edge_f < 1:
        raise ValueError("min_edge_fraction must be in [0, 1)")
    if side == "SELL":
        return round(max(cur_f * (1 + adj_f), fill_f / (1 - edge_f)), dec_i)
    if side == "BUY":
        return round(min(cur_f * (1 - adj_f), fill_f * (1 - edge_f)), dec_i)
    raise ValueError("exit_side must be BUY or SELL")


class PairCoordinator:
    """State machine for one BUY+SELL round in either direction."""

    def __init__(self, venue: PairVenue, qty: float, policy: PairPolicy, *,
                 start_side: str = "BUY",
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep,
                 pair_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex):
        self.venue = venue
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            raise ValueError("rtrade qty must be finite and positive")
        self.qty = qty_f
        self.policy = policy
        self.start_side = "" if start_side is None or isinstance(start_side, bool) else normalize_side(start_side)
        if self.start_side not in {"BUY", "SELL"}:
            raise ValueError("start_side must be BUY or SELL")
        self.clock = clock
        self.sleeper = sleeper
        self.pair_id_factory = pair_id_factory
        self.pair_id = ""
        self.phase = "idle"
        self._reset_round_state(0.0)

    def _reset_round_state(self, started_at: float) -> None:
        self.started_at = started_at
        self.first_fill_at: Optional[float] = None
        self.first_fill_side: Optional[str] = None
        self.shock = False
        self.reason: Optional[str] = None
        self.tickets: list[OrderTicket] = []
        self.snapshots: dict[str, OrderSnapshot] = {}
        self.stop_ticket: Optional[OrderTicket] = None
        self.limit_revisions = {"BUY": 0, "SELL": 0}
        self.hard_stop_revision = 0
        self.hard_stop_reason: Optional[str] = None

    def _is_venue_recovery_pending(self, reason: Optional[str], exc_raised: bool = False) -> bool:
        return (
            exc_raised
            or _is_recovery_pending(reason)
            or bool(getattr(self.venue, "recovery_blocked", False))
        )

    def start(self, mid: Optional[float] = None,
              pair_id: Optional[str] = None) -> PairOutcome:
        if self.phase != "idle" and self.phase not in _TERMINAL_PHASES:
            raise RuntimeError("the existing round is not terminal")
        raw_mid = mid if mid is not None else (self.venue.current_price() or 0.0)
        mid = safe_float(raw_mid, default=None)
        if mid is None:
            raise ValueError("the rtrade mid must be finite")
        buy_price, sell_price = quote_prices(
            mid, self.policy.adjustment_fraction, self.policy.price_decimals)
        clean_pid = clean_order_id(pair_id if pair_id is not None else self.pair_id_factory())
        if clean_pid is None:
            raise ValueError("invalid rtrade pair_id")
        self.pair_id = clean_pid
        started_f = safe_float(self.clock(), default=None)
        if started_f is None:
            raise ValueError("the rtrade clock must be finite")
        self._reset_round_state(started_f)

        prices = {"BUY": buy_price, "SELL": sell_price}
        first_side = self.start_side
        second_side = "SELL" if first_side == "BUY" else "BUY"
        try:
            first = self.venue.place_limit(
                first_side, prices[first_side], self.qty, self.pair_id,
                revision=self.limit_revisions[first_side])
            first_exception = False
        except Exception as exc:  # noqa: BLE001 - Submission may be ambiguous.
            print(f"[rtrade_pair] first {first_side} leg raised before its response was known: {exc}")
            first, first_exception = None, True
        if first is None:
            placement_reason = _place_failure_reason(self.venue, first_side)
            if self._is_venue_recovery_pending(placement_reason, first_exception):
                self.phase = "startup_recovery"
                self.reason = (
                    placement_reason if _is_recovery_pending(placement_reason)
                    else f"{first_side.lower()}_recovery_pending"
                )
            else:
                self.phase, self.reason = "failed", placement_reason
            return self.outcome()
        self.tickets.append(first)

        try:
            second = self.venue.place_limit(
                second_side, prices[second_side], self.qty, self.pair_id,
                revision=self.limit_revisions[second_side])
            second_exception = False
        except Exception as exc:  # noqa: BLE001 - The accepted first leg must remain managed.
            print(f"[rtrade_pair] second {second_side} leg raised after the first leg was accepted: {exc}")
            second, second_exception = None, True
        if second is None:
            placement_reason = _place_failure_reason(self.venue, second_side)
            _snapshot, cancel_reason = self._cancel_and_reconcile(first)
            recovery_pending = self._is_venue_recovery_pending(placement_reason, second_exception)
            pend_reason = (
                placement_reason if _is_recovery_pending(placement_reason)
                else f"{second_side.lower()}_recovery_pending"
            )
            if cancel_reason is not None or recovery_pending:
                self.phase = "startup_recovery"
                self.reason = pend_reason if recovery_pending else placement_reason
                return self.outcome()
            self.phase = "exposed" if abs(self._net_qty()) > 1e-8 else "failed"
            self.reason = placement_reason
            return self.outcome()
        self.tickets.append(second)
        self.phase = "quoting"
        return self.outcome()

    def export_state(self) -> dict:
        """Return a JSON-safe checkpoint sufficient for adoption after restart."""
        return {
            "pair_id": self.pair_id, "qty": self.qty,
            "start_side": self.start_side, "phase": self.phase,
            "reason": self.reason, "shock": self.shock,
            "elapsed_sec": max(0.0, self.clock() - self.started_at),
            "first_fill_elapsed_sec": (
                None if self.first_fill_at is None else max(0.0, self.first_fill_at - self.started_at)
            ),
            "first_fill_side": self.first_fill_side,
            "stop_order_id": None if self.stop_ticket is None else self.stop_ticket.order_id,
            "limit_revisions": dict(self.limit_revisions),
            "hard_stop_revision": self.hard_stop_revision,
            "hard_stop_reason": self.hard_stop_reason,
            "tickets": [vars(ticket).copy() for ticket in self.tickets],
            "snapshots": {str(oid): vars(snap).copy() for oid, snap in self.snapshots.items()},
        }

    @classmethod
    def from_state(cls, venue, policy, state, *,
                   clock=time.monotonic, sleeper=time.sleep):
        required = {"pair_id", "qty", "start_side", "phase", "tickets"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("incomplete rtrade checkpoint")
        obj = cls(venue, state["qty"], policy, start_side=state["start_side"],
                  clock=clock, sleeper=sleeper)
        clean_pair_id = clean_order_id(state.get("pair_id"))
        if clean_pair_id is None:
            raise ValueError("invalid rtrade checkpoint pair_id")
        obj.pair_id = clean_pair_id
        raw_phase = state.get("phase")
        if (
            raw_phase is None
            or isinstance(raw_phase, (bool, dict, list, tuple, set, frozenset))
            or not str(raw_phase).strip()
        ):
            raise ValueError("invalid rtrade checkpoint phase")
        norm_phase = str(raw_phase).strip().lower()
        if norm_phase not in _VALID_PHASES:
            raise ValueError("invalid rtrade checkpoint phase")
        obj.phase = norm_phase
        obj.reason = _clean_optional_reason(state.get("reason"), "reason")
        obj.shock = _parse_bool_flag(state.get("shock", False), default=False)
        raw_elapsed = state.get("elapsed_sec")
        elapsed_f = 0.0 if raw_elapsed is None else safe_float(raw_elapsed, default=None)
        if elapsed_f is None:
            raise ValueError("invalid rtrade checkpoint elapsed_sec")
        now_f = safe_float(clock(), default=None)
        if now_f is None:
            raise ValueError("the rtrade clock must be finite")
        obj.started_at = now_f - max(0.0, elapsed_f)
        ff_elapsed = state.get("first_fill_elapsed_sec")
        if ff_elapsed is None:
            obj.first_fill_at = None
        else:
            ff_f = safe_float(ff_elapsed, default=None)
            if ff_f is None:
                raise ValueError("invalid rtrade checkpoint first_fill_elapsed_sec")
            obj.first_fill_at = obj.started_at + max(0.0, ff_f)
        raw_ffs = state.get("first_fill_side")
        if raw_ffs is None:
            obj.first_fill_side = None
        else:
            ffs_u = "" if isinstance(raw_ffs, bool) else str(raw_ffs).strip().upper()
            if ffs_u not in {"BUY", "SELL", "BOTH"}:
                raise ValueError("invalid rtrade checkpoint first_fill_side")
            obj.first_fill_side = ffs_u
        raw_tickets = state.get("tickets")
        if not isinstance(raw_tickets, list) or not all(isinstance(t, dict) for t in raw_tickets):
            raise ValueError("invalid rtrade checkpoint tickets")
        try:
            obj.tickets = [OrderTicket(**ticket) for ticket in raw_tickets]
        except TypeError as exc:
            raise ValueError("invalid rtrade checkpoint tickets") from exc
        raw_snapshots = state.get("snapshots")
        if raw_snapshots is None:
            raw_snapshots = {}
        elif not isinstance(raw_snapshots, dict):
            raise ValueError("invalid rtrade checkpoint snapshots")
        try:
            cleaned_snaps = {}
            for order_id, snapshot in raw_snapshots.items():
                clean_oid = clean_order_id(order_id)
                if clean_oid is None or not isinstance(snapshot, dict):
                    raise TypeError("invalid snapshot entry")
                cleaned_snaps[clean_oid] = OrderSnapshot(**snapshot)
            obj.snapshots = cleaned_snaps
        except TypeError as exc:
            raise ValueError("invalid rtrade checkpoint snapshots") from exc
        clean_stop_oid = clean_order_id(state.get("stop_order_id"))
        obj.stop_ticket = (
            next((t for t in obj.tickets if t.order_id == clean_stop_oid), None)
            if clean_stop_oid is not None else None
        )
        if obj.stop_ticket is None and clean_stop_oid is None and obj.phase == "stopping" and obj.tickets:
            obj.stop_ticket = obj.tickets[-1]
        stored_limit_revisions = state.get("limit_revisions")
        if stored_limit_revisions is None:
            stored_limit_revisions = {}
        elif not isinstance(stored_limit_revisions, dict):
            raise ValueError("invalid rtrade checkpoint limit_revisions")

        def _parse_revision(raw_val, label: str) -> int:
            return 0 if raw_val is None else _parse_nonneg_int(
                raw_val, max_val=_MAX_LIMIT_REVISION, error_msg=f"invalid {label}: {raw_val}"
            )

        obj.limit_revisions = {
            side: _parse_revision(stored_limit_revisions.get(side, 0), f"{side} limit revision")
            for side in ("BUY", "SELL")
        }
        obj.hard_stop_revision = _parse_revision(state.get("hard_stop_revision", 0), "hard-stop revision")
        obj.hard_stop_reason = _clean_optional_reason(state.get("hard_stop_reason"), "hard_stop_reason")
        return obj

    def _append_unique_ticket(self, ticket: OrderTicket) -> OrderTicket:
        for known in self.tickets:
            if known.order_id == ticket.order_id:
                return known
        self.tickets.append(ticket)
        return ticket

    @staticmethod
    def _intent_revision(kind: str, prefix: str) -> Optional[int]:
        kind = str(kind)
        if kind == prefix:
            return 0
        marker = f"{prefix}_"
        if not kind.startswith(marker):
            return None
        suffix = kind[len(marker):]
        if not suffix.isascii() or not suffix.isdigit():
            return None
        revision = exact_nonneg_int(suffix, default=None)
        return revision if revision is not None and 0 < revision <= _MAX_LIMIT_REVISION else None

    def _recover_persisted_intents(self) -> bool:
        """Adopt venue-confirmed intents without issuing another submit."""
        recover = getattr(self.venue, "recover_pair_intents", None)
        if not callable(recover):
            return False
        try:
            recovered = recover(self.pair_id)
        except Exception as exc:  # noqa: BLE001 - Recovery must remain fail closed.
            print(f"[rtrade_pair] intent recovery remains blocked for {self.pair_id}: {exc}")
            return False
        for ticket, snapshot, kind in recovered:
            if not any(known.order_id == ticket.order_id for known in self.tickets):
                ticket.pair_id = self.pair_id
            existing = self._append_unique_ticket(ticket)
            existing.active = snapshot.status == "open"
            self.snapshots[existing.order_id] = snapshot
            limit_revision = self._intent_revision(kind, "limit")
            if limit_revision is not None:
                existing.revision = limit_revision
                self.limit_revisions[existing.side] = max(self.limit_revisions[existing.side], limit_revision)
            stop_revision = self._intent_revision(kind, "hard_stop")
            if stop_revision is not None:
                existing.revision = stop_revision
                if self.stop_ticket is None or stop_revision >= self.hard_stop_revision:
                    self.stop_ticket = existing
                self.hard_stop_revision = max(self.hard_stop_revision, stop_revision)
                self.hard_stop_reason = self.hard_stop_reason or "inventory_hard_stop"
        blocked = getattr(self.venue, "pair_recovery_blocked", None)
        if callable(blocked):
            return not bool(blocked(self.pair_id))
        return not bool(getattr(self.venue, "recovery_blocked", False))

    def _cancel_and_reconcile(self, ticket: OrderTicket):
        """Return the terminal post-cancel snapshot, or a fail-closed reason."""
        try:
            if not self.venue.cancel(ticket.order_id):
                return None, "cancel_failed"
        except Exception:  # noqa: BLE001 - The caller must defer on ambiguity.
            return None, "cancel_failed"
        try:
            snapshot = self.venue.order_status(ticket.order_id)
        except Exception:  # noqa: BLE001 - Do not replace an ambiguous live order.
            ticket.active = True
            return None, "status_unavailable"
        self.snapshots[ticket.order_id] = snapshot
        ticket.active = snapshot.status == "open"
        if snapshot.status not in _TERMINAL_ORDER_STATUSES:
            return None, "status_unconfirmed"
        return snapshot, None

    def _net_qty(self) -> float:
        buy_qty, _bc, _bf, sell_qty, _sc, _sf = self._totals()
        return buy_qty - sell_qty

    def _refresh(self) -> None:
        for ticket in self.tickets:
            try:
                snap = self.venue.order_status(ticket.order_id)
            except Exception:  # noqa: BLE001 - Preserve fail-closed active tracking on transient status failure.
                ticket.active = True
                continue
            self.snapshots[ticket.order_id] = snap
            ticket.active = snap.status == "open"
        latest_limit_ticket = {}
        for ticket in reversed(self.tickets):
            if ticket is not self.stop_ticket:
                latest_limit_ticket.setdefault(ticket.side.upper(), ticket)
        has_fill_history = any(
            self.snapshots.get(ticket.order_id, OrderSnapshot()).filled_qty > 0
            for ticket in self.tickets
        )
        kept = []
        for ticket in self.tickets:
            snap = self.snapshots.get(ticket.order_id, OrderSnapshot())
            if (
                ticket.active or snap.filled_qty > 0
                or ticket is self.stop_ticket
                or (has_fill_history and ticket is latest_limit_ticket.get(ticket.side.upper()))
                or self.phase == "startup_recovery"
            ):
                kept.append(ticket)
            else:
                self.snapshots.pop(ticket.order_id, None)
        self.tickets = kept

    def _side_totals(self, side: str) -> tuple[float, float, float]:
        qty = cost = fee = 0.0
        for ticket in self.tickets:
            if ticket.side.upper() != side.upper():
                continue
            snap = self.snapshots.get(ticket.order_id, OrderSnapshot())
            fq = max(0.0, snap.filled_qty)
            qty += fq
            cost += max(0.0, snap.cost) if snap.cost > 0 else (fq * ticket.price if fq > 0 else 0.0)
            fee += max(0.0, snap.fee)
        return qty, cost, fee

    def _totals(self):
        return (*self._side_totals("BUY"), *self._side_totals("SELL"))

    def _record_first_fill(self, now: float, buy_qty: float, sell_qty: float) -> None:
        if self.first_fill_at is not None or (buy_qty <= 0 and sell_qty <= 0):
            return
        self.first_fill_at = now
        self.first_fill_side = (
            "BUY" if (buy_qty > 0 and sell_qty <= 0) else
            "SELL" if (sell_qty > 0 and buy_qty <= 0) else "BOTH"
        )
        self.shock = max(0.0, now - self.started_at) <= self.policy.quote_ttl_sec * self.policy.fast_fill_ratio

    def _retry_terminal_hard_stop(self, net_qty: float) -> bool:
        """Submit one bounded new revision for a terminal stop remainder."""
        if abs(net_qty) <= 1e-8:
            return False
        if self.hard_stop_revision >= 1:
            self.phase, self.reason = "hard_stop_recovery", "hard_stop_terminal_with_exposure"
            print(f"[rtrade_pair] hard-stop recovery required for {self.pair_id}: residual net quantity {net_qty}")
            return False
        exit_side = "SELL" if net_qty > 0 else "BUY"
        exit_qty = abs(net_qty)
        reason = self.hard_stop_reason or ("fast_fill_hard_stop" if self.shock else "inventory_hard_stop")
        next_revision = self.hard_stop_revision + 1
        kind = f"rtrade:{reason}:{self.pair_id}:revision:{next_revision}"
        stop_exception = False
        try:
            cache_permit = self.venue.preflight_order(
                exit_side, exit_qty, None, market=True, kind=kind)
            try:
                market = self.venue.place_market_exit(
                    exit_side, exit_qty, reason=reason, pair_id=self.pair_id,
                    cache_permit=cache_permit, revision=next_revision)
            except Exception:
                stop_exception = True
                raise
        except Exception as exc:  # noqa: BLE001 - Keep residual exposure recoverable.
            print(f"[rtrade_pair] hard-stop revision {next_revision} failed for {self.pair_id}: {exc}")
            market = None
        if market is None:
            self.phase = "hard_stop_recovery"
            self.reason = (
                "hard_stop_recovery_pending"
                if self._is_venue_recovery_pending(_place_failure_reason(self.venue, exit_side), stop_exception)
                else "hard_stop_retry_failed"
            )
            return False
        self.hard_stop_revision = next_revision
        self._append_unique_ticket(market)
        self.stop_ticket = market
        self.phase, self.reason = "stopping", reason
        return True

    def _totals_and_net(self, *, refresh: bool = False) -> tuple[float, float, float, float, float]:
        if refresh:
            self._refresh()
        bq, bc, _bf, sq, sc, _sf = self._totals()
        return bq, bc, sq, sc, bq - sq

    def _bump_limit_revision(self, exit_side: str) -> bool:
        current_revision = self.limit_revisions[exit_side]
        if current_revision >= _MAX_LIMIT_REVISION:
            self.reason = "exit_revision_exhausted"
            return False
        self.limit_revisions[exit_side] = current_revision + 1
        return True

    def _cancel_active_exit_ticket(
        self,
        ticket: OrderTicket,
        exit_side: str,
        net: float,
        *,
        prefix: str,
        flat_phase: str,
        flat_reason: Optional[str],
        bump_revision: bool = False,
    ) -> Optional[float]:
        _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
        if cancel_reason is not None:
            self.reason = f"{prefix}_cancel_failed" if cancel_reason == "cancel_failed" else f"{prefix}_cancel_ambiguous"
            return None
        final_net = self._net_qty()
        if abs(final_net) <= 1e-8:
            self.phase, self.reason = flat_phase, flat_reason
            return None
        final_exit_side = "SELL" if final_net > 0 else "BUY"
        rem_qty = abs(final_net)
        if bump_revision and not self._bump_limit_revision(exit_side):
            return None
        if final_exit_side != exit_side or rem_qty > abs(net) + 1e-8:
            self.reason = f"{prefix}_exposure_changed_during_cancel"
            return None
        return rem_qty

    def _try_complete_flat_exposure(self) -> tuple[bool, float, float, float, float, float]:
        cancel_ambiguous = False
        for ticket in self.tickets:
            if ticket.active:
                _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                cancel_ambiguous = cancel_ambiguous or cancel_reason is not None
        bq, bc, sq, sc, net = self._totals_and_net(refresh=True)
        if abs(net) <= 1e-8:
            if cancel_ambiguous or any(ticket.active for ticket in self.tickets):
                self.phase, self.reason = "closing", "balanced_cancel_pending"
            else:
                self.phase, self.reason = "complete", None
            return True, bq, bc, sq, sc, net
        return False, bq, bc, sq, sc, net

    def step(self, now: Optional[float] = None) -> PairOutcome:
        if self.phase in _TERMINAL_PHASES:
            return self.outcome()
        if self.phase == "idle":
            raise RuntimeError("start() must be called before step()")
        now = safe_float(self.clock() if now is None else now, default=None)
        if now is None:
            raise ValueError("the rtrade clock must be finite")
        bq, bc, sq, sc, net = self._totals_and_net(refresh=True)
        if self.phase != "startup_recovery":
            self._record_first_fill(now, bq, sq)

        if self.phase == "startup_recovery":
            if _is_recovery_pending(self.reason) and not self._recover_persisted_intents():
                return self.outcome()
            for ticket in self.tickets:
                if ticket.active and self._cancel_and_reconcile(ticket)[1] is not None:
                    return self.outcome()
            bq, bc, sq, sc, net = self._totals_and_net()
            self._record_first_fill(now, bq, sq)
            if abs(net) <= 1e-8:
                if bq > 0 and sq > 0:
                    self.phase = "complete"
                    self.reason = None
                else:
                    self.phase = "failed"
                return self.outcome()
            self.phase = "exposed"

        if self.phase == "exposed" and _is_recovery_pending(self.reason):
            if not self._recover_persisted_intents():
                return self.outcome()
            self.reason = None
            bq, bc, sq, sc, net = self._totals_and_net()

        if self.phase == "hard_stop_recovery":
            if self.reason == "hard_stop_retry_failed":
                self._retry_terminal_hard_stop(net)
                return self.outcome()
            if not _is_recovery_pending(self.reason) or not self._recover_persisted_intents():
                return self.outcome()
            bq, bc, sq, sc, net = self._totals_and_net()
            if abs(net) <= 1e-8:
                self.phase = "hard_stop"
                self.reason = self.hard_stop_reason or "inventory_hard_stop"
                return self.outcome()
            if self.stop_ticket is None:
                self.reason = "hard_stop_recovery_missing_order"
                return self.outcome()
            self.phase = "stopping"

        if self.phase == "stopping":
            if abs(net) <= 1e-8:
                self.phase = "hard_stop"
                self.reason = self.hard_stop_reason or (
                    "fast_fill_hard_stop" if self.shock else "inventory_hard_stop"
                )
                return self.outcome()
            stop_snapshot = None if self.stop_ticket is None else self.snapshots.get(self.stop_ticket.order_id)
            if stop_snapshot is not None and stop_snapshot.status in _TERMINAL_ORDER_STATUSES:
                self._retry_terminal_hard_stop(net)
            return self.outcome()

        if bq > 0 or sq > 0:
            for closing_phase in (False, True):
                if abs(net) <= 1e-8:
                    if closing_phase:
                        self.phase = "closing"
                    settled, bq, bc, sq, sc, net = self._try_complete_flat_exposure()
                    if settled:
                        return self.outcome()
                if closing_phase:
                    break
                exposure_side = "LONG" if net > 0 else "SOLD"
                self.phase = "exposed"
                entry_side = "BUY" if exposure_side == "LONG" else "SELL"
                for ticket in self.tickets:
                    if ticket.side.upper() == entry_side and ticket.active:
                        _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                        if cancel_reason is not None:
                            self.reason = "entry_cancel_failed" if cancel_reason == "cancel_failed" else "entry_cancel_ambiguous"
                            return self.outcome()
                bq, bc, sq, sc, net = self._totals_and_net(refresh=True)
            exposure_side = "LONG" if net > 0 else "SOLD"
            entry_avg = (bc / bq) if exposure_side == "LONG" else (sc / sq)
            current = safe_pos_float(self.venue.current_price(), default=0.0)
            if current <= 0:
                self.reason = "current_price_unavailable"
                return self.outcome()
            self.reason = None
            stop_fraction = self.policy.shock_hard_stop_fraction if self.shock else self.policy.hard_stop_fraction
            breached = (
                current <= entry_avg * (1 - stop_fraction)
                if exposure_side == "LONG" else current >= entry_avg * (1 + stop_fraction)
            )
            if abs(net) > 1e-8 and breached:
                loss_fraction = max(
                    0.0, ((entry_avg - current) if exposure_side == "LONG" else (current - entry_avg)) / entry_avg
                )
                hs_reason = "fast_fill_hard_stop" if self.shock else "inventory_hard_stop"
                allow = getattr(self.venue, "market_exit_allowed", None)
                if callable(allow) and not allow(exposure_side, loss_fraction, hs_reason):
                    self.reason = "market_exit_waiting_for_trend"
                else:
                    exit_side = "SELL" if exposure_side == "LONG" else "BUY"
                    exit_ticket = next(
                        (t for t in reversed(self.tickets) if t.side.upper() == exit_side and t.active),
                        None,
                    )
                    cache_permit = None
                    exit_qty = abs(net)
                    if exit_ticket is not None:
                        cache_permit = self.venue.preflight_order(
                            exit_side, exit_qty, None, market=True, kind=f"rtrade:{hs_reason}:{self.pair_id}"
                        )
                        rem_qty = self._cancel_active_exit_ticket(
                            exit_ticket, exit_side, net,
                            prefix="hard_stop", flat_phase="hard_stop", flat_reason=hs_reason,
                        )
                        if rem_qty is None:
                            return self.outcome()
                        exit_qty = rem_qty
                    try:
                        market = self.venue.place_market_exit(
                            exit_side, exit_qty, reason=hs_reason, pair_id=self.pair_id,
                            cache_permit=cache_permit, revision=self.hard_stop_revision)
                        stop_exception = False
                    except Exception as exc:  # noqa: BLE001 - Keep residual exposure recoverable.
                        print(f"[rtrade_pair] hard-stop submit failed for {self.pair_id}: {exc}")
                        market, stop_exception = None, True
                    if market is None:
                        if self._is_venue_recovery_pending(_place_failure_reason(self.venue, exit_side), stop_exception):
                            self.phase, self.reason = "hard_stop_recovery", "hard_stop_recovery_pending"
                        else:
                            self.reason = "hard_stop_place_failed"
                        return self.outcome()
                    self._append_unique_ticket(market)
                    self.stop_ticket = market
                    self.hard_stop_reason = hs_reason
                    self.phase = "stopping"
                    return self.outcome()
            exit_side = "SELL" if exposure_side == "LONG" else "BUY"
            target = anchored_exit_price(
                exit_side, entry_avg, current,
                self.policy.adjustment_fraction, self.policy.min_edge_fraction, self.policy.price_decimals,
            )
            ticket = next(
                (t for t in reversed(self.tickets) if t.side.upper() == exit_side and t.active),
                None,
            )
            latest_exit = next(
                (c for c in reversed(self.tickets) if c.side.upper() == exit_side and c is not self.stop_ticket),
                None,
            )
            remaining = (
                max(0.0, ticket.qty - self.snapshots.get(ticket.order_id, OrderSnapshot()).filled_qty)
                if ticket is not None else 0.0
            )
            adequate_price = bool(ticket and (
                (exit_side == "SELL" and ticket.price >= target) or (exit_side == "BUY" and ticket.price <= target)
            ))
            adequate_qty = ticket is not None and math.isclose(remaining, abs(net), rel_tol=1e-6, abs_tol=1e-8)
            if adequate_price and adequate_qty:
                return self.outcome()
            cache_permit = None
            replacement_qty = abs(net)
            if ticket is not None:
                cache_permit = self.venue.preflight_order(
                    exit_side, replacement_qty, target, market=False, kind="rtrade_pair_quote"
                )
                rem_qty = self._cancel_active_exit_ticket(
                    ticket, exit_side, net,
                    prefix="exit", flat_phase="complete", flat_reason=None, bump_revision=True,
                )
                if rem_qty is None:
                    return self.outcome()
                replacement_qty = rem_qty
            elif latest_exit is not None:
                latest_snapshot = self.snapshots.get(latest_exit.order_id, OrderSnapshot())
                if latest_snapshot.status not in _TERMINAL_ORDER_STATUSES:
                    self.reason = "exit_status_unconfirmed"
                    return self.outcome()
                latest_rev = exact_nonneg_int(latest_exit.revision, default=0) or 0
                current_revision = max(self.limit_revisions[exit_side], latest_rev)
                self.limit_revisions[exit_side] = current_revision
                if latest_rev == current_revision and not self._bump_limit_revision(exit_side):
                    return self.outcome()
            try:
                replacement = self.venue.place_limit(
                    exit_side, target, replacement_qty, self.pair_id,
                    cache_permit=cache_permit, revision=self.limit_revisions[exit_side]
                )
                exit_exception = False
            except Exception as exc:  # noqa: BLE001 - Keep exposed exit order recoverable.
                print(f"[rtrade_pair] replacement {exit_side} leg raised for {self.pair_id}: {exc}")
                replacement, exit_exception = None, True
            if replacement is None:
                placement_reason = _place_failure_reason(self.venue, exit_side)
                if self._is_venue_recovery_pending(placement_reason, exit_exception):
                    self.reason = (
                        placement_reason if _is_recovery_pending(placement_reason)
                        else f"{exit_side.lower()}_recovery_pending"
                    )
                else:
                    self.reason = "exit_place_failed"
                return self.outcome()
            self._append_unique_ticket(replacement)
            return self.outcome()

        if now - self.started_at >= self.policy.quote_ttl_sec:
            for ticket in self.tickets:
                if ticket.active:
                    try:
                        self.venue.cancel(ticket.order_id)
                    except Exception:
                        pass
            self._refresh()
            bq, _bc, _bf, sq, _sc, _sf = self._totals()
            if bq > 0 or sq > 0:
                return self.step(now=now)
            if any(ticket.active for ticket in self.tickets):
                self.reason = "quote_cancel_pending"
                return self.outcome()
            self.phase, self.reason = "expired", "quote_ttl"
        return self.outcome()

    def run_cycle(self, mid: Optional[float] = None) -> PairOutcome:
        outcome = self.start(mid)
        while not outcome.terminal:
            self.sleeper(self.policy.poll_sec)
            outcome = self.step()
        return outcome

    def outcome(self) -> PairOutcome:
        bq, bc, bf, sq, sc, sf = self._totals()
        matched_q = min(bq, sq)
        realized_gross_pnl = matched_q * ((sc / sq) - (bc / bq)) if (matched_q > 0 and bq > 0 and sq > 0) else 0.0
        return PairOutcome(
            phase=self.phase,
            pair_id=self.pair_id,
            terminal=self.phase in _TERMINAL_PHASES,
            shock=self.shock,
            first_fill_side=self.first_fill_side,
            fill_latency_sec=(self.first_fill_at - self.started_at if self.first_fill_at is not None else None),
            net_qty=bq - sq,
            buy_qty=bq,
            sell_qty=sq,
            buy_avg=bc / bq if bq > 0 else None,
            sell_avg=sc / sq if sq > 0 else None,
            gross_pnl=realized_gross_pnl,
            fees=bf + sf,
            reason=self.reason,
        )
