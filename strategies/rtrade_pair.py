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
from typing import Callable, Optional, Protocol

from botcore import (
    _parse_bool_flag,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.base import _clean_single_order_id, normalize_side


_MAX_LIMIT_REVISION = 1_000_000
_clean_pair_id = _clean_single_order_id
_clean_ticket_order_id = _clean_single_order_id


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
        clean_oid = _clean_ticket_order_id(self.order_id)
        if clean_oid is None:
            raise ValueError("ticket rtrade invalid")
        self.order_id = clean_oid
        self.side = normalize_side(self.side)
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("ticket rtrade invalid")
        px = safe_pos_float(self.price, default=None)
        q = safe_pos_float(self.qty, default=None)
        rev = safe_int(self.revision, default=None)
        if (
            px is None
            or q is None
            or rev is None
            or not 0 <= rev <= _MAX_LIMIT_REVISION
            or (isinstance(self.revision, float) and not self.revision.is_integer())
        ):
            raise ValueError("ticket rtrade invalid")
        self.price = px
        self.qty = q
        self.revision = rev
        if self.pair_id is not None:
            clean_pid = _clean_pair_id(self.pair_id)
            if clean_pid is None:
                raise ValueError("ticket rtrade invalid")
            self.pair_id = clean_pid
        self.active = _parse_bool_flag(self.active, default=False)


@dataclass(frozen=True)
class OrderSnapshot:
    status: str = "open"       # open|closed|canceled|expired
    filled_qty: float = 0.0
    cost: float = 0.0
    fee: float = 0.0

    def __post_init__(self):
        norm_status = (
            ""
            if self.status is None or isinstance(self.status, bool)
            else str(self.status).strip().lower()
        )
        if norm_status not in {"open", "closed", "canceled", "expired"}:
            raise ValueError(f"invalid rtrade order status: {self.status}")
        object.__setattr__(self, "status", norm_status)
        for name, value in (
            ("filled_qty", self.filled_qty),
            ("cost", self.cost),
            ("fee", self.fee),
        ):
            f = safe_nonneg_float(value, default=None)
            if f is None:
                raise ValueError("invalid rtrade financial snapshot")
            object.__setattr__(self, name, f)


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
        for name, value in (
            ("adjustment_fraction", self.adjustment_fraction),
            ("quote_ttl_sec", self.quote_ttl_sec),
            ("poll_sec", self.poll_sec),
            ("fast_fill_ratio", self.fast_fill_ratio),
            ("min_edge_fraction", self.min_edge_fraction),
            ("shock_hard_stop_fraction", self.shock_hard_stop_fraction),
            ("hard_stop_fraction", self.hard_stop_fraction),
        ):
            f = safe_float(value, default=None)
            if f is None:
                raise ValueError("the rtrade policy must hold finite values")
            object.__setattr__(self, name, f)
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
            raise ValueError(
                "hard_stop_fraction must be >= shock_hard_stop_fraction and < 1")
        pd = safe_int(self.price_decimals, default=None)
        if pd is None or pd < 0 or (isinstance(self.price_decimals, float) and not self.price_decimals.is_integer()):
            raise ValueError("price_decimals must be a non-negative integer")
        object.__setattr__(self, "price_decimals", pd)


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


def quote_prices(mid: float, adjustment_fraction: float,
                 decimals: int = 4) -> tuple[float, float]:
    """Return bid/ask quotes symmetric around the same market snapshot."""
    mid_f = safe_pos_float(mid, default=None)
    if mid_f is None:
        raise ValueError("mid must be positive")
    adj_f = safe_float(adjustment_fraction, default=None)
    if adj_f is None or not 0 < adj_f < 1:
        raise ValueError("adjustment_fraction must be in (0, 1)")
    dec_i = safe_int(decimals, default=None)
    if dec_i is None or dec_i < 0 or (isinstance(decimals, float) and not decimals.is_integer()):
        raise ValueError("decimals must be a non-negative integer")
    return (
        round(mid_f * (1 - adj_f), dec_i),
        round(mid_f * (1 + adj_f), dec_i),
    )


def anchored_exit_price(exit_side: str, fill_price: float, current_price: float,
                        adjustment_fraction: float, min_edge_fraction: float,
                        decimals: int = 4) -> float:
    """Anchor the target so it cannot fall below cost plus edge or chase a loss.

    The edge formula matches the financial guard:
      SELL: (sell-buy)/sell >= edge  -> sell >= buy/(1-edge)
      BUY:  (sell-buy)/sell >= edge  -> buy <= sell*(1-edge)
    """
    side = normalize_side(exit_side)
    fill_f = safe_pos_float(fill_price, default=None)
    cur_f = safe_pos_float(current_price, default=None)
    if fill_f is None or cur_f is None:
        raise ValueError("the prices must be positive")
    adj_f = safe_float(adjustment_fraction, default=None)
    if adj_f is None or not 0 < adj_f < 1:
        raise ValueError("adjustment_fraction must be in (0, 1)")
    edge_f = safe_float(min_edge_fraction, default=None)
    if edge_f is None or not 0 <= edge_f < 1:
        raise ValueError("min_edge_fraction must be in [0, 1)")
    dec_i = safe_int(decimals, default=None)
    if dec_i is None or dec_i < 0 or (isinstance(decimals, float) and not decimals.is_integer()):
        raise ValueError("decimals must be a non-negative integer")
    if side == "SELL":
        market_target = cur_f * (1 + adj_f)
        floor = fill_f / (1 - edge_f)
        return round(max(market_target, floor), dec_i)
    if side == "BUY":
        market_target = cur_f * (1 - adj_f)
        ceiling = fill_f * (1 - edge_f)
        return round(min(market_target, ceiling), dec_i)
    raise ValueError("exit_side must be BUY or SELL")


class PairCoordinator:
    """State machine for one BUY+SELL round in either direction."""

    def __init__(self, venue: PairVenue, qty: float, policy: PairPolicy, *,
                 start_side: str = "BUY",
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep,
                 pair_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex):
        self.venue = venue
        if isinstance(qty, bool) or qty is None:
            raise ValueError("rtrade qty must be finite and positive")
        try:
            self.qty = float(qty)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rtrade qty must be finite and positive") from exc
        if not math.isfinite(self.qty) or self.qty <= 0:
            raise ValueError("rtrade qty must be finite and positive")
        self.policy = policy
        if start_side is None or isinstance(start_side, bool):
            raise ValueError("start_side must be BUY or SELL")
        self.start_side = normalize_side(start_side)
        if self.start_side not in {"BUY", "SELL"}:
            raise ValueError("start_side must be BUY or SELL")
        self.clock = clock
        self.sleeper = sleeper
        self.pair_id_factory = pair_id_factory
        self.pair_id = ""
        self.started_at = 0.0
        self.first_fill_at: Optional[float] = None
        self.first_fill_side: Optional[str] = None
        self.shock = False
        self.phase = "idle"
        self.reason: Optional[str] = None
        self.tickets: list[OrderTicket] = []
        self.snapshots: dict[str, OrderSnapshot] = {}
        self.stop_ticket: Optional[OrderTicket] = None
        self.limit_revisions = {"BUY": 0, "SELL": 0}
        self.hard_stop_revision = 0
        self.hard_stop_reason: Optional[str] = None

    def start(self, mid: Optional[float] = None,
              pair_id: Optional[str] = None) -> PairOutcome:
        if self.phase not in {"idle", "complete", "expired", "failed", "hard_stop"}:
            raise RuntimeError("the existing round is not terminal")
        raw_mid = mid if mid is not None else (self.venue.current_price() or 0.0)
        if isinstance(raw_mid, bool):
            raise ValueError("the rtrade mid must be finite")
        mid = float(raw_mid)
        if not math.isfinite(mid):
            raise ValueError("the rtrade mid must be finite")
        buy_price, sell_price = quote_prices(
            mid, self.policy.adjustment_fraction, self.policy.price_decimals)
        if pair_id is not None:
            clean_pid = _clean_pair_id(pair_id)
            if clean_pid is None:
                raise ValueError("invalid rtrade pair_id")
            self.pair_id = clean_pid
        else:
            clean_pid = _clean_pair_id(self.pair_id_factory())
            if clean_pid is None:
                raise ValueError("invalid rtrade pair_id")
            self.pair_id = clean_pid
        raw_started = self.clock()
        if isinstance(raw_started, bool) or not math.isfinite(float(raw_started)):
            raise ValueError("the rtrade clock must be finite")
        self.started_at = float(raw_started)
        self.first_fill_at = None
        self.first_fill_side = None
        self.shock = False
        self.reason = None
        self.tickets = []
        self.snapshots = {}
        self.stop_ticket = None
        self.limit_revisions = {"BUY": 0, "SELL": 0}
        self.hard_stop_revision = 0
        self.hard_stop_reason = None

        prices = {"BUY": buy_price, "SELL": sell_price}
        first_side = self.start_side
        second_side = "SELL" if first_side == "BUY" else "BUY"
        try:
            first = self.venue.place_limit(
                first_side, prices[first_side], self.qty, self.pair_id,
                revision=self.limit_revisions[first_side])
        except Exception as exc:  # noqa: BLE001 - Submission may be ambiguous.
            print(
                f"[rtrade_pair] first {first_side} leg raised before its "
                f"response was known: {exc}")
            first = None
            first_exception = True
        else:
            first_exception = False
        if first is None:
            placement_reason = _place_failure_reason(self.venue, first_side)
            if (first_exception or _is_recovery_pending(placement_reason)
                    or bool(getattr(self.venue, "recovery_blocked", False))):
                self.phase = "startup_recovery"
                self.reason = (
                    placement_reason if _is_recovery_pending(placement_reason)
                    else f"{first_side.lower()}_recovery_pending")
                return self.outcome()
            self.phase = "failed"
            self.reason = placement_reason
            return self.outcome()
        self.tickets.append(first)

        try:
            second = self.venue.place_limit(
                second_side, prices[second_side], self.qty, self.pair_id,
                revision=self.limit_revisions[second_side])
        except Exception as exc:  # noqa: BLE001 - The accepted first leg must remain managed.
            print(
                f"[rtrade_pair] second {second_side} leg raised after the "
                f"first leg was accepted: {exc}")
            second = None
            second_exception = True
        else:
            second_exception = False
        if second is None:
            placement_reason = _place_failure_reason(self.venue, second_side)
            _snapshot, cancel_reason = self._cancel_and_reconcile(first)
            recovery_pending = (
                second_exception or _is_recovery_pending(placement_reason)
                or bool(getattr(self.venue, "recovery_blocked", False))
            )
            if cancel_reason is not None:
                # The first order may still be live or may have filled despite a
                # lost cancel response. Keep the round nonterminal and checkpointable.
                self.phase = "startup_recovery"
                self.reason = (
                    (placement_reason if _is_recovery_pending(placement_reason)
                     else f"{second_side.lower()}_recovery_pending")
                    if recovery_pending else placement_reason
                )
                return self.outcome()
            if recovery_pending:
                # The second order may have been accepted despite response loss.
                # Keep its persisted intent active even after the first is canceled.
                self.phase = "startup_recovery"
                self.reason = (
                    placement_reason if _is_recovery_pending(placement_reason)
                    else f"{second_side.lower()}_recovery_pending")
                return self.outcome()
            if abs(self._net_qty()) > 1e-8:
                # A fill won the cancel race. The next step will build the exit
                # from the reconciled net exposure rather than forgetting the leg.
                self.phase = "exposed"
                self.reason = placement_reason
                return self.outcome()
            self.phase = "failed"
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
                None if self.first_fill_at is None
                else max(0.0, self.first_fill_at - self.started_at)),
            "first_fill_side": self.first_fill_side,
            "stop_order_id": (
                None if self.stop_ticket is None else self.stop_ticket.order_id),
            "limit_revisions": dict(self.limit_revisions),
            "hard_stop_revision": self.hard_stop_revision,
            "hard_stop_reason": self.hard_stop_reason,
            "tickets": [vars(ticket).copy() for ticket in self.tickets],
            "snapshots": {
                str(order_id): vars(snapshot).copy()
                for order_id, snapshot in self.snapshots.items()
            },
        }

    @classmethod
    def from_state(cls, venue, policy, state, *,
                   clock=time.monotonic, sleeper=time.sleep):
        required = {"pair_id", "qty", "start_side", "phase", "tickets"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("incomplete rtrade checkpoint")
        obj = cls(venue, state["qty"], policy, start_side=state["start_side"],
                  clock=clock, sleeper=sleeper)
        clean_pair_id = _clean_pair_id(state.get("pair_id"))
        if clean_pair_id is None:
            raise ValueError("invalid rtrade checkpoint pair_id")
        obj.pair_id = clean_pair_id
        raw_phase = state.get("phase")
        if (
            raw_phase is None
            or isinstance(raw_phase, (bool, dict, list, tuple, set, frozenset))
            or not str(raw_phase).strip()
            or str(raw_phase).strip().lower() in {"nan", "none", "null", "inf", "-inf", "+inf"}
        ):
            raise ValueError("invalid rtrade checkpoint phase")
        obj.phase = str(raw_phase).strip()
        obj.reason = state.get("reason")
        obj.shock = _parse_bool_flag(state.get("shock", False), default=False)
        raw_elapsed = state.get("elapsed_sec")
        if raw_elapsed is None:
            elapsed_f = 0.0
        else:
            if isinstance(raw_elapsed, bool):
                raise ValueError("invalid rtrade checkpoint elapsed_sec")
            elapsed_f = float(raw_elapsed)
            if not math.isfinite(elapsed_f):
                raise ValueError("invalid rtrade checkpoint elapsed_sec")
        obj.started_at = clock() - max(0.0, elapsed_f)
        ff_elapsed = state.get("first_fill_elapsed_sec")
        if ff_elapsed is None:
            obj.first_fill_at = None
        else:
            if isinstance(ff_elapsed, bool) or not math.isfinite(float(ff_elapsed)):
                raise ValueError("invalid rtrade checkpoint first_fill_elapsed_sec")
            obj.first_fill_at = obj.started_at + max(0.0, float(ff_elapsed))
        obj.first_fill_side = state.get("first_fill_side")
        raw_tickets = state.get("tickets")
        if not isinstance(raw_tickets, list):
            raise ValueError("invalid rtrade checkpoint tickets")
        try:
            obj.tickets = [
                OrderTicket(**ticket) if isinstance(ticket, dict) else (_ for _ in ()).throw(TypeError("ticket not dict"))
                for ticket in raw_tickets
            ]
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
                clean_oid = _clean_ticket_order_id(order_id)
                if clean_oid is None or not isinstance(snapshot, dict):
                    raise TypeError("invalid snapshot entry")
                cleaned_snaps[clean_oid] = OrderSnapshot(**snapshot)
            obj.snapshots = cleaned_snaps
        except TypeError as exc:
            raise ValueError("invalid rtrade checkpoint snapshots") from exc
        clean_stop_oid = _clean_ticket_order_id(state.get("stop_order_id"))
        obj.stop_ticket = (
            next(
                (ticket for ticket in obj.tickets
                 if ticket.order_id == clean_stop_oid),
                None,
            )
            if clean_stop_oid is not None
            else None
        )
        if obj.stop_ticket is None and obj.phase == "stopping" and obj.tickets:
            # Backward-compatible adoption of checkpoints written before the
            # explicit stop-order field existed: the market exit was appended last.
            obj.stop_ticket = obj.tickets[-1]
        stored_limit_revisions = state.get("limit_revisions")
        if stored_limit_revisions is None:
            stored_limit_revisions = {}
        elif not isinstance(stored_limit_revisions, dict):
            raise ValueError("invalid rtrade checkpoint limit_revisions")
        obj.limit_revisions = {}
        for side in ("BUY", "SELL"):
            raw_rev = stored_limit_revisions.get(side, 0)
            if isinstance(raw_rev, bool):
                raise ValueError(f"invalid {side} limit revision: {raw_rev}")
            try:
                rev_f = 0.0 if raw_rev is None else float(raw_rev)
                if not math.isfinite(rev_f) or rev_f != int(rev_f):
                    raise ValueError(f"invalid {side} limit revision: {raw_rev}")
                revision = int(rev_f)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {side} limit revision: {raw_rev}") from exc
            if not 0 <= revision <= _MAX_LIMIT_REVISION:
                raise ValueError(f"invalid {side} limit revision: {revision}")
            obj.limit_revisions[side] = revision
        raw_hs_rev = state.get("hard_stop_revision", 0)
        if isinstance(raw_hs_rev, bool):
            raise ValueError(f"invalid hard-stop revision: {raw_hs_rev}")
        try:
            hs_rev_f = 0.0 if raw_hs_rev is None else float(raw_hs_rev)
            if not math.isfinite(hs_rev_f) or hs_rev_f != int(hs_rev_f):
                raise ValueError(f"invalid hard-stop revision: {raw_hs_rev}")
            hs_revision = int(hs_rev_f)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid hard-stop revision: {raw_hs_rev}") from exc
        if not 0 <= hs_revision <= _MAX_LIMIT_REVISION:
            raise ValueError(f"invalid hard-stop revision: {hs_revision}")
        obj.hard_stop_revision = hs_revision
        obj.hard_stop_reason = state.get("hard_stop_reason")
        return obj

    def _cancel(self, ticket: OrderTicket) -> bool:
        if not ticket.active:
            return True
        try:
            ok = bool(self.venue.cancel(ticket.order_id))
        except Exception:
            return False
        if ok:
            ticket.active = False
        return ok

    @staticmethod
    def _intent_revision(kind: str, prefix: str) -> Optional[int]:
        kind = str(kind)
        if kind == prefix:
            return 0
        marker = f"{prefix}_"
        if not kind.startswith(marker):
            return None
        suffix = kind[len(marker):]
        if not suffix.isdigit():
            return None
        revision = int(suffix)
        return revision if 0 < revision <= _MAX_LIMIT_REVISION else None

    def _recover_persisted_intents(self) -> bool:
        """Adopt venue-confirmed intents without issuing another submit."""
        recover = getattr(self.venue, "recover_pair_intents", None)
        if not callable(recover):
            return False
        try:
            recovered = recover(self.pair_id)
        except Exception as exc:  # noqa: BLE001 - Recovery must remain fail closed.
            print(
                f"[rtrade_pair] intent recovery remains blocked for "
                f"{self.pair_id}: {exc}")
            return False
        for ticket, snapshot, kind in recovered:
            existing = next(
                (known for known in self.tickets
                 if known.order_id == ticket.order_id),
                None,
            )
            if existing is None:
                ticket.pair_id = self.pair_id
                self.tickets.append(ticket)
                existing = ticket
            existing.active = snapshot.status == "open"
            self.snapshots[existing.order_id] = snapshot
            limit_revision = self._intent_revision(kind, "limit")
            if limit_revision is not None:
                existing.revision = limit_revision
                self.limit_revisions[existing.side] = max(
                    self.limit_revisions[existing.side], limit_revision)
            stop_revision = self._intent_revision(kind, "hard_stop")
            if stop_revision is not None:
                existing.revision = stop_revision
                self.hard_stop_revision = max(
                    self.hard_stop_revision, stop_revision)
                self.stop_ticket = existing
                self.hard_stop_reason = (
                    self.hard_stop_reason or "inventory_hard_stop")
                self.phase = "stopping"
        blocked = getattr(self.venue, "pair_recovery_blocked", None)
        if callable(blocked):
            return not bool(blocked(self.pair_id))
        return not bool(getattr(self.venue, "recovery_blocked", False))

    def _cancel_and_reconcile(self, ticket: OrderTicket):
        """Return the terminal post-cancel snapshot, or a fail-closed reason.

        A successful cancel response is not a quantity snapshot: fills may race
        it. Keep the ticket active until a final venue read proves it terminal.
        """
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
        if not snapshot.status in {"closed", "canceled", "expired"}:
            return None, "status_unconfirmed"
        return snapshot, None

    def _net_qty(self) -> float:
        buy_qty, _bc, _bf, sell_qty, _sc, _sf = self._totals()
        return buy_qty - sell_qty

    def _refresh(self) -> None:
        for ticket in self.tickets:
            snap = self.venue.order_status(ticket.order_id)
            self.snapshots[ticket.order_id] = snap
            # Venue truth overrides both stale checkpoints and an optimistic
            # cancel acknowledgement that has not reached a terminal state.
            ticket.active = snap.status == "open"
        self._compact_zero_fill_history()

    def _compact_zero_fill_history(self) -> None:
        """Remove terminal orders that contributed no inventory or P&L.

        Repricing exposure can create many canceled orders. Retain active orders and
        every filled order required for accounting, but prevent zero-fill cancellations
        from growing memory and checkpoints without bound.
        """
        latest_limit_ticket = {}
        for ticket in reversed(self.tickets):
            if ticket is self.stop_ticket:
                continue
            latest_limit_ticket.setdefault(ticket.side.upper(), ticket)
        has_fill_history = any(
            self.snapshots.get(
                ticket.order_id, OrderSnapshot()).filled_qty > 0
            for ticket in self.tickets)
        kept = []
        for ticket in self.tickets:
            snap = self.snapshots.get(ticket.order_id, OrderSnapshot())
            if (ticket.active or snap.filled_qty > 0
                    or ticket is self.stop_ticket
                    or (has_fill_history
                        and ticket is latest_limit_ticket.get(
                            ticket.side.upper()))
                    or self.phase == "startup_recovery"):
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
        bq, bc, bf = self._side_totals("BUY")
        sq, sc, sf = self._side_totals("SELL")
        return bq, bc, bf, sq, sc, sf

    def _record_first_fill(self, now: float, buy_qty: float, sell_qty: float) -> None:
        if self.first_fill_at is not None or (buy_qty <= 0 and sell_qty <= 0):
            return
        self.first_fill_at = now
        if buy_qty > 0 and sell_qty <= 0:
            self.first_fill_side = "BUY"
        elif sell_qty > 0 and buy_qty <= 0:
            self.first_fill_side = "SELL"
        else:
            self.first_fill_side = "BOTH"
        latency = max(0.0, now - self.started_at)
        self.shock = latency <= self.policy.quote_ttl_sec * self.policy.fast_fill_ratio

    def _active_ticket(self, side: str) -> Optional[OrderTicket]:
        for ticket in reversed(self.tickets):
            if ticket.side.upper() == side.upper() and ticket.active:
                return ticket
        return None

    def _cancel_entry_remainder(self, exposure_side: str) -> bool:
        # Once exposure exists, prevent the same side from increasing it further.
        entry_side = "BUY" if exposure_side == "LONG" else "SELL"
        for ticket in self.tickets:
            if ticket.side.upper() == entry_side and ticket.active:
                _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                if cancel_reason is not None:
                    self.reason = (
                        "entry_cancel_failed"
                        if cancel_reason == "cancel_failed"
                        else "entry_cancel_ambiguous")
                    return False
        return True

    def _has_active_tickets(self) -> bool:
        return any(ticket.active for ticket in self.tickets)

    def _ensure_anchored_exit(self, exposure_side: str, net_qty: float,
                              entry_avg: float, current: float) -> bool:
        exit_side = "SELL" if exposure_side == "LONG" else "BUY"
        target = anchored_exit_price(
            exit_side, entry_avg, current,
            self.policy.adjustment_fraction,
            self.policy.min_edge_fraction,
            self.policy.price_decimals,
        )
        ticket = self._active_ticket(exit_side)
        latest_exit = next(
            (candidate for candidate in reversed(self.tickets)
             if candidate.side.upper() == exit_side
             and candidate is not self.stop_ticket),
            None,
        )
        remaining = 0.0
        if ticket is not None:
            filled = self.snapshots.get(ticket.order_id, OrderSnapshot()).filled_qty
            remaining = max(0.0, ticket.qty - filled)
        adequate_price = bool(ticket and (
            (exit_side == "SELL" and ticket.price >= target) or
            (exit_side == "BUY" and ticket.price <= target)))
        adequate_qty = ticket is not None and math.isclose(
            remaining, abs(net_qty), rel_tol=1e-6, abs_tol=1e-8)
        if adequate_price and adequate_qty:
            return True
        cache_permit = None
        replacement_qty = abs(net_qty)
        if ticket is not None:
            cache_permit = self.venue.preflight_order(
                exit_side, replacement_qty, target, market=False,
                kind="rtrade_pair_quote")
            _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
            if cancel_reason is not None:
                self.reason = (
                    "exit_cancel_failed" if cancel_reason == "cancel_failed"
                    else "exit_cancel_ambiguous")
                return False
            final_net = self._net_qty()
            if abs(final_net) <= 1e-8:
                self.phase = "complete"
                self.reason = None
                return True
            final_exit_side = "SELL" if final_net > 0 else "BUY"
            replacement_qty = abs(final_net)
            current_revision = self.limit_revisions[exit_side]
            if current_revision >= _MAX_LIMIT_REVISION:
                self.reason = "exit_revision_exhausted"
                return False
            # Advance only after the old order is confirmed terminal. If submit
            # later fails, retries and restart reuse this same new revision/CID.
            self.limit_revisions[exit_side] = current_revision + 1
            if (final_exit_side != exit_side
                    or replacement_qty > abs(net_qty) + 1e-8):
                self.reason = "exit_exposure_changed_during_cancel"
                return False
        elif latest_exit is not None:
            latest_snapshot = self.snapshots.get(
                latest_exit.order_id, OrderSnapshot())
            if latest_snapshot.status not in {"closed", "canceled", "expired"}:
                self.reason = "exit_status_unconfirmed"
                return False
            current_revision = self.limit_revisions[exit_side]
            represented_revision = int(latest_exit.revision)
            if represented_revision > current_revision:
                current_revision = represented_revision
                self.limit_revisions[exit_side] = current_revision
            if represented_revision == current_revision:
                if current_revision >= _MAX_LIMIT_REVISION:
                    self.reason = "exit_revision_exhausted"
                    return False
                # The terminal client ID is never reused. Advancing the state
                # first also makes a refused/ambiguous submit reuse this new slot
                # on the next tick instead of incrementing repeatedly.
                self.limit_revisions[exit_side] = current_revision + 1
        replacement = self.venue.place_limit(
            exit_side, target, replacement_qty, self.pair_id,
            cache_permit=cache_permit,
            revision=self.limit_revisions[exit_side])
        if replacement is None:
            self.reason = "exit_place_failed"
            return False
        if not any(
                known.order_id == replacement.order_id
                for known in self.tickets):
            self.tickets.append(replacement)
        return True

    def _submit_hard_stop(self, exposure_side: str, net_qty: float,
                          entry_avg: float, current: float) -> bool:
        if abs(net_qty) <= 1e-8:
            return False
        stop_fraction = (self.policy.shock_hard_stop_fraction if self.shock
                         else self.policy.hard_stop_fraction)
        breached = (
            current <= entry_avg * (1 - stop_fraction)
            if exposure_side == "LONG"
            else current >= entry_avg * (1 + stop_fraction)
        )
        if not breached:
            return False
        loss_fraction = (
            max(0.0, (entry_avg - current) / entry_avg)
            if exposure_side == "LONG"
            else max(0.0, (current - entry_avg) / entry_avg)
        )
        reason = "fast_fill_hard_stop" if self.shock else "inventory_hard_stop"
        allow = getattr(self.venue, "market_exit_allowed", None)
        if callable(allow) and not allow(exposure_side, loss_fraction, reason):
            self.reason = "market_exit_waiting_for_trend"
            return False
        exit_side = "SELL" if exposure_side == "LONG" else "BUY"
        exit_ticket = self._active_ticket(exit_side)
        cache_permit = None
        exit_qty = abs(net_qty)
        if exit_ticket is not None:
            cache_permit = self.venue.preflight_order(
                exit_side, exit_qty, None, market=True,
                kind=f"rtrade:{reason}:{self.pair_id}")
            _snapshot, cancel_reason = self._cancel_and_reconcile(exit_ticket)
            if cancel_reason is not None:
                self.reason = (
                    "hard_stop_cancel_failed"
                    if cancel_reason == "cancel_failed"
                    else "hard_stop_cancel_ambiguous")
                return False
            final_net = self._net_qty()
            if abs(final_net) <= 1e-8:
                self.reason = reason
                self.phase = "hard_stop"
                return True
            final_exit_side = "SELL" if final_net > 0 else "BUY"
            exit_qty = abs(final_net)
            if (final_exit_side != exit_side
                    or exit_qty > abs(net_qty) + 1e-8):
                self.reason = "hard_stop_exposure_changed_during_cancel"
                return False
        try:
            market = self.venue.place_market_exit(
                exit_side, exit_qty,
                reason=reason,
                pair_id=self.pair_id,
                cache_permit=cache_permit,
                revision=self.hard_stop_revision)
        except Exception as exc:  # noqa: BLE001 - Keep residual exposure recoverable.
            print(
                f"[rtrade_pair] hard-stop submit failed for "
                f"{self.pair_id}: {exc}")
            market = None
            stop_exception = True
        else:
            stop_exception = False
        if market is None:
            if (stop_exception
                    or bool(getattr(self.venue, "recovery_blocked", False))
                    or _is_recovery_pending(
                        _place_failure_reason(self.venue, exit_side))):
                self.phase = "hard_stop_recovery"
                self.reason = "hard_stop_recovery_pending"
            else:
                self.reason = "hard_stop_place_failed"
            return False
        if not any(
                known.order_id == market.order_id
                for known in self.tickets):
            self.tickets.append(market)
        self.stop_ticket = market
        self.hard_stop_reason = reason
        self.phase = "stopping"
        return True

    def _retry_terminal_hard_stop(self, net_qty: float) -> bool:
        """Submit one bounded new revision for a terminal stop remainder.

        The original client ID may already identify a canceled or expired venue
        order, so the retry uses revision one. A second terminal remainder is kept
        in explicit nonterminal recovery instead of creating an unbounded order loop.
        """
        if abs(net_qty) <= 1e-8:
            return False
        if self.hard_stop_revision >= 1:
            self.phase = "hard_stop_recovery"
            self.reason = "hard_stop_terminal_with_exposure"
            print(
                f"[rtrade_pair] hard-stop recovery required for {self.pair_id}: "
                f"residual net quantity {net_qty}")
            return False
        exit_side = "SELL" if net_qty > 0 else "BUY"
        exit_qty = abs(net_qty)
        reason = self.hard_stop_reason or (
            "fast_fill_hard_stop" if self.shock else "inventory_hard_stop")
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
            print(
                f"[rtrade_pair] hard-stop revision {next_revision} failed for "
                f"{self.pair_id}: {exc}")
            market = None
        if market is None:
            self.phase = "hard_stop_recovery"
            if (stop_exception
                    or bool(getattr(self.venue, "recovery_blocked", False))
                    or _is_recovery_pending(
                        _place_failure_reason(self.venue, exit_side))):
                self.reason = "hard_stop_recovery_pending"
            else:
                self.reason = "hard_stop_retry_failed"
            return False
        self.hard_stop_revision = next_revision
        if not any(
                known.order_id == market.order_id
                for known in self.tickets):
            self.tickets.append(market)
        self.stop_ticket = market
        self.phase = "stopping"
        self.reason = reason
        return True

    def step(self, now: Optional[float] = None) -> PairOutcome:
        if self.phase in {"complete", "expired", "failed", "hard_stop"}:
            return self.outcome()
        if self.phase == "idle":
            raise RuntimeError("start() must be called before step()")
        raw_now = self.clock() if now is None else now
        if isinstance(raw_now, bool) or not math.isfinite(float(raw_now)):
            raise ValueError("the rtrade clock must be finite")
        now = float(raw_now)
        self._refresh()
        bq, bc, _bf, sq, sc, _sf = self._totals()
        self._record_first_fill(now, bq, sq)
        net = bq - sq

        if self.phase == "startup_recovery":
            if _is_recovery_pending(self.reason):
                if not self._recover_persisted_intents():
                    return self.outcome()
            for ticket in self.tickets:
                if not ticket.active:
                    continue
                _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                if cancel_reason is not None:
                    return self.outcome()
            bq, bc, _bf, sq, sc, _sf = self._totals()
            net = bq - sq
            if abs(net) <= 1e-8:
                self.phase = "failed"
                return self.outcome()
            self.phase = "exposed"

        if self.phase == "hard_stop_recovery":
            if self.reason == "hard_stop_retry_failed":
                self._retry_terminal_hard_stop(net)
                return self.outcome()
            if not _is_recovery_pending(self.reason):
                return self.outcome()
            if not self._recover_persisted_intents():
                return self.outcome()
            bq, bc, _bf, sq, sc, _sf = self._totals()
            net = bq - sq
            if abs(net) <= 1e-8:
                self.phase = "hard_stop"
                self.reason = (
                    self.hard_stop_reason or "inventory_hard_stop")
                return self.outcome()
            if self.stop_ticket is None:
                self.reason = "hard_stop_recovery_missing_order"
                return self.outcome()
            self.phase = "stopping"

        if self.phase == "stopping":
            if abs(net) <= 1e-8:
                self.phase = "hard_stop"
                self.reason = self.hard_stop_reason or (
                    "fast_fill_hard_stop" if self.shock
                    else "inventory_hard_stop"
                )
                return self.outcome()
            stop_snapshot = (
                None if self.stop_ticket is None
                else self.snapshots.get(self.stop_ticket.order_id))
            if (stop_snapshot is not None
                    and stop_snapshot.status in {"closed", "canceled", "expired"}):
                self._retry_terminal_hard_stop(net)
            return self.outcome()

        if bq > 0 or sq > 0:
            if abs(net) <= 1e-8:
                cancel_ambiguous = False
                for ticket in self.tickets:
                    if not ticket.active:
                        continue
                    _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                    cancel_ambiguous = (
                        cancel_ambiguous or cancel_reason is not None)
                # Cancellation may race with a fill or fail. Confirm venue state
                # before declaring the round terminal.
                self._refresh()
                bq, bc, _bf, sq, sc, _sf = self._totals()
                net = bq - sq
                if abs(net) <= 1e-8:
                    if cancel_ambiguous or self._has_active_tickets():
                        self.phase = "closing"
                        self.reason = "balanced_cancel_pending"
                        return self.outcome()
                    self.phase = "complete"
                    self.reason = None
                    return self.outcome()

            exposure_side = "LONG" if net > 0 else "SOLD"
            self.phase = "exposed"
            if not self._cancel_entry_remainder(exposure_side):
                return self.outcome()
            # Reconcile after cancellation because an order may fill during the race.
            self._refresh()
            bq, bc, _bf, sq, sc, _sf = self._totals()
            net = bq - sq
            if abs(net) <= 1e-8:
                self.phase = "closing"
                cancel_ambiguous = False
                for ticket in self.tickets:
                    if not ticket.active:
                        continue
                    _snapshot, cancel_reason = self._cancel_and_reconcile(ticket)
                    cancel_ambiguous = (
                        cancel_ambiguous or cancel_reason is not None)
                self._refresh()
                bq, bc, _bf, sq, sc, _sf = self._totals()
                net = bq - sq
                if abs(net) <= 1e-8:
                    if cancel_ambiguous or self._has_active_tickets():
                        self.reason = "balanced_cancel_pending"
                        return self.outcome()
                    self.phase = "complete"
                    self.reason = None
                    return self.outcome()
            exposure_side = "LONG" if net > 0 else "SOLD"
            entry_avg = (bc / bq) if exposure_side == "LONG" else (sc / sq)
            raw_current = self.venue.current_price()
            try:
                current = 0.0 if raw_current is None or isinstance(raw_current, bool) else float(raw_current)
            except (TypeError, ValueError, OverflowError):
                current = 0.0
            if not math.isfinite(current) or current <= 0:
                self.reason = "current_price_unavailable"
                return self.outcome()
            self.reason = None
            if self._submit_hard_stop(exposure_side, net, entry_avg, current):
                return self.outcome()
            if self.reason in {
                    "hard_stop_cancel_failed",
                    "hard_stop_cancel_ambiguous",
                    "hard_stop_exposure_changed_during_cancel",
                    "hard_stop_place_failed",
                    "hard_stop_recovery_pending"}:
                return self.outcome()
            self._ensure_anchored_exit(exposure_side, net, entry_avg, current)
            return self.outcome()

        if now - self.started_at >= self.policy.quote_ttl_sec:
            for ticket in self.tickets:
                self._cancel(ticket)
            # The final read resolves the fill-versus-cancel race.
            self._refresh()
            bq, _bc, _bf, sq, _sc, _sf = self._totals()
            if bq > 0 or sq > 0:
                return self.step(now=now)
            if self._has_active_tickets():
                self.reason = "quote_cancel_pending"
                return self.outcome()
            self.phase = "expired"
            self.reason = "quote_ttl"
        return self.outcome()

    def run_cycle(self, mid: Optional[float] = None) -> PairOutcome:
        outcome = self.start(mid)
        while not outcome.terminal:
            self.sleeper(self.policy.poll_sec)
            outcome = self.step()
        return outcome

    def outcome(self) -> PairOutcome:
        bq, bc, bf, sq, sc, sf = self._totals()
        buy_avg = bc / bq if bq > 0 else None
        sell_avg = sc / sq if sq > 0 else None
        latency = (self.first_fill_at - self.started_at
                   if self.first_fill_at is not None else None)
        return PairOutcome(
            phase=self.phase,
            pair_id=self.pair_id,
            terminal=self.phase in {"complete", "expired", "failed", "hard_stop"},
            shock=self.shock,
            first_fill_side=self.first_fill_side,
            fill_latency_sec=latency,
            net_qty=bq - sq,
            buy_qty=bq,
            sell_qty=sq,
            buy_avg=buy_avg,
            sell_avg=sell_avg,
            gross_pnl=sc - bc,
            fees=bf + sf,
            reason=self.reason,
        )
