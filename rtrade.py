import os
import time
import datetime
import math
import uuid
from collections import deque
import threading
from decimal import Decimal, ROUND_DOWN
from concurrent.futures import ThreadPoolExecutor, wait

# Financial policy and operational invariants are documented in docs/RTRADE.md.
# Effective configuration remains in config.env.


# my imports
import log
import alertnotifiers as alert
import utils as u
import symbols as sym
from instrument_registry import single_symbol_for
from binance_api import bapi as api
from binance_api import bapi_placeorder as po   # Retained for the dead-safe WeightLimitBlock path.
from providers.market_api import api as mkt      # Single guarded Instrument.place proxy.
from providers.execution_audit import AuditedStrategyExecutor
from providers.quantity import decide_quantity
from providers.strategy_executor import ProviderError, SubmissionRefused
from market_regime import MarketRegimeDecision
from order_retry import (
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
    propagate_submission_refusal,
)
from rtrade_pair_store import RTradePairStore, rtrade_client_order_id
from strategies.rtrade_pair import (
    OrderSnapshot as PairOrderSnapshot,
    OrderTicket as PairOrderTicket,
    PairCoordinator,
    PairPolicy,
)


_RTRADE_HEARTBEAT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cachedb", "rtrade.heartbeat")
_RTRADE_HEARTBEAT_INTERVAL_SEC = 30.0
_rtrade_heartbeat_lock = threading.Lock()
_rtrade_heartbeat_last = float("-inf")


def _touch_rtrade_heartbeat(*, force=False, now=None):
    """Publish actual coordinator progress independently from buffered stdout.

    The healthcheck must not infer liveness from ``rtrade.log``: normal trend and
    placement backoffs can make that log quiet, while redirection can delay writes.
    Touching from the bounded coordinator loop still detects a blocked API call or
    dead loop, unlike a separate heartbeat thread that could outlive the work.
    """
    global _rtrade_heartbeat_last
    now = time.monotonic() if now is None else float(now)
    with _rtrade_heartbeat_lock:
        if not force and now - _rtrade_heartbeat_last < _RTRADE_HEARTBEAT_INTERVAL_SEC:
            return
        try:
            os.makedirs(os.path.dirname(_RTRADE_HEARTBEAT_PATH), exist_ok=True)
            with open(_RTRADE_HEARTBEAT_PATH, "a", encoding="utf-8"):
                pass
            os.utime(_RTRADE_HEARTBEAT_PATH, None)
        except OSError:
            # Trading must not stop because an observational heartbeat cannot be
            # written. The healthcheck will surface the stale/missing file.
            return
        _rtrade_heartbeat_last = now


# Load versioned, non-secret tuning parameters before reading the environment below.
# ``botcore.load_dotenv`` does not overwrite variables already set in the real environment.
from botcore import (load_dotenv as _load_dotenv, required_bool_env,
                     required_env, required_float_env, required_int_env)
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))

# Seconds between cancel-and-recreate attempts.
WAIT_FOR_ORDER = required_float_env("RTRADE_WAIT_FOR_ORDER_SEC")
MIN_adjustment_percent = required_float_env("RTRADE_MIN_ADJUSTMENT_PCT")

# Express each round's budget in quote currency. Calculate TAO quantity once from
# the round's starting price.
RTRADE_NOTIONAL_USDC = required_float_env("RTRADE_NOTIONAL_USDC")

# Initial fractional spread estimate for filled prices before the first real fill.
RTRADE_INITIAL_SPREAD_PCT = required_float_env("RTRADE_INITIAL_SPREAD_PCT")

# Relaxation rate for adjustment_percent after the opposite side fills.
# BUY and SELL are intentionally asymmetric.
RTRADE_BUY_DECAY_PCT = required_float_env("RTRADE_BUY_DECAY_PCT")
RTRADE_SELL_DECAY_PCT = required_float_env("RTRADE_SELL_DECAY_PCT")

# Base hours window, divided by failure_count, for the desperate path after the
# opposite side fills. BUY (0.3) and SELL (0.23) are intentionally asymmetric.
RTRADE_BUY_DESPERATE_HOURS_BASE = required_float_env("RTRADE_BUY_DESPERATE_HOURS_BASE")
RTRADE_SELL_DESPERATE_HOURS_BASE = required_float_env("RTRADE_SELL_DESPERATE_HOURS_BASE")

# Shared BUY/SELL lookback for the desperate path (one hour plus 60 seconds).
RTRADE_DESPERATE_SAFEBACK_SEC = required_float_env("RTRADE_DESPERATE_SAFEBACK_SEC")

# Hours for the normal path before desperation. BUY waits longer than SELL.
RTRADE_BUY_NORMAL_HOURS = required_float_env("RTRADE_BUY_NORMAL_HOURS")
RTRADE_SELL_NORMAL_HOURS = required_float_env("RTRADE_SELL_NORMAL_HOURS")

# Fractional price offset and hours window for the desperate follow-up immediately
# after the opposite side fills. Shared by both directions.
RTRADE_FOLLOWUP_OFFSET_PCT = required_float_env("RTRADE_FOLLOWUP_OFFSET_PCT")
RTRADE_FOLLOWUP_HOURS = required_float_env("RTRADE_FOLLOWUP_HOURS")

# ``are_close`` tolerance for detecting a bad day when price crosses the reference,
# plus the adjustment multiplier used then. Shared by both directions.
RTRADE_BAD_DAY_TOLERANCE_PCT = required_float_env("RTRADE_BAD_DAY_TOLERANCE_PCT")
RTRADE_BAD_DAY_MULTIPLIER = required_float_env("RTRADE_BAD_DAY_MULTIPLIER")

# Epsilon preventing division by zero in profit/loss ratio calculations.
RTRADE_ZERO_EPSILON = required_float_env("RTRADE_ZERO_EPSILON")

# Maximum failures accepted before abandoning a BUY or SELL order.
RTRADE_MAX_FAILURES = required_int_env("RTRADE_MAX_FAILURES")
# Trend filter: this spread bot suffers adverse selection in a clear trend. It stays
# idle when ``|gradient_recent| > K * epsilon`` over a short window, where epsilon is
# the volatility-calibrated cacheManager noise floor. Controlled by a kill switch and threshold.
RTRADE_TREND_FILTER_ENABLED = required_bool_env("RTRADE_TREND_FILTER_ENABLED")
RTRADE_TREND_FILTER_K = required_float_env("RTRADE_TREND_FILTER_K")
RTRADE_TREND_WINDOW_SEC = required_float_env("RTRADE_TREND_WINDOW_SEC")
RTRADE_TREND_OHLC_FALLBACK_ENABLED = required_bool_env("RTRADE_TREND_OHLC_FALLBACK_ENABLED")
RTRADE_DYNAMIC_MARKET_EXIT_MODE = required_env("RTRADE_DYNAMIC_MARKET_EXIT_MODE").lower()
RTRADE_EMERGENCY_HARD_STOP_PCT = required_float_env("RTRADE_EMERGENCY_HARD_STOP_PCT")

# The new coordinator remains disabled until replay/walk-forward validation. When enabled,
# one owner manages the pair and exposure; the legacy two-worker path remains behind the switch.
RTRADE_PAIR_COORDINATOR_ENABLED = required_bool_env("RTRADE_PAIR_COORDINATOR_ENABLED")
RTRADE_PAIR_POLL_SEC = required_float_env("RTRADE_PAIR_POLL_SEC")
RTRADE_PAIR_MAX_ACTIVE_ROUNDS = required_int_env("RTRADE_PAIR_MAX_ACTIVE_ROUNDS")
RTRADE_PAIR_START_INTERVAL_SEC = required_float_env("RTRADE_PAIR_START_INTERVAL_SEC")
RTRADE_PAIR_DIRECTIONS = tuple(
    side.strip().upper()
    for side in required_env("RTRADE_PAIR_DIRECTIONS").split(",")
    if side.strip()
)
RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC = required_float_env("RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC")
RTRADE_PLACE_FAILURE_BACKOFF_SEC = required_float_env("RTRADE_PLACE_FAILURE_BACKOFF_SEC")
RTRADE_INTENT_MISSING_CONFIRMATIONS = required_int_env(
    "RTRADE_INTENT_MISSING_CONFIRMATIONS")
RTRADE_INTENT_RECOVERY_HORIZON_SEC = required_float_env(
    "RTRADE_INTENT_RECOVERY_HORIZON_SEC")
RTRADE_FAST_FILL_RATIO = required_float_env("RTRADE_FAST_FILL_RATIO")
RTRADE_MIN_EDGE_PCT = required_float_env("RTRADE_MIN_EDGE_PCT")
RTRADE_SHOCK_HARD_STOP_PCT = required_float_env("RTRADE_SHOCK_HARD_STOP_PCT")
RTRADE_HARD_STOP_PCT = required_float_env("RTRADE_HARD_STOP_PCT")


def _validate_intent_recovery_config(
        missing_confirmations=RTRADE_INTENT_MISSING_CONFIRMATIONS,
        poll_sec=RTRADE_PAIR_POLL_SEC,
        horizon_sec=RTRADE_INTENT_RECOVERY_HORIZON_SEC):
    if (type(missing_confirmations) is not int
            or missing_confirmations < 2):
        raise ValueError(
            "RTRADE_INTENT_MISSING_CONFIRMATIONS must be an integer >= 2")
    try:
        poll_sec = float(poll_sec)
        horizon_sec = float(horizon_sec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "rtrade intent recovery timing must be numeric") from exc
    if not math.isfinite(poll_sec) or poll_sec <= 0:
        raise ValueError("RTRADE_PAIR_POLL_SEC must be finite and positive")
    if (not math.isfinite(horizon_sec)
            or horizon_sec < poll_sec * missing_confirmations):
        raise ValueError(
            "RTRADE_INTENT_RECOVERY_HORIZON_SEC is too short")


_validate_intent_recovery_config()


class _IntentRecoveryPending(RuntimeError):
    """A persisted venue intent is not yet safe to submit or discard."""


class _LivePairVenue:
    """Thin adapter between the pure coordinator and the current Binance venue."""

    def __init__(self, symbol, pair_store=None, *, recovery_clock=time.time):
        self.symbol = symbol
        # Used only to release cooldown on cancellation. Rounds retain their own
        # financial state, so a small LRU is sufficient here.
        self._known_tickets = deque(maxlen=max(32, RTRADE_PAIR_MAX_ACTIVE_ROUNDS * 8))
        self._last_place_failures = {}
        self._recovery_blocked_pairs = set()
        self.recovery_blocked = False
        self._recovery_clock = recovery_clock
        provider_name = mkt.provider_name_for(symbol)
        self.provider_name = provider_name
        self.executor = mkt.provider_by_name(provider_name)
        if self.executor is None:
            raise RuntimeError(f"provider executor unavailable for {symbol}")
        self.audited_executor = AuditedStrategyExecutor(
            self.executor, venue=provider_name)
        self.pair_store = pair_store
        _validate_intent_recovery_config()
        # A Binance client ID is deterministic for one pair leg. Require repeated
        # observations on separate ticks, but never reuse an ambiguous submission ID.
        self.order_lifecycle = TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(self.executor),
            provider_name=provider_name, venue=provider_name,
            missing_confirmations=RTRADE_INTENT_MISSING_CONFIRMATIONS,
            retry_on_lookup_error=False,
            audit=(self.audited_executor.audit
                   if self.pair_store is not None else None),
        )

    def _set_pair_recovery_blocked(self, pair_id, blocked=True):
        pair_id = str(pair_id or "")
        if pair_id:
            if blocked:
                self._recovery_blocked_pairs.add(pair_id)
            else:
                self._recovery_blocked_pairs.discard(pair_id)
        self.recovery_blocked = bool(self._recovery_blocked_pairs)

    def _pair_record(self, pair_id):
        if self.pair_store is None:
            return None
        return next(
            (record for record in self.pair_store.active(self.symbol)
             if record.get("pair_id") == pair_id),
            None,
        )

    def pair_recovery_blocked(self, pair_id):
        return str(pair_id or "") in self._recovery_blocked_pairs

    @staticmethod
    def _pre_submit_refusal(intent):
        return (
            str(intent.get("submission_outcome") or "").lower() == "refused"
            or str(intent.get("submit_status") or "").lower()
            == "refused_before_submit"
        )

    def current_price(self):
        return mkt.get_current_price(self.symbol)

    def preflight_order(self, side, qty, price=None, *, market=False, kind=None):
        return self.audited_executor.preflight_order(
            self.symbol, side, qty, price, market=market, kind=kind)

    @staticmethod
    def _intent_id(pair_id, side, kind):
        return f"rtrade:{pair_id}:{kind}:{str(side).lower()}"

    @staticmethod
    def _limit_kind(revision):
        try:
            revision = int(revision)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("limit revision must be an integer") from exc
        if not 0 <= revision <= 1_000_000:
            raise ValueError("limit revision must be between 0 and 1000000")
        return "limit" if revision == 0 else f"limit_{revision}"

    @staticmethod
    def _limit_revision(kind):
        kind = str(kind)
        if kind == "limit":
            return 0
        if not kind.startswith("limit_"):
            return None
        suffix = kind.removeprefix("limit_")
        if not suffix.isdigit():
            return None
        revision = int(suffix)
        return revision if 0 < revision <= 1_000_000 else None

    def _persist_callback(self, pair_id, side, kind, requested_qty):
        if self.pair_store is None:
            return lambda _pending: None

        def persist(pending):
            self.pair_store.persist_intent(
                pair_id, side, kind, pending,
                symbol=self.symbol, start_side=side, qty=requested_qty)
        return persist

    def _remember_ticket(self, ticket):
        if ticket is None:
            return None
        if not any(existing.order_id == ticket.order_id
                   for existing in self._known_tickets):
            self._known_tickets.append(ticket)
        return ticket

    def _ticket_from_pending(self, pending, pair_id):
        order_id = pending.get("order_id")
        if not order_id:
            return None
        ticket_price = float(
            pending.get("submitted_price")
            or pending.get("requested_price") or 0.0)
        if ticket_price <= 0:
            # Old hard-stop intents stored no limit price. Use a live reference
            # only to reconstruct the ticket; venue status remains authoritative
            # for fill quantity, cost, and fees.
            ticket_price = float(self.current_price() or 0.0)
        ticket = PairOrderTicket(
            order_id=str(order_id), side=str(pending["side"]).upper(),
            price=ticket_price,
            qty=float(
                pending.get("submitted_qty")
                or pending.get("requested_qty")),
            pair_id=pair_id,
            revision=(
                self._limit_revision(pending.get("kind"))
                if self._limit_revision(pending.get("kind")) is not None
                else self._hard_stop_revision_from_kind(
                    pending.get("kind")) or 0),
        )
        return self._remember_ticket(ticket)

    def _submit_limit_intent(self, side, price, qty, pair_id, *, attempt=1,
                             cache_permit=None, revision=0):
        side = side.upper()
        hours = (RTRADE_BUY_NORMAL_HOURS if side == "BUY"
                 else RTRADE_SELL_NORMAL_HOURS)
        store_kind = self._limit_kind(revision)
        client_id = rtrade_client_order_id(pair_id, side, store_kind)
        intent = self.order_lifecycle.new_intent(
            intent_id=self._intent_id(pair_id, side, store_kind),
            client_order_id=client_id, symbol=self.symbol, side=side,
            requested_qty=qty, requested_price=price, kind=store_kind,
            attempt=attempt,
            metadata={"pair_id": pair_id, "limit_revision": int(revision)},
        )
        persist = self._persist_callback(
            pair_id, side, store_kind, requested_qty=qty)
        outcome_context = {}

        def submit_once():
            response = mkt.place(
                self.symbol, side, price, qty,
                force=False, cancelorders=False, hours=hours, smart=False,
                cooldown_pair_id=pair_id,
                # The coordinator owns retry/reconciliation. The global outbox
                # must not later recreate a leg from an expired pair.
                caller_owns_retry=True, motivation="rtrade_pair_quote",
                client_order_id=client_id,
                _outcome_context=outcome_context,
                cache_permit=cache_permit,
            )
            return propagate_submission_refusal(response, outcome_context)

        result = self.order_lifecycle.submit(
            intent, persist=persist,
            submit=submit_once,
        )
        return self._ticket_from_pending(result.intent, pair_id), result.intent

    def _canonical_intent(self, record, stored):
        """Upgrade an old rtrade intent in memory without losing observations."""
        pending = dict(stored)
        side = str(pending.get("side") or record["start_side"]).upper()
        kind = str(pending.get("kind") or "limit")
        requested_qty = pending.get("requested_qty", pending.get("qty"))
        requested_price = pending.get("requested_price", pending.get("price"))
        pending.update({
            "intent_id": pending.get("intent_id")
                         or self._intent_id(record["pair_id"], side, kind),
            "client_order_id": pending["client_order_id"],
            "symbol": pending.get("symbol") or record["symbol"],
            "side": side,
            "kind": kind,
            "requested_qty": float(requested_qty),
            "requested_price": (None if requested_price is None
                                else float(requested_price)),
            "attempt": int(pending.get("attempt") or 1),
            "created_at": float(
                pending.get("created_at") or record.get("created_ts")
                or time.time()),
            "lookup_misses": int(pending.get("lookup_misses") or 0),
            "pair_id": pending.get("pair_id") or record["pair_id"],
        })
        return pending

    def recover_intent(self, record, stored):
        """Recover one fsynced pair intent without guessing venue state.

        Lookup/status ambiguity leaves the round active and blocks new rounds.
        Missing-order confirmations happen on separate configured coordinator
        ticks. Even confirmed absence never permits reuse of an ambiguous client
        order ID because Binance may eventually expose an accepted terminal order.
        """
        pending = self._canonical_intent(record, stored)
        side = pending["side"]
        kind = pending["kind"]
        pair_id = record["pair_id"]
        persist_raw = self._persist_callback(
            pair_id, side, kind, requested_qty=pending["requested_qty"])
        now = float(self._recovery_clock())
        if not math.isfinite(now) or now <= 0:
            raise ValueError("rtrade recovery clock returned invalid time")
        if not pending.get("order_id"):
            recovery_started_at = pending.get("recovery_started_at")
            if recovery_started_at is not None:
                try:
                    recovery_age = max(0.0, now - float(recovery_started_at))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{kind}:{side} has invalid recovery timing") from exc
                if recovery_age >= RTRADE_INTENT_RECOVERY_HORIZON_SEC:
                    pending["recovery_state"] = "horizon_exhausted"
                    pending["recovery_horizon_exhausted_at"] = now
                    persist_raw(pending)
                    raise _IntentRecoveryPending(
                        f"{kind}:{side} recovery horizon is exhausted")
            if pending.get("recovery_state") in {
                    "absence_confirmed_no_reuse", "horizon_exhausted"}:
                persist_raw(pending)
                raise _IntentRecoveryPending(
                    f"{kind}:{side} requires manual reconciliation")
            next_recovery_at = pending.get("next_recovery_at")
            if next_recovery_at is not None:
                try:
                    next_recovery_at = float(next_recovery_at)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{kind}:{side} has invalid next recovery time") from exc
                if not math.isfinite(next_recovery_at):
                    raise ValueError(
                        f"{kind}:{side} has invalid next recovery time")
                if now < next_recovery_at:
                    raise _IntentRecoveryPending(
                        f"{kind}:{side} reconciliation is not due yet")

        retained = dict(pending)

        def persist(value):
            nonlocal retained
            if value is None:
                retained["recovery_state"] = "absence_confirmed_no_reuse"
                retained["absence_confirmed_at"] = now
                retained["last_recovery_lookup_at"] = now
                retained["next_recovery_at"] = now + RTRADE_PAIR_POLL_SEC
                persist_raw(dict(retained))
                return
            retained = dict(value)
            if retained.get("order_id"):
                for field in (
                        "lookup_error", "recovery_state", "recovery_started_at",
                        "last_recovery_lookup_at", "next_recovery_at",
                        "absence_confirmed_at", "recovery_horizon_exhausted_at"):
                    retained.pop(field, None)
            elif ("lookup_error" in retained
                  or int(retained.get("lookup_misses") or 0) > 0):
                retained.setdefault("recovery_started_at", now)
                retained["last_recovery_lookup_at"] = now
                retained["next_recovery_at"] = now + RTRADE_PAIR_POLL_SEC
            persist_raw(dict(retained))

        persist_raw(pending)
        result = self.order_lifecycle.reconcile(pending, persist=persist)

        if result.outcome == "absent":
            raise _IntentRecoveryPending(
                f"{kind}:{side} absence is confirmed; client-ID reuse is blocked")

        if not result.intent.get("order_id"):
            raise _IntentRecoveryPending(
                f"{kind}:{side} intent is ambiguous; lookup/status is unavailable")
        if result.status is None:
            raise _IntentRecoveryPending(
                f"intent {kind}:{side} has an order_id but no available status")

        ticket = self._ticket_from_pending(result.intent, pair_id)
        snapshot = PairOrderSnapshot(
            status=result.status.status,
            filled_qty=result.status.filled_qty,
            cost=result.status.cost,
            fee=result.status.fee,
        )
        return ticket, snapshot

    def _reconcile_existing_intent(self, pair_id, side, kind):
        """Return state, ticket, and snapshot for one persisted submit slot."""
        if self.pair_store is None:
            return "none", None, None
        record = self._pair_record(pair_id)
        if record is None:
            return "none", None, None
        stored = record.get("intents", {}).get(f"{kind}:{side}")
        if stored is None:
            return "none", None, None
        if self._pre_submit_refusal(stored):
            self.pair_store.persist_intent(
                pair_id, side, kind, None, symbol=self.symbol)
            return "safe_refusal", None, None
        try:
            ticket, snapshot = self.recover_intent(record, stored)
        except _IntentRecoveryPending:
            self._set_pair_recovery_blocked(pair_id)
            return "pending", None, None
        self._set_pair_recovery_blocked(pair_id, False)
        return "recovered", ticket, snapshot

    def recover_pair_intents(self, pair_id):
        """Reconcile every persisted intent for one active coordinator tick."""
        record = self._pair_record(pair_id)
        if record is None:
            self._set_pair_recovery_blocked(pair_id)
            raise _IntentRecoveryPending(
                f"pair={pair_id}: persisted recovery record is unavailable")
        recovered = []
        pending_recovery = False
        for stored in list(dict(record.get("intents") or {}).values()):
            pending = self._canonical_intent(record, stored)
            side = pending["side"]
            kind = pending["kind"]
            if self._pre_submit_refusal(pending):
                self.pair_store.persist_intent(
                    pair_id, side, kind, None, symbol=self.symbol)
                continue
            try:
                item = self.recover_intent(record, pending)
            except _IntentRecoveryPending:
                pending_recovery = True
                continue
            if item is not None:
                ticket, snapshot = item
                recovered.append((ticket, snapshot, kind))
        self._set_pair_recovery_blocked(pair_id, pending_recovery)
        return recovered

    @staticmethod
    def _hard_stop_revision_from_kind(kind):
        kind = str(kind)
        if kind == "hard_stop":
            return 0
        if not kind.startswith("hard_stop_"):
            return None
        suffix = kind.removeprefix("hard_stop_")
        if not suffix.isdigit():
            return None
        revision = int(suffix)
        return revision if 0 < revision <= 1_000_000 else None

    def merge_checkpoint_intents(self, record, state):
        """Merge fsynced intents that raced ahead of a coordinator checkpoint.

        Venue submission persists its intent before returning to the coordinator.
        A crash can therefore leave an accepted replacement or hard stop in the
        intent map while the older checkpoint still references the prior order.
        Reconcile every unrepresented intent by its deterministic client ID/order
        ID, then advance persisted revisions before the state machine runs again.
        """
        merged = dict(state or {})
        tickets = [dict(ticket) for ticket in merged.get("tickets", [])]
        snapshots = {
            str(order_id): dict(snapshot)
            for order_id, snapshot in dict(
                merged.get("snapshots") or {}).items()
        }
        represented = {
            str(ticket.get("order_id")) for ticket in tickets
            if ticket.get("order_id") is not None
        }
        limit_revisions = {"BUY": 0, "SELL": 0}
        for side, value in dict(
                merged.get("limit_revisions") or {}).items():
            side_u = str(side).upper()
            if side_u in limit_revisions:
                revision = int(value or 0)
                if not 0 <= revision <= 1_000_000:
                    raise ValueError(f"invalid {side_u} limit revision: {revision}")
                limit_revisions[side_u] = revision
        hard_stop_revision = int(merged.get("hard_stop_revision", 0) or 0)
        if not 0 <= hard_stop_revision <= 1_000_000:
            raise ValueError(
                f"invalid hard-stop revision: {hard_stop_revision}")
        recovery_pending = False
        pending_hard_stop = False

        for stored in dict(record.get("intents") or {}).values():
            pending = self._canonical_intent(record, stored)
            side = pending["side"]
            kind = pending["kind"]
            limit_revision = self._limit_revision(kind)
            stop_revision = self._hard_stop_revision_from_kind(kind)
            if limit_revision is not None:
                limit_revisions[side] = max(
                    limit_revisions[side], limit_revision)
            if stop_revision is not None:
                hard_stop_revision = max(hard_stop_revision, stop_revision)
            if self._pre_submit_refusal(pending):
                self.pair_store.persist_intent(
                    record["pair_id"], side, kind, None, symbol=self.symbol)
                continue

            order_id = pending.get("order_id")
            if order_id is not None and str(order_id) in represented:
                represented_revision = (
                    limit_revision if limit_revision is not None
                    else stop_revision)
                if represented_revision is not None:
                    for ticket in tickets:
                        if str(ticket.get("order_id")) == str(order_id):
                            ticket["revision"] = represented_revision
                            break
                if stop_revision is not None:
                    merged["stop_order_id"] = str(order_id)
                    merged["phase"] = "stopping"
                continue

            try:
                recovered = self.recover_intent(record, pending)
            except _IntentRecoveryPending as exc:
                recovery_pending = True
                pending_hard_stop = pending_hard_stop or stop_revision is not None
                merged["reason"] = f"intent_recovery_pending:{exc}"
                continue
            if recovered is None:
                continue
            ticket, snapshot = recovered
            order_id = str(ticket.order_id)
            if order_id not in represented:
                ticket.active = snapshot.status not in {
                    "closed", "canceled", "expired"}
                tickets.append(vars(ticket).copy())
                represented.add(order_id)
            snapshots[order_id] = vars(snapshot).copy()
            if stop_revision is not None:
                merged["stop_order_id"] = order_id
                merged["phase"] = "stopping"
                merged["hard_stop_reason"] = (
                    merged.get("hard_stop_reason")
                    or "inventory_hard_stop")

        merged["tickets"] = tickets
        merged["snapshots"] = snapshots
        merged["limit_revisions"] = limit_revisions
        merged["hard_stop_revision"] = hard_stop_revision
        if recovery_pending:
            merged["phase"] = (
                "hard_stop_recovery" if pending_hard_stop
                else "startup_recovery")
            merged["recovery_pending"] = True
            self._set_pair_recovery_blocked(record["pair_id"])
        else:
            merged.pop("recovery_pending", None)
            self._set_pair_recovery_blocked(record["pair_id"], False)
        return merged

    def place_limit(self, side, price, qty, pair_id, *, cache_permit=None,
                    revision=0):
        side = side.upper()
        store_kind = self._limit_kind(revision)
        self._last_place_failures.pop(side, None)
        existing_state, existing_ticket, _snapshot = (
            self._reconcile_existing_intent(pair_id, side, store_kind))
        if existing_state == "recovered":
            return existing_ticket
        if existing_state == "pending":
            self._last_place_failures[side] = (
                f"{side.lower()}_recovery_pending")
            return None
        from providers.quantity import balance_cap_quantity
        available_qty, required_asset = balance_cap_quantity(
            self.executor.free_balance, self.symbol, side, price)
        if required_asset:
            # Do not compare against requested quantity: QuantityDecision and venue
            # mechanics already cap it to available balance. Back off only when the
            # required asset has no usable free balance.
            if available_qty is not None and float(available_qty) <= 1e-12:
                self._last_place_failures[side] = (
                    f"{side.lower()}_insufficient_funds:{required_asset}")
                print(
                    f"[{self.symbol}] {side} insufficient funds: "
                    f"available=0.00000000 {required_asset}")
                return None
        ticket, _pending = self._submit_limit_intent(
            side, price, qty, pair_id, cache_permit=cache_permit,
            revision=revision)
        if ticket is None and _pending.get("refusal_reason"):
            # A guard refusal happened before any provider submit. It is safe to
            # finish this round and let the strategy backoff create a fresh intent;
            # venue response-loss recovery would be both false and duplicative.
            return None
        if ticket is None and self.pair_store is not None:
            # A missing response is not retried synchronously. The persisted
            # client ID is reconciled on later coordinator ticks.
            self._set_pair_recovery_blocked(pair_id)
            self._last_place_failures[side] = (
                f"{side.lower()}_recovery_pending")
        elif ticket is not None:
            self._set_pair_recovery_blocked(pair_id, False)
        return ticket

    def last_place_failure_reason(self, side):
        return self._last_place_failures.get(side.upper())

    def market_exit_allowed(self, exposure_side, loss_fraction, reason):
        """Decide whether a normal hard stop has trend confirmation for MARKET.

        ``shadow`` only observes; ``live`` applies the decision. The emergency threshold
        remains mandatory regardless of trend so a stale signal cannot leave unlimited risk.
        """
        emergency = loss_fraction >= RTRADE_EMERGENCY_HARD_STOP_PCT
        decision = _market_regime_decision(self.symbol)
        adverse = decision.adverse_to(exposure_side)
        justified = adverse or emergency
        print(
            f"[{self.symbol}] dynamic-market-exit mode={RTRADE_DYNAMIC_MARKET_EXIT_MODE} "
            f"exposure={exposure_side} regime={decision.regime} "
            f"strength={decision.strength} signal_reason={decision.reason} "
            f"loss={loss_fraction:.4%} emergency={emergency} "
            f"would_market={justified} reason={reason}")
        if RTRADE_DYNAMIC_MARKET_EXIT_MODE == "emergency_only":
            # Discretionary / adverse-regime exits WAIT (the round holds) -- a possibly
            # false drop must not stop us out. Only a real loss >= the emergency
            # threshold still forces a market exit as the catastrophe backstop.
            return emergency
        if RTRADE_DYNAMIC_MARKET_EXIT_MODE in {"off", "shadow"}:
            return True
        return justified

    def place_market_exit(self, side, qty, reason, pair_id=None, *,
                          cache_permit=None, revision=0):
        """Place a spot market exit reserved for reducing hard-stop exposure.

        Normal orders remain on ``mkt.place``/``place_safe_order``. This risk exit
        bypasses profit, cooldown, and weight checks, but never bypasses balance,
        fee-cap, precision, preflight, or client-order-ID auditing.
        """
        side = side.upper()
        revision = int(revision)
        if not 0 <= revision <= 1_000_000:
            raise ValueError("hard-stop revision must be between 0 and 1000000")
        store_kind = "hard_stop" if revision == 0 else f"hard_stop_{revision}"
        self._last_place_failures.pop(side, None)
        if pair_id:
            existing_state, existing_ticket, _snapshot = (
                self._reconcile_existing_intent(
                    pair_id, side, store_kind))
            if existing_state == "recovered":
                return existing_ticket
            if existing_state == "pending":
                self._last_place_failures[side] = (
                    f"{side.lower()}_recovery_pending")
                return None
        price = float(self.current_price() or 0.0)
        if price <= 0:
            print(f"[{self.symbol}] {side} hard-stop BLOCKED: price unavailable")
            return None
        # This audited protective path follows venue filters only. The business
        # notional floor belongs to ordinary Instrument orders, not risk reduction.
        decision = decide_quantity(
            self.executor, self.symbol, side, price, qty,
            apply_policy=False, market=True,
            enforce_business_minimum=False)
        final_qty = float(decision.final_qty)
        if final_qty <= 0:
            print(f"[{self.symbol}] {side} hard-stop BLOCKED: "
                  f"{decision.refuse_reason} asset={decision.balance_asset}")
            return None
        precision = self.audited_executor.pair_precision(self.symbol)
        if precision is not None:
            quantum = Decimal(1).scaleb(-int(precision.volume_decimals))
            final_qty = float(Decimal(str(final_qty)).quantize(quantum, rounding=ROUND_DOWN))
            if final_qty <= 0 or final_qty < float(precision.order_min or 0.0):
                print(f"[{self.symbol}] {side} hard-stop BLOCKED: qty {final_qty} "
                      f"below the minimum {precision.order_min}")
                return None
        kind = f"rtrade:{reason}:{pair_id or 'unknown-pair'}:revision:{revision}"
        intent_id = (
            f"rtrade-{pair_id or 'unknown'}-{reason}-{side.lower()}-r{revision}")
        client_id = rtrade_client_order_id(
            pair_id or "unknown", side, store_kind)
        replacement_after_cancel = cache_permit is not None
        if cache_permit is None:
            cache_permit = self.preflight_order(
                side, final_qty, price=None, market=True, kind=kind)
        if self.pair_store is not None and pair_id:
            self.pair_store.intent(
                pair_id, side, price, final_qty, client_id, kind=store_kind,
                symbol=self.symbol)
        try:
            order_id = self.audited_executor.submit_order_with_intent(
                intent_id, self.symbol, side, final_qty, price=None,
                market=True, kind=kind, reference_price=price,
                client_order_id=client_id,
                cache_permit=cache_permit)
        except SubmissionRefused as exc:
            if self.pair_store is not None and pair_id:
                if not replacement_after_cancel:
                    self.pair_store.persist_intent(
                        pair_id, side, store_kind, None, symbol=self.symbol)
                else:
                    record = self._pair_record(pair_id)
                    stored = (
                        None if record is None else
                        record.get("intents", {}).get(f"{store_kind}:{side}"))
                    if stored is not None:
                        pending = self._canonical_intent(record, stored)
                        pending["submission_outcome"] = "refused"
                        pending["submit_status"] = "refused_before_submit"
                        pending["refusal_reason"] = exc.reason
                        self.pair_store.persist_intent(
                            pair_id, side, store_kind, pending,
                            symbol=self.symbol)
            raise
        except ProviderError as exc:
            if self.pair_store is not None and pair_id:
                record = self._pair_record(pair_id)
                stored = (
                    None if record is None
                    else record.get("intents", {}).get(f"{store_kind}:{side}"))
                if stored is not None:
                    pending = self._canonical_intent(record, stored)
                    pending["submission_outcome"] = "unknown"
                    pending["submit_error"] = f"{exc.__class__.__name__}: {exc}"
                    self.pair_store.persist_intent(
                        pair_id, side, store_kind, pending,
                        symbol=self.symbol)
                self._set_pair_recovery_blocked(pair_id)
                self._last_place_failures[side] = (
                    f"{side.lower()}_recovery_pending")
            print(
                f"[{self.symbol}] {side} hard-stop response is ambiguous; "
                "the persisted client ID must be reconciled")
            return None
        if order_id is None or not str(order_id).strip():
            if self.pair_store is not None and pair_id:
                self._set_pair_recovery_blocked(pair_id)
                self._last_place_failures[side] = (
                    f"{side.lower()}_recovery_pending")
            return None
        ticket = PairOrderTicket(
            order_id=str(order_id), side=side, price=price, qty=final_qty,
            pair_id=pair_id, revision=revision)
        self._known_tickets.append(ticket)
        if self.pair_store is not None and pair_id:
            self.pair_store.accepted(
                pair_id, side, ticket.order_id, kind=store_kind)
            self._set_pair_recovery_blocked(pair_id, False)
        return ticket

    def order_status(self, order_id):
        status = self.executor.order_status(self.symbol, str(order_id))
        return PairOrderSnapshot(
            status=status.status, filled_qty=status.filled_qty,
            cost=status.cost, fee=status.fee)

    def cancel(self, order_id):
        try:
            self.executor.cancel_order(self.symbol, str(order_id))
            for ticket in getattr(self, "_known_tickets", []):
                if ticket.order_id == str(order_id) and ticket.pair_id:
                    from lock import trade_cooldown
                    trade_cooldown.release_pair_leg(
                        self.symbol, ticket.pair_id, ticket.side)
                    break
            return True
        except Exception as exc:  # noqa: BLE001 — the coordinator decides how to fail closed.
            print(f"[{self.symbol}] pair cancel {order_id} failed: {exc}")
            return False


def _trend_too_strong(symbol):
    """Return whether the regime should keep the spread bot idle.

    Stand aside on a clear trend (``|gradient_recent| > K * epsilon``; adverse-selection
    risk) AND when trend data is UNAVAILABLE (regime ``unknown``). With no data we cannot
    tell a safe flat market from a trend, so we do nothing rather than trade blind: a data
    outage must never be read as "flat" and green-light spread rounds. Only a CONFIRMED
    ``sideways`` regime lets the bot run. Fail-CLOSED on no data (deliberately unlike the
    pipeline's trend wait, which fails open).
    """
    if not RTRADE_TREND_FILTER_ENABLED:
        return False
    decision = _market_regime_decision(symbol)
    stand_aside = decision.directional or decision.regime == "unknown"
    if stand_aside:
        print(f"[{symbol}] rtrade STANDING ASIDE: regime {decision.regime} "
              f"strength={decision.strength} reason={decision.reason} "
              f"window={RTRADE_TREND_WINDOW_SEC:.0f}s")
    return stand_aside


def _market_regime_decision(symbol) -> MarketRegimeDecision:
    """Adapt the existing cache source to the provider-neutral regime evaluator."""
    if not RTRADE_TREND_FILTER_ENABLED:
        return mkt.market_regime(
            symbol, horizon="short", snapshot={}, snapshot_source="cache24_dynamic",
            strength_threshold=RTRADE_TREND_FILTER_K, allow_fallback=False)
    try:
        import cacheManager as cm
        dyn = cm.get_short_trend_manager().get_instant_trend_for_window(
            symbol, RTRADE_TREND_WINDOW_SEC)
    except Exception as exc:
        return mkt.market_regime(
            symbol, horizon="short", snapshot={}, snapshot_source="cache24_dynamic",
            strength_threshold=RTRADE_TREND_FILTER_K,
            allow_fallback=RTRADE_TREND_OHLC_FALLBACK_ENABLED)
    return mkt.market_regime(
        symbol, horizon="short", snapshot=dyn or {},
        snapshot_source="cache24_dynamic",
        strength_threshold=RTRADE_TREND_FILTER_K,
        allow_fallback=RTRADE_TREND_OHLC_FALLBACK_ENABLED)


def _order_fully_filled(symbol, order_id):
    """Support the legacy loop through provider-neutral order status."""
    if not order_id:
        return False
    try:
        return mkt.order_status(symbol, str(order_id)).fully_filled
    except Exception as exc:
        print(f"[{symbol}] order status {order_id} unavailable: {exc}")
        return False


def _preflight_routed_order(symbol, side, qty, price=None, *,
                            market=False, kind=None):
    provider_name = mkt.provider_name_for(symbol)
    executor = mkt.provider_by_name(provider_name)
    if executor is None:
        raise RuntimeError(f"provider executor unavailable for {symbol}")
    preflight = getattr(executor, "preflight_order", None)
    if not callable(preflight):
        raise RuntimeError(f"{provider_name} preflight unavailable for {symbol}")
    return preflight(
        symbol, side, qty, price, market=market, kind=kind)


def _cancel_order_confirmed(symbol, order_id):
    cancel_error = None
    try:
        mkt.cancel_order(symbol, str(order_id))
    except Exception as exc:
        cancel_error = exc
        print(f"[{symbol}] order cancel {order_id} unconfirmed: {exc}")
    try:
        status = mkt.order_status(symbol, str(order_id))
    except Exception as exc:
        print(f"[{symbol}] final order status {order_id} unavailable: {exc}")
        return None
    if not status.terminal:
        detail = f" after cancel error {cancel_error}" if cancel_error else ""
        print(f"[{symbol}] order {order_id} remains active{detail}; replacement deferred")
        return None
    return status


def _remaining_after_canceled_order(status, submitted_qty):
    """Return the unfilled terminal remainder, or ``None`` for invalid state."""
    try:
        submitted_qty = float(submitted_qty)
        filled_qty = float(status.filled_qty)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(submitted_qty) or submitted_qty <= 0
            or not math.isfinite(filled_qty) or filled_qty < 0
            or filled_qty > submitted_qty + max(1e-12, submitted_qty * 1e-9)):
        return None
    if status.fully_filled:
        return 0.0
    return max(0.0, submitted_qty - filled_qty)


def _place_failure_backoff(reason):
    """Return ``(side, seconds)`` for a terminal placement failure.

    Completely absent funds have an explicit reason. Other ``*_place_failed`` cases,
    including price guards, minimum notional, weight limits, and API refusal, also need
    throttling so transient errors cannot become an order loop with a new pair ID each time.
    """
    reason = str(reason or "").strip().lower()
    marker = "_insufficient_funds:"
    if marker in reason:
        side = reason.split(marker, 1)[0].upper()
        if side in {"BUY", "SELL"}:
            return side, RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC
    suffix = "_place_failed"
    if reason.endswith(suffix):
        side = reason[:-len(suffix)].upper()
        if side in {"BUY", "SELL"}:
            return side, RTRADE_PLACE_FAILURE_BACKOFF_SEC
    return None, 0.0


def _followup_force(symbol, side):
    """Choose MARKET force for a post-fill flip only when trend is not adverse.

    Selling into a clear decline or buying into a clear rise is adverse, so return False
    and leave a patient limit at the flip price. A confirmed flat (``sideways``) or
    favorable trend keeps immediate market behavior. UNAVAILABLE trend data (regime
    ``unknown``) also returns False: with no data we do not fire an aggressive market
    flip blind -- leave a patient limit instead. This prevents desperate execution
    against the trend or during a data outage.
    """
    if not RTRADE_TREND_FILTER_ENABLED:
        return True
    decision = _market_regime_decision(symbol)
    if decision.regime == "unknown":
        print(f"[{symbol}] follow-up {(side or '').upper()}: trend data unavailable "
              "-> patient limit, NOT a market order")
        return False
    if not decision.directional:
        return True   # A confirmed flat (sideways) regime permits an immediate flip.
    su = (side or "").upper()
    exposure = "LONG" if su == "SELL" else "SOLD"
    adverse = decision.adverse_to(exposure)
    if adverse:
        print(f"[{symbol}] follow-up {su}: adverse {decision.regime} regime "
              "-> use a patient limit, NOT a market order")
        return False
    return True


class TradingBot:
    def __init__(self, symbol, qty, DEFAULT_ADJUSTMENT_PERCENT):
        self.symbol = symbol
        self.qty = qty
        self.transaction_state = "COMPLETED"  # Initial state.
        current_price = api.get_current_price(symbol)
        self.filled_buy_price = round(current_price * (1 - RTRADE_INITIAL_SPREAD_PCT), 4)
        self.filled_sell_price = round(current_price * (1 + RTRADE_INITIAL_SPREAD_PCT), 4)
        self.buy_filled = False
        self.sell_filled = False
        self.DEFAULT_ADJUSTMENT_PERCENT = DEFAULT_ADJUSTMENT_PERCENT
        self.lock = threading.Lock()  # Synchronization lock.

    @property
    def is_buy_filled(self):
        with self.lock:
            return self.buy_filled

    @property
    def is_sell_filled(self):
        with self.lock:
            return self.sell_filled
        
    def mark_buy_filled(self, filled_buy_price=None):
        with self.lock:
            self.buy_filled = True
            self.sell_filled = False
            if filled_buy_price:
                self.filled_buy_price = filled_buy_price
            return self.filled_buy_price

    def mark_sell_filled(self, filled_sell_price=None):
        with self.lock:
            self.buy_filled = False
            self.sell_filled = True
            if filled_sell_price:
                self.filled_sell_price = filled_sell_price
            return self.filled_sell_price
        
    def repetitive_buy(self, current_price, filled_sell_price):
        adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
        failure_count = 1  # Count placement failures.
        max_failures = RTRADE_MAX_FAILURES  # Maximum accepted failures.
        pending_cache_permit = None
        pending_replacement_price = None
        pending_replacement_qty = None

        while True:

            current_price = api.get_current_price(self.symbol)

            if self.is_sell_filled:
                adjustment_percent = max(MIN_adjustment_percent, adjustment_percent - adjustment_percent * RTRADE_BUY_DECAY_PCT)

            calculated_price = round(
                current_price * (1 - adjustment_percent), 4)
            target_buy_price = (
                calculated_price if pending_replacement_price is None
                else pending_replacement_price)
            target_buy_qty = (
                self.qty if pending_replacement_qty is None
                else pending_replacement_qty)
            submit_cache_permit = pending_cache_permit
            pending_cache_permit = pending_replacement_price = None
            pending_replacement_qty = None
            print(f"[{self.symbol}] BUY order initiated at {target_buy_price:.2f}; adjustment {adjustment_percent}%")

            if self.is_buy_filled:
                print(f"[{self.symbol}] Ignore BUY order. It was previously filled at {self.filled_buy_price:.2f}")
                return self.mark_buy_filled(self.filled_buy_price)

            buy_order = None
            h = RTRADE_BUY_DESPERATE_HOURS_BASE / failure_count
            try:
                if self.is_sell_filled:  # Desperate follow-up path.
                    if adjustment_percent == MIN_adjustment_percent:
                        print(f"[{self.symbol}] entering the urgent BUY path")
                    buy_order = mkt.place(
                        self.symbol, "BUY", target_buy_price, target_buy_qty,
                        safeback_seconds=RTRADE_DESPERATE_SAFEBACK_SEC,
                        force=False, cancelorders=True, hours=h, smart=False,
                        kind="rtrade_legacy_quote",
                        cache_permit=submit_cache_permit)
                else:
                    buy_order = mkt.place(
                        self.symbol, "BUY", target_buy_price, target_buy_qty,
                        cancelorders=True, hours=RTRADE_BUY_NORMAL_HOURS,
                        smart=False, kind="rtrade_legacy_quote",
                        cache_permit=submit_cache_permit)
            except po.WeightLimitBlock as e:
                print(f"[{self.symbol}] 24h limit reached — exiting without retry ({e})")
                return None

            if buy_order is None:
                print(f"[{self.symbol}] Order BUY failed, retryed {failure_count} times. Retrying again ...")
                time.sleep(WAIT_FOR_ORDER)
                failure_count += 1
                if failure_count > max_failures:
                    print(f"[{self.symbol}] Order BUY failed {failure_count} times. Exiting.")
                    return None
                continue

            failure_count = 1 # reset failure count after a successful order placement
            
            time.sleep(WAIT_FOR_ORDER)
            order_id = buy_order['orderId']
            self.filled_buy_price = round(float(buy_order['price']), 4)
            
            if _order_fully_filled(self.symbol, order_id):
                print(f"[{self.symbol}] BUY order filled at {self.filled_buy_price:.2f}")
                print(f"[{self.symbol}] Starting urgent SELL follow-up (1)")
                mkt.place(self.symbol, "SELL", api.get_current_price(self.symbol) * (1 + RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                    force=_followup_force(self.symbol, "SELL"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                return self.mark_buy_filled(self.filled_buy_price)


            filled_buy_price = mkt.latest_fill_price(
                self.symbol, "BUY", WAIT_FOR_ORDER)
            if filled_buy_price is not None:
                print(f"[{self.symbol}] BUY order may have been filled :-) at {filled_buy_price:.2f}")
                print(f"[{self.symbol}] Starting urgent SELL follow-up (2)")
                mkt.place(self.symbol, "SELL", api.get_current_price(self.symbol) * (1 + RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                    force=_followup_force(self.symbol, "SELL"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                return self.mark_buy_filled(filled_buy_price)

            current_price = api.get_current_price(self.symbol)
            if current_price > filled_sell_price and not u.are_close(current_price, filled_sell_price, RTRADE_BAD_DAY_TOLERANCE_PCT):
                print(f"[{self.symbol}] Bed day :-(. Trying BUY at current price - x2 {current_price:.2f}")
                adjustment_percent = RTRADE_BAD_DAY_MULTIPLIER * self.DEFAULT_ADJUSTMENT_PERCENT
            # Authorize the exact replacement before cancellation, then submit it
            # on the next iteration.
            replacement_price = round(
                current_price * (1 - adjustment_percent), 4)
            replacement_cache_permit = _preflight_routed_order(
                self.symbol, "BUY", target_buy_qty, replacement_price,
                market=False, kind="rtrade_legacy_quote")
            final_status = _cancel_order_confirmed(self.symbol, order_id)
            if not final_status:
                if _order_fully_filled(self.symbol, order_id):
                    print(f"[{self.symbol}] Cancel BUY order failed. Maybe it was filled :-)? Moving to SELL ...")
                    print(f"[{self.symbol}] Starting urgent SELL follow-up (3)")
                    mkt.place(self.symbol, "SELL", api.get_current_price(self.symbol) * (1 + RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                    force=_followup_force(self.symbol, "SELL"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                    return self.mark_buy_filled(self.filled_buy_price)
                else:
                    print(
                        f"[{self.symbol}] Cancel BUY order remains ambiguous; "
                        "replacement is deferred.")
                    return None
            else:
                remaining_qty = _remaining_after_canceled_order(
                    final_status, target_buy_qty)
                if remaining_qty is None:
                    print(f"[{self.symbol}] Invalid final BUY quantity; replacement deferred.")
                    return None
                if remaining_qty <= max(1e-12, target_buy_qty * 1e-9):
                    print(f"[{self.symbol}] BUY filled during cancellation; moving to SELL.")
                    mkt.place(
                        self.symbol, "SELL",
                        api.get_current_price(self.symbol)
                        * (1 + RTRADE_FOLLOWUP_OFFSET_PCT),
                        self.qty, force=_followup_force(self.symbol, "SELL"),
                        cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                    return self.mark_buy_filled(self.filled_buy_price)
                pending_cache_permit = replacement_cache_permit
                pending_replacement_price = replacement_price
                pending_replacement_qty = remaining_qty


    def repetitive_sell(self, current_price, filled_buy_price):
        adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
        failure_count = 1  # Count placement failures.
        max_failures = RTRADE_MAX_FAILURES  # Maximum accepted failures.
        pending_cache_permit = None
        pending_replacement_price = None
        pending_replacement_qty = None

        while True:

            current_price = api.get_current_price(self.symbol)

            if self.is_buy_filled:
                adjustment_percent = max(MIN_adjustment_percent, adjustment_percent - adjustment_percent * RTRADE_SELL_DECAY_PCT)

            calculated_price = round(
                current_price * (1 + adjustment_percent), 4)
            target_sell_price = (
                calculated_price if pending_replacement_price is None
                else pending_replacement_price)
            target_sell_qty = (
                self.qty if pending_replacement_qty is None
                else pending_replacement_qty)
            submit_cache_permit = pending_cache_permit
            pending_cache_permit = pending_replacement_price = None
            pending_replacement_qty = None
            print(f"[{self.symbol}] SELL order initiated at {target_sell_price:.2f}; adjustment {adjustment_percent}%")

            if self.is_sell_filled:
                print(f"[{self.symbol}] Ignore SELL order. It was previously filled at {self.filled_sell_price:.2f}")
                return self.mark_sell_filled(self.filled_sell_price)

            sell_order = None
            h = RTRADE_SELL_DESPERATE_HOURS_BASE / failure_count
            try:
                if self.is_buy_filled:  # Desperate follow-up path.
                    if adjustment_percent == MIN_adjustment_percent:
                        print(f"[{self.symbol}] entering the urgent SELL path")
                    sell_order = mkt.place(
                        self.symbol, "SELL", target_sell_price, target_sell_qty,
                        safeback_seconds=RTRADE_DESPERATE_SAFEBACK_SEC,
                        force=False, cancelorders=True, hours=h, smart=False,
                        kind="rtrade_legacy_quote",
                        cache_permit=submit_cache_permit)
                else:
                    sell_order = mkt.place(
                        self.symbol, "SELL", target_sell_price, target_sell_qty,
                        cancelorders=True, hours=RTRADE_SELL_NORMAL_HOURS,
                        smart=False, kind="rtrade_legacy_quote",
                        cache_permit=submit_cache_permit)
            except po.WeightLimitBlock as e:
                print(f"[{self.symbol}] 24h limit reached (SELL) — exiting without retry ({e})")
                return None

            if sell_order is None:
                print(f"[{self.symbol}] Order SELL failed, retryed {failure_count} times. Retrying again ...")
                time.sleep(WAIT_FOR_ORDER)
                failure_count += 1
                if failure_count > max_failures:
                    print(f"[{self.symbol}] Order SELL failed {failure_count} times. Exiting.")
                    return None
                continue
            
            failure_count = 1 # reset failure count after a successful order placement

            time.sleep(WAIT_FOR_ORDER)
            order_id = sell_order['orderId']
            self.filled_sell_price = round(float(sell_order['price']), 4)

            if _order_fully_filled(self.symbol, order_id):
                print(f"[{self.symbol}] SELL order filled at {self.filled_sell_price:.2f}")
                print(f"[{self.symbol}] Starting urgent BUY follow-up (1)")
                mkt.place(self.symbol, "BUY", api.get_current_price(self.symbol) * (1 - RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                    force=_followup_force(self.symbol, "BUY"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                return self.mark_sell_filled(self.filled_sell_price)


            filled_sell_price = mkt.latest_fill_price(
                self.symbol, "SELL", WAIT_FOR_ORDER)
            if filled_sell_price is not None:
                print(f"[{self.symbol}] SELL order may have been filled :-) at {filled_sell_price:.2f}")
                print(f"[{self.symbol}] Starting urgent BUY follow-up (2)")
                mkt.place(self.symbol, "BUY", api.get_current_price(self.symbol) * (1 - RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                    force=_followup_force(self.symbol, "BUY"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                return self.mark_sell_filled(filled_sell_price)

            current_price = api.get_current_price(self.symbol)
            if current_price < filled_buy_price and not u.are_close(current_price, filled_buy_price, RTRADE_BAD_DAY_TOLERANCE_PCT):
                print(f"[{self.symbol}] Bed day :-(. Trying SELL at current price + x2 {current_price:.2f}")
                adjustment_percent = RTRADE_BAD_DAY_MULTIPLIER * self.DEFAULT_ADJUSTMENT_PERCENT
            # Authorize the exact replacement before cancellation, then submit it
            # on the next iteration.
            replacement_price = round(
                current_price * (1 + adjustment_percent), 4)
            replacement_cache_permit = _preflight_routed_order(
                self.symbol, "SELL", target_sell_qty, replacement_price,
                market=False, kind="rtrade_legacy_quote")
            final_status = _cancel_order_confirmed(self.symbol, order_id)
            if not final_status:
                if _order_fully_filled(self.symbol, order_id):
                    print(f"[{self.symbol}] Cancel SELL order failed. Maybe it was filled :-)? Moving to BUY ...")
                    print(f"[{self.symbol}] Starting urgent BUY follow-up (3)")
                    mkt.place(self.symbol, "BUY", api.get_current_price(self.symbol) * (1 - RTRADE_FOLLOWUP_OFFSET_PCT), self.qty,
                        force=_followup_force(self.symbol, "BUY"), cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                    return self.mark_sell_filled(self.filled_sell_price)
                else:
                    print(
                        f"[{self.symbol}] Cancel SELL order remains ambiguous; "
                        "replacement is deferred.")
                    return None
            else:
                remaining_qty = _remaining_after_canceled_order(
                    final_status, target_sell_qty)
                if remaining_qty is None:
                    print(f"[{self.symbol}] Invalid final SELL quantity; replacement deferred.")
                    return None
                if remaining_qty <= max(1e-12, target_sell_qty * 1e-9):
                    print(f"[{self.symbol}] SELL filled during cancellation; moving to BUY.")
                    mkt.place(
                        self.symbol, "BUY",
                        api.get_current_price(self.symbol)
                        * (1 - RTRADE_FOLLOWUP_OFFSET_PCT),
                        self.qty, force=_followup_force(self.symbol, "BUY"),
                        cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS)
                    return self.mark_sell_filled(self.filled_sell_price)
                pending_cache_permit = replacement_cache_permit
                pending_replacement_price = replacement_price
                pending_replacement_qty = remaining_qty

    def _run_pair(self, executor, current_price):
        """Run both sides concurrently on the bot's persistent workers.

        ``Future.result`` propagates worker exceptions to the main loop, which already
        performs defensive reconciliation by canceling recent orders. The old version
        created two threads per round and lost their exceptions on stderr.
        """
        buy_future = executor.submit(
            self.repetitive_buy, current_price, self.filled_sell_price)
        sell_future = executor.submit(
            self.repetitive_sell, current_price, self.filled_buy_price)
        # Wait for both sides before propagating an exception. Resolving BUY immediately
        # could leave the old SELL active after a fast BUY failure while the main loop
        # starts another round over it.
        wait((buy_future, sell_future))
        return buy_future.result(), sell_future.result()

    def _run_coordinator_forever(self):
        pair_store = getattr(self, "pair_store", None) or RTradePairStore()
        venue = _LivePairVenue(self.symbol, pair_store=pair_store)
        policy = PairPolicy(
            adjustment_fraction=self.DEFAULT_ADJUSTMENT_PERCENT,
            quote_ttl_sec=WAIT_FOR_ORDER,
            poll_sec=RTRADE_PAIR_POLL_SEC,
            fast_fill_ratio=RTRADE_FAST_FILL_RATIO,
            min_edge_fraction=RTRADE_MIN_EDGE_PCT,
            shock_hard_stop_fraction=RTRADE_SHOCK_HARD_STOP_PCT,
            hard_stop_fraction=RTRADE_HARD_STOP_PCT,
        )
        if RTRADE_PAIR_MAX_ACTIVE_ROUNDS < 1:
            raise ValueError("RTRADE_PAIR_MAX_ACTIVE_ROUNDS must be >= 1")
        if RTRADE_PAIR_START_INTERVAL_SEC <= 0:
            raise ValueError("RTRADE_PAIR_START_INTERVAL_SEC must be > 0")
        if (not RTRADE_PAIR_DIRECTIONS
                or any(side not in {"BUY", "SELL"} for side in RTRADE_PAIR_DIRECTIONS)):
            raise ValueError("RTRADE_PAIR_DIRECTIONS accepts only BUY,SELL")
        if RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC <= 0:
            raise ValueError("RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC must be > 0")
        if RTRADE_PLACE_FAILURE_BACKOFF_SEC <= 0:
            raise ValueError("RTRADE_PLACE_FAILURE_BACKOFF_SEC must be > 0")
        if RTRADE_DYNAMIC_MARKET_EXIT_MODE not in {"off", "shadow", "live", "emergency_only"}:
            raise ValueError("RTRADE_DYNAMIC_MARKET_EXIT_MODE: off|shadow|live|emergency_only")
        if not RTRADE_HARD_STOP_PCT <= RTRADE_EMERGENCY_HARD_STOP_PCT < 1:
            raise ValueError(
                "RTRADE_EMERGENCY_HARD_STOP_PCT must be >= hard-stop and < 1")

        # Each coordinator exclusively owns one round's order IDs and inventory. An
        # exposed round continues managing its exit without blocking other rounds up
        # to the configured per-symbol limit.
        active = []
        recovery_blocked = False
        for record in pair_store.active(self.symbol):
            state = record.get("state")
            if not state:
                state = {
                    "pair_id": record["pair_id"], "qty": record["qty"],
                    "start_side": record["start_side"], "phase": "quoting",
                    "reason": "recovered_by_client_order_id", "shock": False,
                    "elapsed_sec": max(
                        0.0, time.time() - float(record.get("created_ts") or time.time())),
                    "first_fill_elapsed_sec": None, "first_fill_side": None,
                    "stop_order_id": None,
                    "limit_revisions": {"BUY": 0, "SELL": 0},
                    "hard_stop_revision": 0, "hard_stop_reason": None,
                    "tickets": [], "snapshots": {},
                }
            try:
                # The intent store is fsynced before venue submission, while the
                # checkpoint follows the coordinator step. Merge any intent that
                # raced ahead of the older checkpoint before adopting the round.
                state = venue.merge_checkpoint_intents(record, state)
                intents = record.get("intents", {})
                all_exhausted = bool(intents) and all(
                    isinstance(i, dict) and i.get("recovery_state") in {
                        "absence_confirmed_no_reuse", "horizon_exhausted"}
                    for i in intents.values()
                )
                if not state.get("tickets") and (
                        all_exhausted or state.get("phase") not in {
                            "startup_recovery", "hard_stop_recovery"}):
                    state["phase"] = "failed"
                    state["reason"] = (
                        "recovery_horizon_exhausted" if all_exhausted
                        else "recovery_intent_not_submitted"
                    )
                    pair_store.checkpoint(
                        record["pair_id"], state, terminal=True)
                    print(
                        f"[{self.symbol}] pair={record['pair_id']} recovery: "
                        "exhausted intent or no tickets placed; closed as terminal")
                    continue
                pair_store.checkpoint(
                    record["pair_id"], state, terminal=False)
            except Exception as exc:
                print(
                    f"[{self.symbol}] RECOVERY BLOCKED: "
                    f"pair={record.get('pair_id')} {exc}")
                recovery_blocked = True
                continue
            try:
                coordinator = PairCoordinator.from_state(venue, policy, state)
                active.append(coordinator)
                print(f"[{self.symbol}] pair={coordinator.pair_id} adopted "
                      f"phase={coordinator.phase} tickets={len(coordinator.tickets)}")
            except Exception as exc:
                print(f"[{self.symbol}] RECOVERY BLOCKED: {exc}")
                recovery_blocked = True

        try:
            known_client_ids = {
                intent.get("client_order_id")
                for rec in pair_store.active(self.symbol)
                for intent in rec.get("intents", {}).values()
            }
            orphan_orders = [
                order for order in venue.executor.open_orders(self.symbol)
                if str(order.get("clientOrderId") or "").startswith("RT_")
                and order.get("clientOrderId") not in known_client_ids
            ]
            for order in orphan_orders:
                order_id = str(order["orderId"])
                client_id = order.get("clientOrderId")
                # An RT_ order without local intent cannot be safely associated with a
                # round because its ID is hashed. Cancel it instead of inventing state.
                venue.executor.cancel_order(self.symbol, order_id)
                print(f"[{self.symbol}] recovery: orphaned RT_ order cancelled "
                      f"order_id={order_id} client_id={client_id}")
            if orphan_orders:
                remaining = {
                    str(order.get("orderId"))
                    for order in venue.executor.open_orders(self.symbol)
                }
                not_canceled = [
                    str(order["orderId"]) for order in orphan_orders
                    if str(order["orderId"]) in remaining
                ]
                if not_canceled:
                    print(f"[{self.symbol}] RECOVERY BLOCKED: unconfirmed cancellation "
                          f"for the RT_ orders {not_canceled}")
                    recovery_blocked = True
        except Exception as exc:
            print(f"[{self.symbol}] RECOVERY BLOCKED: exchange inventory unavailable ({exc})")
            recovery_blocked = True
        last_start_at = float("-inf")
        last_recovery_retry = float("-inf")
        recovery_blocked_since = time.monotonic() if recovery_blocked else None
        next_direction = 0
        side_backoff_until = {"BUY": 0.0, "SELL": 0.0}
        while True:
            try:
                now = time.monotonic()

                if recovery_blocked:
                    if recovery_blocked_since is None:
                        recovery_blocked_since = now
                    if now - recovery_blocked_since >= 180.0:
                        print(f"[{self.symbol}] FATAL: startup recovery remained blocked for {now - recovery_blocked_since:.0f}s (>180s). Exiting for orchestrator restart.")
                        sys.exit(1)
                    if now - last_recovery_retry >= 60.0:
                        last_recovery_retry = now
                        try:
                            known_client_ids = {
                                intent.get("client_order_id")
                                for rec in pair_store.active(self.symbol)
                                for intent in rec.get("intents", {}).values()
                            }
                            orphan_orders = [
                                order for order in venue.executor.open_orders(self.symbol)
                                if str(order.get("clientOrderId") or "").startswith("RT_")
                                and order.get("clientOrderId") not in known_client_ids
                            ]
                            for order in orphan_orders:
                                order_id = str(order["orderId"])
                                client_id = order.get("clientOrderId")
                                venue.executor.cancel_order(self.symbol, order_id)
                                print(f"[{self.symbol}] recovery: orphaned RT_ order cancelled "
                                      f"order_id={order_id} client_id={client_id}")
                            recovery_blocked = False
                            recovery_blocked_since = None
                            print(f"[{self.symbol}] startup recovery unblocked: exchange inventory verified")
                        except Exception as exc:
                            print(f"[{self.symbol}] startup recovery retry blocked: {exc}")
                else:
                    recovery_blocked_since = None

                # Only touch heartbeat when the process is healthy and actively trading / coordinating
                if not recovery_blocked and not getattr(venue, "recovery_blocked", False):
                    _touch_rtrade_heartbeat(now=now)

                survivors = []
                checkpoints = []
                for coordinator in active:
                    outcome = coordinator.step(now=now)
                    export_state = getattr(coordinator, "export_state", None)
                    if callable(export_state):
                        checkpoints.append((
                            coordinator.pair_id, export_state(), outcome.terminal))
                    if outcome.terminal:
                        print(
                            f"[{self.symbol}] pair={outcome.pair_id} "
                            f"phase={outcome.phase} shock={outcome.shock} "
                            f"latency={outcome.fill_latency_sec} "
                            f"buy={outcome.buy_qty:.6f} sell={outcome.sell_qty:.6f} "
                            f"net={outcome.net_qty:.6f} "
                            f"cashflow={outcome.gross_pnl:.2f} "
                            f"fees={outcome.fees:.2f} reason={outcome.reason}")
                    else:
                        survivors.append(coordinator)
                pair_store.checkpoint_many(checkpoints)
                active = survivors

                can_start = (
                    not recovery_blocked
                    and not getattr(venue, "recovery_blocked", False)
                    and len(active) < RTRADE_PAIR_MAX_ACTIVE_ROUNDS
                    and now - last_start_at >= RTRADE_PAIR_START_INTERVAL_SEC
                )
                if can_start and not _trend_too_strong(self.symbol):
                    current_price = venue.current_price()
                    if current_price is not None:
                        start_side = None
                        for _ in range(len(RTRADE_PAIR_DIRECTIONS)):
                            candidate = RTRADE_PAIR_DIRECTIONS[next_direction]
                            next_direction = (
                                next_direction + 1) % len(RTRADE_PAIR_DIRECTIONS)
                            if now >= side_backoff_until[candidate]:
                                start_side = candidate
                                break
                        if start_side is None:
                            time.sleep(RTRADE_PAIR_POLL_SEC)
                            continue
                        round_qty = RTRADE_NOTIONAL_USDC / float(current_price)
                        reserved_pair_id = uuid.uuid4().hex
                        coordinator = PairCoordinator(
                            venue, round_qty, policy, start_side=start_side)
                        outcome = coordinator.start(
                            current_price, pair_id=reserved_pair_id)
                        export_state = getattr(coordinator, "export_state", None)
                        last_start_at = now
                        if not outcome.terminal:
                            # Keep the in-memory owner before checkpoint I/O. If
                            # persistence fails after venue acceptance, the same
                            # process must still manage the orders and block new
                            # rounds instead of waiting for a restart.
                            active.append(coordinator)
                        try:
                            if callable(export_state):
                                pair_store.checkpoint(
                                    coordinator.pair_id, export_state(),
                                    terminal=outcome.terminal)
                        except Exception:
                            recovery_blocked = True
                            raise
                        if outcome.terminal:
                            failed_side, backoff_sec = _place_failure_backoff(
                                outcome.reason)
                            if failed_side in side_backoff_until:
                                side_backoff_until[failed_side] = now + backoff_sec
                                print(
                                    f"[{self.symbol}] {failed_side} backoff "
                                    f"{backoff_sec:.0f}s after a placement failure "
                                    f"({outcome.reason})")
                            print(
                                f"[{self.symbol}] pair={outcome.pair_id} "
                                f"direction={start_side}-first "
                                f"phase={outcome.phase} reason={outcome.reason}")
                        else:
                            print(
                                f"[{self.symbol}] pair={outcome.pair_id} started "
                                f"direction={start_side}-first "
                                f"active={len(active)}/"
                                f"{RTRADE_PAIR_MAX_ACTIVE_ROUNDS}")
                time.sleep(RTRADE_PAIR_POLL_SEC)
            except Exception as exc:  # noqa: BLE001 — retain rounds for reconciliation.
                print(f"[{self.symbol}] pair coordinator error: {exc}")
                # Do not globally cancel recent orders in a multi-round registry; that
                # could destroy valid legs owned by other pair IDs.
                time.sleep(RTRADE_PAIR_POLL_SEC)

    def run(self):
        _touch_rtrade_heartbeat(force=True)
        if RTRADE_PAIR_COORDINATOR_ENABLED:
            return self._run_coordinator_forever()
        # Reuse exactly two workers per bot across rounds. BUY and SELL remain concurrent
        # while avoiding thread churn and lost exceptions.
        prefix = f"rtrade-{self.symbol}"
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=prefix) as executor:
            while True:
                _touch_rtrade_heartbeat()
                try:
                    current_price = api.get_current_price(self.symbol)
                    if current_price is None:
                        print(f"[{self.symbol}] Failed to fetch current price. Retrying in {WAIT_FOR_ORDER} seconds...")
                        time.sleep(WAIT_FOR_ORDER)
                        continue
                    print(f"[{self.symbol}] Current price: {current_price:.2f}")

                    # If the asset trends clearly, keep the spread bot idle for this cycle
                    # to avoid catching a falling knife. Retry on the next iteration.
                    if _trend_too_strong(self.symbol):
                        time.sleep(WAIT_FOR_ORDER)
                        continue

                    buy_result, sell_result = self._run_pair(executor, current_price)

                    if not buy_result or not sell_result:
                        continue

                    filled_buy_price = buy_result + RTRADE_ZERO_EPSILON  # avoid zero
                    filled_sell_price = sell_result

                    print(f"[{self.symbol}] Transaction complete: Bought at {filled_buy_price:.2f}, Sold at {filled_sell_price:.2f}")
                    if filled_buy_price < filled_sell_price:
                        print(f"[{self.symbol}] PROFIT: Profit ratio {filled_sell_price / filled_buy_price:.2f}")
                    else:
                        print(f"[{self.symbol}] LOSS: Loss ratio {filled_sell_price / filled_buy_price:.2f}")

                    time.sleep(1)

                    # Reset for the next round.
                    with self.lock:
                        self.buy_filled = self.sell_filled = False
                except Exception as e:
                    print(f"[{self.symbol}] Unexpected error: {e}")
                    # Worker exceptions arrive here through Future.result().
                    api.cancel_recent_orders("SELL", self.symbol, WAIT_FOR_ORDER)
                    api.cancel_recent_orders("BUY", self.symbol, WAIT_FOR_ORDER)
                    time.sleep(1)
                
                
DEFAULT_ADJUSTMENT_PERCENT = required_float_env("RTRADE_DEFAULT_ADJUSTMENT_PCT")
print(f"[INFO] DEFAULT_ADJUSTMENT_PERCENT = {DEFAULT_ADJUSTMENT_PERCENT}")

# Keep actual startup, including WebSocket setup and the infinite live order loop, under
# ``__main__``. Historically it ran during import and could start live trading from a test.
# Fleet startup executes ``python rtrade.py``, so production behavior remains unchanged
# while importing the module is safe.
if __name__ == "__main__":
    # This bot owns one pair. Switching it requires an explicit registry selection;
    # silently choosing the first of several pairs would leave assets unmanaged.
    symbol = single_symbol_for("binance", "rtrade")
    # Explicit user-data bridge: placement guards inspect order/fill history, so the WS
    # keeps that cache fresh instead of relying solely on three-minute polling.
    import cacheManager as cm
    cm.enable_real_ws_event_sync()

    # A VPN/feed blip at startup must not crash with a traceback (which trips the
    # watchdog's traceback alarm at threshold 1 and respawn-loops). Retry briefly for
    # the first price, then exit CLEANLY for the supervisor to relaunch later. Wording
    # avoids "unavailable" so it does not trip the blind regex either.
    initial_price = 0.0
    for attempt in range(1, 13):
        initial_price = float(api.get_current_price(symbol) or 0.0)
        if initial_price > 0:
            break
        print(f"[{symbol}] startup: price feed not ready yet (attempt {attempt}/12); waiting 5s...")
        time.sleep(5)
    if initial_price <= 0:
        print(f"[{symbol}] startup: price feed still not ready after 12 tries; exiting cleanly "
              "for the supervisor to relaunch (transient feed gap, not a fault).")
        raise SystemExit(0)
    initial_qty = RTRADE_NOTIONAL_USDC / initial_price
    bot = TradingBot(symbol, initial_qty,
                     DEFAULT_ADJUSTMENT_PERCENT=DEFAULT_ADJUSTMENT_PERCENT)
    bot.run()

    
