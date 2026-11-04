import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from decimal import Decimal, ROUND_DOWN

# Financial policy and operational invariants are documented in docs/RTRADE.md.
# Effective configuration remains in config.env.

import cacheManager as cm
import external_order_guard as eog
import utils as u
from binance_api import bapi as api
from binance_api import bapi_placeorder as po
from botcore import (
    _parse_bool_flag,
    exact_int,
    load_dotenv as _load_dotenv,
    normalize_epoch_sec,
    required_bool_env,
    required_env,
    required_float_env,
    required_int_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    touch_heartbeat_file,
)
from cachepaths import cache_path, resolve_cachedb_filepath
from instrument_registry import single_symbol_for
from lock import trade_cooldown
from market_regime import MarketRegimeDecision
from order_retry import (
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
    is_terminal_filter_refusal,
    propagate_submission_refusal,
    resolve_accepted_order,
)
from providers.base import extract_client_order_id, extract_order_id, extract_order_price
from providers.execution_audit import AuditedStrategyExecutor
from providers.market_api import api as mkt
from providers.quantity import balance_cap_quantity, decide_quantity
from providers.strategy_executor import ProviderError, SubmissionRefused
from rtrade_pair_store import RTradePairStore, rtrade_client_order_id
from strategies.rtrade_pair import (
    OrderSnapshot as PairOrderSnapshot,
    OrderTicket as PairOrderTicket,
    PairCoordinator,
    PairPolicy,
)

_finite_pos_float = safe_pos_float

_RTRADE_HEARTBEAT_PATH = cache_path("rtrade.heartbeat")
_RTRADE_HEARTBEAT_INTERVAL_SEC = 30.0
_RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC = 120.0
_rtrade_heartbeat_lock = threading.Lock()
_rtrade_heartbeat_last = float("-inf")


def _touch_rtrade_heartbeat(*, force=False, now=None):
    """Publish actual coordinator progress independently from buffered stdout."""
    global _rtrade_heartbeat_last
    now_val = safe_float(now, default=None)
    now_m = time.monotonic() if now_val is None else now_val
    with _rtrade_heartbeat_lock:
        _rtrade_heartbeat_last = touch_heartbeat_file(
            resolve_cachedb_filepath(_RTRADE_HEARTBEAT_PATH),
            last_ts=_rtrade_heartbeat_last,
            interval_sec=_RTRADE_HEARTBEAT_INTERVAL_SEC,
            force=force,
            now=now_m,
        )


_load_dotenv()

WAIT_FOR_ORDER = required_float_env("RTRADE_WAIT_FOR_ORDER_SEC")
MIN_adjustment_percent = required_float_env("RTRADE_MIN_ADJUSTMENT_PCT")
RTRADE_NOTIONAL_USDC = required_float_env("RTRADE_NOTIONAL_USDC")
RTRADE_INITIAL_SPREAD_PCT = required_float_env("RTRADE_INITIAL_SPREAD_PCT")
RTRADE_BUY_DECAY_PCT = required_float_env("RTRADE_BUY_DECAY_PCT")
RTRADE_SELL_DECAY_PCT = required_float_env("RTRADE_SELL_DECAY_PCT")
RTRADE_BUY_DESPERATE_HOURS_BASE = required_float_env("RTRADE_BUY_DESPERATE_HOURS_BASE")
RTRADE_SELL_DESPERATE_HOURS_BASE = required_float_env("RTRADE_SELL_DESPERATE_HOURS_BASE")
RTRADE_DESPERATE_SAFEBACK_SEC = required_float_env("RTRADE_DESPERATE_SAFEBACK_SEC")
RTRADE_BUY_NORMAL_HOURS = required_float_env("RTRADE_BUY_NORMAL_HOURS")
RTRADE_SELL_NORMAL_HOURS = required_float_env("RTRADE_SELL_NORMAL_HOURS")
RTRADE_FOLLOWUP_OFFSET_PCT = required_float_env("RTRADE_FOLLOWUP_OFFSET_PCT")
RTRADE_FOLLOWUP_HOURS = required_float_env("RTRADE_FOLLOWUP_HOURS")
RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED = required_bool_env("RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED")
RTRADE_BAD_DAY_TOLERANCE_PCT = required_float_env("RTRADE_BAD_DAY_TOLERANCE_PCT")
RTRADE_BAD_DAY_MULTIPLIER = required_float_env("RTRADE_BAD_DAY_MULTIPLIER")
RTRADE_ZERO_EPSILON = required_float_env("RTRADE_ZERO_EPSILON")
RTRADE_MAX_FAILURES = required_int_env("RTRADE_MAX_FAILURES")

RTRADE_TREND_FILTER_ENABLED = required_bool_env("RTRADE_TREND_FILTER_ENABLED")
RTRADE_TREND_FILTER_K = required_float_env("RTRADE_TREND_FILTER_K")
RTRADE_TREND_WINDOW_SEC = required_float_env("RTRADE_TREND_WINDOW_SEC")
RTRADE_TREND_OHLC_FALLBACK_ENABLED = required_bool_env("RTRADE_TREND_OHLC_FALLBACK_ENABLED")
RTRADE_DYNAMIC_MARKET_EXIT_MODE = required_env("RTRADE_DYNAMIC_MARKET_EXIT_MODE").lower()
RTRADE_EMERGENCY_HARD_STOP_PCT = required_float_env("RTRADE_EMERGENCY_HARD_STOP_PCT")

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
RTRADE_INTENT_MISSING_CONFIRMATIONS = required_int_env("RTRADE_INTENT_MISSING_CONFIRMATIONS")
RTRADE_INTENT_RECOVERY_HORIZON_SEC = required_float_env("RTRADE_INTENT_RECOVERY_HORIZON_SEC")
RTRADE_FAST_FILL_RATIO = required_float_env("RTRADE_FAST_FILL_RATIO")
RTRADE_MIN_EDGE_PCT = required_float_env("RTRADE_MIN_EDGE_PCT")
RTRADE_SHOCK_HARD_STOP_PCT = required_float_env("RTRADE_SHOCK_HARD_STOP_PCT")
RTRADE_HARD_STOP_PCT = required_float_env("RTRADE_HARD_STOP_PCT")
RTRADE_BYPASS_PROFIT_GUARD = required_bool_env("RTRADE_BYPASS_PROFIT_GUARD")


def _validate_intent_recovery_config(
    missing_confirmations=RTRADE_INTENT_MISSING_CONFIRMATIONS,
    poll_sec=RTRADE_PAIR_POLL_SEC,
    horizon_sec=RTRADE_INTENT_RECOVERY_HORIZON_SEC,
):
    if type(missing_confirmations) is not int or missing_confirmations < 2:
        raise ValueError("RTRADE_INTENT_MISSING_CONFIRMATIONS must be an integer >= 2")
    poll = safe_pos_float(poll_sec, default=None)
    horizon = safe_pos_float(horizon_sec, default=None)
    if poll is None:
        raise ValueError("RTRADE_PAIR_POLL_SEC must be finite and positive")
    if horizon is None or horizon < poll * missing_confirmations:
        raise ValueError("RTRADE_INTENT_RECOVERY_HORIZON_SEC is too short")


_validate_intent_recovery_config()


class _IntentRecoveryPending(RuntimeError):
    """A persisted venue intent is not yet safe to submit or discard."""


class _LivePairVenue:
    """Thin adapter between the pure coordinator and the current Binance venue."""

    def __init__(self, symbol, pair_store=None, *, recovery_clock=time.time):
        self.symbol = symbol
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
        self.audited_executor = AuditedStrategyExecutor(self.executor, venue=provider_name)
        self.pair_store = pair_store
        _validate_intent_recovery_config()
        self.order_lifecycle = TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(self.executor),
            provider_name=provider_name,
            venue=provider_name,
            missing_confirmations=RTRADE_INTENT_MISSING_CONFIRMATIONS,
            retry_on_lookup_error=False,
            audit=self.audited_executor.audit if self.pair_store is not None else None,
        )

    def _set_pair_recovery_blocked(self, pair_id, blocked=True):
        pair_id = str(pair_id or "")
        blocked_flag = _parse_bool_flag(blocked, default=True)
        was_blocked = self.recovery_blocked
        prev_count = len(self._recovery_blocked_pairs)
        if pair_id:
            if blocked_flag:
                self._recovery_blocked_pairs.add(pair_id)
            else:
                self._recovery_blocked_pairs.discard(pair_id)
        self.recovery_blocked = bool(self._recovery_blocked_pairs)
        if self.recovery_blocked != was_blocked or len(self._recovery_blocked_pairs) != prev_count:
            sym = getattr(self, "symbol", "UNKNOWN")
            print(
                f"[{sym}] pair recovery state changed: pair={pair_id} blocked={blocked_flag} "
                f"-> total_blocked={len(self._recovery_blocked_pairs)} {list(self._recovery_blocked_pairs)}"
            )

    def _mark_recovery_pending(self, pair_id, side: str) -> None:
        if pair_id:
            self._set_pair_recovery_blocked(pair_id)
        self._last_place_failures[side] = f"{side.lower()}_recovery_pending"

    def _pair_record(self, pair_id):
        if self.pair_store is None:
            return None
        return next(
            (record for record in self.pair_store.active(self.symbol) if record.get("pair_id") == pair_id),
            None,
        )

    def pair_recovery_blocked(self, pair_id):
        return str(pair_id or "") in self._recovery_blocked_pairs

    @staticmethod
    def _pre_submit_refusal(intent):
        return (
            str(intent.get("submission_outcome") or "").lower() == "refused"
            or str(intent.get("submit_status") or "").lower() == "refused_before_submit"
        )

    def current_price(self):
        return mkt.get_current_price(self.symbol)

    def preflight_order(self, side, qty, price=None, *, market=False, kind=None):
        return self.audited_executor.preflight_order(
            self.symbol, side, qty, price, market=_parse_bool_flag(market, default=False), kind=kind
        )

    @staticmethod
    def _intent_id(pair_id, side, kind):
        return f"rtrade:{pair_id}:{kind}:{str(side).lower()}"

    @staticmethod
    def _parse_bounded_revision(value, *, int_err: str, range_err: str | None = None) -> int:
        rev = exact_int(value, default=None)
        if rev is None:
            raise ValueError(int_err)
        if not 0 <= rev <= 1_000_000:
            raise ValueError(range_err or int_err)
        return rev

    @classmethod
    def _kind_for_revision(cls, prefix: str, revision, *, label: str) -> str:
        rev = cls._parse_bounded_revision(
            revision,
            int_err=f"{label} revision must be an integer",
            range_err=f"{label} revision must be between 0 and 1000000",
        )
        return prefix if rev == 0 else f"{prefix}_{rev}"

    _revision_from_kind = staticmethod(PairCoordinator._intent_revision)

    @classmethod
    def _limit_kind(cls, revision):
        return cls._kind_for_revision("limit", revision, label="limit")

    @classmethod
    def _limit_revision(cls, kind):
        return cls._revision_from_kind(kind, "limit")

    @classmethod
    def _hard_stop_revision_from_kind(cls, kind):
        return cls._revision_from_kind(kind, "hard_stop")

    def _persist_callback(self, pair_id, side, kind, requested_qty):
        if self.pair_store is None:
            return lambda _pending: None
        return lambda pending: self.pair_store.persist_intent(
            pair_id, side, kind, pending, symbol=self.symbol, start_side=side, qty=requested_qty
        )

    def _remember_ticket(self, ticket):
        if ticket is not None and not any(existing.order_id == ticket.order_id for existing in self._known_tickets):
            self._known_tickets.append(ticket)
        return ticket

    def _ticket_from_pending(self, pending, pair_id, *, status=None):
        order_id = extract_order_id(pending.get("order_id"), default=None)
        if not order_id:
            return None
        implied_px = 0.0
        if status is not None:
            fq = _finite_pos_float(getattr(status, "filled_qty", None), default=None)
            fc = _finite_pos_float(getattr(status, "cost", None), default=None)
            if fq is not None and fc is not None:
                implied_px = _finite_pos_float(fc / fq, 0.0)
        ticket_price = (
            _finite_pos_float(
                pending.get("submitted_price") or pending.get("requested_price") or pending.get("price"), 0.0
            )
            or implied_px
            or _finite_pos_float(self.current_price(), 0.0)
        )
        if ticket_price <= 0:
            raise _IntentRecoveryPending(
                f"intent {pending.get('kind')}:{pending.get('side')} has an order_id but no available reference price"
            )
        kind = pending.get("kind")
        rev = self._limit_revision(kind)
        if rev is None:
            rev = self._hard_stop_revision_from_kind(kind) or 0
        ticket = PairOrderTicket(
            order_id=str(order_id),
            side=str(pending["side"]).upper(),
            price=ticket_price,
            qty=float(pending.get("submitted_qty") or pending.get("requested_qty")),
            pair_id=pair_id,
            revision=rev,
        )
        return self._remember_ticket(ticket)

    def _submit_limit_intent(self, side, price, qty, pair_id, *, attempt=1, cache_permit=None, revision=0):
        side = side.upper()
        hours = RTRADE_BUY_NORMAL_HOURS if side == "BUY" else RTRADE_SELL_NORMAL_HOURS
        store_kind = self._limit_kind(revision)
        client_id = rtrade_client_order_id(pair_id, side, store_kind)
        intent = self.order_lifecycle.new_intent(
            intent_id=self._intent_id(pair_id, side, store_kind),
            client_order_id=client_id, symbol=self.symbol, side=side,
            requested_qty=qty, requested_price=price, kind=store_kind,
            attempt=attempt, metadata={"pair_id": pair_id, "limit_revision": int(revision)},
        )
        persist = self._persist_callback(pair_id, side, store_kind, requested_qty=qty)
        outcome_context = {}

        def submit_once():
            response = mkt.place(
                self.symbol, side, price, qty,
                force=False, cancelorders=False, hours=hours, smart=False,
                cooldown_pair_id=pair_id,
                bypass_guards={"G_PROF", "GLLM1"} if RTRADE_BYPASS_PROFIT_GUARD else {"GLLM1"},
                allow_math_reference_sell=bool(RTRADE_BYPASS_PROFIT_GUARD and side == "SELL"),
                caller_owns_retry=True, motivation="rtrade_pair_quote",
                client_order_id=client_id, _outcome_context=outcome_context,
                cache_permit=cache_permit,
            )
            return propagate_submission_refusal(response, outcome_context)

        result = self.order_lifecycle.submit(intent, persist=persist, submit=submit_once)
        return self._ticket_from_pending(result.intent, pair_id), result.intent

    def _canonical_intent(self, record, stored):
        """Upgrade an old rtrade intent in memory without losing observations."""
        pending = dict(stored)
        side = str(pending.get("side") or record["start_side"]).upper()
        kind = str(pending.get("kind") or "limit")
        requested_qty = pending.get("requested_qty", pending.get("qty", record.get("qty")))
        requested_price = pending.get("requested_price", pending.get("price"))
        req_qty_f = _finite_pos_float(requested_qty, default=None)
        if req_qty_f is None:
            raise ValueError(f"invalid requested_qty in rtrade intent {kind}:{side}")
        req_px_f = None if requested_price is None else _finite_pos_float(requested_price, default=None)
        if requested_price is not None and req_px_f is None:
            raise ValueError(f"invalid requested_price in rtrade intent {kind}:{side}")
        raw_created = pending.get("created_at") or record.get("created_ts")
        created_f = normalize_epoch_sec(raw_created, default=None)
        pending.update({
            "intent_id": pending.get("intent_id") or self._intent_id(record["pair_id"], side, kind),
            "client_order_id": pending["client_order_id"],
            "symbol": pending.get("symbol") or record["symbol"],
            "side": side,
            "kind": kind,
            "requested_qty": req_qty_f,
            "requested_price": req_px_f,
            "attempt": safe_pos_int(pending.get("attempt"), default=1),
            "created_at": created_f if created_f is not None else time.time(),
            "lookup_misses": safe_nonneg_int(pending.get("lookup_misses"), default=0),
            "pair_id": pending.get("pair_id") or record["pair_id"],
        })
        return pending

    def recover_intent(self, record, stored):
        """Recover one fsynced pair intent without guessing venue state."""
        pending = self._canonical_intent(record, stored)
        side, kind, pair_id = pending["side"], pending["kind"], record["pair_id"]
        persist_raw = self._persist_callback(pair_id, side, kind, requested_qty=pending["requested_qty"])
        now = safe_pos_float(self._recovery_clock(), default=None)
        if now is None:
            raise ValueError("rtrade recovery clock returned invalid time")
        if not pending.get("order_id"):
            recovery_started_at = pending.get("recovery_started_at")
            if recovery_started_at is not None:
                r_started = safe_float(recovery_started_at, default=None)
                if r_started is None:
                    raise ValueError(f"{kind}:{side} has invalid recovery timing")
                if max(0.0, now - r_started) >= RTRADE_INTENT_RECOVERY_HORIZON_SEC:
                    pending["recovery_state"] = "horizon_exhausted"
                    pending["recovery_horizon_exhausted_at"] = now
                    persist_raw(pending)
                    raise _IntentRecoveryPending(f"{kind}:{side} recovery horizon is exhausted")
            if pending.get("recovery_state") in {"absence_confirmed_no_reuse", "horizon_exhausted"}:
                persist_raw(pending)
                raise _IntentRecoveryPending(f"{kind}:{side} requires manual reconciliation")
            next_recovery_at = pending.get("next_recovery_at")
            if next_recovery_at is not None:
                n_rec = safe_float(next_recovery_at, default=None)
                if n_rec is None:
                    raise ValueError(f"{kind}:{side} has invalid next recovery time")
                if now < n_rec:
                    raise _IntentRecoveryPending(f"{kind}:{side} reconciliation is not due yet")

        retained = dict(pending)

        def persist(value):
            nonlocal retained
            if value is None:
                retained.update({
                    "recovery_state": "absence_confirmed_no_reuse",
                    "absence_confirmed_at": now,
                    "last_recovery_lookup_at": now,
                    "next_recovery_at": now + RTRADE_PAIR_POLL_SEC,
                })
                persist_raw(dict(retained))
                return
            retained = dict(value)
            if retained.get("order_id"):
                for field in (
                    "lookup_error", "recovery_state", "recovery_started_at",
                    "last_recovery_lookup_at", "next_recovery_at",
                    "absence_confirmed_at", "recovery_horizon_exhausted_at",
                ):
                    retained.pop(field, None)
            elif "lookup_error" in retained or int(retained.get("lookup_misses") or 0) > 0:
                retained.setdefault("recovery_started_at", now)
                retained["last_recovery_lookup_at"] = now
                retained["next_recovery_at"] = now + RTRADE_PAIR_POLL_SEC
            persist_raw(dict(retained))

        persist_raw(pending)
        result = self.order_lifecycle.reconcile(pending, persist=persist)
        if result.outcome == "absent":
            raise _IntentRecoveryPending(f"{kind}:{side} absence is confirmed; client-ID reuse is blocked")
        if not result.intent.get("order_id"):
            raise _IntentRecoveryPending(f"{kind}:{side} intent is ambiguous; lookup/status is unavailable")
        if result.status is None:
            raise _IntentRecoveryPending(f"intent {kind}:{side} has an order_id but no available status")

        ticket = self._ticket_from_pending(result.intent, pair_id, status=result.status)
        snapshot = PairOrderSnapshot(
            status=result.status.status,
            filled_qty=result.status.filled_qty,
            cost=result.status.cost,
            fee=result.status.fee,
        )
        return ticket, snapshot

    def _reconcile_existing_intent(self, pair_id, side, kind):
        """Return state, ticket, and snapshot for one persisted submit slot."""
        record = self._pair_record(pair_id) if self.pair_store is not None else None
        stored = record.get("intents", {}).get(f"{kind}:{side}") if record else None
        if stored is None:
            return "none", None, None
        if self._pre_submit_refusal(stored):
            self.pair_store.persist_intent(pair_id, side, kind, None, symbol=self.symbol)
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
            raise _IntentRecoveryPending(f"pair={pair_id}: persisted recovery record is unavailable")
        recovered = []
        pending_recovery = False
        for stored in list(dict(record.get("intents") or {}).values()):
            pending = self._canonical_intent(record, stored)
            side, kind = pending["side"], pending["kind"]
            if self._pre_submit_refusal(pending):
                self.pair_store.persist_intent(pair_id, side, kind, None, symbol=self.symbol)
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

    def merge_checkpoint_intents(self, record, state):
        """Merge fsynced intents that raced ahead of a coordinator checkpoint."""
        merged = dict(state or {})
        tickets = [dict(ticket) for ticket in merged.get("tickets", [])]
        snapshots = {str(oid): dict(snap) for oid, snap in dict(merged.get("snapshots") or {}).items()}
        represented = {str(t.get("order_id")) for t in tickets if t.get("order_id") is not None}
        limit_revisions = {"BUY": 0, "SELL": 0}
        for side, value in dict(merged.get("limit_revisions") or {}).items():
            side_u = str(side).upper()
            if side_u in limit_revisions:
                limit_revisions[side_u] = self._parse_bounded_revision(
                    0 if value is None else value, int_err=f"invalid {side_u} limit revision: {value!r}"
                )
        raw_hs_rev = merged.get("hard_stop_revision", 0)
        hard_stop_revision = self._parse_bounded_revision(
            0 if raw_hs_rev is None else raw_hs_rev, int_err=f"invalid hard-stop revision: {raw_hs_rev!r}"
        )
        best_stop_revision = hard_stop_revision if merged.get("stop_order_id") is not None else -1
        recovery_pending = False
        pending_hard_stop = False

        for stored in dict(record.get("intents") or {}).values():
            pending = self._canonical_intent(record, stored)
            side, kind = pending["side"], pending["kind"]
            limit_revision = self._limit_revision(kind)
            stop_revision = self._hard_stop_revision_from_kind(kind)
            if limit_revision is not None:
                limit_revisions[side] = max(limit_revisions[side], limit_revision)
            if stop_revision is not None:
                hard_stop_revision = max(hard_stop_revision, stop_revision)
            if self._pre_submit_refusal(pending):
                self.pair_store.persist_intent(record["pair_id"], side, kind, None, symbol=self.symbol)
                continue

            order_id = pending.get("order_id")
            if order_id is not None and str(order_id) in represented:
                represented_revision = limit_revision if limit_revision is not None else stop_revision
                if represented_revision is not None:
                    for ticket in tickets:
                        if str(ticket.get("order_id")) == str(order_id):
                            ticket["revision"] = represented_revision
                            break
                if stop_revision is not None and stop_revision >= best_stop_revision:
                    best_stop_revision = stop_revision
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
                ticket.active = snapshot.status not in {"closed", "canceled", "expired"}
                tickets.append(vars(ticket).copy())
                represented.add(order_id)
            snapshots[order_id] = vars(snapshot).copy()
            if stop_revision is not None and stop_revision >= best_stop_revision:
                best_stop_revision = stop_revision
                merged["stop_order_id"] = order_id
                merged["phase"] = "stopping"
                merged["hard_stop_reason"] = merged.get("hard_stop_reason") or "inventory_hard_stop"

        merged.update({
            "tickets": tickets,
            "snapshots": snapshots,
            "limit_revisions": limit_revisions,
            "hard_stop_revision": hard_stop_revision,
        })
        if recovery_pending:
            merged["phase"] = "hard_stop_recovery" if pending_hard_stop else "startup_recovery"
            merged["recovery_pending"] = True
            self._set_pair_recovery_blocked(record["pair_id"])
        else:
            merged.pop("recovery_pending", None)
            self._set_pair_recovery_blocked(record["pair_id"], False)
        return merged

    def place_limit(self, side, price, qty, pair_id, *, cache_permit=None, revision=0):
        side = side.upper()
        store_kind = self._limit_kind(revision)
        self._last_place_failures.pop(side, None)
        existing_state, existing_ticket, _snapshot = self._reconcile_existing_intent(pair_id, side, store_kind)
        if existing_state == "recovered":
            return existing_ticket
        if existing_state == "pending":
            self._last_place_failures[side] = f"{side.lower()}_recovery_pending"
            return None
        available_qty, required_asset = balance_cap_quantity(self.executor.free_balance, self.symbol, side, price)
        if required_asset and available_qty is not None and float(available_qty) <= 1e-12:
            self._last_place_failures[side] = f"{side.lower()}_insufficient_funds:{required_asset}"
            print(f"[{self.symbol}] {side} insufficient funds: available=0.00000000 {required_asset}")
            return None
        ticket, _pending = self._submit_limit_intent(
            side, price, qty, pair_id, cache_permit=cache_permit, revision=revision
        )
        refusal = _pending.get("refusal_reason")
        if ticket is None and refusal:
            if refusal in {"insufficient_funds", "balance_unavailable"} or (
                (refusal == "below_min_notional" or is_terminal_filter_refusal(refusal))
                and available_qty is not None
                and float(available_qty) < float(qty)
            ):
                self._last_place_failures[side] = f"{side.lower()}_insufficient_funds:{required_asset or 'UNKNOWN'}"
            return None
        if ticket is None and self.pair_store is not None:
            self._mark_recovery_pending(pair_id, side)
        elif ticket is not None:
            self._set_pair_recovery_blocked(pair_id, False)
        return ticket

    def last_place_failure_reason(self, side):
        return self._last_place_failures.get(side.upper())

    def market_exit_allowed(self, exposure_side, loss_fraction, reason):
        """Decide whether a normal hard stop has trend confirmation for MARKET."""
        emergency = loss_fraction >= RTRADE_EMERGENCY_HARD_STOP_PCT
        decision = _market_regime_decision(self.symbol)
        justified = decision.adverse_to(exposure_side) or emergency
        print(
            f"[{self.symbol}] dynamic-market-exit mode={RTRADE_DYNAMIC_MARKET_EXIT_MODE} "
            f"exposure={exposure_side} regime={decision.regime} "
            f"strength={decision.strength} signal_reason={decision.reason} "
            f"loss={loss_fraction:.4%} emergency={emergency} "
            f"would_market={justified} reason={reason}"
        )
        if RTRADE_DYNAMIC_MARKET_EXIT_MODE == "emergency_only":
            return emergency
        if RTRADE_DYNAMIC_MARKET_EXIT_MODE in {"off", "shadow"}:
            return True
        return justified

    def _update_stored_intent(self, pair_id, side: str, store_kind: str, **updates) -> None:
        record = self._pair_record(pair_id)
        stored = None if record is None else record.get("intents", {}).get(f"{store_kind}:{side}")
        if stored is not None:
            pending = self._canonical_intent(record, stored)
            pending.update(updates)
            self.pair_store.persist_intent(pair_id, side, store_kind, pending, symbol=self.symbol)

    def place_market_exit(self, side, qty, reason, pair_id=None, *, cache_permit=None, revision=0):
        """Place a spot market exit reserved for reducing hard-stop exposure."""
        side = side.upper()
        store_kind = self._kind_for_revision("hard_stop", revision, label="hard-stop")
        revision = int(store_kind.removeprefix("hard_stop_")) if store_kind != "hard_stop" else 0
        self._last_place_failures.pop(side, None)
        if pair_id:
            existing_state, existing_ticket, _snapshot = self._reconcile_existing_intent(pair_id, side, store_kind)
            if existing_state == "recovered":
                return existing_ticket
            if existing_state == "pending":
                self._last_place_failures[side] = f"{side.lower()}_recovery_pending"
                return None
        price = _finite_pos_float(self.current_price(), 0.0)
        if price <= 0:
            print(f"[{self.symbol}] {side} hard-stop BLOCKED: price unavailable")
            return None
        decision = decide_quantity(
            self.executor, self.symbol, side, price, qty,
            apply_policy=False, market=True, enforce_business_minimum=False,
        )
        final_qty = float(decision.final_qty)
        if final_qty <= 0:
            print(f"[{self.symbol}] {side} hard-stop BLOCKED: {decision.refuse_reason} asset={decision.balance_asset}")
            return None
        precision = self.audited_executor.pair_precision(self.symbol)
        if precision is not None:
            quantum = Decimal(1).scaleb(-int(precision.volume_decimals))
            final_qty = float(Decimal(str(final_qty)).quantize(quantum, rounding=ROUND_DOWN))
            if final_qty <= 0 or final_qty < float(precision.order_min or 0.0):
                print(f"[{self.symbol}] {side} hard-stop BLOCKED: qty {final_qty} below the minimum {precision.order_min}")
                return None
        kind = f"rtrade:{reason}:{pair_id or 'unknown-pair'}:revision:{revision}"
        intent_id = f"rtrade-{pair_id or 'unknown'}-{reason}-{side.lower()}-r{revision}"
        client_id = rtrade_client_order_id(pair_id or "unknown", side, store_kind)
        replacement_after_cancel = cache_permit is not None
        if cache_permit is None:
            cache_permit = self.preflight_order(side, final_qty, price=None, market=True, kind=kind)
        if self.pair_store is not None and pair_id:
            self.pair_store.intent(pair_id, side, price, final_qty, client_id, kind=store_kind, symbol=self.symbol)
        try:
            order_id = self.audited_executor.submit_order_with_intent(
                intent_id, self.symbol, side, final_qty, price=None,
                market=True, kind=kind, reference_price=price,
                client_order_id=client_id, cache_permit=cache_permit,
            )
        except SubmissionRefused as exc:
            if self.pair_store is not None and pair_id:
                if not replacement_after_cancel:
                    self.pair_store.persist_intent(pair_id, side, store_kind, None, symbol=self.symbol)
                else:
                    self._update_stored_intent(
                        pair_id, side, store_kind,
                        submission_outcome="refused",
                        submit_status="refused_before_submit",
                        refusal_reason=exc.reason,
                    )
            raise
        except ProviderError as exc:
            if self.pair_store is not None and pair_id:
                self._update_stored_intent(
                    pair_id, side, store_kind,
                    submission_outcome="unknown",
                    submit_error=f"{exc.__class__.__name__}: {exc}",
                )
                self._mark_recovery_pending(pair_id, side)
            print(f"[{self.symbol}] {side} hard-stop response is ambiguous; the persisted client ID must be reconciled")
            return None
        clean_oid = extract_order_id(order_id, default=None)
        if not clean_oid:
            if self.pair_store is not None and pair_id:
                self._mark_recovery_pending(pair_id, side)
            return None
        ticket = PairOrderTicket(
            order_id=clean_oid, side=side, price=price, qty=final_qty, pair_id=pair_id, revision=revision
        )
        self._known_tickets.append(ticket)
        if self.pair_store is not None and pair_id:
            self.pair_store.accepted(pair_id, side, ticket.order_id, kind=store_kind)
            self._set_pair_recovery_blocked(pair_id, False)
        return ticket

    def order_status(self, order_id):
        status = self.executor.order_status(self.symbol, str(order_id))
        return PairOrderSnapshot(
            status=status.status, filled_qty=status.filled_qty, cost=status.cost, fee=status.fee
        )

    def cancel(self, order_id):
        try:
            self.executor.cancel_order(self.symbol, str(order_id))
            for ticket in getattr(self, "_known_tickets", []):
                if ticket.order_id == str(order_id) and ticket.pair_id:
                    trade_cooldown.release_pair_leg(self.symbol, ticket.pair_id, ticket.side)
                    break
            return True
        except Exception as exc:  # noqa: BLE001 — the coordinator decides how to fail closed.
            print(f"[{self.symbol}] pair cancel {order_id} failed: {exc}")
            return False


def _trend_too_strong(symbol):
    """Return whether the regime should keep the spread bot idle (fail-closed on unknown/directional)."""
    if not RTRADE_TREND_FILTER_ENABLED:
        return False
    decision = _market_regime_decision(symbol)
    stand_aside = decision.directional or decision.regime == "unknown"
    if stand_aside:
        print(
            f"[{symbol}] rtrade STANDING ASIDE: regime {decision.regime} "
            f"strength={decision.strength} reason={decision.reason} window={RTRADE_TREND_WINDOW_SEC:.0f}s"
        )
    return stand_aside


def _effective_trend_filter_k(snapshot) -> float:
    """Scale RTRADE_TREND_FILTER_K by sqrt(10 / m) for dense multi-step dynamic windows."""
    base_k = float(RTRADE_TREND_FILTER_K)
    if not isinstance(snapshot, dict):
        return base_k
    n_f = safe_float(snapshot.get("n_samples"), default=None)
    return base_k * math.sqrt(10.0 / (n_f - 1.0)) if (n_f is not None and n_f - 1.0 > 10.0) else base_k


def _market_regime_decision(symbol) -> MarketRegimeDecision:
    """Adapt the existing cache source to the provider-neutral regime evaluator."""
    dyn, allow_fb = None, False
    if RTRADE_TREND_FILTER_ENABLED:
        allow_fb = RTRADE_TREND_OHLC_FALLBACK_ENABLED
        try:
            import cacheManager as cm
            dyn = cm.get_short_trend_manager().get_short_trend_for_window(symbol, RTRADE_TREND_WINDOW_SEC)
        except Exception:
            dyn = None
    return mkt.market_regime(
        symbol,
        horizon="short",
        snapshot=dyn or {},
        snapshot_source="cache24_dynamic",
        strength_threshold=_effective_trend_filter_k(dyn),
        allow_fallback=allow_fb,
    )


def _is_gi_whale_enforced(bypass_guards=None) -> bool:
    """Return True if GI_WHALE is actively enforced and not bypassed in order_guard.conf."""
    return sys.modules.get("external_order_guard", eog).is_gi_whale_enforced(bypass_guards, venue="binance")


def _whale_flow_blocks_side(symbol: str, side: str, *, bypass_guards=None) -> tuple[bool, str]:
    """Check if whale taker flow or OI divergence conflicts with initiating an rtrade round."""
    return sys.modules.get("external_order_guard", eog).evaluate_whale_spread_flow(
        symbol, side, bypass_guards=bypass_guards, enforced=_is_gi_whale_enforced(bypass_guards)
    )


def _effective_adjustment_fraction(symbol: str, base_adjustment: float, *, bypass_guards=None) -> float:
    """Widen grid quote adjustment when moderate taker flow pressure exists."""
    return sys.modules.get("external_order_guard", eog).resolve_whale_spread_adjustment(
        symbol, base_adjustment, bypass_guards=bypass_guards, enforced=_is_gi_whale_enforced(bypass_guards)
    )


def _order_fully_filled(symbol, order_id):
    """Support the legacy loop through provider-neutral order status."""
    if not order_id:
        return False
    try:
        return mkt.order_status(symbol, str(order_id)).fully_filled
    except Exception as exc:
        print(f"[{symbol}] order status {order_id} unavailable: {exc}")
        return False


def _preflight_routed_order(symbol, side, qty, price=None, *, market=False, kind=None):
    provider_name = mkt.provider_name_for(symbol)
    executor = mkt.provider_by_name(provider_name)
    if executor is None:
        raise RuntimeError(f"provider executor unavailable for {symbol}")
    preflight = getattr(executor, "preflight_order", None)
    if not callable(preflight):
        raise RuntimeError(f"{provider_name} preflight unavailable for {symbol}")
    return preflight(symbol, side, qty, price, market=market, kind=kind)


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
    try:
        resolve_accepted_order(symbol, order_id)
    except Exception:
        pass
    return status


def _remaining_after_canceled_order(status, submitted_qty):
    """Return the unfilled terminal remainder, or ``None`` for invalid state."""
    sub_f = _finite_pos_float(submitted_qty, default=None)
    fill_f = safe_nonneg_float(getattr(status, "filled_qty", None), default=None)
    if sub_f is None or fill_f is None or fill_f > sub_f + max(1e-12, sub_f * 1e-9):
        return None
    return 0.0 if status.fully_filled else max(0.0, sub_f - fill_f)


def _place_failure_backoff(reason):
    """Return ``(side, seconds)`` for a terminal placement failure."""
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


def _followup_exit_price(symbol, side, fill_ref_price):
    """Anchor follow-up exit price to ensure the limit order targets profit."""
    current_price = _finite_pos_float(api.get_current_price(symbol), 0.0)
    fill_ref_price = _finite_pos_float(fill_ref_price, 0.0) or current_price
    if current_price <= 0:
        current_price = fill_ref_price
    side_u = str(side).upper()
    if side_u == "SELL":
        return round(max(current_price, fill_ref_price) * (1 + RTRADE_FOLLOWUP_OFFSET_PCT), 4)
    if side_u == "BUY":
        return round(min(current_price, fill_ref_price) * (1 - RTRADE_FOLLOWUP_OFFSET_PCT), 4)
    raise ValueError(f"Invalid side: {side}")


def _followup_force(symbol, side):
    """Choose MARKET force for a post-fill flip only when trend is not adverse."""
    su = (side or "").upper()
    if not RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED:
        print(f"[{symbol}] follow-up {su}: market flip disabled -> use a patient limit, NOT a market order")
        return False
    if not RTRADE_TREND_FILTER_ENABLED:
        return True
    decision = _market_regime_decision(symbol)
    if decision.regime == "unknown":
        print(f"[{symbol}] follow-up {su}: trend data unavailable -> patient limit, NOT a market order")
        return False
    if not decision.directional:
        return True
    if decision.adverse_to("LONG" if su == "SELL" else "SOLD"):
        print(f"[{symbol}] follow-up {su}: adverse {decision.regime} regime -> use a patient limit, NOT a market order")
        return False
    return True


_TERMINAL_QTY_OR_LIMIT_REFUSALS = frozenset({
    "qty_zero_after_weight", "qty_zero_after_policy", "below_min_notional",
    "insufficient_funds", "balance_unavailable", "medium_trend_qty_zero", "daily_limit",
})

_TERMINAL_UNFILLED_REFUSALS = _TERMINAL_QTY_OR_LIMIT_REFUSALS | frozenset({
    "profit_guard", "trend_deferred", "medium_trend_deferred", "aborted_by_caller",
})


def _is_terminal_unfilled_refusal(refusal_reason) -> bool:
    return bool(refusal_reason) and (
        refusal_reason in _TERMINAL_UNFILLED_REFUSALS or is_terminal_filter_refusal(refusal_reason)
    )


def _validate_coordinator_config() -> None:
    if RTRADE_PAIR_MAX_ACTIVE_ROUNDS < 1:
        raise ValueError("RTRADE_PAIR_MAX_ACTIVE_ROUNDS must be >= 1")
    if RTRADE_PAIR_START_INTERVAL_SEC <= 0:
        raise ValueError("RTRADE_PAIR_START_INTERVAL_SEC must be > 0")
    if not RTRADE_PAIR_DIRECTIONS or any(s not in {"BUY", "SELL"} for s in RTRADE_PAIR_DIRECTIONS):
        raise ValueError("RTRADE_PAIR_DIRECTIONS accepts only BUY,SELL")
    if RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC <= 0:
        raise ValueError("RTRADE_INSUFFICIENT_FUNDS_BACKOFF_SEC must be > 0")
    if RTRADE_PLACE_FAILURE_BACKOFF_SEC <= 0:
        raise ValueError("RTRADE_PLACE_FAILURE_BACKOFF_SEC must be > 0")
    if RTRADE_DYNAMIC_MARKET_EXIT_MODE not in {"off", "shadow", "live", "emergency_only"}:
        raise ValueError("RTRADE_DYNAMIC_MARKET_EXIT_MODE: off|shadow|live|emergency_only")
    if not RTRADE_HARD_STOP_PCT <= RTRADE_EMERGENCY_HARD_STOP_PCT < 1:
        raise ValueError("RTRADE_EMERGENCY_HARD_STOP_PCT must be >= hard-stop and < 1")


class TradingBot:
    def __init__(self, symbol, qty, DEFAULT_ADJUSTMENT_PERCENT):
        self.symbol = symbol
        self.qty = qty
        self.transaction_state = "COMPLETED"
        current_price = _finite_pos_float(api.get_current_price(symbol), 0.0)
        try:
            last_buy = _finite_pos_float(mkt.latest_fill_price(symbol, "BUY", RTRADE_BUY_NORMAL_HOURS * 3600.0), 0.0)
            last_sell = _finite_pos_float(mkt.latest_fill_price(symbol, "SELL", RTRADE_SELL_NORMAL_HOURS * 3600.0), 0.0)
        except Exception:
            last_buy = last_sell = 0.0
        self.filled_buy_price = round(
            last_buy if last_buy > 0 else current_price * (1 - RTRADE_INITIAL_SPREAD_PCT), 4
        )
        self.filled_sell_price = round(
            last_sell if last_sell > 0 else current_price * (1 + RTRADE_INITIAL_SPREAD_PCT), 4
        )
        self.buy_filled = False
        self.sell_filled = False
        self.DEFAULT_ADJUSTMENT_PERCENT = DEFAULT_ADJUSTMENT_PERCENT
        self.lock = threading.RLock()
        self._needs_startup_reconcile = True

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
            self.buy_filled, self.sell_filled = True, False
            if filled_buy_price:
                self.filled_buy_price = filled_buy_price
            return self.filled_buy_price

    def mark_sell_filled(self, filled_sell_price=None):
        with self.lock:
            self.buy_filled, self.sell_filled = False, True
            if filled_sell_price:
                self.filled_sell_price = filled_sell_price
            return self.filled_sell_price

    def _get_pair_cond(self):
        cond = getattr(self, "_pair_precheck_cond", None)
        if cond is None:
            cond = threading.Condition(threading.RLock())
            self._pair_precheck_cond = cond
        return cond

    def _abort_round(self):
        cond = self._get_pair_cond()
        with cond:
            self._round_aborted = True
            pending = getattr(self, "_pair_precheck_pending", None)
            if isinstance(pending, set):
                pending.clear()
            cond.notify_all()

    def _finish_side_precheck(self, side, *, aborted=False):
        cond = self._get_pair_cond()
        with cond:
            if aborted:
                self._round_aborted = True
            pending = getattr(self, "_pair_precheck_pending", None)
            if isinstance(pending, set):
                pending.discard(side)
                if aborted:
                    pending.clear()
            cond.notify_all()

    def _pair_abort_if(self, side, opposite_filled_fn):
        cond = self._get_pair_cond()
        with cond:
            if opposite_filled_fn():
                return False
            if getattr(self, "_round_aborted", False):
                return True
            pending = getattr(self, "_pair_precheck_pending", None)
            if isinstance(pending, set):
                pending.discard(side)
                cond.notify_all()
                deadline = time.monotonic() + 10.0
                while pending and not getattr(self, "_round_aborted", False) and not opposite_filled_fn():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    cond.wait(timeout=remaining)
            return bool(getattr(self, "_round_aborted", False) and not opposite_filled_fn())

    def _place_legacy_side_order(
        self, side: str, is_buy: bool, is_opp_filled, target_price: float, target_qty: float,
        adjustment_percent: float, failure_count: int, desp_hours_base: float, normal_hours: float,
        submit_cache_permit, outcome_context: dict,
    ):
        if is_opp_filled():
            if adjustment_percent == MIN_adjustment_percent:
                print(f"[{self.symbol}] entering the urgent {side} path")
            desp_extra = {} if is_buy else {"allow_math_reference_sell": True}
            return mkt.place(
                self.symbol, side, target_price, target_qty,
                safeback_seconds=RTRADE_DESPERATE_SAFEBACK_SEC,
                force=False, cancelorders=True, hours=desp_hours_base / failure_count, smart=False,
                bypass_guards={"G_PROF", "GLLM1"},
                kind="rtrade_legacy_quote", caller_owns_retry=True,
                cache_permit=submit_cache_permit, _outcome_context=outcome_context,
                **desp_extra,
            )
        extra_place_kwargs = {}
        if isinstance(getattr(self, "_pair_precheck_pending", None), set):
            abort_fn = lambda: self._pair_abort_if(side, is_opp_filled)
            abort_fn.is_aborted = lambda: bool(getattr(self, "_round_aborted", False) and not is_opp_filled())
            extra_place_kwargs["abort_if"] = abort_fn
        if not is_buy:
            extra_place_kwargs["allow_math_reference_sell"] = bool(RTRADE_BYPASS_PROFIT_GUARD)
        return mkt.place(
            self.symbol, side, target_price, target_qty,
            cancelorders=True, hours=normal_hours,
            smart=False, kind="rtrade_legacy_quote",
            bypass_guards={"G_PROF", "GLLM1"} if RTRADE_BYPASS_PROFIT_GUARD else {"GLLM1"},
            caller_owns_retry=True,
            cache_permit=submit_cache_permit, _outcome_context=outcome_context,
            **extra_place_kwargs,
        )

    def _should_exit_on_legacy_refusal(
        self, side: str, opp_side: str, is_buy: bool, is_opp_filled, refusal_reason
    ) -> bool:
        if refusal_reason in _TERMINAL_QTY_OR_LIMIT_REFUSALS or is_terminal_filter_refusal(refusal_reason):
            print(f"[{self.symbol}] 24h limit reached ({side}: {refusal_reason}) — exiting without retry")
            return True
        if refusal_reason == "profit_guard" and (is_buy or not is_opp_filled()):
            print(f"[{self.symbol}] Blocked by profit guard ({side}: {refusal_reason}) — exiting without retry")
            return True
        if refusal_reason in {"trend_deferred", "medium_trend_deferred"} and not is_opp_filled():
            print(f"[{self.symbol}] Deferred by trend ({side}: {refusal_reason}) — exiting without retry")
            return True
        if (refusal_reason == "aborted_by_caller" or getattr(self, "_round_aborted", False)) and not is_opp_filled():
            print(f"[{self.symbol}] Aborting {side} loop because {opp_side} worker exited without fill.")
            return True
        return False

    def _repetitive_side_loop(self, side: str, current_price, filled_opp_price):
        is_buy = side == "BUY"
        opp_side = "SELL" if is_buy else "BUY"
        filled_attr = "filled_buy_price" if is_buy else "filled_sell_price"
        is_side_filled = (lambda: self.is_buy_filled) if is_buy else (lambda: self.is_sell_filled)
        is_opp_filled = (lambda: self.is_sell_filled) if is_buy else (lambda: self.is_buy_filled)
        mark_filled = self.mark_buy_filled if is_buy else self.mark_sell_filled
        decay_pct = RTRADE_BUY_DECAY_PCT if is_buy else RTRADE_SELL_DECAY_PCT
        desp_hours_base = RTRADE_BUY_DESPERATE_HOURS_BASE if is_buy else RTRADE_SELL_DESPERATE_HOURS_BASE
        normal_hours = RTRADE_BUY_NORMAL_HOURS if is_buy else RTRADE_SELL_NORMAL_HOURS
        sign = -1.0 if is_buy else 1.0

        adjustment_percent = self.DEFAULT_ADJUSTMENT_PERCENT
        failure_count = 1
        max_failures = RTRADE_MAX_FAILURES
        pending_cache_permit = pending_replacement_price = pending_replacement_qty = None
        cumulative_filled_qty = cumulative_cost = 0.0
        completed = False

        def _complete_followup(chunk_qty=0.0, chunk_price=0.0):
            nonlocal completed
            add_qty = max(0.0, float(chunk_qty or 0.0))
            total_qty = round(cumulative_filled_qty + add_qty, 8)
            if total_qty <= 1e-12:
                return None
            total_cost = cumulative_cost + add_qty * float(chunk_price or 0.0)
            avg_px = round(total_cost / total_qty, 4) if total_cost > 0 else float(chunk_price or getattr(self, filled_attr))
            setattr(self, filled_attr, avg_px)
            opp_price = _followup_exit_price(self.symbol, opp_side, avg_px)
            followup_extra = {"allow_math_reference_sell": True} if opp_side == "SELL" else {}
            mkt.place(
                self.symbol, opp_side, opp_price, total_qty,
                force=_followup_force(self.symbol, opp_side),
                cancelorders=True, hours=RTRADE_FOLLOWUP_HOURS,
                bypass_guards={"G_PROF", "GLLM1"}, caller_owns_retry=True,
                **followup_extra,
            )
            completed = True
            return mark_filled(avg_px)

        try:
            while True:
                _touch_rtrade_heartbeat()
                if getattr(self, "_round_aborted", False) and not is_opp_filled():
                    print(f"[{self.symbol}] Aborting {side} loop because {opp_side} worker exited without fill.")
                    return _complete_followup()
                latest_price = _finite_pos_float(api.get_current_price(self.symbol), 0.0)
                if latest_price > 0:
                    current_price = latest_price
                current_price = _finite_pos_float(current_price, 0.0) or _finite_pos_float(filled_opp_price, 0.0)
                if current_price <= 0:
                    time.sleep(WAIT_FOR_ORDER)
                    failure_count += 1
                    if failure_count > max_failures:
                        return _complete_followup()
                    continue

                if is_opp_filled():
                    adjustment_percent = max(MIN_adjustment_percent, adjustment_percent * (1.0 - decay_pct))

                calculated_price = round(current_price * (1.0 + sign * adjustment_percent), 4)
                target_price = calculated_price if pending_replacement_price is None else pending_replacement_price
                target_qty = self.qty if pending_replacement_qty is None else pending_replacement_qty
                submit_cache_permit = pending_cache_permit
                pending_cache_permit = pending_replacement_price = pending_replacement_qty = None
                print(f"[{self.symbol}] {side} order initiated at {target_price:.2f}; adjustment {adjustment_percent}%")

                if is_side_filled():
                    cur_filled_px = getattr(self, filled_attr)
                    print(f"[{self.symbol}] Ignore {side} order. It was previously filled at {cur_filled_px:.2f}")
                    self._finish_side_precheck(side, aborted=False)
                    completed = True
                    return mark_filled(cur_filled_px)

                if getattr(self, "_round_aborted", False) and not is_opp_filled():
                    print(f"[{self.symbol}] Aborting {side} order placement because {opp_side} worker exited without fill.")
                    return _complete_followup()

                outcome_context = {}
                try:
                    placed_order = self._place_legacy_side_order(
                        side, is_buy, is_opp_filled, target_price, target_qty,
                        adjustment_percent, failure_count, desp_hours_base, normal_hours,
                        submit_cache_permit, outcome_context,
                    )
                except po.WeightLimitBlock as e:
                    self._finish_side_precheck(side, aborted=not is_opp_filled())
                    print(f"[{self.symbol}] 24h limit reached{'' if is_buy else ' (SELL)'} — exiting without retry ({e})")
                    return _complete_followup()
                except Exception:
                    self._finish_side_precheck(side, aborted=not is_opp_filled())
                    raise

                if placed_order is None:
                    refusal_reason = outcome_context.get("reason")
                    self._finish_side_precheck(
                        side, aborted=bool(_is_terminal_unfilled_refusal(refusal_reason) and not is_opp_filled())
                    )
                    if self._should_exit_on_legacy_refusal(side, opp_side, is_buy, is_opp_filled, refusal_reason):
                        return _complete_followup()
                    print(f"[{self.symbol}] Order {side} failed, retryed {failure_count} times. Retrying again ...")
                    _touch_rtrade_heartbeat()
                    time.sleep(WAIT_FOR_ORDER)
                    failure_count += 1
                    if failure_count > max_failures:
                        print(f"[{self.symbol}] Order {side} failed {failure_count} times. Exiting.")
                        return _complete_followup()
                    continue

                self._finish_side_precheck(side, aborted=False)
                failure_count = 1
                with self.lock:
                    self._round_placed_order = True

                time.sleep(WAIT_FOR_ORDER)
                order_id = (
                    placed_order["orderId"]
                    if isinstance(placed_order, dict) and "orderId" in placed_order
                    else extract_order_id(placed_order)
                )
                extracted_px = extract_order_price(placed_order, default=0.0)
                filled_px = round(float(extracted_px if extracted_px > 0 else target_price), 4)
                setattr(self, filled_attr, filled_px)

                if _order_fully_filled(self.symbol, order_id):
                    print(f"[{self.symbol}] {side} order filled at {filled_px:.2f}")
                    print(f"[{self.symbol}] Starting urgent {opp_side} follow-up (1)")
                    return _complete_followup(target_qty, filled_px)

                recent_fill_px = mkt.latest_fill_price(self.symbol, side, WAIT_FOR_ORDER)
                if recent_fill_px is not None:
                    print(f"[{self.symbol}] {side} order may have been filled :-) at {recent_fill_px:.2f}")
                    print(f"[{self.symbol}] Starting urgent {opp_side} follow-up (2)")
                    return _complete_followup(target_qty, recent_fill_px)

                latest_price = _finite_pos_float(api.get_current_price(self.symbol), 0.0)
                if latest_price > 0:
                    current_price = latest_price
                bad_day = (
                    (current_price > filled_opp_price if is_buy else current_price < filled_opp_price)
                    and not u.are_close(current_price, filled_opp_price, RTRADE_BAD_DAY_TOLERANCE_PCT)
                )
                if bad_day:
                    print(f"[{self.symbol}] Bed day :-(. Trying {side} at current price {'-' if is_buy else '+'} x2 {current_price:.2f}")
                    adjustment_percent = RTRADE_BAD_DAY_MULTIPLIER * self.DEFAULT_ADJUSTMENT_PERCENT
                aborting_unfilled = getattr(self, "_round_aborted", False) and not is_opp_filled()
                if not aborting_unfilled:
                    replacement_price = round(current_price * (1.0 + sign * adjustment_percent), 4)
                    replacement_cache_permit = _preflight_routed_order(
                        self.symbol, side, target_qty, replacement_price, market=False, kind="rtrade_legacy_quote"
                    )
                else:
                    replacement_price = replacement_cache_permit = None

                final_status = _cancel_order_confirmed(self.symbol, order_id)
                if not final_status:
                    if _order_fully_filled(self.symbol, order_id):
                        print(f"[{self.symbol}] Cancel {side} order failed. Maybe it was filled :-)? Moving to {opp_side} ...")
                        print(f"[{self.symbol}] Starting urgent {opp_side} follow-up (3)")
                        return _complete_followup(target_qty, getattr(self, filled_attr))
                    print(f"[{self.symbol}] Cancel {side} order remains ambiguous; replacement is deferred.")
                    return None

                remaining_qty = _remaining_after_canceled_order(final_status, target_qty)
                if remaining_qty is None:
                    print(f"[{self.symbol}] Invalid final {side} quantity; replacement deferred.")
                    return None
                if remaining_qty <= max(1e-12, target_qty * 1e-9):
                    print(f"[{self.symbol}] {side} filled during cancellation; moving to {opp_side}.")
                    return _complete_followup(target_qty, getattr(self, filled_attr))
                step_filled = max(0.0, target_qty - remaining_qty)
                if step_filled > 1e-12:
                    cumulative_filled_qty += step_filled
                    cumulative_cost += step_filled * getattr(self, filled_attr)
                if aborting_unfilled or (getattr(self, "_round_aborted", False) and not is_opp_filled()):
                    print(f"[{self.symbol}] Aborting {side} loop after cancel because {opp_side} worker exited without fill.")
                    return _complete_followup()
                pending_cache_permit = replacement_cache_permit
                pending_replacement_price = replacement_price
                pending_replacement_qty = remaining_qty
        finally:
            if not completed:
                self._abort_round()

    def repetitive_buy(self, current_price, filled_sell_price):
        return self._repetitive_side_loop("BUY", current_price, filled_sell_price)

    def repetitive_sell(self, current_price, filled_buy_price):
        return self._repetitive_side_loop("SELL", current_price, filled_buy_price)

    def _run_pair(self, executor, current_price):
        """Run both sides concurrently on the bot's persistent workers."""
        cond = self._get_pair_cond()
        with cond:
            self._round_placed_order = False
            self._round_aborted = False
            self._pair_precheck_pending = {"BUY", "SELL"}
        buy_future = executor.submit(self.repetitive_buy, current_price, self.filled_sell_price)
        sell_future = executor.submit(self.repetitive_sell, current_price, self.filled_buy_price)
        wait((buy_future, sell_future))
        return buy_future.result(), sell_future.result()

    def _build_pair_policy(self, adjustment_fraction: float | None = None) -> PairPolicy:
        return PairPolicy(
            adjustment_fraction=(
                self.DEFAULT_ADJUSTMENT_PERCENT if adjustment_fraction is None else adjustment_fraction
            ),
            quote_ttl_sec=WAIT_FOR_ORDER, poll_sec=RTRADE_PAIR_POLL_SEC,
            fast_fill_ratio=RTRADE_FAST_FILL_RATIO, min_edge_fraction=RTRADE_MIN_EDGE_PCT,
            shock_hard_stop_fraction=RTRADE_SHOCK_HARD_STOP_PCT, hard_stop_fraction=RTRADE_HARD_STOP_PCT,
        )

    @staticmethod
    def _clear_venue_blocked_pair(venue, pid):
        unblock = getattr(venue, "_set_pair_recovery_blocked", None)
        if callable(unblock):
            unblock(pid, False)

    @staticmethod
    def _default_recovered_state(record) -> dict:
        created_ts = normalize_epoch_sec(record.get("created_ts"), default=None)
        now_s = time.time()
        return {
            "pair_id": record["pair_id"], "qty": record["qty"], "start_side": record["start_side"],
            "phase": "quoting", "reason": "recovered_by_client_order_id", "shock": False,
            "elapsed_sec": max(0.0, now_s - (created_ts if created_ts is not None else now_s)),
            "first_fill_elapsed_sec": None, "first_fill_side": None, "stop_order_id": None,
            "limit_revisions": {"BUY": 0, "SELL": 0}, "hard_stop_revision": 0,
            "hard_stop_reason": None, "tickets": [], "snapshots": {},
        }

    def _adopt_pair_record(self, venue, pair_store, policy: PairPolicy, record, *, retry_label: str = ""):
        state = record.get("state") or self._default_recovered_state(record)
        state = venue.merge_checkpoint_intents(record, state)
        intents = record.get("intents", {})
        all_exhausted = bool(intents) and all(
            isinstance(i, dict) and i.get("recovery_state") in {"absence_confirmed_no_reuse", "horizon_exhausted"}
            for i in intents.values()
        )
        if not state.get("tickets") and (
            all_exhausted or state.get("phase") not in {"startup_recovery", "hard_stop_recovery"}
        ):
            state["phase"] = "failed"
            state["reason"] = "recovery_horizon_exhausted" if all_exhausted else "recovery_intent_not_submitted"
            pair_store.checkpoint(record["pair_id"], state, terminal=True)
            self._clear_venue_blocked_pair(venue, record["pair_id"])
            print(
                f"[{self.symbol}] pair={record['pair_id']} recovery{retry_label}: "
                "exhausted intent or no tickets placed; closed as terminal"
            )
            return None
        pair_store.checkpoint(record["pair_id"], state, terminal=False)
        return PairCoordinator.from_state(venue, policy, state)

    def _cleanup_orphan_rt_orders(self, venue, pair_store) -> bool:
        known_client_ids = {
            intent.get("client_order_id")
            for rec in pair_store.active(self.symbol)
            for intent in (rec.get("intents") or {}).values()
            if isinstance(intent, dict) and intent.get("client_order_id")
        }
        orphan_orders = [
            order for order in venue.executor.open_orders(self.symbol)
            if isinstance(order, dict)
            and (coid := extract_client_order_id(order))
            and coid.startswith("RT_")
            and coid not in known_client_ids
        ]
        for order in orphan_orders:
            order_id = extract_order_id(order) or ""
            client_id = extract_client_order_id(order)
            try:
                venue.executor.cancel_order(self.symbol, order_id)
                print(f"[{self.symbol}] recovery: orphaned RT_ order cancelled order_id={order_id} client_id={client_id}")
            except Exception as exc:
                print(f"[{self.symbol}] recovery: cancel orphaned RT_ order {order_id} raised {exc}")
        if orphan_orders:
            remaining = {extract_order_id(order) for order in venue.executor.open_orders(self.symbol)}
            not_canceled = []
            for order in orphan_orders:
                oid = extract_order_id(order)
                if oid in remaining:
                    not_canceled.append(oid)
                elif oid:
                    try:
                        resolve_accepted_order(self.symbol, oid)
                    except Exception:
                        pass
            if not_canceled:
                print(f"[{self.symbol}] RECOVERY BLOCKED: unconfirmed cancellation for the RT_ orders {not_canceled}")
                return True
        return False

    def _step_active_coordinators(self, venue, pair_store, active: list, now: float) -> list:
        survivors, checkpoints = [], []
        for coordinator in active:
            outcome = coordinator.step(now=now)
            export_state = getattr(coordinator, "export_state", None)
            if callable(export_state):
                checkpoints.append((coordinator.pair_id, export_state(), outcome.terminal))
            if outcome.terminal:
                self._clear_venue_blocked_pair(venue, coordinator.pair_id)
                print(
                    f"[{self.symbol}] pair={outcome.pair_id} "
                    f"phase={outcome.phase} shock={outcome.shock} "
                    f"latency={outcome.fill_latency_sec} "
                    f"buy={outcome.buy_qty:.6f} sell={outcome.sell_qty:.6f} "
                    f"net={outcome.net_qty:.6f} "
                    f"cashflow={outcome.gross_pnl:.2f} "
                    f"fees={outcome.fees:.2f} reason={outcome.reason}"
                )
            else:
                survivors.append(coordinator)
        pair_store.checkpoint_many(checkpoints)
        return survivors

    def _run_coordinator_forever(self):
        pair_store = getattr(self, "pair_store", None) or RTradePairStore()
        venue = _LivePairVenue(self.symbol, pair_store=pair_store)
        policy = self._build_pair_policy()
        _validate_coordinator_config()

        active = []
        recovery_blocked = False
        for record in pair_store.active(self.symbol):
            try:
                coordinator = self._adopt_pair_record(venue, pair_store, policy, record)
                if coordinator is not None:
                    active.append(coordinator)
                    print(f"[{self.symbol}] pair={coordinator.pair_id} adopted phase={coordinator.phase} tickets={len(coordinator.tickets)}")
            except Exception as exc:
                print(f"[{self.symbol}] RECOVERY BLOCKED: pair={record.get('pair_id')} {exc}")
                recovery_blocked = True

        try:
            if self._cleanup_orphan_rt_orders(venue, pair_store):
                recovery_blocked = True
        except Exception as exc:
            print(f"[{self.symbol}] RECOVERY BLOCKED: exchange inventory unavailable ({exc})")
            recovery_blocked = True

        last_start_at = last_recovery_retry = last_recovery_diag_log = last_active_summary_log = last_whale_log = float("-inf")
        recovery_blocked_since = (
            time.monotonic() if (recovery_blocked or bool(getattr(venue, "recovery_blocked", False))) else None
        )
        next_direction = 0
        side_backoff_until = {"BUY": 0.0, "SELL": 0.0}

        while True:
            try:
                now = time.monotonic()
                is_recovery_blocked = recovery_blocked or bool(getattr(venue, "recovery_blocked", False))

                if is_recovery_blocked:
                    if recovery_blocked_since is None:
                        recovery_blocked_since = now
                    elapsed_blocked = now - recovery_blocked_since
                    if elapsed_blocked >= _RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC:
                        print(
                            f"[{self.symbol}] FATAL: startup recovery remained blocked for {elapsed_blocked:.0f}s "
                            f"(>{_RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC:.0f}s). Exiting for orchestrator restart."
                        )
                        sys.exit(1)

                    if now - last_recovery_diag_log >= 15.0:
                        last_recovery_diag_log = now
                        rem = max(0.0, _RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC - elapsed_blocked)
                        blocked_list = list(getattr(venue, "_recovery_blocked_pairs", []))
                        print(
                            f"[{self.symbol}] RECOVERY IN PROGRESS / BLOCKED: elapsed={elapsed_blocked:.1f}s / max={_RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC:.1f}s "
                            f"(orchestrator timeout in {rem:.1f}s) | venue_blocked={bool(getattr(venue, 'recovery_blocked', False))} "
                            f"init_blocked={recovery_blocked} blocked_pairs={blocked_list}"
                        )

                    if recovery_blocked and now - last_recovery_retry >= 60.0:
                        last_recovery_retry = now
                        try:
                            adopted_ids = {c.pair_id for c in active}
                            retry_failed = False
                            for record in pair_store.active(self.symbol):
                                if record.get("pair_id") in adopted_ids:
                                    continue
                                try:
                                    coordinator = self._adopt_pair_record(
                                        venue, pair_store, policy, record, retry_label=" retry"
                                    )
                                    if coordinator is not None:
                                        active.append(coordinator)
                                        adopted_ids.add(coordinator.pair_id)
                                except Exception as adopt_exc:
                                    print(f"[{self.symbol}] startup recovery pair adoption still blocked: {adopt_exc}")
                                    retry_failed = True
                            if self._cleanup_orphan_rt_orders(venue, pair_store):
                                retry_failed = True
                            if not retry_failed:
                                recovery_blocked = False
                                print(f"[{self.symbol}] startup recovery unblocked: exchange inventory verified")
                        except Exception as exc:
                            print(f"[{self.symbol}] startup recovery retry blocked: {exc}")
                elif recovery_blocked_since is not None:
                    print(f"[{self.symbol}] RECOVERY UNBLOCKED: all pairs clear, resuming trading.")
                    recovery_blocked_since = None

                if not is_recovery_blocked:
                    _touch_rtrade_heartbeat(now=now)
                    if now - last_active_summary_log >= 120.0:
                        last_active_summary_log = now
                        print(f"[{self.symbol}] health: active rounds={len(active)} heartbeat touched.")

                active = self._step_active_coordinators(venue, pair_store, active, now)

                can_start = (
                    not recovery_blocked
                    and not getattr(venue, "recovery_blocked", False)
                    and len(active) < RTRADE_PAIR_MAX_ACTIVE_ROUNDS
                    and now - last_start_at >= RTRADE_PAIR_START_INTERVAL_SEC
                )
                if can_start and not _trend_too_strong(self.symbol):
                    current_price = _finite_pos_float(venue.current_price(), 0.0)
                    if current_price > 0:
                        start_side = None
                        for _ in range(len(RTRADE_PAIR_DIRECTIONS)):
                            candidate = RTRADE_PAIR_DIRECTIONS[next_direction]
                            next_direction = (next_direction + 1) % len(RTRADE_PAIR_DIRECTIONS)
                            if now >= side_backoff_until[candidate]:
                                blocked, b_reason = _whale_flow_blocks_side(self.symbol, candidate)
                                if blocked:
                                    if now - last_whale_log >= 30.0:
                                        last_whale_log = now
                                        print(f"[{self.symbol}] rtrade DEFER {candidate}: {b_reason}")
                                    continue
                                start_side = candidate
                                break
                        if start_side is None:
                            time.sleep(RTRADE_PAIR_POLL_SEC)
                            continue
                        round_qty = RTRADE_NOTIONAL_USDC / float(current_price)
                        reserved_pair_id = uuid.uuid4().hex
                        eff_adjustment = _effective_adjustment_fraction(self.symbol, self.DEFAULT_ADJUSTMENT_PERCENT)
                        round_policy = (
                            policy
                            if eff_adjustment == self.DEFAULT_ADJUSTMENT_PERCENT
                            else self._build_pair_policy(eff_adjustment)
                        )
                        coordinator = PairCoordinator(venue, round_qty, round_policy, start_side=start_side)
                        outcome = coordinator.start(current_price, pair_id=reserved_pair_id)
                        export_state = getattr(coordinator, "export_state", None)
                        last_start_at = now
                        if not outcome.terminal:
                            active.append(coordinator)
                        try:
                            if callable(export_state):
                                pair_store.checkpoint(
                                    coordinator.pair_id, export_state(), terminal=outcome.terminal
                                )
                        except Exception:
                            recovery_blocked = True
                            raise
                        if outcome.terminal:
                            self._clear_venue_blocked_pair(venue, coordinator.pair_id)
                            failed_side, backoff_sec = _place_failure_backoff(outcome.reason)
                            if failed_side in side_backoff_until:
                                side_backoff_until[failed_side] = now + backoff_sec
                                print(
                                    f"[{self.symbol}] {failed_side} backoff "
                                    f"{backoff_sec:.0f}s after a placement failure ({outcome.reason})"
                                )
                            print(
                                f"[{self.symbol}] pair={outcome.pair_id} "
                                f"direction={start_side}-first phase={outcome.phase} reason={outcome.reason}"
                            )
                        else:
                            print(
                                f"[{self.symbol}] pair={outcome.pair_id} started "
                                f"direction={start_side}-first active={len(active)}/{RTRADE_PAIR_MAX_ACTIVE_ROUNDS}"
                            )
                time.sleep(RTRADE_PAIR_POLL_SEC)
            except Exception as exc:  # noqa: BLE001 — retain rounds for reconciliation.
                print(f"[{self.symbol}] pair coordinator error: {exc}")
                time.sleep(RTRADE_PAIR_POLL_SEC)

    def _cancel_all_rtrade_open_orders(self):
        api.cancel_open_orders("SELL", self.symbol, allowed_owners={"rtrade"})
        api.cancel_open_orders("BUY", self.symbol, allowed_owners={"rtrade"})

    def _resume_after_single_side_fill(self, filled_side: str, stale_side: str, normal_hours: float, attr_name: str, mark_fn):
        last_px = _finite_pos_float(
            mkt.latest_fill_price(self.symbol, filled_side, normal_hours * 3600.0), getattr(self, attr_name)
        )
        if last_px > 0:
            mark_fn(round(last_px, 4))
            print(
                f"[{self.symbol}] startup recovery: resumed after {filled_side} fill at "
                f"{getattr(self, attr_name):.2f}; cancelling stale {stale_side} order(s) to continue follow-up."
            )
        api.cancel_open_orders(stale_side, self.symbol, allowed_owners={"rtrade"})

    def _reconcile_startup_from_cache(self):
        """Reconcile open RT_ orders and recent fill state from the cached API on reboot."""
        try:
            open_buys = api.get_open_orders("BUY", self.symbol, allowed_owners={"rtrade"})
            open_sells = api.get_open_orders("SELL", self.symbol, allowed_owners={"rtrade"})
        except Exception as exc:
            print(f"[{self.symbol}] startup cache check warning: {exc}")
            return
        has_buys = isinstance(open_buys, dict) and bool(open_buys)
        has_sells = isinstance(open_sells, dict) and bool(open_sells)
        if not has_buys and not has_sells:
            return
        try:
            if has_sells and not has_buys:
                self._resume_after_single_side_fill(
                    "BUY", "SELL", RTRADE_BUY_NORMAL_HOURS, "filled_buy_price", self.mark_buy_filled
                )
            elif has_buys and not has_sells:
                self._resume_after_single_side_fill(
                    "SELL", "BUY", RTRADE_SELL_NORMAL_HOURS, "filled_sell_price", self.mark_sell_filled
                )
            else:
                print(
                    f"[{self.symbol}] startup recovery: cancelling {len(open_buys)} BUY and "
                    f"{len(open_sells)} SELL leftover RT_ orders."
                )
                self._cancel_all_rtrade_open_orders()
        except Exception as exc:
            print(f"[{self.symbol}] startup order cleanup warning: {exc}")

    def run(self):
        _touch_rtrade_heartbeat(force=True)
        if RTRADE_PAIR_COORDINATOR_ENABLED:
            return self._run_coordinator_forever()
        if getattr(self, "_needs_startup_reconcile", False):
            self._needs_startup_reconcile = False
            self._reconcile_startup_from_cache()
        prefix = f"rtrade-{self.symbol}"
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=prefix) as executor:
            while True:
                _touch_rtrade_heartbeat()
                try:
                    current_price = _finite_pos_float(api.get_current_price(self.symbol), 0.0)
                    if current_price <= 0:
                        print(f"[{self.symbol}] Failed to fetch current price. Retrying in {WAIT_FOR_ORDER} seconds...")
                        time.sleep(WAIT_FOR_ORDER)
                        continue
                    print(f"[{self.symbol}] Current price: {current_price:.2f}")

                    if _trend_too_strong(self.symbol):
                        self._cancel_all_rtrade_open_orders()
                        time.sleep(WAIT_FOR_ORDER)
                        continue

                    buy_result, sell_result = self._run_pair(executor, current_price)
                    if not buy_result or not sell_result:
                        with self.lock:
                            self.buy_filled = self.sell_filled = False
                        if getattr(self, "_round_placed_order", True):
                            self._cancel_all_rtrade_open_orders()
                        time.sleep(WAIT_FOR_ORDER)
                        continue

                    filled_buy_price = buy_result + RTRADE_ZERO_EPSILON
                    filled_sell_price = sell_result
                    print(
                        f"[{self.symbol}] Transaction complete: Bought at {filled_buy_price:.2f}, "
                        f"Sold at {filled_sell_price:.2f}"
                    )
                    ratio_label = "PROFIT: Profit ratio" if filled_buy_price < filled_sell_price else "LOSS: Loss ratio"
                    print(f"[{self.symbol}] {ratio_label} {filled_sell_price / filled_buy_price:.2f}")
                    time.sleep(1)
                    with self.lock:
                        self.buy_filled = self.sell_filled = False
                except Exception as e:
                    print(f"[{self.symbol}] Unexpected error: {e}")
                    with self.lock:
                        self.buy_filled = self.sell_filled = False
                    self._cancel_all_rtrade_open_orders()
                    time.sleep(1)


DEFAULT_ADJUSTMENT_PERCENT = required_float_env("RTRADE_DEFAULT_ADJUSTMENT_PCT")
print(f"[INFO] DEFAULT_ADJUSTMENT_PERCENT = {DEFAULT_ADJUSTMENT_PERCENT}")

if __name__ == "__main__":
    symbol = single_symbol_for("binance", "rtrade")
    cm.enable_real_ws_event_sync()

    initial_price = 0.0
    for attempt in range(1, 13):
        initial_price = _finite_pos_float(api.get_current_price(symbol), 0.0)
        if initial_price > 0:
            break
        print(f"[{symbol}] startup: price feed not ready yet (attempt {attempt}/12); waiting 5s...")
        time.sleep(5)
    if initial_price <= 0:
        print(
            f"[{symbol}] startup: price feed still not ready after 12 tries; exiting cleanly "
            "for the supervisor to relaunch (transient feed gap, not a fault)."
        )
        raise SystemExit(0)
    initial_qty = RTRADE_NOTIONAL_USDC / initial_price
    bot = TradingBot(symbol, initial_qty, DEFAULT_ADJUSTMENT_PERCENT=DEFAULT_ADJUSTMENT_PERCENT)
    bot.run()
