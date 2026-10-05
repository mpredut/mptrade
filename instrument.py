# instrument.py
"""Explicit instrument descriptor: provider, symbol, assets, and parameters.

Consumers operate on ``Instrument`` instances instead of inferring a venue from a
symbol. The provider name is resolved through ``MarketApi.provider_by_name`` so the
same asset can be represented independently on multiple venues.

``free()`` queries the instrument provider with the base asset (or the full symbol
when no base is configured). ``place()`` either delegates to an internally guarded
provider or runs the shared policy pipeline before provider mechanics.
"""
import inspect
import os
import math
import sys
import time
from typing import Optional, List, Callable, Any

from providers.market_api import api as _default_api
from providers.execution_audit import ExecutionAudit
from providers.strategy_executor import SubmissionRefused
import order_guard
import order_outcomes_log as _outcomes_log
import accepted_order_persistence
from lock import trade_cooldown


_EXECUTION_AUDIT = ExecutionAudit()


def _accepted_order_payload(payload):
    """Return whether a provider payload proves venue acceptance, never fill."""
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").upper()
    if status in {"REJECTED", "CANCELED", "CANCELLED", "EXPIRED", "FAILED"}:
        return False
    native_id = payload.get("orderId", payload.get("id", payload.get("txid")))
    if isinstance(native_id, (list, tuple)):
        return any(str(item).strip() for item in native_id)
    return native_id is not None and str(native_id).strip() != ""


def _order_id_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    native_id = payload.get("orderId", payload.get("id", payload.get("txid")))
    if isinstance(native_id, (list, tuple)):
        native_id = next((item for item in native_id if str(item).strip()), None)
    return None if native_id is None else str(native_id)


def _is_terminal_filter_refusal(reason) -> bool:
    """Use the retry subsystem's shared deterministic-refusal classification."""
    import order_retry
    return order_retry.is_terminal_filter_refusal(reason)


class Instrument:
    """Provider-bound symbol with namespaced consumer parameters.

    ``params`` is a flat mapping such as ``{"mt.gain": "9.2"}``; use ``param``
    for optional casting and defaults.
    """

    def __init__(self, name: str, symbol: str, provider: str,
                 base: Optional[str] = None, quote: Optional[str] = None,
                 enabled: bool = True, isolation: str = "own_ledger",
                 market_hours: str = "24x7",
                 params: Optional[dict] = None, api=None):
        self.name = name
        self.symbol = symbol
        self.provider_name = provider
        self.base = base
        self.quote = quote
        self.enabled = enabled
        self.isolation = isolation          # 'dedicated' | 'own_ledger' (see design)
        self.market_hours = market_hours    # '24x7' | 'rth' | ...
        self.params = dict(params or {})
        self._api = api or _default_api
        self._provider = self._api.provider_by_name(provider)
        if self._provider is None:
            raise ValueError(
                f"Instrument {name!r} ({symbol}): unknown provider {provider!r}. "
                "Is it registered in market_api?")
        self._provider.validate_symbol(self.symbol)

    def _call_profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None
    ):
        method = getattr(self._provider, "profit_guard_window_ref", None)
        if not callable(method):
            return None
        try:
            parameters = inspect.signature(method).parameters.values()
            accepts_context = any(
                (
                    parameter.name == "regime_context"
                    and parameter.kind in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                )
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_context = False
        if accepts_context:
            return method(symbol, side, safeback_sec, regime_context=regime_context)
        return method(symbol, side, safeback_sec)

    def _apply_late_intelligence_scale(
        self, qty: float, price: float, is_market: bool,
        enforce_business_minimum: bool, regime_context,
        applied_scale: float, side_u: str = "BUY"
    ) -> tuple[float, float, Optional[str]]:
        """Scale down quantity if intelligence guards recommended risk reduction after initial calculation."""
        if regime_context is None or qty <= 0 or side_u != "BUY":
            return qty, applied_scale, None
        scale = getattr(regime_context, "suggested_scale", None)
        if scale is not None and 0.0 < float(scale) < float(applied_scale):
            ratio = float(scale) / float(applied_scale)
            scaled_qty = qty * ratio
            if hasattr(self._provider, "round_amount"):
                scaled_qty = self._provider.round_amount(self.symbol, scaled_qty)
            filter_check = getattr(self._provider, "order_filter_refusal", None)
            refusal = (
                filter_check(
                    self.symbol, side_u, price, scaled_qty,
                    market=is_market, enforce_business_minimum=enforce_business_minimum)
                if callable(filter_check) else None
            )
            if refusal:
                print(f"[{self.symbol}] {side_u} late-scaled qty {scaled_qty} refused by filter: {refusal}")
                return 0.0, applied_scale, str(refusal)
            print(f"[{self.symbol}] {side_u} late-scaled by intelligence guard: {qty} -> {scaled_qty} (scale={float(scale):.2f})")
            return scaled_qty, float(scale), None
        return qty, applied_scale, None

    # -- Provider identity and access. -----------------------------------------
    @property
    def provider(self):
        return self._provider

    @property
    def provider_label(self) -> str:
        return self._provider.name

    # -- Market data delegated to the provider for this instrument symbol. -----
    def price(self) -> Optional[float]:
        return self._provider.get_current_price(self.symbol)

    def history(self, lookback_h: float) -> Optional[List]:
        return self._provider.get_price_history(self.symbol, lookback_h)

    # -- Account: free asset balance; orders and trades by symbol. -------------
    def free(self) -> Optional[float]:
        return self._provider.free_balance(self.base or self.symbol)

    def orders(self, side: Optional[str], since_s: float) -> List[dict]:
        return self._provider.get_orders(self.symbol, side, since_s)

    def trades(self, since_s: float) -> List[dict]:
        return self._provider.get_trades(self.symbol, since_s)

    def open_orders(self) -> List[dict]:
        return self._provider.open_orders(self.symbol)

    # -- Order placement: dry or real after provider gates. --------------------
    def place(self, side: str, price: float, qty: float, **kwargs):
        # Complete provider-agnostic guard: profit, daily cap/anti-spam, rapid-fire
        # cooldown, instantaneous trend deferral, and fleet-wide logging. Apply only to providers
        # without internal guards. Binance retains its richer implementation using
        # real API-permission data, avoiding duplicated or conflicting cooldowns.
        # Kraken and Hyperliquid receive the equivalent protections through shared
        # order_guard, trade_cooldown, cacheManager, and outcome-log components.
        #
        # bypass_profit_guard skips both profit and weight guards for emergency
        # flows. A protective market SELL also uses venue-only order minima so an
        # internal business floor cannot prevent risk reduction.
        # bypass_profit_reference skips only the historical price-reference
        # check while retaining quantity/weight policy and every other guard.
        # bypass_quantity_policy skips only the dynamic quantity/weight policy;
        # balance, fee, profit, daily-cap, cooldown, and trend guards stay active.
        # A caller such as rtrade or the outbox worker may own its lifecycle and
        # retry; in that case this pipeline does not touch the global queue.
        caller_owns_retry = bool(kwargs.pop("caller_owns_retry", False))
        # Internal mutable result channel used by the outbox worker.  It keeps the
        # public return type backward compatible while exposing the exact refusal
        # reason (notably trend deferral) without parsing logs or blocking.
        outcome_context = kwargs.pop("_outcome_context", None)
        if outcome_context is not None and not isinstance(outcome_context, dict):
            raise TypeError("_outcome_context must be a dict")
        # This process-local value is deliberately removed before any retry,
        # outbox, or audit payload is built. It authorizes one immediate Binance
        # submit only and is never durable state.
        cache_permit = kwargs.pop("cache_permit", None)
        caller_supplied_cache_permit = cache_permit is not None
        retry_requested_price_raw = kwargs.pop("_retry_requested_price", None)
        retry_price_tolerance_raw = kwargs.pop("_retry_price_tolerance", None)
        retry_constraint_supplied = (
            retry_requested_price_raw is not None
            or retry_price_tolerance_raw is not None)
        bypass = bool(kwargs.pop("bypass_profit_guard", False))
        bypass_profit_reference = bool(kwargs.pop("bypass_profit_reference", False))
        bypass_quantity_policy = bool(kwargs.pop("bypass_quantity_policy", False))
        wait_for_trend = bool(kwargs.pop("wait_for_trend", True))
        cooldown_pair_id = kwargs.pop("cooldown_pair_id", None)
        side_u = side.upper()
        # This narrow escape hatch exists only for drawdown BUY flows.  A SELL
        # must always prove profitability against its historical BUY reference;
        # silently ignore the flag there so a misplaced caller argument cannot
        # weaken the exit guard.
        if bypass_profit_reference and side_u != "BUY":
            print(
                f"[GUARD] {side_u} {self.symbol}: bypass_profit_reference ignored; "
                "is allowed only for BUY")
            bypass_profit_reference = False
        # Quantity-policy bypass reduces exposure and is therefore SELL-only.
        # Ignore it on BUY so a misplaced argument can never increase exposure.
        if bypass_quantity_policy and side_u != "SELL":
            print(
                f"[GUARD] {side_u} {self.symbol}: bypass_quantity_policy ignored; "
                "is allowed only for SELL")
            bypass_quantity_policy = False
        is_binance = str(self._provider.name).casefold() == "binance"
        if cache_permit is not None and not is_binance:
            reason = "invalid_cache_permit_provider"
            print(f"[{self.symbol}] {side_u} BLOCKED (pre-submit): {reason}")
            if outcome_context is not None:
                outcome_context.update(
                    accepted=False, reason=reason, state="refused")
            return None
        execution_enabled = getattr(self._provider, "execution_enabled", None)
        try:
            provider_execution_enabled = (
                True if not callable(execution_enabled)
                else bool(execution_enabled()))
        except Exception as exc:  # noqa: BLE001 - An unreadable gate must fail closed.
            reason = "execution_gate_unavailable"
            print(
                f"[{self.symbol}] {side_u} BLOCKED (pre-submit): "
                f"{reason} ({exc})")
            if outcome_context is not None:
                outcome_context.update(
                    accepted=False, reason=reason, state="refused")
            return None
        if not provider_execution_enabled:
            reason = "execution_disabled"
            print(f"[{self.symbol}] {side_u} DRY: real provider execution is disabled")
            if outcome_context is not None:
                outcome_context.update(
                    accepted=False, reason=reason, state="refused")
            return None
        if self._provider.guards_internally():
            if retry_constraint_supplied:
                reason = "retry_price_constraint_unsupported"
                if outcome_context is not None:
                    outcome_context.update(
                        accepted=False, reason=reason, state="refused")
                return None
            return self._provider.place_order(self.symbol, side, price, qty, **kwargs)

        # Capture the original uncapped quantity, requested price, and reproducible
        # arguments before mutation in case the placement must be queued for retry.
        orig_qty = qty
        orig_price = price   # Requested price preserves caller intent for the retry guard.
        reason = None
        order = None
        submit_attempted = False
        prequeued_record_id = None
        prequeued_claim = None
        prequeued_client_order_id = None
        prequeued_intent_id = None
        guarded_order_state = None
        # safeback_seconds is explicitly overridden by monitortrades (normally 12
        # days) and tradeall (14 days), so Binance rarely uses the 48-hour config
        # default. Apply the same window to enabled Kraken instruments. Retain it
        # in kwargs for provider.place_order even where currently ignored.
        safeback_override = kwargs.get("safeback_seconds")
        # ``smart`` distinguishes two legacy interfaces: place_order_smart cancels
        # opposing orders and nudges price before guards, while place_safe_order
        # does neither. Default true for the main path; former safe-order callers
        # explicitly pass false.
        smart = bool(kwargs.pop("smart", True))
        regime_context = kwargs.pop("regime_context", None)
        # Reconstruct this call exactly for retry after bypass/smart were popped;
        # all other placement metadata remains in kwargs.
        retry_kwargs = dict(kwargs)
        retry_kwargs["bypass_profit_guard"] = bypass
        retry_kwargs["bypass_profit_reference"] = bypass_profit_reference
        retry_kwargs["bypass_quantity_policy"] = bypass_quantity_policy
        retry_kwargs["wait_for_trend"] = wait_for_trend
        retry_kwargs["smart"] = smart
        if cooldown_pair_id:
            retry_kwargs["cooldown_pair_id"] = cooldown_pair_id
        # A market order ignores the requested price, so revalidate immediately
        # before submission at the current executable price. This prevents a
        # force-market sell from passing at +1% then executing below the validated
        # margin, a financial time-of-check/time-of-use issue.
        is_market = bool(kwargs.get("force", False) or kwargs.get("market", False))
        enforce_business_minimum = not (
            bypass and side_u == "SELL" and is_market)
        profit_margin = None
        profit_window_ref = None
        retry_requested_price = None
        retry_price_tolerance = None
        try:
            if retry_constraint_supplied:
                try:
                    retry_requested_price = float(retry_requested_price_raw)
                    retry_price_tolerance = float(retry_price_tolerance_raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise SubmissionRefused(
                        "invalid_retry_price_constraint") from exc
                if (not math.isfinite(retry_requested_price)
                        or retry_requested_price <= 0
                        or not math.isfinite(retry_price_tolerance)
                        or not 0 <= retry_price_tolerance < 1):
                    raise SubmissionRefused("invalid_retry_price_constraint")
            preflight_order = getattr(self._provider, "preflight_order", None)
            if is_binance and smart and caller_supplied_cache_permit:
                # Smart cancellation accepts only the exact permit issued inside
                # this pipeline after its guards. External replacement flows use
                # smart=False and own their durable cancel/recovery lifecycle.
                raise SubmissionRefused(
                    "external_cache_permit_not_allowed_for_smart_cancel")
            if is_binance and cache_permit is None:
                prepare_order_state = getattr(
                    self._provider, "prepare_order_state", None)
                if not callable(prepare_order_state):
                    raise SubmissionRefused("account_cache_not_fresh")
                guarded_order_state = prepare_order_state()
                if guarded_order_state is None:
                    raise SubmissionRefused("account_cache_not_fresh")
            if not is_binance and callable(preflight_order):
                preflight_order(
                    self.symbol, side_u, qty, price, market=is_market,
                    kind=kwargs.get("kind") or kwargs.get("motivation"))

            # 0. Compute smart venue pricing before the guards. Binance defers the
            # separate opposing-order cancellation until after finite quantity
            # resolution, durable intent persistence, and exact permit issuance.
            # Other providers retain their existing combined hook behavior.
            if smart:
                price = self._provider.adjust_order_price(
                    self.symbol, side_u, price,
                    cancel_opposite=not is_binance)

            # 1. Provider-agnostic daily cap and anti-spam, never bypassed.
            ok, reason = order_guard.daily_limit_guard(self._provider, self.symbol, side_u,
                                                       safeback_sec=safeback_override)
            if not ok:
                return None

            if not bypass:
                if regime_context is None and side_u == "BUY":
                    try:
                        resolve_context = getattr(
                            self._api, "market_regime_context", None)
                        if callable(resolve_context):
                            regime_context = resolve_context(
                                self.symbol,
                                provider_name=self.provider_name,
                            )
                        else:
                            regime_context = order_guard.symbol_regime_context(
                                self.symbol, provider=self._provider)
                    except Exception:
                        regime_context = None

                if bypass_profit_reference:
                    print(
                        f"[GUARD] {side_u} {self.symbol}: the historical price reference "
                        "was explicitly bypassed; the quantity/weight guard remains active")
                else:
                    profit_margin = order_guard.margin_for(self._provider.name)
                    # Tier-one min/max reference comes from the provider hook. Kraken
                    # and Hyperliquid use their configured venue window; Binance uses
                    # safeback_sec from the order cache. An empty window falls back to
                    # the last opposite fill.
                    profit_window_ref = self._call_profit_guard_window_ref(
                        self.symbol, side_u, safeback_override, regime_context=regime_context)
                    ok = order_guard.profit_guard(
                        self._provider, self.symbol, side_u, price, profit_margin,
                        window_ref=profit_window_ref, regime_context=regime_context,
                        qty=qty,
                    )
                    if not ok:
                        reason = "profit_guard"
                        return None
            # Balance, fee, and venue filters are mechanics, not optional profit
            # policy. Apply them even when an emergency flow bypasses profit/weight.
            quantity_price = price
            if is_market:
                quantity_price = self._provider.get_current_price(self.symbol)
                try:
                    quantity_price = float(quantity_price)
                except (TypeError, ValueError, OverflowError):
                    quantity_price = float("nan")
                if not math.isfinite(quantity_price) or quantity_price <= 0:
                    print(f"[{self.symbol}] {side_u} MARKET BLOCKED: current price unavailable")
                    reason = "market_price_unavailable"
                    return None
            decision = self._provider.quantity_decision(
                self.symbol, side_u, quantity_price, qty,
                base=self.base, quote=self.quote,
                cancelorders=bool(kwargs.get("cancelorders", False)),
                hours=float(kwargs.get("hours", 5) or 5),
                apply_policy=not (bypass or bypass_quantity_policy),
                market=is_market,
                enforce_business_minimum=enforce_business_minimum,
                regime_context=regime_context,
            )
            qty = decision.final_qty
            applied_scale = getattr(decision, "applied_scale", 1.0) if getattr(decision, "scale_applied", False) else 1.0
            if qty <= 0:
                print(f"[{self.symbol}] {side_u} qty refused: "
                      f"{decision.refuse_reason} asset={decision.balance_asset}")
                reason = decision.refuse_reason or "qty_zero_after_weight"
                return None

            # Fail-safe check for intelligence scaling if not applied in quantity_decision
            if (
                regime_context is not None
                and side_u == "BUY"
                and qty > 0
                and not getattr(decision, "scale_applied", False)
            ):
                qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                    qty, quantity_price, is_market, enforce_business_minimum,
                    regime_context, applied_scale, side_u=side_u)
                if refusal:
                    reason = refusal
                    return None

            # 2. Optional provider-agnostic trend gate is instantaneous.  Placement
            # must never sleep or poll: a negative decision returns immediately and
            # the durable outbox retries it on a later worker tick.  Callers that own
            # their lifecycle (AssetGuardian) apply the same check in their strategy
            # loop and disable this duplicate gate.
            if wait_for_trend:
                try:
                    import cacheManager as cm
                    if cm.get_short_trend_manager().should_wait(side_u, self.symbol):
                        print(
                            f"[{self.symbol}] {side_u} deferred by trend; "
                            "placement does not wait, the intent stays queued for retry")
                        reason = "trend_deferred"
                        return None
                except Exception as e:  # noqa: BLE001 — Opportunistic gate.
                    print(f"[{self.symbol}] {side_u} trend gate unavailable: {e}")

            # An early executable-price check avoids persisting an intent that is
            # already known to be unprofitable. It is repeated at the final dispatch
            # boundary because cooldown, persistence, and cache validation can take
            # long enough for a market quote to move.
            if not bypass and not bypass_profit_reference:
                if is_market:
                    guard_price = quantity_price
                    ok = order_guard.profit_guard(
                        self._provider, self.symbol, side_u, guard_price, profit_margin,
                        window_ref=profit_window_ref, regime_context=regime_context,
                        qty=qty,
                    )
                    if not ok:
                        reason = "profit_guard"
                        return None
                    qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                        qty, guard_price, is_market, enforce_business_minimum,
                        regime_context, applied_scale, side_u=side_u)
                    if refusal:
                        reason = refusal
                        return None
                elif orig_qty is None and qty > 0:
                    ok = order_guard.profit_guard(
                        self._provider, self.symbol, side_u, price, profit_margin,
                        window_ref=profit_window_ref, regime_context=regime_context,
                        qty=qty,
                    )
                    if not ok:
                        reason = "profit_guard"
                        return None
                    qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                        qty, price, is_market, enforce_business_minimum,
                        regime_context, applied_scale, side_u=side_u)
                    if refusal:
                        reason = refusal
                        return None

            # 3. Provider-agnostic rapid-fire cooldown shared with Binance. Keys are
            # symbols, preventing collisions between different venues.
            with trade_cooldown.trade_slot(
                    side_u, self.symbol, pair_id=cooldown_pair_id) as slot:
                if not slot.allowed:
                    age = time.time() - slot.info.get("timestamp", 0)
                    print(f"[{self.symbol}] {side_u} BLOCKED by cooldown: last order "
                          f"({slot.info.get('side')}) {age:.0f}s ago")
                    reason = "cooldown"
                    return None
                if callable(execution_enabled) and not bool(execution_enabled()):
                    # Recheck next to the durability boundary. A disabled dry-run
                    # must never leave an intent that a later live worker can send.
                    raise SubmissionRefused("execution_disabled")
                # Persist the exact intent before the external side effect.  This
                # closes the process-crash/response-loss gap for every normal caller
                # of Instrument.place. Lifecycle-owned callers already persist in
                # their own campaign state and opt out with caller_owns_retry=True.
                if not caller_owns_retry:
                    try:
                        import order_retry
                        if order_retry.RETRY_ENABLED:
                            prequeue_kwargs = dict(retry_kwargs)
                            prequeue_kwargs["smart"] = smart
                            prequeued_claim = order_retry.enqueue_claimed(
                                self.symbol, side_u, qty, prequeue_kwargs,
                                requested_price=orig_price, ref_price=price,
                                failure_reason="submit_pending",
                                provider_name=self.provider_name,
                                intent_id=kwargs.get("intent_id"),
                                kind=kwargs.get("kind") or kwargs.get("motivation"))
                            if prequeued_claim is None:
                                raise RuntimeError("the intent cannot be read back after persisting")
                            prequeued_record_id = prequeued_claim.get("id")
                            prequeued = prequeued_claim
                            prequeued_client_order_id = dict(
                                prequeued.get("place_kwargs") or {}).get(
                                    "client_order_id")
                            prequeued_intent_id = prequeued.get("intent_id")
                            if not prequeued_client_order_id:
                                raise RuntimeError("the persisted intent has no client_order_id")
                            kwargs["client_order_id"] = prequeued_client_order_id
                    except Exception as exc:
                        print(
                            f"[{self.symbol}] {side_u} BLOCKED: the intent cannot be "
                            f"persisted before the submit ({exc})")
                        reason = "pre_submit_persist_failed"
                        return None

                if prequeued_intent_id:
                    _EXECUTION_AUDIT.record(
                        "submit_requested", intent_id=prequeued_intent_id,
                        venue=self.provider_name, symbol=self.symbol,
                        side=side_u.lower(), qty=qty, price=price,
                        market=is_market,
                        kind=kwargs.get("kind") or kwargs.get("motivation"),
                        client_order_id=prequeued_client_order_id,
                    )

                permit_requested_price = None if is_market else orig_price
                if is_binance and cache_permit is None:
                    if not callable(preflight_order):
                        raise SubmissionRefused("account_cache_not_fresh")
                    cache_permit = preflight_order(
                        self.symbol, side_u, qty, price, market=is_market,
                        kind=kwargs.get("kind") or kwargs.get("motivation"))
                    if cache_permit is None:
                        raise SubmissionRefused("account_cache_not_fresh")
                    permit_requested_price = None if is_market else price
                    validate_order_state = getattr(
                        self._provider, "validate_order_state", None)
                    if not callable(validate_order_state):
                        raise SubmissionRefused("account_cache_not_fresh")
                    validate_order_state(guarded_order_state)

                # Market profitability is a dispatch-time property. Re-read the
                # executable quote after final cache/version validation and before
                # any Binance cancellation. Protective bypasses intentionally
                # remain exempt.
                final_profit_check = (
                    not bypass and not bypass_profit_reference and is_market)
                if is_market and (final_profit_check
                                  or retry_constraint_supplied):
                    final_market_price = self._provider.get_current_price(
                        self.symbol)
                    try:
                        final_market_price = float(final_market_price)
                    except (TypeError, ValueError, OverflowError):
                        final_market_price = float("nan")
                    if (not math.isfinite(final_market_price) or
                            final_market_price <= 0):
                        reason = "market_price_unavailable"
                        print(
                            f"[{self.symbol}] {side_u} MARKET BLOCKED at dispatch: "
                            "current price unavailable")
                        return None
                    if retry_constraint_supplied:
                        favorable = (
                            final_market_price
                            >= retry_requested_price
                            * (1.0 - retry_price_tolerance)
                            if side_u == "SELL" else
                            final_market_price
                            <= retry_requested_price
                            * (1.0 + retry_price_tolerance)
                        )
                        if not favorable:
                            reason = "retry_price_unfavorable"
                            return None
                    if final_profit_check:
                        if not order_guard.profit_guard(
                            self._provider, self.symbol, side_u,
                            final_market_price, profit_margin,
                            window_ref=profit_window_ref,
                            regime_context=regime_context,
                            qty=qty):
                            reason = "profit_guard"
                            return None
                        qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                            qty, final_market_price, is_market, enforce_business_minimum,
                            regime_context, applied_scale, side_u=side_u)
                        if refusal:
                            reason = refusal
                            return None
                        if prequeued_claim is not None:
                            try:
                                import order_retry
                                if order_retry.RETRY_ENABLED:
                                    synced_claim = order_retry.update_claimed_qty(
                                        prequeued_claim, qty)
                                    if synced_claim is not None:
                                        prequeued_claim = synced_claim
                            except Exception:
                                pass
                if callable(execution_enabled) and not bool(execution_enabled()):
                    # The switch can change while guards and persistence run. This
                    # final check converts that race into a terminal pre-submit
                    # refusal under the producer's exact durable claim.
                    raise SubmissionRefused("execution_disabled")
                if prequeued_claim is not None:
                    # Revalidate exact durable ownership next to the external side
                    # effect. An expired/stolen/revision-changed claim must never
                    # race a worker or allow two provider submissions.
                    import order_retry
                    refreshed_claim = order_retry.begin_claimed_submit(
                        prequeued_claim, qty=qty)
                    if refreshed_claim is None:
                        raise SubmissionRefused("producer_claim_lost")
                    prequeued_claim = refreshed_claim
                reason = "submit_ambiguous"
                submit_attempted = True
                provider_kwargs = dict(kwargs)
                if is_binance:
                    provider_kwargs["enforce_business_minimum"] = (
                        enforce_business_minimum)
                if cache_permit is not None:
                    provider_kwargs["permit_requested_price"] = permit_requested_price
                    provider_kwargs["cache_permit"] = cache_permit
                if smart and is_binance:
                    # The Binance low-level dispatch consumes the exact permit
                    # before applying this cancellation prerequisite. Keeping the
                    # two venue side effects together leaves no local policy gate
                    # between a successful cancellation and the replacement.
                    provider_kwargs["_cancel_opposite_requested_price"] = orig_price
                response = self._provider.place_order(
                    self.symbol, side, price, qty, **provider_kwargs)
                if _accepted_order_payload(response):
                    order = response
                    reason = None
                    slot.commit(_order_id_from_payload(order))
                else:
                    reason = "response_without_order_id"
                    order = None
                return order
        except SubmissionRefused as exc:
            reason = exc.reason
            submit_attempted = False
            print(f"[{self.symbol}] {side_u} BLOCKED (pre-submit): {reason}")
            return None
        except Exception as e:  # noqa: BLE001 — Fail closed when guards cannot be verified.
            print(f"[{self.symbol}] {side_u} BLOCKED (fail-closed): {e}")
            reason = reason or "guard_check_failed"
            return None
        finally:
            if outcome_context is not None:
                outcome_context["accepted"] = order is not None
                outcome_context["reason"] = None if order is not None else reason
                outcome_context["state"] = (
                    "accepted" if order is not None else
                    "unknown" if submit_attempted else "refused"
                )
            try:
                caller = os.path.basename(sys._getframe(1).f_code.co_filename)
            except Exception:
                caller = None
            submission_state = (
                "accepted" if order is not None else
                "unknown" if submit_attempted else "refused"
            )
            terminal_filter_refusal = bool(
                reason and not submit_attempted
                and _is_terminal_filter_refusal(reason))
            _outcomes_log.log_order_outcome(
                self.symbol, side_u, price, qty, submission_state,
                None if order else reason, kwargs.get("motivation"), caller=caller,
                regime_context=regime_context)
            # Acceptance is not a fill. Preserve the venue ID in the exact outbox
            # record so the single worker follows open/partial/terminal state. If
            # this persistence step fails, the pre-submit record and deterministic
            # client ID remain available for response-loss recovery.
            if order is not None and not caller_owns_retry:
                try:
                    import order_retry
                    if prequeued_claim is not None:
                        tracked = (
                            accepted_order_persistence.complete_accepted_claim(
                                order_retry, prequeued_claim, order=order,
                                provider_name=self.provider_name,
                                symbol=self.symbol, side=side_u))
                        if not tracked:
                            print(
                                f"[{self.symbol}] {side_u} accepted, but the tracking "
                                "could not be updated; the intent stays pending")
                        elif prequeued_intent_id:
                            _EXECUTION_AUDIT.record(
                                "submit_accepted", intent_id=prequeued_intent_id,
                                venue=self.provider_name, symbol=self.symbol,
                                side=side_u.lower(), qty=qty, price=price,
                                market=is_market,
                                kind=kwargs.get("kind") or kwargs.get("motivation"),
                                client_order_id=prequeued_client_order_id,
                                order_id=_order_id_from_payload(order),
                            )
                except Exception as _e:  # noqa: BLE001
                    print(f"[{self.symbol}] {side_u} tracking accepted failed: {_e}")
            # Persist a failed intent unless this is already a retry. Queue handling
            # is best effort and never changes the placement return value.
            elif (order is None and not caller_owns_retry
                  and (prequeued_record_id is not None
                       or (reason != "account_cache_not_fresh"
                           and not terminal_filter_refusal))):
                try:
                    import order_retry
                    if prequeued_intent_id:
                        _EXECUTION_AUDIT.record(
                            "submit_unknown" if submit_attempted else "submit_refused",
                            intent_id=prequeued_intent_id,
                            venue=self.provider_name, symbol=self.symbol,
                            side=side_u.lower(), qty=qty, price=price,
                            market=is_market,
                            kind=kwargs.get("kind") or kwargs.get("motivation"),
                            client_order_id=prequeued_client_order_id,
                            failure_reason=reason,
                        )
                    if order_retry.RETRY_ENABLED:
                        if prequeued_claim is not None:
                            claim_outcome = (
                                "success" if (
                                    terminal_filter_refusal or reason in {
                                        "execution_disabled", "trading_disabled"}) else
                                "deferred" if order_retry.is_non_failure_deferral(reason)
                                else "failure")
                            order_retry.complete_claim(
                                prequeued_claim, claim_outcome,
                                failure_reason=reason,
                                submission_state=(
                                    "unknown" if submit_attempted
                                    else "refused"))
                        else:
                            # Capture best-effort market price only when enqueueing,
                            # for retry price guarding and deduplication.
                            queue_qty = orig_qty
                            try:
                                if queue_qty is None or float(queue_qty) <= 0:
                                    queue_qty = qty
                            except (TypeError, ValueError, OverflowError):
                                queue_qty = qty
                            ref_price = None
                            try:
                                ref_price = self._provider.get_current_price(self.symbol)
                            except Exception:  # noqa: BLE001
                                ref_price = None
                            order_retry.enqueue(
                                self.symbol, side_u, queue_qty, retry_kwargs,
                                requested_price=orig_price, ref_price=ref_price,
                                failure_reason=reason,
                                provider_name=self.provider_name,
                                intent_id=kwargs.get("intent_id"),
                                kind=kwargs.get("kind") or kwargs.get("motivation"))
                except Exception as _e:  # noqa: BLE001
                    print(f"[{self.symbol}] {side_u} enqueue retry failed (ignored): {_e}")

    def min_qty(self) -> float:
        """Return the venue minimum order quantity; ``0`` means no known minimum."""
        try:
            return float(self._provider.min_order_qty(self.symbol) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    # ── params namespaced (mt.* / tradeall.* / rtrade.*) ───────────────────────
    def param(self, consumer: str, key: str, default: Any = None,
              cast: Optional[Callable] = None) -> Any:
        """Return ``consumer.key`` or ``default`` when missing or conversion fails."""
        v = self.params.get(f"{consumer}.{key}")
        if v is None:
            return default
        if cast is None:
            return v
        try:
            return cast(v)
        except (ValueError, TypeError):
            return default

    def __repr__(self) -> str:
        st = "on" if self.enabled else "off"
        return f"<Instrument {self.name} {self.symbol}@{self.provider_name} {st}>"
