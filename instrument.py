# instrument.py
"""Explicit instrument descriptor: provider, symbol, assets, and parameters.

Consumers operate on ``Instrument`` instances instead of inferring a venue from a
symbol. The provider name is resolved through ``MarketApi.provider_by_name`` so the
same asset can be represented independently on multiple venues.

``free()`` queries the instrument provider with the base asset (or the full symbol
when no base is configured). ``place()`` either delegates to an internally guarded
provider or runs the shared policy pipeline before provider mechanics.
"""
from dataclasses import dataclass, field
import inspect
import os
import sys
from typing import Optional, List, Callable, Any

from cachepaths import normalize_venue_name
from cacheprice import resolve_cached_symbol_price
import cachetrend as ct
from guard_hierarchy import (
    GUARD_REGISTRY,
    revoke_guard_bypass,
)
from instrument_registry import (
    InstrumentSpec,
    discover_tradable_specs,
    load_registry,
    resolve_instrument_spec,
    select_instruments,
)
import pricewindow as pw
from providers.base import MarketDataProvider, extract_order_id
from providers import order_id_context
from providers.order_ohlc_utils import resolve_market_flag
from botcore import (
    _parse_bool_flag,
    is_us_rth_open,
    normalize_side,
    obj_attr,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.market_api import api as _default_api
from providers.execution_audit import ExecutionAudit
from providers.quantity import (
    check_order_filter_refusal,
    extract_regime_buy_scale,
    parse_scale_factor,
    resolve_assets,
    round_provider_qty,
)
from providers.strategy_executor import SubmissionRefused
import order_guard
import order_lifecycle
import order_retry


_EXECUTION_AUDIT = ExecutionAudit()


def _accepted_order_payload(payload):
    """Return whether a provider payload proves venue acceptance, never fill."""
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status") or "").upper() in {"REJECTED", "CANCELED", "CANCELLED", "EXPIRED", "FAILED"}:
        return False
    return extract_order_id(payload) is not None


def _is_aborted(cb: Any) -> bool:
    return callable(cb) and bool(cb())


@dataclass(slots=True)
class _PlaceContext:
    """Mutable pipeline state carrier for a single ``Instrument.place()`` execution."""

    side_u: str
    price: Any
    qty: Any
    orig_price: Any
    orig_qty: Any
    kwargs: dict
    caller_owns_retry: bool = False
    outcome_context: Optional[dict] = None
    cache_permit: Any = None
    caller_supplied_cache_permit: bool = False
    retry_requested_price_raw: Any = None
    retry_price_tolerance_raw: Any = None
    retry_constraint_supplied: bool = False
    profit_reference_mode: Any = None
    caller_module: Any = None
    allow_math_reference_sell: bool = False
    order_kind: Any = None
    wait_for_trend: bool = True
    abort_if: Any = None
    early_abort_if: Any = None
    v_norm: str = ""
    medium_trend_action_cfg: Any = None
    medium_trend_scale_cfg: Any = None
    bypassed_guards: set = field(default_factory=set)
    bypass_all: bool = False
    bypass_profit_guard: bool = False
    bypass_math_ref: bool = False
    bypass_quantity_policy: bool = False
    requires_permit: bool = False
    defers_smart_cancel: bool = False
    supports_biz_min: bool = False
    execution_enabled: Any = None
    safeback_override: Any = None
    smart: bool = True
    regime_context: Any = None
    caller_applied_scale: float = 1.0
    applied_scale: float = 1.0
    is_monitor_replace: bool = False
    profit_margin: Any = None
    retry_kwargs: dict = field(default_factory=dict)
    is_market: bool = False
    enforce_business_minimum: bool = True
    guarded_order_state: Any = None
    preflight_order: Any = None
    retry_requested_price: Any = None
    retry_price_tolerance: Any = None
    quantity_price: Any = None
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    known_balance: Optional[float] = None
    profit_window_ref: Any = None
    prequeued_claim: Any = None
    prequeued_record_id: Any = None
    prequeued_client_order_id: Any = None
    prequeued_intent_id: Any = None
    permit_requested_price: Any = None
    submit_attempted: bool = False
    order: Any = None
    reason: Optional[str] = None


def _is_instrument_spec(item: Any) -> bool:
    return isinstance(item, InstrumentSpec) or type(item).__name__ == "InstrumentSpec"


class Instrument:
    """Provider-bound symbol with namespaced consumer parameters.

    ``params`` is a flat mapping such as ``{"mt.gain": "9.2"}``; use ``param``
    for optional casting and defaults.
    """

    def __init__(self, name: str, symbol: str, provider: str,
                 base: Optional[str] = None, quote: Optional[str] = None,
                 enabled: bool = True, isolation: str = "own_ledger",
                 market_hours: str = "24x7",
                 params: Optional[dict] = None, api=None,
                 spec: Optional[InstrumentSpec] = None):
        self.name = name
        self.symbol = symbol
        self.provider_name = provider
        self.base = base
        self.quote = quote
        self.enabled = _parse_bool_flag(enabled, True)
        self.isolation = isolation          # 'dedicated' | 'own_ledger' (see design)
        self.market_hours = market_hours    # '24x7' | 'rth' | ...
        self.params = dict(params or {})
        self.spec = (
            spec
            if _is_instrument_spec(spec)
            else InstrumentSpec(
                name=self.name,
                provider=normalize_venue_name(provider) or str(provider or "").casefold(),
                symbol=self.symbol,
                base=self.base or "",
                quote=self.quote or "",
                enabled=self.enabled,
                isolation=self.isolation,
                market_hours=self.market_hours,
                params=dict(self.params),
            )
        )
        self._api = api or _default_api
        self._provider = self._api.provider_by_name(provider)
        if self._provider is None:
            raise ValueError(
                f"Instrument {name!r} ({symbol}): unknown provider {provider!r}. "
                "Is it registered in market_api?")
        self._provider.validate_symbol(self.symbol)

    @classmethod
    def from_spec(cls, spec: InstrumentSpec, *, api=None) -> "Instrument":
        """Bind an immutable ``InstrumentSpec`` to a provider adapter via ``MarketApi``."""
        return cls(
            name=spec.name,
            symbol=spec.symbol,
            provider=spec.provider,
            base=spec.base,
            quote=spec.quote,
            enabled=spec.enabled,
            isolation=spec.isolation,
            market_hours=spec.market_hours,
            params=spec.params,
            api=api,
            spec=spec,
        )

    def _call_profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None,
        reference_mode=None, module=None
    ):
        method = getattr(self._provider, "profit_guard_window_ref", None)
        if not callable(method):
            return None
        call_kw = {}
        try:
            params = inspect.signature(method).parameters.values()
            var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
            accepted = {
                p.name for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            for k, v in (("regime_context", regime_context), ("reference_mode", reference_mode), ("module", module)):
                if (var_kw or k in accepted) and (k == "regime_context" or v is not None):
                    call_kw[k] = v
        except (TypeError, ValueError):
            pass
        return method(symbol, side, safeback_sec, **call_kw)

    @staticmethod
    def _apply_outbox_lineage_prefix(target_kwargs: dict, call_kwargs: dict) -> None:
        """Attach composite bot lineage prefix (<BOT>_OR) for outbox retries unless overridden."""
        if "client_order_id" in call_kwargs:
            target_kwargs["client_order_id"] = call_kwargs["client_order_id"]
        elif "prefix" not in target_kwargs:
            try:
                pname = order_id_context.get_process_name()
                if pname and pname in order_id_context.KNOWN_BOT_PROCESS_CODES:
                    target_kwargs["prefix"] = f"{pname}_OR"
            except Exception:
                pass

    def _resolve_regime_context(self):
        try:
            resolve_context = getattr(self._api, "market_regime_context", None)
            if callable(resolve_context):
                return resolve_context(self.symbol, provider_name=self.provider_name)
            return order_guard.symbol_regime_context(self.symbol, provider=self._provider)
        except Exception:
            return None

    @staticmethod
    def _build_profit_guard_kwargs(
        profit_window_ref, regime_context, qty, bypassed_guards,
        profit_reference_mode, caller_module, *, include_qty: bool = True
    ) -> dict:
        pg_kwargs = {"window_ref": profit_window_ref, "regime_context": regime_context}
        if include_qty:
            pg_kwargs["qty"] = qty
        if bypassed_guards:
            pg_kwargs["bypass_guards"] = bypassed_guards
        if profit_reference_mode is not None:
            pg_kwargs["reference_mode"] = profit_reference_mode
        if caller_module is not None:
            pg_kwargs["module"] = caller_module
        return pg_kwargs

    @staticmethod
    def _sync_claimed_qty(prequeued_claim, qty, applied_scale):
        try:
            if order_retry.RETRY_ENABLED:
                synced_claim = order_retry.update_claimed_qty(
                    prequeued_claim, qty, applied_scale=applied_scale)
                if synced_claim is None:
                    raise SubmissionRefused("durable_qty_sync_failed")
                return synced_claim
            return prequeued_claim
        except SubmissionRefused:
            raise
        except Exception as exc:
            raise SubmissionRefused("durable_qty_sync_failed") from exc

    def _apply_late_intelligence_scale(
        self, qty: float, price: float, is_market: bool,
        enforce_business_minimum: bool, regime_context,
        applied_scale: float, side_u: str = "BUY"
    ) -> tuple[float, float, Optional[str]]:
        """Scale down quantity if intelligence guards recommended risk reduction after initial calculation."""
        if regime_context is None or isinstance(qty, bool) or side_u not in ("BUY", "SELL"):
            return qty, applied_scale, None
        qty_f = safe_pos_float(qty, default=None)
        scale_f = extract_regime_buy_scale(regime_context, side=side_u)
        app_f = 1.0 if isinstance(applied_scale, bool) else safe_pos_float(applied_scale, default=None)
        if qty_f is None or scale_f is None or app_f is None or not (0.0 < scale_f < app_f):
            return qty, applied_scale, None
        scaled_qty = round_provider_qty(self._provider, self.symbol, qty_f * (scale_f / app_f))
        refusal = check_order_filter_refusal(
            self._provider, self.symbol, side_u, price, scaled_qty,
            market=is_market, enforce_business_minimum=enforce_business_minimum,
        )
        if refusal:
            print(f"[{self.symbol}] {side_u} late-scaled qty {scaled_qty} refused by filter: {refusal}")
            return 0.0, applied_scale, refusal
        if scaled_qty <= 0.0:
            return 0.0, applied_scale, "qty_zero_after_policy"
        print(f"[{self.symbol}] {side_u} late-scaled by intelligence guard: {qty} -> {scaled_qty} (scale={scale_f:.2f})")
        return scaled_qty, scale_f, None

    def _init_and_gate_pre_pipeline(
        self,
        side: str,
        price: Any,
        qty: Any,
        kwargs: dict,
    ) -> tuple[Optional[_PlaceContext], Any]:
        caller_owns_retry = _parse_bool_flag(kwargs.pop("caller_owns_retry", False), False)
        outcome_context = kwargs.pop("_outcome_context", None)
        if outcome_context is not None and not isinstance(outcome_context, dict):
            raise TypeError("_outcome_context must be a dict")
        cache_permit = kwargs.pop("cache_permit", None)
        retry_requested_price_raw = kwargs.pop("_retry_requested_price", None)
        retry_price_tolerance_raw = kwargs.pop("_retry_price_tolerance", None)
        bypass_guards_raw = kwargs.pop("bypass_guards", None)
        caller_profit_margin = kwargs.pop("profit_margin", None)
        profit_reference_mode = kwargs.pop("reference_mode", None)
        caller_module = kwargs.pop("module", None)
        allow_math_reference_sell = _parse_bool_flag(
            kwargs.pop("allow_math_reference_sell", False), False
        )
        order_kind = kwargs.get("kind") or kwargs.get("motivation")
        if order_kind is not None:
            kwargs["kind"] = order_kind

        def _refuse_early(r: str):
            if outcome_context is not None:
                outcome_context.update(accepted=False, reason=r, state="refused")
            return None

        side_u = normalize_side(side)
        if not side_u:
            print(f"[{self.symbol}] {side!r} BLOCKED (pre-submit): invalid_side")
            return None, _refuse_early("invalid_side")

        ctx = _PlaceContext(
            side_u=side_u,
            price=price,
            qty=qty,
            orig_price=price,
            orig_qty=qty,
            kwargs=kwargs,
            caller_owns_retry=caller_owns_retry,
            outcome_context=outcome_context,
            cache_permit=cache_permit,
            caller_supplied_cache_permit=cache_permit is not None,
            retry_requested_price_raw=retry_requested_price_raw,
            retry_price_tolerance_raw=retry_price_tolerance_raw,
            retry_constraint_supplied=(
                retry_requested_price_raw is not None or retry_price_tolerance_raw is not None
            ),
            profit_reference_mode=profit_reference_mode,
            caller_module=caller_module,
            allow_math_reference_sell=allow_math_reference_sell,
            order_kind=order_kind,
        )
        ctx.medium_trend_action_cfg = ctx.kwargs.get("medium_trend_action")
        ctx.medium_trend_scale_cfg = ctx.kwargs.get("medium_trend_scale")
        inst_bypass_raw = None
        if hasattr(self, "params") and isinstance(self.params, dict):
            inst_bypass_raw = self.params.get("bypass_guards")
            if ctx.profit_reference_mode is None:
                ctx.profit_reference_mode = self.params.get(
                    f"{ctx.side_u.lower()}_reference_mode",
                    self.params.get("reference_mode"),
                )
            if ctx.caller_module is None:
                ctx.caller_module = self.params.get("module")
            if ctx.medium_trend_action_cfg is None:
                ctx.medium_trend_action_cfg = self.params.get("medium_trend_action")
            if ctx.medium_trend_scale_cfg is None:
                ctx.medium_trend_scale_cfg = self.params.get("medium_trend_scale")
        ctx.caller_module = (
            order_guard.resolve_caller_module(
                None,
                prefix=ctx.kwargs.get("prefix"),
                client_order_id=ctx.kwargs.get("client_order_id"),
            )
            if ctx.caller_module is None
            else order_guard.resolve_caller_module(ctx.caller_module)
        )
        ctx.v_norm = normalize_venue_name(
            getattr(self._provider, "name", "") or self.provider_name
        )
        ctx.bypassed_guards = order_guard.resolve_effective_bypass_guards(
            ctx.v_norm, bypass_guards=bypass_guards_raw, inst_bypass=inst_bypass_raw
        )
        ctx.bypass_all = "all" in ctx.bypassed_guards
        ctx.bypass_profit_guard = "profit_guard" in ctx.bypassed_guards or ctx.bypass_all
        ctx.bypass_math_ref = (
            GUARD_REGISTRY["G_PROF"].is_bypassed(ctx.bypassed_guards)
            or ctx.bypass_profit_guard
        )
        ctx.bypass_quantity_policy = (
            GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(ctx.bypassed_guards)
            or ctx.bypass_profit_guard
        )

        if (
            ctx.bypass_math_ref
            and not ctx.bypass_profit_guard
            and ctx.side_u != "BUY"
            and not ctx.allow_math_reference_sell
        ):
            print(
                f"[GUARD] {ctx.side_u} {self.symbol}: bypass math_profit/math_reference "
                "ignored; is allowed only for BUY"
            )
            revoke_guard_bypass(ctx.bypassed_guards, "G_PROF")
            ctx.bypass_math_ref = False
        if (
            ctx.bypass_quantity_policy
            and not ctx.bypass_profit_guard
            and ctx.side_u != "SELL"
            and ctx.caller_module != "assetguardian"
        ):
            print(
                f"[GUARD] {ctx.side_u} {self.symbol}: bypass_quantity_policy "
                "ignored; is allowed only for SELL"
            )
            revoke_guard_bypass(ctx.bypassed_guards, "G_QLIMIT1")
            ctx.bypass_quantity_policy = False

        ctx.wait_for_trend = _parse_bool_flag(kwargs.pop("wait_for_trend", True), True)
        ctx.abort_if = kwargs.pop("abort_if", None)
        ctx.early_abort_if = getattr(ctx.abort_if, "is_aborted", None)
        ctx.requires_permit = bool(
            getattr(self._provider, "requires_cache_permit", lambda: False)()
        )
        ctx.defers_smart_cancel = bool(
            getattr(self._provider, "defers_smart_cancellation", lambda: False)()
        )
        ctx.supports_biz_min = bool(
            getattr(self._provider, "supports_business_minimum", lambda: False)()
        )
        if ctx.cache_permit is not None and not ctx.requires_permit:
            print(f"[{self.symbol}] {side_u} BLOCKED (pre-submit): invalid_cache_permit_provider")
            return None, _refuse_early("invalid_cache_permit_provider")
        ctx.execution_enabled = getattr(self._provider, "execution_enabled", None)
        inst_live = self.param("execution", "live_enabled", None, bool)
        try:
            provider_execution_enabled = (
                bool(inst_live)
                if inst_live is not None
                else (True if not callable(ctx.execution_enabled) else bool(ctx.execution_enabled()))
            )
        except Exception as exc:  # noqa: BLE001 - An unreadable gate must fail closed.
            print(
                f"[{self.symbol}] {side_u} BLOCKED (pre-submit): execution_gate_unavailable ({exc})"
            )
            return None, _refuse_early("execution_gate_unavailable")
        if not provider_execution_enabled:
            print(f"[{self.symbol}] {side_u} DRY: real provider execution is disabled")
            return None, _refuse_early("execution_disabled")
        if inst_live is not None and "live_enabled" not in kwargs:
            kwargs["live_enabled"] = provider_execution_enabled
        supports_validity = getattr(
            self._provider,
            "supports_order_validity",
            lambda: MarketDataProvider.supports_order_validity(self._provider),
        )()
        if supports_validity and "order_validity" not in kwargs:
            inst_validity = self.param("execution", "order_validity")
            if isinstance(inst_validity, str) and inst_validity.strip():
                kwargs["order_validity"] = inst_validity.strip().upper()
        if self._provider.guards_internally():
            if ctx.retry_constraint_supplied:
                return None, _refuse_early("retry_price_constraint_unsupported")
            return None, self._provider.place_order(self.symbol, side_u, price, qty, **kwargs)

        ctx.safeback_override = kwargs.get("safeback_seconds")
        ctx.smart = _parse_bool_flag(kwargs.pop("smart", True), True)
        ctx.regime_context = kwargs.pop("regime_context", None)
        cas_f = safe_pos_float(kwargs.pop("applied_scale", None), default=None)
        ctx.caller_applied_scale = cas_f if (cas_f is not None and cas_f < 1.0) else 1.0
        ctx.applied_scale = ctx.caller_applied_scale
        ctx.is_monitor_replace = "monitor_order_replace" in str(order_kind or "")
        ctx.profit_margin = safe_float(caller_profit_margin, default=None)
        ctx.retry_kwargs = {
            **ctx.kwargs,
            "bypass_guards": sorted(ctx.bypassed_guards),
            "wait_for_trend": ctx.wait_for_trend,
            "smart": ctx.smart,
        }
        if ctx.caller_applied_scale < 1.0:
            ctx.retry_kwargs["applied_scale"] = ctx.caller_applied_scale
        for rk, rv in (
            ("profit_margin", ctx.profit_margin),
            ("reference_mode", ctx.profit_reference_mode),
            ("module", ctx.caller_module),
            ("medium_trend_action", ctx.medium_trend_action_cfg),
            ("medium_trend_scale", ctx.medium_trend_scale_cfg),
        ):
            if rv is not None:
                ctx.retry_kwargs[rk] = rv
        if ctx.allow_math_reference_sell:
            ctx.retry_kwargs["allow_math_reference_sell"] = ctx.allow_math_reference_sell

        ctx.is_market = resolve_market_flag(**ctx.kwargs)
        caller_enforce_biz_min = ctx.kwargs.pop("enforce_business_minimum", None)
        if caller_enforce_biz_min is not None:
            ctx.enforce_business_minimum = _parse_bool_flag(caller_enforce_biz_min, True)
            ctx.retry_kwargs["enforce_business_minimum"] = ctx.enforce_business_minimum
        else:
            ctx.enforce_business_minimum = not (
                (ctx.bypass_profit_guard and ctx.side_u == "SELL" and ctx.is_market)
                or (ctx.caller_module == "assetguardian" and ctx.side_u == "SELL")
            )
        return ctx, None

    def _finalize_place_outcome(
        self, ctx: _PlaceContext, *, caller: Optional[str]
    ) -> None:
        submission_state = (
            "accepted"
            if ctx.order is not None
            else "unknown"
            if ctx.submit_attempted
            else "refused"
        )
        if ctx.outcome_context is not None:
            ctx.outcome_context.update(
                accepted=ctx.order is not None,
                reason=None if ctx.order is not None else ctx.reason,
                state=submission_state,
            )
            if ctx.order is not None or ctx.submit_attempted:
                ctx.outcome_context["submitted_qty"] = ctx.qty
                ctx.outcome_context["submitted_price"] = ctx.price
            cid = ctx.prequeued_client_order_id or ctx.kwargs.get("client_order_id")
            if cid:
                ctx.outcome_context["client_order_id"] = cid
            iid = ctx.prequeued_intent_id or ctx.kwargs.get("intent_id")
            if iid:
                ctx.outcome_context["intent_id"] = iid
        terminal_filter_refusal = bool(
            ctx.reason
            and not ctx.submit_attempted
            and order_retry.is_terminal_filter_refusal(ctx.reason)
        )
        order_lifecycle.log_order_outcome(
            self.symbol,
            ctx.side_u,
            ctx.price,
            ctx.qty,
            submission_state,
            None if ctx.order else ctx.reason,
            ctx.order_kind,
            caller=caller,
            regime_context=ctx.regime_context,
        )
        if ctx.order is not None and not ctx.caller_owns_retry:
            try:
                if ctx.prequeued_claim is not None:
                    tracked = order_lifecycle.complete_accepted_claim(
                        order_retry,
                        ctx.prequeued_claim,
                        order=ctx.order,
                        provider_name=self.provider_name,
                        symbol=self.symbol,
                        side=ctx.side_u,
                    )
                    if not tracked:
                        print(
                            f"[{self.symbol}] {ctx.side_u} accepted, but the tracking "
                            "could not be updated; the intent stays pending"
                        )
                    elif ctx.prequeued_intent_id:
                        _EXECUTION_AUDIT.record(
                            "submit_accepted",
                            intent_id=ctx.prequeued_intent_id,
                            venue=self.provider_name,
                            symbol=self.symbol,
                            side=ctx.side_u.lower(),
                            qty=ctx.qty,
                            price=ctx.price,
                            market=ctx.is_market,
                            kind=ctx.order_kind,
                            client_order_id=ctx.prequeued_client_order_id,
                            order_id=extract_order_id(ctx.order),
                        )
            except Exception as _e:  # noqa: BLE001
                print(f"[{self.symbol}] {ctx.side_u} tracking accepted failed: {_e}")
        elif (
            ctx.order is None
            and not ctx.caller_owns_retry
            and (
                ctx.prequeued_record_id is not None
                or (
                    ctx.reason
                    not in {
                        "account_cache_not_fresh",
                        "execution_disabled",
                        "trading_disabled",
                        "aborted_by_caller",
                        "invalid_price",
                    }
                    and not terminal_filter_refusal
                )
            )
        ):
            try:
                if ctx.prequeued_intent_id:
                    _EXECUTION_AUDIT.record(
                        "submit_unknown" if ctx.submit_attempted else "submit_refused",
                        intent_id=ctx.prequeued_intent_id,
                        venue=self.provider_name,
                        symbol=self.symbol,
                        side=ctx.side_u.lower(),
                        qty=ctx.qty,
                        price=ctx.price,
                        market=ctx.is_market,
                        kind=ctx.order_kind,
                        client_order_id=ctx.prequeued_client_order_id,
                        failure_reason=ctx.reason,
                    )
                if order_retry.RETRY_ENABLED:
                    if ctx.prequeued_claim is not None:
                        claim_outcome = (
                            "success"
                            if (
                                terminal_filter_refusal
                                or ctx.reason
                                in {
                                    "execution_disabled",
                                    "trading_disabled",
                                    "aborted_by_caller",
                                    "invalid_price",
                                }
                            )
                            else "deferred"
                            if order_retry.is_non_failure_deferral(ctx.reason)
                            else "failure"
                        )
                        order_retry.complete_claim(
                            ctx.prequeued_claim,
                            claim_outcome,
                            failure_reason=ctx.reason,
                            submission_state="unknown" if ctx.submit_attempted else "refused",
                        )
                    else:
                        q_orig = safe_pos_float(ctx.orig_qty, default=None)
                        if q_orig is None:
                            queue_qty = ctx.qty
                            app_sc = safe_pos_float(ctx.applied_scale, default=1.0) or 1.0
                            if 0.0 < app_sc < 1.0:
                                ctx.retry_kwargs["applied_scale"] = app_sc
                        else:
                            queue_qty = q_orig
                        if safe_pos_float(queue_qty, default=None) is not None:
                            try:
                                ref_price = self._provider.get_current_price(self.symbol)
                            except Exception:  # noqa: BLE001
                                ref_price = None
                            self._apply_outbox_lineage_prefix(ctx.retry_kwargs, ctx.kwargs)
                            order_retry.enqueue(
                                self.symbol,
                                ctx.side_u,
                                queue_qty,
                                ctx.retry_kwargs,
                                requested_price=ctx.orig_price,
                                ref_price=ref_price,
                                failure_reason=ctx.reason,
                                provider_name=self.provider_name,
                                intent_id=ctx.kwargs.get("intent_id"),
                                kind=ctx.order_kind,
                                ttl_sec=ctx.kwargs.get("ttl_sec"),
                            )
            except Exception as _e:  # noqa: BLE001
                print(f"[{self.symbol}] {ctx.side_u} enqueue retry failed (ignored): {_e}")

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

    def cancel_order(self, order_id: str) -> None:
        cancel_fn = getattr(self._provider, "cancel_order", None)
        if callable(cancel_fn):
            return cancel_fn(self.symbol, str(order_id))
        return None

    # -- Order placement: dry or real after provider gates. --------------------
    def place(self, side: str, price: float, qty: Optional[float] = None, **kwargs):
        ctx, early_return = self._init_and_gate_pre_pipeline(side, price, qty, kwargs)
        if ctx is None:
            return early_return
        try:
            if ctx.retry_constraint_supplied:
                ctx.retry_requested_price = safe_pos_float(ctx.retry_requested_price_raw, default=None)
                ctx.retry_price_tolerance = safe_nonneg_float(
                    ctx.retry_price_tolerance_raw, default=None
                )
                if (
                    ctx.retry_requested_price is None
                    or ctx.retry_price_tolerance is None
                    or ctx.retry_price_tolerance >= 1.0
                ):
                    raise SubmissionRefused("invalid_retry_price_constraint")
            ctx.preflight_order = getattr(self._provider, "preflight_order", None)
            if ctx.requires_permit and ctx.smart and ctx.caller_supplied_cache_permit:
                raise SubmissionRefused("external_cache_permit_not_allowed_for_smart_cancel")
            if ctx.requires_permit and ctx.cache_permit is None:
                prepare_order_state = getattr(self._provider, "prepare_order_state", None)
                if not callable(prepare_order_state):
                    raise SubmissionRefused("account_cache_not_fresh")
                ctx.guarded_order_state = prepare_order_state()
                if ctx.guarded_order_state is None:
                    raise SubmissionRefused("account_cache_not_fresh")
            if not ctx.requires_permit and callable(ctx.preflight_order):
                defers_qty = getattr(
                    self._provider,
                    "defers_preflight_quantity",
                    lambda: MarketDataProvider.defers_preflight_quantity(self._provider),
                )()
                preflight_qty = None if defers_qty else ctx.qty
                ctx.preflight_order(
                    self.symbol,
                    ctx.side_u,
                    preflight_qty,
                    ctx.price,
                    market=ctx.is_market,
                    kind=ctx.order_kind,
                )

            # 0. Compute smart venue pricing before the guards.
            if ctx.smart and not isinstance(ctx.price, bool):
                ctx.price = self._provider.adjust_order_price(
                    self.symbol,
                    ctx.side_u,
                    ctx.price,
                    cancel_opposite=not ctx.defers_smart_cancel,
                )
            if not ctx.is_market and ctx.price is not None and not isinstance(ctx.price, bool):
                round_px = getattr(self._provider, "round_price", None)
                if callable(round_px):
                    px_f = safe_pos_float(round_px(self.symbol, ctx.price), default=None)
                    if px_f is not None and not (
                        ctx.caller_supplied_cache_permit and ctx.side_u == "BUY" and px_f > ctx.price
                    ):
                        ctx.price = px_f

            # 1. Provider-agnostic daily trade cap (30 orders / 24h per symbol and side).
            if not GUARD_REGISTRY["GO1"].is_bypassed(ctx.bypassed_guards):
                ok, ctx.reason = order_guard.daily_limit_guard(
                    self._provider, self.symbol, ctx.side_u
                )
                if not ok:
                    return None

            # Stage 0 & Stage 1: Authoritative Balance/Price & Mathematical Reference Check (G_PROF).
            if isinstance(ctx.price, bool) or (
                ctx.price is not None and (ctx.side_u == "BUY" or not ctx.is_market)
            ):
                price_f = safe_pos_float(ctx.price, default=None)
                if price_f is None:
                    print(f"[{self.symbol}] {ctx.side_u} preflight refused: price invalid {ctx.price!r}")
                    ctx.quantity_price = ctx.price
                    ctx.reason = "invalid_price"
                    return None
                ctx.price = price_f

            ctx.base_asset, ctx.quote_asset = resolve_assets(self.symbol, self.base, self.quote)
            target_asset = ctx.quote_asset if ctx.side_u == "BUY" else ctx.base_asset

            ctx.known_balance = None
            if (
                hasattr(self._provider, "free_balance")
                and callable(self._provider.free_balance)
                and target_asset
            ):
                try:
                    kb_f = safe_float(self._provider.free_balance(target_asset), default=None)
                    if kb_f is not None:
                        if kb_f <= 0:
                            print(
                                f"[{self.symbol}] {ctx.side_u} preflight refused: "
                                f"balance_unavailable asset={target_asset}"
                            )
                            ctx.quantity_price = ctx.price
                            ctx.reason = "balance_unavailable"
                            return None
                        ctx.known_balance = kb_f
                except Exception:
                    ctx.known_balance = None

            ctx.quantity_price = ctx.price
            if ctx.is_market:
                ctx.quantity_price = safe_pos_float(
                    self._provider.get_current_price(self.symbol), default=None
                )
                if ctx.quantity_price is None:
                    print(f"[{self.symbol}] {ctx.side_u} MARKET BLOCKED: current price unavailable")
                    ctx.reason = "market_price_unavailable"
                    return None

            ctx.profit_window_ref = None
            if not ctx.bypass_profit_guard:
                try:
                    eff_ref_mode = order_guard.get_profit_guard_mode_cfg(
                        self._provider.name,
                        ctx.side_u,
                        module=ctx.caller_module,
                        reference_mode=ctx.profit_reference_mode,
                    )
                except Exception:
                    eff_ref_mode = None
                if ctx.regime_context is None:
                    ctx.regime_context = self._resolve_regime_context()
                if ctx.profit_margin is None:
                    ctx.profit_margin = order_guard.get_margin_cfg(self._provider.name)
                if not ctx.bypass_math_ref and eff_ref_mode != "off":
                    ctx.profit_window_ref = self._call_profit_guard_window_ref(
                        self.symbol,
                        ctx.side_u,
                        ctx.safeback_override,
                        regime_context=ctx.regime_context,
                        reference_mode=ctx.profit_reference_mode,
                        module=ctx.caller_module,
                    )

                if ctx.bypass_math_ref:
                    print(
                        f"[GUARD] {ctx.side_u} {self.symbol}: the historical price reference "
                        "was explicitly bypassed; the quantity/weight guard remains active"
                    )
                elif (
                    not ctx.bypass_all
                    and not hasattr(order_guard.profit_guard, "_mock_name")
                    and getattr(order_guard.profit_guard, "__name__", "") == "profit_guard"
                ):
                    guard_px_stage1 = ctx.quantity_price if ctx.is_market else ctx.price
                    cm_kwargs = self._build_profit_guard_kwargs(
                        ctx.profit_window_ref,
                        ctx.regime_context,
                        None,
                        None,
                        ctx.profit_reference_mode,
                        ctx.caller_module,
                        include_qty=False,
                    )
                    if not order_guard.check_math_profit_reference(
                        self._provider,
                        self.symbol,
                        ctx.side_u,
                        guard_px_stage1,
                        ctx.profit_margin,
                        **cm_kwargs,
                    ):
                        ctx.reason = "profit_guard"
                        return None

            if _is_aborted(ctx.early_abort_if):
                ctx.reason = "aborted_by_caller"
                return None

            # Stage 2: Quantity Sizing, Gaussian/Chop Weights & Balance Capping.
            hours_f = safe_pos_float(ctx.kwargs.get("hours", 5), default=5.0)
            decision = self._provider.quantity_decision(
                self.symbol,
                ctx.side_u,
                ctx.quantity_price,
                ctx.qty,
                base=ctx.base_asset,
                quote=ctx.quote_asset,
                cancelorders=_parse_bool_flag(ctx.kwargs.get("cancelorders", False), False),
                hours=hours_f,
                apply_policy=not ctx.bypass_quantity_policy,
                market=ctx.is_market,
                enforce_business_minimum=ctx.enforce_business_minimum,
                regime_context=(
                    None
                    if (
                        ctx.caller_applied_scale < 1.0
                        or ctx.bypass_profit_guard
                        or ctx.is_monitor_replace
                    )
                    else ctx.regime_context
                ),
                known_balance=ctx.known_balance,
            )
            ctx.qty = decision.final_qty
            raw_dec_scale = (
                getattr(decision, "applied_scale", 1.0)
                if getattr(decision, "scale_applied", False)
                else 1.0
            )
            dec_scale = safe_pos_float(raw_dec_scale, default=1.0)
            ctx.applied_scale = min(float(ctx.caller_applied_scale), dec_scale)
            if ctx.qty <= 0:
                print(
                    f"[{self.symbol}] {ctx.side_u} qty refused: "
                    f"{decision.refuse_reason} asset={decision.balance_asset}"
                )
                ctx.reason = decision.refuse_reason or "qty_zero_after_weight"
                return None

            if (
                not ctx.bypass_profit_guard
                and not ctx.is_monitor_replace
                and ctx.regime_context is not None
                and ctx.side_u == "BUY"
                and ctx.qty > 0
                and not getattr(decision, "scale_applied", False)
            ):
                ctx.qty, ctx.applied_scale, refusal = self._apply_late_intelligence_scale(
                    ctx.qty,
                    ctx.quantity_price,
                    ctx.is_market,
                    ctx.enforce_business_minimum,
                    ctx.regime_context,
                    ctx.applied_scale,
                    side_u=ctx.side_u,
                )
                if refusal:
                    ctx.reason = refusal
                    return None

            if _is_aborted(ctx.early_abort_if):
                ctx.reason = "aborted_by_caller"
                return None

            # 2 & 2b: Short Trend Gate (GT_S) & Medium-Term Trend Gate (GT_M).
            if ctx.wait_for_trend:
                try:
                    if pw.should_wait_short_trend(
                        ct.get_short_trend_manager(),
                        ctx.side_u,
                        self.symbol,
                        short_trend_cls=ct.CacheShortTrendManager,
                    ):
                        retry_note = (
                            "caller owns retry"
                            if ctx.caller_owns_retry
                            else "the intent stays queued for retry"
                        )
                        print(
                            f"[{self.symbol}] {ctx.side_u} deferred by trend; "
                            f"placement does not wait, {retry_note}"
                        )
                        ctx.reason = "trend_deferred"
                        return None
                except Exception as e:  # noqa: BLE001 — Opportunistic gate.
                    print(f"[{self.symbol}] {ctx.side_u} trend gate unavailable: {e}")

            check_medium_trend = not GUARD_REGISTRY["GT_M"].is_bypassed(ctx.bypassed_guards)
            medium_trend_action = (
                ctx.medium_trend_action_cfg.strip().lower()
                if isinstance(ctx.medium_trend_action_cfg, str)
                else None
            )
            gt2_context = ctx.regime_context
            if check_medium_trend and gt2_context is None and medium_trend_action in {"defer", "scale"}:
                gt2_context = self._resolve_regime_context()
                if gt2_context is not None and ctx.regime_context is None:
                    ctx.regime_context = gt2_context
            if check_medium_trend and gt2_context is not None:
                raw_reg = obj_attr(gt2_context, "regime") or obj_attr(gt2_context, "resolved_trend")
                resolved_regime = raw_reg.strip().lower() if isinstance(raw_reg, str) else None
                adverse_medium_trend = order_guard.is_adverse_trend(ctx.side_u, resolved_regime)
                if medium_trend_action == "defer" and adverse_medium_trend:
                    regime_label = (
                        "BEAR regime" if ctx.side_u == "BUY" else "BULL regime, riding trend"
                    )
                    print(
                        f"[{self.symbol}] {ctx.side_u} deferred by medium-term trend "
                        f"(4h-24h {regime_label})"
                    )
                    ctx.reason = "medium_trend_deferred"
                    return None
                if (
                    medium_trend_action == "scale"
                    and adverse_medium_trend
                    and not ctx.is_monitor_replace
                ):
                    scale_factor = parse_scale_factor(ctx.medium_trend_scale_cfg, default=0.5)
                    if scale_factor <= 0.0:
                        ctx.reason = "medium_trend_qty_zero"
                        return None
                    if scale_factor < float(ctx.applied_scale):
                        ctx.qty = round_provider_qty(
                            self._provider,
                            self.symbol,
                            max(0.0, ctx.qty * (scale_factor / float(ctx.applied_scale))),
                        )
                        print(
                            f"[{self.symbol}] {ctx.side_u} scaled by medium-term trend "
                            f"({scale_factor*100:.0f}%, new qty={ctx.qty})"
                        )
                        if ctx.qty <= 0:
                            ctx.reason = "medium_trend_qty_zero"
                            return None
                        gt2_refusal = check_order_filter_refusal(
                            self._provider,
                            self.symbol,
                            ctx.side_u,
                            ctx.quantity_price if ctx.is_market else ctx.price,
                            ctx.qty,
                            market=ctx.is_market,
                            enforce_business_minimum=ctx.enforce_business_minimum,
                        )
                        if gt2_refusal:
                            print(
                                f"[{self.symbol}] {ctx.side_u} medium-trend scaled qty {ctx.qty} "
                                f"refused by filter: {gt2_refusal}"
                            )
                            ctx.reason = gt2_refusal
                            return None
                        ctx.applied_scale = min(float(ctx.applied_scale), scale_factor)

            if _is_aborted(ctx.early_abort_if):
                ctx.reason = "aborted_by_caller"
                return None

            # Stage 3: Post-sizing Profit & Intelligence Guards.
            if not ctx.bypass_profit_guard and ctx.qty > 0:
                guard_price = ctx.quantity_price if ctx.is_market else ctx.price
                pg_kwargs = self._build_profit_guard_kwargs(
                    ctx.profit_window_ref,
                    ctx.regime_context,
                    ctx.qty,
                    ctx.bypassed_guards,
                    ctx.profit_reference_mode,
                    ctx.caller_module,
                )
                if not order_guard.profit_guard(
                    self._provider,
                    self.symbol,
                    ctx.side_u,
                    guard_price,
                    ctx.profit_margin,
                    **pg_kwargs,
                ):
                    ctx.reason = "profit_guard"
                    return None
                if not ctx.is_monitor_replace:
                    ctx.qty, ctx.applied_scale, refusal = self._apply_late_intelligence_scale(
                        ctx.qty,
                        guard_price,
                        ctx.is_market,
                        ctx.enforce_business_minimum,
                        ctx.regime_context,
                        ctx.applied_scale,
                        side_u=ctx.side_u,
                    )
                    if refusal:
                        ctx.reason = refusal
                        return None

            if _is_aborted(ctx.abort_if):
                ctx.reason = "aborted_by_caller"
                return None
            if callable(ctx.execution_enabled) and not bool(ctx.execution_enabled()):
                raise SubmissionRefused("execution_disabled")

            # Stage 4: Persistent Outbox Claim Reservation & Cache Permit Acquisition.
            if not ctx.caller_owns_retry:
                try:
                    if order_retry.RETRY_ENABLED:
                        prequeue_kwargs = {**ctx.retry_kwargs, "smart": ctx.smart}
                        if 0.0 < float(ctx.applied_scale) < 1.0:
                            prequeue_kwargs["applied_scale"] = float(ctx.applied_scale)
                        self._apply_outbox_lineage_prefix(prequeue_kwargs, ctx.kwargs)
                        ctx.prequeued_claim = order_retry.enqueue_claimed(
                            self.symbol,
                            ctx.side_u,
                            ctx.qty,
                            prequeue_kwargs,
                            requested_price=ctx.orig_price,
                            ref_price=ctx.price,
                            failure_reason="submit_pending",
                            provider_name=self.provider_name,
                            intent_id=ctx.kwargs.get("intent_id"),
                            kind=ctx.order_kind,
                            ttl_sec=ctx.kwargs.get("ttl_sec"),
                        )
                        if ctx.prequeued_claim is None:
                            raise RuntimeError("the intent cannot be read back after persisting")
                        ctx.prequeued_client_order_id = dict(
                            ctx.prequeued_claim.get("place_kwargs") or {}
                        ).get("client_order_id")
                        if not ctx.prequeued_client_order_id:
                            raise RuntimeError("the persisted intent has no client_order_id")
                        ctx.kwargs["client_order_id"] = ctx.prequeued_client_order_id
                except Exception as exc:
                    print(
                        f"[{self.symbol}] {ctx.side_u} BLOCKED: the intent cannot be "
                        f"persisted before the submit ({exc})"
                    )
                    ctx.reason = "pre_submit_persist_failed"
                    return None
            else:
                ctx.prequeued_claim = ctx.kwargs.get("_retry_claim")

            if ctx.prequeued_claim is not None:
                ctx.prequeued_record_id = ctx.prequeued_claim.get("id")
                ctx.prequeued_client_order_id = dict(
                    ctx.prequeued_claim.get("place_kwargs") or {}
                ).get("client_order_id")
                ctx.prequeued_intent_id = ctx.prequeued_claim.get("intent_id")

            if ctx.prequeued_intent_id and not ctx.caller_owns_retry:
                _EXECUTION_AUDIT.record(
                    "submit_requested",
                    intent_id=ctx.prequeued_intent_id,
                    venue=self.provider_name,
                    symbol=self.symbol,
                    side=ctx.side_u.lower(),
                    qty=ctx.qty,
                    price=ctx.price,
                    market=ctx.is_market,
                    kind=ctx.order_kind,
                    client_order_id=ctx.prequeued_client_order_id,
                )

            ctx.permit_requested_price = None if ctx.is_market else ctx.orig_price
            if ctx.requires_permit and ctx.cache_permit is None:
                if not callable(ctx.preflight_order):
                    raise SubmissionRefused("account_cache_not_fresh")
                ctx.cache_permit = ctx.preflight_order(
                    self.symbol,
                    ctx.side_u,
                    ctx.qty,
                    ctx.price,
                    market=ctx.is_market,
                    kind=ctx.order_kind,
                )
                if ctx.cache_permit is None:
                    raise SubmissionRefused("account_cache_not_fresh")
                ctx.permit_requested_price = None if ctx.is_market else ctx.price
                validate_order_state = getattr(self._provider, "validate_order_state", None)
                if not callable(validate_order_state):
                    raise SubmissionRefused("account_cache_not_fresh")
                validate_order_state(ctx.guarded_order_state)

            # Stage 5: Pre-dispatch Market Revalidation & Venue Submission.
            final_profit_check = not ctx.bypass_profit_guard and ctx.is_market
            if ctx.is_market and (final_profit_check or ctx.retry_constraint_supplied):
                final_market_price = safe_pos_float(
                    self._provider.get_current_price(self.symbol), default=None
                )
                if final_market_price is None:
                    print(
                        f"[{self.symbol}] {ctx.side_u} MARKET BLOCKED at dispatch: "
                        "current price unavailable"
                    )
                    ctx.reason = "market_price_unavailable"
                    return None
                if ctx.retry_constraint_supplied and not order_retry.is_retry_price_favorable(
                    ctx.side_u,
                    final_market_price,
                    ctx.retry_requested_price,
                    ctx.retry_price_tolerance,
                ):
                    ctx.reason = "retry_price_unfavorable"
                    return None
                if final_profit_check:
                    pg_kwargs = self._build_profit_guard_kwargs(
                        ctx.profit_window_ref,
                        ctx.regime_context,
                        ctx.qty,
                        ctx.bypassed_guards,
                        ctx.profit_reference_mode,
                        ctx.caller_module,
                    )
                    if not order_guard.profit_guard(
                        self._provider,
                        self.symbol,
                        ctx.side_u,
                        final_market_price,
                        ctx.profit_margin,
                        **pg_kwargs,
                    ):
                        ctx.reason = "profit_guard"
                        return None
                    pre_final_qty = ctx.qty
                    ctx.qty, ctx.applied_scale, refusal = self._apply_late_intelligence_scale(
                        ctx.qty,
                        final_market_price,
                        ctx.is_market,
                        ctx.enforce_business_minimum,
                        ctx.regime_context,
                        ctx.applied_scale,
                        side_u=ctx.side_u,
                    )
                    if refusal:
                        ctx.reason = refusal
                        return None
                    if (
                        ctx.qty != pre_final_qty
                        and ctx.requires_permit
                        and not ctx.caller_supplied_cache_permit
                        and callable(ctx.preflight_order)
                    ):
                        ctx.cache_permit = ctx.preflight_order(
                            self.symbol,
                            ctx.side_u,
                            ctx.qty,
                            ctx.price,
                            market=ctx.is_market,
                            kind=ctx.order_kind,
                        )
                        if ctx.cache_permit is None:
                            raise SubmissionRefused("account_cache_not_fresh")
                    if ctx.prequeued_claim is not None:
                        ctx.prequeued_claim = self._sync_claimed_qty(
                            ctx.prequeued_claim, ctx.qty, ctx.applied_scale
                        )
                        if ctx.prequeued_intent_id and ctx.applied_scale < 1.0:
                            _EXECUTION_AUDIT.record(
                                "submit_scaled",
                                intent_id=ctx.prequeued_intent_id,
                                venue=self.provider_name,
                                symbol=self.symbol,
                                side=ctx.side_u.lower(),
                                qty=ctx.qty,
                                scale=ctx.applied_scale,
                            )

            if _is_aborted(ctx.abort_if):
                ctx.reason = "aborted_by_caller"
                return None
            if callable(ctx.execution_enabled) and not bool(ctx.execution_enabled()):
                raise SubmissionRefused("execution_disabled")

            if ctx.prequeued_claim is not None:
                if not ctx.caller_owns_retry:
                    refreshed_claim = order_retry.begin_claimed_submit(
                        ctx.prequeued_claim, qty=ctx.qty, applied_scale=ctx.applied_scale
                    )
                    if refreshed_claim is None:
                        raise SubmissionRefused("producer_claim_lost")
                    ctx.prequeued_claim = refreshed_claim
                else:
                    ctx.prequeued_claim = self._sync_claimed_qty(
                        ctx.prequeued_claim, ctx.qty, ctx.applied_scale
                    )
            ctx.reason = "submit_ambiguous"
            ctx.submit_attempted = True
            provider_kwargs = {
                **ctx.kwargs,
                "market": ctx.is_market,
                "force": ctx.is_market,
                "_balance_verified": True,
            }
            if ctx.known_balance is not None:
                provider_kwargs["known_balance"] = ctx.known_balance
            if ctx.supports_biz_min:
                provider_kwargs["enforce_business_minimum"] = ctx.enforce_business_minimum
            if ctx.cache_permit is not None:
                provider_kwargs["permit_requested_price"] = ctx.permit_requested_price
                provider_kwargs["cache_permit"] = ctx.cache_permit
            if ctx.smart and ctx.defers_smart_cancel:
                provider_kwargs["_cancel_opposite_requested_price"] = ctx.orig_price
            response = self._provider.place_order(
                self.symbol, ctx.side_u, ctx.price, ctx.qty, **provider_kwargs
            )
            if _accepted_order_payload(response):
                ctx.order, ctx.reason = response, None
            else:
                ctx.order, ctx.reason = None, "response_without_order_id"
            return ctx.order
        except SubmissionRefused as exc:
            ctx.reason = exc.reason
            ctx.submit_attempted = False
            print(f"[{self.symbol}] {ctx.side_u} BLOCKED (pre-submit): {ctx.reason}")
            return None
        except Exception as e:  # noqa: BLE001 — Fail closed when guards cannot be verified.
            print(f"[{self.symbol}] {ctx.side_u} BLOCKED (fail-closed): {e}")
            ctx.reason = ctx.reason or "guard_check_failed"
            return None
        finally:
            try:
                caller = os.path.basename(sys._getframe(1).f_code.co_filename)
            except Exception:
                caller = None
            self._finalize_place_outcome(ctx, caller=caller)

    def min_qty(self) -> float:
        """Return the venue minimum order quantity; ``0`` means no known minimum."""
        try:
            return safe_pos_float(self._provider.min_order_qty(self.symbol), default=0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def cached_fee_and_limits(self, *, price: Optional[float] = None) -> dict:
        """Return non-blocking cached fee tier and exchange limit overrides from the bound provider."""
        return MarketDataProvider.resolve_provider_cached_fee_and_limits(
            self.provider_name,
            self.symbol,
            self._provider,
            price=price,
        )

    @classmethod
    def resolve_cached_fee_and_limits(
        cls,
        inst: Any,
        venue: str,
        symbol: str,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Resolve cached fee and limit metadata from an ``Instrument`` or compatible stub."""
        return MarketDataProvider.resolve_instrument_cached_fee_and_limits(
            inst, venue, symbol, price=price
        )

    # ── params namespaced (mt.* / tradeall.* / rtrade.* / execution.*) ──────────
    def param(self, consumer: str, key: str, default: Any = None,
              cast: Optional[Callable] = None) -> Any:
        """Return ``consumer.key`` or ``default`` when missing or conversion fails."""
        return InstrumentSpec.parse_param(self.params, consumer, key, default=default, cast=cast)

    def __repr__(self) -> str:
        return f"<Instrument {self.name} {self.symbol}@{self.provider_name} {'on' if self.enabled else 'off'}>"


def bind_instruments(specs: dict, api=None) -> dict:
    """Bind a mapping of ``InstrumentSpec`` records to executable ``Instrument`` instances."""
    cls = resolve_patched_attr("Instrument", Instrument, modules=(__name__,), require_callable=True)
    out = {}
    for name, item in (specs or {}).items():
        if _is_instrument_spec(item):
            out[name] = (
                cls.from_spec(item, api=api)
                if callable(getattr(cls, "from_spec", None))
                else cls(
                    name=item.name,
                    symbol=item.symbol,
                    provider=item.provider,
                    base=item.base,
                    quote=item.quote,
                    enabled=item.enabled,
                    isolation=item.isolation,
                    market_hours=item.market_hours,
                    params=item.params,
                    api=api,
                )
            )
        else:
            out[name] = item
    return out


def load_instruments(path=None, api=None) -> dict:
    """Load and validate ``instruments.conf``, then bind all records to ``MarketApi``."""
    return bind_instruments(load_registry(path), api=api)


def load_for(consumer, path=None, api=None, only_enabled=True) -> dict:
    """Bind instruments explicitly enabled for the requested consumer role."""
    return bind_instruments(
        select_instruments(role=consumer, path=path, only_enabled=only_enabled),
        api=api,
    )


def load_tradable_instruments(
    workspace_dir: str | None = None,
    *,
    only_workspace_conf: bool = False,
    include_closed_rth: bool = False,
    now_ts: float | None = None,
    asset_scope: str = "all",
    is_rth_open_fn=None,
    api=None,
) -> dict:
    """Discover enabled specs across configured venues and bind them to providers."""
    ws_dir = workspace_dir or os.path.dirname(os.path.abspath(__file__))
    MarketDataProvider.load_venue_env_defaults(ws_dir, asset_scope=asset_scope or "all")
    ws_inst_conf = os.path.join(ws_dir, "instruments.conf")
    base_specs: dict | None = None
    if os.path.exists(ws_inst_conf) or not only_workspace_conf:
        try:
            base_specs = (
                load_instruments(ws_inst_conf, api=api)
                if os.path.exists(ws_inst_conf)
                else load_instruments(api=api)
            )
        except Exception:
            base_specs = None

    specs_or_insts = discover_tradable_specs(
        ws_dir,
        only_workspace_conf=only_workspace_conf,
        include_closed_rth=include_closed_rth,
        now_ts=now_ts,
        asset_scope=asset_scope,
        is_rth_open_fn=is_rth_open_fn or is_us_rth_open,
        base_specs=base_specs,
    )
    active_map: dict = {}
    for name, item in specs_or_insts.items():
        if _is_instrument_spec(item):
            try:
                active_map[name] = bind_instruments({name: item}, api=api)[name]
            except Exception:
                continue
        else:
            active_map[name] = item
    return active_map


def resolve_tradable_instrument(
    symbol: str,
    venue: str | None = None,
    instruments_map: dict | None = None,
    *,
    workspace_dir: str | None = None,
    api=None,
):
    """Resolve an enabled ``Instrument`` instance for the given symbol and venue."""
    if instruments_map is None:
        instruments_map = load_tradable_instruments(workspace_dir, api=api)
    return resolve_instrument_spec(
        symbol,
        venue=venue,
        instruments_map=instruments_map,
        workspace_dir=workspace_dir,
    )


def resolve_instrument_live_price(inst, cache_dir: str = "cachedb") -> float:
    """Fetch live market price for ``inst`` from its bound provider or ``cacheprice`` SSOT fallbacks."""
    try:
        prov = getattr(inst, "_provider", None)
        if prov is not None and hasattr(prov, "get_current_price"):
            px_f = safe_pos_float(prov.get_current_price(inst.symbol), default=None)
            if px_f is not None:
                return px_f
    except Exception:
        pass

    venue = getattr(inst, "provider_name", None)
    if not isinstance(venue, str):
        venue = getattr(inst, "provider", None)
    raw_base = getattr(inst, "base", None)
    raw_quote = getattr(inst, "quote", None)
    return resolve_cached_symbol_price(
        str(getattr(inst, "symbol", "") or ""),
        venue=venue if isinstance(venue, str) else None,
        base=raw_base if isinstance(raw_base, str) else None,
        quote=raw_quote if isinstance(raw_quote, str) else None,
        cache_dir=cache_dir,
    )


MarketDataProvider.bind_instrument_class(Instrument)

