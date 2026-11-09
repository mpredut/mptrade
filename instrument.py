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
import sys
from typing import Optional, List, Callable, Any

from cachepaths import normalize_venue_name
from guard_hierarchy import (
    GUARD_CATALOG,  # noqa: F401 (re-export: test compat)
    GUARD_REGISTRY,
    normalize_bypassed_guards,
    revoke_guard_bypass,
)
from instrument_registry import casefold_get
from providers.base import extract_order_id, resolve_market_flag
from botcore import _parse_bool_flag, normalize_side, safe_float, safe_int, safe_nonneg_float, safe_pos_float
from providers.market_api import api as _default_api
from providers.execution_audit import ExecutionAudit
from providers.quantity import (
    check_order_filter_refusal,
    extract_regime_buy_scale,
    parse_scale_factor as _parse_scale_factor,
    resolve_assets,
    round_provider_qty,
)
from providers.strategy_executor import SubmissionRefused
import order_guard
import order_lifecycle as _outcomes_log
import order_lifecycle as accepted_order_persistence
import order_retry
import lock.cooldown as trade_cooldown  # noqa: F401 (re-export: test compat)


_EXECUTION_AUDIT = ExecutionAudit()


def _accepted_order_payload(payload):
    """Return whether a provider payload proves venue acceptance, never fill."""
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status") or "").upper() in {"REJECTED", "CANCELED", "CANCELLED", "EXPIRED", "FAILED"}:
        return False
    return extract_order_id(payload) is not None


_order_id_from_payload = extract_order_id


def _is_terminal_filter_refusal(reason) -> bool:
    """Use the retry subsystem's shared deterministic-refusal classification."""
    return order_retry.is_terminal_filter_refusal(reason)


def _is_aborted(cb: Any) -> bool:
    return callable(cb) and bool(cb())


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
        self.enabled = _parse_bool_flag(enabled, True)
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
                from binance_api.order_id_context import get_process_name
                pname = get_process_name()
                if pname and pname not in {"PY", "PYTEST", "MAIN", "UNITTE"}:
                    target_kwargs["prefix"] = f"{pname}_OR"
            except Exception:
                pass

    def _round_provider_qty(self, qty: float) -> float:
        return round_provider_qty(self._provider, self.symbol, qty)

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
        profit_reference_mode, caller_module
    ) -> dict:
        pg_kwargs = {"window_ref": profit_window_ref, "regime_context": regime_context, "qty": qty}
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
        scaled_qty = self._round_provider_qty(qty_f * (scale_f / app_f))
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
        caller_owns_retry = _parse_bool_flag(kwargs.pop("caller_owns_retry", False), False)
        outcome_context = kwargs.pop("_outcome_context", None)
        if outcome_context is not None and not isinstance(outcome_context, dict):
            raise TypeError("_outcome_context must be a dict")
        cache_permit = kwargs.pop("cache_permit", None)
        caller_supplied_cache_permit = cache_permit is not None
        retry_requested_price_raw = kwargs.pop("_retry_requested_price", None)
        retry_price_tolerance_raw = kwargs.pop("_retry_price_tolerance", None)
        retry_constraint_supplied = retry_requested_price_raw is not None or retry_price_tolerance_raw is not None
        bypass_guards_raw = kwargs.pop("bypass_guards", None)
        caller_profit_margin = kwargs.pop("profit_margin", None)
        profit_reference_mode = kwargs.pop("reference_mode", None)
        caller_module = kwargs.pop("module", None)
        allow_math_reference_sell = _parse_bool_flag(kwargs.pop("allow_math_reference_sell", False), False)
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
            return _refuse_early("invalid_side")

        medium_trend_action_cfg = kwargs.get("medium_trend_action")
        medium_trend_scale_cfg = kwargs.get("medium_trend_scale")
        inst_bypass_raw = None
        if hasattr(self, "params") and isinstance(self.params, dict):
            inst_bypass_raw = self.params.get("bypass_guards")
            if profit_reference_mode is None:
                profit_reference_mode = self.params.get(
                    f"{side_u.lower()}_reference_mode", self.params.get("reference_mode")
                )
            caller_module = caller_module if caller_module is not None else self.params.get("module")
            if medium_trend_action_cfg is None:
                medium_trend_action_cfg = self.params.get("medium_trend_action")
            if medium_trend_scale_cfg is None:
                medium_trend_scale_cfg = self.params.get("medium_trend_scale")
        caller_module = (
            order_guard.resolve_caller_module(
                None, prefix=kwargs.get("prefix"), client_order_id=kwargs.get("client_order_id")
            )
            if caller_module is None
            else order_guard.resolve_caller_module(caller_module)
        )
        v_norm = normalize_venue_name(getattr(self._provider, "name", "") or self.provider_name)
        bypassed_guards = order_guard.resolve_effective_bypass_guards(
            v_norm, bypass_guards=bypass_guards_raw, inst_bypass=inst_bypass_raw,
        )
        bypass_all = "all" in bypassed_guards
        bypass_profit_guard = "profit_guard" in bypassed_guards or bypass_all
        bypass_math_ref = GUARD_REGISTRY["G_PROF"].is_bypassed(bypassed_guards) or bypass_profit_guard
        bypass_quantity_policy = GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(bypassed_guards) or bypass_profit_guard
        wait_for_trend = _parse_bool_flag(kwargs.pop("wait_for_trend", True), True)
        kwargs.pop("cooldown_pair_id", None)
        abort_if = kwargs.pop("abort_if", None)

        if bypass_math_ref and not bypass_profit_guard and side_u != "BUY" and not allow_math_reference_sell:
            print(f"[GUARD] {side_u} {self.symbol}: bypass math_profit/math_reference ignored; is allowed only for BUY")
            revoke_guard_bypass(bypassed_guards, "G_PROF")
            bypass_math_ref = False
        if (
            bypass_quantity_policy
            and not bypass_profit_guard
            and side_u != "SELL"
            and caller_module != "assetguardian"
        ):
            print(f"[GUARD] {side_u} {self.symbol}: bypass_quantity_policy ignored; is allowed only for SELL")
            revoke_guard_bypass(bypassed_guards, "G_QLIMIT1")
            bypass_quantity_policy = False
        requires_permit = bool(getattr(self._provider, "requires_cache_permit", lambda: False)())
        defers_smart_cancel = bool(getattr(self._provider, "defers_smart_cancellation", lambda: False)())
        supports_biz_min = bool(getattr(self._provider, "supports_business_minimum", lambda: False)())
        if cache_permit is not None and not requires_permit:
            print(f"[{self.symbol}] {side_u} BLOCKED (pre-submit): invalid_cache_permit_provider")
            return _refuse_early("invalid_cache_permit_provider")
        execution_enabled = getattr(self._provider, "execution_enabled", None)
        try:
            provider_execution_enabled = True if not callable(execution_enabled) else bool(execution_enabled())
        except Exception as exc:  # noqa: BLE001 - An unreadable gate must fail closed.
            print(f"[{self.symbol}] {side_u} BLOCKED (pre-submit): execution_gate_unavailable ({exc})")
            return _refuse_early("execution_gate_unavailable")
        if not provider_execution_enabled:
            print(f"[{self.symbol}] {side_u} DRY: real provider execution is disabled")
            return _refuse_early("execution_disabled")
        if self._provider.guards_internally():
            if retry_constraint_supplied:
                return _refuse_early("retry_price_constraint_unsupported")
            return self._provider.place_order(self.symbol, side_u, price, qty, **kwargs)

        orig_qty, orig_price = qty, price
        reason = order = prequeued_record_id = prequeued_claim = None
        prequeued_client_order_id = prequeued_intent_id = guarded_order_state = None
        submit_attempted = False
        safeback_override = kwargs.get("safeback_seconds")
        smart = _parse_bool_flag(kwargs.pop("smart", True), True)
        regime_context = kwargs.pop("regime_context", None)
        cas_f = safe_pos_float(kwargs.pop("applied_scale", None), default=None)
        caller_applied_scale = cas_f if (cas_f is not None and cas_f < 1.0) else 1.0
        applied_scale = caller_applied_scale
        is_monitor_replace = "monitor_order_replace" in str(order_kind or "")
        profit_margin = safe_float(caller_profit_margin, default=None)

        retry_kwargs = {
            **kwargs,
            "bypass_guards": sorted(list(bypassed_guards)),
            "wait_for_trend": wait_for_trend,
            "smart": smart,
        }
        if caller_applied_scale < 1.0:
            retry_kwargs["applied_scale"] = caller_applied_scale
        for rk, rv in (
            ("profit_margin", profit_margin),
            ("reference_mode", profit_reference_mode),
            ("module", caller_module),
            ("medium_trend_action", medium_trend_action_cfg),
            ("medium_trend_scale", medium_trend_scale_cfg),
        ):
            if rv is not None:
                retry_kwargs[rk] = rv
        if allow_math_reference_sell:
            retry_kwargs["allow_math_reference_sell"] = allow_math_reference_sell

        is_market = resolve_market_flag(**kwargs)
        caller_enforce_biz_min = kwargs.pop("enforce_business_minimum", None)
        if caller_enforce_biz_min is not None:
            enforce_business_minimum = _parse_bool_flag(caller_enforce_biz_min, True)
            retry_kwargs["enforce_business_minimum"] = enforce_business_minimum
        else:
            enforce_business_minimum = not (
                (bypass_profit_guard and side_u == "SELL" and is_market)
                or (caller_module == "assetguardian" and side_u == "SELL")
            )
        profit_window_ref = retry_requested_price = retry_price_tolerance = None
        try:
            if retry_constraint_supplied:
                retry_requested_price = safe_pos_float(retry_requested_price_raw, default=None)
                retry_price_tolerance = safe_nonneg_float(retry_price_tolerance_raw, default=None)
                if (
                    retry_requested_price is None
                    or retry_price_tolerance is None
                    or retry_price_tolerance >= 1.0
                ):
                    raise SubmissionRefused("invalid_retry_price_constraint")
            preflight_order = getattr(self._provider, "preflight_order", None)
            if requires_permit and smart and caller_supplied_cache_permit:
                raise SubmissionRefused("external_cache_permit_not_allowed_for_smart_cancel")
            if requires_permit and cache_permit is None:
                prepare_order_state = getattr(self._provider, "prepare_order_state", None)
                if not callable(prepare_order_state):
                    raise SubmissionRefused("account_cache_not_fresh")
                guarded_order_state = prepare_order_state()
                if guarded_order_state is None:
                    raise SubmissionRefused("account_cache_not_fresh")
            if not requires_permit and callable(preflight_order):
                preflight_qty = None if v_norm in {"kraken", "hyperliquid"} else qty
                preflight_order(
                    self.symbol, side_u, preflight_qty, price, market=is_market, kind=order_kind
                )

            # 0. Compute smart venue pricing before the guards.
            if smart and not isinstance(price, bool):
                price = self._provider.adjust_order_price(
                    self.symbol, side_u, price, cancel_opposite=not defers_smart_cancel
                )
            if not is_market and price is not None and not isinstance(price, bool):
                round_px = getattr(self._provider, "round_price", None)
                if callable(round_px):
                    px_f = safe_pos_float(round_px(self.symbol, price), default=None)
                    if px_f is not None and not (caller_supplied_cache_permit and side_u == "BUY" and px_f > price):
                        price = px_f

            # 1. Provider-agnostic daily trade cap (30 orders / 24h per symbol and side).
            if not GUARD_REGISTRY["GO1"].is_bypassed(bypassed_guards):
                ok, reason = order_guard.daily_limit_guard(self._provider, self.symbol, side_u)
                if not ok:
                    return None

            # Stage 0: Vendor Pre-flight & Unified Account Balance Check
            if isinstance(price, bool) or (price is not None and (side_u == "BUY" or not is_market)):
                price_f = safe_pos_float(price, default=None)
                if price_f is None:
                    print(f"[{self.symbol}] {side_u} preflight refused: price invalid {price!r}")
                    reason = "invalid_price"
                    return None
                price = price_f

            base_asset, quote_asset = resolve_assets(self.symbol, self.base, self.quote)
            target_asset = quote_asset if side_u == "BUY" else base_asset

            known_balance = None
            if hasattr(self._provider, "free_balance") and callable(self._provider.free_balance) and target_asset:
                try:
                    kb_f = safe_float(self._provider.free_balance(target_asset), default=None)
                    if kb_f is not None:
                        if kb_f <= 0:
                            print(f"[{self.symbol}] {side_u} preflight refused: balance_unavailable asset={target_asset}")
                            reason = "balance_unavailable"
                            return None
                        known_balance = kb_f
                except Exception:
                    known_balance = None

            quantity_price = price
            if is_market:
                quantity_price = safe_pos_float(self._provider.get_current_price(self.symbol), default=None)
                if quantity_price is None:
                    print(f"[{self.symbol}] {side_u} MARKET BLOCKED: current price unavailable")
                    reason = "market_price_unavailable"
                    return None

            profit_window_ref = None
            if not bypass_profit_guard:
                try:
                    eff_ref_mode = order_guard.get_profit_guard_mode_cfg(
                        self._provider.name, side_u, module=caller_module, reference_mode=profit_reference_mode,
                    )
                except Exception:
                    eff_ref_mode = None
                if regime_context is None:
                    regime_context = self._resolve_regime_context()
                if profit_margin is None:
                    profit_margin = order_guard.get_margin_cfg(self._provider.name)
                if not bypass_math_ref and eff_ref_mode != "off":
                    profit_window_ref = self._call_profit_guard_window_ref(
                        self.symbol, side_u, safeback_override,
                        regime_context=regime_context, reference_mode=profit_reference_mode, module=caller_module,
                    )

                # Stage 1: Deterministic Mathematical Reference Check (G_PROF).
                if bypass_math_ref:
                    print(
                        f"[GUARD] {side_u} {self.symbol}: the historical price reference "
                        "was explicitly bypassed; the quantity/weight guard remains active")
                elif (
                    not bypass_all
                    and not hasattr(order_guard.profit_guard, "_mock_name")
                    and getattr(order_guard.profit_guard, "__name__", "") == "profit_guard"
                ):
                    guard_px_stage1 = quantity_price if is_market else price
                    cm_kwargs = self._build_profit_guard_kwargs(
                        profit_window_ref, regime_context, None, None, profit_reference_mode, caller_module,
                    )
                    cm_kwargs.pop("qty", None)
                    if not order_guard.check_math_profit_reference(
                        self._provider, self.symbol, side_u, guard_px_stage1, profit_margin, **cm_kwargs
                    ):
                        reason = "profit_guard"
                        return None

            early_abort_if = getattr(abort_if, "is_aborted", None)
            if _is_aborted(early_abort_if):
                reason = "aborted_by_caller"
                return None

            # Stage 2: Quantity Sizing, Gaussian/Chop Weights & Balance Capping.
            hours_f = safe_pos_float(kwargs.get("hours", 5), default=5.0)
            decision = self._provider.quantity_decision(
                self.symbol, side_u, quantity_price, qty,
                base=base_asset, quote=quote_asset,
                cancelorders=_parse_bool_flag(kwargs.get("cancelorders", False), False),
                hours=hours_f,
                apply_policy=not bypass_quantity_policy,
                market=is_market,
                enforce_business_minimum=enforce_business_minimum,
                regime_context=(None if (caller_applied_scale < 1.0 or bypass_profit_guard or is_monitor_replace) else regime_context),
                known_balance=known_balance,
            )
            qty = decision.final_qty
            raw_dec_scale = getattr(decision, "applied_scale", 1.0) if getattr(decision, "scale_applied", False) else 1.0
            dec_scale = safe_pos_float(raw_dec_scale, default=1.0)
            applied_scale = min(float(caller_applied_scale), dec_scale)
            if qty <= 0:
                print(f"[{self.symbol}] {side_u} qty refused: "
                      f"{decision.refuse_reason} asset={decision.balance_asset}")
                reason = decision.refuse_reason or "qty_zero_after_weight"
                return None

            if (
                not bypass_profit_guard
                and not is_monitor_replace
                and regime_context is not None
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

            if _is_aborted(early_abort_if):
                reason = "aborted_by_caller"
                return None

            # 2. Optional provider-agnostic short trend gate (GT_S).
            if wait_for_trend:
                try:
                    import cacheManager as cm
                    import pricewindow as pw
                    if pw.should_wait_short_trend(
                        cm.get_short_trend_manager(), side_u, self.symbol,
                        short_trend_cls=cm.CacheShortTrendManager
                    ):
                        retry_note = "caller owns retry" if caller_owns_retry else "the intent stays queued for retry"
                        print(f"[{self.symbol}] {side_u} deferred by trend; placement does not wait, {retry_note}")
                        reason = "trend_deferred"
                        return None
                except Exception as e:  # noqa: BLE001 — Opportunistic gate.
                    print(f"[{self.symbol}] {side_u} trend gate unavailable: {e}")

            # 2b. Medium-Term Trend Gate (GT_M / medium_trend, 4h - 24h).
            check_medium_trend = not GUARD_REGISTRY["GT_M"].is_bypassed(bypassed_guards)
            medium_trend_action = (
                medium_trend_action_cfg.strip().lower()
                if isinstance(medium_trend_action_cfg, str)
                else None
            )
            gt2_context = regime_context
            if check_medium_trend and gt2_context is None and medium_trend_action in {"defer", "scale"}:
                gt2_context = self._resolve_regime_context()
                if gt2_context is not None and regime_context is None:
                    regime_context = gt2_context
            if check_medium_trend and gt2_context is not None:
                raw_reg = (
                    (gt2_context.get("regime") or gt2_context.get("resolved_trend"))
                    if isinstance(gt2_context, dict)
                    else (getattr(gt2_context, "regime", None) or getattr(gt2_context, "resolved_trend", None))
                )
                resolved_regime = raw_reg.strip().lower() if isinstance(raw_reg, str) else None
                adverse_medium_trend = (
                    (side_u == "BUY" and resolved_regime == "bear")
                    or (side_u == "SELL" and resolved_regime == "bull")
                )
                if medium_trend_action == "defer" and adverse_medium_trend:
                    regime_label = "BEAR regime" if side_u == "BUY" else "BULL regime, riding trend"
                    print(f"[{self.symbol}] {side_u} deferred by medium-term trend (4h-24h {regime_label})")
                    reason = "medium_trend_deferred"
                    return None
                elif medium_trend_action == "scale" and adverse_medium_trend and not is_monitor_replace:
                    scale_factor = _parse_scale_factor(medium_trend_scale_cfg, default=0.5)
                    if scale_factor <= 0.0:
                        reason = "medium_trend_qty_zero"
                        return None
                    if scale_factor < float(applied_scale):
                        qty = self._round_provider_qty(max(0.0, qty * (scale_factor / float(applied_scale))))
                        print(f"[{self.symbol}] {side_u} scaled by medium-term trend ({scale_factor*100:.0f}%, new qty={qty})")
                        if qty <= 0:
                            reason = "medium_trend_qty_zero"
                            return None
                        gt2_refusal = check_order_filter_refusal(
                            self._provider, self.symbol, side_u,
                            quantity_price if is_market else price, qty,
                            market=is_market, enforce_business_minimum=enforce_business_minimum,
                        )
                        if gt2_refusal:
                            print(f"[{self.symbol}] {side_u} medium-trend scaled qty {qty} refused by filter: {gt2_refusal}")
                            reason = gt2_refusal
                            return None
                        applied_scale = min(float(applied_scale), scale_factor)

            if _is_aborted(early_abort_if):
                reason = "aborted_by_caller"
                return None

            # Stage 3: Post-sizing Profit & Intelligence Guards.
            if not bypass_profit_guard and qty > 0:
                guard_price = quantity_price if is_market else price
                pg_kwargs = self._build_profit_guard_kwargs(
                    profit_window_ref, regime_context, qty, bypassed_guards,
                    profit_reference_mode, caller_module,
                )
                if not order_guard.profit_guard(
                    self._provider, self.symbol, side_u, guard_price, profit_margin, **pg_kwargs
                ):
                    reason = "profit_guard"
                    return None
                if not is_monitor_replace:
                    qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                        qty, guard_price, is_market, enforce_business_minimum,
                        regime_context, applied_scale, side_u=side_u)
                    if refusal:
                        reason = refusal
                        return None

            if _is_aborted(abort_if):
                reason = "aborted_by_caller"
                return None
            if callable(execution_enabled) and not bool(execution_enabled()):
                raise SubmissionRefused("execution_disabled")

            if not caller_owns_retry:
                try:
                    if order_retry.RETRY_ENABLED:
                        prequeue_kwargs = {**retry_kwargs, "smart": smart}
                        if 0.0 < float(applied_scale) < 1.0:
                            prequeue_kwargs["applied_scale"] = float(applied_scale)
                        self._apply_outbox_lineage_prefix(prequeue_kwargs, kwargs)
                        prequeued_claim = order_retry.enqueue_claimed(
                            self.symbol, side_u, qty, prequeue_kwargs,
                            requested_price=orig_price, ref_price=price,
                            failure_reason="submit_pending",
                            provider_name=self.provider_name,
                            intent_id=kwargs.get("intent_id"),
                            kind=order_kind, ttl_sec=kwargs.get("ttl_sec"))
                        if prequeued_claim is None:
                            raise RuntimeError("the intent cannot be read back after persisting")
                        prequeued_client_order_id = dict(
                            prequeued_claim.get("place_kwargs") or {}).get("client_order_id")
                        if not prequeued_client_order_id:
                            raise RuntimeError("the persisted intent has no client_order_id")
                        kwargs["client_order_id"] = prequeued_client_order_id
                except Exception as exc:
                    print(
                        f"[{self.symbol}] {side_u} BLOCKED: the intent cannot be "
                        f"persisted before the submit ({exc})")
                    reason = "pre_submit_persist_failed"
                    return None
            else:
                prequeued_claim = kwargs.get("_retry_claim")

            if prequeued_claim is not None:
                prequeued_record_id = prequeued_claim.get("id")
                prequeued_client_order_id = dict(
                    prequeued_claim.get("place_kwargs") or {}).get("client_order_id")
                prequeued_intent_id = prequeued_claim.get("intent_id")

            if prequeued_intent_id and not caller_owns_retry:
                _EXECUTION_AUDIT.record(
                    "submit_requested", intent_id=prequeued_intent_id,
                    venue=self.provider_name, symbol=self.symbol,
                    side=side_u.lower(), qty=qty, price=price,
                    market=is_market, kind=order_kind,
                    client_order_id=prequeued_client_order_id,
                )

            permit_requested_price = None if is_market else orig_price
            if requires_permit and cache_permit is None:
                if not callable(preflight_order):
                    raise SubmissionRefused("account_cache_not_fresh")
                cache_permit = preflight_order(
                    self.symbol, side_u, qty, price, market=is_market, kind=order_kind)
                if cache_permit is None:
                    raise SubmissionRefused("account_cache_not_fresh")
                permit_requested_price = None if is_market else price
                validate_order_state = getattr(self._provider, "validate_order_state", None)
                if not callable(validate_order_state):
                    raise SubmissionRefused("account_cache_not_fresh")
                validate_order_state(guarded_order_state)

            final_profit_check = not bypass_profit_guard and is_market
            if is_market and (final_profit_check or retry_constraint_supplied):
                final_market_price = safe_pos_float(self._provider.get_current_price(self.symbol), default=None)
                if final_market_price is None:
                    reason = "market_price_unavailable"
                    print(f"[{self.symbol}] {side_u} MARKET BLOCKED at dispatch: current price unavailable")
                    return None
                if retry_constraint_supplied and not order_retry.is_retry_price_favorable(
                    side_u, final_market_price, retry_requested_price, retry_price_tolerance
                ):
                    reason = "retry_price_unfavorable"
                    return None
                if final_profit_check:
                    pg_kwargs = self._build_profit_guard_kwargs(
                        profit_window_ref, regime_context, qty, bypassed_guards,
                        profit_reference_mode, caller_module,
                    )
                    if not order_guard.profit_guard(
                        self._provider, self.symbol, side_u, final_market_price, profit_margin, **pg_kwargs
                    ):
                        reason = "profit_guard"
                        return None
                    pre_final_qty = qty
                    qty, applied_scale, refusal = self._apply_late_intelligence_scale(
                        qty, final_market_price, is_market, enforce_business_minimum,
                        regime_context, applied_scale, side_u=side_u)
                    if refusal:
                        reason = refusal
                        return None
                    if (
                        qty != pre_final_qty
                        and requires_permit
                        and not caller_supplied_cache_permit
                        and callable(preflight_order)
                    ):
                        cache_permit = preflight_order(
                            self.symbol, side_u, qty, price, market=is_market, kind=order_kind)
                        if cache_permit is None:
                            raise SubmissionRefused("account_cache_not_fresh")
                    if prequeued_claim is not None:
                        prequeued_claim = self._sync_claimed_qty(prequeued_claim, qty, applied_scale)
                        if prequeued_intent_id and applied_scale < 1.0:
                            _EXECUTION_AUDIT.record(
                                "submit_scaled", intent_id=prequeued_intent_id,
                                venue=self.provider_name, symbol=self.symbol,
                                side=side_u.lower(), qty=qty, scale=applied_scale,
                            )
            if _is_aborted(abort_if):
                reason = "aborted_by_caller"
                return None
            if callable(execution_enabled) and not bool(execution_enabled()):
                raise SubmissionRefused("execution_disabled")
            if prequeued_claim is not None:
                if not caller_owns_retry:
                    refreshed_claim = order_retry.begin_claimed_submit(
                        prequeued_claim, qty=qty, applied_scale=applied_scale)
                    if refreshed_claim is None:
                        raise SubmissionRefused("producer_claim_lost")
                    prequeued_claim = refreshed_claim
                else:
                    prequeued_claim = self._sync_claimed_qty(prequeued_claim, qty, applied_scale)
            reason = "submit_ambiguous"
            submit_attempted = True
            provider_kwargs = {
                **kwargs, "market": is_market, "force": is_market, "_balance_verified": True
            }
            if known_balance is not None:
                provider_kwargs["known_balance"] = known_balance
            if supports_biz_min:
                provider_kwargs["enforce_business_minimum"] = enforce_business_minimum
            if cache_permit is not None:
                provider_kwargs["permit_requested_price"] = permit_requested_price
                provider_kwargs["cache_permit"] = cache_permit
            if smart and defers_smart_cancel:
                provider_kwargs["_cancel_opposite_requested_price"] = orig_price
            response = self._provider.place_order(
                self.symbol, side_u, price, qty, **provider_kwargs)
            if _accepted_order_payload(response):
                order, reason = response, None
            else:
                order, reason = None, "response_without_order_id"
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
            submission_state = (
                "accepted" if order is not None else
                "unknown" if submit_attempted else "refused"
            )
            if outcome_context is not None:
                outcome_context.update(
                    accepted=order is not None,
                    reason=None if order is not None else reason,
                    state=submission_state,
                )
                if order is not None or submit_attempted:
                    outcome_context["submitted_qty"] = qty
                    outcome_context["submitted_price"] = price
                cid = prequeued_client_order_id or kwargs.get("client_order_id")
                if cid:
                    outcome_context["client_order_id"] = cid
                iid = prequeued_intent_id or kwargs.get("intent_id")
                if iid:
                    outcome_context["intent_id"] = iid
            try:
                caller = os.path.basename(sys._getframe(1).f_code.co_filename)
            except Exception:
                caller = None
            terminal_filter_refusal = bool(
                reason and not submit_attempted and _is_terminal_filter_refusal(reason)
            )
            _outcomes_log.log_order_outcome(
                self.symbol, side_u, price, qty, submission_state,
                None if order else reason, order_kind, caller=caller,
                regime_context=regime_context)
            if order is not None and not caller_owns_retry:
                try:
                    if prequeued_claim is not None:
                        tracked = accepted_order_persistence.complete_accepted_claim(
                            order_retry, prequeued_claim, order=order,
                            provider_name=self.provider_name,
                            symbol=self.symbol, side=side_u)
                        if not tracked:
                            print(
                                f"[{self.symbol}] {side_u} accepted, but the tracking "
                                "could not be updated; the intent stays pending")
                        elif prequeued_intent_id:
                            _EXECUTION_AUDIT.record(
                                "submit_accepted", intent_id=prequeued_intent_id,
                                venue=self.provider_name, symbol=self.symbol,
                                side=side_u.lower(), qty=qty, price=price,
                                market=is_market, kind=order_kind,
                                client_order_id=prequeued_client_order_id,
                                order_id=_order_id_from_payload(order),
                            )
                except Exception as _e:  # noqa: BLE001
                    print(f"[{self.symbol}] {side_u} tracking accepted failed: {_e}")
            elif (order is None and not caller_owns_retry
                  and (prequeued_record_id is not None
                       or (reason not in {
                           "account_cache_not_fresh", "execution_disabled",
                           "trading_disabled", "aborted_by_caller", "invalid_price"}
                           and not terminal_filter_refusal))):
                try:
                    if prequeued_intent_id:
                        _EXECUTION_AUDIT.record(
                            "submit_unknown" if submit_attempted else "submit_refused",
                            intent_id=prequeued_intent_id,
                            venue=self.provider_name, symbol=self.symbol,
                            side=side_u.lower(), qty=qty, price=price,
                            market=is_market, kind=order_kind,
                            client_order_id=prequeued_client_order_id,
                            failure_reason=reason,
                        )
                    if order_retry.RETRY_ENABLED:
                        if prequeued_claim is not None:
                            claim_outcome = (
                                "success" if (
                                    terminal_filter_refusal or reason in {
                                        "execution_disabled", "trading_disabled",
                                        "aborted_by_caller", "invalid_price"}) else
                                "deferred" if order_retry.is_non_failure_deferral(reason)
                                else "failure")
                            order_retry.complete_claim(
                                prequeued_claim, claim_outcome,
                                failure_reason=reason,
                                submission_state="unknown" if submit_attempted else "refused")
                        else:
                            q_orig = safe_pos_float(orig_qty, default=None)
                            if q_orig is None:
                                queue_qty = qty
                                app_sc = safe_pos_float(applied_scale, default=1.0) or 1.0
                                if 0.0 < app_sc < 1.0:
                                    retry_kwargs["applied_scale"] = app_sc
                            else:
                                queue_qty = q_orig
                            if safe_pos_float(queue_qty, default=None) is not None:
                                try:
                                    ref_price = self._provider.get_current_price(self.symbol)
                                except Exception:  # noqa: BLE001
                                    ref_price = None
                                self._apply_outbox_lineage_prefix(retry_kwargs, kwargs)
                                order_retry.enqueue(
                                    self.symbol, side_u, queue_qty, retry_kwargs,
                                    requested_price=orig_price, ref_price=ref_price,
                                    failure_reason=reason,
                                    provider_name=self.provider_name,
                                    intent_id=kwargs.get("intent_id"),
                                    kind=order_kind, ttl_sec=kwargs.get("ttl_sec"))
                except Exception as _e:  # noqa: BLE001
                    print(f"[{self.symbol}] {side_u} enqueue retry failed (ignored): {_e}")

    def min_qty(self) -> float:
        """Return the venue minimum order quantity; ``0`` means no known minimum."""
        try:
            return safe_pos_float(self._provider.min_order_qty(self.symbol), default=0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    # ── params namespaced (mt.* / tradeall.* / rtrade.*) ───────────────────────
    def param(self, consumer: str, key: str, default: Any = None,
              cast: Optional[Callable] = None) -> Any:
        """Return ``consumer.key`` or ``default`` when missing or conversion fails."""
        if isinstance(consumer, bool) or isinstance(key, bool):
            return default
        v = casefold_get(self.params, f"{str(consumer or '').strip()}.{str(key or '').strip()}")
        if v is None or cast is None:
            return default if v is None else v
        if cast is bool:
            return _parse_bool_flag(v, default=bool(default))
        if cast is float:
            return safe_float(v, default=default)
        if cast is int:
            return safe_int(v, default=default)
        try:
            return cast(v)
        except (ValueError, TypeError, OverflowError):
            return default

    def __repr__(self) -> str:
        return f"<Instrument {self.name} {self.symbol}@{self.provider_name} {'on' if self.enabled else 'off'}>"
