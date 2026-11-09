"""LLM high-stake guard validating large BUY and SELL orders (e.g. >= 700 EUR/USD)."""
from __future__ import annotations

from abc import ABC
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    is_cache_fresh,
    normalize_side,
    resolve_order_notional,
    resolve_patched_attr,
    safe_nonneg_float,
    safe_nonneg_ts,
)
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    OrderGuardEvaluatorBase,
    OrderGuardEvaluatorFactory,
    normalize_guard_mode,
)
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    clean_symbol_key,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
    PretradeTelemetrySnapshot,
    collect_pretrade_telemetry as _default_collect_pretrade_telemetry,
    read_pretrade_telemetry_snapshot as _default_read_pretrade_telemetry_snapshot,
)
from intelligence.sentiment.gemini_client import (
    GeminiClient,
    query_topic_json as _default_query_topic_json,
)
from state_io import read_json_dict_recovering as read_json_cached

collect_pretrade_telemetry = _default_collect_pretrade_telemetry
read_pretrade_telemetry_snapshot = _default_read_pretrade_telemetry_snapshot
query_topic_json = _default_query_topic_json

logger = logging.getLogger("intelligence.sentiment.high_stake_guard")

DEFAULT_MIN_NOTIONAL_USD = 700.0
DEFAULT_LLM_GUARD_CACHE_TTL_SEC = 600.0
HIGH_STAKE_NOTIONAL_JUMP_RATIO = 1.0
_HIGH_STAKE_DECISION_CACHE: Dict[tuple[str, ...], tuple[float, GuardDecision]] = {}

LLM_GUARD_JSON_SCHEMA_FOOTER = (
    "Respond strictly in valid JSON format with exact keys:\n"
    "{\n"
    '  "decision": "APPROVED" | "REJECTED" | "DOWNSCALE",\n'
    '  "suggested_scale": 0.0 to 1.0,\n'
    '  "reason": "Concise 1-2 sentence rationale for the decision"\n'
    "}"
)


def is_llm_guard_exempt(explicit_client: bool = False) -> bool:
    """True when LLM pre-trade guards are suppressed via environment flags (backtest/replay/disabled)."""
    return not _parse_bool_flag(explicit_client, False) and any(
        _parse_bool_flag(os.environ.get(k)) for k in ("DISABLE_GEMINI_LLM", "BACKTEST_MODE", "REPLAY_MODE")
    )


def evaluate_unavailable_telemetry_decision(
    guard_name: str,
    reason: str = "telemetry_unavailable",
    *,
    fallback_action: str,
    notional_eur: float,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    """Convert unavailable or expired pre-trade telemetry into a GuardDecision respecting configured fallback_action."""
    safe_notional = safe_nonneg_float(notional_eur, 0.0)
    fb_norm = str(fallback_action or "allow").strip().lower()
    base_reason = str(reason or "telemetry_unavailable").strip()
    meta: Dict[str, Any] = {
        **(extra_metadata if isinstance(extra_metadata, dict) else {}),
        "notional_eur": safe_notional,
        "evaluated_notional_eur": safe_notional,
        "telemetry_unavailable_reason": base_reason,
    }
    if fb_norm in ("veto", "block"):
        return GuardDecision.veto(guard_name, f"{base_reason}_fail_closed", **meta)
    if fb_norm == "downscale":
        return GuardDecision.downscale(guard_name, scale=0.5, reason=f"{base_reason}_downscaled", **meta)
    return GuardDecision.allow(guard_name, f"{base_reason}_fallback_allowed", **meta)


def evaluate_llm_guard_response(
    guard_name: str,
    action_label: str,
    resp: Optional[Dict[str, Any]],
    *,
    fallback_action: str,
    notional_eur: float,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    """SSOT converter from a pre-trade LLM JSON response (or timeout None) into a canonical GuardDecision."""
    safe_notional = safe_nonneg_float(notional_eur, 0.0)
    fb_norm = str(fallback_action or "allow").strip().lower()
    meta_extra = dict(extra_metadata) if isinstance(extra_metadata, dict) else {}
    if not resp or not isinstance(resp, dict):
        if fb_norm in ("veto", "block"):
            return GuardDecision.veto(
                guard_name,
                "llm_query_timeout_fail_closed",
                notional_eur=safe_notional,
                evaluated_notional_eur=safe_notional,
                **meta_extra,
            )
        if fb_norm == "downscale":
            return GuardDecision.downscale(
                guard_name,
                scale=0.5,
                reason="llm_query_timeout_downscaled",
                notional_eur=safe_notional,
                evaluated_notional_eur=safe_notional,
                **meta_extra,
            )
        return GuardDecision.allow(
            guard_name,
            "llm_query_timeout_fallback_allowed",
            notional_eur=safe_notional,
            evaluated_notional_eur=safe_notional,
            **meta_extra,
        )

    raw_dec = resp.get("decision")
    decision_str = str(raw_dec if (isinstance(raw_dec, str) and raw_dec.strip()) else "APPROVED").upper().strip()
    raw_reason = resp.get("reason")
    reason = str(raw_reason).strip() if (isinstance(raw_reason, str) and raw_reason.strip()) else "evaluated by LLM"
    scale = clean_scale(resp.get("suggested_scale"), 0.5 if decision_str == "DOWNSCALE" else 1.0, allow_percentage=True)

    if decision_str in ("REJECTED", "REJECT", "VETO", "BLOCK", "DENIED", "DENY"):
        return GuardDecision.veto(
            guard_name,
            f"Gemini vetoed {action_label}: {reason}",
            notional_eur=safe_notional,
            evaluated_notional_eur=safe_notional,
            gemini_reason=reason,
            **meta_extra,
        )
    if decision_str == "DOWNSCALE":
        eff_scale = 0.5 if (scale <= 0.0 or scale >= 1.0) else max(0.1, min(1.0, scale))
        return GuardDecision.downscale(
            guard_name,
            scale=eff_scale,
            reason=f"Gemini downscaled {action_label}: {reason}",
            notional_eur=safe_notional,
            evaluated_notional_eur=safe_notional,
            gemini_reason=reason,
            **meta_extra,
        )
    return GuardDecision.allow(
        guard_name,
        f"Gemini approved {action_label}: {reason}",
        notional_eur=safe_notional,
        evaluated_notional_eur=safe_notional,
        gemini_reason=reason,
        **meta_extra,
    )


def resolve_llm_guard_cache_ttl(
    cache_ttl_sec: Any, explicit_client: bool, gemini_client: Any, default_ttl_sec: float
) -> float:
    """Resolve effective LLM guard decision cache TTL (0.0 when custom client/runner is injected)."""
    if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
        return safe_nonneg_ts(cache_ttl_sec, 0.0)
    return default_ttl_sec if (not explicit_client and getattr(gemini_client, "_custom_runner", None) is None) else 0.0


class PreTradeLLMGuardBase(OrderGuardEvaluatorBase, ABC):
    """Unified base class (Level 2) for real-time pre-trade LLM guards (HighStakeGuard & PostStopReentryGuard)."""

    cls_name = "PreTradeLLMGuardBase"
    GUARD_NAME: str = "PreTradeLLMGuard"
    TOPIC_KEY: str = "guard_high_stake"
    DEFAULT_TIMEOUT_SEC: float = 120.0
    DEFAULT_FALLBACK: str = "allow"
    DEFAULT_CACHE_TTL_SEC: float = DEFAULT_LLM_GUARD_CACHE_TTL_SEC

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        timeout_sec: float = 120.0,
        fallback_action: str = "allow",
        mode: str = "enforce",
        notify_enabled: bool = True,
        cache_ttl_sec: Optional[float] = None,
        max_telemetry_age_sec: Optional[float] = None,
    ) -> None:
        if type(self) is PreTradeLLMGuardBase:
            raise TypeError("Cannot instantiate abstract class PreTradeLLMGuardBase")
        super().__init__()
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        self.timeout_sec = safe_pos_float(timeout_sec, self.DEFAULT_TIMEOUT_SEC)
        self.fallback_action = str(fallback_action or self.DEFAULT_FALLBACK).strip().lower()
        self.mode = normalize_guard_mode(mode, default="enforce")
        self.notify_enabled = _parse_bool_flag(notify_enabled)
        self.cache_ttl_sec = resolve_llm_guard_cache_ttl(
            cache_ttl_sec, self._explicit_client, self.gemini_client, self.DEFAULT_CACHE_TTL_SEC
        )
        self.max_telemetry_age_sec: Optional[float] = (
            safe_pos_float(max_telemetry_age_sec, default=None)
            if max_telemetry_age_sec is not None and not isinstance(max_telemetry_age_sec, bool)
            else None
        )

    def _wrap_shadow_decision(
        self,
        dec: GuardDecision,
        notional: float,
        valid_resp: Optional[Dict[str, Any]] = None,
        *,
        evaluated_notional_eur: Optional[float] = None,
        cache_hit: bool = False,
    ) -> GuardDecision:
        if isinstance(getattr(dec, "shadow_decision", None), GuardDecision):
            return dec
        if valid_resp is not None and isinstance(valid_resp, dict):
            raw_dec, raw_reason = valid_resp.get("decision"), valid_resp.get("reason")
            decision_str = str(raw_dec if (isinstance(raw_dec, str) and raw_dec.strip()) else "APPROVED").upper().strip()
            reason = str(raw_reason).strip() if (isinstance(raw_reason, str) and raw_reason.strip()) else "evaluated by LLM"
            shadow_msg = f"shadow_mode (LLM suggested {decision_str}: {reason})"
        else:
            shadow_msg = f"shadow_mode ({dec.reason})"
        dec_meta = getattr(dec, "metadata", None)
        extra_meta: Dict[str, Any] = {
            **(dec_meta if isinstance(dec_meta, dict) else {}),
            "notional_eur": notional,
            "evaluated_notional_eur": evaluated_notional_eur if evaluated_notional_eur is not None else notional,
        }
        if cache_hit:
            extra_meta["cache_hit"] = True
        return GuardDecision.allow(self.GUARD_NAME, shadow_msg, shadow_decision=dec, **extra_meta)

    def _resolve_pretrade_telemetry_snapshot(
        self,
        symbol: str,
        *,
        snapshot: Optional[PretradeTelemetrySnapshot] = None,
        telemetry: Optional[Any] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        now_ts: float,
        max_telemetry_age_sec: Optional[float] = None,
    ) -> PretradeTelemetrySnapshot:
        """Resolve an explicit read-only ``PretradeTelemetrySnapshot`` without triggering network or cache writes."""
        eff_max_age = (
            safe_pos_float(max_telemetry_age_sec, default=None)
            if max_telemetry_age_sec is not None and not isinstance(max_telemetry_age_sec, bool)
            else self.max_telemetry_age_sec
        )
        enforce_fresh = eff_max_age is not None
        if snapshot is not None:
            return PretradeTelemetrySnapshot.from_raw(
                snapshot,
                symbol=symbol,
                now_ts=now_ts,
                source="explicit",
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )
        if telemetry is not None:
            return PretradeTelemetrySnapshot.from_raw(
                telemetry,
                symbol=symbol,
                now_ts=now_ts,
                source="explicit",
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )

        target_modules = (
            self.__class__.__module__,
            __name__,
            "intelligence.sentiment.collectors.macro_telemetry_collector",
            "intelligence.sentiment.collectors.pretrade_telemetry_collector",
            "intelligence.macro_sentiment_ssot",
        )
        _telem_fn = resolve_patched_attr(
            "collect_pretrade_telemetry",
            collect_pretrade_telemetry,
            defaults=(_default_collect_pretrade_telemetry,),
            modules=target_modules,
            require_callable=True,
        )
        _snap_fn = resolve_patched_attr(
            "read_pretrade_telemetry_snapshot",
            read_pretrade_telemetry_snapshot,
            defaults=(_default_read_pretrade_telemetry_snapshot,),
            modules=target_modules,
            require_callable=True,
        )
        try:
            if _telem_fn is not _default_collect_pretrade_telemetry:
                call_kwargs: Dict[str, Any] = {"regime_context": regime_context}
                if dur_sec is not None:
                    call_kwargs["dur_sec"] = dur_sec
                try:
                    raw_telem = _telem_fn(symbol, **call_kwargs, allow_network=False, now_ts=now_ts)
                except TypeError:
                    raw_telem = _telem_fn(symbol, **call_kwargs)
                if isinstance(raw_telem, PretradeTelemetrySnapshot):
                    return PretradeTelemetrySnapshot.from_raw(
                        raw_telem,
                        symbol=symbol,
                        now_ts=now_ts,
                        max_age_sec=eff_max_age,
                        enforce_freshness=enforce_fresh,
                    )
                return PretradeTelemetrySnapshot.from_raw(
                    raw_telem,
                    symbol=symbol,
                    now_ts=now_ts,
                    source="collector",
                    max_age_sec=eff_max_age,
                    enforce_freshness=enforce_fresh,
                )

            resolved = _snap_fn(
                symbol,
                regime_context=regime_context,
                dur_sec=dur_sec,
                now_ts=now_ts,
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )
            if isinstance(resolved, PretradeTelemetrySnapshot):
                return resolved
            return PretradeTelemetrySnapshot.from_raw(
                resolved,
                symbol=symbol,
                now_ts=now_ts,
                source="cachedb",
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )
        except Exception as exc:
            logger.warning("Pre-trade telemetry read failed for %s: %s", symbol, exc)
            return PretradeTelemetrySnapshot.unavailable(
                symbol,
                reason="telemetry_unavailable",
                now_ts=now_ts,
                max_age_sec=eff_max_age,
            )

    def _execute_pretrade_guard_eval(
        self,
        *,
        symbol: str,
        notional: float,
        cache_key: Any,
        decision_cache: Dict[Any, tuple[float, GuardDecision]],
        now_ts: float,
        header_lines: Sequence[str],
        build_prompt_fn: Callable[[str], str],
        subtitle: str,
        action_label: str,
        timeout_log_label: str,
        snapshot: Optional[PretradeTelemetrySnapshot] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        max_telemetry_age_sec: Optional[float] = None,
        max_notional_jump_ratio: Optional[float] = None,
        on_decision: Optional[Callable[[GuardDecision], None]] = None,
        logger_inst: Optional[logging.Logger] = None,
    ) -> GuardDecision:
        """Shared pre-trade LLM guard pipeline: cache lookup -> snapshot read -> topic query -> decision -> shadow wrap -> cache write."""
        if self.cache_ttl_sec > 0:
            cached = decision_cache.get(cache_key)
            if (
                isinstance(cached, (tuple, list))
                and len(cached) >= 2
                and isinstance(cached[1], GuardDecision)
                and is_cache_fresh(cached[0], now_ts=now_ts, ttl_sec=self.cache_ttl_sec)
            ):
                cached_dec: GuardDecision = cached[1]
                base_dec = (
                    cached_dec.shadow_decision
                    if isinstance(getattr(cached_dec, "shadow_decision", None), GuardDecision)
                    else cached_dec
                )
                base_meta = getattr(base_dec, "metadata", None)
                base_meta = base_meta if isinstance(base_meta, dict) else {}
                eval_notional = safe_pos_float(
                    base_meta.get("evaluated_notional_eur") or base_meta.get("notional_eur"),
                    default=None,
                )
                notional_jumped = (
                    max_notional_jump_ratio is not None
                    and eval_notional is not None
                    and (notional - eval_notional) / max(eval_notional, 1.0) > max_notional_jump_ratio
                )
                if not notional_jumped:
                    baseline_notional = eval_notional if eval_notional is not None else notional
                    if self.mode == "shadow":
                        return self._wrap_shadow_decision(
                            base_dec,
                            notional,
                            evaluated_notional_eur=baseline_notional,
                            cache_hit=True,
                        )
                    from dataclasses import replace as _dc_replace

                    return _dc_replace(
                        base_dec,
                        metadata={
                            **base_meta,
                            "notional_eur": notional,
                            "evaluated_notional_eur": baseline_notional,
                            "cache_hit": True,
                        },
                    )

        resolved_snapshot = self._resolve_pretrade_telemetry_snapshot(
            symbol,
            snapshot=snapshot,
            telemetry=telemetry,
            regime_context=regime_context,
            dur_sec=dur_sec,
            now_ts=now_ts,
            max_telemetry_age_sec=max_telemetry_age_sec,
        )
        snap_meta = resolved_snapshot.to_guard_metadata()
        if not resolved_snapshot.is_available:
            unavail_reason = resolved_snapshot.unavailable_reason or "telemetry_unavailable"
            (logger_inst or logger).warning(
                "Pre-trade telemetry unavailable for %s €%.2f (%s). Fallback: %s",
                symbol,
                notional,
                unavail_reason,
                self.fallback_action,
            )
            dec = evaluate_unavailable_telemetry_decision(
                self.GUARD_NAME,
                unavail_reason,
                fallback_action=self.fallback_action,
                notional_eur=notional,
                extra_metadata=snap_meta,
            )
            if on_decision is not None:
                on_decision(dec)
            return self._wrap_shadow_decision(dec, notional, None) if self.mode == "shadow" else dec

        context_items = list(header_lines)
        if resolved_snapshot:
            context_items.append(resolved_snapshot.format_identity_header_line())
            context_items.extend(f"- {k}: {v}" for k, v in resolved_snapshot.items())
        context_str = "\n".join(context_items)

        _query_fn = resolve_patched_attr(
            "query_topic_json",
            query_topic_json,
            defaults=(_default_query_topic_json,),
            modules=(
                self.__class__.__module__,
                __name__,
                "intelligence.sentiment.gemini_client",
                "intelligence.macro_sentiment_ssot",
            ),
            require_callable=True,
        )

        prompt = f"{build_prompt_fn(context_str)}\n\n{LLM_GUARD_JSON_SCHEMA_FOOTER}"
        valid_resp = _query_fn(
            self.gemini_client,
            self.TOPIC_KEY,
            prompt,
            subtitle=subtitle,
            timeout_sec=self.timeout_sec,
        )
        if valid_resp is None:
            (logger_inst or logger).warning(
                "LLM %s query failed or timed out for %s €%.2f. Fallback: %s",
                timeout_log_label,
                symbol,
                notional,
                self.fallback_action,
            )

        dec = evaluate_llm_guard_response(
            self.GUARD_NAME,
            action_label,
            valid_resp,
            fallback_action=self.fallback_action,
            notional_eur=notional,
            extra_metadata=snap_meta,
        )
        if on_decision is not None:
            on_decision(dec)

        final_dec = self._wrap_shadow_decision(dec, notional, valid_resp) if self.mode == "shadow" else dec

        if self.cache_ttl_sec > 0:
            decision_cache[cache_key] = (now_ts, final_dec)
        return final_dec


class HighStakeGuard(PreTradeLLMGuardBase):
    """Invokes LLM reasoning before executing large BUY/SELL orders (>= 700 EUR/USD)."""

    cls_name = "HighStakeGuard"
    GUARD_ID = "sentiment_high_stake_llm"
    GUARD_NAME = "HighStakeGuard"
    TOPIC_KEY = "guard_high_stake"
    DEFAULT_CACHE_TTL_SEC = DEFAULT_LLM_GUARD_CACHE_TTL_SEC

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        min_notional_eur: float = DEFAULT_MIN_NOTIONAL_USD,
        min_notional_usd: Optional[float] = None,
        timeout_sec: float = 120.0,
        fallback_action: str = "allow",
        mode: str = "enforce",
        notify_enabled: bool = True,
        cache_ttl_sec: Optional[float] = None,
        max_telemetry_age_sec: Optional[float] = None,
    ) -> None:
        super().__init__(
            gemini_client=gemini_client,
            timeout_sec=timeout_sec,
            fallback_action=fallback_action,
            mode=mode,
            notify_enabled=notify_enabled,
            cache_ttl_sec=cache_ttl_sec,
            max_telemetry_age_sec=max_telemetry_age_sec,
        )
        resolved_min = safe_pos_float(min_notional_usd if min_notional_usd is not None else min_notional_eur, DEFAULT_MIN_NOTIONAL_USD)
        self.min_notional_usd = self.min_notional_eur = resolved_min

    def _build_high_stake_prompt(self, side_norm: str, notional: float, context_str: str) -> str:
        if side_norm == "BUY":
            action_desc = "HIGH-VALUE BUY order"
            risk_context = (
                "evaluating whether this entry is sound or if there are critical red flags "
                "(e.g., euphoric top buying, aggressive whale distribution, severe funding imbalance, or active cascade)."
            )
            policy_rules = (
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions (% Top Traders Long) and Orderbook Bid Depth. Static long holdings can be trapped or liquidated. If Whale Taker Flow is below 0.85 (net sell aggression), you must NEVER issue an unrestricted APPROVED. If Whale Taker Flow is below 0.80 in a FLAT or BEAR market, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural support).\n"
                "2. ORDERBOOK SUPPORT LIMITATION: Bid walls and orderbook depth are passive and can be pulled or absorbed by aggressive market selling. Do NOT use orderbook depth alone to override persistent taker selling.\n"
                "3. MACRO, GEOPOLITICAL & CONTRARIAN BOTTOM ACCUMULATION: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow. However, when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM' (especially with 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE') AND Whale Taker Flow is >= 1.0 (or 15m CVD is positive/spot_absorbing), issue APPROVED — do NOT veto or downscale a verified contrarian bottom entry merely because lagging 24h breadth or residual geopolitical headlines remain elevated.\n"
                "4. CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW SYMMETRY (Equities <-> Crypto): Synthesize 'Equity & Macro Regime' across both Stocks and Crypto: weigh SPY/QQQ/VIX/DXY/US10Y and Futures_Lead (FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN) alongside IBIT (US ETF flow), CB_Premium (Coinbase vs Binance institutional spot basis bps), BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options(VRP, PC_OI, Skew_25d), and Cross_Asset (COUPLED_RISK_ON | COUPLED_RISK_OFF | CRYPTO_BULLISH_DECOUPLING | EQUITY_BULLISH_DECOUPLING). When Cross_Asset is COUPLED_RISK_OFF with surging VIX, deeply negative CB_Premium (<= -8 bps), or negative Skew_25d (put-buying fear), require strict Taker/CVD absorption before approving large BUY entries; conversely, when Cross_Asset is CRYPTO_BULLISH_DECOUPLING or COUPLED_RISK_ON with positive CB_Premium / IBIT inflow or FUTURES_PREMARKET_GAP_UP, do NOT veto a crypto BUY solely due to soft prior-day equity indices."
            )
        else:
            action_desc = "HIGH-VALUE SELL order"
            risk_context = (
                "evaluating whether this sale is sound or if there are critical red flags "
                "(e.g., dumping directly into major whale support/walls, panic-selling at local bottom during whale accumulation, or excessive slippage)."
            )
            policy_rules = (
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions and Orderbook Ask Depth. Static short holdings can be squeezed or liquidated. If Whale Taker Flow is above 1.18 (net buy aggression), you must NEVER issue an unrestricted APPROVED for a non-stop-loss SELL into strength. If Whale Taker Flow is above 1.25 in a FLAT or BULL market during whale accumulation or short squeeze, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural resistance).\n"
                "2. ORDERBOOK RESISTANCE LIMITATION: Ask walls and orderbook ask depth are passive and can be pulled or absorbed by aggressive market buying. Do NOT use orderbook ask depth alone to override persistent taker buying.\n"
                "3. CAPITULATION & SQUEEZE PROTECTION (6H DOUBLE-BOTTOM & RECLAIM DEFENSE): Do not approve panic-selling into extreme negative funding, active whale accumulation, or a verified 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'FRESH_6H_LOW_TEST') at local bottoms.\n"
                "4. CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW SYMMETRY (Equities <-> Crypto): Synthesize 'Equity & Macro Regime' across both Stocks and Crypto: weigh SPY/QQQ/VIX/DXY/US10Y and Futures_Lead (FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN) alongside IBIT, CB_Premium, BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options(VRP, PC_OI, Skew_25d), and Cross_Asset (COUPLED_RISK_ON | COUPLED_RISK_OFF | CRYPTO_BULLISH_DECOUPLING | EQUITY_BULLISH_DECOUPLING). Do NOT approve non-stop-loss panic SELLs at local lows when Cross_Asset is CRYPTO_BULLISH_DECOUPLING or COUPLED_RISK_ON with positive CB_Premium (>= +5 bps), IBIT accumulation, or FUTURES_PREMARKET_GAP_UP; conversely, approve defensive profit-taking or risk-reduction SELLs when Cross_Asset is COUPLED_RISK_OFF or EQUITY_BULLISH_DECOUPLING with negative CB_Premium, FUTURES_PREMARKET_GAP_DOWN, and rising VIX/BTC_DVOL."
            )
        return (
            "You are a principal quantitative crypto risk officer. "
            f"An automated trading bot is about to place a {action_desc}:\n\n{context_str}\n\n"
            f"Because this order exceeds €{self.min_notional_eur:,.2f} EUR, you must evaluate if executing {risk_context}\n\n"
            f"{policy_rules}"
        )

    def check(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        notional_eur: Optional[float] = None,
        snapshot: Optional[PretradeTelemetrySnapshot] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        max_telemetry_age_sec: Optional[float] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order through LLM if value exceeds threshold."""
        side_norm = normalize_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow("HighStakeGuard", "unsupported_side_exempt")

        notional = resolve_order_notional(notional_eur, price, qty, default=0.0, fallback_on_nonfinite=False) or 0.0
        notional_f = safe_float(notional, default=None)
        if notional_f is None or notional_f < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard", f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)", notional_eur=notional
            )

        if self.mode in ("off", "0", "0.0", "disabled", "false", "no"):
            return GuardDecision.allow("HighStakeGuard", "high_stake_guard_disabled", notional_eur=notional)
        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("HighStakeGuard", "backtest_mode_exempt", notional_eur=notional)

        now_ts = resolve_now_ts(now)
        cache_key = (clean_symbol_key(symbol), side_norm)
        header_lines = (
            f"- Symbol: {symbol}",
            f"- Side: {side_norm}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        )
        return self._execute_pretrade_guard_eval(
            symbol=symbol,
            notional=notional,
            cache_key=cache_key,
            decision_cache=_HIGH_STAKE_DECISION_CACHE,
            now_ts=now_ts,
            header_lines=header_lines,
            build_prompt_fn=lambda ctx: self._build_high_stake_prompt(side_norm, notional, ctx),
            subtitle=f"{side_norm} {symbol} €{notional:,.2f}",
            action_label=f"high-stake {side_norm}",
            timeout_log_label="high-stake",
            snapshot=snapshot,
            telemetry=telemetry,
            regime_context=regime_context,
            dur_sec=dur_sec,
            max_telemetry_age_sec=max_telemetry_age_sec,
            max_notional_jump_ratio=HIGH_STAKE_NOTIONAL_JUMP_RATIO,
            logger_inst=logger,
        )


OrderGuardEvaluatorFactory.register(HighStakeGuard.GUARD_ID, HighStakeGuard)
