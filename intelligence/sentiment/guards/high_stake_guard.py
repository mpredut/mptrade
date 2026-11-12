"""LLM high-stake guard validating large BUY and SELL orders (e.g. >= 700 EUR/USD)."""
from __future__ import annotations

from abc import ABC
from dataclasses import replace as _dc_replace
import logging
from typing import Any, Callable, Dict, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    clean_scale,
    first_pos_float,
    is_cache_fresh,
    is_gemini_llm_disabled,
    normalize_side,
    resolve_now_ts,
    resolve_order_notional,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import clean_symbol_key, has_equity_suffix
from intelligence.internal.guards.guard_decision import (
    GuardDecision,
    OrderGuardEvaluatorBase,
    OrderGuardEvaluatorFactory,
    normalize_guard_mode,
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
    return not _parse_bool_flag(explicit_client, False) and is_gemini_llm_disabled()


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
    scale = clean_scale(resp.get("suggested_scale"), 0.5 if decision_str == "DOWNSCALE" else 1.0, allow_percent=True)

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
        return safe_nonneg_float(cache_ttl_sec, default=0.0)
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
                eval_notional = first_pos_float(
                    base_meta.get("evaluated_notional_eur"),
                    base_meta.get("notional_eur"),
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
                    return _dc_replace(
                        base_dec,
                        metadata={
                            **base_meta,
                            "notional_eur": notional,
                            "evaluated_notional_eur": baseline_notional,
                            "cache_hit": True,
                        },
                    )

        eff_max_age = (
            safe_pos_float(max_telemetry_age_sec, default=None)
            if max_telemetry_age_sec is not None and not isinstance(max_telemetry_age_sec, bool)
            else self.max_telemetry_age_sec
        )
        enforce_fresh = eff_max_age is not None
        if snapshot is not None:
            resolved_snapshot = PretradeTelemetrySnapshot.from_raw(
                snapshot,
                symbol=symbol,
                now_ts=now_ts,
                source="explicit",
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )
        elif telemetry is not None:
            resolved_snapshot = PretradeTelemetrySnapshot.from_raw(
                telemetry,
                symbol=symbol,
                now_ts=now_ts,
                source="explicit",
                max_age_sec=eff_max_age,
                enforce_freshness=enforce_fresh,
            )
        else:
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
                        resolved_snapshot = PretradeTelemetrySnapshot.from_raw(
                            raw_telem,
                            symbol=symbol,
                            now_ts=now_ts,
                            max_age_sec=eff_max_age,
                            enforce_freshness=enforce_fresh,
                        )
                    else:
                        resolved_snapshot = PretradeTelemetrySnapshot.from_raw(
                            raw_telem,
                            symbol=symbol,
                            now_ts=now_ts,
                            source="collector",
                            max_age_sec=eff_max_age,
                            enforce_freshness=enforce_fresh,
                        )
                else:
                    resolved = _snap_fn(
                        symbol,
                        regime_context=regime_context,
                        dur_sec=dur_sec,
                        now_ts=now_ts,
                        max_age_sec=eff_max_age,
                        enforce_freshness=enforce_fresh,
                    )
                    resolved_snapshot = (
                        resolved
                        if isinstance(resolved, PretradeTelemetrySnapshot)
                        else PretradeTelemetrySnapshot.from_raw(
                            resolved,
                            symbol=symbol,
                            now_ts=now_ts,
                            source="cachedb",
                            max_age_sec=eff_max_age,
                            enforce_freshness=enforce_fresh,
                        )
                    )
            except Exception as exc:
                logger.warning("Pre-trade telemetry read failed for %s: %s", symbol, exc)
                resolved_snapshot = PretradeTelemetrySnapshot.unavailable(
                    symbol,
                    reason="telemetry_unavailable",
                    now_ts=now_ts,
                    max_age_sec=eff_max_age,
                )
        snap_meta: Dict[str, Any] = {
            "telemetry_symbol": resolved_snapshot.canonical_symbol,
            "telemetry_asset_scope": resolved_snapshot.asset_scope,
            "telemetry_source": resolved_snapshot.source,
            "telemetry_pillar_sources": dict(resolved_snapshot.pillar_sources),
            "telemetry_ts": resolved_snapshot.ts,
            "telemetry_age_sec": round(resolved_snapshot.age_sec, 3) if resolved_snapshot.age_sec is not None else None,
            "telemetry_max_age_sec": resolved_snapshot.max_age_sec,
            "telemetry_fresh": resolved_snapshot.is_fresh,
            "telemetry_available": resolved_snapshot.is_available,
        }
        if not resolved_snapshot.is_available:
            unavail_reason = str(resolved_snapshot.unavailable_reason or "telemetry_unavailable").strip()
            (logger_inst or logger).warning(
                "Pre-trade telemetry unavailable for %s €%.2f (%s). Fallback: %s",
                symbol,
                notional,
                unavail_reason,
                self.fallback_action,
            )
            safe_notional = safe_nonneg_float(notional, 0.0)
            fb_norm = str(self.fallback_action or "allow").strip().lower()
            meta: Dict[str, Any] = {
                **snap_meta,
                "notional_eur": safe_notional,
                "evaluated_notional_eur": safe_notional,
                "telemetry_unavailable_reason": unavail_reason,
            }
            if fb_norm in ("veto", "block"):
                dec = GuardDecision.veto(self.GUARD_NAME, f"{unavail_reason}_fail_closed", **meta)
            elif fb_norm == "downscale":
                dec = GuardDecision.downscale(self.GUARD_NAME, scale=0.5, reason=f"{unavail_reason}_downscaled", **meta)
            else:
                dec = GuardDecision.allow(self.GUARD_NAME, f"{unavail_reason}_fallback_allowed", **meta)
            if on_decision is not None:
                on_decision(dec)
            return self._wrap_shadow_decision(dec, notional, None) if self.mode == "shadow" else dec

        context_items = list(header_lines)
        if resolved_snapshot:
            age_str = f"{resolved_snapshot.age_sec:.1f}s" if resolved_snapshot.age_sec is not None else "n/a"
            src_str = resolved_snapshot.source or "cachedb"
            context_items.append(
                f"- Telemetry Snapshot: symbol={resolved_snapshot.canonical_symbol} "
                f"(scope={resolved_snapshot.asset_scope}, source={src_str}, age={age_str}, fresh={resolved_snapshot.is_fresh})"
            )
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
        min_notional_eur: Optional[float] = None,
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
        self._legacy_eur_unit = min_notional_eur is not None and min_notional_usd is None
        raw_min = (
            min_notional_usd
            if min_notional_usd is not None
            else (min_notional_eur if min_notional_eur is not None else DEFAULT_MIN_NOTIONAL_USD)
        )
        resolved_min = safe_pos_float(raw_min, DEFAULT_MIN_NOTIONAL_USD)
        self.min_notional_usd = self.min_notional_eur = resolved_min

    def _resolve_currency_display(self, symbol: str) -> tuple[str, str]:
        sym_u = "" if isinstance(symbol, bool) else str(symbol or "").strip().upper()
        if sym_u.endswith("EUR"):
            return "€", "EUR"
        if has_equity_suffix(symbol, include_compact=True):
            return "$", "USD"
        if self._legacy_eur_unit:
            return "€", "EUR"
        if sym_u.endswith("USDC"):
            return "$", "USDC"
        return "$", "USD"

    def _build_high_stake_prompt(
        self,
        side_norm: str,
        notional: float,
        context_str: str,
        *,
        symbol: str = "",
        ccy_sym: str = "€",
        ccy_code: str = "EUR",
    ) -> str:
        is_eq = has_equity_suffix(symbol, include_compact=True)
        if side_norm == "BUY":
            action_desc = "HIGH-VALUE BUY order"
            risk_context = (
                "whether this entry is sound or if there are critical red flags "
                "(e.g., euphoric top buying, aggressive institutional/whale distribution, severe positioning imbalance, or active cascade)."
            )
            policy_rules = (
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions (% Top Traders Long) and Orderbook Bid Depth. Static long holdings can be trapped or liquidated (note: on US cash equities *_US_EQ without perpetual futures, Open Interest reflects Cboe options chain OI rather than leveraged perp positions). If Whale Taker Flow is below 0.85 (net sell aggression), you must NEVER issue an unrestricted APPROVED. If Whale Taker Flow is below 0.80 in a FLAT or BEAR market, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural support).\n"
                "2. ORDERBOOK SUPPORT LIMITATION: Bid walls and orderbook depth are passive and can be pulled or absorbed by aggressive market selling. Do NOT use orderbook depth alone to override persistent taker selling.\n"
                "3. MACRO, GEOPOLITICAL & CONTRARIAN BOTTOM ACCUMULATION: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow. However, when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM' (especially with 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE') AND Whale Taker Flow is >= 1.0 (or 15m CVD is positive/spot_absorbing), issue APPROVED — do NOT veto or downscale a verified contrarian bottom entry merely because lagging 24h breadth or residual geopolitical headlines remain elevated.\n"
                "4. CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW SYMMETRY (Equities <-> Crypto): Synthesize 'Equity & Macro Regime' across both Stocks and Crypto: weigh SPY/QQQ/VIX/DXY/US10Y and Futures_Lead (FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN) alongside IBIT (US ETF flow), CB_Premium (Coinbase vs Binance institutional spot basis bps), BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options(VRP, PC_OI, Skew_25d), and Cross_Asset (COUPLED_RISK_ON | COUPLED_RISK_OFF | CRYPTO_BULLISH_DECOUPLING | EQUITY_BULLISH_DECOUPLING). When Cross_Asset is COUPLED_RISK_OFF with surging VIX, deeply negative CB_Premium (<= -8 bps), or negative Skew_25d (put-buying fear), require strict Taker/CVD absorption before approving large BUY entries; conversely, when Cross_Asset is CRYPTO_BULLISH_DECOUPLING or COUPLED_RISK_ON with positive CB_Premium / IBIT inflow or FUTURES_PREMARKET_GAP_UP, do NOT veto a crypto BUY solely due to soft prior-day equity indices."
            )
        else:
            action_desc, risk_context, policy_rules = (
                "HIGH-VALUE SELL order",
                "whether this sale is sound or if there are critical red flags "
                "(e.g., dumping directly into major institutional/whale support/walls, panic-selling at local bottom during accumulation, or excessive slippage).",
                "CORE RISK POLICIES & PRIORITY RULES:\n"
                "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions and Orderbook Ask Depth. Static short holdings can be squeezed or liquidated (on US cash equities *_US_EQ, Open Interest reflects Cboe options chain OI rather than leveraged perps). If Whale Taker Flow is above 1.18 (net buy aggression), you must NEVER issue an unrestricted APPROVED for a non-stop-loss SELL into strength. If Whale Taker Flow is above 1.25 in a FLAT or BULL market during whale accumulation or short squeeze, you must issue REJECTED (or DOWNSCALE <= 0.25 only if at verified structural resistance).\n"
                "2. ORDERBOOK RESISTANCE LIMITATION: Ask walls and orderbook ask depth are passive and can be pulled or absorbed by aggressive market buying. Do NOT use orderbook ask depth alone to override persistent taker buying.\n"
                "3. CAPITULATION & SQUEEZE PROTECTION (6H DOUBLE-BOTTOM & RECLAIM DEFENSE): Do not approve panic-selling into extreme negative funding, active whale accumulation, or a verified 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'FRESH_6H_LOW_TEST') at local bottoms.\n"
                "4. CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW SYMMETRY (Equities <-> Crypto): Synthesize 'Equity & Macro Regime' across both Stocks and Crypto: weigh SPY/QQQ/VIX/DXY/US10Y and Futures_Lead (FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN) alongside IBIT, CB_Premium, BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options(VRP, PC_OI, Skew_25d), and Cross_Asset (COUPLED_RISK_ON | COUPLED_RISK_OFF | CRYPTO_BULLISH_DECOUPLING | EQUITY_BULLISH_DECOUPLING). Do NOT approve non-stop-loss panic SELLs at local lows when Cross_Asset is CRYPTO_BULLISH_DECOUPLING or COUPLED_RISK_ON with positive CB_Premium (>= +5 bps), IBIT accumulation, or FUTURES_PREMARKET_GAP_UP; conversely, approve defensive profit-taking or risk-reduction SELLs when Cross_Asset is COUPLED_RISK_OFF or EQUITY_BULLISH_DECOUPLING with negative CB_Premium, FUTURES_PREMARKET_GAP_DOWN, and rising VIX/BTC_DVOL.",
            )
        persona_role = (
            "principal quantitative equities and multi-asset risk officer"
            if is_eq
            else "principal quantitative crypto risk officer"
        )
        return (
            f"You are a {persona_role}. "
            f"An automated trading bot is about to place a {action_desc}:\n\n{context_str}\n\n"
            f"Because this order exceeds {ccy_sym}{self.min_notional_eur:,.2f} {ccy_code}, you must evaluate {risk_context}\n\n"
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

        ccy_sym, ccy_code = self._resolve_currency_display(symbol)
        notional = resolve_order_notional(notional_eur, price, qty, default=0.0, fallback_on_nonfinite=False) or 0.0
        notional_f = safe_float(notional, default=None)
        if notional_f is None or notional_f < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} {ccy_code})",
                notional_eur=notional,
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
            f"- Total Notional Value: {ccy_sym}{notional:,.2f} {ccy_code}",
        )
        return self._execute_pretrade_guard_eval(
            symbol=symbol,
            notional=notional,
            cache_key=cache_key,
            decision_cache=_HIGH_STAKE_DECISION_CACHE,
            now_ts=now_ts,
            header_lines=header_lines,
            build_prompt_fn=lambda ctx: self._build_high_stake_prompt(
                side_norm, notional, ctx, symbol=symbol, ccy_sym=ccy_sym, ccy_code=ccy_code
            ),
            subtitle=f"{side_norm} {symbol} {ccy_sym}{notional:,.2f}",
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
