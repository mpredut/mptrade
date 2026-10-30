"""LLM high-stake guard validating large BUY and SELL orders (e.g. >= 700 EUR/USD)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from botcore import (
    _parse_bool_flag,
    fetch_public_json,
)
from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    clean_symbol_key,
    is_cache_fresh,
    normalize_order_side,
    resolve_now_ts,
    resolve_order_notional,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    MACRO_BENCHMARK_SPECS,
    _read_first_cached_dict,
    apply_asset_scope_filter,
    build_cycle_fallback_prices,
    collect_macro_benchmarks_snapshot,
    collect_pretrade_telemetry,
    format_bottom_structure_6h_parts,
    format_macro_benchmarks_summary_line,
    harvest_initial_prices_and_trends,
    harvest_macro_and_news_sensors,
    harvest_portfolio_and_venue_cycles,
    populate_market_and_portfolio_snapshot,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _blend_structural_liq_anchor,
    _build_vp_position_dict,
    _compute_horizon_range_dict,
    _extract_cached_items_series,
    _first_variant_dict,
    _parse_ts_px_row,
    _prices_diverged,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_24h_price_metrics,
    compute_7d_range_metrics,
    compute_cross_btc_metrics,
    compute_liquidation_clusters,
    compute_swing_wave_metrics,
    compute_volume_profile_metrics,
    enrich_microstructure_cvd,
    extract_symbol_recent_fills,
    load_symbol_microstructure,
    load_symbol_price_series,
    round_opt_px,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _VENUE_DEFAULT_FEE_SPECS,
    _compute_kalman_from_series,
    _load_recent_audit_origins,
    build_cross_market_context,
    enrich_24h_and_microstructure,
    enrich_cross_venue_trends,
    harvest_bot_runtime_states,
    harvest_kraken_xstocks,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles,
    load_venue_trailing_maps,
    parse_venue_cycle_state,
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
    set_pos_anchor,
    sync_sibling_symbol_telemetry,
)
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import topic_spec

logger = logging.getLogger("intelligence.sentiment.high_stake_guard")

DEFAULT_MIN_NOTIONAL_USD = 700.0
DEFAULT_MIN_NOTIONAL_EUR = DEFAULT_MIN_NOTIONAL_USD


DEFAULT_HIGH_STAKE_CACHE_TTL_SEC = 600.0
_HIGH_STAKE_DECISION_CACHE: Dict[tuple[str, str, int], tuple[float, GuardDecision]] = {}


def is_llm_guard_exempt(explicit_client: bool = False) -> bool:
    """True when LLM pre-trade guards are suppressed via environment flags (backtest/replay/disabled)."""
    exp_b = _parse_bool_flag(explicit_client, False)
    return not exp_b and any(
        _parse_bool_flag(os.environ.get(k))
        for k in ("DISABLE_GEMINI_LLM", "BACKTEST_MODE", "REPLAY_MODE")
    )


def evaluate_llm_guard_response(
    guard_name: str,
    action_label: str,
    resp: Optional[Dict[str, Any]],
    *,
    fallback_action: str,
    notional_eur: float,
) -> GuardDecision:
    """SSOT converter from a pre-trade LLM JSON response (or timeout None) into a canonical GuardDecision."""
    safe_notional = safe_nonneg_float(notional_eur, 0.0)
    fb_norm = str(fallback_action or "allow").strip().lower()
    if not resp or not isinstance(resp, dict):
        if fb_norm in ("veto", "block"):
            return GuardDecision.veto(guard_name, "llm_query_timeout_fail_closed", notional_eur=safe_notional)
        if fb_norm == "downscale":
            return GuardDecision.downscale(
                guard_name, scale=0.5, reason="llm_query_timeout_downscaled", notional_eur=safe_notional
            )
        return GuardDecision.allow(guard_name, "llm_query_timeout_fallback_allowed", notional_eur=safe_notional)

    raw_dec = resp.get("decision")
    decision_str = str(raw_dec if (isinstance(raw_dec, str) and raw_dec.strip()) else "APPROVED").upper().strip()
    raw_reason = resp.get("reason")
    reason = str(raw_reason).strip() if (isinstance(raw_reason, str) and raw_reason.strip()) else "evaluated by LLM"
    default_scale = 0.5 if decision_str == "DOWNSCALE" else 1.0
    scale = clean_scale(resp.get("suggested_scale"), default_scale, allow_percentage=True)

    if decision_str in ("REJECTED", "REJECT", "VETO", "BLOCK", "DENIED", "DENY"):
        return GuardDecision.veto(
            guard_name,
            f"Gemini vetoed {action_label}: {reason}",
            notional_eur=safe_notional,
            gemini_reason=reason,
        )
    if decision_str == "DOWNSCALE":
        if scale <= 0.0 or scale >= 1.0:
            scale = 0.5
        return GuardDecision.downscale(
            guard_name,
            scale=max(0.1, min(1.0, scale)),
            reason=f"Gemini downscaled {action_label}: {reason}",
            notional_eur=safe_notional,
            gemini_reason=reason,
        )
    return GuardDecision.allow(
        guard_name,
        f"Gemini approved {action_label}: {reason}",
        notional_eur=safe_notional,
        gemini_reason=reason,
    )


def resolve_llm_guard_cache_ttl(
    cache_ttl_sec: Any,
    explicit_client: bool,
    gemini_client: Any,
    default_ttl_sec: float,
) -> float:
    """Resolve effective LLM guard decision cache TTL (0.0 when custom client/runner is injected)."""
    if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
        return safe_nonneg_ts(cache_ttl_sec, 0.0)
    if not explicit_client and getattr(gemini_client, "_custom_runner", None) is None:
        return default_ttl_sec
    return 0.0


class HighStakeGuard:
    """Invokes LLM reasoning before executing large BUY purchases (>= 1000 EUR).

    Sub-1000 EUR orders pass immediately with zero overhead (0 ms).
    Orders exceeding the threshold undergo real-time LLM risk vetting against
    market psychology, whale positioning, and liquidation cascades.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        min_notional_eur: float = DEFAULT_MIN_NOTIONAL_USD,
        min_notional_usd: Optional[float] = None,
        timeout_sec: float = 120.0,
        fallback_action: str = "allow",  # "allow" | "downscale" | "veto"
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        raw_min = min_notional_usd if min_notional_usd is not None else min_notional_eur
        resolved_min = safe_pos_float(raw_min, DEFAULT_MIN_NOTIONAL_USD)
        self.min_notional_usd = resolved_min
        self.min_notional_eur = resolved_min
        self.timeout_sec = safe_pos_float(timeout_sec, 120.0)
        self.fallback_action = str(fallback_action or "allow").strip().lower()
        self.cache_ttl_sec = resolve_llm_guard_cache_ttl(
            cache_ttl_sec, self._explicit_client, self.gemini_client, DEFAULT_HIGH_STAKE_CACHE_TTL_SEC
        )

    def check(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        notional_eur: Optional[float] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order through LLM if value exceeds threshold."""
        side_norm = normalize_order_side(side)
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow("HighStakeGuard", "unsupported_side_exempt")

        notional = resolve_order_notional(notional_eur, price, qty, default=0.0, fallback_on_nonfinite=False) or 0.0
        notional_f = safe_float(notional, default=None)
        if notional_f is None or notional_f < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)",
                notional_eur=notional,
            )

        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("HighStakeGuard", "backtest_mode_exempt", notional_eur=notional)

        now_ts = resolve_now_ts(now)
        cache_key = (clean_symbol_key(symbol), side_norm, int(notional // 100.0))
        if self.cache_ttl_sec > 0:
            cached = _HIGH_STAKE_DECISION_CACHE.get(cache_key)
            if isinstance(cached, (tuple, list)) and len(cached) >= 2 and is_cache_fresh(cached[0], now_ts=now_ts, ttl_sec=self.cache_ttl_sec):
                return cached[1]

        effective_telemetry = telemetry if isinstance(telemetry, dict) else collect_pretrade_telemetry(
            symbol, regime_context=regime_context, dur_sec=dur_sec
        )
        context_items = [
            f"- Symbol: {symbol}",
            f"- Side: {side_norm}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        ]
        if isinstance(effective_telemetry, dict) and effective_telemetry:
            context_items.extend(f"- {k}: {v}" for k, v in effective_telemetry.items())

        context_str = "\n".join(context_items)
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
                "3. MACRO, GEOPOLITICAL & CONTRARIAN BOTTOM ACCUMULATION: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow. However, when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM' (especially with 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE') AND Whale Taker Flow is >= 1.0 (or 15m CVD is positive/spot_absorbing), issue APPROVED — do NOT veto or downscale a verified contrarian bottom entry merely because lagging 24h breadth or residual geopolitical headlines remain elevated.\n\n"
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
                "3. CAPITULATION & SQUEEZE PROTECTION (6H DOUBLE-BOTTOM & RECLAIM DEFENSE): Do not approve panic-selling into extreme negative funding, active whale accumulation, or a verified 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'FRESH_6H_LOW_TEST') at local bottoms.\n\n"
            )

        prompt = (
            "You are a principal quantitative crypto risk officer. "
            f"An automated trading bot is about to place a {action_desc}:\n\n"
            f"{context_str}\n\n"
            f"Because this order exceeds €{self.min_notional_eur:,.2f} EUR, you must evaluate if executing "
            f"{risk_context}\n\n"
            f"{policy_rules}"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "decision": "APPROVED" | "REJECTED" | "DOWNSCALE",\n'
            '  "suggested_scale": 0.0 to 1.0,\n'
            '  "reason": "Concise 1-2 sentence rationale for the decision"\n'
            "}"
        )

        thread_title = f"{topic_spec('guard_high_stake').title} — {side_norm} {symbol} €{notional:,.2f}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=self.timeout_sec,
            thread_title=thread_title,
            topic_key="guard_high_stake",
        )
        if not resp:
            logger.warning(
                "LLM high-stake query failed or timed out for %s €%.2f. Fallback: %s",
                symbol,
                notional,
                self.fallback_action,
            )

        dec = evaluate_llm_guard_response(
            "HighStakeGuard",
            f"high-stake {side_norm}",
            resp or None,
            fallback_action=self.fallback_action,
            notional_eur=notional,
        )
        if self.cache_ttl_sec > 0:
            _HIGH_STAKE_DECISION_CACHE[cache_key] = (now_ts, dec)
        return dec
