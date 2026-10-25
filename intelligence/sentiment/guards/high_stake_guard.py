"""LLM high-stake guard validating large BUY orders (e.g. >= 1000 EUR)."""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import topic_spec
from state_io import read_recovering_previous

logger = logging.getLogger("intelligence.sentiment.high_stake_guard")

DEFAULT_MIN_NOTIONAL_USD = 700.0
DEFAULT_MIN_NOTIONAL_EUR = DEFAULT_MIN_NOTIONAL_USD


def compute_24h_price_metrics(series: list) -> Dict[str, Any]:
    """Compute multi-horizon percentage changes (15m, 30m, 1h, 3h) and 24h range from a cache_24_price series."""
    if not isinstance(series, list) or len(series) < 2:
        return {}
    try:
        now_ms = float(series[-1][0])
        cur_px = float(series[-1][1])
        if cur_px <= 0:
            return {}

        def _px_ago(mins: float) -> float:
            target = now_ms - mins * 60000.0
            for ts_ms, p_val in reversed(series):
                if float(ts_ms) <= target:
                    return float(p_val)
            return float(series[0][1])

        p15, p30, p60, p180 = _px_ago(15), _px_ago(30), _px_ago(60), _px_ago(180)
        sample_prices = [float(x[1]) for x in series[::max(1, len(series) // 500)] if len(x) > 1]
        out: Dict[str, Any] = {
            "low_24h": min(sample_prices) if sample_prices else cur_px,
            "high_24h": max(sample_prices) if sample_prices else cur_px,
        }
        if p15 > 0:
            out["chg_15m_pct"] = round((cur_px / p15 - 1.0) * 100.0, 2)
        if p30 > 0:
            out["chg_30m_pct"] = round((cur_px / p30 - 1.0) * 100.0, 2)
        if p60 > 0:
            out["chg_1h_pct"] = round((cur_px / p60 - 1.0) * 100.0, 2)
        if p180 > 0:
            out["chg_3h_pct"] = round((cur_px / p180 - 1.0) * 100.0, 2)
        return out
    except Exception:
        return {}


def extract_symbol_recent_fills(orders_list: list) -> Optional[Dict[str, Any]]:
    """Extract last action side, active cycle avg BUY price, last BUY/SELL fills, and recent tranches."""
    if not isinstance(orders_list, list) or not orders_list:
        return None
    valid = [o for o in orders_list if isinstance(o, dict)]
    if not valid:
        return None
    ordered = sorted(
        enumerate(valid),
        key=lambda item: (
            float(
                item[1].get("timestamp")
                if item[1].get("timestamp") is not None
                else (item[1].get("time") or 0.0)
            ),
            item[0],
        ),
    )
    chron = [o for _, o in ordered]
    last_buy = next((o for o in reversed(chron) if str(o.get("side") or "").upper() == "BUY"), None)
    last_sell = next((o for o in reversed(chron) if str(o.get("side") or "").upper() == "SELL"), None)
    last_any = chron[-1]

    def _norm_fill(o: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Any]:
        q_raw = (
            o.get("quantity")
            if o.get("quantity") is not None
            else (o.get("qty") if o.get("qty") is not None else o.get("executedQty"))
        )
        p_raw = o.get("price")
        ts_raw = o.get("timestamp") if o.get("timestamp") is not None else o.get("time")
        try:
            q_f = float(q_raw) if q_raw is not None else 0.0
        except (TypeError, ValueError):
            q_f = 0.0
        try:
            p_f = float(p_raw) if p_raw is not None else 0.0
        except (TypeError, ValueError):
            p_f = 0.0
        if p_f <= 0 and q_f > 0 and o.get("cummulativeQuoteQty") is not None:
            try:
                p_f = round(float(o["cummulativeQuoteQty"]) / q_f, 6)
            except (TypeError, ValueError, ZeroDivisionError):
                p_f = 0.0
        return (p_f if p_f > 0 else None), (q_f if q_f > 0 else None), ts_raw

    cost_qty = 0.0
    active_avg = 0.0
    buys_fmt: list[Dict[str, Any]] = []
    sells_fmt: list[Dict[str, Any]] = []
    for o in chron:
        side_o = str(o.get("side") or "").upper()
        p_f, q_f, ts_raw = _norm_fill(o)
        if side_o == "BUY":
            buys_fmt.append({"price": p_f, "qty": q_f, "ts_ms": ts_raw})
            if q_f is not None and p_f is not None and math.isfinite(q_f) and math.isfinite(p_f):
                new_cost_qty = cost_qty + q_f
                active_avg = ((cost_qty * active_avg) + (q_f * p_f)) / new_cost_qty if new_cost_qty > 0 else 0.0
                cost_qty = new_cost_qty
        elif side_o == "SELL":
            sells_fmt.append({"price": p_f, "qty": q_f, "ts_ms": ts_raw})
            if q_f is not None and math.isfinite(q_f):
                if q_f >= cost_qty * 0.98 or cost_qty - q_f <= 1e-12:
                    cost_qty = 0.0
                    active_avg = 0.0
                else:
                    cost_qty -= q_f

    lb_norm = _norm_fill(last_buy) if last_buy else (None, None, None)
    ls_norm = _norm_fill(last_sell) if last_sell else (None, None, None)
    return {
        "last_action_side": last_any.get("side"),
        "active_avg_buy_px": round(active_avg, 6) if active_avg > 0 else None,
        "last_buy": (
            {"price": lb_norm[0], "qty": lb_norm[1], "ts_ms": lb_norm[2]}
            if last_buy else None
        ),
        "last_sell": (
            {"price": ls_norm[0], "qty": ls_norm[1], "ts_ms": ls_norm[2]}
            if last_sell else None
        ),
        "recent_buys": buys_fmt[-4:],
        "recent_sells": sells_fmt[-3:],
    }




def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    if cachedb_dir == "cachedb":
        try:
            from cachepaths import get_cachedb_dir
            cachedb_dir = get_cachedb_dir()
        except Exception:
            cachedb_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
    telemetry: Dict[str, Any] = {}

    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
        if ob_snap is not None:
            bid_pct = int(ob_snap.imbalance_ratio * 100)
            ask_pct = int((1.0 - ob_snap.imbalance_ratio) * 100)
            telemetry["Orderbook Depth Imbalance"] = f"{ob_snap.imbalance_ratio:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            if ob_snap.largest_ask_wall_usd > 100_000.0:
                telemetry["Orderbook Ask Wall Ahead"] = f"${ob_snap.largest_ask_wall_usd:,.0f} @ ${ob_snap.largest_ask_wall_price:,.2f}"
            else:
                telemetry["Orderbook Ask Wall Ahead"] = "None detected within 2%"
            if getattr(ob_snap, "largest_bid_wall_usd", 0.0) > 100_000.0:
                telemetry["Orderbook Bid Wall Ahead"] = f"${ob_snap.largest_bid_wall_usd:,.0f} @ ${ob_snap.largest_bid_wall_price:,.2f}"
            else:
                telemetry["Orderbook Bid Wall Ahead"] = "None detected within 2%"
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
        if f_snap is not None:
            telemetry["Perpetual Funding Rate"] = f"{f_snap.funding_rate * 100:+.4f}% (8h)"
            if f_snap.open_interest_usd > 0:
                telemetry["Derivatives Open Interest"] = f"${f_snap.open_interest_usd:,.0f}"
    except Exception:
        pass

    # 3. Whale Positioning: Top Traders & Aggressive Flow (Pillar 2)
    try:
        from external_order_guard import get_whale_collector
        w_snap = get_whale_collector().fetch(symbol, allow_network=False)
        if w_snap is not None:
            long_pct = w_snap.top_traders_long_pct * 100
            telemetry["Whale Positioning"] = f"{long_pct:.1f}% Top Traders Long (L/S ratio: {w_snap.top_traders_long_ratio:.2f})"
            telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {w_snap.taker_buy_sell_ratio:.2f} (regime: {w_snap.divergence_regime})"
    except Exception:
        pass

    # 4. Technical Trend & Regime (Pillars 0 & 1)
    if regime_context is not None:
        try:
            trend = getattr(regime_context, "resolved_trend", None)
            dec = getattr(regime_context, "decision", None)
            grad = getattr(dec, "gradient", None) if dec else None
            eps = getattr(dec, "epsilon", None) if dec else None
            trend_str = f"{trend}" if trend else "unspecified"
            if grad is not None and eps is not None:
                trend_str += f" (gradient={grad:+.4f}, epsilon={eps:.4f})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    if dur_sec and dur_sec > 0:
        telemetry["Trend Duration"] = f"{dur_sec / 3600.0:.1f} hours"

    # 4B. 24h Multi-Horizon Price Evolution & Dynamic Multi-Window Slopes (shared with Reconciler)
    from intelligence.external.collectors.orderbook_depth import _symbol_variants
    sym_variants = _symbol_variants(symbol)
    try:
        m24: Dict[str, Any] = {}
        for cand in sym_variants:
            if "/" in cand:
                continue
            p24_path = os.path.join(cachedb_dir, f"cache_24_price_{cand}.json")
            p24_data = read_recovering_previous(p24_path, json.loads, default=None)
            if not isinstance(p24_data, dict):
                continue
            raw_items = p24_data.get("items", {})
            series = []
            if isinstance(raw_items, dict):
                for k_cand in sym_variants:
                    series = raw_items.get(k_cand, [])
                    if series:
                        break
                if not series and raw_items:
                    series = next(iter(raw_items.values()), [])
            elif isinstance(raw_items, list):
                series = raw_items
            m24 = compute_24h_price_metrics(series)
            if m24:
                break
        if m24:
            telemetry["24h Multi-Horizon Price Evolution"] = (
                f"15m={m24.get('chg_15m_pct', 0):+.2f}%, 30m={m24.get('chg_30m_pct', 0):+.2f}%, "
                f"1h={m24.get('chg_1h_pct', 0):+.2f}%, 3h={m24.get('chg_3h_pct', 0):+.2f}%, "
                f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
            )
    except Exception:
        pass

    try:
        st_path = os.path.join(cachedb_dir, "cache_short_trend.json")
        st_data = read_recovering_previous(st_path, json.loads, default=None)
        if isinstance(st_data, dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            tinfo = None
            for cand in sym_variants:
                tinfo = st_items.get(cand)
                if isinstance(tinfo, dict):
                    break
            if isinstance(tinfo, dict):
                dyn = tinfo.get("dynamic", {}) if isinstance(tinfo.get("dynamic"), dict) else {}
                s37 = (dyn.get("222") or {}).get("slope_full", 0.0)
                s15 = (dyn.get("900") or {}).get("slope_full", 0.0)
                s25h = (dyn.get("9000") or {}).get("slope_full", 0.0)
                telemetry["Dynamic Multi-Window Slopes"] = f"3.7m={s37:+.5f}, 15m={s15:+.5f}, 2.5h={s25h:+.5f}"
    except Exception:
        pass

    try:
        fills = {}
        for ord_file in ("cache_order.json", "cache_trade_kraken.json"):
            ord_path = os.path.join(cachedb_dir, ord_file)
            ord_data = read_recovering_previous(ord_path, json.loads, default=None)
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) else {}
            if not isinstance(o_items, dict):
                continue
            orders_for_sym = []
            for cand in sym_variants:
                orders_for_sym = o_items.get(cand, [])
                if orders_for_sym:
                    break
            fills = extract_symbol_recent_fills(orders_for_sym)
            if fills:
                break
        if fills:
            lb = fills.get("last_buy") or {}
            ls = fills.get("last_sell") or {}
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias & Fear/Greed Index (Pillar 3)
    try:
        for m_file in ("sentiment_advisor_eval.json", "macro_advisor_eval.json", "gemini_macro_advisor.json"):
            m_path = os.path.join(cachedb_dir, m_file)
            m_data = read_recovering_previous(m_path, json.loads, default=None)
            if isinstance(m_data, dict):
                bias = m_data.get("market_bias", "")
                risk = m_data.get("risk_level", "")
                action = m_data.get("recommended_action", "")
                telemetry["Macro Sentiment Bias"] = f"{bias} (Risk: {risk}, Recommendation: {action})"
                break
    except Exception:
        pass

    try:
        for fg_file in ("fear_greed_collect.json", "fear_greed_cache.json"):
            fg_path = os.path.join(cachedb_dir, fg_file)
            fg_data = read_recovering_previous(fg_path, json.loads, default=None)
            if isinstance(fg_data, dict):
                val = fg_data.get("value", "")
                label = fg_data.get("classification", "")
                if val is not None and val != "":
                    telemetry["Fear & Greed Index"] = f"{val} ({label})"
                    break
    except Exception:
        pass

    # 6. Geopolitical Threat Shield (Pillar 4)
    try:
        for g_file in ("geopolitical_threat_eval.json", "geopolitical_threat_state.json"):
            g_path = os.path.join(cachedb_dir, g_file)
            g_data = read_recovering_previous(g_path, json.loads, default=None)
            if isinstance(g_data, dict):
                level = g_data.get("threat_level", "")
                if level is not None and level != "":
                    telemetry["Geopolitical Threat Level"] = level
                    break
    except Exception:
        pass

    return telemetry


import time

DEFAULT_HIGH_STAKE_CACHE_TTL_SEC = 600.0
_HIGH_STAKE_DECISION_CACHE: Dict[tuple[str, str, int], tuple[float, GuardDecision]] = {}


def is_llm_guard_exempt(explicit_client: bool = False) -> bool:
    """True when LLM pre-trade guards are suppressed via environment flags (backtest/replay/disabled)."""
    return not explicit_client and any(
        os.environ.get(k, "").strip().lower() in ("1", "true", "yes", "on")
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
    fb_norm = str(fallback_action or "allow").strip().lower()
    if not resp:
        if fb_norm in ("veto", "block"):
            return GuardDecision.veto(guard_name, "llm_query_timeout_fail_closed", notional_eur=notional_eur)
        if fb_norm == "downscale":
            return GuardDecision.downscale(
                guard_name, scale=0.5, reason="llm_query_timeout_downscaled", notional_eur=notional_eur
            )
        return GuardDecision.allow(guard_name, "llm_query_timeout_fallback_allowed", notional_eur=notional_eur)

    decision_str = str(resp.get("decision") or "APPROVED").upper().strip()
    reason = str(resp.get("reason") or "evaluated by LLM")
    raw_scale = resp.get("suggested_scale")
    default_scale = 0.5 if decision_str == "DOWNSCALE" else 1.0
    if raw_scale is None or isinstance(raw_scale, bool):
        scale = default_scale
    else:
        try:
            scale = float(raw_scale)
            if not math.isfinite(scale) or not (0.0 <= scale <= 10.0):
                scale = default_scale
        except (TypeError, ValueError):
            scale = default_scale

    if decision_str in ("REJECTED", "VETO", "BLOCK"):
        return GuardDecision.veto(
            guard_name,
            f"Gemini vetoed {action_label}: {reason}",
            notional_eur=notional_eur,
            gemini_reason=reason,
        )
    if decision_str == "DOWNSCALE":
        if scale <= 0.0 or scale >= 1.0:
            scale = 0.5
        clamped_scale = max(0.1, min(1.0, scale))
        return GuardDecision.downscale(
            guard_name,
            scale=clamped_scale,
            reason=f"Gemini downscaled {action_label}: {reason}",
            notional_eur=notional_eur,
            gemini_reason=reason,
        )
    return GuardDecision.allow(
        guard_name,
        f"Gemini approved {action_label}: {reason}",
        notional_eur=notional_eur,
        gemini_reason=reason,
    )



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
        timeout_sec: float = 25.0,
        fallback_action: str = "allow",  # "allow" | "downscale" | "veto"
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        resolved_min = float(min_notional_usd) if min_notional_usd is not None else float(min_notional_eur)
        self.min_notional_usd = resolved_min
        self.min_notional_eur = resolved_min
        self.timeout_sec = float(timeout_sec)
        self.fallback_action = fallback_action.strip().lower()
        if cache_ttl_sec is not None:
            self.cache_ttl_sec = float(cache_ttl_sec)
        elif not self._explicit_client and getattr(self.gemini_client, "_custom_runner", None) is None:
            self.cache_ttl_sec = DEFAULT_HIGH_STAKE_CACHE_TTL_SEC
        else:
            self.cache_ttl_sec = 0.0

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
    ) -> GuardDecision:
        """Evaluate order through LLM if value exceeds threshold."""
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow("HighStakeGuard", "unsupported_side_exempt")

        # 2. Compute effective notional value in EUR/USD
        notional = float(notional_eur if notional_eur is not None else (price * qty))
        if notional < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)",
                notional_eur=notional,
            )

        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("HighStakeGuard", "backtest_mode_exempt", notional_eur=notional)

        now = time.time()
        cache_key = (symbol.upper(), side_norm, int(notional // 100.0))
        if self.cache_ttl_sec > 0:
            cached = _HIGH_STAKE_DECISION_CACHE.get(cache_key)
            if cached is not None and 0.0 <= (now - cached[0]) < self.cache_ttl_sec:
                return cached[1]

        # 3. High-stake order detected: build rich multi-pillar telemetry context for LLM
        effective_telemetry = telemetry if telemetry is not None else collect_pretrade_telemetry(
            symbol, regime_context=regime_context, dur_sec=dur_sec
        )
        context_items = [
            f"- Symbol: {symbol}",
            f"- Side: {side_norm}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        ]
        if effective_telemetry:
            for k, v in effective_telemetry.items():
                context_items.append(f"- {k}: {v}")

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
                "3. MACRO & GEOPOLITICAL RISK: If Macro Sentiment recommends CAUTION or Geopolitical threat is elevated, do not fight negative taker flow.\n\n"
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
                "3. CAPITULATION & SQUEEZE PROTECTION: Do not approve panic-selling into extreme negative funding or active whale accumulation at local bottoms.\n\n"
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
            return evaluate_llm_guard_response(
                "HighStakeGuard",
                f"high-stake {side_norm}",
                None,
                fallback_action=self.fallback_action,
                notional_eur=notional,
            )

        dec = evaluate_llm_guard_response(
            "HighStakeGuard",
            f"high-stake {side_norm}",
            resp,
            fallback_action=self.fallback_action,
            notional_eur=notional,
        )

        if self.cache_ttl_sec > 0:
            _HIGH_STAKE_DECISION_CACHE[cache_key] = (now, dec)
        return dec
