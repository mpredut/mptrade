"""Composite market intelligence coordinator orchestrating triggers and guards across all pillars."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from guard_hierarchy import BaseGuard, GUARD_REGISTRY, _ALL_GUARD_ALIASES, normalize_bypassed_guards

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.state.volatility import calculate_volatility_1h
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.external.collectors.bybit_liquidations import LiquidationSummary
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
from intelligence.sentiment.triggers.sentiment_contrarian_trigger import SentimentContrarianTrigger
from intelligence.sentiment.triggers.market_breadth_trigger import MarketBreadthTrigger
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard
from intelligence.sentiment.guards.high_stake_guard import DEFAULT_MIN_NOTIONAL_USD, HighStakeGuard
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard


@dataclass(frozen=True)
class MarketIntelligenceEvaluation:
    """Consolidated outcome of evaluating directional triggers against all guards."""

    symbol: str
    price: float
    ts: float
    triggers: Tuple[TriggerEvent, ...]
    active_trigger: Optional[TriggerEvent]
    guard_decision: GuardDecision
    can_execute: bool
    effective_scale: float
    metrics: Dict[str, object] = field(default_factory=dict)


class CompositeMarketIntelligence:
    """Unified coordinator enforcing the Drivers/Triggers vs Guards/Brakes paradigm across all pillars."""

    def __init__(
        self,
        parabolic_surge_pct: float = 4.0,
        exhaustion_policy: str = "downscale",  # "downscale" | "veto"
        exhaustion_scale: float = 0.25,
        max_active_cascade_usd: float = 500_000.0,
        max_long_funding_rate: float = 0.0005,
        greed_downscale_threshold: int = 80,
        greed_hard_veto_threshold: int = 90,
        gemini_min_notional_eur: float = DEFAULT_MIN_NOTIONAL_USD,
        gemini_guard: Optional[HighStakeGuard] = None,
        geopolitical_guard: Optional[GeopoliticalShockGuard] = None,
    ):
        # Triggers (Drivers - Signals IN and Signals OUT)
        self.kalman_triggers: Dict[str, KalmanTrendTrigger] = {}
        self.gradient_triggers: Dict[str, LinearGradientTrigger] = {}
        self.liquidation_trigger = LiquidationCapitulationTrigger()
        self.sentiment_trigger = SentimentContrarianTrigger()
        self.breadth_trigger = MarketBreadthTrigger()

        # Pillar 1: Internal Guards (Brakes)
        self.parabolic_guard = ParabolicSurgeGuard(surge_threshold_pct=parabolic_surge_pct)
        self.exhaustion_guard = WeibullExhaustionGuard(policy=exhaustion_policy, exhausted_scale=exhaustion_scale)
        self.noise_guard = NoiseFloorGuard(min_strength_ratio=1.0)

        # Pillar 2: External Guards (Brakes)
        self.cascade_guard = LiquidationCascadeGuard(max_active_cascade_usd=max_active_cascade_usd)
        self.funding_guard = FundingCrowdingGuard(max_long_funding_rate=max_long_funding_rate)

        # Pillar 3: Sentiment & LLM Guards (Brakes)
        self.greed_guard = ExtremeGreedGuard(
            downscale_threshold=greed_downscale_threshold,
            hard_veto_threshold=greed_hard_veto_threshold,
        )
        self.panic_guard = PanicWashoutGuard()
        self.gemini_guard = gemini_guard or HighStakeGuard(min_notional_eur=gemini_min_notional_eur)

        # Macro Geopolitical & Energy Shock Shield (Black Swan Brake)
        self.geopolitical_guard = geopolitical_guard or GeopoliticalShockGuard()

    def _get_kalman(self, symbol: str) -> KalmanTrendTrigger:
        sym = (symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")) if isinstance(symbol, str) else ""
        if sym not in self.kalman_triggers:
            self.kalman_triggers[sym] = KalmanTrendTrigger()
        return self.kalman_triggers[sym]

    def _get_gradient(self, symbol: str) -> LinearGradientTrigger:
        sym = (symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")) if isinstance(symbol, str) else ""
        if sym not in self.gradient_triggers:
            self.gradient_triggers[sym] = LinearGradientTrigger()
        return self.gradient_triggers[sym]

    def evaluate_guards(
        self,
        symbol: str,
        side: str,
        price: float,
        *,
        gradient: Optional[float] = None,
        epsilon: Optional[float] = None,
        trend_duration_seconds: float = 0.0,
        price_history: Optional[List[Tuple[float, float]]] = None,
        volatility_1h_pct: Optional[float] = None,
        liquidation_summary: Optional[LiquidationSummary] = None,
        derivatives_telemetry: Optional[DerivativesTelemetry] = None,
        fear_greed_snapshot: Optional[FearGreedSnapshot] = None,
        market_breadth_snapshot: Optional[MarketBreadthSnapshot] = None,
        geopolitical_assessment: Optional[GeopoliticalThreatAssessment] = None,
        asset_24h_change_pct: Optional[float] = None,
        qty: Optional[float] = None,
        notional_eur: Optional[float] = None,
        now: Optional[float] = None,
        bypass_guards: Optional[object] = None,
    ) -> GuardDecision:
        """Run all protection guards (brakes) across all pillars in sequence.

        Returns the first blocking/downscaling decision or an approved allow decision.
        """
        effective_scale = 1.0
        downscale_dec: Optional[GuardDecision] = None
        raw_bypass_tokens = BaseGuard._coerce_bypass_set(bypass_guards)
        recognized_tokens = [t for t in raw_bypass_tokens if t in _ALL_GUARD_ALIASES]
        bypassed = normalize_bypassed_guards(recognized_tokens if recognized_tokens else None) | raw_bypass_tokens
        side = side.strip().upper() if isinstance(side, str) else ""
        if side in ("B", "BID", "LONG"):
            side = "BUY"
        elif side in ("S", "A", "ASK", "SHORT"):
            side = "SELL"

        def _is_bypassed(code: Optional[str], *legacy_names: str) -> bool:
            if not bypassed:
                return False
            if "all" in bypassed or "*" in bypassed:
                return True
            if code:
                spec = GUARD_REGISTRY.get(code)
                if spec is not None and spec.is_bypassed(bypassed):
                    return True
                if code.lower() in bypassed:
                    return True
            for ln in legacy_names:
                if ln.lower() in bypassed:
                    return True
            return False

        def _record_downscale(dec: GuardDecision) -> None:
            nonlocal effective_scale, downscale_dec
            raw_scale = getattr(dec, "suggested_scale", 1.0)
            if raw_scale is None or isinstance(raw_scale, bool):
                s_val = 1.0
            else:
                try:
                    sf = float(raw_scale)
                    if math.isfinite(sf) and 1.0 < sf <= 100.0:
                        sf = sf / 100.0
                    s_val = sf if (math.isfinite(sf) and 0.0 <= sf <= 1.0) else 1.0
                except (TypeError, ValueError):
                    s_val = 1.0
            if s_val < effective_scale or downscale_dec is None:
                effective_scale = min(effective_scale, s_val)
                downscale_dec = GuardDecision.downscale(
                    dec.guard_name,
                    scale=effective_scale,
                    reason=dec.reason,
                    **dec.metadata,
                )

        grad_finite: Optional[float] = None
        if gradient is not None and not isinstance(gradient, bool):
            try:
                gf = float(gradient)
                if math.isfinite(gf):
                    grad_finite = gf
            except (TypeError, ValueError):
                pass
        eps_finite: Optional[float] = None
        if epsilon is not None and not isinstance(epsilon, bool):
            try:
                ef = float(epsilon)
                if math.isfinite(ef):
                    eps_finite = ef
            except (TypeError, ValueError):
                pass

        # 1. Noise Floor Guard
        if grad_finite is not None and eps_finite is not None and not _is_bypassed(None, "noise_guard", "noisefloorguard"):
            noise_dec = self.noise_guard.check(symbol, side, grad_finite, eps_finite)
            if not noise_dec.allowed:
                return noise_dec

        # 2. Parabolic Surge Guard (Anti-FOMO)
        if not _is_bypassed("G_QLIMIT2", "parabolic_guard", "parabolicsurgeguard"):
            parabolic_dec = self.parabolic_guard.check(
                symbol, side, price, price_history=price_history,
                volatility_1h_pct=volatility_1h_pct, now=now,
            )
            if not parabolic_dec.allowed:
                return parabolic_dec

        # 3. Weibull Trend Exhaustion Guard (G_QLIMIT2)
        dur_sec = 0.0
        if trend_duration_seconds is not None and not isinstance(trend_duration_seconds, bool):
            try:
                df = float(trend_duration_seconds)
                if math.isfinite(df) and df > 0.0:
                    dur_sec = df
            except (TypeError, ValueError):
                dur_sec = 0.0
        if dur_sec > 0.0 and not _is_bypassed("G_QLIMIT2", "exhaustion_guard", "weibullexhaustionguard"):
            surv = get_trend_survival_metrics(symbol, dur_sec)
            qlimit1_active = not _is_bypassed("G_QLIMIT1", "qlimit1", "gaussian_horizon")
            inferred_trend_dir = (
                "bull" if (grad_finite is not None and grad_finite > 0)
                else ("bear" if (grad_finite is not None and grad_finite < 0) else None)
            )
            exh_dec = self.exhaustion_guard.check(
                symbol, side, dur_sec,
                p90_days=surv.get("p90_days"),
                median_days=surv.get("median_days"),
                gaussian_horizon_days=surv.get("gaussian_horizon_days") if qlimit1_active else None,
                trend_direction=inferred_trend_dir,
            )
            if not exh_dec.allowed:
                return exh_dec
            if exh_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(exh_dec)

        # 4. External Liquidation Cascade Guard (Active Falling Knife)
        if liquidation_summary is not None and not _is_bypassed(None, "cascade_guard", "liquidationcascadeguard"):
            cascade_dec = self.cascade_guard.check(symbol, side, summary=liquidation_summary, now=now)
            if not cascade_dec.allowed:
                return cascade_dec

        # 5. External Funding Rate Crowding Guard (GI_FUNDING)
        if derivatives_telemetry is not None and not _is_bypassed("GI_FUNDING", "funding_guard", "fundingcrowdingguard"):
            try:
                fund_dec = self.funding_guard.check(symbol, side, telemetry=derivatives_telemetry, now=now)
            except TypeError:
                fund_dec = self.funding_guard.check(symbol, side, telemetry=derivatives_telemetry)
            if not fund_dec.allowed:
                return fund_dec
            if fund_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(fund_dec)

        # 6. Sentiment Extreme Greed Guard (GLLM2)
        if fear_greed_snapshot is not None and not _is_bypassed("GLLM2", "greed_guard", "extremegreedguard"):
            greed_dec = self.greed_guard.check(symbol, side, fear_greed_snapshot)
            if not greed_dec.allowed:
                return greed_dec
            if greed_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(greed_dec)

        # 7. Sentiment Panic Washout Guard (GLLM2)
        if (market_breadth_snapshot is not None or fear_greed_snapshot is not None) and not _is_bypassed("GLLM2", "panic_guard", "panicwashoutguard"):
            panic_dec = self.panic_guard.check(
                symbol,
                side,
                breadth_snapshot=market_breadth_snapshot,
                fear_greed_snapshot=fear_greed_snapshot,
                asset_24h_change_pct=asset_24h_change_pct,
            )
            if not panic_dec.allowed:
                return panic_dec
            if panic_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(panic_dec)

        # 8. Google Gemini High-Stake Guard (GLLM1)
        computed_notional: Optional[float] = None
        if notional_eur is not None and not isinstance(notional_eur, bool):
            try:
                nf = float(notional_eur)
                if math.isfinite(nf) and nf > 0.0:
                    computed_notional = nf
            except (TypeError, ValueError):
                computed_notional = None
        qty_finite: Optional[float] = None
        if qty is not None and not isinstance(qty, bool):
            try:
                qf = float(qty)
                if math.isfinite(qf) and qf > 0.0:
                    qty_finite = qf
            except (TypeError, ValueError):
                qty_finite = None
        price_finite: Optional[float] = None
        if price is not None and not isinstance(price, bool):
            try:
                pf = float(price)
                if math.isfinite(pf) and pf > 0.0:
                    price_finite = pf
            except (TypeError, ValueError):
                price_finite = None
        if computed_notional is None and qty_finite is not None and price_finite is not None:
            computed_notional = price_finite * qty_finite
        if (
            computed_notional is not None
            and computed_notional >= self.gemini_guard.min_notional_eur
            and not _is_bypassed("GLLM1", "gemini_guard", "highstakeguard")
        ):
            telemetry_ctx = {
                "fear_greed": (
                    fear_greed_snapshot.get("value")
                    if isinstance(fear_greed_snapshot, dict)
                    else getattr(fear_greed_snapshot, "value", None)
                ) if fear_greed_snapshot else None,
                "breadth_regime": (
                    market_breadth_snapshot.get("regime")
                    if isinstance(market_breadth_snapshot, dict)
                    else getattr(market_breadth_snapshot, "regime", None)
                ) if market_breadth_snapshot else None,
            }
            eff_price = price_finite if price_finite is not None else price
            eff_qty = qty_finite if qty_finite is not None else 1.0
            try:
                gemini_dec = self.gemini_guard.check(
                    symbol,
                    side,
                    eff_price,
                    eff_qty,
                    notional_eur=computed_notional,
                    telemetry=telemetry_ctx,
                    dur_sec=dur_sec if dur_sec > 0 else None,
                    now=now,
                )
            except TypeError:
                gemini_dec = self.gemini_guard.check(
                    symbol,
                    side,
                    eff_price,
                    eff_qty,
                    notional_eur=computed_notional,
                    telemetry=telemetry_ctx,
                )
            if not gemini_dec.allowed:
                return gemini_dec
            if gemini_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(gemini_dec)

        # 9. Geopolitical & Energy Shock Guard (GLLM3)
        if geopolitical_assessment is not None and not _is_bypassed("GLLM3", "geopolitical_guard", "geopoliticalshockguard"):
            geo_dec = self.geopolitical_guard.check(symbol, side, geopolitical_assessment)
            if not geo_dec.allowed:
                return geo_dec
            if geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                _record_downscale(geo_dec)

        if downscale_dec is not None:
            return downscale_dec

        return GuardDecision.allow("CompositeGuards", "all_guards_cleared")

    def evaluate(
        self,
        symbol: str,
        price: float,
        *,
        ts: Optional[float] = None,
        gradient: Optional[float] = None,
        epsilon: Optional[float] = None,
        trend_duration_seconds: float = 0.0,
        price_history: Optional[List[Tuple[float, float]]] = None,
        prices_for_vol: Optional[Sequence[float]] = None,
        sample_rate_sec: float = 60.0,
        liquidation_summary: Optional[LiquidationSummary] = None,
        derivatives_telemetry: Optional[DerivativesTelemetry] = None,
        fear_greed_snapshot: Optional[FearGreedSnapshot] = None,
        market_breadth_snapshot: Optional[MarketBreadthSnapshot] = None,
        geopolitical_assessment: Optional[GeopoliticalThreatAssessment] = None,
        asset_24h_change_pct: Optional[float] = None,
        qty: Optional[float] = None,
        notional_eur: Optional[float] = None,
        bypass_guards: Optional[object] = None,
    ) -> MarketIntelligenceEvaluation:
        """Evaluate directional triggers and validate against all guards."""
        now_ts = time.time()
        if ts is not None and not isinstance(ts, bool):
            try:
                tf = float(ts)
                if math.isfinite(tf) and tf > 0.0:
                    now_ts = tf
            except (TypeError, ValueError):
                pass
        price_clean = 0.0
        if price is not None and not isinstance(price, bool):
            try:
                pf = float(price)
                if math.isfinite(pf) and pf > 0.0:
                    price_clean = pf
            except (TypeError, ValueError, OverflowError):
                price_clean = 0.0
        price = price_clean
        triggered_events: List[TriggerEvent] = []

        # 1. Kalman trigger
        kalman = self._get_kalman(symbol)
        k_out, k_event = kalman.evaluate(symbol, now_ts, price, epsilon)
        if k_event:
            triggered_events.append(k_event)

        # 2. Gradient trigger
        if gradient is not None and epsilon is not None:
            grad = self._get_gradient(symbol)
            _, g_event = grad.evaluate(symbol, price, gradient, epsilon, ts=now_ts)
            if g_event:
                triggered_events.append(g_event)

        # 3. External Liquidation Capitulation Trigger
        if liquidation_summary is not None:
            l_res = self.liquidation_trigger.evaluate(symbol, price, liquidation_summary, now=now_ts)
            l_event = l_res[1] if isinstance(l_res, tuple) else l_res
            if l_event:
                triggered_events.append(l_event)

        # 4. Sentiment Contrarian Trigger (Fear & Greed)
        if fear_greed_snapshot is not None:
            _, s_event = self.sentiment_trigger.evaluate(symbol, price, fear_greed_snapshot, ts=now_ts)
            if s_event:
                triggered_events.append(s_event)

        # 5. Market Breadth Trigger
        if market_breadth_snapshot is not None:
            _, b_event = self.breadth_trigger.evaluate(
                symbol,
                price,
                market_breadth_snapshot,
                asset_24h_change_pct=asset_24h_change_pct,
                ts=now_ts,
            )
            if b_event:
                triggered_events.append(b_event)

        # Calculate volatility if prices given
        vol1h: Optional[float] = None
        sr_sec = 60.0
        if sample_rate_sec is not None and not isinstance(sample_rate_sec, bool):
            try:
                sr_f = float(sample_rate_sec)
                if math.isfinite(sr_f) and sr_f > 0.0:
                    sr_sec = sr_f
            except (TypeError, ValueError, OverflowError):
                sr_sec = 60.0
        if prices_for_vol is not None and not isinstance(prices_for_vol, (bool, str, bytes)):
            vol1h = calculate_volatility_1h(prices_for_vol, sr_sec)

        # Select primary active trigger (prioritize highest strength among executable triggers)
        executable_triggers = [t for t in triggered_events if t.is_executable()]
        if executable_triggers:
            active_trigger = max(
                executable_triggers,
                key=lambda t: (
                    float(t.strength)
                    if isinstance(getattr(t, "strength", None), (int, float))
                    and not isinstance(getattr(t, "strength", None), bool)
                    and math.isfinite(float(t.strength))
                    else 0.0
                ),
            )
        else:
            active_trigger = triggered_events[0] if triggered_events else None
        raw_side = getattr(active_trigger.side, "value", active_trigger.side) if active_trigger else "HOLD"
        side = str(raw_side or "HOLD").strip().upper() if not isinstance(raw_side, bool) else "HOLD"
        if side in ("B", "BID", "LONG"):
            side = "BUY"
        elif side in ("S", "A", "ASK", "SHORT"):
            side = "SELL"

        # Evaluate guards against active side
        if active_trigger and active_trigger.is_executable():
            guard_dec = self.evaluate_guards(
                symbol,
                side,
                price,
                gradient=gradient,
                epsilon=epsilon,
                trend_duration_seconds=trend_duration_seconds,
                price_history=price_history,
                volatility_1h_pct=vol1h,
                liquidation_summary=liquidation_summary,
                derivatives_telemetry=derivatives_telemetry,
                fear_greed_snapshot=fear_greed_snapshot,
                market_breadth_snapshot=market_breadth_snapshot,
                geopolitical_assessment=geopolitical_assessment,
                asset_24h_change_pct=asset_24h_change_pct,
                qty=qty,
                notional_eur=notional_eur,
                now=now_ts,
                bypass_guards=bypass_guards,
            )
        else:
            guard_dec = GuardDecision.allow("NoActiveOrder", "idle")

        can_exec = bool(active_trigger and active_trigger.is_executable() and guard_dec.allowed)
        raw_sc = getattr(guard_dec, "suggested_scale", 1.0)
        sc_val = 1.0
        if raw_sc is not None and not isinstance(raw_sc, bool):
            try:
                sf = float(raw_sc)
                if math.isfinite(sf) and 1.0 < sf <= 100.0:
                    sf = sf / 100.0
                sc_val = min(1.0, max(0.0, sf)) if math.isfinite(sf) else 0.0
            except (TypeError, ValueError, OverflowError):
                sc_val = 0.0
        else:
            sc_val = 0.0
        scale = (1.0 if side == "SELL" else sc_val) if can_exec else 0.0
        if scale <= 0.0:
            can_exec = False
            scale = 0.0

        return MarketIntelligenceEvaluation(
            symbol=symbol,
            price=price,
            ts=now_ts,
            triggers=tuple(triggered_events),
            active_trigger=active_trigger,
            guard_decision=guard_dec,
            can_execute=can_exec,
            effective_scale=scale,
            metrics={
                "kalman": k_out,
                "volatility_1h": vol1h,
                "gradient": gradient,
                "epsilon": epsilon,
                "fear_greed": (
                    fear_greed_snapshot.get("value")
                    if isinstance(fear_greed_snapshot, dict)
                    else getattr(fear_greed_snapshot, "value", None)
                ) if fear_greed_snapshot else None,
                "market_regime": (
                    market_breadth_snapshot.get("regime")
                    if isinstance(market_breadth_snapshot, dict)
                    else getattr(market_breadth_snapshot, "regime", None)
                ) if market_breadth_snapshot else None,
                "geopolitical_threat": (
                    geopolitical_assessment.get("threat_level")
                    if isinstance(geopolitical_assessment, dict)
                    else getattr(geopolitical_assessment, "threat_level", None)
                ) if geopolitical_assessment else None,
            },
        )
