"""Composite market intelligence coordinator orchestrating triggers and guards across all pillars."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
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
        gemini_min_notional_eur: float = 1000.0,
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
        if symbol not in self.kalman_triggers:
            self.kalman_triggers[symbol] = KalmanTrendTrigger()
        return self.kalman_triggers[symbol]

    def _get_gradient(self, symbol: str) -> LinearGradientTrigger:
        if symbol not in self.gradient_triggers:
            self.gradient_triggers[symbol] = LinearGradientTrigger()
        return self.gradient_triggers[symbol]

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
    ) -> GuardDecision:
        """Run all protection guards (brakes) across all pillars in sequence.

        Returns the first blocking/downscaling decision or an approved allow decision.
        """
        # 1. Noise Floor Guard
        if gradient is not None and epsilon is not None:
            noise_dec = self.noise_guard.check(symbol, side, gradient, epsilon)
            if not noise_dec.allowed:
                return noise_dec

        # 2. Parabolic Surge Guard (Anti-FOMO)
        parabolic_dec = self.parabolic_guard.check(
            symbol, side, price, price_history=price_history,
            volatility_1h_pct=volatility_1h_pct, now=now,
        )
        if not parabolic_dec.allowed:
            return parabolic_dec

        # 3. Weibull Trend Exhaustion Guard
        if trend_duration_seconds > 0:
            surv = get_trend_survival_metrics(symbol, trend_duration_seconds)
            exh_dec = self.exhaustion_guard.check(
                symbol, side, trend_duration_seconds,
                p90_days=surv.get("p90_days"),
                median_days=surv.get("median_days"),
            )
            if not exh_dec.allowed or exh_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return exh_dec

        # 4. External Liquidation Cascade Guard (Active Falling Knife)
        if liquidation_summary is not None:
            cascade_dec = self.cascade_guard.check(symbol, side, summary=liquidation_summary, now=now)
            if not cascade_dec.allowed:
                return cascade_dec

        # 5. External Funding Rate Crowding Guard
        if derivatives_telemetry is not None:
            fund_dec = self.funding_guard.check(symbol, side, telemetry=derivatives_telemetry)
            if not fund_dec.allowed or fund_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return fund_dec

        # 6. Sentiment Extreme Greed Guard (Anti-Euphoria / Anti-Top Buying)
        if fear_greed_snapshot is not None:
            greed_dec = self.greed_guard.check(symbol, side, fear_greed_snapshot)
            if not greed_dec.allowed or greed_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return greed_dec

        # 7. Sentiment Panic Washout Guard (Falling Knife / Market-Wide Panic)
        if market_breadth_snapshot is not None or fear_greed_snapshot is not None:
            panic_dec = self.panic_guard.check(
                symbol,
                side,
                breadth_snapshot=market_breadth_snapshot,
                fear_greed_snapshot=fear_greed_snapshot,
                asset_24h_change_pct=asset_24h_change_pct,
            )
            if not panic_dec.allowed or panic_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return panic_dec

        # 8. Google Gemini High-Stake Guard (> 1000 EUR purchases)
        computed_notional = notional_eur
        if computed_notional is None and qty is not None and price > 0:
            computed_notional = price * qty
        if computed_notional is not None and computed_notional >= self.gemini_guard.min_notional_eur:
            gemini_dec = self.gemini_guard.check(
                symbol,
                side,
                price,
                qty if qty is not None else 1.0,
                notional_eur=computed_notional,
                telemetry={
                    "fear_greed": fear_greed_snapshot.value if fear_greed_snapshot else None,
                    "breadth_regime": market_breadth_snapshot.regime if market_breadth_snapshot else None,
                },
            )
            if not gemini_dec.allowed or gemini_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return gemini_dec

        # 9. Geopolitical & Energy Shock Guard (Black Swan Shield)
        if geopolitical_assessment is not None:
            geo_dec = self.geopolitical_guard.check(symbol, side, geopolitical_assessment)
            if not geo_dec.allowed or geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                return geo_dec

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
    ) -> MarketIntelligenceEvaluation:
        """Evaluate directional triggers and validate against all guards."""
        now_ts = ts if ts is not None else time.time()
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
            _, l_event = self.liquidation_trigger.evaluate(symbol, price, liquidation_summary, now=now_ts)
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
        if prices_for_vol and len(prices_for_vol) >= 20:
            vol1h = calculate_volatility_1h(prices_for_vol, sample_rate_sec)

        # Select primary active trigger (prioritize highest strength or first actionable)
        active_trigger = triggered_events[0] if triggered_events else None
        side = active_trigger.side.value if active_trigger else "HOLD"

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
            )
        else:
            guard_dec = GuardDecision.allow("NoActiveOrder", "idle")

        can_exec = bool(active_trigger and active_trigger.is_executable() and guard_dec.allowed)
        scale = guard_dec.suggested_scale if can_exec else 0.0

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
                "fear_greed": fear_greed_snapshot.value if fear_greed_snapshot else None,
                "market_regime": market_breadth_snapshot.regime if market_breadth_snapshot else None,
                "geopolitical_threat": geopolitical_assessment.threat_level if geopolitical_assessment else None,
            },
        )
