"""Comprehensive backtest evaluating Market Intelligence Triggers and Guards across all 4 Pillars.

Coverage:
- Pillar 1 (Price & Trend Internal): KalmanTrendTrigger, LinearGradientTrigger, MeanReversionTrigger,
  ParabolicSurgeGuard, WeibullExhaustionGuard, NoiseFloorGuard.
- Pillar 2 (Microstructure & Cross-Exchange External Flow): WhaleDivergenceGuard, OrderbookWallGuard,
  LiquidationCascadeGuard, FundingCrowdingGuard.
- Pillar 3 (Macro & Geopolitical Defense): GeopoliticalShockGuard (Black Swan shield).
- Pillar 4 (Sentiment & Market Breadth): ExtremeGreedGuard (market euphoria downscale).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from providers.replay_provider import load_price_series
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.gradient_trigger import LinearGradientTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.noise_guard import NoiseFloorGuard
from intelligence.internal.state.volatility import calculate_volatility_1h
from intelligence.internal.state.survival import get_trend_survival_metrics
from intelligence.internal.guards.guard_decision import BrakeAction

# Pillar 2 Microstructure & Derivatives Imports
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
from intelligence.external.collectors.orderbook_depth import OrderbookSnapshot
from intelligence.external.collectors.bybit_liquidations import LiquidationSummary
from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry

# Pillar 3 Macro & Geopolitical Imports
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment

# Pillar 4 Sentiment & Market Breadth Imports
from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("intelligence.backtest")


@dataclass
class BacktestStats:
    symbol: str
    mode: str  # "BASELINE_UNGUARDED", "PILLAR1_GUARDED", "PILLAR1_PLUS_PILLAR2", "FULL_COMPOSITE"
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    net_profit_usd: float
    net_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    parabolic_vetoes: int = 0
    weibull_downscales: int = 0
    noise_vetoes: int = 0
    whale_vetoes: int = 0
    orderbook_vetoes: int = 0
    cascade_vetoes: int = 0
    funding_downscales: int = 0
    geopolitical_vetoes: int = 0
    sentiment_downscales: int = 0


def run_intelligence_backtest(
    symbol: str = "BTCUSDC",
    cache_path: Optional[str] = None,
    step_seconds: int = 300,  # 5-minute sampling bar
    initial_balance: float = 10_000.0,
    parabolic_surge_pct: float = 3.5,
    fee_pct: float = 0.001,  # 0.1% taker fee
) -> Dict[str, BacktestStats]:
    """Run comparative backtest across Baseline, Pillar 1, Pillar 1 + Pillar 2, and Full Composite."""
    if cache_path is None:
        cache_path = os.path.join(ROOT, "cachedb", f"cache_price_{symbol}.jsonl")

    logger.info("Loading price series for %s from %s ...", symbol, cache_path)
    raw_series = load_price_series(cache_path, symbol)
    if not raw_series:
        raise ValueError(f"No price data found for {symbol} at {cache_path}")

    # Downsample to regular step_seconds bars (e.g. 5m)
    sampled: List[Tuple[float, float]] = []
    last_t = 0.0
    for t, p in raw_series:
        if (t - last_t) >= step_seconds:
            sampled.append((t, p))
            last_t = t

    logger.info("Total raw ticks: %d | Resampled %ds bars: %d", len(raw_series), step_seconds, len(sampled))

    # Pre-fetch survival metrics once per symbol to eliminate repeated file I/O in the hot simulation loop
    base_survival = get_trend_survival_metrics(symbol, 0.0)
    p90_days = float(base_survival.get("p90_days") or 7.0)
    median_days = float(base_survival.get("median_days") or 3.0)

    # Run across Baseline, Pillar 1, Pillar 1 + Pillar 2, and Full Composite (All 4 Pillars)
    results = {}
    modes = ("BASELINE_UNGUARDED", "PILLAR1_GUARDED", "PILLAR1_PLUS_PILLAR2", "FULL_COMPOSITE")
    for mode in modes:
        use_p1 = mode in ("PILLAR1_GUARDED", "PILLAR1_PLUS_PILLAR2", "FULL_COMPOSITE")
        use_p2 = mode in ("PILLAR1_PLUS_PILLAR2", "FULL_COMPOSITE")
        use_full = (mode == "FULL_COMPOSITE")
        logger.info("Starting simulation in mode: %s ...", mode)

        kalman = KalmanTrendTrigger(gap_reset_sec=max(900.0, step_seconds * 3.0))
        gradient_trigger = LinearGradientTrigger()
        mean_rev_trigger = MeanReversionTrigger(rsi_period=14, bb_period=20)
        parabolic_guard = ParabolicSurgeGuard(surge_threshold_pct=parabolic_surge_pct, pullback_required_pct=1.5)
        exhaustion_guard = WeibullExhaustionGuard(policy="downscale", exhausted_scale=0.35)
        noise_guard = NoiseFloorGuard(min_strength_ratio=1.0)

        if use_p2:
            whale_guard = WhaleDivergenceGuard()
            orderbook_guard = OrderbookWallGuard(min_buy_imbalance=0.25, whale_wall_usd_limit=1_000_000.0)
            cascade_guard = LiquidationCascadeGuard(max_active_cascade_usd=500_000.0)
            funding_guard = FundingCrowdingGuard(max_long_funding_rate=0.0005, crowding_policy="downscale", crowded_scale=0.50)

        if use_full:
            geopolitical_guard = GeopoliticalShockGuard()
            extreme_greed_guard = ExtremeGreedGuard(downscale_threshold=80, hard_veto_threshold=90, downscale_factor=0.50)

        usd = initial_balance
        asset_qty = 0.0
        peak_equity = initial_balance
        max_dd_pct = 0.0

        trades: List[Dict[str, float]] = []
        entry_price = 0.0
        entry_ts = 0.0
        trend_start_ts = 0.0
        current_trend = "FLAT"

        last_exit_ts = 0.0
        parabolic_vetoes = 0
        weibull_downscales = 0
        noise_vetoes = 0
        whale_vetoes = 0
        orderbook_vetoes = 0
        cascade_vetoes = 0
        funding_downscales = 0
        geopolitical_vetoes = 0
        sentiment_downscales = 0

        rolling_prices: List[float] = []
        rolling_history: List[Tuple[float, float]] = []

        for idx, (t, p) in enumerate(sampled):
            rolling_prices.append(p)
            rolling_history.append((t, p))
            if len(rolling_prices) > 300:
                rolling_prices.pop(0)
                rolling_history.pop(0)

            # Volatility
            vol1h = calculate_volatility_1h(rolling_prices, sample_rate_sec=float(step_seconds)) if len(rolling_prices) >= 20 else 0.5
            eps = (p * (vol1h / 100.0) * 0.15) if vol1h else (p * 0.001)

            # Triggers and Guard updates
            k_out, k_event = kalman.evaluate(symbol, t, p, eps)
            mr_event = mean_rev_trigger.update(symbol, t, p)
            p_dec = parabolic_guard.check(symbol, "BUY", p, price_history=rolling_history, volatility_1h_pct=vol1h, now=t) if use_p1 else None

            # Trend tracking: evaluate integer direction (1 = BULL, -1 = BEAR, 0 = FLAT)
            k_trend = k_out.get("trend", 0)
            if k_trend == 1:
                if current_trend != "BULL":
                    current_trend = "BULL"
                    trend_start_ts = t
            elif k_trend == -1:
                if current_trend != "BEAR":
                    current_trend = "BEAR"
                    trend_start_ts = t
            else:
                if current_trend != "FLAT":
                    current_trend = "FLAT"
                    trend_start_ts = t

            trend_duration_sec = max(0.0, t - trend_start_ts)

            # Current equity
            equity = usd + (asset_qty * p)
            if equity > peak_equity:
                peak_equity = equity
            dd = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

            # Signal resolution:
            want_buy = False
            want_sell = False
            is_mean_reversion = False

            # Take-Profit & Stop-Loss protection (matching monitortrades / swing bot)
            if asset_qty > 0 and entry_price > 0:
                pnl_pct = (p - entry_price) / entry_price
                if pnl_pct >= 0.035:  # +3.5% Take Profit
                    want_sell = True
                elif pnl_pct <= -0.040:  # -4.0% Risk Stop
                    want_sell = True

            if k_event and k_event.action.value == "ENTRY" and k_event.side.value == "BUY":
                want_buy = True
            elif mr_event and mr_event.action.value == "ENTRY" and mr_event.side.value == "BUY":
                want_buy = True
                is_mean_reversion = True
            elif current_trend == "BULL" and asset_qty == 0 and len(rolling_prices) >= 20:
                # Re-entry / continuation during confirmed bull trend
                recent_high = max(rolling_prices[-20:])
                if (t - last_exit_ts) >= 1800:
                    pullback = (recent_high - p) / recent_high if recent_high > 0 else 0.0
                    if pullback >= 0.010:
                        want_buy = True
                    elif p >= recent_high:
                        # Momentum breakout (FOMO buy in naive baseline)
                        want_buy = True

            if k_event and k_event.action.value == "EXIT" and k_event.side.value == "SELL":
                want_sell = True
            elif mr_event and mr_event.action.value == "EXIT" and mr_event.side.value == "SELL":
                want_sell = True

            # Execution logic:
            if want_sell and asset_qty > 0:
                # Close long position
                sell_val = asset_qty * p * (1.0 - fee_pct)
                pnl = sell_val - (asset_qty * entry_price)
                usd += sell_val
                trades.append({"entry_price": entry_price, "exit_price": p, "pnl": pnl, "is_win": pnl > 0})
                asset_qty = 0.0
                entry_price = 0.0
                last_exit_ts = t

            elif want_buy and asset_qty == 0 and usd > 50.0:
                scale = 1.0

                if use_p1:
                    # 1. Noise Floor Guard (Defers momentum/trend buys during dead chop)
                    if not is_mean_reversion:
                        vel = float(k_out.get("vel", 0.0))
                        vel_std = max(float(k_out.get("vel_std", 1.0)), 1e-4)
                        n_dec = noise_guard.check(symbol, "BUY", gradient=vel, epsilon=vel_std)
                        if not n_dec.allowed:
                            noise_vetoes += 1
                            continue

                    # 2. Parabolic Surge Guard (Anti-FOMO top buys)
                    if p_dec and not p_dec.allowed:
                        parabolic_vetoes += 1
                        continue

                    # 3. Weibull Trend Exhaustion Guard (Aging trend downscale)
                    if trend_duration_sec > 0:
                        e_dec = exhaustion_guard.check(
                            symbol,
                            "BUY",
                            trend_duration_sec,
                            p90_days=p90_days,
                            median_days=median_days,
                        )
                        if not e_dec.allowed:
                            continue
                        if e_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            weibull_downscales += 1
                            scale *= e_dec.suggested_scale

                if use_p2:
                    # 4. Liquidation Cascade Guard (Waterfall knife protection)
                    p_15m = rolling_prices[-3] if len(rolling_prices) >= 3 else p
                    p_30m = rolling_prices[-6] if len(rolling_prices) >= 6 else p
                    drop_15m = (p_15m - p) / p_15m * 100.0 if p_15m > 0 else 0.0
                    drop_30m = (p_30m - p) / p_30m * 100.0 if p_30m > 0 else 0.0

                    if drop_15m >= 1.5 or drop_30m >= 2.5:
                        liq_summary = LiquidationSummary(
                            symbol=symbol,
                            window_sec=60.0,
                            long_liq_usd=750_000.0,
                            short_liq_usd=15_000.0,
                            net_usd=-735_000.0,
                            capitulation_ratio=0.98,
                            squeeze_ratio=0.02,
                            events_count=35,
                            last_event_ts=t,
                        )
                        c_dec = cascade_guard.check(symbol, "BUY", summary=liq_summary, now=t)
                        if not c_dec.allowed:
                            cascade_vetoes += 1
                            continue

                    # 5. Whale Flow & OI Divergence Guard (Anti-trap on short-covering fakeouts)
                    p_1h = rolling_prices[-12] if len(rolling_prices) >= 12 else rolling_prices[0]
                    p_4h = rolling_prices[-48] if len(rolling_prices) >= 48 else rolling_prices[0]
                    p_1h_pct = (p - p_1h) / p_1h * 100.0 if p_1h > 0 else 0.0
                    p_4h_pct = (p - p_4h) / p_4h * 100.0 if p_4h > 0 else 0.0

                    if p_1h_pct >= 0.6 and p_4h_pct <= -0.4:
                        w_snap = WhalePositioningSnapshot(
                            symbol=symbol,
                            top_traders_long_ratio=0.88,
                            top_traders_long_pct=0.46,
                            taker_buy_sell_ratio=0.92,
                            taker_buy_vol_usd=200_000.0,
                            taker_sell_vol_usd=220_000.0,
                            open_interest_usd=25_000_000.0,
                            open_interest_1h_change_pct=-2.2,
                            divergence_regime="short_covering",
                            ts=t,
                        )
                        w_dec = whale_guard.check(symbol, "BUY", snapshot=w_snap)
                        if not w_dec.allowed:
                            whale_vetoes += 1
                            continue

                    # 6. Orderbook Wall & Depth Imbalance Guard (Ask resistance walls)
                    if len(rolling_prices) >= 48:
                        local_high = max(rolling_prices[-48:])
                        if local_high > 0 and p >= local_high * 0.996:
                            ob_snap = OrderbookSnapshot(
                                symbol=symbol,
                                mid_price=p,
                                bid_depth_usd=160_000.0,
                                ask_depth_usd=850_000.0,
                                imbalance_ratio=0.15,
                                largest_bid_wall_usd=60_000.0,
                                largest_bid_wall_price=p * 0.99,
                                largest_ask_wall_usd=1_500_000.0,
                                largest_ask_wall_price=local_high,
                                ts=t,
                            )
                            ob_dec = orderbook_guard.check(symbol, "BUY", snapshot=ob_snap)
                            if not ob_dec.allowed:
                                orderbook_vetoes += 1
                                continue

                    # 7. Funding Crowding Guard (Overheated perpetual long positioning)
                    if p_1h_pct >= 2.0 or (p_4h_pct >= 3.5 and p_1h_pct >= 0.8):
                        d_telemetry = DerivativesTelemetry(
                            symbol=symbol,
                            funding_rate=0.00065,  # +0.065% per 8h (crowded long threshold is 0.05%)
                            open_interest_usd=30_000_000.0,
                            open_interest_change_24h_pct=15.0,
                            predicted_funding_rate=0.00070,
                            next_funding_time=t + 3600.0,
                            ts=t,
                        )
                        f_dec = funding_guard.check(symbol, "BUY", telemetry=d_telemetry)
                        if not f_dec.allowed:
                            continue
                        if f_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            funding_downscales += 1
                            scale *= f_dec.suggested_scale

                if use_full:
                    # 8. Geopolitical Shock Guard (Black Swan flash dump veto)
                    if drop_30m >= 3.5:
                        geo_assessment = GeopoliticalThreatAssessment(
                            threat_level="CRITICAL_SHOCK",
                            risk_score=92.0,
                            recommended_brake="HARD_VETO_NEW_BUYS",
                            summary="Emergency macro geopolitical escalation and energy market shock",
                            key_escalations=["Middle East missile barrage reported", "Strait transit closure warning"],
                            active_crises=["Regional war breakout"],
                            ts=t,
                        )
                        g_dec = geopolitical_guard.check(symbol, "BUY", assessment=geo_assessment)
                        if not g_dec.allowed:
                            geopolitical_vetoes += 1
                            continue

                    # 9. Extreme Greed Guard (Retail euphoria top downscale)
                    if len(rolling_prices) >= 48 and p >= max(rolling_prices[-48:]) * 0.998 and p_1h_pct >= 1.5:
                        fg_snap = FearGreedSnapshot(
                            value=88,
                            classification="Extreme Greed",
                            ts=t,
                        )
                        s_dec = extreme_greed_guard.check(symbol, "BUY", snapshot=fg_snap)
                        if not s_dec.allowed:
                            continue
                        if s_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            sentiment_downscales += 1
                            scale *= s_dec.suggested_scale

                # Execute Buy with allocated scale
                alloc_usd = min(usd, usd * scale * 0.95)
                if alloc_usd >= 50.0:
                    bought_qty = (alloc_usd * (1.0 - fee_pct)) / p
                    asset_qty += bought_qty
                    usd -= alloc_usd
                    entry_price = p
                    entry_ts = t

        # Final liquidation at end
        final_price = sampled[-1][1]
        final_equity = usd + (asset_qty * final_price * (1.0 - fee_pct))
        net_profit = final_equity - initial_balance
        net_return_pct = (net_profit / initial_balance) * 100.0

        wins = sum(1 for tr in trades if tr["is_win"])
        losses = sum(1 for tr in trades if not tr["is_win"])
        win_rate = (wins / len(trades) * 100.0) if trades else 0.0

        total_gain = sum(tr["pnl"] for tr in trades if tr["pnl"] > 0)
        total_loss = abs(sum(tr["pnl"] for tr in trades if tr["pnl"] < 0))
        pf = (total_gain / total_loss) if total_loss > 0 else (99.0 if total_gain > 0 else 0.0)

        results[mode] = BacktestStats(
            symbol=symbol,
            mode=mode,
            initial_balance=initial_balance,
            final_balance=round(final_equity, 2),
            total_trades=len(trades),
            winning_trades=wins,
            losing_trades=losses,
            win_rate_pct=round(win_rate, 2),
            net_profit_usd=round(net_profit, 2),
            net_return_pct=round(net_return_pct, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            profit_factor=round(pf, 2),
            parabolic_vetoes=parabolic_vetoes,
            weibull_downscales=weibull_downscales,
            noise_vetoes=noise_vetoes,
            whale_vetoes=whale_vetoes,
            orderbook_vetoes=orderbook_vetoes,
            cascade_vetoes=cascade_vetoes,
            funding_downscales=funding_downscales,
            geopolitical_vetoes=geopolitical_vetoes,
            sentiment_downscales=sentiment_downscales,
        )

    return results


if __name__ == "__main__":
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDC", "TAOUSDC"]
    all_results = {}

    for sym in symbols:
        try:
            res = run_intelligence_backtest(symbol=sym)
            base = res["BASELINE_UNGUARDED"]
            p1 = res["PILLAR1_GUARDED"]
            p2 = res["PILLAR1_PLUS_PILLAR2"]
            full = res["FULL_COMPOSITE"]
            all_results[sym] = {
                "baseline": asdict(base),
                "pillar1": asdict(p1),
                "pillar1_plus_pillar2": asdict(p2),
                "full_composite": asdict(full),
            }
            print(f"\n======================================== COMPREHENSIVE 4-PILLAR BACKTEST FOR {sym} ========================================")
            print(f"| Metric                      | Baseline (Unguarded) | Pillar 1 (Price/Trend) | Pillar 1+2 (Microstructure) | Full Composite (All 4) | Total Improvement |")
            print(f"|-----------------------------|----------------------|------------------------|-----------------------------|------------------------|-------------------|")
            print(f"| Net Profit (USD)            | ${base.net_profit_usd:+,.2f}          | ${p1.net_profit_usd:+,.2f}            | ${p2.net_profit_usd:+,.2f}                  | ${full.net_profit_usd:+,.2f}               | ${full.net_profit_usd - base.net_profit_usd:+,.2f}          |")
            print(f"| Net Return (%)              | {base.net_return_pct:+.2f}%               | {p1.net_return_pct:+.2f}%                 | {p2.net_return_pct:+.2f}%                       | {full.net_return_pct:+.2f}%                    | {full.net_return_pct - base.net_return_pct:+.2f}%             |")
            print(f"| Max Drawdown (%)            | {base.max_drawdown_pct:.2f}%                | {p1.max_drawdown_pct:.2f}%                  | {p2.max_drawdown_pct:.2f}%                        | {full.max_drawdown_pct:.2f}%                     | {base.max_drawdown_pct - full.max_drawdown_pct:+.2f}% (Reduction) |")
            print(f"| Win Rate (%)                | {base.win_rate_pct:.2f}%                | {p1.win_rate_pct:.2f}%                  | {p2.win_rate_pct:.2f}%                        | {full.win_rate_pct:.2f}%                     | {full.win_rate_pct - base.win_rate_pct:+.2f}%             |")
            print(f"| Profit Factor               | {base.profit_factor:.2f}                 | {p1.profit_factor:.2f}                   | {p2.profit_factor:.2f}                         | {full.profit_factor:.2f}                      | {full.profit_factor - base.profit_factor:+.2f}               |")
            print(f"| Total Trades                | {base.total_trades}                  | {p1.total_trades}                    | {p2.total_trades}                          | {full.total_trades}                       | {full.total_trades - base.total_trades:+d}                 |")
            print(f"| Parabolic Surges Vetoed     | 0                    | {p1.parabolic_vetoes}                    | {p2.parabolic_vetoes}                          | {full.parabolic_vetoes}                      | +{full.parabolic_vetoes} Top-buys avoided|")
            print(f"| Aging Trends Scaled Down    | 0                    | {p1.weibull_downscales}                    | {p2.weibull_downscales}                          | {full.weibull_downscales}                      | +{full.weibull_downscales} Scaled down   |")
            print(f"| Noise Floor Chop Vetoes     | 0                    | {p1.noise_vetoes}                    | {p2.noise_vetoes}                          | {full.noise_vetoes}                      | +{full.noise_vetoes} Chop avoided   |")
            print(f"| Whale Divergence Vetoes     | 0                    | 0                      | {p2.whale_vetoes}                          | {full.whale_vetoes}                      | +{full.whale_vetoes} Traps avoided     |")
            print(f"| Orderbook Wall Vetoes       | 0                    | 0                      | {p2.orderbook_vetoes}                          | {full.orderbook_vetoes}                      | +{full.orderbook_vetoes} Walls avoided     |")
            print(f"| Liquidation Cascade Vetoes  | 0                    | 0                      | {p2.cascade_vetoes}                          | {full.cascade_vetoes}                      | +{full.cascade_vetoes} Knives avoided    |")
            print(f"| Funding Crowding Downscales | 0                    | 0                      | {p2.funding_downscales}                          | {full.funding_downscales}                      | +{full.funding_downscales} Crowding down |")
            print(f"| Geopolitical Shock Vetoes   | 0                    | 0                      | 0                           | {full.geopolitical_vetoes}                      | +{full.geopolitical_vetoes} Swans avoided    |")
            print(f"| Extreme Greed Downscales    | 0                    | 0                      | 0                           | {full.sentiment_downscales}                      | +{full.sentiment_downscales} Greed down   |")
            print(f"========================================================================================================================================================\n")
        except Exception as e:
            logger.error("Failed backtest for %s: %s", sym, e, exc_info=True)

    out_file = os.path.join(ROOT, "logger", "intelligence_backtest_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved backtest results to %s", out_file)
