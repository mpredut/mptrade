"""Comprehensive backtest evaluating Market Intelligence Triggers and Guards on historical tick data."""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("intelligence.backtest")


@dataclass
class BacktestStats:
    symbol: str
    mode: str  # "BASELINE_UNGUARDED" vs "INTELLIGENCE_GUARDED"
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


def run_intelligence_backtest(
    symbol: str = "BTCUSDC",
    cache_path: Optional[str] = None,
    step_seconds: int = 300,  # 5-minute sampling bar
    initial_balance: float = 10_000.0,
    parabolic_surge_pct: float = 8.0,
    fee_pct: float = 0.001,  # 0.1% taker fee
) -> Tuple[BacktestStats, BacktestStats]:
    """Run comparative backtest between Unguarded baseline and Intelligence-Guarded strategies."""
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

    # Run for both modes
    results = {}
    for mode in ("BASELINE_UNGUARDED", "INTELLIGENCE_GUARDED"):
        use_guards = (mode == "INTELLIGENCE_GUARDED")
        logger.info("Starting simulation in mode: %s ...", mode)

        kalman = KalmanTrendTrigger(q=1e-4, r=1e-2)
        gradient_trigger = LinearGradientTrigger()
        mean_rev_trigger = MeanReversionTrigger(rsi_period=14, bollinger_period=20)
        parabolic_guard = ParabolicSurgeGuard(surge_threshold_pct=parabolic_surge_pct)
        exhaustion_guard = WeibullExhaustionGuard(policy="downscale", exhausted_scale=0.35)
        noise_guard = NoiseFloorGuard(min_strength_ratio=1.0)

        usd = initial_balance
        asset_qty = 0.0
        peak_equity = initial_balance
        max_dd_pct = 0.0

        trades: List[Dict[str, float]] = []
        entry_price = 0.0
        entry_ts = 0.0
        trend_start_ts = 0.0
        current_trend = "FLAT"

        parabolic_vetoes = 0
        weibull_downscales = 0
        noise_vetoes = 0

        rolling_prices: List[float] = []
        rolling_history: List[Tuple[float, float]] = []

        for idx, (t, p) in enumerate(sampled):
            rolling_prices.append(p)
            rolling_history.append((t, p))
            if len(rolling_prices) > 100:
                rolling_prices.pop(0)
                rolling_history.pop(0)

            # Volatility
            vol1h = calculate_volatility_1h(rolling_prices, sample_rate_sec=float(step_seconds)) if len(rolling_prices) >= 20 else 0.5
            eps = (p * (vol1h / 100.0) * 0.15) if vol1h else (p * 0.001)

            # Triggers
            k_out, k_event = kalman.evaluate(symbol, t, p, eps)
            mr_sig, mr_event = mean_rev_trigger.evaluate(symbol, rolling_prices, t=t) if len(rolling_prices) >= 30 else (None, None)

            # Trend tracking
            if k_out == "BULL":
                if current_trend != "BULL":
                    current_trend = "BULL"
                    trend_start_ts = t
            elif k_out == "BEAR":
                if current_trend != "BEAR":
                    current_trend = "BEAR"
                    trend_start_ts = t
            else:
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

            if k_event and k_event.action.value == "ENTRY" and k_event.side.value == "BUY":
                want_buy = True
            elif mr_event and mr_event.side.value == "BUY":
                want_buy = True

            if k_event and k_event.action.value == "EXIT" and k_event.side.value == "SELL":
                want_sell = True
            elif mr_event and mr_event.side.value == "SELL":
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

            elif want_buy and asset_qty == 0 and usd > 50.0:
                scale = 1.0

                if use_guards:
                    # 1. Parabolic Surge Guard (Anti-FOMO)
                    p_dec = parabolic_guard.check(symbol, "BUY", p, price_history=rolling_history, volatility_1h_pct=vol1h, now=t)
                    if not p_dec.allowed:
                        parabolic_vetoes += 1
                        continue

                    # 2. Weibull Trend Exhaustion Guard
                    if trend_duration_sec > 0:
                        surv = get_trend_survival_metrics(symbol, trend_duration_sec)
                        e_dec = exhaustion_guard.check(symbol, "BUY", trend_duration_sec, p90_days=surv.get("p90_days"), median_days=surv.get("median_days"))
                        if not e_dec.allowed:
                            continue
                        if e_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                            weibull_downscales += 1
                            scale *= e_dec.suggested_scale

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
        )

    return results["BASELINE_UNGUARDED"], results["INTELLIGENCE_GUARDED"]


if __name__ == "__main__":
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDC", "TAOUSDC"]
    all_results = {}

    for sym in symbols:
        try:
            base, guarded = run_intelligence_backtest(symbol=sym)
            all_results[sym] = {"baseline": asdict(base), "guarded": asdict(guarded)}
            print(f"\n==================== BACKTEST RESULTS FOR {sym} ====================")
            print(f"| Metric                   | Baseline (Unguarded) | Intelligence Guarded | Delta / Improvement |")
            print(f"|--------------------------|----------------------|----------------------|---------------------|")
            print(f"| Net Profit (USD)         | ${base.net_profit_usd:+,.2f}          | ${guarded.net_profit_usd:+,.2f}          | ${guarded.net_profit_usd - base.net_profit_usd:+,.2f}           |")
            print(f"| Net Return (%)           | {base.net_return_pct:+.2f}%               | {guarded.net_return_pct:+.2f}%               | {guarded.net_return_pct - base.net_return_pct:+.2f}%              |")
            print(f"| Max Drawdown (%)         | {base.max_drawdown_pct:.2f}%                | {guarded.max_drawdown_pct:.2f}%                | {base.max_drawdown_pct - guarded.max_drawdown_pct:+.2f}% (Reduction)  |")
            print(f"| Win Rate (%)             | {base.win_rate_pct:.2f}%                | {guarded.win_rate_pct:.2f}%                | {guarded.win_rate_pct - base.win_rate_pct:+.2f}%              |")
            print(f"| Profit Factor            | {base.profit_factor:.2f}                 | {guarded.profit_factor:.2f}                 | {guarded.profit_factor - base.profit_factor:+.2f}                |")
            print(f"| Total Trades             | {base.total_trades}                  | {guarded.total_trades}                  | {guarded.total_trades - base.total_trades:+d}                  |")
            print(f"| FOMO Surges Vetoed       | 0                    | {guarded.parabolic_vetoes}                  | +{guarded.parabolic_vetoes} Top-buys avoided|")
            print(f"| Aging Trends Scaled Down | 0                    | {guarded.weibull_downscales}                  | +{guarded.weibull_downscales} Downscaled   |")
            print(f"====================================================================\n")
        except Exception as e:
            logger.error("Failed backtest for %s: %s", sym, e, exc_info=True)

    out_file = os.path.join(ROOT, "logger", "intelligence_backtest_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved backtest results to %s", out_file)
