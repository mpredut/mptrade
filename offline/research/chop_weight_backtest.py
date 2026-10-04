#!/usr/bin/env python3
"""Chop Regime & Fallback Weight Research Backtest.

Evaluates trade allocation and profitability during consolidation/sideways market
regimes across different chop weight policies:
- CHOP_0.00: 0% weight (complete pause during chop/no-trend)
- CHOP_0.01: 1% micro-DCA allocation
- CHOP_0.02: 2% small-DCA allocation
- CHOP_0.03: 3% baseline default (current system parameter)
- CHOP_0.05: 5% moderate allocation
- CHOP_0.10: 10% aggressive consolidation accumulation
- PROXY_BTC: Beta proxy (uses BTC's active trend weight when altcoin is in chop)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from priceAnalysis import get_trade_weight
from providers.replay_provider import load_price_series
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.state.volatility import calculate_volatility_1h
from intelligence.internal.state.persistence import calculate_mann_kendall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chop_backtest")


@dataclass
class ChopBacktestStats:
    policy: str
    symbol: str
    initial_balance: float
    final_balance: float
    net_profit_usd: float
    net_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    total_trades: int
    chop_trades: int
    trend_trades: int
    chop_pnl_usd: float
    trend_pnl_usd: float
    avg_trade_pnl_usd: float
    avg_chop_pnl_usd: float


def detect_regime_slope(prices: np.ndarray, mk_alpha: float = 0.05) -> Tuple[str, float]:
    """Detect local trend regime from price window using linear slope and Mann-Kendall test."""
    if len(prices) < 10:
        return "CHOP", 0.0
    x = np.arange(len(prices), dtype=float)
    slope, _ = np.polyfit(x, prices, 1)
    norm_slope = slope / prices[-1] if prices[-1] > 0 else 0.0

    _, _, p_mk = calculate_mann_kendall(prices)
    if p_mk > mk_alpha:
        return "CHOP", norm_slope
    if norm_slope > 1e-4:
        return "BULL", norm_slope
    if norm_slope < -1e-4:
        return "BEAR", norm_slope
    return "CHOP", norm_slope


def run_single_simulation(
    symbol: str,
    sampled: List[Tuple[float, float]],
    btc_sampled: Optional[List[Tuple[float, float]]],
    policy: str,
    step_seconds: int = 900,
    initial_balance: float = 10_000.0,
    base_notional: float = 1_000.0,
    fee_pct: float = 0.001,
    tp_pct: float = 0.035,
    stop_pct: float = 0.040,
    T_days: int = 8,
) -> ChopBacktestStats:
    """Run a single backtest simulation under a specific chop allocation policy."""
    kalman = KalmanTrendTrigger(gap_reset_sec=max(1800.0, step_seconds * 3.0))
    mean_rev = MeanReversionTrigger(rsi_period=14, bb_period=20)

    # Build BTC lookup index for proxy policies
    btc_dict = {}
    if btc_sampled:
        btc_dict = {int(t // step_seconds): p for t, p in btc_sampled}

    usd = initial_balance
    asset_qty = 0.0
    entry_price = 0.0
    entry_regime = "CHOP"
    peak_equity = initial_balance
    max_dd_pct = 0.0

    trades: List[Dict[str, Any]] = []
    rolling_prices: List[float] = []

    current_trend = "CHOP"
    trend_start_ts = sampled[0][0]
    last_regime_check_ts = 0.0
    regime = "CHOP"

    # BTC tracking for proxy
    btc_rolling: List[float] = []
    btc_regime = "CHOP"
    btc_trend_start_ts = sampled[0][0]

    for idx, (t, p) in enumerate(sampled):
        rolling_prices.append(p)
        if len(rolling_prices) > 96:  # 24 hours of 15m bars
            rolling_prices.pop(0)

        # Track BTC for proxy if available
        btc_p = btc_dict.get(int(t // step_seconds))
        if btc_p is not None:
            btc_rolling.append(btc_p)
            if len(btc_rolling) > 96:
                btc_rolling.pop(0)

        # Check regime every 4 hours (16 bars)
        if (t - last_regime_check_ts) >= 14400.0 and len(rolling_prices) >= 24:
            last_regime_check_ts = t
            regime, _ = detect_regime_slope(np.array(rolling_prices), mk_alpha=0.05)
            if regime != current_trend:
                current_trend = regime
                trend_start_ts = t

            if len(btc_rolling) >= 24:
                b_reg, _ = detect_regime_slope(np.array(btc_rolling), mk_alpha=0.05)
                if b_reg != btc_regime:
                    btc_regime = b_reg
                    btc_trend_start_ts = t

        # Volatility & Triggers
        vol1h = calculate_volatility_1h(rolling_prices, sample_rate_sec=float(step_seconds)) if len(rolling_prices) >= 12 else 0.5
        eps = (p * (vol1h / 100.0) * 0.15) if vol1h else (p * 0.001)

        k_out, k_event = kalman.evaluate(symbol, t, p, eps)
        mr_event = mean_rev.update(symbol, t, p)

        # Track equity & drawdown
        equity = usd + (asset_qty * p)
        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        want_buy = False
        want_sell = False

        # Take Profit & Stop Loss
        if asset_qty > 0 and entry_price > 0:
            pnl_pct = (p - entry_price) / entry_price
            if pnl_pct >= tp_pct:
                want_sell = True
            elif pnl_pct <= -stop_pct:
                want_sell = True

        # Signal Generation
        if k_event and k_event.action.value == "ENTRY" and k_event.side.value == "BUY":
            want_buy = True
        elif mr_event and mr_event.action.value == "ENTRY" and mr_event.side.value == "BUY":
            want_buy = True
        elif current_trend == "BULL" and asset_qty == 0 and len(rolling_prices) >= 16:
            recent_high = max(rolling_prices[-16:])
            pullback = (recent_high - p) / recent_high if recent_high > 0 else 0.0
            if pullback >= 0.012:
                want_buy = True

        if k_event and k_event.action.value == "EXIT" and k_event.side.value == "SELL":
            want_sell = True

        # Execute SELL
        if want_sell and asset_qty > 0:
            proceeds = asset_qty * p * (1.0 - fee_pct)
            cost = asset_qty * entry_price
            pnl = proceeds - cost
            usd += proceeds
            trades.append({
                "entry_price": entry_price,
                "exit_price": p,
                "pnl": pnl,
                "regime": entry_regime,
                "is_win": pnl > 0,
            })
            asset_qty = 0.0
            entry_price = 0.0

        # Execute BUY
        elif want_buy and asset_qty == 0 and usd > 50.0:
            # Determine weight based on policy and current regime
            trend_len_days = max(0.0, (t - trend_start_ts) / 86400.0)

            if current_trend == "BULL":
                # Confirmed bull trend: Gaussian weight with Lindy plateau
                _, ws = get_trade_weight(T=T_days, trend_len=trend_len_days, trend="up", order_type="BUY", lindy_plateau=True)
                weight = float(ws[0]) if len(ws) > 0 else 0.95
                trade_regime = "TREND"
            else:
                # CHOP or BEAR regime: evaluate candidate chop policy
                trade_regime = "CHOP"
                if policy == "CHOP_0.00":
                    weight = 0.00  # Completely skip buy in chop
                elif policy == "CHOP_0.01":
                    weight = 0.01
                elif policy == "CHOP_0.02":
                    weight = 0.02
                elif policy == "CHOP_0.03":
                    weight = 0.03  # Default baseline
                elif policy == "CHOP_0.05":
                    weight = 0.05
                elif policy == "CHOP_0.10":
                    weight = 0.10
                elif policy == "PROXY_BTC":
                    if btc_regime == "BULL":
                        b_dur = max(0.0, (t - btc_trend_start_ts) / 86400.0)
                        _, b_ws = get_trade_weight(T=8, trend_len=b_dur, trend="up", order_type="BUY", lindy_plateau=True)
                        weight = float(b_ws[0]) if len(b_ws) > 0 else 0.86
                    else:
                        weight = 0.03
                else:
                    weight = 0.03

            if weight <= 0.0001:
                continue  # Skip order under 0% chop policy

            # Allocation
            trade_budget = min(usd * 0.98, base_notional * weight)
            if trade_budget >= 10.0:
                cost = trade_budget * (1.0 + fee_pct)
                if cost <= usd:
                    buy_qty = trade_budget / p
                    asset_qty += buy_qty
                    entry_price = p
                    entry_regime = trade_regime
                    usd -= cost

    # Close remaining open position at end
    if asset_qty > 0 and len(sampled) > 0:
        last_p = sampled[-1][1]
        proceeds = asset_qty * last_p * (1.0 - fee_pct)
        pnl = proceeds - (asset_qty * entry_price)
        usd += proceeds
        trades.append({
            "entry_price": entry_price,
            "exit_price": last_p,
            "pnl": pnl,
            "regime": entry_regime,
            "is_win": pnl > 0,
        })

    # Stats Calculation
    total_trades = len(trades)
    winning = [tr for tr in trades if tr["is_win"]]
    losing = [tr for tr in trades if not tr["is_win"]]
    win_rate = (len(winning) / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(tr["pnl"] for tr in winning)
    gross_loss = abs(sum(tr["pnl"] for tr in losing))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

    chop_tr = [tr for tr in trades if tr["regime"] == "CHOP"]
    trend_tr = [tr for tr in trades if tr["regime"] == "TREND"]
    chop_pnl = sum(tr["pnl"] for tr in chop_tr)
    trend_pnl = sum(tr["pnl"] for tr in trend_tr)

    net_profit = usd - initial_balance
    net_return = (net_profit / initial_balance) * 100.0
    avg_pnl = (net_profit / total_trades) if total_trades > 0 else 0.0
    avg_chop_pnl = (chop_pnl / len(chop_tr)) if len(chop_tr) > 0 else 0.0

    return ChopBacktestStats(
        policy=policy,
        symbol=symbol,
        initial_balance=initial_balance,
        final_balance=usd,
        net_profit_usd=round(net_profit, 2),
        net_return_pct=round(net_return, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        profit_factor=round(profit_factor, 2),
        win_rate_pct=round(win_rate, 2),
        total_trades=total_trades,
        chop_trades=len(chop_tr),
        trend_trades=len(trend_tr),
        chop_pnl_usd=round(chop_pnl, 2),
        trend_pnl_usd=round(trend_pnl, 2),
        avg_trade_pnl_usd=round(avg_pnl, 2),
        avg_chop_pnl_usd=round(avg_chop_pnl, 2),
    )


def run_chop_benchmark(
    symbols: List[str] = ("TAOUSDC", "BTCUSDC"),
    step_seconds: int = 900,
) -> Dict[str, List[ChopBacktestStats]]:
    """Run full comparative benchmark across all chop weight policies for given symbols."""
    # Policies to test
    policies = [
        "CHOP_0.00",  # Skip / 0%
        "CHOP_0.01",  # 1% micro
        "CHOP_0.02",  # 2% small
        "CHOP_0.03",  # 3% current default
        "CHOP_0.05",  # 5% moderate
        "CHOP_0.10",  # 10% aggressive
        "PROXY_BTC",  # BTC proxy
    ]

    all_results: Dict[str, List[ChopBacktestStats]] = {}

    # Load BTC series for proxy reference
    btc_path = os.path.join(ROOT, "cachedb", "cache_price_BTCUSDC.jsonl")
    raw_btc = load_price_series(btc_path, "BTCUSDC")
    sampled_btc: List[Tuple[float, float]] = []
    last_t = 0.0
    for t, p in raw_btc:
        if (t - last_t) >= step_seconds:
            sampled_btc.append((t, p))
            last_t = t

    for sym in symbols:
        cache_path = os.path.join(ROOT, "cachedb", f"cache_price_{sym}.jsonl")
        logger.info("Loading price series for %s from %s ...", sym, cache_path)
        raw_series = load_price_series(cache_path, sym)
        if not raw_series:
            logger.warning("No price data found for %s, skipping.", sym)
            continue

        sampled: List[Tuple[float, float]] = []
        last_t = 0.0
        for t, p in raw_series:
            if (t - last_t) >= step_seconds:
                sampled.append((t, p))
                last_t = t

        logger.info("Symbol %s: %d ticks -> %d %ds bars (%.1f days)", sym, len(raw_series), len(sampled), step_seconds, (sampled[-1][0] - sampled[0][0]) / 86400.0)

        sym_stats: List[ChopBacktestStats] = []
        for pol in policies:
            # Skip PROXY_BTC on BTC itself
            if sym == "BTCUSDC" and pol == "PROXY_BTC":
                continue
            logger.info("Testing policy %s on %s ...", pol, sym)
            stat = run_single_simulation(
                symbol=sym,
                sampled=sampled,
                btc_sampled=sampled_btc if sym != "BTCUSDC" else None,
                policy=pol,
                step_seconds=step_seconds,
            )
            sym_stats.append(stat)

        all_results[sym] = sym_stats

    return all_results


def print_comparison_table(symbol: str, stats_list: List[ChopBacktestStats]) -> None:
    """Print a clean Markdown table comparing all chop policies for a symbol."""
    print(f"\n{'='*115}")
    print(f"CHOP REGIME & FALLBACK WEIGHT BENCHMARK — {symbol}".center(115))
    print(f"{'='*115}")
    header = (
        f"| {'Policy':<14} | {'Net Return':<11} | {'Net Profit':<12} | "
        f"{'Max DD':<9} | {'Win Rate':<9} | {'Profit Factor':<14} | "
        f"{'Total Trades':<13} | {'Chop Trades':<12} | {'Chop PnL':<11} |"
    )
    sep = (
        f"|{'-'*16}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*11}|{'-'*16}|{'-'*15}|{'-'*14}|{'-'*13}|"
    )
    print(header)
    print(sep)
    for s in stats_list:
        pol_name = s.policy
        if s.policy == "CHOP_0.03":
            pol_name += " (BASE)"
        elif s.policy == "CHOP_0.00":
            pol_name += " (SKIP)"
        row = (
            f"| {pol_name:<14} | {s.net_return_pct:>+8.2f}%   | ${s.net_profit_usd:>+9.2f}   | "
            f"{s.max_drawdown_pct:>6.2f}%   | {s.win_rate_pct:>6.2f}%   | {s.profit_factor:>12.2f}   | "
            f"{s.total_trades:>11}   | {s.chop_trades:>10}   | ${s.chop_pnl_usd:>+8.2f}   |"
        )
        print(row)
    print(f"{'='*115}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chop Weight Backtest")
    parser.add_argument("--symbols", default="TAOUSDC,BTCUSDC", help="Comma-separated symbols")
    parser.add_argument("--step-sec", type=int, default=900, help="Bar resolution in seconds (default 900 = 15m)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = run_chop_benchmark(symbols=symbols, step_seconds=args.step_sec)

    for sym, stats in results.items():
        print_comparison_table(sym, stats)

    # Save to JSON in tracked research directory and logger
    out_paths = [
        os.path.join(ROOT, "offline", "research", "chop_weight_backtest_results.json"),
        os.path.join(ROOT, "logger", "chop_weight_backtest_results.json"),
    ]
    json_data = {
        sym: [asdict(s) for s in stats]
        for sym, stats in results.items()
    }
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        logger.info("Saved chop benchmark results to %s", p)
