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
from bisect import bisect_right
from dataclasses import asdict, dataclass
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import normalize_epoch_sec, safe_float, safe_int, safe_nonneg_float, safe_pos_float
from cachepaths import cache_path, canonical_base_asset
from instrument_registry import symbols_for
from intelligence.internal.state.survival import get_trade_weight
from providers.replay_provider import load_price_series
from state_io import atomic_write_json
from intelligence.internal.triggers.kalman_trigger import KalmanTrendTrigger
from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
from intelligence.internal.state.volatility import calculate_volatility_1h
from intelligence.internal.state.persistence import calculate_mann_kendall

logger = logging.getLogger("chop_backtest")

_CHOP_POLICY_WEIGHTS = {
    "CHOP_0.00": 0.00,
    "CHOP_0.01": 0.01,
    "CHOP_0.02": 0.02,
    "CHOP_0.03": 0.03,
    "CHOP_0.05": 0.05,
    "CHOP_0.10": 0.10,
}


def _req_pos_num(val: Any, name: str) -> float:
    fval = (
        safe_pos_float(val, default=None)
        if isinstance(val, (int, float)) and not isinstance(val, bool)
        else None
    )
    if fval is None:
        raise ValueError(f"{name} must be finite and > 0")
    return fval


def _downsample_price_series(
    raw_series: List[Tuple[float, float]], step_seconds: int
) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    needs_sort = False
    prev_t: Optional[float] = None
    for item in raw_series or ():
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t = normalize_epoch_sec(item[0], default=None, allow_zero=True)
        p = safe_pos_float(item[1], default=None)
        if t is None or p is None:
            continue
        if prev_t is not None and t < prev_t:
            needs_sort = True
        prev_t = t
        cleaned.append((t, p))
    if needs_sort:
        cleaned.sort(key=lambda x: x[0])
    sampled: List[Tuple[float, float]] = []
    last_t: Optional[float] = None
    for t, p in cleaned:
        if last_t is None or (t - last_t) >= step_seconds:
            sampled.append((t, p))
            last_t = t
    return sampled


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
    if prices is None:
        return "CHOP", 0.0
    p_arr = np.asarray(prices, dtype=np.float64)
    p_arr = p_arr[np.isfinite(p_arr) & (p_arr > 0)]
    if len(p_arr) < 10:
        return "CHOP", 0.0
    x = np.arange(len(p_arr), dtype=np.float64)
    try:
        slope, _ = np.polyfit(x, p_arr, 1)
    except Exception:
        return "CHOP", 0.0
    slope_f = safe_float(slope, default=None)
    if slope_f is None:
        return "CHOP", 0.0
    last_px = safe_pos_float(p_arr[-1].item(), default=0.0)
    norm_slope = slope_f / last_px if last_px > 0.0 else 0.0

    _, _, p_mk = calculate_mann_kendall(p_arr)
    af = safe_pos_float(mk_alpha, default=None)
    alpha = af if (af is not None and af < 1.0) else 0.05
    if p_mk > alpha:
        return "CHOP", norm_slope
    if norm_slope > 1e-4:
        return "BULL", norm_slope
    if norm_slope < -1e-4:
        return "BEAR", norm_slope
    return "CHOP", norm_slope


def _precompute_chop_bar_tape(
    symbol: str,
    sampled: List[Tuple[float, float]],
    btc_sampled: Optional[List[Tuple[float, float]]],
    step_seconds: int,
) -> List[Dict[str, Any]]:
    """Precompute policy-independent regime, volatility, and trigger signals once per symbol."""
    step_f = _req_pos_num(step_seconds, "step_seconds")
    step_seconds = max(1, safe_int(round(step_f), 1))
    step_sec_f = safe_pos_float(step_seconds, 1.0)
    if not sampled:
        return []

    kalman = KalmanTrendTrigger(gap_reset_sec=max(1800.0, step_sec_f * 3.0))
    mean_rev = MeanReversionTrigger(rsi_period=14, bb_period=20)
    btc_ts: List[float] = []
    btc_px: List[float] = []
    if btc_sampled:
        btc_pairs: List[Tuple[float, float]] = []
        for item in btc_sampled:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            bt_f = normalize_epoch_sec(item[0], default=None, allow_zero=True)
            bp_f = safe_pos_float(item[1], default=None)
            if bt_f is not None and bp_f is not None:
                btc_pairs.append((bt_f, bp_f))
        btc_pairs.sort(key=lambda x: x[0])
        btc_ts = [x[0] for x in btc_pairs]
        btc_px = [x[1] for x in btc_pairs]

    bars_24h = max(12, safe_int(round(86400.0 / step_sec_f), 12))
    bars_6h = max(10, safe_int(round(21600.0 / step_sec_f), 10))
    bars_4h = max(4, safe_int(round(14400.0 / step_sec_f), 4))
    bars_3h = max(4, safe_int(round(10800.0 / step_sec_f), 4))
    btc_max_staleness = max(step_sec_f * 2.0, 1800.0)

    rolling_prices: List[float] = []
    btc_rolling: List[float] = []
    btc_rolling_ts: List[float] = []
    last_btc_idx = -1
    current_trend = "CHOP"
    trend_start_ts: Optional[float] = None
    btc_regime = "CHOP"
    btc_trend_start_ts: Optional[float] = None
    last_regime_check_ts = 0.0
    tape: List[Dict[str, Any]] = []

    cleaned_samples: List[Tuple[float, float]] = []
    needs_sort = False
    prev_t: Optional[float] = None
    for item in sampled:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t_c = normalize_epoch_sec(item[0], default=None, allow_zero=True)
        p_c = safe_pos_float(item[1], default=None)
        if t_c is None or p_c is None:
            continue
        if prev_t is not None and t_c < prev_t:
            needs_sort = True
        prev_t = t_c
        cleaned_samples.append((t_c, p_c))
    if needs_sort:
        cleaned_samples.sort(key=lambda x: x[0])

    for t, p in cleaned_samples:
        if trend_start_ts is None:
            trend_start_ts = t
        if btc_trend_start_ts is None:
            btc_trend_start_ts = t
        rolling_prices.append(p)
        if len(rolling_prices) > bars_24h:
            rolling_prices.pop(0)

        if btc_ts:
            b_idx = bisect_right(btc_ts, t) - 1
            if b_idx >= 0 and (t - btc_ts[b_idx]) <= btc_max_staleness and b_idx != last_btc_idx:
                last_btc_idx = b_idx
                btc_rolling.append(btc_px[b_idx])
                btc_rolling_ts.append(btc_ts[b_idx])
            while btc_rolling_ts and (t - btc_rolling_ts[0]) > 86400.0:
                btc_rolling.pop(0)
                btc_rolling_ts.pop(0)
            if len(btc_rolling) < 2 or (btc_rolling_ts and (t - btc_rolling_ts[-1]) > btc_max_staleness):
                if btc_regime != "CHOP":
                    btc_regime = "CHOP"
                    btc_trend_start_ts = t

        # Check regime every 4 hours over a 6h-24h rolling window
        if (t - last_regime_check_ts) >= 14400.0 and len(rolling_prices) >= bars_6h:
            last_regime_check_ts = t
            regime, _ = detect_regime_slope(np.array(rolling_prices), mk_alpha=0.05)
            if regime != current_trend:
                current_trend = regime
                trend_start_ts = t

            if (
                len(btc_rolling) >= 10
                and btc_rolling_ts
                and (btc_rolling_ts[-1] - btc_rolling_ts[0]) >= max(0.0, 21600.0 - btc_max_staleness)
            ):
                b_reg, _ = detect_regime_slope(np.array(btc_rolling), mk_alpha=0.05)
                if b_reg != btc_regime:
                    btc_regime = b_reg
                    btc_trend_start_ts = t

        vol1h = calculate_volatility_1h(rolling_prices, sample_rate_sec=step_sec_f) if len(rolling_prices) >= bars_3h else 0.5
        eps = (p * (vol1h / 100.0) * 0.15) if vol1h else (p * 0.001)

        _, k_event = kalman.evaluate(symbol, t, p, eps)
        mr_event = mean_rev.update(symbol, t, p)

        entry_signal = (
            (k_event is not None and k_event.action.value == "ENTRY" and k_event.side.value == "BUY")
            or (mr_event is not None and mr_event.action.value == "ENTRY" and mr_event.side.value == "BUY")
        )
        pullback_signal = False
        if not entry_signal and current_trend == "BULL" and len(rolling_prices) >= bars_4h + 1:
            recent_high = max(rolling_prices[-bars_4h - 1:-1])
            pullback = (recent_high - p) / recent_high if recent_high > 0 else 0.0
            pullback_signal = pullback >= 0.012

        exit_signal = (
            (k_event is not None and k_event.action.value == "EXIT" and k_event.side.value == "SELL")
            or (mr_event is not None and mr_event.action.value == "EXIT" and mr_event.side.value == "SELL")
        )

        tape.append({
            "t": t,
            "p": p,
            "current_trend": current_trend,
            "trend_start_ts": trend_start_ts,
            "btc_regime": btc_regime,
            "btc_trend_start_ts": btc_trend_start_ts,
            "entry_signal": entry_signal,
            "pullback_signal": pullback_signal,
            "exit_signal": exit_signal,
        })
    return tape


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
    precomputed_tape: Optional[List[Dict[str, Any]]] = None,
) -> ChopBacktestStats:
    """Run a single backtest simulation under a specific chop allocation policy."""
    if not isinstance(policy, str) or (policy != "PROXY_BTC" and policy not in _CHOP_POLICY_WEIGHTS):
        raise ValueError(f"Unknown chop policy: {policy!r}")
    step_f = _req_pos_num(step_seconds, "step_seconds")
    initial_balance = _req_pos_num(initial_balance, "initial_balance")
    base_notional = _req_pos_num(base_notional, "base_notional")
    fee_f = (
        safe_nonneg_float(fee_pct, default=None)
        if isinstance(fee_pct, (int, float)) and not isinstance(fee_pct, bool)
        else None
    )
    if fee_f is None or fee_f >= 1.0:
        raise ValueError("fee_pct must be in [0, 1)")
    tp_pct = _req_pos_num(tp_pct, "tp_pct")
    stop_pct = _req_pos_num(stop_pct, "stop_pct")
    _req_pos_num(T_days, "T_days")
    step_seconds = max(1, safe_int(round(step_f), 1))
    fee_pct = fee_f
    usd = initial_balance
    asset_qty = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    entry_regime = "CHOP"
    peak_equity = initial_balance
    max_dd_pct = 0.0

    if not sampled:
        return ChopBacktestStats(
            policy=policy,
            symbol=symbol,
            initial_balance=initial_balance,
            final_balance=initial_balance,
            net_profit_usd=0.0,
            net_return_pct=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            win_rate_pct=0.0,
            total_trades=0,
            chop_trades=0,
            trend_trades=0,
            chop_pnl_usd=0.0,
            trend_pnl_usd=0.0,
            avg_trade_pnl_usd=0.0,
            avg_chop_pnl_usd=0.0,
        )

    tape = (
        precomputed_tape
        if precomputed_tape is not None
        else _precompute_chop_bar_tape(symbol, sampled, btc_sampled, step_seconds)
    )
    trades: List[Dict[str, Any]] = []

    for bar in tape:
        t = bar["t"]
        p = bar["p"]
        current_trend = bar["current_trend"]
        trend_start_ts = bar["trend_start_ts"]
        btc_regime = bar["btc_regime"]
        btc_trend_start_ts = bar["btc_trend_start_ts"]

        # Track equity & drawdown (net of exit fee)
        equity = usd + (asset_qty * p * (1.0 - fee_pct))
        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        want_buy = bar["entry_signal"] or (asset_qty == 0 and bar["pullback_signal"])
        want_sell = bar["exit_signal"]

        # Take Profit & Stop Loss
        if asset_qty > 0 and entry_price > 0:
            pnl_pct = (p - entry_price) / entry_price
            if pnl_pct >= tp_pct or pnl_pct <= -stop_pct:
                want_sell = True

        # Execute SELL
        if want_sell and asset_qty > 0:
            proceeds = asset_qty * p * (1.0 - fee_pct)
            pnl = proceeds - entry_cost
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
            entry_cost = 0.0

        # Execute BUY
        elif want_buy and not want_sell and asset_qty == 0 and usd > 50.0:
            # Determine weight based on policy and current regime
            trend_len_days = max(0.0, (t - trend_start_ts) / 86400.0)

            if current_trend == "BULL":
                # Confirmed bull trend: Gaussian weight with Lindy plateau
                _, ws = get_trade_weight(T=T_days, trend_len=trend_len_days, trend="up", order_type="BUY", lindy_plateau=True)
                weight = safe_nonneg_float(ws[0] if len(ws) > 0 else None, default=0.95)
                trade_regime = "TREND"
            else:
                # CHOP or BEAR regime: evaluate candidate chop policy
                trade_regime = "CHOP"
                if policy == "PROXY_BTC":
                    if btc_regime == "BULL":
                        b_dur = max(0.0, (t - btc_trend_start_ts) / 86400.0)
                        _, b_ws = get_trade_weight(T=T_days, trend_len=b_dur, trend="up", order_type="BUY", lindy_plateau=True)
                        weight = safe_nonneg_float(b_ws[0] if len(b_ws) > 0 else None, default=0.86)
                    else:
                        weight = 0.03
                else:
                    weight = _CHOP_POLICY_WEIGHTS[policy]

            if weight <= 0.0001:
                continue  # Skip order under 0% chop policy

            # Allocation
            max_affordable_notional = usd / (1.0 + fee_pct)
            trade_budget = min(usd * 0.98, max_affordable_notional, base_notional * weight)
            if trade_budget >= 10.0:
                cost = trade_budget * (1.0 + fee_pct)
                if cost <= usd + 1e-9:
                    cost = min(cost, usd)
                    buy_qty = trade_budget / p
                    asset_qty += buy_qty
                    entry_price = p
                    entry_cost += cost
                    entry_regime = trade_regime
                    usd -= cost

        post_equity = usd + (asset_qty * p * (1.0 - fee_pct))
        if post_equity > peak_equity:
            peak_equity = post_equity
        post_dd = ((peak_equity - post_equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        if post_dd > max_dd_pct:
            max_dd_pct = post_dd

    # Close remaining open position at end
    if asset_qty > 0 and (tape or sampled):
        last_p = safe_pos_float(tape[-1]["p"], default=0.0) if tape else safe_pos_float(sampled[-1][1], default=0.0)
        proceeds = asset_qty * last_p * (1.0 - fee_pct)
        pnl = proceeds - entry_cost
        usd += proceeds
        trades.append({
            "entry_price": entry_price,
            "exit_price": last_p,
            "pnl": pnl,
            "regime": entry_regime,
            "is_win": pnl > 0,
        })
        asset_qty = 0.0
    if usd > peak_equity:
        peak_equity = usd
    final_dd = ((peak_equity - usd) / peak_equity * 100.0) if peak_equity > 0 else 0.0
    if final_dd > max_dd_pct:
        max_dd_pct = final_dd

    # Stats Calculation
    total_trades = len(trades)
    winning = [tr for tr in trades if tr["is_win"]]
    losing = [tr for tr in trades if not tr["is_win"]]
    win_rate = (len(winning) / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(tr["pnl"] for tr in winning)
    gross_loss = abs(sum(tr["pnl"] for tr in losing))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

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
    step_f = _req_pos_num(step_seconds, "step_seconds")
    step_seconds = max(1, safe_int(round(step_f), 1))
    # Policies to test
    policies = [*_CHOP_POLICY_WEIGHTS.keys(), "PROXY_BTC"]

    all_results: Dict[str, List[ChopBacktestStats]] = {}

    # Load BTC series for proxy reference
    btc_path = cache_path("cache_sparse_price_BTCUSDC.jsonl")
    raw_btc = load_price_series(btc_path, "BTCUSDC")
    sampled_btc = _downsample_price_series(raw_btc, step_seconds)

    for sym in symbols:
        cache_path_sym = cache_path(f"cache_sparse_price_{sym}.jsonl")
        logger.info("Loading price series for %s from %s ...", sym, cache_path_sym)
        raw_series = load_price_series(cache_path_sym, sym)
        if not raw_series:
            logger.warning("No price data found for %s, skipping.", sym)
            continue

        sampled = _downsample_price_series(raw_series, step_seconds)
        if not sampled:
            logger.warning("No sampled bars for %s, skipping.", sym)
            continue

        logger.info("Symbol %s: %d ticks -> %d %ds bars (%.1f days)", sym, len(raw_series), len(sampled), step_seconds, (sampled[-1][0] - sampled[0][0]) / 86400.0)

        is_btc = canonical_base_asset(sym) == "BTC"
        btc_ref = sampled_btc if not is_btc else None
        tape = _precompute_chop_bar_tape(sym, sampled, btc_ref, step_seconds)

        sym_stats: List[ChopBacktestStats] = []
        for pol in policies:
            # Skip PROXY_BTC on BTC itself
            if is_btc and pol == "PROXY_BTC":
                continue
            logger.info("Testing policy %s on %s ...", pol, sym)
            stat = run_single_simulation(
                symbol=sym,
                sampled=sampled,
                btc_sampled=btc_ref,
                policy=pol,
                step_seconds=step_seconds,
                precomputed_tape=tape,
            )
            sym_stats.append(stat)

        all_results[sym] = sym_stats

    return all_results


def print_comparison_table(symbol: str, stats_list: List[ChopBacktestStats]) -> None:
    """Print a clean Markdown table comparing all chop policies for a symbol."""
    print(f"\n{'='*115}")
    print(f"CHOP REGIME & FALLBACK WEIGHT BENCHMARK -- {symbol}".center(115))
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    default_syms = ",".join(symbols_for("binance", role="tradeall_fire"))
    parser = argparse.ArgumentParser(description="Run Chop Weight Backtest")
    parser.add_argument("--symbols", default=default_syms, help="Comma-separated symbols")
    parser.add_argument("--step-sec", type=int, default=900, help="Bar resolution in seconds (default 900 = 15m)")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "logger", "chop_weight_backtest_results.json"),
        help="Output JSON path (defaults to logger/chop_weight_backtest_results.json)",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = run_chop_benchmark(symbols=symbols, step_seconds=args.step_sec)

    for sym, stats in results.items():
        print_comparison_table(sym, stats)

    json_data = {
        sym: [asdict(s) for s in stats]
        for sym, stats in results.items()
    }
    atomic_write_json(args.out, json_data, indent=2)
    logger.info("Saved chop benchmark results to %s", args.out)
