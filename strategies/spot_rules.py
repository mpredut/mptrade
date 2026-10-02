"""PURE spot-strategy decision rules for DCA, take profit, stop loss, and reentry.

Shared by the event-driven LIVE engine in strategies/spot_engine.py and the OHLC
backtest in kraken/backtest.py. With no state, client, or I/O, this module is the
SINGLE source of truth for price thresholds and prevents live/backtest divergence.

Motivation: the stop-aware reentry added on August 4 originally required editing
both strategy.py step() and backtest.py simulate() with duplicate `sl_bounce_pct`
logic, creating drift risk. These formulas are IDENTICAL to live behavior. Callers
retain their own event/OHLC loops and quantity/spend/count accounting, but all use
the same price thresholds.
"""
from __future__ import annotations

import math


def diff_percent(v1: float, v2: float) -> float:
    """Return symmetric percentage difference relative to the absolute mean.
    Identical to botcore.diff_percent but self-contained for isolated backtests."""
    if v1 == 0 and v2 == 0:
        return 0.0
    return abs(v1 - v2) / ((abs(v1) + abs(v2)) / 2) * 100


def are_close(v1: float, v2: float, tol_pct: float) -> bool:
    """Return whether v1 is within tol_pct%% of v2, matching botcore.are_close.
    Treat near-threshold prices as reached to avoid missing entries by a few cents."""
    return diff_percent(v1, v2) <= tol_pct


def entry_price(close: float, disc_pct: float) -> float:
    """Return the entry/DCA price at `disc_pct`%% below close."""
    return close * (1 - disc_pct / 100)


def tp_price(avg: float, tp_pct: float) -> float:
    """Return the take-profit price at `tp_pct`%% above average cost."""
    return avg * (1 + tp_pct / 100)


def hit_stop(avg: float, price: float, sl_pct: float) -> bool:
    """Return whether unrealized long loss reaches stop loss.
    A non-positive sl_pct or missing average disables the stop."""
    if sl_pct <= 0 or not avg:
        return False
    return (avg - price) / avg * 100 >= sl_pct


def reentry_stop_blocked(price: float, sl_low: float, bounce_pct: float, tol_pct: float) -> bool:
    """After STOP-LOSS, block until price rebounds `bounce_pct`%% above the post-sale low.
    True means reentry remains blocked."""
    prag = sl_low * (1 + bounce_pct / 100)
    return price < prag and not are_close(price, prag, tol_pct)


def reentry_drop_blocked(price: float, last_sell: float, drop_pct: float, tol_pct: float) -> bool:
    """After take profit, block until price falls `drop_pct`%% below the sale price.
    True means reentry remains blocked. A non-positive drop or missing sale disables
    the barrier."""
    if drop_pct <= 0 or not last_sell:
        return False
    prag = last_sell * (1 - drop_pct / 100)
    return price > prag and not are_close(price, prag, tol_pct)


def dca_price_hit(price: float, last_buy: float, drop_pct: float, tol_pct: float) -> bool:
    """Return whether price is `drop_pct`%% below the latest buy, including tolerance.
    This checks PRICE only; the caller retains DCA-count, budget, and open-order caps.
    A zero tolerance requires price to be at or below the threshold."""
    if not last_buy:
        return False
    prag = last_buy * (1 - drop_pct / 100)
    return price <= prag or are_close(price, prag, tol_pct)


def progressive_dca_drop_pct(
    base_drop_pct: float,
    growth_pct: float,
    completed_dca_buys: int,
) -> float:
    """Return the next DCA distance, increased gradually after every completed DCA.

    ``growth_pct=0`` preserves existing live behavior exactly. Clamp growth to
    non-negative values so a bad configuration cannot compress levels and accelerate
    exposure during a decline.
    """
    growth = max(0.0, float(growth_pct))
    completed = max(0, int(completed_dca_buys))
    return max(0.0, float(base_drop_pct)) + growth * completed


def reentry_hybrid_blocked(
    price: float,
    last_sell: float | None,
    post_sell_peak: float | None,
    is_bull: bool,
    drop_pct: float,
    pullback_pct: float,
    tol_pct: float = 0.05,
    elapsed_sec: float | None = None,
    ttl_sec: float | None = None,
    post_sell_trough: float | None = None,
    bounce_pct: float = 0.0,
) -> tuple[bool, str]:
    """Hybrid re-entry decision separating trend confirmation from execution trigger.

    State A/B/C logic:
      - If no valid last_sell: unblocked.
      - If TTL barrier expired (elapsed_sec >= ttl_sec > 0): unblocked.
      - If is_bull (State B - TREND_CONFIRMED):
          Bypasses last_sell price barrier; requires a pullback_pct drop
          relative to post_sell_peak (peak = max(last_sell, post_sell_peak, price)).
      - If not is_bull (State C - NO_CONFIRMED_TREND):
          Enforces standard drop_pct below last_sell. If bounce_pct > 0, also
          requires price to rebound >= bounce_pct above post_sell_trough to prevent
          catching falling knives during free-fall downtrends.

    Returns:
      (blocked: bool, reason: str)
    """
    if not last_sell or last_sell <= 0:
        return False, "no_last_sell"

    if (ttl_sec is not None and ttl_sec > 0
            and elapsed_sec is not None and elapsed_sec >= ttl_sec):
        return False, "ttl_expired"

    if is_bull:
        peak = max(float(last_sell), float(post_sell_peak or price), float(price))
        if pullback_pct <= 0:
            return False, "bull_immediate"
        prag = peak * (1.0 - pullback_pct / 100.0)
        blocked = price > prag and not are_close(price, prag, tol_pct)
        return blocked, ("bull_pullback_pending" if blocked else "bull_pullback_met")

    if drop_pct <= 0:
        return False, "range_immediate"
    prag = float(last_sell) * (1.0 - drop_pct / 100.0)
    if price > prag and not are_close(price, prag, tol_pct):
        return True, "range_drop_pending"
    if bounce_pct > 0:
        trough = min(float(last_sell), float(post_sell_trough or price), float(price))
        prag_bounce = trough * (1.0 + bounce_pct / 100.0)
        if price < prag_bounce and not are_close(price, prag_bounce, tol_pct):
            return True, "range_bounce_pending"
        return False, "range_bounce_met"
    return False, "range_drop_met"


def dynamic_surge_gain_pct(
    hourly_volatility_pct: float | None,
    min_gain_pct: float = 24.0,
    max_gain_pct: float = 32.0,
    vol_multiplier: float = 10.0,
    fallback_gain_pct: float = 25.0,
) -> float:
    """Calculate asset-adaptive parabolic surge trigger based on trailing volatility.

    Centered around the proven 25.0% baseline:
    - Normal volatility (~2.5% vol_1h) scales directly to ~25.0%.
    - Calmer assets (vol_1h <= 2.2%) stay securely clamped to min_gain_pct (24.0%),
      preventing normal +15% bull trend moves from being choked prematurely.
    - Highly volatile assets scale up to max_gain_pct (32.0%), allowing explosive
      candle bursts room to develop before arming the tight exhaustion exit.
    """
    if hourly_volatility_pct is None or not math.isfinite(hourly_volatility_pct) or hourly_volatility_pct <= 0:
        return float(fallback_gain_pct)
    raw = float(hourly_volatility_pct) * float(vol_multiplier)
    min_g = min(float(min_gain_pct), float(max_gain_pct))
    max_g = max(float(min_gain_pct), float(max_gain_pct))
    return max(min_g, min(max_g, raw))


def dynamic_flat_tp_pct(
    strength: float | None,
    strength_threshold: float = 2.0,
    tp_min_pct: float = 3.0,
    tp_max_pct: float = 7.0,
    base_tp_pct: float = 5.0,
) -> float:
    """Calculate regime-aware take-profit % for flat/sideways markets.

    c_flat = clamp(strength / strength_threshold, 0.0, 1.0).
    When strength is low (calm chop), returns near tp_min_pct.
    When strength approaches strength_threshold (breakout cusp), returns near tp_max_pct.
    Falls back safely to base_tp_pct if strength is None, negative, or non-finite.
    """
    if strength is None or not math.isfinite(strength) or strength < 0:
        return float(base_tp_pct)
    thresh = float(strength_threshold) if strength_threshold > 0 else 2.0
    c_flat = min(1.0, max(0.0, float(strength) / thresh))
    min_tp = min(float(tp_min_pct), float(tp_max_pct))
    max_tp = max(float(tp_min_pct), float(tp_max_pct))
    return min_tp + c_flat * (max_tp - min_tp)


def dynamic_trend_trail_pct(
    peak_gain_pct: float,
    base_trail_pct: float = 8.0,
    min_trail_pct: float = 3.0,
    ratchet_k: float = 0.5,
    gain_threshold_pct: float = 6.0,
    surge_guard_active: bool = False,
) -> float:
    """Calculate ratcheted trailing stop distance based on peak unrealized gain.

    Ratchets tighter as accumulated peak profit expands beyond gain_threshold_pct:
    trail = base_trail_pct - ratchet_k * max(0.0, peak_gain_pct - gain_threshold_pct)
    clamped to [min_trail_pct, base_trail_pct].

    Harmonization: When surge_guard_active is True, clamps the floor to max(min_trail_pct, 5.0)
    to preserve a healthy breathing cushion (preventing 3-4% pullbacks from prematurely
    liquidating the position before it can reach the 25% parabolic surge trigger).
    """
    base = max(0.0, float(base_trail_pct))
    eff_min = max(float(min_trail_pct), 5.0) if surge_guard_active else float(min_trail_pct)
    floor = min(base, max(0.0, eff_min))
    thresh = max(0.0, float(gain_threshold_pct))
    k = max(0.0, float(ratchet_k))
    gain = float(peak_gain_pct)
    if gain <= thresh or k == 0.0:
        return base
    ratcheted = base - k * (gain - thresh)
    return max(floor, min(base, ratcheted))



def check_fast_profit_reversal(
    current_price: float,
    avg_cost: float,
    base_tp_pct: float,
    mult: float,
    shadow_prices: list[tuple[float, float]],
    window_sec: float,
    drop_pct: float,
    current_time: float | None = None,
) -> tuple[bool, float, float]:
    """Return (triggered, current_gain_pct, micro_drop_pct) for fast 2X profit reversal guard.

    Triggered when:
    1. current_gain_pct >= mult * base_tp_pct (e.g. >= 10.0%)
    2. micro_drop_pct >= drop_pct from the highest price in the last window_sec.
    """
    if avg_cost <= 0 or current_price <= 0 or base_tp_pct <= 0 or mult <= 0:
        return False, 0.0, 0.0
    current_gain_pct = (current_price - avg_cost) / avg_cost * 100.0
    required_gain = base_tp_pct * mult
    if current_gain_pct < required_gain:
        return False, current_gain_pct, 0.0
    if not shadow_prices:
        return False, current_gain_pct, 0.0
    now = current_time if current_time is not None else shadow_prices[-1][0]
    cutoff = now - max(0.0, window_sec)
    recent = [p for t, p in shadow_prices if t >= cutoff]
    if not recent:
        recent = [shadow_prices[-1][1]]
    local_peak = max(max(recent), current_price)
    if local_peak <= 0:
        return False, current_gain_pct, 0.0
    micro_drop_pct = (local_peak - current_price) / local_peak * 100.0
    triggered = micro_drop_pct >= drop_pct
    return triggered, current_gain_pct, micro_drop_pct


def parabolic_surge_profit_floor_pct(
    surge_gain_pct: float,
    exit_pullback_pct: float,
    min_profit_pct: float = 1.0,
) -> float:
    """Return the gross profit floor shared by surge signals and pending exits."""
    return max(min_profit_pct, surge_gain_pct - exit_pullback_pct * 1.5)


def check_parabolic_surge_exhaustion(
    current_price: float,
    avg_cost: float,
    surge_peak: float,
    surge_gain_pct: float,
    exit_pullback_pct: float,
    window_move_pct: float | None = None,
    surge_move_pct: float = 25.0,
    surge_active: bool = False,
    min_profit_pct: float = 1.0,
) -> tuple[bool, str]:
    """Return (triggered, reason) for 2-3 day parabolic surge exhaustion.

    Surge is active/armed if:
    - surge_active is True (latched state), OR
    - current gain from avg_cost >= surge_gain_pct (e.g. 18.0%), OR
    - peak gain from avg_cost >= surge_gain_pct (e.g. 18.0%).
    Note: window_move_pct cannot arm surge alone unless position itself captured the move.
    In all cases, Parabolic Surge Guard is a TAKE-PROFIT exit: it must never exit below
    the minimum surge profit floor (min_floor = max(min_profit_pct, surge_gain_pct - exit_pullback_pct * 1.5)).
    If surge is active and price >= floor, triggers exit when price pulls back >= exit_pullback_pct from surge_peak.
    """
    if avg_cost <= 0 or current_price <= 0 or surge_peak <= 0:
        return False, ""
    current_gain_pct = (current_price - avg_cost) / avg_cost * 100.0
    peak_gain_pct = (surge_peak - avg_cost) / avg_cost * 100.0

    # Invariant: A parabolic surge exhaustion exit is a TAKE-PROFIT guard.
    # It must never execute below the minimum surge profit floor!
    min_floor = parabolic_surge_profit_floor_pct(
        surge_gain_pct, exit_pullback_pct, min_profit_pct)
    if current_gain_pct < min_floor:
        return False, ""

    pos_surge = surge_gain_pct > 0 and (current_gain_pct >= surge_gain_pct or peak_gain_pct >= surge_gain_pct)
    win_surge = (window_move_pct is not None and surge_move_pct > 0 and window_move_pct >= surge_move_pct
                 and current_gain_pct >= surge_gain_pct * 0.8)
    if not (surge_active or pos_surge or win_surge):
        return False, ""
    pullback = (surge_peak - current_price) / surge_peak * 100.0
    if pullback >= exit_pullback_pct:
        reason = f"surge_pullback_{pullback:.2f}%_ge_{exit_pullback_pct:.2f}%"
        return True, reason
    return False, ""


def check_slow_grind_exhaustion(
    current_price: float,
    avg_cost: float,
    entry_ts: float | None,
    current_ts: float,
    min_days: float,
    min_gain_pct: float,
    recent_peak: float,
    shadow_prices: list[tuple[float, float]],
    flash_window_sec: float,
    flash_drop_pct: float,
    structural_drop_pct: float,
    sma_value: float | None = None,
) -> tuple[bool, str]:
    """Return (triggered, reason) for 1-3 week slow-grind accumulation exit.

    Qualifies when position is held >= min_days (e.g. 7d) with accumulated gain >= min_gain_pct (e.g. 15%).
    Exits if:
    - Flash sensor: price drops >= flash_drop_pct within flash_window_sec (e.g. 1.5% in 15m), OR
    - Structural sensor: price drops >= structural_drop_pct from recent_peak (e.g. 2.5%), OR
    - SMA break: sma_value is provided and current_price < sma_value.
    """
    if avg_cost <= 0 or current_price <= 0 or entry_ts is None:
        return False, ""
    elapsed_days = (current_ts - entry_ts) / 86400.0
    if elapsed_days < min_days:
        return False, ""
    current_gain_pct = (current_price - avg_cost) / avg_cost * 100.0
    if current_gain_pct < min_gain_pct:
        return False, ""
    # Check flash drop
    cutoff = current_ts - max(0.0, flash_window_sec)
    recent = [p for t, p in shadow_prices if t >= cutoff]
    if recent:
        local_flash_peak = max(max(recent), current_price)
        if local_flash_peak > 0:
            flash_drop = (local_flash_peak - current_price) / local_flash_peak * 100.0
            if flash_drop >= flash_drop_pct:
                return True, f"slow_grind_flash_drop_{flash_drop:.2f}%_ge_{flash_drop_pct:.2f}%"
    # Check structural drop
    if recent_peak > 0:
        struct_drop = (recent_peak - current_price) / recent_peak * 100.0
        if struct_drop >= structural_drop_pct:
            return True, f"slow_grind_structural_drop_{struct_drop:.2f}%_ge_{structural_drop_pct:.2f}%"
    # Check SMA break
    if sma_value is not None and current_price < sma_value:
        return True, f"slow_grind_sma_break_{current_price:.4f}_lt_{sma_value:.4f}"
    return False, ""


