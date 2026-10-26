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


def _finite_float(val, default=None):
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


def diff_percent(v1: float, v2: float) -> float:
    """Return symmetric percentage difference relative to the absolute mean.
    Identical to botcore.diff_percent but self-contained for isolated backtests."""
    f1 = _finite_float(v1)
    f2 = _finite_float(v2)
    if f1 is None or f2 is None:
        return float("inf")
    if f1 == f2:
        return 0.0
    denom = (abs(f1) + abs(f2)) / 2
    if denom <= 0:
        return float("inf")
    return abs(f1 - f2) / denom * 100


def are_close(v1: float, v2: float, tol_pct: float) -> bool:
    """Return whether v1 is within tol_pct%% of v2, matching botcore.are_close.
    Treat near-threshold prices as reached to avoid missing entries by a few cents."""
    tol_f = _finite_float(tol_pct)
    if tol_f is None or tol_f < 0:
        return False
    return diff_percent(v1, v2) <= tol_f


def entry_price(close: float, disc_pct: float) -> float:
    """Return the entry/DCA price at `disc_pct`%% below close."""
    return close * (1 - disc_pct / 100)


def tp_price(avg: float, tp_pct: float) -> float:
    """Return the take-profit price at `tp_pct`%% above average cost."""
    return avg * (1 + tp_pct / 100)


def hit_stop(avg: float, price: float, sl_pct: float) -> bool:
    """Return whether unrealized long loss reaches stop loss.
    A non-positive sl_pct or missing average disables the stop."""
    avg_f = _finite_float(avg)
    px_f = _finite_float(price)
    sl_f = _finite_float(sl_pct)
    if avg_f is None or avg_f <= 0 or px_f is None or px_f <= 0 or sl_f is None or sl_f <= 0:
        return False
    return (avg_f - px_f) / avg_f * 100 >= sl_f


def reentry_stop_blocked(price: float, sl_low: float, bounce_pct: float, tol_pct: float) -> bool:
    """After STOP-LOSS, block until price rebounds `bounce_pct`%% above the post-sale low.
    True means reentry remains blocked."""
    low_f = _finite_float(sl_low)
    bounce_f = _finite_float(bounce_pct)
    if low_f is None or low_f <= 0 or bounce_f is None or bounce_f <= 0:
        return False
    px_f = _finite_float(price)
    if px_f is None or px_f <= 0:
        return True
    prag = low_f * (1 + bounce_f / 100)
    return px_f < prag and not are_close(px_f, prag, tol_pct)


def reentry_drop_blocked(price: float, last_sell: float, drop_pct: float, tol_pct: float) -> bool:
    """After take profit, block until price falls `drop_pct`%% below the sale price.
    True means reentry remains blocked. A non-positive drop or missing sale disables
    the barrier."""
    drop_f = _finite_float(drop_pct)
    sell_f = _finite_float(last_sell)
    if drop_f is None or drop_f <= 0 or sell_f is None or sell_f <= 0:
        return False
    px_f = _finite_float(price)
    if px_f is None or px_f <= 0:
        return True
    prag = sell_f * (1 - drop_f / 100)
    return px_f > prag and not are_close(px_f, prag, tol_pct)


def dca_price_hit(price: float, last_buy: float, drop_pct: float, tol_pct: float) -> bool:
    """Return whether price is `drop_pct`%% below the latest buy, including tolerance.
    This checks PRICE only; the caller retains DCA-count, budget, and open-order caps.
    A zero tolerance requires price to be at or below the threshold."""
    px_f = _finite_float(price)
    lb_f = _finite_float(last_buy)
    drop_f = _finite_float(drop_pct)
    if px_f is None or px_f <= 0 or lb_f is None or lb_f <= 0 or drop_f is None:
        return False
    prag = lb_f * (1 - drop_f / 100)
    return px_f <= prag or are_close(px_f, prag, tol_pct)


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
    base_f = _finite_float(base_drop_pct, 0.0)
    growth_f = _finite_float(growth_pct, 0.0)
    comp_f = _finite_float(completed_dca_buys, 0.0)
    growth = max(0.0, growth_f)
    completed = max(0, int(comp_f))
    return max(0.0, base_f) + growth * completed


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
    sell_f = _finite_float(last_sell)
    if sell_f is None or sell_f <= 0:
        return False, "no_last_sell"

    ttl_f = _finite_float(ttl_sec)
    elapsed_f = _finite_float(elapsed_sec)
    if (ttl_f is not None and ttl_f > 0
            and elapsed_f is not None and elapsed_f >= ttl_f):
        return False, "ttl_expired"

    px_f = _finite_float(price)
    if px_f is None or px_f <= 0:
        return True, "invalid_price"

    if is_bull:
        peak_cand = _finite_float(post_sell_peak)
        peak = max(sell_f, peak_cand if (peak_cand is not None and peak_cand > 0) else px_f, px_f)
        pullback_f = _finite_float(pullback_pct, 0.0)
        if pullback_f <= 0:
            return False, "bull_immediate"
        prag = peak * (1.0 - pullback_f / 100.0)
        blocked = px_f > prag and not are_close(px_f, prag, tol_pct)
        return blocked, ("bull_pullback_pending" if blocked else "bull_pullback_met")

    drop_f = _finite_float(drop_pct, 0.0)
    if drop_f <= 0:
        return False, "range_immediate"
    prag = sell_f * (1.0 - drop_f / 100.0)
    if px_f > prag and not are_close(px_f, prag, tol_pct):
        return True, "range_drop_pending"
    bounce_f = _finite_float(bounce_pct, 0.0)
    if bounce_f > 0:
        trough_cand = _finite_float(post_sell_trough)
        trough = min(sell_f, trough_cand if (trough_cand is not None and trough_cand > 0) else px_f, px_f)
        prag_bounce = trough * (1.0 + bounce_f / 100.0)
        if px_f < prag_bounce and not are_close(px_f, prag_bounce, tol_pct):
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
    vol_f = _finite_float(hourly_volatility_pct)
    if vol_f is None or vol_f <= 0:
        return float(fallback_gain_pct)
    raw = vol_f * float(vol_multiplier)
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
    str_f = _finite_float(strength)
    if str_f is None or str_f < 0:
        return float(base_tp_pct)
    thresh_f = _finite_float(strength_threshold)
    thresh = thresh_f if (thresh_f is not None and thresh_f > 0) else 2.0
    c_flat = min(1.0, max(0.0, str_f / thresh))
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
    base = max(0.0, float(_finite_float(base_trail_pct, 8.0)))
    min_t = float(_finite_float(min_trail_pct, 3.0))
    eff_min = max(min_t, 5.0) if surge_guard_active else min_t
    floor = min(base, max(0.0, eff_min))
    thresh = max(0.0, float(_finite_float(gain_threshold_pct, 6.0)))
    k = max(0.0, float(_finite_float(ratchet_k, 0.5)))
    gain = _finite_float(peak_gain_pct)
    if gain is None or gain <= thresh or k == 0.0:
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
    avg_f = _finite_float(avg_cost)
    px_f = _finite_float(current_price)
    tp_f = _finite_float(base_tp_pct)
    mult_f = _finite_float(mult)
    drop_f = _finite_float(drop_pct)
    if (
        avg_f is None
        or px_f is None
        or tp_f is None
        or mult_f is None
        or drop_f is None
        or avg_f <= 0
        or px_f <= 0
        or tp_f <= 0
        or mult_f <= 0
        or drop_f <= 0
    ):
        return False, 0.0, 0.0
    current_gain_pct = (px_f - avg_f) / avg_f * 100.0
    required_gain = tp_f * mult_f
    if not (current_gain_pct >= required_gain):
        return False, current_gain_pct, 0.0
    valid_shadows = [
        (float(t), float(p))
        for t, p in (shadow_prices or ())
        if t is not None
        and p is not None
        and not isinstance(t, bool)
        and not isinstance(p, bool)
        and math.isfinite(float(t))
        and math.isfinite(float(p))
        and float(p) > 0
    ]
    if not valid_shadows:
        return False, current_gain_pct, 0.0
    now = (
        float(current_time)
        if (current_time is not None and not isinstance(current_time, bool) and math.isfinite(float(current_time)))
        else valid_shadows[-1][0]
    )
    win_s = float(window_sec) if (window_sec is not None and not isinstance(window_sec, bool) and math.isfinite(float(window_sec))) else 0.0
    cutoff = now - max(0.0, win_s)
    recent = [p for t, p in valid_shadows if t >= cutoff]
    if not recent:
        recent = [valid_shadows[-1][1]]
    local_peak = max(max(recent), px_f)
    if local_peak <= 0:
        return False, current_gain_pct, 0.0
    micro_drop_pct = (local_peak - px_f) / local_peak * 100.0
    triggered = bool(micro_drop_pct >= drop_f)
    return triggered, current_gain_pct, micro_drop_pct


def parabolic_surge_profit_floor_pct(
    surge_gain_pct: float,
    exit_pullback_pct: float,
    min_profit_pct: float = 1.0,
) -> float:
    """Return the gross profit floor shared by surge signals and pending exits."""
    sg_f = _finite_float(surge_gain_pct, 25.0)
    ep_f = _finite_float(exit_pullback_pct, 3.2)
    mp_f = _finite_float(min_profit_pct, 1.0)
    return max(mp_f, sg_f - ep_f * 1.5)


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
    avg_f = _finite_float(avg_cost)
    px_f = _finite_float(current_price)
    peak_f = _finite_float(surge_peak)
    sg_f = _finite_float(surge_gain_pct)
    ep_f = _finite_float(exit_pullback_pct)
    mp_f = _finite_float(min_profit_pct)
    if (
        avg_f is None
        or px_f is None
        or peak_f is None
        or sg_f is None
        or ep_f is None
        or mp_f is None
        or avg_f <= 0
        or px_f <= 0
        or peak_f <= 0
        or ep_f <= 0
    ):
        return False, ""
    current_gain_pct = (px_f - avg_f) / avg_f * 100.0
    peak_gain_pct = (peak_f - avg_f) / avg_f * 100.0

    # Invariant: A parabolic surge exhaustion exit is a TAKE-PROFIT guard.
    # It must never execute below the minimum surge profit floor!
    min_floor = parabolic_surge_profit_floor_pct(
        surge_gain_pct, exit_pullback_pct, min_profit_pct)
    if not (math.isfinite(min_floor) and current_gain_pct >= min_floor):
        return False, ""

    pos_surge = surge_gain_pct > 0 and (current_gain_pct >= surge_gain_pct or peak_gain_pct >= surge_gain_pct)
    win_surge = (
        window_move_pct is not None
        and not isinstance(window_move_pct, bool)
        and math.isfinite(float(window_move_pct))
        and math.isfinite(float(surge_move_pct))
        and surge_move_pct > 0
        and window_move_pct >= surge_move_pct
        and current_gain_pct >= surge_gain_pct * 0.8
    )
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
    avg_f = _finite_float(avg_cost)
    px_f = _finite_float(current_price)
    ent_f = _finite_float(entry_ts)
    cur_t_f = _finite_float(current_ts)
    md_f = _finite_float(min_days)
    mg_f = _finite_float(min_gain_pct)
    if (
        avg_f is None
        or px_f is None
        or ent_f is None
        or cur_t_f is None
        or md_f is None
        or mg_f is None
        or avg_f <= 0
        or px_f <= 0
    ):
        return False, ""
    elapsed_days = (cur_t_f - ent_f) / 86400.0
    if not (elapsed_days >= md_f):
        return False, ""
    current_gain_pct = (px_f - avg_f) / avg_f * 100.0
    if not (current_gain_pct >= mg_f):
        return False, ""
    # Check flash drop
    win_s = _finite_float(flash_window_sec, 0.0)
    cutoff = cur_t_f - max(0.0, win_s)
    recent = [
        float(p)
        for t, p in (shadow_prices or ())
        if t is not None
        and p is not None
        and not isinstance(t, bool)
        and not isinstance(p, bool)
        and math.isfinite(float(t))
        and math.isfinite(float(p))
        and float(p) > 0
        and float(t) >= cutoff
    ]
    fd_f = _finite_float(flash_drop_pct)
    if fd_f is not None and fd_f > 0 and recent:
        local_flash_peak = max(max(recent), px_f)
        if local_flash_peak > 0:
            flash_drop = (local_flash_peak - px_f) / local_flash_peak * 100.0
            if flash_drop >= fd_f:
                return True, f"slow_grind_flash_drop_{flash_drop:.2f}%_ge_{fd_f:.2f}%"
    # Check structural drop
    sd_f = _finite_float(structural_drop_pct)
    rp_f = _finite_float(recent_peak)
    if sd_f is not None and sd_f > 0 and rp_f is not None and rp_f > 0:
        struct_drop = (rp_f - px_f) / rp_f * 100.0
        if struct_drop >= sd_f:
            return True, f"slow_grind_structural_drop_{struct_drop:.2f}%_ge_{sd_f:.2f}%"
    # Check SMA break
    sma_f = _finite_float(sma_value)
    if sma_f is not None and px_f < sma_f:
        return True, f"slow_grind_sma_break_{px_f:.4f}_lt_{sma_f:.4f}"
    return False, ""


