"""PURE spot-strategy decision rules for DCA, take profit, stop loss, and reentry.

Shared by the event-driven LIVE engine in strategies/spot_engine.py and the OHLC
backtest in kraken/backtest.py. With no state, client, or I/O, this module is the
SINGLE source of truth for price thresholds and prevents live/backtest divergence.
"""
from __future__ import annotations

from botcore import (
    _parse_bool_flag,
    are_close,
    normalize_epoch_sec,
    safe_float,
    safe_pos_float,
)


def _iter_valid_shadow_prices(shadow_prices: object) -> list[tuple[float, float]]:
    if shadow_prices is None or isinstance(shadow_prices, (bool, str, bytes, dict)) or not hasattr(shadow_prices, "__iter__"):
        return []
    out: list[tuple[float, float]] = []
    for item in shadow_prices:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        tf, pf = safe_float(item[0], default=None), safe_pos_float(item[1], default=None)
        if tf is not None and pf is not None:
            if tf >= 1e11:
                tf /= 1000.0
            out.append((tf, pf))
    if len(out) > 1 and any(out[i][0] > out[i + 1][0] for i in range(len(out) - 1)):
        out.sort(key=lambda pair: pair[0])
    return out


def entry_price(close: float, disc_pct: float) -> float:
    """Return the entry/DCA price at `disc_pct`%% below close."""
    close_f, disc_f = safe_float(close, 0.0), safe_float(disc_pct, 0.0)
    return max(0.0, close_f * (1.0 - disc_f / 100.0)) if close_f > 0 else 0.0


def tp_price(avg: float, tp_pct: float) -> float:
    """Return the take-profit price at `tp_pct`%% above average cost."""
    avg_f, tp_f = safe_float(avg, 0.0), safe_float(tp_pct, 0.0)
    return max(0.0, avg_f * (1.0 + tp_f / 100.0)) if avg_f > 0 else 0.0


def hit_stop(avg: float, price: float, sl_pct: float) -> bool:
    """Return whether unrealized long loss reaches stop loss."""
    avg_f, px_f, sl_f = safe_pos_float(avg, default=None), safe_pos_float(price, default=None), safe_pos_float(sl_pct, default=None)
    if avg_f is None or px_f is None or sl_f is None:
        return False
    return (avg_f - px_f) / avg_f * 100 >= sl_f


def dca_price_hit(price: float, last_buy: float, drop_pct: float, tol_pct: float) -> bool:
    """Return whether price is `drop_pct`%% below the latest buy, including tolerance."""
    px_f, lb_f, drop_f = safe_pos_float(price, default=None), safe_pos_float(last_buy, default=None), safe_float(drop_pct, default=None)
    if px_f is None or lb_f is None or drop_f is None:
        return False
    threshold = lb_f * (1 - drop_f / 100)
    return px_f <= threshold or are_close(px_f, threshold, tol_pct)


def progressive_dca_drop_pct(
    base_drop_pct: float,
    growth_pct: float,
    completed_dca_buys: int,
) -> float:
    """Return the next DCA distance, increased gradually after every completed DCA."""
    base_f = safe_float(base_drop_pct, 0.0)
    growth = max(0.0, safe_float(growth_pct, 0.0))
    completed = max(0, int(safe_float(completed_dca_buys, 0.0)))
    return max(0.0, base_f) + growth * completed


def dynamic_surge_gain_pct(
    hourly_volatility_pct: float | None,
    min_gain_pct: float = 24.0,
    max_gain_pct: float = 32.0,
    vol_multiplier: float = 10.0,
    fallback_gain_pct: float = 25.0,
) -> float:
    """Calculate asset-adaptive parabolic surge trigger based on trailing volatility."""
    vol_f, fb_f = safe_float(hourly_volatility_pct, default=None), safe_float(fallback_gain_pct, 25.0)
    if vol_f is None or vol_f <= 0:
        return fb_f
    raw = vol_f * safe_float(vol_multiplier, 10.0)
    min_raw, max_raw = safe_float(min_gain_pct, 24.0), safe_float(max_gain_pct, 32.0)
    return max(min(min_raw, max_raw), min(max(min_raw, max_raw), raw))


def dynamic_flat_tp_pct(
    strength: float | None,
    strength_threshold: float = 2.0,
    tp_min_pct: float = 3.0,
    tp_max_pct: float = 7.0,
    base_tp_pct: float = 5.0,
) -> float:
    """Calculate regime-aware take-profit % for flat/sideways markets."""
    str_f, base_f = safe_float(strength, default=None), safe_float(base_tp_pct, 5.0)
    if str_f is None or str_f < 0:
        return base_f
    thresh_f = safe_float(strength_threshold, default=None)
    thresh = thresh_f if (thresh_f is not None and thresh_f > 0) else 2.0
    c_flat = min(1.0, max(0.0, str_f / thresh))
    min_raw, max_raw = safe_float(tp_min_pct, 3.0), safe_float(tp_max_pct, 7.0)
    min_tp, max_tp = min(min_raw, max_raw), max(min_raw, max_raw)
    return min_tp + c_flat * (max_tp - min_tp)


def dynamic_trend_trail_pct(
    peak_gain_pct: float,
    base_trail_pct: float = 8.0,
    min_trail_pct: float = 3.0,
    ratchet_k: float = 0.5,
    gain_threshold_pct: float = 6.0,
    surge_guard_active: bool = False,
) -> float:
    """Calculate ratcheted trailing stop distance based on peak unrealized gain."""
    base = max(0.0, safe_float(base_trail_pct, 8.0))
    min_t = safe_float(min_trail_pct, 3.0)
    eff_min = max(min_t, 5.0) if _parse_bool_flag(surge_guard_active) else min_t
    floor = min(base, max(0.0, eff_min))
    thresh = max(0.0, safe_float(gain_threshold_pct, 6.0))
    k = max(0.0, safe_float(ratchet_k, 0.5))
    gain = safe_float(peak_gain_pct, default=None)
    if gain is None or gain <= thresh or k == 0.0:
        return base
    return max(floor, min(base, base - k * (gain - thresh)))


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
    """Return (triggered, current_gain_pct, micro_drop_pct) for fast 2X profit reversal guard."""
    avg_f, px_f = safe_pos_float(avg_cost, default=None), safe_pos_float(current_price, default=None)
    tp_f, mult_f, drop_f = safe_pos_float(base_tp_pct, default=None), safe_pos_float(mult, default=None), safe_pos_float(drop_pct, default=None)
    if any(v is None for v in (avg_f, px_f, tp_f, mult_f, drop_f)):
        return False, 0.0, 0.0
    current_gain_pct = (px_f - avg_f) / avg_f * 100.0
    if not (current_gain_pct >= tp_f * mult_f):
        return False, current_gain_pct, 0.0
    valid_shadows = _iter_valid_shadow_prices(shadow_prices)
    if not valid_shadows:
        return False, current_gain_pct, 0.0
    now_f = normalize_epoch_sec(current_time, default=None, allow_zero=True)
    now = now_f if now_f is not None else valid_shadows[-1][0]
    cutoff = now - max(0.0, safe_float(window_sec, 0.0))
    recent = [p for t, p in valid_shadows if cutoff <= t <= now]
    if not recent:
        past_or_now = [p for t, p in valid_shadows if t <= now]
        recent = [past_or_now[-1]] if past_or_now else [valid_shadows[-1][1]]
    local_peak = max(max(recent), px_f)
    if local_peak <= 0:
        return False, current_gain_pct, 0.0
    micro_drop_pct = (local_peak - px_f) / local_peak * 100.0
    return bool(micro_drop_pct >= drop_f), current_gain_pct, micro_drop_pct


def parabolic_surge_profit_floor_pct(
    surge_gain_pct: float,
    exit_pullback_pct: float,
    min_profit_pct: float = 1.0,
) -> float:
    """Return the gross profit floor shared by surge signals and pending exits."""
    sg_f = safe_float(surge_gain_pct, 25.0)
    ep_f = safe_float(exit_pullback_pct, 3.2)
    mp_f = safe_float(min_profit_pct, 1.0)
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
    """Return (triggered, reason) for 2-3 day parabolic surge exhaustion."""
    avg_f, px_f, peak_f = safe_pos_float(avg_cost, default=None), safe_pos_float(current_price, default=None), safe_pos_float(surge_peak, default=None)
    sg_f, ep_f, mp_f = safe_pos_float(surge_gain_pct, default=None), safe_pos_float(exit_pullback_pct, default=None), safe_float(min_profit_pct, default=None)
    if mp_f is None or any(v is None for v in (avg_f, px_f, peak_f, sg_f, ep_f)):
        return False, ""
    current_gain_pct = (px_f - avg_f) / avg_f * 100.0
    peak_gain_pct = (peak_f - avg_f) / avg_f * 100.0

    # Invariant: A parabolic surge exhaustion exit is a TAKE-PROFIT guard.
    min_floor = safe_float(parabolic_surge_profit_floor_pct(sg_f, ep_f, mp_f), default=None)
    if min_floor is None or current_gain_pct < min_floor:
        return False, ""

    pos_surge = sg_f > 0 and (current_gain_pct >= sg_f or peak_gain_pct >= sg_f)
    wm_f, sm_f = safe_float(window_move_pct, default=None), safe_pos_float(surge_move_pct, default=None)
    win_surge = wm_f is not None and sm_f is not None and wm_f >= sm_f and current_gain_pct >= sg_f * 0.8
    if not (_parse_bool_flag(surge_active) or pos_surge or win_surge):
        return False, ""
    pullback = (peak_f - px_f) / peak_f * 100.0
    if pullback >= ep_f:
        return True, f"surge_pullback_{pullback:.2f}%_ge_{ep_f:.2f}%"
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
    """Return (triggered, reason) for 1-3 week slow-grind accumulation exit."""
    avg_f, px_f = safe_pos_float(avg_cost, default=None), safe_pos_float(current_price, default=None)
    ent_f = normalize_epoch_sec(entry_ts, default=None, allow_zero=True)
    cur_t_f = normalize_epoch_sec(current_ts, default=None, allow_zero=True)
    md_f, mg_f = safe_pos_float(min_days, default=None), safe_pos_float(min_gain_pct, default=None)
    if (
        any(v is None for v in (avg_f, px_f, md_f, mg_f))
        or ent_f is None or cur_t_f is None or ent_f < 0 or cur_t_f < 0
    ):
        return False, ""
    if not ((cur_t_f - ent_f) / 86400.0 >= md_f):
        return False, ""
    if not ((px_f - avg_f) / avg_f * 100.0 >= mg_f):
        return False, ""
    win_s = safe_float(flash_window_sec, 0.0)
    cutoff = cur_t_f - max(0.0, win_s)
    recent = [pf for tf, pf in _iter_valid_shadow_prices(shadow_prices) if cutoff <= tf <= cur_t_f]
    fd_f = safe_pos_float(flash_drop_pct, default=None)
    if fd_f is not None and win_s > 0 and recent:
        local_flash_peak = max(max(recent), px_f)
        if local_flash_peak > 0:
            flash_drop = (local_flash_peak - px_f) / local_flash_peak * 100.0
            if flash_drop >= fd_f:
                return True, f"slow_grind_flash_drop_{flash_drop:.2f}%_ge_{fd_f:.2f}%"
    sd_f, rp_f = safe_pos_float(structural_drop_pct, default=None), safe_pos_float(recent_peak, default=None)
    if sd_f is not None and rp_f is not None:
        struct_drop = (rp_f - px_f) / rp_f * 100.0
        if struct_drop >= sd_f:
            return True, f"slow_grind_structural_drop_{struct_drop:.2f}%_ge_{sd_f:.2f}%"
    sma_f = safe_pos_float(sma_value, default=None)
    if sma_f is not None and px_f < sma_f:
        return True, f"slow_grind_sma_break_{px_f:.4f}_lt_{sma_f:.4f}"
    return False, ""
