"""replay_trend_source.py — a trend signal computed from the REPLAYED history (not from
cache_short_trend.json, which is written LIVE), for monitortrades backtests.

It uses EXACTLY the same PriceWindow class (pricewindow.py) and formula
(get_short_trend: growth_coefficient from pricewindow.SHORT_TREND_ESTIMATOR — since
27 Oct the median-filtered Step-WOLS cubic; legacy (slope_full + gradient_recent)/2 via
MPTRADE_SHORT_TREND_ESTIMATOR=legacy -> semn) ca live
(cacheManager.py, after the 29 Jul race fix — see below; identical also to
tradeall.handle_symbol's own formula), fed by hand from the replay ticks
— the same pattern already used by offline/backtests/tradeall.py for its windows.

Why we do NOT model "tradeall absent": tradeall.py ALWAYS runs live (it is not an
optional process) — so this signal is always available, exactly as in live. Neutral
(gradient≈0) appears ONLY when the market itself has no clear direction in the
chosen window, never as a fallback for a process that might not be running.

Context (29 iul, investigatie cursa fast/slow in cacheManager.py): masurat empiric
pe istoric real ca cele 2 cai (rapida=get_recent_gradient simplu, lenta=
the rich get_instant_trend) differ in SIGN on 14.9% (BTC) / 21.0% (TAO) of the ticks,
and the fast path alone changes its sign on 32-35% of the ticks (a window too
mica/zgomotoasa pt un semnal de "trend" stabil). Fix aplicat live (cacheManager.py,
not yet committed at that date): the fast path no longer writes gradient_recent/final_trend,
only the slow path does. This module REPLICATES the slow-path formula (get_instant_trend),
the only one that reaches is_trend_up() in monitortrades.py today.

Parameterised on the window length (window_seconds) — 3 distinct horizons, ALL
valid (a user decision: they are not unified into a single one, each trading model
uses what suits it):
  - "instant"  ~3.7 min (the default, identical to WINDOW_SECONDS[0] in
                cacheManager.CacheShortTrendManager — what is_trend_up() reads today)
  - "mediu"    ~1.5-6h  (orizontul "big" al lui tradeall, slope_big — azi nepublicat
                cross-process, but computable identically here for A/B tests)
  - "long"     not defined yet (the exact window remains to be decided; supported generically
                through the same window_seconds parameter, with no default)
"""
from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import pricewindow as pw
from botcore import float_env, safe_float, safe_int, safe_nonneg_float, safe_pos_float
from providers.base import normalize_side

# The default "instant" window — identical today to WINDOW_SECONDS[0] in
# cacheManager.CacheShortTrendManager (the window is_trend_up() relies on).
DEFAULT_WINDOW_SECONDS = 3.7 * 60

# The proposed ranges (user, 29 Jul) for A/B sensitivity sweeps — NOT default
# values, only starting points for a future backtest on a medium/long horizon.
SMALL_WINDOW_RANGE_SEC = (60.0, 7 * 60.0)          # 1-7 minute
MEDIUM_WINDOW_RANGE_SEC = (1.5 * 3600.0, 6 * 3600.0)  # 1.5-6 ore
# The long horizon: the interval is not decided yet (user: "I don't know") — to be settled later,
# based on a backtest of the interval rather than assumed a priori.


class ReplayTrendSource:
    """One PriceWindow per symbol, fed by hand from the replay ticks —
    entirely isolated (it never touches cache_short_trend.json). Instantiate it
    o data per (simbol-set, window_seconds); pt teste A/B pe orizonturi diferite,
    use SEPARATE instances (do not reuse one across horizons)."""

    def __init__(self, symbols, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        cand_w = safe_pos_float(window_seconds, default=None)
        w_sec = cand_w if cand_w is not None else DEFAULT_WINDOW_SECONDS
        self.window_seconds = w_sec
        is_default_w = math.isclose(w_sec, DEFAULT_WINDOW_SECONDS, rel_tol=0.0, abs_tol=1e-6)
        self._windows: Dict[str, pw.PriceWindow] = {}
        for s in symbols:
            sym_w = self.window_seconds
            if is_default_w:
                fast_w = pw.fast_trend_window_for_symbol(s)
                if fast_w is not None and fast_w > 0.0:
                    sym_w = float(fast_w)
            self._windows[s] = pw.PriceWindow(s, window_size=200, window_seconds=sym_w)
        self._prev_ts: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._cached_trend_up: Dict[str, Optional[bool]] = {s: None for s in symbols}

    def advance(self, symbol: str, ts: float, price: float) -> None:
        """Feed the symbol's window with ONE new replay tick. It must be
        called for EVERY tick, BEFORE reading is_trend_up() for the same moment
        — otherwise the signal "sees" prices from the replay's future (look-ahead)."""
        win = self._windows.get(symbol)
        if win is None:
            return
        ts_f = safe_float(ts, default=None)
        px_f = safe_pos_float(price, default=None)
        if ts_f is None or px_f is None:
            return
        prev = self._prev_ts.get(symbol)
        if prev is not None and ts_f > prev:
            win.set_sample_rate(ts_f - prev)
        self._prev_ts[symbol] = ts_f
        win.process_price(px_f)
        self._cached_trend_up[symbol] = None

    def is_trend_up(self, symbol: str, window_seconds: Optional[float] = None,
                    now: Optional[float] = None) -> bool:
        """Replicates EXACTLY the condition in monitortrades.is_trend_up() (slope>0 or
        slope==0 and gradient>0), reading from the REPLAY window, not the live file.
        Without enough data yet -> False (the same "neutral" fallback as today:
        monitortrades.is_trend_up() returns False when it has no snapshot yet)."""
        win = self._windows.get(symbol)
        use_cache = window_seconds is None
        if not use_cache and win is not None:
            ws_f = safe_float(window_seconds, default=None)
            use_cache = ws_f is not None and math.isclose(
                ws_f, win.window_seconds, rel_tol=0.0, abs_tol=1e-6
            )
        if use_cache:
            cached = self._cached_trend_up.get(symbol)
            if cached is not None:
                return cached
        if win is None or len(win.prices) < 2:
            res = False
        else:
            res = _is_trend_up_tuple(
                win.get_short_trend(),
                eps=win.get_noise_epsilon(k=1.0),
                n_samples=len(win.prices),
            )
        if use_cache:
            self._cached_trend_up[symbol] = res
        return res

    def should_wait(
        self,
        side: str,
        symbol: str,
        window_seconds: Optional[float] = None,
        fast: bool = False,
        use_noise_gate: bool = True,
        now: Optional[float] = None,
    ) -> bool:
        """Replicate CacheShortTrendManager.should_wait against the replay PriceWindow."""
        side_str = normalize_side(side)
        if side_str not in ("BUY", "SELL"):
            return False
        win = self._windows.get(symbol)
        if win is None or len(win.prices) < 2:
            return False
        _ft, gc, _sf, gr = win.get_short_trend()
        g = gr if fast else gc
        if use_noise_gate:
            eps = win.get_noise_epsilon(k=1.0)
            if abs(g) <= eps:
                return True
        if side_str == "BUY":
            return g < 0
        return g > 0


# ──────────────────────────────────────────────────────────────────────────────
# A prototype (29 Jul, the user's idea): a small window that is CONFIRMABLE and DYNAMIC — instead
# of a single FIXED window (3.7 min, chosen empirically by the user from observation),
# is_trend_up(symbol, timeout) accepts ANY duration in the small range (14-30s up
# at 9-11 min), computed ON DEMAND from a RAW buffer (ts, price) — not N
# PriceWindow windows precomputed on fixed buckets. "Confirmable" = you check
# agreement between several timeouts in the range before declaring a trend, instead
# rather than relying on a single sample (the same principle as the margin/confirmation in
# of the 2-window rule already used throughout the session's backtests).
#
# NOT DECIDED YET (to be validated empirically, not assumed): whether "confirmation
# timeouts" really beats a single well-chosen window (3.7 min is already
# empirically validated by the user) — to be compared directly on real data before any
# concluzie.
# ──────────────────────────────────────────────────────────────────────────────

SMALL_TIMEOUT_RANGE_SEC = (14.0, 11 * 60.0)   # 14s - 11 min (user, 29 iul)


UPWARD_EPSILON_K = 0.25


def resolve_upward_eps_k() -> float:
    """Resolve upward-trend noise-gate multiplier k from env, clamping negatives to UPWARD_EPSILON_K."""
    v = float_env("MPTRADE_UPWARD_EPSILON_K")
    if v is None:
        v = float_env("MPTRADE_TREND_NOISE_GATE_K")
    if v is None or v < 0.0:
        return UPWARD_EPSILON_K
    return v


def _is_trend_up_tuple(
    trend_tuple: Tuple[int, float, float, float],
    eps: float = 0.0,
    n_samples: Optional[int] = None,
    up_k: Optional[float] = None,
) -> bool:
    """Canonical monitortrades.is_trend_up() rule on (final_trend, gc, slope_full, gradient_recent)."""
    final_trend, gc, _slope_full, _gradient_recent = trend_tuple
    k_val = resolve_upward_eps_k() if up_k is None else safe_nonneg_float(up_k, UPWARD_EPSILON_K)
    eps_f = safe_pos_float(eps, default=None)
    if eps_f is not None and k_val > 0.0:
        n_f = safe_float(n_samples, default=None)
        m_steps = (n_f - 1.0) if n_f is not None else 0.0
        scale = math.sqrt(10.0 / m_steps) if m_steps > 10.0 else 1.0
        return gc > k_val * scale * eps_f
    return gc > 0 or (gc == 0 and final_trend > 0)


def _instant_trend_from_slice(
    prices: Sequence[float],
    sample_rate_sec: float,
    symbol: Optional[str] = None,
) -> Tuple[int, float, float, float]:
    """The same formula as PriceWindow.get_instant_trend(), applied directly to a
    a list of prices (without rebuilding a PriceWindow) — for slices computed
    PE CERERE dintr-un buffer brut. Intoarce (final_trend, growth_coefficient,
    slope_full, gradient_recent), identical in signature to get_instant_trend().
    30 Jul: a thin wrapper over pw.instant_trend_from_prices() (the single source of
    of the formula, extracted so it is not duplicated — cacheManager.py now uses
    the same function for the live dynamic window)."""
    return pw.instant_trend_from_prices(prices, sample_rate_sec, symbol=symbol)


class DynamicReplayTrendSource:
    """A RAW (ts, price) buffer per symbol, covering max(the timeouts used) —
    instead of a FIXED PriceWindow window. is_trend_up_at(symbol, timeout_sec)
    it trims the buffer to the last timeout_sec and computes the formula ON DEMAND
    (not precomputed) — any timeout in the range, not just fixed buckets."""

    def __init__(self, symbols, max_timeout_sec: float = SMALL_TIMEOUT_RANGE_SEC[1]):
        cand_m = safe_pos_float(max_timeout_sec, default=None)
        self.max_timeout_sec = cand_m if cand_m is not None else SMALL_TIMEOUT_RANGE_SEC[1]
        self._buf: Dict[str, deque] = {s: deque() for s in symbols}   # [(ts, price), ...] crescator

    def advance(self, symbol: str, ts: float, price: float) -> None:
        buf = self._buf.get(symbol)
        if buf is None:
            return
        ts_f = safe_float(ts, default=None)
        px_f = safe_pos_float(price, default=None)
        if ts_f is None or px_f is None:
            return
        buf.append((ts_f, px_f))
        cutoff = ts_f - self.max_timeout_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _slice(self, symbol: str, timeout_sec: float) -> List[Tuple[float, float]]:
        buf = self._buf.get(symbol)
        t_sec = safe_pos_float(timeout_sec, default=None)
        if not buf or t_sec is None:
            return []
        cutoff = buf[-1][0] - t_sec
        return [(t, p) for t, p in buf if t >= cutoff]

    def _prices_and_rate(self, symbol: str, timeout_sec: float) -> Tuple[List[float], float]:
        pts = self._slice(symbol, timeout_sec)
        if len(pts) < 2:
            return [], pw.DEFAULT_SAMPLE_RATE_SEC
        timestamps = [t for t, _ in pts]
        prices = [p for _, p in pts]
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
                if timestamps[i + 1] > timestamps[i]]
        sample_rate = float(np.median(gaps)) if gaps else pw.DEFAULT_SAMPLE_RATE_SEC
        return prices, sample_rate

    def is_trend_up_at(self, symbol: str, timeout_sec: float) -> bool:
        """is_trend_up(timeout) — what the user's idea asks for: a small DYNAMIC window, any
        any duration in the range (14s-11min), not just a fixed 3.7min."""
        prices, sample_rate = self._prices_and_rate(symbol, timeout_sec)
        if len(prices) < 2:
            return False
        eps = pw.PriceWindow._std_of_gradient(prices) if len(prices) >= 3 else 0.0
        return _is_trend_up_tuple(
            _instant_trend_from_slice(prices, sample_rate, symbol=symbol),
            eps=eps,
            n_samples=len(prices),
        )

    def is_trend_up_confirmed(self, symbol: str, timeouts_sec: Sequence[float],
                               min_agree: Optional[int] = None) -> bool:
        """"Confirmable": True only if at least `min_agree` of the given timeouts
        agree that the trend is up (default min_agree=all — the most
        strict). An answer to the user's idea: do not rely on ONE noisy sample."""
        if not timeouts_sec:
            return False
        votes = [self.is_trend_up_at(symbol, t) for t in timeouts_sec]
        ma_i = safe_int(min_agree, default=None)
        threshold = max(1, ma_i) if ma_i is not None else len(timeouts_sec)
        return sum(votes) >= threshold

    def should_wait(self, symbol: str, side: str, timeout_sec: float,
                    use_noise_gate: bool = False, epsilon_k: float = 1.0) -> bool:
        """True = WAIT (this is not yet a good moment to execute a queued BUY/SELL
        intent), False = OK, execute now. BUY waits while the price FALLS and executes
        at the FIRST sign of a rise (g>=0); SELL is the mirror image (it waits while the price rises and executes at
        the first sign of a fall, g<=0).

        A user clarification (29 Jul, after empirical measurement): "the first SLIGHT sign of
        [a reversal] -> I enter" — any sign, however weak, is enough. By default
        (use_noise_gate=False) no longer requires exceeding a noise threshold (eps) —
        the initial variant (copied faithfully from cacheManager.is_favorable_to_wait) required
        it, and that was EXACTLY the problem found: at the 2 real events
        testate, growth_coefficient era in interiorul lui eps (flat/zgomot la scara
        short) although the wider trend (on which tradeall's decision to fire was already
        based) was already confirmed — the noise rule kept it waiting LONGER
        precisely when the signal was weak or marginal, the opposite of what was wanted. use_noise_gate=
        True keeps the old variant (faithful to is_favorable_to_wait) for comparison."""
        prices, _sample_rate = self._prices_and_rate(symbol, timeout_sec)
        if len(prices) < 3:
            return True   # Not enough data -> the safe behaviour: wait.
        _final_trend, growth_coefficient, _slope_full, _gradient_recent = \
            _instant_trend_from_slice(prices, _sample_rate, symbol=symbol)
        g = growth_coefficient
        if use_noise_gate:
            cand_ek = safe_nonneg_float(epsilon_k, default=None)
            ek = cand_ek if cand_ek is not None else 1.0
            m_steps = float(len(prices) - 1)
            scale = math.sqrt(10.0 / m_steps) if m_steps > 10.0 else 1.0
            eps = float(ek * scale * pw.PriceWindow._std_of_gradient(prices))
            if abs(g) <= eps:
                return True
        side_str = normalize_side(side)
        if side_str == "BUY":
            return g < 0
        if side_str == "SELL":
            return g > 0
        return False
