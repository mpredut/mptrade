"""replay_trend_source.py -- a trend signal computed from the REPLAYED history (not from
cache_short_trend.json, which is written LIVE), for monitortrades backtests.

It uses EXACTLY the same PriceWindow class (pricewindow.py) and formula
(get_short_trend: growth_coefficient from pricewindow.SHORT_TREND_ESTIMATOR -- since
27 Oct the median-filtered Step-WOLS cubic; legacy (slope_full + gradient_recent)/2 via
MPTRADE_SHORT_TREND_ESTIMATOR=legacy -> sign) as live
(cacheManager.py, after the 29 Jul race fix -- see below; identical also to
tradeall.handle_symbol's own formula), fed directly from replay ticks
-- the same pattern already used by offline/backtests/tradeall.py for its windows.

Why we do NOT model "tradeall absent": tradeall.py ALWAYS runs live (it is not an
optional process) -- so this signal is always available, exactly as in live. Neutral
(gradient ~ 0) appears ONLY when the market itself has no clear direction in the
chosen window, never as a fallback for a process that might not be running.

Context (29 Jul, fast/slow race investigation in cacheManager.py): empirically measured
on real history that the 2 paths (fast = simple get_recent_gradient, slow =
the rich get_instant_trend) differ in SIGN on 14.9% (BTC) / 21.0% (TAO) of ticks,
and the fast path alone flips sign on 32-35% of ticks (a window too small and noisy
for a stable trend signal). Fix applied live (cacheManager.py): the fast path no
longer writes gradient_recent/final_trend; only the slow path does. This module
replicates the slow-path formula (get_instant_trend), the only one that reaches
is_trend_up() in monitortrades.py today.

Parameterised on the window length (window_seconds) -- 3 distinct horizons, ALL
valid (they are not unified into a single one; each trading model uses what suits it):
  - "instant"  ~3.7 min (the default, identical to WINDOW_SECONDS[0] in
                cacheManager.CacheShortTrendManager -- what is_trend_up() reads today)
  - "medium"   ~1.5-6h  (tradeall's "big" horizon, slope_big -- not published
                cross-process today, but computable identically here for A/B tests)
  - "long"     configurable through the same window_seconds parameter, with no default
"""
from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import pricewindow as pw
from botcore import float_env, normalize_epoch_sec, safe_float, safe_nonneg_float, safe_pos_float
from providers.base import normalize_side

# The default "instant" window -- identical today to WINDOW_SECONDS[0] in
# cacheManager.CacheShortTrendManager (the window is_trend_up() relies on).
DEFAULT_WINDOW_SECONDS = 3.7 * 60


def parse_optional_window_seconds(
    window_seconds: Optional[float],
    default_window_seconds: Optional[float] = None,
) -> Tuple[bool, Optional[float], bool]:
    """Return ``(is_valid, ws_f, is_same_as_default)`` for an optional ``window_seconds`` argument."""
    if window_seconds is None:
        return True, None, True
    ws_f = safe_pos_float(window_seconds, default=None)
    if ws_f is None:
        return False, None, False
    same = (
        default_window_seconds is not None
        and math.isclose(ws_f, default_window_seconds, rel_tol=0.0, abs_tol=1e-6)
    )
    return True, ws_f, same


def _slice_since_cutoff(buf, window_sec: float) -> List[Tuple[float, float]]:
    cutoff = buf[-1][0] - window_sec
    pts: List[Tuple[float, float]] = []
    for t, p in reversed(buf):
        if t < cutoff:
            break
        pts.append((t, p))
    pts.reverse()
    return pts


class ReplayTrendSource:
    """One PriceWindow per symbol, fed directly from replay ticks --
    entirely isolated (it never touches cache_short_trend.json). Instantiate it
    once per (symbol-set, window_seconds); for A/B tests on different horizons,
    use SEPARATE instances (do not reuse one across horizons)."""

    def __init__(self, symbols, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        cand_w = safe_pos_float(window_seconds, default=None)
        w_sec = cand_w if cand_w is not None else DEFAULT_WINDOW_SECONDS
        self.window_seconds = w_sec
        is_default_w = math.isclose(w_sec, DEFAULT_WINDOW_SECONDS, rel_tol=0.0, abs_tol=1e-6)
        self._windows: Dict[str, pw.PriceWindow] = {}
        self._raw_ticks: Dict[str, deque] = {}
        for s in symbols:
            sym_w = self.window_seconds
            if is_default_w:
                fast_w = safe_pos_float(pw.fast_trend_window_for_symbol(s), default=None)
                if fast_w is not None:
                    sym_w = fast_w
            self._windows[s] = pw.PriceWindow(s, window_size=200, window_seconds=sym_w)
            self._raw_ticks[s] = deque()
        self._max_keep_sec = max(self.window_seconds, SMALL_TIMEOUT_RANGE_SEC[1], 21600.0)
        self._prev_ts: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._cached_trend_up: Dict[str, Optional[bool]] = {s: None for s in symbols}

    def advance(self, symbol: str, ts: float, price: float) -> None:
        """Feed the symbol's window with ONE new replay tick. It must be
        called for EVERY tick, BEFORE reading is_trend_up() for the same moment
        -- otherwise the signal sees prices from the replay's future (look-ahead)."""
        if (win := self._windows.get(symbol)) is None or (parsed := _parse_advance_tick(ts, price)) is None:
            return
        ts_f, px_f = parsed
        prev = self._prev_ts.get(symbol)
        if prev is not None and ts_f < prev:
            return
        if prev is not None and ts_f > prev:
            win.set_sample_rate(ts_f - prev)
        self._prev_ts[symbol] = ts_f
        win.process_price(px_f)
        raw_q = self._raw_ticks.get(symbol)
        if raw_q is not None:
            raw_q.append((ts_f, px_f))
            cutoff = ts_f - self._max_keep_sec
            while len(raw_q) > 200 and raw_q[0][0] < cutoff:
                raw_q.popleft()
        self._cached_trend_up[symbol] = None

    def _custom_window_trend(self, symbol: str, ws_f: float) -> Optional[Tuple[Tuple[int, float, float, float], float, int]]:
        if ws_f > self._max_keep_sec:
            self._max_keep_sec = ws_f
        raw_q = self._raw_ticks.get(symbol)
        if not raw_q or len(raw_q) < 2:
            return None
        pts = _slice_since_cutoff(raw_q, ws_f)
        if len(pts) < 2:
            return None
        gaps = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1) if pts[i + 1][0] > pts[i][0]]
        sr = safe_pos_float(np.median(gaps), default=pw.DEFAULT_SAMPLE_RATE_SEC) if gaps else pw.DEFAULT_SAMPLE_RATE_SEC
        prices = [p for _, p in pts]
        eps = pw.PriceWindow._std_of_gradient(prices) if len(prices) >= 3 else 0.0
        tup = pw.instant_trend_from_prices(prices, sr, window_seconds=ws_f, symbol=symbol)
        return tup, eps, len(prices)

    def is_trend_up(self, symbol: str, window_seconds: Optional[float] = None,
                    now: Optional[float] = None) -> bool:
        """Replicates EXACTLY the condition in monitortrades.is_trend_up() (slope>0 or
        slope==0 and gradient>0), reading from the REPLAY window, not the live file.
        Without enough data yet -> False (the same "neutral" fallback as today:
        monitortrades.is_trend_up() returns False when it has no snapshot yet)."""
        win = self._windows.get(symbol)
        valid_ws, ws_f, same_win = parse_optional_window_seconds(
            window_seconds, win.window_seconds if win is not None else None
        )
        if not valid_ws:
            return False
        use_cache = window_seconds is None or same_win
        if use_cache:
            cached = self._cached_trend_up.get(symbol)
            if cached is not None:
                return cached
        elif win is not None and ws_f is not None:
            if (custom := self._custom_window_trend(symbol, ws_f)) is None:
                return False
            return _is_trend_up_tuple(custom[0], eps=custom[1], n_samples=custom[2])
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
        if (side_str := normalize_side(side)) not in ("BUY", "SELL") or (win := self._windows.get(symbol)) is None:
            return False
        valid_ws, ws_f, same_win = parse_optional_window_seconds(
            window_seconds, win.window_seconds
        )
        if not valid_ws:
            return False
        if ws_f is not None and not same_win:
            if (custom := self._custom_window_trend(symbol, ws_f)) is None:
                return False
            (_ft, gc, _sf, gr), base_eps, n_p = custom
        else:
            if len(win.prices) < 2:
                return False
            _ft, gc, _sf, gr = win.get_short_trend()
            base_eps = win.get_noise_epsilon(k=1.0) if use_noise_gate else 0.0
            n_p = len(win.prices)
        return _should_wait_for_gradient(
            side_str,
            gr if fast else gc,
            base_eps=base_eps,
            n_samples=n_p,
            use_noise_gate=use_noise_gate,
        )


# ------------------------------------------------------------------------------
# Prototype (29 Jul): a small window that is CONFIRMABLE and DYNAMIC -- instead
# of a single FIXED window (3.7 min, chosen empirically from observation),
# is_trend_up(symbol, timeout) accepts ANY duration in the small range (14-30s up
# to 9-11 min), computed ON DEMAND from a RAW buffer (ts, price) -- not N
# PriceWindow windows precomputed on fixed buckets. "Confirmable" = check
# agreement between several timeouts in the range before declaring a trend,
# rather than relying on a single sample.
#
# NOT DECIDED YET (to be validated empirically, not assumed): whether confirmation
# across timeouts beats a single well-chosen window (3.7 min is already
# empirically validated) -- to be compared directly on real data before any
# conclusion.
# ------------------------------------------------------------------------------

SMALL_TIMEOUT_RANGE_SEC = (14.0, 11 * 60.0)   # 14s - 11 min (29 Jul)
UPWARD_EPSILON_K = 0.25


def _parse_advance_tick(ts: float, price: float) -> Optional[Tuple[float, float]]:
    ts_f = normalize_epoch_sec(ts, default=None, allow_zero=True)
    px_f = safe_pos_float(price, default=None)
    return (ts_f, px_f) if ts_f is not None and px_f is not None else None


def _should_wait_for_gradient(
    side_str: str,
    g: float,
    *,
    base_eps: float = 0.0,
    n_samples: int = 0,
    use_noise_gate: bool = False,
) -> bool:
    if use_noise_gate:
        m_steps = n_samples - 1
        scale = math.sqrt(10.0 / m_steps) if m_steps > 10 else 1.0
        eps = safe_float(scale * base_eps, 0.0)
        if abs(g) <= eps:
            return True
    return g < 0 if side_str == "BUY" else g > 0


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


class DynamicReplayTrendSource:
    """A RAW (ts, price) buffer per symbol, covering max(the timeouts used) --
    instead of a FIXED PriceWindow window. is_trend_up_at(symbol, timeout_sec)
    trims the buffer to the last timeout_sec and computes the formula ON DEMAND
    (not precomputed) -- any timeout in the range, not just fixed buckets."""

    def __init__(self, symbols, max_timeout_sec: float = SMALL_TIMEOUT_RANGE_SEC[1]):
        cand_m = safe_pos_float(max_timeout_sec, default=None)
        self.max_timeout_sec = cand_m if cand_m is not None else SMALL_TIMEOUT_RANGE_SEC[1]
        self._buf: Dict[str, deque] = {s: deque() for s in symbols}   # [(ts, price), ...] ascending

    def advance(self, symbol: str, ts: float, price: float) -> None:
        if (buf := self._buf.get(symbol)) is None or (parsed := _parse_advance_tick(ts, price)) is None:
            return
        ts_f, px_f = parsed
        if buf and ts_f < buf[-1][0]:
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
        return _slice_since_cutoff(buf, t_sec)

    def _prices_and_rate(self, symbol: str, timeout_sec: float) -> Tuple[List[float], float]:
        pts = self._slice(symbol, timeout_sec)
        if len(pts) < 2:
            return [], pw.DEFAULT_SAMPLE_RATE_SEC
        timestamps = [t for t, _ in pts]
        prices = [p for _, p in pts]
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
                if timestamps[i + 1] > timestamps[i]]
        sample_rate = safe_pos_float(np.median(gaps), default=pw.DEFAULT_SAMPLE_RATE_SEC) if gaps else pw.DEFAULT_SAMPLE_RATE_SEC
        return prices, sample_rate

    def is_trend_up_at(self, symbol: str, timeout_sec: float) -> bool:
        """is_trend_up(timeout) -- a small DYNAMIC window for any duration in
        the range (14s-11min), not just a fixed 3.7min."""
        prices, sample_rate = self._prices_and_rate(symbol, timeout_sec)
        if len(prices) < 2:
            return False
        eps = pw.PriceWindow._std_of_gradient(prices) if len(prices) >= 3 else 0.0
        return _is_trend_up_tuple(
            pw.instant_trend_from_prices(
                prices, sample_rate, window_seconds=timeout_sec, symbol=symbol
            ),
            eps=eps,
            n_samples=len(prices),
        )

    def should_wait(self, symbol: str, side: str, timeout_sec: float,
                    use_noise_gate: bool = False, epsilon_k: float = 1.0) -> bool:
        """True = WAIT (this is not yet a good moment to execute a queued BUY/SELL
        intent), False = OK, execute now. BUY waits while the price FALLS and executes
        at the FIRST sign of a rise (g>=0); SELL is the mirror image (it waits while
        the price rises and executes at the first sign of a fall, g<=0).

        By default (use_noise_gate=False), execution does not require exceeding a
        noise threshold (eps): at tested real events, growth_coefficient was inside
        eps (flat/noisy at short scale) even though the wider trend had already
        confirmed, so requiring |g| > eps kept execution waiting longer when the
        short-scale signal was marginal. Setting use_noise_gate=True retains the
        noise-gated variant for comparison."""
        if (
            symbol not in self._buf
            or (side_str := normalize_side(side)) not in ("BUY", "SELL")
            or safe_pos_float(timeout_sec, default=None) is None
        ):
            return False
        prices, _sample_rate = self._prices_and_rate(symbol, timeout_sec)
        if len(prices) < 3:
            return True   # Not enough data -> the safe behaviour: wait.
        _final_trend, growth_coefficient, _slope_full, _gradient_recent = (
            pw.instant_trend_from_prices(
                prices, _sample_rate, window_seconds=timeout_sec, symbol=symbol
            )
        )
        cand_ek = safe_nonneg_float(epsilon_k, default=None) if use_noise_gate else None
        ek = cand_ek if cand_ek is not None else 1.0
        base_eps = ek * pw.PriceWindow._std_of_gradient(prices) if use_noise_gate else 0.0
        return _should_wait_for_gradient(
            side_str,
            growth_coefficient,
            base_eps=base_eps,
            n_samples=len(prices),
            use_noise_gate=use_noise_gate,
        )
