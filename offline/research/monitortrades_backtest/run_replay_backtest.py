#!/usr/bin/env python3
"""
A monitortrades.py backtest (Phase 1, step 2 of UNIFIED_BACKTEST_PLAN.md) over
real BTC/TAO history, using ReplayMarketDataProvider -- evaluates
candidates #4-5 from BACKTEST_CANDIDATES.md: are the per-symbol gain/lost/maxage
(instruments.conf) values well calibrated?

Methodology (necessary, not an artefact): monitor_price_and_trade() MANAGES
an EXISTING position -- it never initiates the first BUY without a prior SELL
to react to (verified in the code: `if not (trade_orders_buy or
trade_orders_sell): return`). A cold-start backtest would stay idle the entire
time. We simulate holding a position CONTINUOUSLY: every time the position is
flat and there is NO trade (BUY or SELL) at all in the age window
(mt.maxage_days) -- that is, a cycle has closed and aged out -- a new BUY is
re-seeded at the current price with the same fixed notional. That produces many
independent cycles over 329 days (not just one), comparable as a method
with the Kraken/tradeall sweeps from the same session.

It uses the REAL values from instruments.conf (not arbitrary constants):
  BTCUSDC: gain=7.0% lost=3.3% maxage=7d hardtp=17%/0.5/6h
  TAOUSDC: gain=9.2% lost=4.9% maxage=17d hardtp=17%/0.5/6h

It does NOT modify monitortrades.py or instruments.conf on disk. It never touches
the network (ReplayMarketDataProvider reads only from cachedb/cache_price_*.jsonl).

Usage: python3 offline/research/monitortrades_backtest/run_replay_backtest.py
"""
import argparse
from bisect import bisect_left
import math
from multiprocessing import Pool
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import (
    bool_env,
    int_env,
    load_ini_config,
    lru_dict_put,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)

from cachepaths import cache_path
from instrument_registry import select_instruments
from offline.backtests.evaluation import parse_cli_float
from providers.replay_provider import (
    ReplayMarketDataProvider,
    backtest_isolation_context,
    load_price_series,
)
from state_io import read_text_file
from providers.market_api import MarketApi
from instrument import Instrument
from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss
from intelligence.internal.state.volatility import calculate_volatility_1h
import monitortrades as mt
import pricewindow as pw
from offline.research.monitortrades_backtest.replay_trend_source import (
    DEFAULT_WINDOW_SECONDS,
    ReplayTrendSource,
    _is_trend_up_tuple,
    parse_optional_window_seconds,
    resolve_upward_eps_k,
)

SEED_NOTIONAL_USD = 1000.0
SBS = 12 * 24 * 3600 + 60   # The same default as live (MT_GUARD_WINDOW_DAYS=12).
FEE_PCT = 0.1
_VENUE_FEE_PCT = {
    "binance": FEE_PCT,
    "kraken": 0.26,
    "hyperliquid": 0.07,
    "t212": 0.15,
}
_REPLAY_MT_LOCK = threading.RLock()


def _resolve_symbol_meta(symbol: str) -> tuple[str, float]:
    """Resolve (quote_currency, taker_fee_pct) for an mt symbol from instruments.conf."""
    conf_path = os.path.join(ROOT, "instruments.conf")
    try:
        for spec in select_instruments(role="mt", path=conf_path).values():
            if spec.symbol == symbol:
                venue = str(getattr(spec, "provider", "") or "").strip().lower()
                quote = str(getattr(spec, "quote", "") or "USDC").strip().upper() or "USDC"
                fee = _VENUE_FEE_PCT.get(venue, FEE_PCT)
                return quote, fee
    except Exception:  # noqa: BLE001
        pass
    return "USDC", FEE_PCT


def _live_mt_params(section):
    """Read the CURRENT mt.* parameters from a section of instruments.conf."""
    cp = load_ini_config(os.path.join(ROOT, "instruments.conf"), must_exist=False)
    if section not in cp:
        return {}
    return {k: v for k, v in cp[section].items() if k.startswith("mt.") or k.startswith("trailing.")}


class _LiveSymbols:
    """dict-like: SYMBOLS[symbol]["params"] reads instruments.conf ON EVERY
    ACCESS (not at import), so it cannot become a stale copy again."""
    @property
    def _SECTIONS(self):
        return {
            spec.symbol: (spec.name, spec.base)
            for spec in select_instruments(role="mt", path=os.path.join(ROOT, "instruments.conf")).values()
        }

    def __getitem__(self, symbol):
        section, base = self._SECTIONS[symbol]
        return {"base": base, "params": _live_mt_params(section)}

    def items(self):
        return [(s, self[s]) for s in self._SECTIONS]


SYMBOLS = _LiveSymbols()

_TREND_TAPE_CACHE: dict[tuple, tuple[bool, ...]] = {}


def _series_cache_key(symbol: str, series: list, trend_window_seconds: float | None) -> tuple:
    return (
        str(symbol),
        len(series),
        series[0] if series else None,
        series[-1] if series else None,
        pw.step_wols_power_for_symbol(symbol),
        pw.fast_trend_window_for_symbol(symbol),
        resolve_upward_eps_k(),
        safe_pos_float(trend_window_seconds, DEFAULT_WINDOW_SECONDS),
    )


_LAST_HARD_TP_TOUCHED: dict[tuple, bool] = {}
_LAST_RUN_DIAGNOSTICS: dict[tuple, dict] = {}


def step_adaptive_trend_window(
    ts_f: float,
    prev_ts: float | None,
    win_size: int,
    cur_len: int,
    sample_rate: float,
    eff_win_sec: float,
    *,
    default_win_size: int = 200,
) -> tuple[float, int, int, float]:
    """Advance adaptive short-trend window state for a single timestamp ``ts_f``."""
    if prev_ts is not None and ts_f > prev_ts:
        rate = safe_pos_float(ts_f - prev_ts, default=None)
        if rate is not None:
            sample_rate = rate
            new_size = min(100000, max(10, safe_int(eff_win_sec / rate, default=default_win_size)))
            if new_size != win_size:
                win_size = new_size
                if cur_len > new_size:
                    cur_len = new_size
    next_prev_ts = ts_f if (prev_ts is None or ts_f >= prev_ts) else prev_ts
    cur_len = win_size if cur_len >= win_size else cur_len + 1
    return next_prev_ts, win_size, cur_len, sample_rate


def prewarm_trend_series(
    symbol: str,
    series: list,
    trend_window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> tuple[bool, ...]:
    """Precompute and cache the deterministic is_trend_up boolean sequence for series."""
    if not series:
        return ()
    win_sec = safe_pos_float(trend_window_seconds, DEFAULT_WINDOW_SECONDS)
    key = _series_cache_key(symbol, series, win_sec)
    cached = _TREND_TAPE_CACHE.get(key)
    if cached is not None:
        return cached
    eff_win_sec = win_sec
    if math.isclose(win_sec, DEFAULT_WINDOW_SECONDS, rel_tol=0.0, abs_tol=1e-6):
        fast_w = safe_pos_float(pw.fast_trend_window_for_symbol(symbol), default=None)
        if fast_w is not None:
            eff_win_sec = fast_w
    buf = pw.IncrementalTrendBuffer()
    prev_ts, win_size, cur_len, sample_rate = None, 200, 0, pw.DEFAULT_SAMPLE_RATE_SEC
    up_k = resolve_upward_eps_k()
    sym_pwr = pw.step_wols_power_for_symbol(symbol)
    tape: list[bool] = []
    for ts, px in series:
        ts_f = safe_float(ts, default=None)
        px_f = safe_pos_float(px, default=None)
        if ts_f is None or px_f is None:
            tape.append(False)
            continue
        prev_ts, win_size, cur_len, sample_rate = step_adaptive_trend_window(
            ts_f,
            prev_ts,
            win_size,
            cur_len,
            sample_rate,
            eff_win_sec,
            default_win_size=200,
        )
        buf.append(px_f)
        blen = len(buf.prices)
        if blen - cur_len >= max(512, win_size):
            buf.trim_front(blen - cur_len)
            blen = len(buf.prices)
        if cur_len < 2:
            tape.append(False)
        else:
            ft, gc, _sf, _gr, eps = buf.eval_slice(
                blen - cur_len,
                blen - 1,
                sample_rate_sec=sample_rate,
                eps_k=1.0,
                power=sym_pwr,
                symbol=symbol,
            )
            tape.append(
                _is_trend_up_tuple(
                    (ft, gc, _sf, _gr),
                    eps=eps,
                    n_samples=cur_len,
                    up_k=up_k,
                )
            )
    res = tuple(tape)
    lru_dict_put(_TREND_TAPE_CACHE, key, res, max_entries=16)
    return res


def resolve_archive_jsonl_path(symbol: str, use_365: bool = False) -> str:
    """Resolve `cache_365_price_{symbol}.jsonl` with fallback to `cache_sparse_price_{symbol}.jsonl`."""
    fname = f"cache_365_price_{symbol}.jsonl" if use_365 else f"cache_sparse_price_{symbol}.jsonl"
    path = cache_path(fname)
    if use_365 and not os.path.exists(path):
        path = cache_path(f"cache_sparse_price_{symbol}.jsonl")
    return path


def run_symbol(
    symbol,
    params,
    base,
    quiet=True,
    trend_window_seconds=DEFAULT_WINDOW_SECONDS,
    series=None,
):
    from_disk = series is None
    if from_disk:
        use_365 = bool_env("MPTRADE_REPLAY_USE_365", default=False)
        path = resolve_archive_jsonl_path(symbol, use_365=use_365)
        series = load_price_series(path, symbol)
        if not series:
            sys.stderr.write(f"[{symbol}] no history at {path}\n")
            return None
    elif not series:
        return None

    sym_quote, sym_fee_pct = _resolve_symbol_meta(symbol)
    provider = ReplayMarketDataProvider({symbol: series}, fee_pct=sym_fee_pct)
    api = MarketApi([provider])
    inst = Instrument(name=symbol, symbol=symbol, provider="replay",
                      base=base, quote=sym_quote, params=dict(params), api=api)

    (
        _cfg_gain, _cfg_lost, _cfg_maxage_s,
        _cfg_hard_tp_pct, _cfg_hard_tp_frac, _cfg_hard_tp_cd,
        _cfg_tp_ref, _cfg_buy_budget, _cfg_buy_qty, _cfg_max_budget,
        _cfg_min_qty, _cfg_prov_label,
    ) = mt._resolve_mt_inst_cfg(inst)
    maxage_s = max(60, safe_int(_cfg_maxage_s, default=60))

    eff_tape_win_sec = safe_pos_float(trend_window_seconds, DEFAULT_WINDOW_SECONDS)
    tape_key = _series_cache_key(symbol, series, eff_tape_win_sec)
    cached_tape = _TREND_TAPE_CACHE.get(tape_key)
    if cached_tape is None:
        cached_tape = prewarm_trend_series(symbol, series, eff_tape_win_sec)
    trend_source = None if cached_tape is not None else ReplayTrendSource([symbol], window_seconds=eff_tape_win_sec)
    recorded_tape: list[bool] = []
    tick_idx = 0

    def _replay_is_trend_up(sym, window_seconds=None, now=None):
        valid_ws, ws_f, same_win = parse_optional_window_seconds(
            window_seconds, eff_tape_win_sec
        )
        if not valid_ws:
            return False
        if cached_tape is not None and sym == symbol and same_win:
            if 0 <= tick_idx < len(cached_tape):
                return cached_tape[tick_idx]
            return bool(cached_tape[-1]) if cached_tape else False
        if sym == symbol and ws_f is not None and not same_win:
            alt_tape = prewarm_trend_series(sym, series, trend_window_seconds=ws_f)
            if alt_tape:
                idx = min(max(tick_idx, 0), len(alt_tape) - 1)
                return bool(alt_tape[idx])
            return False
        if trend_source is not None:
            return trend_source.is_trend_up(sym, window_seconds=window_seconds, now=now)
        if cached_tape and sym == symbol:
            idx = min(max(tick_idx, 0), len(cached_tape) - 1)
            return cached_tape[idx]
        return False

    orig_is_trend_up = mt.is_trend_up
    orig_get_vol = mt.get_symbol_volatility_1h
    orig_persist = mt._MT_PERSISTENCE_ENABLED
    orig_post_stop_ai = mt._apply_post_stop_ai_guard
    orig_get_relevant_trade = mt.get_relevant_trade
    orig_summarize_orders = mt._summarize_orders
    orig_eval_hard_tp = mt._evaluate_hard_take_profit
    hard_tp_touched = False
    min_hard_tp_gain_ratio = math.inf
    hard_tp_cooldown_dts: set[float] = set()

    def _fast_get_relevant_trade(trade_orders, trade_type, symbol=None):
        if not trade_orders:
            if quiet:
                return None, 0, True
            return orig_get_relevant_trade(trade_orders, trade_type, symbol)
        if not quiet:
            return orig_get_relevant_trade(list(trade_orders), trade_type, symbol)
        if getattr(trade_orders, "_sorted_desc", False):
            latest = trade_orders[0]
        elif hasattr(trade_orders, "_sorted_desc"):
            latest = trade_orders[-1]
        else:
            latest = max(trade_orders, key=lambda x: x["timestamp"])
        return safe_float(latest["price"], 0.0), safe_float(latest["timestamp"], 0.0) / 1000.0, True

    def _fast_summarize_orders(orders):
        if not orders:
            return 0.0, 0.0, (0, 0.0, 0.0, 0, 0)
        cached_sum = getattr(orders, "_summary", None)
        if cached_sum is not None:
            return cached_sum
        res = orig_summarize_orders(orders)
        if hasattr(orders, "_summary"):
            orders._summary = res
        return res

    def _tracked_eval_hard_tp(*a, **kw):
        nonlocal hard_tp_touched, min_hard_tp_gain_ratio
        hard_tp_touched = True
        if len(a) >= 2:
            snap = a[1]
            last_h = mt._hard_tp_last.get(getattr(snap, "symbol", symbol), 0)
            if last_h > 0:
                hard_tp_cooldown_dts.add(round(safe_float(snap.current_time_s - last_h, 0.0), 1))
        res = orig_eval_hard_tp(*a, **kw)
        if res and len(a) >= 2:
            g_ratio = safe_float(getattr(a[1], "gain_ratio", None), default=None)
            if g_ratio is not None and g_ratio < min_hard_tp_gain_ratio:
                min_hard_tp_gain_ratio = g_ratio
        return res

    def _clear_symbol_mt_state(sym: str) -> None:
        mt.clear_position_stats_cache()
        clear_smart_stop_loss(sym, provider="replay")
        for st_dict in (
            mt._SL_BOUNCE_STATE,
            mt._SL_EPISODE_STATE,
            mt._TP_RIDE_STATE,
            mt._REENTRY_STATE,
            mt._LAST_EXIT_KIND,
            mt._REENTRY_DONE_SELL_TIME,
            mt._PENDING_STOP_EXIT,
            mt._hard_tp_last,
        ):
            st_dict.pop(sym, None)

    series_ts: list[float] | None = None

    def _replay_get_vol_1h(sym):
        nonlocal series_ts
        if sym != symbol or not series or tick_idx < 10:
            return None
        if series_ts is None:
            series_ts = [safe_float(item[0], 0.0) for item in series]
        hi = min(tick_idx + 1, len(series))
        cur_t = series_ts[hi - 1]
        lo = bisect_left(series_ts, cur_t - 3600.0, 0, hi)
        if hi - lo < 10:
            return None
        dt_span = cur_t - series_ts[lo]
        if dt_span <= 0:
            return None
        sample_rate = dt_span / (hi - lo - 1)
        win_prices = [safe_float(series[k][1], 0.0) for k in range(lo, hi)]
        return calculate_volatility_1h(win_prices, sample_rate_sec=sample_rate)

    with _REPLAY_MT_LOCK, backtest_isolation_context(quiet=quiet):
        try:
            mt.is_trend_up = _replay_is_trend_up
            mt.get_symbol_volatility_1h = _replay_get_vol_1h
            mt._MT_PERSISTENCE_ENABLED = False
            mt._apply_post_stop_ai_guard = (
                lambda sym, px, spx, st, snap, bb, bq, q, inst=None: (True, bb, bq, q)
            )
            mt.get_relevant_trade = _fast_get_relevant_trade
            mt._summarize_orders = _fast_summarize_orders
            mt._evaluate_hard_take_profit = _tracked_eval_hard_tp
            _clear_symbol_mt_state(symbol)
            first_price = provider.advance(symbol)
            if first_price is None:
                _LAST_HARD_TP_TOUCHED.pop(tape_key, None)
                _LAST_RUN_DIAGNOSTICS.pop(tape_key, None)
                return None
            last_price = first_price
            now_s = provider.now(symbol)
            if trend_source is not None:
                trend_source.advance(symbol, now_s, first_price)
                recorded_tape.append(bool(trend_source.is_trend_up(symbol)))
            seed_order = provider.place_order(symbol, "BUY", first_price, SEED_NOTIONAL_USD / first_price)
            n_seeds = 1 if seed_order is not None else 0
            n_ticks = 0

            trail_pct = safe_nonneg_float(params.get("trailing.pct"), 0.0)
            trail_ratio = trail_pct / 100.0
            trail_peak = first_price
            min_trail_trigger_ratio = math.inf
            max_untriggered_trail_ratio = 0.0

            _now_fn = lambda: now_s
            _eff_up_thresh = (
                min(_cfg_gain, _cfg_hard_tp_pct)
                if (mt.HARD_TP_ENABLED and _cfg_hard_tp_pct > 0.0 and _cfg_hard_tp_frac > 0.0)
                else _cfg_gain
            )
            _fast_ver = -1
            _fast_min_ts_ms = -1.0
            _fast_low_buy_px = -1.0
            _fast_high_buy_px = -1.0
            _fast_has_sells = False
            _fast_sell_time = 0.0
            _fast_reentry_high_px = math.inf
            _fast_exit_is_stop = False

            while True:
                price = provider.advance(symbol)
                if price is None:
                    break
                last_price = price
                n_ticks += 1
                tick_idx = n_ticks
                now_s = provider.now(symbol)
                if trend_source is not None:
                    trend_source.advance(symbol, now_s, price)
                    recorded_tape.append(bool(trend_source.is_trend_up(symbol)))

                open_qty, open_cost = provider.position(symbol)
                if open_qty > _cfg_min_qty:
                    if price >= trail_peak:
                        trail_peak = price
                    elif trail_ratio > 0:
                        dd_ratio = (trail_peak - price) / trail_peak
                        if dd_ratio >= trail_ratio:
                            if dd_ratio < min_trail_trigger_ratio:
                                min_trail_trigger_ratio = dd_ratio
                            avg_cost = (open_cost / open_qty) if open_qty > 0 else 0.0
                            exit_kind = "STOP" if (avg_cost > 0.0 and price < avg_cost) else "TP"
                            if provider.place_order(symbol, "SELL", price, open_qty) is not None:
                                mt._resolve_exit(symbol, inst, kind=exit_kind)
                            trail_peak = price
                            continue
                        elif dd_ratio > max_untriggered_trail_ratio:
                            max_untriggered_trail_ratio = dd_ratio
                else:
                    trail_peak = price

                if open_qty <= _cfg_min_qty and not provider.has_orders_since(symbol, maxage_s):
                    if provider.place_order(symbol, "BUY", price, SEED_NOTIONAL_USD / price) is not None:
                        trail_peak = price
                        n_seeds += 1
                    continue

                if (
                    quiet
                    and provider._orders_version == _fast_ver
                    and (now_s - maxage_s) * 1000.0 <= _fast_min_ts_ms
                    and _fast_low_buy_px < price < _fast_high_buy_px
                    and symbol not in mt._SL_BOUNCE_STATE
                    and symbol not in mt._SL_EPISODE_STATE
                    and symbol not in mt._TP_RIDE_STATE
                    and symbol not in mt._PENDING_STOP_EXIT
                ):
                    if not _fast_has_sells:
                        if from_disk and n_ticks % 20000 == 0:
                            sys.stderr.write(f"[{symbol}] {n_ticks} ticks, seeds={n_seeds}\n")
                        continue
                    reentry_st = mt._REENTRY_STATE.get(symbol)
                    if reentry_st is not None and reentry_st.get("sell_time") == _fast_sell_time:
                        if cached_tape is not None:
                            t_up = (
                                cached_tape[tick_idx]
                                if 0 <= tick_idx < len(cached_tape)
                                else (bool(cached_tape[-1]) if cached_tape else False)
                            )
                        else:
                            t_up = bool(trend_source.is_trend_up(symbol)) if trend_source is not None else False
                        if not t_up or (
                            not _fast_exit_is_stop
                            and not mt.MT_REENTRY_HYBRID_ENABLED
                            and price > _fast_reentry_high_px
                        ):
                            if price < reentry_st["trough"]:
                                reentry_st["trough"] = price
                                reentry_st["peak_bounce"] = price
                            else:
                                reentry_st["peak_bounce"] = max(reentry_st.get("peak_bounce", price), price)
                            if price > reentry_st["peak"]:
                                reentry_st["peak"] = price
                            if from_disk and n_ticks % 20000 == 0:
                                sys.stderr.write(f"[{symbol}] {n_ticks} ticks, seeds={n_seeds}\n")
                            continue

                mt.monitor_price_and_trade(inst, sbs=SBS, maxage_trade_s=maxage_s, now_fn=_now_fn)
                if (
                    quiet
                    and symbol not in mt._SL_BOUNCE_STATE
                    and symbol not in mt._SL_EPISODE_STATE
                    and symbol not in mt._TP_RIDE_STATE
                    and symbol not in mt._PENDING_STOP_EXIT
                ):
                    cur_ver = provider._orders_version
                    if cur_ver != _fast_ver or (now_s - maxage_s) * 1000.0 > _fast_min_ts_ms:
                        b_ords = provider.get_orders(symbol, "BUY", maxage_s)
                        s_ords = provider.get_orders(symbol, "SELL", maxage_s)
                        min_ts = math.inf
                        for o in b_ords:
                            o_ts = safe_float(o["timestamp"], default=None)
                            if o_ts is not None and o_ts < min_ts:
                                min_ts = o_ts
                        for o in s_ords:
                            o_ts = safe_float(o["timestamp"], default=None)
                            if o_ts is not None and o_ts < min_ts:
                                min_ts = o_ts
                        _fast_min_ts_ms = min_ts
                        pos = mt.get_position_stats(
                            symbol, maxage_s, api=provider,
                            buy_orders=b_ords, sell_orders=s_ords, now_ts=safe_int(now_s, default=0),
                        )
                        b_px, b_t, _ = _fast_get_relevant_trade(b_ords, "BUY", symbol=symbol)
                        s_px, s_t, _ = _fast_get_relevant_trade(s_ords, "SELL", symbol=symbol)
                        active_avg_buy = safe_float(
                            pos.get("active_average_buy_price")
                            or pos.get("average_buy_price")
                            or 0.0,
                            0.0,
                        )
                        if _cfg_tp_ref == "average" and active_avg_buy > 0:
                            b_px = active_avg_buy
                        if _cfg_tp_ref == "average" and pos["average_sell_price"] > 0:
                            s_px = pos["average_sell_price"]
                        avail_q = provider.free_balance(base) or 0.0
                        owned_q = max(safe_float(pos["net_qty"], 0.0), 0.0)
                        sellable_q = min(avail_q, owned_q) if inst.isolation == "own_ledger" else avail_q
                        pos_closed = (
                            bool(s_ords)
                            and s_t > b_t
                            and mt._is_untradable_dust(inst, symbol, sellable_q, price, _cfg_min_qty)
                        )
                        _tol = max(safe_float(mt.MT_ARE_CLOSE_TOLERANCE_PCT, 0.0), 0.0)
                        _band_mult = max((200.0 - _tol) / max(200.0 + _tol, 1e-9) - 1e-9, 0.0)
                        _re_mult = (200.0 + _tol) / max(200.0 - _tol, 1e-9) + 1e-9
                        if not b_ords or pos_closed or not b_px:
                            _fast_low_buy_px = -1.0
                            _fast_high_buy_px = math.inf
                        else:
                            _fast_low_buy_px = b_px * (1.0 - _cfg_lost * _band_mult)
                            _fast_high_buy_px = b_px * (1.0 + _eff_up_thresh * _band_mult)
                        _fast_has_sells = bool(s_ords) and (not b_ords or s_t >= b_t)
                        _fast_sell_time = safe_float(s_t, 0.0) if s_ords else 0.0
                        _fast_exit_is_stop = (mt._LAST_EXIT_KIND.get(symbol) == "STOP")
                        _fast_reentry_high_px = (
                            s_px * (1.0 - _cfg_gain) * _re_mult
                        ) if (s_ords and s_px) else math.inf
                        _fast_ver = cur_ver

                if from_disk and n_ticks % 20000 == 0:
                    sys.stderr.write(f"[{symbol}] {n_ticks} ticks, seeds={n_seeds}\n")
            if cached_tape is None and len(recorded_tape) == len(series):
                lru_dict_put(_TREND_TAPE_CACHE, tape_key, tuple(recorded_tape), max_entries=16)
            lru_dict_put(_LAST_HARD_TP_TOUCHED, tape_key, hard_tp_touched, max_entries=64)
            lru_dict_put(
                _LAST_RUN_DIAGNOSTICS,
                tape_key,
                {
                    "hard_tp_touched": hard_tp_touched,
                    "min_hard_tp_gain_ratio": min_hard_tp_gain_ratio,
                    "hard_tp_cooldown_dts": tuple(sorted(hard_tp_cooldown_dts)),
                    "min_trail_trigger_ratio": min_trail_trigger_ratio,
                    "max_untriggered_trail_ratio": max_untriggered_trail_ratio,
                },
                max_entries=64,
            )
        finally:
            _clear_symbol_mt_state(symbol)
            mt.is_trend_up = orig_is_trend_up
            mt.get_symbol_volatility_1h = orig_get_vol
            mt._MT_PERSISTENCE_ENABLED = orig_persist
            mt._apply_post_stop_ai_guard = orig_post_stop_ai
            mt.get_relevant_trade = orig_get_relevant_trade
            mt._summarize_orders = orig_summarize_orders
            mt._evaluate_hard_take_profit = orig_eval_hard_tp

    all_buys = provider.get_orders(symbol, "BUY", since_s=1e12)
    all_sells = provider.get_orders(symbol, "SELL", since_s=1e12)
    total_bought = sum(o["qty"] * o["price"] for o in all_buys)
    total_sold = sum(o["qty"] * o["price"] for o in all_sells)
    open_qty, _open_cost = provider.position(symbol)
    open_value = open_qty * last_price
    fees = sum(o["qty"] * o["price"] * sym_fee_pct / 100 for o in all_buys + all_sells)
    liquidation_fee = max(open_qty, 0.0) * last_price * sym_fee_pct / 100
    net = total_sold - total_bought + open_value - fees - liquidation_fee

    bh_qty = provider.round_quantity(symbol, SEED_NOTIONAL_USD / first_price)
    buy_hold_net = (last_price - first_price) * bh_qty - (first_price + last_price) * bh_qty * sym_fee_pct / 100

    result = dict(symbol=symbol, ticks=n_ticks, seeds=n_seeds, buys=len(all_buys), sells=len(all_sells),
                  net=round(net, 2), buy_hold=round(buy_hold_net, 2),
                  first_price=first_price, last_price=last_price,
                  trend_window_sec=eff_tape_win_sec)
    if from_disk:
        sys.stderr.write(f"[{symbol}] RESULT: {result}\n")
    return result


def _cli_run_target(task_args):
    symbol, params, base, trend_window_sec = task_args
    return run_symbol(symbol, params, base, trend_window_seconds=trend_window_sec)


def _resolve_replay_worker_cap(use_365_cli: bool) -> int:
    raw_w = int_env("WORKERS")
    if raw_w is not None and raw_w > 0:
        return raw_w
    cpu_cap = min(2 if use_365_cli else 4, os.cpu_count() or 1)
    meminfo = read_text_file("/proc/meminfo", default="")
    if isinstance(meminfo, str):
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_kb = safe_nonneg_int(parts[1], default=0)
                    if 0 < mem_kb <= 4608 * 1024:
                        return min(2, cpu_cap)
                break
    return cpu_cap


if __name__ == "__main__":
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", choices=list(SYMBOLS._SECTIONS), default=None,
                     help="run ONLY this symbol (default: BTCUSDC + TAOUSDC)")
    ap.add_argument("--trend-window-sec", type=parse_cli_float, default=DEFAULT_WINDOW_SECONDS,
                     help=f"window horizon for ReplayTrendSource (seconds), default "
                          f"{DEFAULT_WINDOW_SECONDS:.0f}s (~3.7 min, matching live "
                          f"is_trend_up()). For A/B tests: short 60-420 (1-7 min), "
                          f"medium 5400-21600 (1.5-6h, matching tradeall slope_big).")
    args = ap.parse_args()
    if args.trend_window_sec <= 0.0:
        ap.error("--trend-window-sec must be positive")

    targets = [args.symbol] if args.symbol else list(SYMBOLS._SECTIONS)
    cli_tasks = [
        (sym, SYMBOLS[sym]["params"], SYMBOLS[sym]["base"], args.trend_window_sec)
        for sym in targets
    ]
    use_365_cli = bool_env("MPTRADE_REPLAY_USE_365", default=False)
    worker_cap = _resolve_replay_worker_cap(use_365_cli)
    n_workers = min(len(cli_tasks), max(1, worker_cap))
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            pool.map(_cli_run_target, cli_tasks)
    else:
        for t in cli_tasks:
            _cli_run_target(t)
