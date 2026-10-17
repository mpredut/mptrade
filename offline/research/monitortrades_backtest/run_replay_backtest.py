#!/usr/bin/env python3
"""
A monitortrades.py backtest (Phase 1, step 2 of UNIFIED_BACKTEST_PLAN.md) over
istoric REAL BTC/TAO, folosind ReplayMarketDataProvider — raspunde la
candidates #4-5 from BACKTEST_CANDIDATES.md: are the per-symbol gain/lost/maxage
(instruments.conf) buni?

Methodology (necessary, not an artefact): monitor_price_and_trade() MANAGES an
an EXISTING position — it never initiates the first BUY without a prior SELL to
to react to (verified in the code: `if not (trade_orders_buy or
trade_orders_sell): return`). Un backtest "de la zero" ar sta degeaba tot
time. We simulate holding a position CONTINUOUSLY: every time there is NO
there is NO trade (BUY or SELL) at all in the age window
(mt.maxage_days) — that is, a cycle has just closed (normal TP or HARD-TP)
OR the previous position "expired" with nothing happening — it is re-seeded
a new BUY at the current price, with the same fixed notional. That produces MANY
independent cycles over 329 days (not just one), comparable as a method
with the Kraken/tradeall sweeps from the same session.

It uses the REAL values from instruments.conf (not arbitrary constants):
  BTCUSDC: gain=7.0% lost=3.3% maxage=7z hardtp=17%/0.5/6h
  TAOUSDC: gain=9.2% lost=4.9% maxage=17z hardtp=17%/0.5/6h

It does NOT modify monitortrades.py or instruments.conf on disk. It never touches
the network (ReplayMarketDataProvider reads only from cachedb/cache_price_*.jsonl).

Rulare: python3 offline/research/monitortrades_backtest/run_replay_backtest.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.replay_provider import ReplayMarketDataProvider, load_price_series
from providers.market_api import MarketApi
from instrument import Instrument
import monitortrades as mt
from offline.research.monitortrades_backtest.replay_trend_source import (
    DEFAULT_WINDOW_SECONDS,
    ReplayTrendSource,
)

SEED_NOTIONAL_USD = 1000.0
SBS = 12 * 24 * 3600 + 60   # The same default as live (MT_GUARD_WINDOW_DAYS=12).
FEE_PCT = 0.1


def _live_mt_params(section):
    """Read the CURRENT mt.* parameters from a section of instruments.conf —
    NOT a hardcoded copy (found today, 23 Jul: SYMBOLS was a frozen copy from
    from before mt.buy_budget/mt.max_budget were added — scheduled_pilot.py tested
    without that protection, producing a false catastrophic result of -$200k on a
    "buy again" with qty=1 whole BTC, which was not modelled correctly because of it)."""
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.join(ROOT, "instruments.conf"))
    if section not in cp:
        return {}
    return {k: v for k, v in cp[section].items() if k.startswith("mt.") or k.startswith("trailing.")}


class _LiveSymbols:
    """dict-like: SYMBOLS[symbol]["params"] reads instruments.conf ON EVERY
    ACCESS (not at import), so it cannot become a stale copy again."""
    @property
    def _SECTIONS(self):
        from instrument_registry import select_instruments
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


def _series_cache_key(symbol: str, series: list, trend_window_seconds: float) -> tuple:
    return (
        str(symbol),
        len(series),
        series[0] if series else None,
        series[-1] if series else None,
        float(trend_window_seconds),
    )


_LAST_HARD_TP_TOUCHED: dict[tuple, bool] = {}
_LAST_RUN_DIAGNOSTICS: dict[tuple, dict] = {}


def prewarm_trend_series(
    symbol: str,
    series: list,
    trend_window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> tuple[bool, ...]:
    """Precompute and cache the deterministic is_trend_up boolean sequence for series."""
    if not series:
        return ()
    key = _series_cache_key(symbol, series, trend_window_seconds)
    cached = _TREND_TAPE_CACHE.get(key)
    if cached is not None:
        return cached
    import pricewindow as pw
    buf = pw.IncrementalTrendBuffer()
    win_sec = float(trend_window_seconds)
    win_size = 200
    sample_rate = pw.DEFAULT_SAMPLE_RATE_SEC
    cur_len = 0
    prev_ts = None
    tape: list[bool] = []
    for ts, px in series:
        if prev_ts is not None and ts > prev_ts:
            rate = ts - prev_ts
            if rate > 0:
                sample_rate = rate
                new_size = max(10, int(win_sec / rate))
                if new_size != win_size:
                    win_size = new_size
                    if cur_len > new_size:
                        cur_len = new_size
        prev_ts = ts
        if cur_len >= win_size:
            cur_len = win_size
        else:
            cur_len += 1
        buf.append(px)
        blen = len(buf.prices)
        if blen - cur_len >= max(512, win_size):
            buf.trim_front(blen - cur_len)
            blen = len(buf.prices)
        if cur_len < 2:
            tape.append(False)
        else:
            ft, _gc, _sf, gr, _ = buf.eval_slice(
                blen - cur_len, blen - 1, sample_rate_sec=sample_rate, eps_k=1.0
            )
            tape.append(gr > 0 or (gr == 0 and ft > 0))
    res = tuple(tape)
    _TREND_TAPE_CACHE[key] = res
    return res


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
        path = os.path.join(ROOT, "cachedb", f"cache_price_sparse_{symbol}.jsonl")
        series = load_price_series(path, symbol)
        if not series:
            sys.stderr.write(f"[{symbol}] no history at {path}\n")
            return None
    elif not series:
        return None

    if quiet:
        try:
            import log
            log.disable_print()
        except Exception:
            pass

    provider = ReplayMarketDataProvider({symbol: series}, fee_pct=FEE_PCT)
    api = MarketApi([provider])
    inst = Instrument(name=symbol, symbol=symbol, provider="replay",
                      base=base, quote="USDC", params=dict(params), api=api)

    maxage_s = int(float(params["mt.maxage_days"]) * 24 * 3600)

    # 29 Jul: is_trend_up() REPLICATED from the replayed history (ReplayTrendSource),
    # neutralizat — vezi replay_trend_source.py pt de ce (cursa fast/slow investigata
    # and fixed live in cacheManager.py; tradeall.py ALWAYS runs live, so
    # the signal is always available, just as it is here). trend_window_seconds is parameterised
    # for A/B tests on another horizon (instant/medium/long), without changing the default.
    tape_key = _series_cache_key(symbol, series, trend_window_seconds)
    cached_tape = _TREND_TAPE_CACHE.get(tape_key)
    if cached_tape is None:
        cached_tape = prewarm_trend_series(symbol, series, trend_window_seconds)
    trend_source = None if cached_tape is not None else ReplayTrendSource([symbol], window_seconds=trend_window_seconds)
    recorded_tape: list[bool] = []
    tick_idx = 0

    def _replay_is_trend_up(sym, window_seconds=None, now=None):
        if cached_tape is not None and sym == symbol and window_seconds is None:
            return cached_tape[tick_idx]
        return trend_source.is_trend_up(sym, window_seconds=window_seconds, now=now)

    orig_is_trend_up = mt.is_trend_up
    mt.is_trend_up = _replay_is_trend_up
    orig_get_vol = mt.get_symbol_volatility_1h
    mt.get_symbol_volatility_1h = lambda s: None
    orig_persist = mt._MT_PERSISTENCE_ENABLED
    mt._MT_PERSISTENCE_ENABLED = False
    orig_post_stop_ai = mt._apply_post_stop_ai_guard
    mt._apply_post_stop_ai_guard = (
        lambda sym, px, spx, st, snap, bb, bq, q: (True, bb, bq, q)
    )
    orig_get_relevant_trade = mt.get_relevant_trade
    orig_summarize_orders = mt._summarize_orders

    def _fast_get_relevant_trade(trade_orders, trade_type, threshold_s=0, symbol=None, now_fn=None):
        if not trade_orders:
            if quiet:
                return None, 0, True
            return orig_get_relevant_trade(trade_orders, trade_type, threshold_s, symbol, now_fn)
        if len(trade_orders) > 1 and not getattr(trade_orders, "_sorted_desc", False):
            trade_orders.sort(key=lambda x: x["timestamp"], reverse=True)
            if hasattr(trade_orders, "_sorted_desc"):
                trade_orders._sorted_desc = True
                trade_orders._summary = None
        if not quiet:
            return orig_get_relevant_trade(trade_orders, trade_type, threshold_s, symbol, now_fn)
        first = trade_orders[0]
        return float(first["price"]), float(first["timestamp"]) / 1000.0, True

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

    mt.get_relevant_trade = _fast_get_relevant_trade
    mt._summarize_orders = _fast_summarize_orders
    orig_eval_hard_tp = mt._evaluate_hard_take_profit
    hard_tp_touched = False
    min_hard_tp_gain_ratio = float("inf")
    hard_tp_cooldown_dts: list[float] = []

    def _tracked_eval_hard_tp(*a, **kw):
        nonlocal hard_tp_touched, min_hard_tp_gain_ratio
        hard_tp_touched = True
        if len(a) >= 2:
            snap = a[1]
            last_h = mt._hard_tp_last.get(getattr(snap, "symbol", symbol), 0)
            if last_h > 0:
                hard_tp_cooldown_dts.append(float(snap.current_time_s - last_h))
        res = orig_eval_hard_tp(*a, **kw)
        if res and len(a) >= 2:
            g_ratio = getattr(a[1], "gain_ratio", None)
            if g_ratio is not None and g_ratio < min_hard_tp_gain_ratio:
                min_hard_tp_gain_ratio = float(g_ratio)
        return res

    mt._evaluate_hard_take_profit = _tracked_eval_hard_tp
    mt.clear_position_stats_cache()
    try:
        first_price = provider.advance(symbol)
        if first_price is None:
            return None
        last_price = first_price
        if trend_source is not None:
            trend_source.advance(symbol, provider.now(symbol), first_price)
            recorded_tape.append(bool(trend_source.is_trend_up(symbol)))
        provider.place_order(symbol, "BUY", first_price, SEED_NOTIONAL_USD / first_price)
        n_seeds = 1
        n_ticks = 0

        trail_pct = float(params.get("trailing.pct") or 0.0)
        trail_ratio = trail_pct / 100.0
        trail_peak = first_price
        min_trail_trigger_ratio = float("inf")
        max_untriggered_trail_ratio = 0.0

        now_s = 0.0
        _now_fn = lambda: now_s
        (
            _cfg_gain, _cfg_lost, _cfg_maxage_s,
            _cfg_hard_tp_pct, _cfg_hard_tp_frac, _cfg_hard_tp_cd,
            _cfg_tp_ref, _cfg_buy_budget, _cfg_buy_qty, _cfg_max_budget,
            _cfg_min_qty, _cfg_prov_label,
        ) = mt._resolve_mt_inst_cfg(inst)
        _eff_up_thresh = min(_cfg_gain, _cfg_hard_tp_pct) if mt.HARD_TP_ENABLED else _cfg_gain
        _fast_ver = -1
        _fast_min_ts_ms = -1.0
        _fast_low_buy_px = -1.0
        _fast_high_buy_px = -1.0
        _fast_has_sells = False
        _fast_sell_time = 0.0
        _fast_reentry_high_px = float("inf")
        _fast_exit_is_stop = False

        while True:
            price = provider.advance(symbol)
            if price is None:
                break
            last_price = price
            n_ticks += 1
            tick_idx = n_ticks
            # Feed the trend window BEFORE the decision — that way the signal
            # it reflects only what would have been visible up to this tick (no look-ahead).
            if trend_source is not None:
                trend_source.advance(symbol, provider.now(symbol), price)
                recorded_tape.append(bool(trend_source.is_trend_up(symbol)))

            open_qty, _ = provider.position(symbol)
            if open_qty > 0:
                if price >= trail_peak:
                    trail_peak = price
                elif trail_ratio > 0:
                    dd_ratio = (trail_peak - price) / trail_peak
                    if dd_ratio >= trail_ratio:
                        if dd_ratio < min_trail_trigger_ratio:
                            min_trail_trigger_ratio = dd_ratio
                        provider.place_order(symbol, "SELL", price, open_qty)
                        trail_peak = price
                        continue
                    elif dd_ratio > max_untriggered_trail_ratio:
                        max_untriggered_trail_ratio = dd_ratio
            else:
                trail_peak = price

            now_s = provider.now(symbol)
            if not provider.has_orders_since(symbol, maxage_s):
                provider.place_order(symbol, "BUY", price, SEED_NOTIONAL_USD / price)
                trail_peak = price
                n_seeds += 1
                continue

            if (
                quiet
                and provider._orders_version == _fast_ver
                and (int(now_s) - maxage_s) * 1000.0 <= _fast_min_ts_ms
                and _fast_low_buy_px < price < _fast_high_buy_px
                and symbol not in mt._SL_BOUNCE_STATE
                and symbol not in mt._SL_EPISODE_STATE
                and symbol not in mt._TP_RIDE_STATE
                and symbol not in mt._PENDING_STOP_EXIT
            ):
                if not _fast_has_sells:
                    if from_disk and n_ticks % 20000 == 0:
                        sys.stderr.write(f"[{symbol}] {n_ticks} tick-uri, seed-uri={n_seeds}\n")
                    continue
                reentry_st = mt._REENTRY_STATE.get(symbol)
                if reentry_st is not None and reentry_st.get("sell_time") == _fast_sell_time:
                    t_up = (
                        cached_tape[tick_idx]
                        if cached_tape is not None
                        else bool(trend_source.is_trend_up(symbol))
                    )
                    if not t_up or (
                        not _fast_exit_is_stop
                        and not mt.MT_REENTRY_HYBRID_ENABLED
                        and price > _fast_reentry_high_px
                    ):
                        if price < reentry_st["trough"]:
                            reentry_st["trough"] = price
                            reentry_st["peak_bounce"] = price
                        if price > reentry_st["peak"]:
                            reentry_st["peak"] = price
                        if from_disk and n_ticks % 20000 == 0:
                            sys.stderr.write(f"[{symbol}] {n_ticks} tick-uri, seed-uri={n_seeds}\n")
                        continue

            try:
                mt.monitor_price_and_trade(inst, sbs=SBS, now_fn=_now_fn)
                if (
                    quiet
                    and symbol not in mt._SL_BOUNCE_STATE
                    and symbol not in mt._SL_EPISODE_STATE
                    and symbol not in mt._TP_RIDE_STATE
                    and symbol not in mt._PENDING_STOP_EXIT
                ):
                    cur_ver = provider._orders_version
                    if cur_ver != _fast_ver or (int(now_s) - maxage_s) * 1000.0 > _fast_min_ts_ms:
                        b_ords = provider.get_orders(symbol, "BUY", maxage_s)
                        s_ords = provider.get_orders(symbol, "SELL", maxage_s)
                        min_ts = float("inf")
                        for o in b_ords:
                            if o["timestamp"] < min_ts:
                                min_ts = float(o["timestamp"])
                        for o in s_ords:
                            if o["timestamp"] < min_ts:
                                min_ts = float(o["timestamp"])
                        _fast_min_ts_ms = min_ts
                        pos = mt.get_position_stats(
                            symbol, maxage_s, api=provider,
                            buy_orders=b_ords, sell_orders=s_ords, now_ts=int(now_s),
                        )
                        b_px, b_t, _ = _fast_get_relevant_trade(b_ords, "BUY", symbol=symbol, now_fn=_now_fn)
                        s_px, s_t, _ = _fast_get_relevant_trade(s_ords, "SELL", symbol=symbol, now_fn=_now_fn)
                        active_avg_buy = float(
                            pos.get("active_average_buy_price")
                            or pos.get("average_buy_price")
                            or 0.0
                        )
                        if _cfg_tp_ref == "average" and active_avg_buy > 0:
                            b_px = active_avg_buy
                        if _cfg_tp_ref == "average" and pos["average_sell_price"] > 0:
                            s_px = pos["average_sell_price"]
                        avail_q = provider.free_balance(base) or 0.0
                        owned_q = max(float(pos["net_qty"]), 0.0)
                        sellable_q = min(avail_q, owned_q) if inst.isolation == "own_ledger" else avail_q
                        pos_closed = (
                            bool(s_ords)
                            and s_t > b_t
                            and mt._is_untradable_dust(inst, symbol, sellable_q, price, _cfg_min_qty)
                        )
                        _tol = max(float(mt.MT_ARE_CLOSE_TOLERANCE_PCT), 0.0)
                        _band_mult = max((200.0 - _tol) / max(200.0 + _tol, 1e-9) - 1e-9, 0.0)
                        _re_mult = (200.0 + _tol) / max(200.0 - _tol, 1e-9) + 1e-9
                        if not b_ords or pos_closed or not b_px:
                            _fast_low_buy_px = -1.0
                            _fast_high_buy_px = float("inf")
                        else:
                            _fast_low_buy_px = b_px * (1.0 - _cfg_lost * _band_mult)
                            _fast_high_buy_px = b_px * (1.0 + _eff_up_thresh * _band_mult)
                        _fast_has_sells = bool(s_ords) and (not b_ords or s_t >= b_t)
                        _fast_sell_time = float(s_t) if s_ords else 0.0
                        _fast_exit_is_stop = (mt._LAST_EXIT_KIND.get(symbol) == "STOP")
                        _fast_reentry_high_px = (
                            s_px * (1.0 - _cfg_gain) * _re_mult
                        ) if (s_ords and s_px) else float("inf")
                        _fast_ver = cur_ver
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[{symbol}] eroare in monitor_price_and_trade: {e}\n")

            if from_disk and n_ticks % 20000 == 0:
                sys.stderr.write(f"[{symbol}] {n_ticks} tick-uri, seed-uri={n_seeds}\n")
        if cached_tape is None and len(recorded_tape) == len(series):
            _TREND_TAPE_CACHE[tape_key] = tuple(recorded_tape)
        _LAST_HARD_TP_TOUCHED[tape_key] = hard_tp_touched
        _LAST_RUN_DIAGNOSTICS[tape_key] = {
            "hard_tp_touched": hard_tp_touched,
            "min_hard_tp_gain_ratio": min_hard_tp_gain_ratio,
            "hard_tp_cooldown_dts": tuple(hard_tp_cooldown_dts),
            "min_trail_trigger_ratio": min_trail_trigger_ratio,
            "max_untriggered_trail_ratio": max_untriggered_trail_ratio,
        }
    finally:
        if quiet:
            try:
                import log
                log.enable_print()
            except Exception:
                pass
        mt.clear_position_stats_cache()
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
    fees = sum(o["qty"] * o["price"] * FEE_PCT / 100 for o in all_buys + all_sells)
    net = total_sold - total_bought + open_value - fees

    bh_qty = SEED_NOTIONAL_USD / first_price
    buy_hold_net = (last_price - first_price) * bh_qty - 2 * bh_qty * first_price * FEE_PCT / 100

    result = dict(symbol=symbol, ticks=n_ticks, seeds=n_seeds, buys=len(all_buys), sells=len(all_sells),
                  net=round(net, 2), buy_hold=round(buy_hold_net, 2),
                  first_price=first_price, last_price=last_price,
                  trend_window_sec=trend_window_seconds)
    if from_disk:
        sys.stderr.write(f"[{symbol}] REZULTAT: {result}\n")
    return result


def _cli_run_target(task_args):
    symbol, params, base, trend_window_sec = task_args
    return run_symbol(symbol, params, base, trend_window_seconds=trend_window_sec)


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", choices=list(SYMBOLS._SECTIONS), default=None,
                     help="run ONLY this symbol (default: BTCUSDC + TAOUSDC)")
    ap.add_argument("--trend-window-sec", type=float, default=DEFAULT_WINDOW_SECONDS,
                     help=f"orizontul ferestrei pt ReplayTrendSource (secunde), implicit "
                          f"{DEFAULT_WINDOW_SECONDS:.0f}s (~instant, 3.7min, what is read by "
                          f"is_trend_up() azi pe live). Pt teste A/B: mic 60-420 (1-7 min), "
                          f"mediu 5400-21600 (1.5-6h, ca slope_big al lui tradeall).")
    args = ap.parse_args()

    targets = [args.symbol] if args.symbol else list(SYMBOLS._SECTIONS)
    cli_tasks = [
        (sym, SYMBOLS[sym]["params"], SYMBOLS[sym]["base"], args.trend_window_sec)
        for sym in targets
    ]
    n_workers = min(len(cli_tasks), os.cpu_count() or 1)
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            pool.map(_cli_run_target, cli_tasks)
    else:
        for t in cli_tasks:
            _cli_run_target(t)
