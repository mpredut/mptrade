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
    orig_eval_hard_tp = mt._evaluate_hard_take_profit
    hard_tp_touched = False

    def _tracked_eval_hard_tp(*a, **kw):
        nonlocal hard_tp_touched
        hard_tp_touched = True
        return orig_eval_hard_tp(*a, **kw)

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
        trail_peak = first_price

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
                trail_peak = max(trail_peak, price)
                if trail_pct > 0 and (trail_peak - price) / trail_peak >= trail_pct / 100.0:
                    provider.place_order(symbol, "SELL", price, open_qty)
                    trail_peak = price
                    continue
            else:
                trail_peak = price

            now_s = provider.now(symbol)
            if not provider.has_orders_since(symbol, maxage_s):
                provider.place_order(symbol, "BUY", price, SEED_NOTIONAL_USD / price)
                trail_peak = price
                n_seeds += 1
                continue
            try:
                mt.monitor_price_and_trade(inst, sbs=SBS, now_fn=lambda: now_s)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[{symbol}] eroare in monitor_price_and_trade: {e}\n")

            if from_disk and n_ticks % 20000 == 0:
                sys.stderr.write(f"[{symbol}] {n_ticks} tick-uri, seed-uri={n_seeds}\n")
        if cached_tape is None and len(recorded_tape) == len(series):
            _TREND_TAPE_CACHE[tape_key] = tuple(recorded_tape)
        _LAST_HARD_TP_TOUCHED[tape_key] = hard_tp_touched
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


if __name__ == "__main__":
    import argparse
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
    for symbol in targets:
        cfg = SYMBOLS[symbol]
        run_symbol(symbol, cfg["params"], cfg["base"], trend_window_seconds=args.trend_window_sec)
