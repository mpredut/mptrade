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
    return {k: v for k, v in cp[section].items() if k.startswith("mt.")}


class _LiveSymbols:
    """dict-like: SYMBOLS[symbol]["params"] reads instruments.conf ON EVERY
    ACCESS (not at import), so it cannot become a stale copy again."""
    _SECTIONS = {
        "BTCUSDC": ("BINANCE_BTC", "BTC"),
        "TAOUSDC": ("BINANCE_TAO", "TAO"),
        "ARBUSDC": ("BINANCE_ARB", "ARB"),
    }

    def __getitem__(self, symbol):
        section, base = self._SECTIONS[symbol]
        return {"base": base, "params": _live_mt_params(section)}

    def items(self):
        return [(s, self[s]) for s in self._SECTIONS]


SYMBOLS = _LiveSymbols()


def _neutral_is_trend_up(symbol):
    """KEPT as a reference/manual fallback (it is no longer run_symbol's default,
    see ReplayTrendSource below) — still used by scheduled_pilot.py::_run_one
    (a separate loop of its own, UNTOUCHED in this pass — see 29 Jul). Deterministic,
    Backtest ONLY: no trend signal (neutral). FOUND 23 Jul: is_trend_up()
    the real one reads cacheManager.get_short_trend_manager() — for REAL symbols
    (BTCUSDC/TAOUSDC), that is the LIVE cache, updated RIGHT NOW by
    tradeall.py/cacheManager.py, which run on the same machine. Running the same
    backtest twice gave DIFFERENT results, because the "live" trend being read
    changes between runs, contaminating a historical replay with the REAL, current state
    of the market. False = exactly what is_trend_up() would return anyway for a symbol WITHOUT
    snapshot in the cache (already the "safe" default in the code: "no snapshot ->
    neutral, it does not block a profitable sale")."""
    return False


def run_symbol(symbol, params, base, quiet=True, trend_window_seconds=DEFAULT_WINDOW_SECONDS):
    path = os.path.join(ROOT, "cachedb", f"cache_price_{symbol}.jsonl")
    series = load_price_series(path, symbol)
    if not series:
        sys.stderr.write(f"[{symbol}] no history at {path}\n")
        return None

    if quiet:
        mt.log.disable_print() if hasattr(mt, "log") and hasattr(mt.log, "disable_print") else None

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
    trend_source = ReplayTrendSource([symbol], window_seconds=trend_window_seconds)

    orig_is_trend_up = mt.is_trend_up
    mt.is_trend_up = trend_source.is_trend_up
    try:
        first_price = provider.advance(symbol)
        if first_price is None:
            return None
        last_price = first_price
        trend_source.advance(symbol, provider.now(symbol), first_price)
        provider.place_order(symbol, "BUY", first_price, SEED_NOTIONAL_USD / first_price)
        n_seeds = 1
        n_ticks = 0

        while True:
            price = provider.advance(symbol)
            if price is None:
                break
            last_price = price
            n_ticks += 1
            # Feed the trend window BEFORE the decision — that way the signal
            # it reflects only what would have been visible up to this tick (no look-ahead).
            trend_source.advance(symbol, provider.now(symbol), price)

            buys = provider.get_orders(symbol, "BUY", since_s=maxage_s)
            sells = provider.get_orders(symbol, "SELL", since_s=maxage_s)
            if not buys and not sells:
                provider.place_order(symbol, "BUY", price, SEED_NOTIONAL_USD / price)
                n_seeds += 1
                continue

            try:
                mt.monitor_price_and_trade(inst, sbs=SBS, now_fn=lambda: provider.now(symbol))
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[{symbol}] eroare in monitor_price_and_trade: {e}\n")

            if n_ticks % 20000 == 0:
                sys.stderr.write(f"[{symbol}] {n_ticks} tick-uri, seed-uri={n_seeds}\n")
    finally:
        mt.is_trend_up = orig_is_trend_up

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
