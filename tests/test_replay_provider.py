"""
Tests for ReplayMarketDataProvider (offline/research/UNIFIED_BACKTEST_PLAN.md
Phase 1 — implementation step 1: monitortrades.py testable on historical data,
without touching the network).

Coverage:
  - load_price_series: correctly parses the real cache_price_{symbol}.jsonl format.
  - ReplayMarketDataProvider: advance, get_current_price, now() (= timestamp of
    the last price read, NOT wall clock), get_orders/free_balance after BUY/SELL.
  - Integration: monitortrades.monitor_price_and_trade() runs a complete cycle
    (an old BUY -> the price rises past gain_threshold -> SELL) through an Instrument
    built with api=MarketApi([replay]) — WITHOUT any change to the real code
    from monitortrades.py (only now_fn injected).

We use a SYNTHETIC symbol ("ZZZTESTUSDC") — not a real one tracked by the live fleet
(BTCUSDC/TAOUSDC) — so that is_trend_up() is deterministically FALSE (cacheManager has
no snapshot for a non-existent symbol) rather than depending on the REAL trend on
the market when the test runs.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from providers.replay_provider import ReplayMarketDataProvider, load_price_series, _base_asset
from providers.market_api import MarketApi
from instrument import Instrument
import monitortrades as mt

SYMBOL = "ZZZTESTUSDC"


def _make_provider(prices, start_ts=1_000_000.0, step_s=60.0):
    """prices: a list of prices -> a (ts, price) series with a fixed step of step_s seconds."""
    series = [(start_ts + i * step_s, p) for i, p in enumerate(prices)]
    return ReplayMarketDataProvider({SYMBOL: series})


class TestLoadPriceSeries(unittest.TestCase):

    def test_parses_real_cache_format(self):
        tmpdir = tempfile.mkdtemp(prefix="replay_series_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
        path = os.path.join(tmpdir, "price_series.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"s": "BTCUSDC", "i": [1756248610634, 111702.25]}\n')
            f.write('{"s": "TAOUSDC", "i": [1756248611000, 200.0]}\n')   # other symbol, ignored
            f.write('{"s": "BTCUSDC", "i": [1756248620944, 111720.0]}\n')
            f.write('not json at all\n')                                 # corrupt line, ignored
        series = load_price_series(path, "BTCUSDC")
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0], (1756248610634 / 1000.0, 111702.25))
        self.assertEqual(series[1], (1756248620944 / 1000.0, 111720.0))

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_price_series("/tmp/does_not_exist_xyz.jsonl", "BTCUSDC"), [])


class TestBaseAsset(unittest.TestCase):

    def test_strips_known_quote_suffixes(self):
        self.assertEqual(_base_asset("BTCUSDC"), "BTC")
        self.assertEqual(_base_asset("TAOUSDC"), "TAO")
        self.assertEqual(_base_asset("HYPEUSD"), "HYPE")

    def test_symbol_without_known_suffix_unchanged(self):
        self.assertEqual(_base_asset("WEIRD"), "WEIRD")


class TestReplayMarketDataProvider(unittest.TestCase):

    def test_price_none_before_first_advance(self):
        p = _make_provider([100.0, 105.0])
        self.assertIsNone(p.get_current_price(SYMBOL))

    def test_advance_returns_sequential_prices(self):
        p = _make_provider([100.0, 105.0, 110.0])
        self.assertEqual(p.advance(SYMBOL), 100.0)
        self.assertEqual(p.get_current_price(SYMBOL), 100.0)
        self.assertEqual(p.advance(SYMBOL), 105.0)
        self.assertEqual(p.advance(SYMBOL), 110.0)
        self.assertIsNone(p.advance(SYMBOL))   # end of series

    def test_now_reflects_last_read_timestamp_not_wallclock(self):
        p = _make_provider([100.0, 105.0], start_ts=1_000_000.0, step_s=60.0)
        p.advance(SYMBOL)
        self.assertEqual(p.now(SYMBOL), 1_000_000.0)
        p.advance(SYMBOL)
        self.assertEqual(p.now(SYMBOL), 1_000_060.0)

    def test_place_buy_then_sell_updates_position_and_balance(self):
        p = _make_provider([100.0, 100.0])
        p.advance(SYMBOL)
        p.place_order(SYMBOL, "BUY", 100.0, 2.0)
        self.assertEqual(p.free_balance("ZZZTEST"), 2.0)
        p.advance(SYMBOL)
        p.place_order(SYMBOL, "SELL", 110.0, 2.0)
        self.assertEqual(p.free_balance("ZZZTEST"), 0.0)

    def test_oversized_sell_records_only_the_executed_quantity(self):
        p = _make_provider([100.0, 110.0])
        p.advance(SYMBOL)
        p.place_order(SYMBOL, "BUY", 100.0, 1.0)
        p.advance(SYMBOL)
        p.place_order(SYMBOL, "SELL", 110.0, 2.0)

        sells = p.get_orders(SYMBOL, "SELL", since_s=1e9)
        self.assertEqual(sells[0]["qty"], 1.0)
        self.assertEqual(p.position(SYMBOL), (0.0, 0.0))

    def test_sell_without_holdings_is_refused_and_not_recorded(self):
        p = _make_provider([110.0])
        p.advance(SYMBOL)

        self.assertIsNone(p.place_order(SYMBOL, "SELL", 110.0, 1.0))
        self.assertEqual(p.get_orders(SYMBOL, "SELL", since_s=1e9), [])
        self.assertEqual(p.position(SYMBOL), (0.0, 0.0))

    def test_get_orders_filters_by_side_and_window(self):
        p = _make_provider([100.0] * 5, start_ts=1_000_000.0, step_s=3600.0)  # 1h/step
        p.advance(SYMBOL)                       # ts=1_000_000
        p.place_order(SYMBOL, "BUY", 100.0, 1.0)
        for _ in range(4):
            p.advance(SYMBOL)                    # advance up to ts=1_014_400 (4h later)
        recent = p.get_orders(SYMBOL, "BUY", since_s=2 * 3600)   # only the last 2h
        self.assertEqual(recent, [])             # BUY is 4h ago -> outside window
        all_buys = p.get_orders(SYMBOL, "BUY", since_s=10 * 3600)
        self.assertEqual(len(all_buys), 1)

    def test_guards_internally_true_phase1_simplification(self):
        p = _make_provider([100.0])
        self.assertTrue(p.guards_internally())


class TestNowFnDefaultsToRealTime(unittest.TestCase):
    """Regression: without now_fn (the LIVE path, unchanged), get_relevant_trade and
    monitor_price_and_trade must keep using real time.time()
    — verified directly, not merely assumed by reading the code."""

    def test_get_relevant_trade_uses_real_time_by_default(self):
        import time as real_time
        trade_orders = [{"timestamp": int(real_time.time() * 1000) - 1000, "price": 42.0}]
        _, trade_time, _ = mt.get_relevant_trade(trade_orders, "BUY", threshold_s=3600, symbol=SYMBOL)
        # trade_time must be ~now (one second ago), not something arbitrary
        self.assertAlmostEqual(trade_time, real_time.time(), delta=5)


class TestMonitorPriceAndTradeIntegration(unittest.TestCase):
    """A complete cycle through monitortrades.monitor_price_and_trade(), on an
    Instrument built with api=replay — WITHOUT any code change in
    monitortrades.py (only now_fn, already injected today)."""

    def test_full_cycle_buy_then_gain_triggers_sell(self):
        # Price: 100 (a BUY here) -> it stays at 100 for a few steps -> it rises to 115 (+15%,
        # above the default gain_threshold of 7%) -> monitor_price_and_trade must
        # sell (is_trend_up is deterministically False: a synthetic symbol, with no
        # snapshot in cacheManager).
        prices = [100.0, 100.0, 100.0, 115.0]
        provider = _make_provider(prices, start_ts=2_000_000.0, step_s=3600.0)
        provider.advance(SYMBOL)                              # ts=2_000_000, price=100
        provider.place_order(SYMBOL, "BUY", 100.0, 1.0)        # an old BUY, recorded

        api = MarketApi([provider])
        inst = Instrument(name="ZZZTEST", symbol=SYMBOL, provider="replay",
                          base="ZZZTEST", quote="USDC", api=api)

        provider.advance(SYMBOL)                              # ts+3600, price=100 (no change)
        provider.advance(SYMBOL)                              # ts+7200, price=100
        provider.advance(SYMBOL)                              # ts+10800, price=115 (+15%)

        mt.monitor_price_and_trade(inst, sbs=3600, maxage_trade_s=100000,
                                    now_fn=lambda: provider.now(SYMBOL))

        sells = provider.get_orders(SYMBOL, "SELL", since_s=1e9)
        self.assertEqual(len(sells), 1, "it should have sold after +15% above gain_threshold")
        self.assertAlmostEqual(sells[0]["price"], 115.0, places=2)


class TestSimClock(unittest.TestCase):
    """Formerly test_replay_clock.py (unified 28 Jul) — SimClock, the other piece
    of the replay infrastructure in providers/, shared by tradeall and monitortrades."""

    def test_callable_returns_ts(self):
        from providers.replay_clock import SimClock
        clock = SimClock()
        clock.ts = 12345.0
        self.assertEqual(clock(), 12345.0)

    def test_advancing_ts_reflected_immediately(self):
        from providers.replay_clock import SimClock
        clock = SimClock()
        clock.ts = 100.0
        self.assertEqual(clock(), 100.0)
        clock.ts = 200.0
        self.assertEqual(clock(), 200.0)

    def test_tradeall_backtest_uses_shared_class(self):
        """tradeall_backtest._SimClock must be the SAME type as
        providers.replay_clock.SimClock (not a separate reimplementation)."""
        from offline.backtests import tradeall as tb
        from providers.replay_clock import SimClock
        self.assertIs(tb._SimClock, SimClock)


if __name__ == "__main__":
    unittest.main()
