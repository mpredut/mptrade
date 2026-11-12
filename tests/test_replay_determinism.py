"""Tests verifying backtest and replay determinism, wall-clock immunity, and cache isolation."""

import json
import monitortrades as mt
import os
import pytest
import time
import unittest
from binance_api.market_data import fetch_ohlc_candles, parse_interval_minutes
from instrument import Instrument
from offline.backtests import tradeall as tb
from offline.backtests.evaluation import evaluate_segment, evaluate_walk_forward
from offline.backtests.metrics import validate_replay_metrics
from providers.market_api import MarketApi
from providers.replay_provider import (
    ReplayMarketDataProvider,
    SimClock,
    _base_asset,
    load_price_series,
)
from strategies.position_exit_manager import (
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)
from strategies.spot_engine import StratParams, Strategy
from strategies.spot_replay import make_spot_replay_adapter, run_replay
from tests.conftest import TempDirTestCase, load_venue_modules, make_spot_test_params
from unittest.mock import MagicMock, patch




def _sample_ohlc(length: int = 40, base_price: float = 100.0) -> list[tuple[float, float, float, float]]:
    bars = []
    p = base_price
    for i in range(length):
        # Oscillate to trigger DCA buys, rebounds, and exits
        if i < 10:
            delta = -1.2
        elif i < 20:
            delta = 1.5
        elif i < 30:
            delta = -2.0
        else:
            delta = 1.8
        open_ = p
        close = p + delta
        high = max(open_, close) + 0.5
        low = min(open_, close) - 0.5
        bars.append((open_, high, low, close))
        p = close
    return bars


def _make_params(**over) -> StratParams:
    defaults = dict(
        dca_drop_pct=1.5,
        takeprofit_pct=3.0,
        order_ttl_min=10.0,
        stop_loss_pct=8.0,
        reentry_sl_bounce_pct=1.5,
    )
    defaults.update(over)
    return make_spot_test_params(**defaults)


class ReplayDeterminismTest(unittest.TestCase):
    def test_spot_replay_wallclock_independence(self):
        """Spot replay must yield bit-for-bit identical results regardless of wall-clock time."""
        bars = _sample_ohlc(40, 100.0)
        params = _make_params(
            sl_mhdsl_enabled=True,
            tp_mdhtp_enabled=True,
        )

        # Run 1: with standard time.time()
        res1 = run_replay(
            bars,
            params,
            bar_minutes=5.0,
            include_decision_trace=True,
        )

        # Run 2: monkeypatch time.time to return an arbitrary future timestamp that advances arbitrarily
        warp_t = [2_000_000_000.0]

        def _time_warp():
            warp_t[0] += 137.5
            return warp_t[0]

        with patch("time.time", side_effect=_time_warp):
            res2 = run_replay(
                bars,
                params,
                bar_minutes=5.0,
                include_decision_trace=True,
            )

        self.assertEqual(res1["fills"], res2["fills"])
        self.assertEqual(res1["cycles"], res2["cycles"])
        self.assertEqual(res1["wins"], res2["wins"])
        self.assertAlmostEqual(res1["net"], res2["net"], places=9)
        self.assertAlmostEqual(res1["maxdd"], res2["maxdd"], places=9)
        self.assertEqual(res1["decision_trace"], res2["decision_trace"])

    def test_spot_replay_external_cache_immunity(self):
        """Historical replay must never read disk cache volatility or alter its behavior."""
        bars = _sample_ohlc(40, 100.0)
        params = _make_params(
            sl_mhdsl_enabled=True,
            reentry_adaptive=True,
            tp_trail_adaptive=True,
            tp_trail_vol_interval=5,
        )

        res_clean = run_replay(
            bars,
            params,
            bar_minutes=5.0,
            include_decision_trace=True,
        )

        # Inject extreme fake volatility in shadow cache resolver
        with patch(
            "intelligence.internal.state.volatility.resolve_cached_volatility_1h",
            return_value=0.50,  # 50% hourly volatility!
        ), patch(
            "intelligence.internal.state.volatility.resolve_cached_volatility_1h_pct",
            return_value=50.0,
        ), patch(
            "strategies.position_exit_manager.resolve_cached_volatility_1h",
            return_value=0.50,
        ):
            res_polluted = run_replay(
                bars,
                params,
                bar_minutes=5.0,
                include_decision_trace=True,
            )

        self.assertEqual(res_clean["fills"], res_polluted["fills"])
        self.assertEqual(res_clean["cycles"], res_polluted["cycles"])
        self.assertEqual(res_clean["decision_trace"], res_polluted["decision_trace"])
        self.assertAlmostEqual(res_clean["net"], res_polluted["net"], places=9)

    def test_position_exit_manager_zero_timestamp_fidelity(self):
        """PositionExitManager must treat current_time_s=0.0 as a valid timestamp and avoid wall-clock fallback."""
        cfg = PositionExitConfig(
            lost_threshold=0.05,
            gain_threshold=0.05,
            sl_bounce_min_recovery_pct=0.01,
            sl_bounce_pullback_pct=0.01,
            reentry_enabled=True,
            allow_external_cache=False,
        )
        mgr = PositionExitManager(config=cfg)

        snap = PositionSnapshot(
            symbol="BTCUSDC",
            current_price=94.0,
            avg_buy_price=100.0,
            net_qty=1.0,
            sellable_qty=1.0,
            current_time_s=0.0,  # Explicit 0.0 timestamp at the start of a simulation
            trend_up=False,
            vol_1h=0.02,
            vol_source="test",
            provider="test",
            shadow_prices=[(0.0, 94.0)],
        )

        def forbidden_time():
            raise AssertionError("time.time() was invoked despite current_time_s being explicitly provided!")

        with patch("time.time", side_effect=forbidden_time):
            # 1. evaluate_stop_loss
            decision_sl = mgr.evaluate_stop_loss(snap)
            self.assertIsNotNone(decision_sl)

            # 2. evaluate_take_profit
            snap_tp = PositionSnapshot(
                symbol="BTCUSDC",
                current_price=106.0,
                avg_buy_price=100.0,
                net_qty=1.0,
                sellable_qty=1.0,
                current_time_s=0.0,
                trend_up=True,
                vol_1h=0.02,
                vol_source="test",
                provider="test",
                shadow_prices=[(0.0, 106.0)],
            )
            decision_tp = mgr.evaluate_take_profit(snap_tp)
            self.assertIsNotNone(decision_tp)

            # 3. evaluate_reentry
            re_snap = ReentrySnapshot(
                symbol="BTCUSDC",
                current_price=98.0,
                last_sell_price=105.0,
                last_exit_kind="TP",
                current_time_s=0.0,
                trend_up=True,
                vol_1h=0.02,
                vol_source="test",
            )
            decision_re = mgr.evaluate_reentry(re_snap)
            self.assertIsNotNone(decision_re)

    def test_strategy_simulated_ts_provider(self):
        """Strategy._now() must reflect set_simulated_time in replay mode and time.time in live mode."""
        client = MagicMock()
        client.pair_precision.return_value = MagicMock(price_decimals=2, volume_decimals=4, order_min=0.001)
        params = _make_params()

        # In replay_mode:
        strat_replay = Strategy(
            client=client,
            pair="BTCUSD",
            params=params,
            dry_run=True,
            replay_mode=True,
        )
        self.assertEqual(strat_replay._now(), 0.0)

        strat_replay.set_simulated_time(12345.67)
        self.assertEqual(strat_replay._now(), 12345.67)

        strat_replay._place("buy", 0.01, 50000.0, "ENTRY")
        self.assertEqual(strat_replay.s["orders"][-1]["ts"], 12345.67)

        # In live mode (replay_mode=False):
        strat_live = Strategy(
            client=client,
            pair="BTCUSD",
            params=params,
            dry_run=False,
            replay_mode=False,
        )
        t_before = time.time()
        now_live = strat_live._now()
        t_after = time.time()
        self.assertTrue(t_before <= now_live <= t_after)

    def test_t212_strategy_replay_cache_immunity(self):
        """T212 strategy in replay mode must isolate from external volatility cache."""
        mod = load_venue_modules("212trading", "strategy")

        client = MagicMock()
        client.get_order.return_value = None
        params = mod.StratParams(
            currency="USD",
            entry_amount=100.0,
            entry_discount_pct=0.2,
            dca_amount=50.0,
            dca_drop_pct=2.0,
            check_minutes=5.0,
            takeprofit_pct=3.0,
            max_budget=500.0,
            max_dca_buys=3,
            validity="DAY",
            enable_takeprofit=True,
            order_ttl_min=10.0,
            stop_loss_pct=8.0,
            sl_mhdsl_enabled=True,
            tp_mdhtp_enabled=True,
            reentry_pullback_adaptive=True,
        )

        strat = mod.Strategy(
            client=client,
            ticker="AAPL_US_EQ",
            params=params,
            dry_run=True,
            clock=lambda: 1000.0,
        )

        self.assertTrue(strat.replay_mode)
        self.assertFalse(strat.exit_manager.config.allow_external_cache)
        self.assertIsNotNone(strat.exit_manager.vol_resolver)

        # Even with mocked external cached volatility returning 0.35, replay must ignore it
        with patch("intelligence.internal.state.volatility.resolve_cached_volatility_1h", return_value=0.35), \
             patch("intelligence.internal.state.volatility.resolve_cached_volatility_1h_pct", return_value=35.0):
            vol_ratio, vol_src = strat._vol_1h_ratio()
            self.assertEqual(vol_ratio, 0.0)
            self.assertEqual(vol_src, "none")

            val, src = strat._effective_reentry_pullback_pct()
            self.assertIn("fallback", src)


# ---------------------------------------------------------------------------
# Consolidated from tests/test_binance_backtest.py & tests/test_replay_provider.py
# ---------------------------------------------------------------------------

SYMBOL = "ZZZTESTUSDC"


def _sample_params(**overrides) -> StratParams:
    return make_spot_test_params(**{
        "currency": "USDT",
        "takeprofit_pct": 1.0,
        "max_dca_buys": 2,
        "check_minutes": 1.0,
        **overrides,
    })


def test_interval_parsing():
    assert parse_interval_minutes(1) == 1
    assert parse_interval_minutes(60) == 60
    assert parse_interval_minutes("1h") == 60
    assert parse_interval_minutes("4h") == 240
    assert parse_interval_minutes("1d") == 1440

    with pytest.raises(ValueError):
        parse_interval_minutes(0)
    with pytest.raises(ValueError):
        parse_interval_minutes("invalid_interval")
    with pytest.raises(ValueError):
        fetch_ohlc_candles("BTCUSDC", interval=17)


def test_fetch_ohlc_candles_mocked():
    now_ms = 1700000000000
    mock_batch = [
        [now_ms - 7200000, "50000.0", "50500.0", "49500.0", "50200.0", "10.0", now_ms - 3600001, "0", 0, "0", "0", "0"],
        [now_ms - 3600000, "50200.0", "51000.0", "50100.0", "50800.0", "15.0", now_ms - 1, "0", 0, "0", "0", "0"],
        [now_ms, "50800.0", "51200.0", "50700.0", "51100.0", "5.0", now_ms + 3599999, "0", 0, "0", "0", "0"],
    ]

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_batch).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch("time.time", return_value=now_ms / 1000.0):
        candles = fetch_ohlc_candles("BTCUSDT", interval="1h", start_ms=now_ms - 8000000, end_ms=now_ms + 4000000, drop_incomplete=True)
        assert len(candles) == 2
        assert candles[0]["timestamp"] == (now_ms - 7200000) // 1000
        assert candles[0]["open"] == 50000.0
        assert candles[0]["close"] == 50200.0
        assert candles[1]["close"] == 50800.0

        all_candles = fetch_ohlc_candles("BTCUSDT", interval="1h", start_ms=now_ms - 8000000, end_ms=now_ms + 4000000, drop_incomplete=False)
        assert len(all_candles) == 3


def test_binance_replay_metrics_contract():
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 99.0, 101.0),
        (101.0, 103.0, 100.5, 102.5),
    ]
    params = _sample_params()
    res = run_replay(ohlc, params, fee_pct=0.10, bar_minutes=60.0)

    validate_replay_metrics(res)
    assert res["account_currency"] == "USDT"
    assert res["fees"] >= 0.0
    assert "return_pct" in res
    assert "sharpe" in res
    assert "calmar" in res
    assert "sortino" in res


def test_binance_adapter_in_evaluate_walk_forward():
    records = [
        {"timestamp": 1000 + i * 3600, "open": 100.0 + (i % 5), "high": 102.0 + (i % 5),
         "low": 98.0 + (i % 5), "close": 100.5 + (i % 5)}
        for i in range(60)
    ]
    params = _sample_params()
    adapter = make_spot_replay_adapter(params, fee_pct=0.10, bar_minutes=60.0)

    seg = evaluate_segment(records[:20], adapter, validate_contract=True)
    assert seg["bars"] == 20
    validate_replay_metrics(seg["metrics"])

    wf = evaluate_walk_forward(
        records,
        adapter,
        train_size=20,
        validation_size=10,
        test_size=10,
        step_size=10,
        validate_contract=True,
    )
    assert wf["aggregate_test"]["fold_count"] > 0
    assert "mean_return_pct" in wf["aggregate_test"]


def test_cross_venue_alignment_matrix():
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 99.0, 101.0),
        (101.0, 102.5, 100.0, 102.0),
    ]

    bn_adapter = make_spot_replay_adapter(_sample_params(), fee_pct=0.10, bar_minutes=60.0)
    bn_res = bn_adapter(ohlc, ())
    validate_replay_metrics(bn_res)

    kr_params = _sample_params(currency="USD")
    kr_adapter = make_spot_replay_adapter(kr_params, fee_pct=0.25, bar_minutes=60.0)
    kr_res = kr_adapter(ohlc, ())
    validate_replay_metrics(kr_res)

    if bn_res["fills"] > 0 and kr_res["fills"] > 0:
        assert bn_res["fees"] < kr_res["fees"]


def _make_provider(prices, start_ts=1_000_000.0, step_s=60.0):
    series = [(start_ts + i * step_s, p) for i, p in enumerate(prices)]
    return ReplayMarketDataProvider({SYMBOL: series})


class TestLoadPriceSeries(TempDirTestCase):
    def test_parses_real_cache_format(self):
        path = os.path.join(self.tmp, "price_series.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"s": "BTCUSDC", "i": [1756248610634, 111702.25]}\n')
            f.write('{"s": "TAOUSDC", "i": [1756248611000, 200.0]}\n')
            f.write('{"s": "BTCUSDC", "i": [1756248620944, 111720.0]}\n')
            f.write('not json at all\n')
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
        self.assertIsNone(p.advance(SYMBOL))

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
        p = _make_provider([100.0] * 5, start_ts=1_000_000.0, step_s=3600.0)
        p.advance(SYMBOL)
        p.place_order(SYMBOL, "BUY", 100.0, 1.0)
        for _ in range(4):
            p.advance(SYMBOL)
        recent = p.get_orders(SYMBOL, "BUY", since_s=2 * 3600)
        self.assertEqual(recent, [])
        all_buys = p.get_orders(SYMBOL, "BUY", since_s=10 * 3600)
        self.assertEqual(len(all_buys), 1)

    def test_guards_internally_true_phase1_simplification(self):
        p = _make_provider([100.0])
        self.assertTrue(p.guards_internally())


class TestNowFnDefaultsToRealTime(unittest.TestCase):
    def test_get_relevant_trade_uses_real_time_by_default(self):
        trade_orders = [{"timestamp": int(time.time() * 1000) - 1000, "price": 42.0}]
        _, trade_time, _ = mt.get_relevant_trade(trade_orders, "BUY", symbol=SYMBOL)
        self.assertAlmostEqual(trade_time, time.time(), delta=5)


class TestMonitorPriceAndTradeIntegration(unittest.TestCase):
    def test_full_cycle_buy_then_gain_triggers_sell(self):
        prices = [100.0, 100.0, 100.0, 115.0]
        provider = _make_provider(prices, start_ts=2_000_000.0, step_s=3600.0)
        provider.advance(SYMBOL)
        provider.place_order(SYMBOL, "BUY", 100.0, 1.0)

        api = MarketApi([provider])
        inst = Instrument(name="ZZZTEST", symbol=SYMBOL, provider="replay",
                          base="ZZZTEST", quote="USDC", api=api)

        provider.advance(SYMBOL)
        provider.advance(SYMBOL)
        provider.advance(SYMBOL)

        mt.monitor_price_and_trade(inst, sbs=3600, maxage_trade_s=100000,
                                    now_fn=lambda: provider.now(SYMBOL))

        sells = provider.get_orders(SYMBOL, "SELL", since_s=1e9)
        self.assertEqual(len(sells), 1, "it should have sold after +15% above gain_threshold")
        self.assertAlmostEqual(sells[0]["price"], 115.0, places=2)


class TestSimClock(unittest.TestCase):
    def test_callable_returns_ts(self):
        clock = SimClock()
        clock.ts = 12345.0
        self.assertEqual(clock(), 12345.0)

    def test_advancing_ts_reflected_immediately(self):
        clock = SimClock()
        clock.ts = 100.0
        self.assertEqual(clock(), 100.0)
        clock.ts = 200.0
        self.assertEqual(clock(), 200.0)

    def test_tradeall_backtest_uses_shared_class(self):
        self.assertIs(tb.SimClock, SimClock)


if __name__ == "__main__":
    unittest.main()
