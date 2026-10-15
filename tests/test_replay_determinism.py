"""Tests verifying backtest and replay determinism, wall-clock immunity, and cache isolation."""
from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.position_exit_manager import (
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)
from strategies.spot_engine import StratParams, Strategy
from strategies.spot_replay import run_replay


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
    d = dict(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=1.5,
        check_minutes=5.0,
        takeprofit_pct=3.0,
        max_budget=500.0,
        max_dca_buys=3,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        stop_loss_pct=8.0,
        adopt_cost=0.0,
        adopt_qty=0.0,
        reentry_drop_pct=0.0,
        reentry_tolerance_pct=0.05,
        reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5,
        tp_tranches=[],
    )
    d.update(over)
    return StratParams(**d)


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
        class TimeWarp:
            def __init__(self):
                self._t = 2_000_000_000.0

            def __call__(self):
                self._t += 137.5
                return self._t

        with patch("time.time", side_effect=TimeWarp()):
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
            "strategies.spot_engine.resolve_cached_volatility_1h",
            return_value=0.50,  # 50% hourly volatility!
        ), patch(
            "strategies.spot_engine.resolve_cached_volatility_1h_pct",
            return_value=50.0,
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
        import importlib.util
        t212_dir = os.path.join(ROOT, "212trading")
        colliding = ("strategy", "market_data", "notify", "ipo_notify", "replay")
        preloaded = {name: sys.modules.pop(name) for name in colliding if name in sys.modules}
        try:
            if t212_dir in sys.path:
                sys.path.remove(t212_dir)
            sys.path.insert(0, t212_dir)
            _md_spec = importlib.util.spec_from_file_location("t212_md_test", os.path.join(t212_dir, "market_data.py"))
            _md = importlib.util.module_from_spec(_md_spec)
            sys.modules["market_data"] = _md
            _md_spec.loader.exec_module(_md)

            spec = importlib.util.spec_from_file_location(
                "t212_strat_test", os.path.join(t212_dir, "strategy.py")
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            sys.modules["strategy"] = mod
            spec.loader.exec_module(mod)
        finally:
            for name in colliding:
                sys.modules.pop(name, None)
            sys.modules.update(preloaded)

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

        class SimClock:
            def __call__(self):
                return 1000.0

        strat = mod.Strategy(
            client=client,
            ticker="AAPL_US_EQ",
            params=params,
            dry_run=True,
            clock=SimClock(),
        )

        self.assertTrue(strat.replay_mode)
        self.assertFalse(strat.exit_manager.config.allow_external_cache)
        self.assertIsNotNone(strat.exit_manager.vol_resolver)

        # Even with mocked external cached volatility returning 0.35, replay must ignore it
        with patch.object(mod, "resolve_cached_volatility_1h", return_value=0.35), \
             patch.object(mod, "resolve_cached_volatility_1h_pct", return_value=35.0):
            vol_ratio, vol_src = strat._vol_1h_ratio()
            self.assertEqual(vol_ratio, 0.0)
            self.assertEqual(vol_src, "none")

            val, src = strat._effective_reentry_pullback_pct()
            self.assertIn("fallback", src)


if __name__ == "__main__":
    unittest.main()
