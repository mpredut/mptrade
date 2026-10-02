"""Characterisation for the Kraken trend overlay.

The overlay is experimental and off by default, but it must preserve the same
financial invariants as the range engine: an order is not a position until it fills,
the stop-loss takes priority, and paper-live reads the real OHLC bars.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KRAKEN_DIR = os.path.join(ROOT, "kraken")
sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from strategies import spot_engine as strat  # noqa: E402
from providers.strategy_executor import PairPrecision  # noqa: E402


def _make_strategy(*, replay_mode=False, **overrides):
    client = MagicMock()
    client.pair_precision.return_value = PairPrecision(2, 6, 0.01, "TEST")
    defaults = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=1000.0, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=0.0,
        reentry_tolerance_pct=0.0, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[], trend_overlay=True,
        trend_sma_n=2, trend_interval=240, regime_min_samples=3,
        trend_topup=400.0, trend_trail_pct=5.0, trend_exit_break=False,
    )
    defaults.update(overrides)
    params = strat.StratParams(**defaults)
    strategy = strat.Strategy(
        client, "TESTPAIR_OVERLAY", params, dry_run=True,
        initial_state=strat._new_state(), replay_mode=replay_mode,
    )
    strategy._save = lambda: None
    return strategy


class TrendOverlayTest(unittest.TestCase):
    def test_replay_never_emits_external_notifications(self):
        notifier = MagicMock()
        s = _make_strategy(replay_mode=True)
        s._notifier = notifier

        s._emit(
            title="SL TESTPAIR -20%", body="test paper",
            source="kraken", symbol="TESTPAIR",
        )

        notifier.assert_not_called()

    def test_paper_live_keeps_external_notifications(self):
        notifier = MagicMock()
        s = _make_strategy(replay_mode=False)
        s._notifier = notifier
        event = {
            "title": "paper-live", "body": "test",
            "source": "kraken", "symbol": "TESTPAIR",
        }

        s._emit(**event)

        notifier.assert_called_once_with(**event)

    def test_topup_enters_trend_mode_only_after_fill(self):
        s = _make_strategy(replay_mode=True)
        s._shadow_prices.extend([(1.0, 100.0), (2.0, 101.0)])

        s.step(102.0)

        order = s._find_open("buy")
        self.assertIsNotNone(order)
        self.assertEqual(order["kind"], "TREND_ENTRY")
        self.assertFalse(s.s["trend_mode"], "an unfilled order is not a trend position yet")

        s._remove(order)
        with patch.object(strat, "notify"):
            s._apply_fill(order, order["vol"], order["price"], fee=0.0)
        self.assertTrue(s.s["trend_mode"])
        self.assertEqual(s.s["trend_peak"], order["price"])

    def test_unfilled_topup_is_cancelled_if_signal_disappears(self):
        s = _make_strategy(replay_mode=True)
        closes = [[100.0, 101.0, 102.0]]
        with patch.object(s, "_trend_closes", side_effect=lambda: closes[0]):
            s.step(102.0)
            self.assertEqual(s._find_open("buy")["kind"], "TREND_ENTRY")

            closes[0] = [102.0, 101.0, 100.0]
            s.step(100.0)

        order = s._find_open("buy")
        self.assertIsNotNone(order, "after the top-up is cancelled it returns to the range logic")
        self.assertEqual(order["kind"], "ENTRY")
        self.assertFalse(s.s["trend_mode"])

    def test_stop_loss_has_priority_over_overlay(self):
        s = _make_strategy(replay_mode=True, stop_loss_pct=10.0, trend_trail_pct=50.0)
        s.s.update({
            "qty": 1.0, "cost": 100.0, "spent": 100.0,
            "entry_price": 100.0, "last_buy_price": 100.0,
            "trend_mode": True, "trend_peak": 100.0,
        })
        s._shadow_prices.extend([(1.0, 100.0), (2.0, 101.0)])

        s.step(80.0)

        sell = s._find_open("sell")
        self.assertIsNotNone(sell)
        self.assertEqual(sell["kind"], "STOP")

    def test_paper_live_uses_client_ohlc_not_tick_shadow(self):
        s = _make_strategy(replay_mode=False)
        s.client.ohlc_closes.return_value = [10.0, 11.0, 12.0]
        s._shadow_prices.extend([(1.0, 99.0), (2.0, 98.0)])

        self.assertEqual(s._trend_closes(), [10.0, 11.0, 12.0])
        s.client.ohlc_closes.assert_called_once_with("TESTPAIR_OVERLAY", 240)

    def test_step_uses_injected_replay_timestamp(self):
        s = _make_strategy(replay_mode=True, trend_overlay=False)
        s.step(100.0, timestamp=14_400.0)
        self.assertEqual(s._shadow_prices[-1], (14_400.0, 100.0))

    def test_adaptive_trail_uses_fixed_closed_ohlc_in_paper_live(self):
        s = _make_strategy(
            replay_mode=False,
            trend_overlay=False,
            tp_trail_adaptive=True,
            tp_trail_vol_interval=240,
        )
        closes = [100.0 + (index % 3) for index in range(20)]
        s.client.ohlc_closes.return_value = closes

        self.assertIsNotNone(s._trail_vol_1h())
        s.client.ohlc_closes.assert_called_once_with("TESTPAIR_OVERLAY", 240)

    def test_adaptive_trail_clamps_and_falls_back_to_fixed(self):
        s = _make_strategy(
            replay_mode=True,
            trend_overlay=False,
            tp_trail_adaptive=True,
            tp_trail_pct=3.0,
            tp_trail_k=2.0,
            tp_trail_min=1.5,
            tp_trail_max=8.0,
        )
        with patch.object(s, "_trail_vol_1h", return_value=None):
            self.assertEqual(s._effective_trail_pct(), 3.0)
        with patch.object(s, "_trail_vol_1h", return_value=0.1):
            self.assertEqual(s._effective_trail_pct(), 1.5)
        with patch.object(s, "_trail_vol_1h", return_value=2.0):
            self.assertEqual(s._effective_trail_pct(), 4.0)
        with patch.object(s, "_trail_vol_1h", return_value=10.0):
            self.assertEqual(s._effective_trail_pct(), 8.0)

    def test_dca_brake_uses_common_regime_from_fixed_ohlc_in_paper_live(self):
        s = _make_strategy(
            replay_mode=False,
            trend_overlay=False,
            dca_trend_brake=True,
            dca_brake_min_pct=1.5,
        )
        s.client.ohlc_closes.return_value = [100.0 - index for index in range(20)]
        s._shadow_prices.extend((float(index), 100.0 + index) for index in range(20))

        decision, _ = s._regime_context()
        self.assertTrue(s._regime_matches(
            decision, "bear", min_move_pct=s.p.dca_brake_min_pct,
            min_samples=20,
        ))
        s.client.ohlc_closes.assert_called_once_with("TESTPAIR_OVERLAY", 240)

    def test_bear_regime_blocks_actual_dca(self):
        s = _make_strategy(
            replay_mode=False,
            trend_overlay=False,
            dca_trend_brake=True,
            dca_brake_min_pct=1.5,
        )
        s.s.update({
            "qty": 1.0,
            "cost": 100.0,
            "spent": 100.0,
            "entry_price": 100.0,
            "last_buy_price": 100.0,
        })
        s.client.ohlc_closes.return_value = [100.0 - index for index in range(20)]

        s.step(95.0)

        self.assertFalse(s._has_open("buy"))

    def test_unknown_regime_preserves_classic_dca_policy(self):
        s = _make_strategy(
            replay_mode=False,
            trend_overlay=False,
            dca_trend_brake=True,
        )
        s.s.update({
            "qty": 1.0,
            "cost": 100.0,
            "spent": 100.0,
            "entry_price": 100.0,
            "last_buy_price": 100.0,
        })
        s.client.ohlc_closes.return_value = []
        s.client.free_balance.return_value = 1000.0

        s.step(95.0)

        self.assertEqual(s._find_open("buy")["kind"], "DCA")

    def test_all_enabled_policies_share_one_ohlc_read_per_step(self):
        s = _make_strategy(
            replay_mode=False,
            tp_trend_hold=True,
            tp_regime_gate=True,
            dca_trend_brake=True,
        )
        s.s.update({
            "qty": 1.0,
            "cost": 100.0,
            "spent": 100.0,
            "entry_price": 100.0,
            "last_buy_price": 100.0,
        })
        s.client.ohlc_closes.return_value = [100.0] * 40

        s.step(105.5)

        s.client.ohlc_closes.assert_called_once_with("TESTPAIR_OVERLAY", 240)

    def test_live_regime_context_reuses_a_bar_and_refreshes_after_settle(self):
        s = _make_strategy(replay_mode=False)
        s.client.ohlc_closes.return_value = [100.0 + index for index in range(42)]
        interval_sec = s.p.trend_interval * 60
        slot_start = 10 * interval_sec

        with patch.object(strat.time, "time", return_value=slot_start + 1) as clock:
            first, _ = s._regime_context()
            clock.return_value = slot_start + 600
            cached, _ = s._regime_context()
            clock.return_value = slot_start + 901
            refreshed, _ = s._regime_context()
            clock.return_value = slot_start + 1200
            stable, _ = s._regime_context()

        self.assertEqual(first, cached)
        self.assertEqual(refreshed, stable)
        self.assertEqual(s.client.ohlc_closes.call_count, 2)

    def test_unavailable_regime_context_is_not_cached_for_the_whole_bar(self):
        s = _make_strategy(replay_mode=False)
        s.client.ohlc_closes.return_value = []
        with patch.object(strat.time, "time", return_value=1_000_000):
            s._regime_context()
            s._regime_context()
        self.assertEqual(s.client.ohlc_closes.call_count, 2)

    def test_trend_topup_rejects_non_finite_or_non_positive_sizing(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            with self.subTest(trend_topup=invalid), self.assertRaisesRegex(
                ValueError, "STRAT_TREND_TOPUP",
            ):
                _make_strategy(trend_topup=invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
