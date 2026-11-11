"""
Tests for tradeall.py — PriceWindow, PriceTrendAnalyzer, TrendState
and the integration with Cache24PriceManager.

Coverage:
  - PriceTrendAnalyzer: linreg, gradient, insufficient data
  - PriceWindow: sample_rate_sec, recent_n, process_price, get_trend (4 values)
  - PriceWindow.from_cache24: the factory from Cache24PriceManager with real data
  - PriceWindow._sample_rate_from_entries: rate computed from timestamps
  - TrendState: complete lifecycle
  - TrendState: cooldown per trend instance (fire_limit_reached/mark_confirmed
    with FIRE_MAX_PER_TREND executions, can_retry_fire/mark_fire_attempt with
    FIRE_MIN_RETRY_INTERVAL_SEC, reset on new start_trend) — Jul 22
  - CacheCurrentPriceManager: get_sample_rate / get_update_frequency
"""
import os, sys, json, time, tempfile, threading, unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import order_retry
import pricewindow as pw_mod
import pytest
from tests.conftest import TempDirTestCase, make_cache24_test_manager, make_synthetic_price_entries

# ── mocks for the external dependencies (before any local import) ─────────
_mock_bapi = MagicMock()
_mock_bapi.get_current_price = MagicMock(return_value=60000.0)
_mock_bapi.cancel_order      = MagicMock(return_value=True)
_mock_bapi.cancel_expired_orders = MagicMock()
_mock_bapi.client            = MagicMock()

_mock_market_api = MagicMock()
_mock_market_api.api = _mock_bapi

_IMPORT_MOCKS = dict([
    ("bapi",            _mock_bapi),
    ("binance_api.bapi", _mock_bapi),
    ("bapi_trades",     MagicMock(**{"get_my_trades_24.return_value": []})),
    ("binance_api.bapi_trades", MagicMock(**{"get_my_trades_24.return_value": []})),
    ("bapi_allorders",  MagicMock()),
    ("binance_api.bapi_allorders", MagicMock()),
    ("bapi_placeorder", MagicMock()),
    ("binance_api.bapi_placeorder", MagicMock()),
    ("providers.market_api", _mock_market_api),
    ("alertnotifiers",  MagicMock()),
    ("log",             MagicMock()),
    ("credentials",     MagicMock(**{"api_key_ws": "fake"})),
])
_PRELOADED_IMPORTS = {
    name: sys.modules[name] for name in _IMPORT_MOCKS if name in sys.modules
}
_BINANCE_CHILDREN = ("bapi", "bapi_trades", "bapi_allorders", "bapi_placeorder")
sys.modules.update(_IMPORT_MOCKS)

try:
    import cacheManager as cm
    import tradeall as ta
finally:
    # pytest collection imports every file before running the tests. Without
    # restoring, the venue mocks would change the Binance tests collected earlier.
    for _name in _IMPORT_MOCKS:
        sys.modules.pop(_name, None)
    sys.modules.update(_PRELOADED_IMPORTS)
    _binance_package = sys.modules.get("binance_api")
    if _binance_package is not None:
        for _child in _BINANCE_CHILDREN:
            _full_name = f"binance_api.{_child}"
            if _full_name in _PRELOADED_IMPORTS:
                setattr(_binance_package, _child, _PRELOADED_IMPORTS[_full_name])
            elif getattr(_binance_package, _child, None) is _IMPORT_MOCKS[_full_name]:
                delattr(_binance_package, _child)
    import cacheaccount as _ca_mod
    if getattr(_ca_mod, "apiorders", None) is _IMPORT_MOCKS["binance_api.bapi_allorders"]:
        from binance_api import bapi_allorders as _real_bapi_allorders
        _ca_mod.apiorders = _real_bapi_allorders

mock_bapi = _mock_bapi
from pricewindow import PriceWindow, WindowAnalyzer


# ═══════════════════════════════════════════════════════════════════════════
# Tradeall order boundary
# ═══════════════════════════════════════════════════════════════════════════

class TestTradeallOrderBoundary(unittest.TestCase):

    def test_unsafe_runtime_configuration_fails_closed(self):
        fields = (
            "TREND_TO_BE_OLD_SECONDS", "PRICE_CHANGE_THRESHOLD_EUR",
            "PRICE_CHANGE_THRESHOLD_BIG_EUR", "TREND_UNIFORM_RATE_THRESHOLD",
            "SLOPE_EXTREME_THRESHOLD", "FIRE_MIN_RETRY_INTERVAL_SEC",
            "FIRE_SAFEBACK_SEC", "FIRE_MAX_PER_TREND",
            "TREND_MIN_VALIDATED_SECONDS", "TREND_MIN_VALIDATED_CONFIRMS",
            "TREND_CONSISTENT_CONFIRMS",
        )
        for field in fields:
            with self.subTest(field=field), patch.object(ta, field, -1):
                with self.assertRaises(ValueError):
                    ta._validate_tradeall_config()

    def test_fire_order_allowlist_and_retry_policy(self):
        with (patch.object(ta, "_kalman_gate_blocks", return_value=(False, "off", None)),
              patch.object(ta, "TRADEALL_FIRE_SYMBOLS", {"BTCUSDT"}),
              patch.object(ta.mkt, "place", return_value={"orderId": "accepted"}) as place):
            result = ta._fire_order("BTCUSDT", "BUY", 100.0, "test")
            self.assertIsNotNone(result)
            self.assertNotIn("caller_owns_retry", place.call_args.kwargs)

            place.reset_mock()
            blocked = ta._fire_order("ARBUSDC", "SELL", 0.17, "test")
            allowed = ta._fire_order("BTCUSDT", "SELL", 100.0, "test")
            self.assertIsNone(blocked)
            self.assertIsNotNone(allowed)
            self.assertEqual(place.call_count, 1)
            self.assertEqual(place.call_args.args[0], "BTCUSDT")

    def test_invalid_side_or_price_never_reaches_executor(self):
        invalid = (("HOLD", 100.0), ("BUY", None), ("SELL", float("nan")),
                   ("BUY", float("inf")), ("SELL", 0), ("BUY", -1))
        with patch.object(ta.mkt, "place") as place:
            for side, price in invalid:
                with self.subTest(side=side, price=price):
                    self.assertIsNone(ta._fire_order("BTCUSDT", side, price, "test"))
        place.assert_not_called()

    def test_disabled_logic_does_not_consume_retry_slot(self):
        state = MagicMock()
        state.is_trend_up.return_value = 1
        state.is_trend_down.return_value = 0
        state.is_trend_uniform_confirmed.return_value = True
        state.is_trend_fresh.return_value = True
        state.is_trend_consistent_validated.return_value = False
        state.is_started_trend_older_than.return_value = False
        state.fire_limit_reached.return_value = False
        state.can_retry_fire.return_value = True

        with patch.object(ta, "_fire_order") as fire:
            ta.logic("BIG", False, "BTCUSDT", 1.0, -1.0, state, 100.0)

        state.mark_fire_attempt.assert_not_called()
        fire.assert_not_called()

    def test_kalman_gate_behaviors(self):
        with self.subTest("fails closed for blind buy but allows sell"):
            shadows = (
                SimpleNamespace(current_trend=lambda _sym: (None, 0.0)),
                SimpleNamespace(current_trend=lambda _sym: (1, ta.GATE_STALE_SEC + 1.0)),
            )
            for shadow in shadows:
                with (patch.object(ta, "_kalman_ref", shadow),
                      patch.dict(ta.KALMAN_MODES, {"TAOUSDC": "strict"})):
                    self.assertTrue(ta._kalman_gate_blocks("TAOUSDC", "BUY")[0])
                    self.assertFalse(ta._kalman_gate_blocks("TAOUSDC", "SELL")[0])

        with self.subTest("ages signal on replay clock"):
            import shadow_signals
            sim_now = [1_780_000_000.0]
            shadow = shadow_signals.KalmanStateTracker(
                journal=shadow_signals.KalmanJournal(fixed_path=os.devnull),
                now_fn=lambda: sim_now[0])
            shadow._state["TAOUSDC"] = {"kalman_trend": 1, "ts": sim_now[0] - 10.0}
            with (patch.object(ta, "_kalman_ref", shadow),
                  patch.dict(ta.KALMAN_MODES, {"TAOUSDC": "strict"})):
                self.assertFalse(ta._kalman_gate_blocks("TAOUSDC", "BUY")[0])
                # Ordinary SELL is blocked in strict mode while Kalman trend is +1 (UP)
                self.assertTrue(ta._kalman_gate_blocks("TAOUSDC", "SELL", reason="trend_confirmed_down")[0])
                # Kinematic plunge SELLs bypass Kalman lag even when trend is +1
                self.assertFalse(ta._kalman_gate_blocks("TAOUSDC", "SELL", reason="accel_plunge_down")[0])
                self.assertFalse(ta._kalman_gate_blocks("TAOUSDC", "SELL", reason="small_window_plunge_down")[0])
                sim_now[0] += ta.GATE_STALE_SEC + 1.0
                self.assertTrue(ta._kalman_gate_blocks("TAOUSDC", "BUY")[0])


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _window(prices, sample_rate=0.8, symbol="BTCUSDT"):
    pw = ta.PriceWindow(symbol, len(prices), sample_rate_sec=sample_rate)
    for p in prices:
        pw.process_price(p)
    return pw


def _make_cache24_manager(symbol, entries, tmp_dir):
    return make_cache24_test_manager(symbol, entries, tmp_dir, api_client=mock_bapi)


_synthetic_entries = make_synthetic_price_entries


# ═══════════════════════════════════════════════════════════════════════════
# PriceTrendAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceTrendAnalyzer(unittest.TestCase):

    def test_linear_regression_cases(self):
        cases = (
            ("up", [100 + i for i in range(10)], 1),
            ("down", [100 - i for i in range(10)], -1),
            ("single", [42.0], 0),
            ("constant", [100.0] * 10, 0),
        )
        for label, prices, expected_sign in cases:
            with self.subTest(case=label):
                _, slope, correlation = ta.PriceTrendAnalyzer(prices).linear_regression_trend()
                if expected_sign == 0:
                    self.assertIsNone(slope)
                else:
                    self.assertGreater(slope * expected_sign, 0)
                    self.assertAlmostEqual(abs(correlation), 1.0, places=6)

    def test_gradient_cases(self):
        cases = (
            ("up", list(range(10)), 1),
            ("down", [10 - i for i in range(10)], -1),
            ("single", [5.0], 0),
        )
        for label, prices, expected_sign in cases:
            with self.subTest(case=label):
                gradients, average = ta.PriceTrendAnalyzer(prices).calculate_gradient()
                if expected_sign == 0:
                    self.assertEqual(gradients, [])
                    self.assertEqual(average, 0)
                else:
                    self.assertGreater(average * expected_sign, 0)


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — sample_rate_sec + recent_n
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowSampleRate(unittest.TestCase):

    def test_sample_rate_initialization_and_update(self):
        with self.subTest(msg="default_sample_rate"):
            pw = ta.PriceWindow("BTCUSDT", 50)
            self.assertAlmostEqual(pw.sample_rate_sec, ta.TIME_SLEEP_GET_PRICE)
        with self.subTest(msg="custom_sample_rate"):
            pw = ta.PriceWindow("BTCUSDT", 50, sample_rate_sec=2.0)
            self.assertAlmostEqual(pw.sample_rate_sec, 2.0)
        with self.subTest(msg="sample_rate_updatable"):
            pw = ta.PriceWindow("BTCUSDT", 50)
            pw.sample_rate_sec = 1.5
            self.assertAlmostEqual(pw.sample_rate_sec, 1.5)

    def test_recent_n_calculation(self):
        with self.subTest(msg="recent_n_formula"):
            pw = ta.PriceWindow("BTCUSDT", 50, sample_rate_sec=1.0)
            expected = max(2, int(ta.RECENT_GRADIENT_SECONDS / 1.0))
            self.assertEqual(pw.recent_n, expected)
        with self.subTest(msg="recent_n_minimum_two"):
            pw = ta.PriceWindow("BTCUSDT", 50, sample_rate_sec=9999.0)
            self.assertEqual(pw.recent_n, 2)
        with self.subTest(msg="recent_n_larger_for_faster_rate"):
            pw = ta.PriceWindow("BTCUSDT", 50, sample_rate_sec=0.5)
            n_fast = pw.recent_n
            pw.sample_rate_sec = 2.0
            n_slow = pw.recent_n
            self.assertGreater(n_fast, n_slow)

    def test_set_sample_rate_resizing(self):
        with self.subTest(msg="no_resize_without_window_seconds"):
            pw = ta.PriceWindow("BTCUSDT", 50)
            pw.set_sample_rate(2.0)
            self.assertAlmostEqual(pw.sample_rate_sec, 2.0)
            self.assertEqual(pw.window_size, 50)
        with self.subTest(msg="resizes_to_target_duration"):
            pw = ta.PriceWindow("BTCUSDT", 60, sample_rate_sec=1.0, window_seconds=60.0)
            pw.set_sample_rate(2.0)
            self.assertEqual(pw.window_size, 30)
            self.assertEqual(pw.prices.maxlen, 30)
        with self.subTest(msg="resize_keeps_recent_prices"):
            pw = ta.PriceWindow("BTCUSDT", 100, sample_rate_sec=1.0, window_seconds=100.0)
            for p in range(100):
                pw.process_price(float(p))
            pw.set_sample_rate(4.0)
            self.assertEqual(pw.window_size, 25)
            self.assertEqual(len(pw.prices), 25)
            self.assertIn(99.0, pw.prices)
            self.assertEqual(len(pw.sorted_prices), len(pw.prices))
        with self.subTest(msg="ignores_invalid"):
            pw = ta.PriceWindow("BTCUSDT", 50, window_seconds=60.0)
            pw.set_sample_rate(0)
            pw.set_sample_rate(None)
            self.assertEqual(pw.window_size, 50)

    def test_from_cache24_stores_window_seconds(self):
        tmp = tempfile.mkdtemp()
        mgr = _make_cache24_manager("BTCUSDT", _synthetic_entries(30), tmp)
        pw = ta.PriceWindow.from_cache24("BTCUSDT", 24.0, mgr)
        self.assertAlmostEqual(pw.window_seconds, 24.0)


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow._sample_rate_from_entries
# ═══════════════════════════════════════════════════════════════════════════

class TestSampleRateFromEntries(unittest.TestCase):

    def test_regular_and_outlier_intervals(self):
        regular = [[i * 800, 100.0] for i in range(10)]
        with_outlier = [[i * 800, 100.0] for i in range(9)]
        with_outlier.append([with_outlier[-1][0] + 60_000, 100.0])
        for label, entries, places in (("regular", regular, 3), ("outlier", with_outlier, 1)):
            with self.subTest(case=label):
                rate = ta.PriceWindow._sample_rate_from_entries(entries)
                self.assertAlmostEqual(rate, 0.8, places=places)

    def test_insufficient_entries_return_default(self):
        for label, entries in (("empty", []), ("single", [[0, 100.0]])):
            with self.subTest(case=label):
                rate = ta.PriceWindow._sample_rate_from_entries(entries)
                self.assertAlmostEqual(rate, ta.TIME_SLEEP_GET_PRICE)

# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow.from_cache24 — the factory with Cache24PriceManager
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowFromCache24(TempDirTestCase):

    def _make(self, entries, symbol="BTCUSDT", window_seconds=None):
        mgr = _make_cache24_manager(symbol, entries, self.tmp)
        if window_seconds is None:
            window_seconds = len(entries) * 0.8
        return ta.PriceWindow.from_cache24(symbol, window_seconds, mgr)

    # ── synthetic data ──────────────────────────────────────────────────────

    def test_window_properties_initialization(self):
        with self.subTest(msg="prices_loaded"):
            entries = _synthetic_entries(50)
            pw = self._make(entries, window_seconds=50 * 0.8)
            self.assertGreater(len(pw.prices), 0)
        
        with self.subTest(msg="size_bounded_by_window_seconds"):
            entries = _synthetic_entries(200, interval_ms=800)
            pw = self._make(entries, window_seconds=60.0)  # 60s / 0.8s = 75 samples
            self.assertLessEqual(pw.window_size, 100)
            
        with self.subTest(msg="sample_rate_computed_from_intervals"):
            entries = _synthetic_entries(50, interval_ms=2000)  # 2s per sample
            pw = self._make(entries, window_seconds=100.0)
            self.assertAlmostEqual(pw.sample_rate_sec, 2.0, delta=0.1)
            
        with self.subTest(msg="window_within_24h_only"):
            entries = _synthetic_entries(100, interval_ms=800)
            max_seconds = cm.Cache24PriceManager.KEEP_HOURS * 3600
            pw = self._make(entries, window_seconds=max_seconds)
            self.assertLessEqual(pw.window_size, max_seconds / pw.sample_rate_sec + 1)
            
        with self.subTest(msg="minimum_window_size_ten"):
            entries = _synthetic_entries(2)
            pw = self._make(entries, window_seconds=1.0)
            self.assertGreaterEqual(pw.window_size, 10)

    def test_trend_detection(self):
        with self.subTest(msg="uptrend"):
            entries = _synthetic_entries(60, delta=10.0, interval_ms=800)
            pw = self._make(entries, window_seconds=60 * 0.8)
            final_trend, gc, sf, gr = pw.get_short_trend()
            self.assertEqual(final_trend, 1)
            self.assertGreater(gc, 0)
            
        with self.subTest(msg="downtrend"):
            entries = _synthetic_entries(60, delta=-10.0, interval_ms=800)
            pw = self._make(entries, window_seconds=60 * 0.8)
            final_trend, gc, sf, gr = pw.get_short_trend()
            self.assertEqual(final_trend, -1)
            self.assertLess(gc, 0)
            
        with self.subTest(msg="small_window_detects_reversal"):
            now_ms = int(time.time() * 1000)
            prices = [100.0 + i for i in range(100)]
            prices.extend(prices[-1] - i for i in range(1, 21))
            entries = [[now_ms - (len(prices) - 1 - i) * 1000, price]
                       for i, price in enumerate(prices)]
            manager = _make_cache24_manager("BTCUSDC", entries, self.tmp)
    
            large = ta.PriceWindow.from_cache24("BTCUSDC", 119.0, manager)
            small = ta.PriceWindow.from_cache24("BTCUSDC", 15.0, manager)
            _, _, large_slope, _ = large.get_short_trend()
            _, _, small_slope, _ = small.get_short_trend()
    
            self.assertGreater(large_slope, 0)
            self.assertLess(small_slope, 0)
            self.assertLess(len(small.prices), len(large.prices))


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — get_short_trend()'s 4 values
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowGetTrend(unittest.TestCase):

    def test_price_window_trend_calculations(self):
        with self.subTest("returns four values"):
            pw = _window([100 + i for i in range(20)])
            self.assertEqual(len(pw.get_short_trend()), 4)

        with self.subTest("directional trends"):
            cases = (
                ("up", [100 + i * 2 for i in range(30)], 1),
                ("down", [200 - i * 2 for i in range(30)], -1),
            )
            for label, prices, expected in cases:
                with self.subTest(direction=label):
                    final_trend, growth, slope, gradient = _window(prices).get_short_trend()
                    self.assertEqual(final_trend, expected)
                    self.assertGreater(growth * expected, 0)
                    self.assertGreater(slope * expected, 0)
                    self.assertGreater(gradient * expected, 0)
                    self.assertEqual(final_trend, 1 if growth > 0 else -1)

        with self.subTest("gc is Step-WOLS on the median-filtered window (default SSOT)"):
            prices = [100 + i for i in range(20)]
            pw = _window(prices)
            _, gc, sf, gr = pw.get_short_trend()
            self.assertIn(pw_mod.SHORT_TREND_ESTIMATOR, (pw_mod.SHORT_TREND_ESTIMATOR_STEP_WOLS, pw_mod.SHORT_TREND_ESTIMATOR_STEP_WOLS_CUBIC))
            self.assertAlmostEqual(gc, pw_mod.step_wols_slope(prices, pw.sample_rate_sec), places=10)

        with self.subTest("power parameterization in PriceWindow"):
            prices = [100 + i for i in range(20)]
            pw = _window(prices)
            with patch.object(pw_mod, "STEP_WOLS_POWER", 3.0):
                pw._short_trend_cache = None
                _, gc, sf, gr = pw.get_short_trend()
                self.assertAlmostEqual(gc, pw_mod.step_wols_slope(prices, pw.sample_rate_sec, power=3.0), places=10)

        with self.subTest("neutral series return zero trend"):
            for label, prices in (("single", [100.0]), ("constant", [100.0] * 20)):
                with self.subTest(case=label):
                    final_trend, growth, slope, gradient = _window(prices).get_short_trend()
                    self.assertEqual(final_trend, 0)
                    self.assertAlmostEqual(growth, 0.0, places=6)
                    self.assertAlmostEqual(slope, 0.0, places=6)
                    self.assertAlmostEqual(gradient, 0.0, places=6)

        with self.subTest("recent gradient captures late reversal"):
            prices = [100 + i for i in range(30)] + [129 - i * 8 for i in range(1, 6)]
            pw = _window(prices, sample_rate=0.8)
            _, _, _, gr = pw.get_short_trend()
            self.assertLess(gr, 0)

        with self.subTest("slope full sees whole window"):
            prices = [100 + i for i in range(30)] + [129, 128]
            pw = _window(prices)
            _, _, sf, _ = pw.get_short_trend()
            self.assertGreater(sf, 0)


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — min/max/slope/proximities
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowMinMax(unittest.TestCase):

    def test_min_max_and_proximity_behaviors(self):
        with self.subTest("get min max"):
            pw = _window([100, 105, 95, 110, 90])
            self.assertAlmostEqual(pw.get_min(), 90.0, delta=1.0)
            self.assertAlmostEqual(pw.get_max(), 110.0, delta=1.0)

        with self.subTest("sorted consistency"):
            pw = _window([50, 80, 60, 70, 90])
            self.assertEqual(len(pw.prices), len(pw.sorted_prices))

        with self.subTest("eviction"):
            pw = ta.PriceWindow("BTCUSDT", 3)
            for p in [10, 20, 30, 40]:
                pw.process_price(p)
            self.assertEqual(len(pw.prices), 3)
            self.assertNotIn(10, pw.prices)


# ═══════════════════════════════════════════════════════════════════════════
# TrendState
# ═══════════════════════════════════════════════════════════════════════════

class TestTrendState(unittest.TestCase):

    def setUp(self):
        self.now = 1000.0

    def _ts(self, exp_time=9999, fresh_time=60):
        return ta.TrendState(exp_time, fresh_time, now_fn=lambda: self.now)

    def test_trend_lifecycle(self):
        with self.subTest(msg="initial_state"):
            ts = self._ts()
            self.assertEqual(ts.state, "HOLD")
            self.assertEqual(ts.confirm_count, 0)
            
        with self.subTest(msg="start_confirm_and_direction"):
            ts = self._ts()
            ts.start_trend("UP")
            self.assertEqual(ts.state, "UP")
            self.assertEqual(ts.confirm_count, 1)
            self.assertGreater(ts.is_trend_up(), 0)
            self.assertEqual(ts.is_trend_down(), 0)
            self.assertFalse(ts.is_trend_a_minim_validated())
            ts.confirm_trend()
            self.assertEqual(ts.confirm_count, 2)
            
        with self.subTest(msg="start_invalid_raises"):
            with self.assertRaises(AssertionError):
                self._ts().start_trend("INVALID")

    def test_trend_timing(self):
        with self.subTest(msg="trend_expiration"):
            ts = self._ts(exp_time=1)
            ts.start_trend("UP")
            self.now += 1.1
            self.assertTrue(ts.check_trend_expiration())
            
        with self.subTest(msg="freshness_window"):
            fresh = self._ts(fresh_time=60)
            stale = self._ts(fresh_time=1)
            fresh.start_trend("UP")
            stale.start_trend("UP")
            self.assertTrue(fresh.is_trend_fresh())
            self.now += 1.1
            self.assertFalse(stale.is_trend_fresh())
            
        with self.subTest(msg="older_than"):
            ts = self._ts()
            ts.start_trend("UP")
            self.now += 0.1
            self.assertTrue(ts.is_started_trend_older_than(0.05))
            self.assertFalse(ts.is_started_trend_older_than(9999))

class TestTrendStateCooldown(unittest.TestCase):
    """Cooldown per trend instance (Jul 22) — see FIRE_MIN_RETRY_INTERVAL_SEC.
    A fake clock (a mutable counter) so we do not depend on real 30-minute sleeps."""

    def _ts_with_clock(self):
        clock = {"t": 1000.0}
        ts = ta.TrendState(9999, 60, now_fn=lambda: clock["t"])
        return ts, clock

    def test_confirmation_limit_and_reset(self):
        ts, _ = self._ts_with_clock()
        ts.start_trend("UP")
        self.assertFalse(ts.fire_limit_reached("UP"))
        self.assertFalse(ts.fire_limit_reached("DOWN"))
        ts.mark_confirmed("UP")
        self.assertFalse(ts.fire_limit_reached("UP"))
        self.assertFalse(ts.fire_limit_reached("DOWN"))
        for _ in range(ta.FIRE_MAX_PER_TREND - 1):
            ts.mark_confirmed("UP")
        self.assertTrue(ts.fire_limit_reached("UP"))
        self.assertFalse(ts.fire_limit_reached("DOWN"))
        ts.start_trend("DOWN")
        self.assertFalse(ts.fire_limit_reached("UP"))
        self.assertFalse(ts.fire_limit_reached("DOWN"))

    def test_retry_interval_and_reset(self):
        ts, clock = self._ts_with_clock()
        ts.start_trend("UP")
        self.assertTrue(ts.can_retry_fire("UP"))
        ts.mark_fire_attempt("UP")
        self.assertFalse(ts.can_retry_fire("UP"))
        self.assertTrue(ts.can_retry_fire("DOWN"))
        clock["t"] += ta.FIRE_MIN_RETRY_INTERVAL_SEC - 1
        self.assertFalse(ts.can_retry_fire("UP"))
        clock["t"] += 2
        self.assertTrue(ts.can_retry_fire("UP"))
        ts.start_trend("UP")
        self.assertTrue(ts.can_retry_fire("UP"))

# ═══════════════════════════════════════════════════════════════════════════
# CacheCurrentPriceManager — get_sample_rate / get_update_frequency
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheCurrentPriceFrequency(TempDirTestCase):

    def setUp(self):
        super().setUp()
        fname = os.path.join(self.tmp, "cp.json")
        self.mgr = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        # Construction (missing file -> init fetch) can record a timestamp.
        # We start the frequency measurement clean to test the mechanism in isolation.
        self.mgr._update_timestamps.clear()

    def test_current_price_frequency_and_sample_rate(self):
        with self.subTest("empty state uses fallback"):
            self.assertAlmostEqual(self.mgr.get_sample_rate("BTCUSDT", fallback=0.8), 0.8)

        with self.subTest("updates produce sample rate"):
            t0 = time.time()
            self.mgr.on_items_update("BTCUSDT", [[int(t0 * 1000), 60000.0]])
            time.sleep(0.15)
            self.mgr.on_items_update("BTCUSDT", [[int(time.time() * 1000), 60001.0]])
            elapsed = time.time() - t0
            rate = self.mgr.get_sample_rate("BTCUSDT", fallback=9.9)
            self.assertGreater(rate, 0.0)
            self.assertLess(rate, 9.9)
            self.assertLessEqual(rate, elapsed + 0.5)

        with self.subTest("old timestamps trimmed"):
            self.mgr._update_timestamps.clear()
            old_ts = time.time() - cm.CacheCurrentPriceManager.FREQ_WINDOW_SEC - 10
            self.mgr._update_timestamps["BTCUSDT"].append(old_ts)
            self.mgr.on_items_update("BTCUSDT", [[int(time.time() * 1000), 60000.0]])
            dq = self.mgr._update_timestamps["BTCUSDT"]
            cutoff = time.time() - cm.CacheCurrentPriceManager.FREQ_WINDOW_SEC - 1
            self.assertTrue(all(t > cutoff for t in dq))

        with self.subTest("single update returns fallback"):
            self.mgr._update_timestamps.clear()
            self.mgr.on_items_update("BTCUSDT", [[int(time.time() * 1000), 60000.0]])
            self.assertAlmostEqual(self.mgr.get_sample_rate("BTCUSDT", fallback=1.23), 1.23)


# ═══════════════════════════════════════════════════════════════════════════
# Subscriber pattern inherited from CacheManagerInterface + WS attachment
# ═══════════════════════════════════════════════════════════════════════════

class _FakeWSManager:
    """Mimics BinanceWebSocketManager: subscribe(sub) + push() -> on_items_update."""
    def __init__(self):
        self._subs = []
    def subscribe(self, sub):
        if sub not in self._subs:
            self._subs.append(sub)
    def push(self, symbol, price):
        now_ms = int(time.time() * 1000)
        for s in list(self._subs):
            s.on_items_update(symbol, [[now_ms, price]])


class _RecordingSubscriber:
    def __init__(self):
        self.events = []
    def on_price_update(self, symbol, ts_ms, price):
        self.events.append((symbol, price))


class TestSubscriberPatternInheritance(TempDirTestCase):

    def _cache24(self, symbol="BTCUSDT"):
        return _make_cache24_manager(symbol, _synthetic_entries(5), self.tmp)

    def test_subscriber_pattern_and_ws_attachment(self):
        with self.subTest("managers inherit subscribe_price"):
            cases = (
                ("cache24", type(self._cache24()).subscribe_price),
                ("current-price", cm.CacheCurrentPriceManager.subscribe_price),
            )
            for label, method in cases:
                with self.subTest(manager=label):
                    self.assertIs(method, cm.CachePriceManagerBase.subscribe_price)

        with self.subTest("inherited notify reaches subscriber"):
            mgr = self._cache24("BTCUSDT")
            rec = _RecordingSubscriber()
            mgr.subscribe_price(rec)
            mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 123.0)
            self.assertIn(("BTCUSDT", 123.0), rec.events)

        with self.subTest("attach ws manager wires chain"):
            fname = os.path.join(self.tmp, "cp_ws.json")
            mgr = cm.CacheCurrentPriceManager(
                sync_ts=9999, symbols=["BTCUSDT"],
                filename=fname, ws_manager=None, api_client=mock_bapi,
            )
            ws = _FakeWSManager()
            mgr.attach_ws_manager(ws)

            rec = _RecordingSubscriber()
            mgr.subscribe_price(rec)

            ws.push("BTCUSDT", 67000.0)
            self.assertIn(("BTCUSDT", 67000.0), rec.events)

        with self.subTest("attach ws manager idempotent"):
            fname = os.path.join(self.tmp, "cp_ws2.json")
            mgr = cm.CacheCurrentPriceManager(
                sync_ts=9999, symbols=["BTCUSDT"],
                filename=fname, ws_manager=None, api_client=mock_bapi,
            )
            ws = _FakeWSManager()
            mgr.attach_ws_manager(ws)
            mgr.attach_ws_manager(ws)
            self.assertEqual(ws._subs.count(mgr), 1)

        with self.subTest("ws tick marks ws healthy"):
            fname = os.path.join(self.tmp, "cp_ws3.json")
            mgr = cm.CacheCurrentPriceManager(
                sync_ts=9999, symbols=["BTCUSDT"],
                filename=fname, ws_manager=None, api_client=mock_bapi,
            )
            self.assertFalse(mgr._ws_is_healthy())
            mgr.on_items_update("BTCUSDT", [[int(time.time() * 1000), 50000.0]])
            self.assertTrue(mgr._ws_is_healthy())


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — complete Cache24PriceManager → PriceWindow wiring
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowCache24Wiring(TempDirTestCase):

    def _make_wired(self, entries, symbol="BTCUSDT", window_seconds=None):
        """Create a Cache24PriceManager plus a PriceWindow subscribed to it."""
        mgr = _make_cache24_manager(symbol, entries, self.tmp)
        if window_seconds is None:
            window_seconds = len(entries) * 0.8
        pw = ta.PriceWindow.from_cache24(symbol, window_seconds, mgr)
        return pw, mgr

    def test_subscription_lifecycle_and_flags(self):
        with self.subTest(msg="subscription_flags"):
            entries = _synthetic_entries(20)
            wired, _ = self._make_wired(entries)
            plain = ta.PriceWindow("BTCUSDT", 50)
            self.assertTrue(wired._subscribed_to_cache24)
            self.assertFalse(plain._subscribed_to_cache24)
            
        with self.subTest(msg="subscribe_method_directly"):
            entries = _synthetic_entries(10)
            mgr = _make_cache24_manager("BTCUSDT", entries, self.tmp)
            pw = ta.PriceWindow("BTCUSDT", 20)
            self.assertFalse(pw._subscribed_to_cache24)
            pw.subscribe_to_cache24(mgr)
            self.assertTrue(pw._subscribed_to_cache24)
            mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 88888.0)
            self.assertIn(88888.0, pw.prices)
            
        with self.subTest(msg="unsubscribe_stops_updates"):
            entries = _synthetic_entries(10)
            pw, mgr = self._make_wired(entries, symbol="BTCUSDT")
            mgr.unsubscribe_price(pw)
            n_before = len(pw.prices)
            mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 55555.0)
            self.assertEqual(len(pw.prices), n_before)

    def test_price_update_routing(self):
        with self.subTest(msg="direct_price_updates_filter_symbol"):
            entries = _synthetic_entries(10)
            pw, _ = self._make_wired(entries, symbol="BTCUSDT")
            n_before = len(pw.prices)
            pw.on_price_update("ETHUSDT", int(time.time() * 1000), 3000.0)
            self.assertEqual(len(pw.prices), n_before)
            pw.on_price_update("BTCUSDT", int(time.time() * 1000), 99999.0)
            self.assertEqual(len(pw.prices), min(n_before + 1, pw.window_size))
            
        with self.subTest(msg="cache24_notifies_pricewindow"):
            entries = _synthetic_entries(10)
            pw, mgr = self._make_wired(entries, symbol="BTCUSDT")
            n_before = len(pw.prices)
            ts_ms = int(time.time() * 1000)
            mgr.on_price_update("BTCUSDT", ts_ms, 77777.0)
            self.assertEqual(len(pw.prices), min(n_before + 1, pw.window_size))
            self.assertIn(77777.0, pw.prices)
            
        with self.subTest(msg="multiple_windows_same_cache24"):
            entries = _synthetic_entries(50)
            mgr = _make_cache24_manager("BTCUSDT", entries, self.tmp)
            pw_small = ta.PriceWindow.from_cache24("BTCUSDT", 20 * 0.8, mgr)
            pw_big   = ta.PriceWindow.from_cache24("BTCUSDT", 50 * 0.8, mgr)
    
            ts_ms = int(time.time() * 1000)
            mgr.on_price_update("BTCUSDT", ts_ms, 12345.0)
    
            self.assertIn(12345.0, pw_small.prices)
            self.assertIn(12345.0, pw_big.prices)


# ═══════════════════════════════════════════════════════════════════════════
# WindowAnalyzer — the metrics moved out of PriceWindow
# ═══════════════════════════════════════════════════════════════════════════

class TestWindowAnalyzer(unittest.TestCase):

    def test_window_analyzer_integration_and_api_surface(self):
        with self.subTest("api surface is lean"):
            pw = _window([100, 110, 105])
            an = ta.WindowAnalyzer(pw)
            for method in ("calculate_proximities", "calculate_slope_max_min",
                           "calculate_positions", "_analyze_price_movement",
                           "evaluate_buy_sell_opportunity",
                           "get_trend", "get_instant_trend"):
                with self.subTest(absent=method):
                    self.assertFalse(hasattr(pw, method))
                    self.assertFalse(hasattr(an, method))
            for method in ("get_min", "get_max", "get_short_trend"):
                with self.subTest(present=method):
                    self.assertTrue(hasattr(pw, method))
            for method in ("check_price_change", "calculate_acceleration"):
                with self.subTest(analyzer_present=method):
                    self.assertTrue(hasattr(an, method))

    def test_analyzer_gradients_and_noise(self):
        with self.subTest("recent gradient cases"):
            cases = (
                ("up", [100 + i for i in range(20)], 1),
                ("down", [200 - i for i in range(20)], -1),
                ("insufficient", [100.0], 0),
            )
            for label, prices, expected_sign in cases:
                with self.subTest(case=label):
                    gradient = _window(prices).get_recent_gradient()
                    if expected_sign == 0:
                        self.assertEqual(gradient, 0.0)
                    else:
                        self.assertGreater(gradient * expected_sign, 0)

        with self.subTest("noise epsilon cases"):
            import random
            random.seed(1)
            constant = _window([100.0] * 20).get_noise_epsilon()
            volatile = _window([100 + random.uniform(-5, 5) for _ in range(30)]).get_noise_epsilon()
            calm = _window([100 + (i % 2) * 0.1 for i in range(30)])
            wild = _window([100 + (i % 2) * 10 for i in range(30)])
            insufficient = _window([100.0, 101.0]).get_noise_epsilon()
            self.assertAlmostEqual(constant, 0.0, places=6)
            self.assertGreater(volatile, 0.0)
            self.assertLess(calm.get_noise_epsilon(), wild.get_noise_epsilon())
            self.assertEqual(insufficient, 0.0)

    def test_analyzer_price_movement_and_evaluations(self):
        with self.subTest("check price change cases"):
            cases = (
                ("below", [100.0, 100.05, 100.02], 5.0, True, None),
                ("above", [100.0, 100.0, 110.0], 1.0, False, None),
                ("insufficient", [100.0], 1.0, True, 1),
            )
            for label, prices, threshold, expect_zero, expected_pos in cases:
                with self.subTest(case=label):
                    slope, position = ta.WindowAnalyzer(_window(prices)).check_price_change(threshold)
                    self.assertEqual(slope, 0) if expect_zero else self.assertNotEqual(slope, 0)
                    if expected_pos is not None:
                        self.assertEqual(position, expected_pos)

        with self.subTest("analyzer shares window mutation"):
            pw = _window([100, 101, 102])
            an = ta.WindowAnalyzer(pw)
            before = pw.get_max()
            pw.process_price(200.0)
            self.assertGreater(pw.get_max(), before)
            self.assertIs(an.window, pw)


# ═══════════════════════════════════════════════════════════════════════════
# TrendCoordinator — event-driven + heartbeat + cache
# ═══════════════════════════════════════════════════════════════════════════

class TestTrendCoordinator(TempDirTestCase):

    def setUp(self):
        super().setUp()
        entries = _synthetic_entries(60, interval_ms=800)
        self.cache24 = _make_cache24_manager("BTCUSDT", entries, self.tmp)

        fname = os.path.join(self.tmp, "cp.json")
        self.cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        self.cpm.on_items_update("BTCUSDT", [[int(time.time() * 1000), 60000.0]])

        # The manager owns the windows + calc + cross-process cache
        self.mgr = cm.CacheShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "trend.json"))
        self.mgr.start_computation({"BTCUSDT": self.cache24}, self.cpm)
        self._orig_kalman_ref = ta._kalman_ref

    def tearDown(self):
        ta._kalman_ref = self._orig_kalman_ref

    def _make_coord(self):
        return ta.TrendCoordinator(
            symbols=["BTCUSDT"],
            instant_mgr=self.mgr,
            current_price_mgr=self.cpm,
            cache24_managers={"BTCUSDT": self.cache24},
            min_interval=2.0, max_interval=30.0,
        )

    def test_coordinator_and_manager_lifecycle_and_snapshots(self):
        with self.subTest("manager owns windows"):
            self.assertIsNotNone(self.mgr.get_window("BTCUSDT"))
            self.assertIsNotNone(self.mgr.get_analyzer("BTCUSDT"))
            self.assertGreater(len(self.mgr.get_window("BTCUSDT").prices), 0)

        with self.subTest("coordinator and windows subscribed to cache24"):
            coord = self._make_coord()
            self.assertIn(coord, self.cache24._price_subscribers)
            self.assertTrue(self.mgr.get_window("BTCUSDT")._subscribed_to_cache24)
            self.assertTrue(self.mgr.get_window("BTCUSDT", self.mgr.window_big_sec)._subscribed_to_cache24)

        with self.subTest("evaluation cache lifecycle"):
            coord = self._make_coord()
            self.assertIsNone(coord.instant_mgr.get_snapshot("BTCUSDT"))
            snap = coord.evaluate("BTCUSDT")
            self.assertIsNotNone(snap)
            cached = coord.instant_mgr.get_snapshot("BTCUSDT")
            for key in ("final_trend", "growth_coefficient", "slope_full",
                        "gradient_recent", "slope_small", "slope_big",
                        "pos", "current_price", "ts"):
                with self.subTest(field=key):
                    self.assertIn(key, cached)
            self.assertNotIn("slope_max_min", cached)
            self.assertFalse(coord._dirty["BTCUSDT"])
            manager_snapshot = self.mgr.get_snapshot("BTCUSDT")
            self.assertIsNotNone(manager_snapshot)
            self.assertIn("slope_big", manager_snapshot)

        with self.subTest("manager tick publishes instant gradient"):
            self.mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 60500.0)
            snap = self.mgr.get_snapshot("BTCUSDT")
            self.assertIsNotNone(snap)
            self.assertIn("gradient_recent_fast", snap)
            self.assertIn("epsilon", snap)
            self.assertEqual(snap["current_price"], 60500.0)

    def test_coordinator_due_dirty_and_run_loop(self):
        with self.subTest("dirty set on price update"):
            coord = self._make_coord()
            coord._dirty["BTCUSDT"] = False
            coord.on_price_update("BTCUSDT", int(time.time() * 1000), 60001.0)
            self.assertTrue(coord._dirty["BTCUSDT"])
            self.assertTrue(coord._event.is_set())

        with self.subTest("due policy for dirty and heartbeat"):
            coord = self._make_coord()
            now = time.time()
            coord._last_eval["BTCUSDT"] = now
            coord._dirty["BTCUSDT"] = True
            self.assertFalse(coord._is_due("BTCUSDT", now + 0.5))
            self.assertTrue(coord._is_due("BTCUSDT", now + 2.5))
            coord._dirty["BTCUSDT"] = False
            self.assertTrue(coord._is_due("BTCUSDT", now + 31.0))
            self.assertFalse(coord._is_due("BTCUSDT", now + 5.0))

        with self.subTest("tick updates window and marks dirty"):
            coord = self._make_coord()
            coord._dirty["BTCUSDT"] = False
            win = self.mgr.get_window("BTCUSDT")
            self.cache24.on_price_update("BTCUSDT", int(time.time() * 1000), 61234.0)
            self.assertTrue(coord._dirty["BTCUSDT"])
            self.assertIn(61234.0, win.prices)

        with self.subTest("stop wakes run loop"):
            coord = self._make_coord()
            thread = threading.Thread(target=coord.run)
            thread.start()
            coord.stop()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_coordinator_validation_and_rejection_rules(self):
        with self.subTest("invalid intervals fail fast and symbols deduplicated"):
            for minimum, maximum in ((0, 30), (-1, 30), (31, 30), (1, float("nan"))):
                with self.subTest(minimum=minimum, maximum=maximum):
                    with self.assertRaises(ValueError):
                        ta.TrendCoordinator(
                            ["BTCUSDT"], self.mgr, self.cpm,
                            min_interval=minimum, max_interval=maximum,
                        )
            coord = ta.TrendCoordinator(
                ["BTCUSDT", "BTCUSDT"], self.mgr, self.cpm,
                min_interval=1, max_interval=2,
            )
            self.assertEqual(coord.symbols, ["BTCUSDT"])

        with self.subTest("invalid price is not evaluated"):
            coord = self._make_coord()
            with (patch.object(self.cpm, "get_price", return_value=[time.time() * 1000, float("nan")]),
                  patch.object(ta, "handle_symbol") as handle):
                self.assertIsNone(coord.evaluate("BTCUSDT"))
            handle.assert_not_called()

        with self.subTest("stale or future price is not evaluated"):
            coord = self._make_coord()
            now_ms = time.time() * 1000
            with patch.object(ta, "handle_symbol") as handle:
                for timestamp in (now_ms - self.cpm.STALE_THRESHOLD_MS - 1, now_ms + 5_000):
                    with self.subTest(timestamp=timestamp):
                        with patch.object(self.cpm, "get_price", return_value=[timestamp, 60_000.0]):
                            self.assertIsNone(coord.evaluate("BTCUSDT"))
            handle.assert_not_called()

    def test_concurrent_update_and_read_no_crash(self):
        """A WS thread updates the window while the evaluation reads it."""
        coord = self._make_coord()
        stop = threading.Event()
        errors = []

        def writer():
            i = 0
            while not stop.is_set():
                try:
                    self.cache24.on_price_update("BTCUSDT", int(time.time() * 1000), 60000.0 + (i % 50))
                except Exception as e:
                    errors.append(("writer", e))
                i += 1

        def reader():
            while not stop.is_set():
                try:
                    coord.evaluate("BTCUSDT")
                    self.mgr.get_short_trend("BTCUSDT")
                    self.mgr.get_analyzer("BTCUSDT").check_price_change(0.7)
                except Exception as e:
                    errors.append(("reader", e))

        threads = [
            threading.Thread(target=writer, name="writer"),
            threading.Thread(target=reader, name="reader_1"),
            threading.Thread(target=reader, name="reader_2"),
        ]
        for th in threads:
            th.start()
        time.sleep(1.0)
        stop.set()
        for th in threads:
            th.join(timeout=5)
        self.assertEqual(errors, [], f"concurrency errors: {errors[:3]}")


class TestTrackAndPlaceOrder(unittest.TestCase):
    def setUp(self):
        ta._symbol_order_ids.clear()

    def test_buy_cancels_expired_applies_discount_and_tracks_order(self):
        with patch.object(ta.api, "cancel_expired_orders") as mock_exp, \
             patch.object(ta.alert, "notify") as mock_notify, \
             patch.object(ta, "_fire_order", return_value={"orderId": 1001}) as mock_fire:
            order_ids = ta.track_and_place_order("BUY", "BTCUSDT", 1, 100.0, 100.0)
            mock_exp.assert_called_once_with("BUY", "BTCUSDT", ta.EXP_TIME_BUY_ORDER, allowed_owners={"tradeall"})
            mock_fire.assert_called_once_with(
                "BTCUSDT", "BUY", 100.0 * (1.0 - ta.order_lifecycle.MONITOR_MIN_PROC / 100.0), "trend_signal"
            )
            # Placement alerts are centralized in monitororder, not emitted from tradeall
            mock_notify.assert_not_called()
            self.assertEqual(order_ids, [1001])
            self.assertEqual(ta._symbol_order_ids["BTCUSDT"], [1001])

    def test_sell_cancels_expired_applies_premium_and_tracks_order(self):
        with patch.object(ta.api, "cancel_expired_orders") as mock_exp, \
             patch.object(ta.alert, "notify") as mock_notify, \
             patch.object(ta, "_fire_order", return_value={"orderId": 2002}) as mock_fire:
            order_ids = ta.track_and_place_order("SELL", "BTCUSDT", 1, 100.0, 100.0)
            mock_exp.assert_called_once_with("SELL", "BTCUSDT", ta.EXP_TIME_SELL_ORDER, allowed_owners={"tradeall"})
            mock_fire.assert_called_once_with(
                "BTCUSDT", "SELL", 100.0 * (1.0 + ta.order_lifecycle.MONITOR_MIN_PROC / 100.0), "trend_signal"
            )
            # Placement alerts are centralized in monitororder, not emitted from tradeall
            mock_notify.assert_not_called()
            self.assertEqual(order_ids, [2002])

    def test_track_and_place_order_scales_entry_offset_by_tick_size(self):
        ta._SYMBOL_TICK_SIZE_CACHE.clear()
        with patch.object(ta.api, "cancel_expired_orders"), \
             patch.object(ta.mkt, "price_tick_size", return_value=0.10), \
             patch.object(ta, "_fire_order", return_value={"orderId": 2003}) as mock_fire:
            ta.track_and_place_order("BUY", "TAOUSDC", 1, 200.0, 200.0)
            eff_min, _, _, _ = ta.order_lifecycle.resolve_dynamic_reprice_bounds("TAOUSDC", 200.0, tick_size=0.10)
            mock_fire.assert_called_once_with(
                "TAOUSDC", "BUY", 200.0 * (1.0 - eff_min / 100.0), "trend_signal"
            )

    def test_order_skipped_or_refused_does_not_alert(self):
        with patch.object(ta.api, "cancel_expired_orders"), \
             patch.object(ta.alert, "notify") as mock_notify, \
             patch.object(ta, "_fire_order", return_value=None):
            order_ids = ta.track_and_place_order("SELL", "ARBUSDC", 1, 0.21, 0.21, reason="trend_confirmed_down")
            mock_notify.assert_not_called()
            self.assertEqual(order_ids, [])

    def test_cancels_previous_order_without_duplicate_alert_when_filled(self):
        with patch.object(ta.mkt, "cancel_order", side_effect=RuntimeError("filled")) as mock_cancel, \
             patch.object(ta.alert, "notify") as mock_notify, \
             patch.object(ta, "_fire_order", return_value={"orderId": 3003}):
            existing = [9999]
            order_ids = ta.track_and_place_order("BUY", "BTCUSDT", 2, 100.0, 100.0, order_ids=existing)
            mock_cancel.assert_called_once_with("BTCUSDT", "9999")
            # Redundant placement and execution alerts are centralized in monitororder and monitortrades
            mock_notify.assert_not_called()
            self.assertEqual(order_ids, [3003])

    def test_hold_does_not_fire_or_cancel(self):
        with patch.object(ta.mkt, "cancel_order") as mock_cancel, \
             patch.object(ta.alert, "notify") as mock_notify, \
             patch.object(ta, "_fire_order") as mock_fire:
            existing = [777]
            order_ids = ta.track_and_place_order("HOLD", "BTCUSDT", 1, 100.0, 100.0, order_ids=existing)
            mock_cancel.assert_not_called()
            mock_fire.assert_not_called()
            mock_notify.assert_not_called()
            self.assertEqual(order_ids, [777])

    def test_kalman_blocked_or_disallowed_signal_preserves_existing_venue_orders(self):
        existing = [8888]
        with patch.object(ta, "TRADEALL_FIRE_SYMBOLS", {"BTCUSDT"}), \
             patch.object(ta, "_kalman_gate_blocks", return_value=(True, "strict", -1)), \
             patch.object(ta, "GATE_OUTCOME_LOG", MagicMock()), \
             patch.object(ta.mkt, "cancel_order") as mock_cancel, \
             patch.object(ta.api, "cancel_expired_orders") as mock_exp, \
             patch.object(ta.mkt, "place") as mock_place:
            res = ta.track_and_place_order("BUY", "BTCUSDT", 1, 100.0, 100.0, order_ids=existing)
            self.assertEqual(res, [])
            self.assertEqual(existing, [8888])
            mock_cancel.assert_not_called()
            mock_exp.assert_not_called()
            mock_place.assert_not_called()


class TestLogicSmall(unittest.TestCase):
    def test_logic_small_fires_buy_on_positive_surge(self):
        trend_state = ta.TrendState(300, 300)
        with patch.object(ta, "track_and_place_order", return_value=[123]) as mock_order:
            ta.logic_small("SMALL", True, "BTCUSDT", 1, 4.0, trend_state, 60000.0)
            mock_order.assert_called_once_with(
                "BUY", "BTCUSDT", 1, 60000.0, 60000.0,
                reason="small_window_surge_up",
                safeback_seconds=ta.FIRE_SAFEBACK_SEC, force=False,
                cancelorders=True, hours=0.3, regime_context=None
            )
            self.assertEqual(trend_state.state, "UP")
            self.assertEqual(trend_state._confirmed_count_up, 1)

    def test_logic_small_fires_sell_on_negative_plunge(self):
        trend_state = ta.TrendState(300, 300)
        with patch.object(ta, "track_and_place_order", return_value=[456]) as mock_order:
            ta.logic_small("SMALL", True, "BTCUSDT", -1, -4.0, trend_state, 60000.0)
            mock_order.assert_called_once_with(
                "SELL", "BTCUSDT", 1, 60000.0, 60000.0,
                reason="small_window_plunge_down",
                safeback_seconds=ta.FIRE_SAFEBACK_SEC, force=False,
                cancelorders=True, hours=0.3, regime_context=None
            )
            self.assertEqual(trend_state.state, "DOWN")
            self.assertEqual(trend_state._confirmed_count_down, 1)

    def test_logic_small_respects_enable_and_threshold(self):
        trend_state = ta.TrendState(300, 300)
        with patch.object(ta, "track_and_place_order") as mock_order:
            # Below threshold
            ta.logic_small("SMALL", True, "BTCUSDT", 1, ta.SMALL_SLOPE_THRESHOLD - 0.5, trend_state, 60000.0)
            mock_order.assert_not_called()
            # Disabled
            ta.logic_small("SMALL", False, "BTCUSDT", 1, 4.0, trend_state, 60000.0)
            mock_order.assert_not_called()

    def test_logic_small_fires_sell_on_acceleration_plunge(self):
        trend_state = ta.TrendState(300, 300)
        pw = MagicMock()
        pw.calculate_acceleration.side_effect = [(-0.45, -0.35), (-0.30, -0.10)]
        with patch.object(ta, "track_and_place_order", return_value=[789]) as mock_order:
            ta.logic_small("SMALL", True, "TAOUSDC", -1, -1.25, trend_state, 275.0, price_window=pw)
            mock_order.assert_called_once_with(
                "SELL", "TAOUSDC", 1, 275.0, 275.0,
                reason="accel_plunge_down",
                safeback_seconds=ta.FIRE_SAFEBACK_SEC, force=False,
                cancelorders=True, hours=0.3, regime_context=None
            )
            self.assertEqual(trend_state.state, "DOWN")
            self.assertEqual(trend_state._confirmed_count_down, 1)

    def test_logic_small_resets_fire_limit_after_trend_expiration_or_direction_flip(self):
        clock = [1000.0]
        trend_state = ta.TrendState(21.0, 222.0, now_fn=lambda: clock[0])
        pw = MagicMock()
        pw.calculate_acceleration.return_value = (-0.45, -0.35)
        with patch.object(ta, "track_and_place_order", return_value=[789]) as mock_order:
            # First DOWN acceleration impulse fires
            ta.logic_small("SMALL", True, "TAOUSDC", -1, -1.25, trend_state, 218.0, price_window=pw)
            self.assertEqual(mock_order.call_count, 1)
            self.assertEqual(trend_state._confirmed_count_down, 1)

            # Same impulse 5s later (< 21s expiration) is throttled by FIRE_MAX_PER_TREND
            clock[0] = 1005.0
            ta.logic_small("SMALL", True, "TAOUSDC", -1, -1.25, trend_state, 217.9, price_window=pw)
            self.assertEqual(mock_order.call_count, 1)
            self.assertEqual(trend_state.confirm_count, 2)

            # New impulse 30s later (> 21s expiration) starts a fresh DOWN trend and fires again
            clock[0] = 1036.0
            ta.logic_small("SMALL", True, "TAOUSDC", -1, -1.25, trend_state, 218.3, price_window=pw)
            self.assertEqual(mock_order.call_count, 2)
            self.assertEqual(trend_state.confirm_count, 1)
            self.assertEqual(trend_state._confirmed_count_down, 1)

            # Immediate direction flip to UP starts a fresh UP trend and fires
            clock[0] = 1040.0
            ta.logic_small("SMALL", True, "TAOUSDC", 1, 4.0, trend_state, 219.0)
            self.assertEqual(mock_order.call_count, 3)
            self.assertEqual(trend_state.state, "UP")
            self.assertEqual(trend_state._confirmed_count_up, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_order_retry_enqueue_sanitizes_nan_bool_requested_price_ref_price_and_iter44_s3(monkeypatch, tmp_path):
    """[ITER44 #3] order_retry: enqueue sanitizes NaN/bool requested_price & ref_price, and is_expired/is_due reject bool now."""
    # 3. order_retry: enqueue sanitizes NaN/bool requested_price & ref_price, and is_expired/is_due reject bool now
    monkeypatch.setattr(order_retry, "QUEUE_FILE", str(tmp_path / "q.jsonl"))
    monkeypatch.setattr(order_retry, "LOCK_FILE", str(tmp_path / "q.lock"))
    rid = order_retry.enqueue(
        "BTCUSDC",
        "BUY",
        0.25,
        requested_price=float("nan"),
        ref_price=True,
        failure_reason="profit_guard",
        now=1700000000.0,
    )
    assert rid is not None
    loaded = order_retry.get(rid)
    assert loaded["requested_price"] is None
    assert loaded["ref_price"] is None
    assert order_retry.is_expired(loaded, now=True) is True
    assert order_retry.is_due(loaded, now=True) is False
    assert order_retry.resolve_accepted_order("BTCUSDC", True) is False
    assert order_retry.get_accepted_order_ids_for_owner(True, "tradeall") == []

def test_pricewindow_windowanalyzer_iter54_s4():
    """[ITER54 #4] PriceWindow & WindowAnalyzer."""
    # 4. PriceWindow & WindowAnalyzer
    pw = PriceWindow("BTCUSDC", window_size=True, sample_rate_sec=float("nan"), window_seconds=True)
    assert pw.window_size == 10
    assert pw.window_seconds is None
    for p in (100.0, 101.0, 102.0, 103.0):
        pw.process_price(p)
    eps_default = pw.get_noise_epsilon(1.0)
    assert pw.get_noise_epsilon(float("nan")) == eps_default
    assert pw.get_noise_epsilon(True) == eps_default

    wa = WindowAnalyzer(pw)
    assert wa.check_price_change(float("nan")) == (0, 0)
    assert wa.check_price_change(True) == (0, 0)

def test_pricewindow_symbol_clamps_to_while_bool_nan_inf_iter55_s1():
    """[ITER55 #1] PriceWindow(symbol, 0) clamps to 1, while bool/NaN/Inf fall back to 10."""
    # 1. PriceWindow(symbol, 0) clamps to 1, while bool/NaN/Inf fall back to 10
    assert PriceWindow("BTCUSDC", 0).window_size == 1
    assert PriceWindow("BTCUSDC", -5).window_size == 1
    assert PriceWindow("BTCUSDC", True).window_size == 10
    assert PriceWindow("BTCUSDC", float("nan")).window_size == 10

def test_tradeall_observe_format_state_text_does_not_treat_bool_as_trend_iter55_s2(tmp_path):
    """[ITER55 #2] tradeall_observe: format_state_text does not treat bool as 1/0 trend, and backtest loaders reject NaN/Inf."""
    from visuals.tradeall_observe import _parse_shadow_rows, format_state_text, load_backtest_order_events, load_backtest_price_samples, load_backtest_trend_starts
    # 2. tradeall_observe: format_state_text does not treat bool as 1/0 trend, and backtest loaders reject NaN/Inf
    txt = format_state_text({"final_trend": True, "trend_fast": False, "kalman_trend": True}, "HDR")
    assert "trend:     ?" in txt
    assert "fast:      ?" in txt
    assert "kalman:    ?" in txt

    shadow_rows = [
        ["nan", "BTCUSDC", "kalman", "trend_start", "1", "0", "95000.0", "0.1", "0.02"],
        ["1700000000", "BTCUSDC", "kalman", "trend_start", "1", "0", "nan", "0.1", "0.02"],
        ["1700000000", "BTCUSDC", "kalman", "trend_start", "1", "0", "95000.0", "0.1", "0.02"],
    ]
    parsed_sh = _parse_shadow_rows(shadow_rows, "BTCUSDC")
    assert len(parsed_sh) == 1
    assert parsed_sh[0]["ts"] == 1700000000.0

    (tmp_path / "tradeall_price_samples.log").write_text("nan|BTCUSDC|95000\n1700000000|BTCUSDC|nan\n1700000000|BTCUSDC|95000\n")
    assert load_backtest_price_samples(str(tmp_path), "BTCUSDC") == ([1700000000.0], [95000.0])

    (tmp_path / "tradeall_decisions.log").write_text("nan|BTCUSDC|trend_start|UP|HOLD|95000|1\n1700000000|BTCUSDC|trend_start|UP|HOLD|95000|1\n")
    assert len(load_backtest_trend_starts(str(tmp_path), "BTCUSDC")) == 1

    (tmp_path / "order_outcomes.log").write_text("nan|BTCUSDC|BUY|95000|0.1|accepted||tradeall.py|trend_buy\n1700000000|BTCUSDC|BUY|95000|0.1|accepted||tradeall.py|trend_buy\n")
    assert len(load_backtest_order_events(str(tmp_path), "BTCUSDC")) == 1

def test_pricetrendanalyzer_calculate_gradient_handles_empty_prices_iter63_s3():
    """[ITER63 #3] PriceTrendAnalyzer.calculate_gradient handles empty prices;."""
    from pricewindow import PriceTrendAnalyzer, instant_trend_from_prices
    # 3. PriceTrendAnalyzer.calculate_gradient handles empty prices;
    #    instant_trend_from_prices handles bool/inf/tiny sample_rate_sec
    assert PriceTrendAnalyzer([]).calculate_gradient() == ([], 0)
    assert len(PriceTrendAnalyzer([10.0, 12.0, 14.0]).calculate_gradient()[0]) == 3
    for bad_sr in (True, float("inf"), float("nan"), 1e-309, -1.0):
        t, gc, sf, gr = instant_trend_from_prices([10.0, 11.0, 12.0], bad_sr)
        assert t == 1 and sf > 0

def test_visuals_tradeall_observe__log_dates_and_build_backtest_state_text_iter77_s2(tmp_path):
    """[ITER77 #2] visuals.tradeall_observe _log_dates and build_backtest_state_text."""
    from visuals import tradeall_observe
    # 2. visuals.tradeall_observe _log_dates and build_backtest_state_text
    assert tradeall_observe._log_dates(False) == []
    assert tradeall_observe._log_dates(-3) == []
    assert tradeall_observe._log_dates(float("nan")) == []
    assert len(tradeall_observe._log_dates(2)) == 2

    state_file = tmp_path / "analysis_state.json"
    state_file.write_text(json.dumps({"BTCUSDC": {"ts": 1e20, "price": 50000.0}}), encoding="utf-8")
    state_txt = tradeall_observe.build_backtest_state_text(str(tmp_path), "BTCUSDC")
    assert state_txt is not None and "SIMULATED ANALYSIS (unknown)" in state_txt

def test_pricewindow_input_validation_and_synchronized_normalization_when_window_size_iter78_s3():
    """[ITER78 #3] PriceWindow input validation and synchronized normalization when window_size >= max_index."""
    from offline.simulations.sim import PriceWindow
    # 3. PriceWindow input validation and synchronized normalization when window_size >= max_index
    with pytest.raises(ValueError):
        PriceWindow(True)
    with pytest.raises(ValueError):
        PriceWindow(5, max_index=0)
    with pytest.raises(ValueError):
        PriceWindow(5, epsilon=-0.1)

    pw = PriceWindow(window_size=5, max_index=3)
    for px in [10.0, 20.0, 5.0, 30.0, 15.0, 25.0]:
        pw.process_price(px)
    min_px, min_pos = pw.get_min()
    max_px, max_pos = pw.get_max()
    assert min_px == 5.0 and min_pos == 1
    assert max_px == 30.0 and max_pos == 2

def test_fleethealthmonitor_parses_float_hb_stale_s_and_rejects_invalid_loop_iter80_s3(tmp_path):
    """[ITER80 #3] FleetHealthMonitor parses float hb_stale_s and rejects invalid loop intervals."""
    from orchestratorOS.livecheck.fleet_health_monitor import FleetHealthMonitor
    # 3. FleetHealthMonitor parses float hb_stale_s and rejects invalid loop intervals
    procs_file = tmp_path / "procs.conf"
    procs_file.write_text(
        "tradeall.py|$ROOT|python tradeall.py|tradeall|logs/tradeall.log|180.5|bot\n",
        encoding="utf-8",
    )
    from orchestratorOS.lib.technical_repair_ssot import parse_procs_manifest
    mon = FleetHealthMonitor(workspace_dir=str(tmp_path))
    parsed = parse_procs_manifest(mon.workspace_dir)
    assert len(parsed) == 1
    assert parsed[0]["hb_stale_s"] == 180.5

    with pytest.raises(ValueError):
        mon.loop(interval_seconds=False)
    with pytest.raises(ValueError):
        mon.loop(interval_seconds=0)
    with pytest.raises(ValueError):
        mon.loop(interval_seconds=float("nan"))

def test_os_resources_watchdog__parse_proc_stat_raises_valueerror_on_truncated_stat_lines_iter82_s3():
    """[ITER82 #3] os_resources_watchdog._parse_proc_stat raises ValueError on truncated stat lines and _managed_script handles malformed proc dicts."""
    from orchestratorOS.livecheck import os_resources_watchdog
    # 3. os_resources_watchdog._parse_proc_stat raises ValueError on truncated stat lines and _managed_script handles malformed proc dicts
    with pytest.raises(ValueError, match="malformed /proc/stat line"):
        os_resources_watchdog._parse_proc_stat("123 (python) S 1 2 3")
    assert os_resources_watchdog._managed_script(None) is None
    assert os_resources_watchdog._managed_script({"command": "python tradeall.py", "pid": False}) is None

def test_backtestbroker_normalizes_lowercase_whitespace_order_type_and_sanitizes_nan_iter134_s2(tmp_path):
    """[ITER134 #2] BacktestBroker normalizes lowercase/whitespace order_type and sanitizes NaN trend_gate_max_wait_sec."""
    from offline.backtests.tradeall import BacktestBroker
    # 2. BacktestBroker normalizes lowercase/whitespace order_type and sanitizes NaN trend_gate_max_wait_sec
    broker = BacktestBroker(str(tmp_path), clock=lambda: 1000.0, trend_gate_max_wait_sec=float("nan"))
    assert broker.trend_gate_max_wait_sec == pytest.approx(3600.0)
    assert broker.place_order_smart("invalid_side", "BTCUSDC", 50000.0, 0.1) is None
    res_buy = broker.place_order_smart(" buy ", "BTCUSDC", 50000.0, 0.1)
    assert res_buy is not None
    assert broker.n_buy == 1
    assert broker.n_sell == 0

def test_tradeall_track_and_place_order_deduplicates_ids_ignores_and_stores_clean_iter158_s4(monkeypatch):
    """[ITER158 #4] tradeall.track_and_place_order deduplicates .0 IDs, ignores '0', and stores clean orderId."""
    # 4. tradeall.track_and_place_order deduplicates .0 IDs, ignores "0", and stores clean orderId
    cancelled_oids = []
    monkeypatch.setattr(ta, "TRADEALL_FIRE_SYMBOLS", {"BTCUSDC"})
    monkeypatch.setitem(ta.KALMAN_MODES, "BTCUSDC", "off")
    monkeypatch.setattr(ta.mkt, "cancel_order", lambda sym, oid: cancelled_oids.append((sym, oid)))
    monkeypatch.setattr(ta.api, "cancel_expired_orders", lambda *a, **kw: None)
    monkeypatch.setattr(ta, "_fire_order", lambda sym, act, px, reason, **kw: {"orderId": 444.0})
    monkeypatch.setattr(order_retry, "get_accepted_order_ids_for_owner", lambda sym, owner: ["333", "0"])
    monkeypatch.setattr(order_retry, "resolve_accepted_order", lambda sym, oid: None)
    tracked = ["333.0", "0", "nan"]
    out_ids = ta.track_and_place_order("BUY", "BTCUSDC", 1, 60000.0, 60100.0, order_ids=tracked)
    assert cancelled_oids == [("BTCUSDC", "333")]
    assert out_ids == ["444"]

def test_pricetrendanalyzer_linear_regression_trend_and_instant_trend_from_prices_edge_cases_iter166_s1():
    """[ITER166 #1] PriceTrendAnalyzer.linear_regression_trend and instant_trend_from_prices edge cases."""
    # 1. PriceTrendAnalyzer.linear_regression_trend and instant_trend_from_prices edge cases
    empty_pta = pw_mod.PriceTrendAnalyzer([])
    assert empty_pta.linear_regression_trend() == (None, None, None)
    pta = pw_mod.PriceTrendAnalyzer([100.0, 101.0, 103.0])
    trend_line, slp, rv = pta.linear_regression_trend()
    assert len(trend_line) == 3 and slp > 0
    assert pw_mod.instant_trend_from_prices([100.0, None], 0.8) == (0, 0.0, 0.0, 0.0)
    assert pw_mod.instant_trend_from_prices((x for x in [100.0, 102.0, 104.0]), 0.8)[0] == 1

def test_incrementaltrendbuffer_trim_front_true_eval_slice_eps_k_validation_and_log_return_std_iter166_s2():
    """[ITER166 #2] IncrementalTrendBuffer: trim_front(True), eval_slice eps_k validation, and log_return_std bounds check."""
    # 2. IncrementalTrendBuffer: trim_front(True), eval_slice eps_k validation, and log_return_std bounds check
    buf = pw_mod.IncrementalTrendBuffer()
    for idx in range(25):
        buf.append(100.0 + idx, 1700000000000 + idx * 1000)
    buf.trim_front(True)
    assert len(buf.prices) == 25
    ft, gc, sf, gr, eps = buf.eval_slice(0, 24, eps_k=-5.0)
    assert ft == 1 and eps >= 0.0
    assert buf.log_return_std(-5, 20) is None
    assert buf.log_return_std(0, 100) is None
    assert buf.log_return_std(0, 24) is not None

def test_pricewindow_from_cache24_with_string_window_seconds_and_windowanalyzer_check_price_change_iter166_s3():
    """[ITER166 #3] PriceWindow.from_cache24 with string window_seconds and WindowAnalyzer.check_price_change with string threshold."""
    # 3. PriceWindow.from_cache24 with string window_seconds and WindowAnalyzer.check_price_change with string threshold
    fake_c24 = SimpleNamespace(
        get_recent_entries=lambda sym, last_seconds: [[1700000000000 + i * 1000, 100.0 + i * 0.5] for i in range(15)],
        subscribe_price=lambda sub: None,
    )
    win = pw_mod.PriceWindow.from_cache24("BTCUSDC", "60.0", fake_c24)
    assert win.window_seconds == pytest.approx(60.0)
    wa = pw_mod.WindowAnalyzer(win)
    slope, pos = wa.check_price_change("1.0")
    assert slope != 0 and pos == 0
