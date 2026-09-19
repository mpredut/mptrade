"""
Tests for tradeall.py — PriceWindow, PriceTrendAnalyzer, TrendState
and the integration with Cache24PriceManager.

Acoperire:
  - PriceTrendAnalyzer: linreg, gradient, date insuficiente
  - PriceWindow: sample_rate_sec, recent_n, process_price, get_trend (4 values)
  - PriceWindow.from_cache24: the factory from Cache24PriceManager with real data
  - PriceWindow._sample_rate_from_entries: rate computed from timestamps
  - TrendState: lifecycle complet
  - TrendState: cooldown per instanta de trend (fire_limit_reached/mark_confirmed
    with FIRE_MAX_PER_TREND executions, can_retry_fire/mark_fire_attempt with
    FIRE_MIN_RETRY_INTERVAL_SEC, reset la start_trend nou) — 22 iul
  - CacheCurrentPriceManager: get_sample_rate / get_update_frequency
"""
import os, sys, json, time, tempfile, threading, unittest
from collections import deque
from unittest.mock import MagicMock, patch

# ── mocks for the external dependencies (before any local import) ─────────
_mock_bapi = MagicMock()
_mock_bapi.get_current_price = MagicMock(return_value=60000.0)
_mock_bapi.cancel_order      = MagicMock(return_value=True)
_mock_bapi.cancel_expired_orders = MagicMock()
_mock_bapi.client            = MagicMock()

_mock_sym = MagicMock()
_mock_sym.symbols   = ["BTCUSDT"]
_mock_sym.btcsymbol = "BTCUSDT"
_mock_sym.validate_ordertype = MagicMock()
_mock_market_api = MagicMock()
_mock_market_api.api = MagicMock()

_IMPORT_MOCKS = dict([
    ("bapi",            _mock_bapi),
    ("binance_api.bapi", _mock_bapi),
    ("symbols",         _mock_sym),
    ("bapi_trades",     MagicMock(**{"get_my_trades_24.return_value": []})),
    ("binance_api.bapi_trades", MagicMock(**{"get_my_trades_24.return_value": []})),
    ("bapi_allorders",  MagicMock()),
    ("binance_api.bapi_allorders", MagicMock()),
    ("bapi_placeorder", MagicMock()),
    ("binance_api.bapi_placeorder", MagicMock()),
    ("providers.market_api", _mock_market_api),
    ("alertnotifiers",  MagicMock()),
    ("generateweb",     MagicMock()),
    ("log",             MagicMock()),
    ("keys",            MagicMock()),
    ("keys.apikeys",    MagicMock(**{"api_key_ws": "fake"})),
])
_PRELOADED_IMPORTS = {
    name: sys.modules[name] for name in _IMPORT_MOCKS if name in sys.modules
}
_BINANCE_CHILDREN = ("bapi", "bapi_trades", "bapi_allorders", "bapi_placeorder")
sys.modules.update(_IMPORT_MOCKS)

try:
    # cacheManager starts a WS thread at import — we block it
    with patch("cacheManager._initialize_once", return_value=None):
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

mock_bapi = _mock_bapi


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

    def test_strategy_preserves_common_persistent_retry_policy(self):
        with (patch.object(ta, "_kalman_gate_blocks", return_value=(False, "off", None)),
              patch.object(ta, "TRADEALL_FIRE_SYMBOLS", {"BTCUSDT"}),
              patch.object(ta.mkt, "place", return_value={"orderId": "accepted"}) as place):
            result = ta._fire_order("BTCUSDT", "BUY", 100.0, "test")
        self.assertIsNotNone(result)
        self.assertNotIn("caller_owns_retry", place.call_args.kwargs)

    def test_symbol_outside_fire_allowlist_is_never_traded(self):
        # A coin can be trend-tracked (present in symbols.py) without tradeall
        # trading it: a manual position guarded by the trailing stop. The gate is
        # the single choke point, so neither logic() nor Kalman-primary can fire.
        with (patch.object(ta, "_kalman_gate_blocks", return_value=(False, "off", None)),
              patch.object(ta, "TRADEALL_FIRE_SYMBOLS", {"BTCUSDT"}),
              patch.object(ta.mkt, "place", return_value={"orderId": "accepted"}) as place):
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

    def test_kalman_gate_fails_closed_for_blind_buy_but_allows_sell(self):
        # Missing/stale confirming signal: block an additive BUY (do not buy blind) but
        # never a SELL -- a risk-reducing exit must never be trapped by a data outage.
        class _AbsentShadow:
            def current_trend(self, _sym):
                return None, 0.0
        class _StaleShadow:
            def current_trend(self, _sym):
                return 1, ta.GATE_STALE_SEC + 1.0
        for shadow in (_AbsentShadow(), _StaleShadow()):
            with (patch.object(ta, "_shadow_ref", shadow),
                  patch.dict(ta.KALMAN_MODES, {"TAOUSDC": "strict"})):
                self.assertTrue(ta._kalman_gate_blocks("TAOUSDC", "BUY")[0])
                self.assertFalse(ta._kalman_gate_blocks("TAOUSDC", "SELL")[0])


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _window(prices, sample_rate=0.8, symbol="BTCUSDT"):
    pw = ta.PriceWindow(symbol, len(prices), sample_rate_sec=sample_rate)
    for p in prices:
        pw.process_price(p)
    return pw


def _make_cache24_manager(symbol, entries, tmp_dir):
    """Create a Cache24PriceManager pre-populated with entries = [[ts_ms, price], ...]."""
    fname = os.path.join(tmp_dir, f"cache_24_{symbol}.json")
    with open(fname, "w") as f:
        json.dump({"items": {symbol: entries}, "fetchtime": {}}, f)
    mgr = cm.Cache24PriceManager(
        sync_ts   = 9999,
        symbols   = [symbol],
        filename  = fname,
        api_client= mock_bapi,
    )
    return mgr


def _synthetic_entries(n=100, start_price=60000.0, delta=10.0,
                       interval_ms=800, start_ts_ms=None):
    """Generate n [ts_ms, price] entries with a constant interval and delta."""
    if start_ts_ms is None:
        start_ts_ms = int(time.time() * 1000) - n * interval_ms
    return [
        [start_ts_ms + i * interval_ms, start_price + i * delta]
        for i in range(n)
    ]


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
        import tempfile
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

class TestPriceWindowFromCache24(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self, entries, symbol="BTCUSDT", window_seconds=None):
        mgr = _make_cache24_manager(symbol, entries, self.tmp)
        if window_seconds is None:
            window_seconds = len(entries) * 0.8
        return ta.PriceWindow.from_cache24(symbol, window_seconds, mgr)

    # ── date sintetice ──────────────────────────────────────────────────────

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
            final_trend, gc, sf, gr = pw.get_trend()
            self.assertEqual(final_trend, 1)
            self.assertGreater(gc, 0)
            
        with self.subTest(msg="downtrend"):
            entries = _synthetic_entries(60, delta=-10.0, interval_ms=800)
            pw = self._make(entries, window_seconds=60 * 0.8)
            final_trend, gc, sf, gr = pw.get_trend()
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
            _, _, large_slope, _ = large.get_trend()
            _, _, small_slope, _ = small.get_trend()
    
            self.assertGreater(large_slope, 0)
            self.assertLess(small_slope, 0)
            self.assertLess(len(small.prices), len(large.prices))


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — get_trend()'s 4 values
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowGetTrend(unittest.TestCase):

    def test_returns_four_values(self):
        pw = _window([100 + i for i in range(20)])
        self.assertEqual(len(pw.get_trend()), 4)

    def test_directional_trends(self):
        cases = (
            ("up", [100 + i * 2 for i in range(30)], 1),
            ("down", [200 - i * 2 for i in range(30)], -1),
        )
        for label, prices, expected in cases:
            with self.subTest(direction=label):
                final_trend, growth, slope, gradient = _window(prices).get_trend()
                self.assertEqual(final_trend, expected)
                self.assertGreater(growth * expected, 0)
                self.assertGreater(slope * expected, 0)
                self.assertGreater(gradient * expected, 0)
                self.assertEqual(final_trend, 1 if growth > 0 else -1)

    def test_gc_is_average_of_sf_and_gr(self):
        pw = _window([100 + i for i in range(20)])
        _, gc, sf, gr = pw.get_trend()
        self.assertAlmostEqual(gc, (sf + gr) / 2.0, places=10)

    def test_neutral_series_return_zero_trend(self):
        for label, prices in (("single", [100.0]), ("constant", [100.0] * 20)):
            with self.subTest(case=label):
                final_trend, growth, slope, gradient = _window(prices).get_trend()
                self.assertEqual(final_trend, 0)
                self.assertAlmostEqual(growth, 0.0, places=6)
                self.assertAlmostEqual(slope, 0.0, places=6)
                self.assertAlmostEqual(gradient, 0.0, places=6)

    def test_recent_gradient_captures_late_reversal(self):
        # Overall UP trend, but the last 5 prices drop sharply
        prices = [100 + i for i in range(30)] + [129 - i * 8 for i in range(1, 6)]
        pw = _window(prices, sample_rate=0.8)
        _, _, _, gr = pw.get_trend()
        self.assertLess(gr, 0)   # momentumul recent e negativ

    def test_slope_full_sees_whole_window(self):
        # Overall UP trend even though the last 2 prices dip slightly
        prices = [100 + i for i in range(30)] + [129, 128]
        pw = _window(prices)
        _, _, sf, _ = pw.get_trend()
        self.assertGreater(sf, 0)

# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — min/max/slope/proximities
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowMinMax(unittest.TestCase):

    def test_get_min_max(self):
        pw = _window([100, 105, 95, 110, 90])
        self.assertAlmostEqual(pw.get_min(), 90.0, delta=1.0)
        self.assertAlmostEqual(pw.get_max(), 110.0, delta=1.0)

    def test_sorted_consistency(self):
        pw = _window([50, 80, 60, 70, 90])
        self.assertEqual(len(pw.prices), len(pw.sorted_prices))

    def test_eviction(self):
        pw = ta.PriceWindow("BTCUSDT", 3)
        for p in [10, 20, 30, 40]:
            pw.process_price(p)
        self.assertEqual(len(pw.prices), 3)
        self.assertNotIn(10, pw.prices)

    def test_proximities(self):
        analyzer = ta.WindowAnalyzer(_window([100, 200]))
        cases = (("midpoint", 150, 0.5, 0.5), ("minimum", 100, 0.0, 1.0))
        for label, price, expected_min, expected_max in cases:
            with self.subTest(position=label):
                min_proximity, max_proximity = analyzer.calculate_proximities(price)
                self.assertAlmostEqual(min_proximity, expected_min, places=5)
                self.assertAlmostEqual(max_proximity, expected_max, places=5)


# ═══════════════════════════════════════════════════════════════════════════
# TrendState
# ═══════════════════════════════════════════════════════════════════════════

class TestTrendState(unittest.TestCase):

    def setUp(self):
        self.now = 1000.0

    def _ts(self, exp_time=9999, fresh_time=60):
        return ta.TrendState(3600, exp_time, fresh_time, now_fn=lambda: self.now)

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
    """Cooldown per instanta de trend (22 iul) — vezi FIRE_MIN_RETRY_INTERVAL_SEC.
    A fake clock (a mutable counter) so we do not depend on real 30-minute sleeps."""

    def _ts_with_clock(self):
        clock = {"t": 1000.0}
        ts = ta.TrendState(3600, 9999, 60, now_fn=lambda: clock["t"])
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

class TestCacheCurrentPriceFrequency(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        fname = os.path.join(self.tmp, "cp.json")
        self.mgr = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        # Construction (missing file -> init fetch) can record a timestamp.
        # We start the frequency measurement clean to test the mechanism in isolation.
        self.mgr._update_timestamps.clear()

    def test_empty_state_uses_fallback_and_zero_frequency(self):
        self.assertAlmostEqual(self.mgr.get_sample_rate("BTCUSDT", fallback=0.8), 0.8)
        self.assertEqual(self.mgr.get_update_frequency("BTCUSDT"), 0.0)

    def test_updates_produce_sample_rate_and_frequency(self):
        t0 = time.time()
        self.mgr.on_items_update("BTCUSDT", [60000.0])
        time.sleep(0.15)
        self.mgr.on_items_update("BTCUSDT", [60001.0])
        elapsed = time.time() - t0
        rate = self.mgr.get_sample_rate("BTCUSDT", fallback=9.9)
        # the measured rate is approximately the real interval between the 2 updates (not the fallback)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 9.9)                  # It is not the fallback.
        self.assertLessEqual(rate, elapsed + 0.5)   # robust la jitter de scheduling
        self.assertGreater(self.mgr.get_update_frequency("BTCUSDT"), 0.0)

    def test_old_timestamps_trimmed(self):
        old_ts = time.time() - cm.CacheCurrentPriceManager.FREQ_WINDOW_SEC - 10
        self.mgr._update_timestamps["BTCUSDT"].append(old_ts)
        self.mgr.on_items_update("BTCUSDT", [60000.0])
        dq = self.mgr._update_timestamps["BTCUSDT"]
        cutoff = time.time() - cm.CacheCurrentPriceManager.FREQ_WINDOW_SEC - 1
        self.assertTrue(all(t > cutoff for t in dq))

    def test_single_update_returns_fallback(self):
        self.mgr.on_items_update("BTCUSDT", [60000.0])
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
        for s in list(self._subs):
            s.on_items_update(symbol, [price])


class _RecordingSubscriber:
    def __init__(self):
        self.events = []
    def on_price_update(self, symbol, ts_ms, price):
        self.events.append((symbol, price))


class TestSubscriberPatternInheritance(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _cache24(self, symbol="BTCUSDT"):
        return _make_cache24_manager(symbol, _synthetic_entries(5), self.tmp)

    def test_managers_inherit_subscribe_price(self):
        cases = (
            ("cache24", type(self._cache24()).subscribe_price),
            ("current-price", cm.CacheCurrentPriceManager.subscribe_price),
        )
        for label, method in cases:
            with self.subTest(manager=label):
                self.assertIs(method, cm.CacheManagerInterface.subscribe_price)

    def test_inherited_notify_reaches_subscriber(self):
        mgr = self._cache24("BTCUSDT")
        rec = _RecordingSubscriber()
        mgr.subscribe_price(rec)
        mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 123.0)
        self.assertIn(("BTCUSDT", 123.0), rec.events)

    def test_attach_ws_manager_wires_chain(self):
        """WS tick → CacheCurrentPrice.on_items_update → subscriber.on_price_update."""
        fname = os.path.join(self.tmp, "cp_ws.json")
        mgr = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        ws = _FakeWSManager()
        mgr.attach_ws_manager(ws)

        rec = _RecordingSubscriber()
        mgr.subscribe_price(rec)

        ws.push("BTCUSDT", 67000.0)   # simulates a WS tick
        self.assertIn(("BTCUSDT", 67000.0), rec.events)

    def test_attach_ws_manager_idempotent(self):
        fname = os.path.join(self.tmp, "cp_ws2.json")
        mgr = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        ws = _FakeWSManager()
        mgr.attach_ws_manager(ws)
        mgr.attach_ws_manager(ws)
        self.assertEqual(ws._subs.count(mgr), 1)

    def test_ws_tick_marks_ws_healthy(self):
        fname = os.path.join(self.tmp, "cp_ws3.json")
        mgr = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        self.assertFalse(mgr._ws_is_healthy())   # no event yet
        mgr.on_items_update("BTCUSDT", [50000.0])
        self.assertTrue(mgr._ws_is_healthy())    # WS marcat activ


# ═══════════════════════════════════════════════════════════════════════════
# PriceWindow — wiring complet Cache24PriceManager → PriceWindow
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWindowCache24Wiring(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

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
            pw.unsubscribe_from_cache24(mgr)
            self.assertFalse(pw._subscribed_to_cache24)
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

    def test_pricewindow_api_surface_is_lean(self):
        pw = _window([100, 110, 105])
        for method in ("calculate_proximities", "calculate_slope_max_min",
                       "check_price_change", "evaluate_buy_sell_opportunity"):
            with self.subTest(absent=method):
                self.assertFalse(hasattr(pw, method))
        for method in ("get_min", "get_max", "get_instant_trend"):
            with self.subTest(present=method):
                self.assertTrue(hasattr(pw, method))

    def test_get_trend_alias(self):
        pw = _window([100 + i for i in range(20)])
        self.assertEqual(pw.get_trend(), pw.get_instant_trend())

    def test_recent_gradient_cases(self):
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

    def test_noise_epsilon_cases(self):
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

    def test_slope_max_min_cases(self):
        for label, prices, positive in (
            ("up", [100 + i for i in range(20)], True),
            ("constant", [100.0] * 10, False),
        ):
            with self.subTest(case=label):
                slope = ta.WindowAnalyzer(_window(prices)).calculate_slope_max_min()
                self.assertGreater(slope, 0) if positive else self.assertEqual(slope, 0)

    def test_check_price_change_cases(self):
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

    def test_evaluate_buy_sell_cases(self):
        cases = (
            ("directional", [100 + i for i in range(20)], 120.0, None, None),
            ("below-threshold", [100.0, 100.01, 100.02], 100.02, 5.0, "HOLD"),
        )
        for label, prices, current, threshold, expected in cases:
            with self.subTest(case=label):
                analyzer = ta.WindowAnalyzer(_window(prices))
                if threshold is None:
                    action, _, _, _ = analyzer.evaluate_buy_sell_opportunity(current)
                else:
                    action, _, _, _ = analyzer.evaluate_buy_sell_opportunity(
                        current, threshold_percent=threshold)
                self.assertIn(action, ("BUY", "SELL", "HOLD"))
                if expected is not None:
                    self.assertEqual(action, expected)

    def test_calculate_positions_returns_fractions(self):
        pw = _window([100 + i for i in range(10)])
        an = ta.WindowAnalyzer(pw)
        min_pos, max_pos = an.calculate_positions()
        self.assertIsNotNone(min_pos)
        self.assertIsNotNone(max_pos)

    def test_analyze_price_movement_returns_tuple(self):
        # the complicated logic restored — it must return (slope, price_diff)
        pw = _window([100 + i for i in range(20)])
        an = ta.WindowAnalyzer(pw)
        result = an._analyze_price_movement(100, 0, 119, 19, 119, 19, 19.0)
        self.assertEqual(len(result), 2)

    def test_analyzer_shares_window_mutation(self):
        # composition: the analyzer sees the window's changes (the same object)
        pw = _window([100, 101, 102])
        an = ta.WindowAnalyzer(pw)
        before = pw.get_max()
        pw.process_price(200.0)
        self.assertGreater(pw.get_max(), before)
        self.assertIs(an.window, pw)


# ═══════════════════════════════════════════════════════════════════════════
# TrendCoordinator — event-driven + heartbeat + cache
# ═══════════════════════════════════════════════════════════════════════════

class TestTrendCoordinator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        entries = _synthetic_entries(60, interval_ms=800)
        base = entries[0][0]
        now_ms = int(time.time() * 1000)
        entries = [[now_ms + (e[0] - base), e[1]] for e in entries]
        self.cache24 = _make_cache24_manager("BTCUSDT", entries, self.tmp)

        fname = os.path.join(self.tmp, "cp.json")
        self.cpm = cm.CacheCurrentPriceManager(
            sync_ts=9999, symbols=["BTCUSDT"],
            filename=fname, ws_manager=None, api_client=mock_bapi,
        )
        self.cpm.on_items_update("BTCUSDT", [60000.0])

        # The manager owns the windows + calc + cross-process cache
        self.mgr = cm.CachePriceShortTrendManager(["BTCUSDT"], os.path.join(self.tmp, "trend.json"))
        self.mgr.start_computation({"BTCUSDT": self.cache24}, self.cpm)

    def _make_coord(self):
        return ta.TrendCoordinator(
            symbols=["BTCUSDT"],
            instant_mgr=self.mgr,
            current_price_mgr=self.cpm,
            cache24_managers={"BTCUSDT": self.cache24},
            min_interval=2.0, max_interval=30.0,
        )

    def test_manager_owns_windows(self):
        self.assertIsNotNone(self.mgr.get_window("BTCUSDT"))
        self.assertIsNotNone(self.mgr.get_analyzer("BTCUSDT"))
        self.assertGreater(len(self.mgr.get_window("BTCUSDT").prices), 0)

    def test_dirty_set_on_price_update(self):
        coord = self._make_coord()
        coord._dirty["BTCUSDT"] = False
        coord.on_price_update("BTCUSDT", int(time.time() * 1000), 60001.0)
        self.assertTrue(coord._dirty["BTCUSDT"])
        self.assertTrue(coord._event.is_set())

    def test_due_policy_for_dirty_and_heartbeat(self):
        coord = self._make_coord()
        now = time.time()
        coord._last_eval["BTCUSDT"] = now
        coord._dirty["BTCUSDT"] = True
        self.assertFalse(coord._is_due("BTCUSDT", now + 0.5))
        self.assertTrue(coord._is_due("BTCUSDT", now + 2.5))
        coord._dirty["BTCUSDT"] = False
        self.assertTrue(coord._is_due("BTCUSDT", now + 31.0))
        self.assertFalse(coord._is_due("BTCUSDT", now + 5.0))

    def test_invalid_intervals_fail_fast_and_symbols_are_deduplicated(self):
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

    def test_invalid_price_is_not_evaluated(self):
        coord = self._make_coord()
        with (patch.object(self.cpm, "get_price", return_value=[time.time() * 1000, float("nan")]),
              patch.object(ta, "handle_symbol") as handle):
            self.assertIsNone(coord.evaluate("BTCUSDT"))
        handle.assert_not_called()

    def test_stale_or_future_price_is_not_evaluated(self):
        coord = self._make_coord()
        now_ms = time.time() * 1000
        with patch.object(ta, "handle_symbol") as handle:
            for timestamp in (now_ms - self.cpm.STALE_THRESHOLD_MS - 1, now_ms + 5_000):
                with self.subTest(timestamp=timestamp):
                    with patch.object(self.cpm, "get_price", return_value=[timestamp, 60_000.0]):
                        self.assertIsNone(coord.evaluate("BTCUSDT"))
        handle.assert_not_called()

    def test_stop_wakes_run_loop(self):
        coord = self._make_coord()
        thread = threading.Thread(target=coord.run)
        thread.start()
        coord.stop()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_evaluation_cache_lifecycle(self):
        coord = self._make_coord()
        self.assertIsNone(coord.get_cached_trend("BTCUSDT"))
        self.assertEqual(coord.get_all_cached_trends(), {})
        snap = coord.evaluate("BTCUSDT")
        self.assertIsNotNone(snap)
        cached = coord.get_cached_trend("BTCUSDT")
        for key in ("final_trend", "growth_coefficient", "slope_full",
                    "gradient_recent", "slope_small", "slope_big",
                    "slope_max_min", "pos", "current_price", "ts"):
            with self.subTest(field=key):
                self.assertIn(key, cached)
        self.assertFalse(coord._dirty["BTCUSDT"])
        self.assertIn("BTCUSDT", coord.get_all_cached_trends())
        manager_snapshot = self.mgr.get_snapshot("BTCUSDT")
        self.assertIsNotNone(manager_snapshot)
        self.assertIn("slope_big", manager_snapshot)

    def test_manager_tick_publishes_instant_gradient(self):
        # the fast channel lives in the MANAGER: on_price_update publishes the gradient, under
        # gradient_recent_fast (29 Jul: gradient_recent now belongs EXCLUSIVELY to the
        # lente, evaluate_full — vezi cachemanager-trend-race-investigation).
        self.mgr.on_price_update("BTCUSDT", int(time.time() * 1000), 60500.0)
        snap = self.mgr.get_snapshot("BTCUSDT")
        self.assertIsNotNone(snap)
        self.assertIn("gradient_recent_fast", snap)
        self.assertIn("epsilon", snap)
        self.assertEqual(snap["current_price"], 60500.0)

    def test_coordinator_and_windows_subscribed_to_cache24(self):
        coord = self._make_coord()
        self.assertIn(coord, self.cache24._price_subscribers)
        self.assertTrue(self.mgr.get_window("BTCUSDT")._subscribed_to_cache24)
        self.assertTrue(self.mgr.get_window("BTCUSDT", self.mgr.window_big_sec)._subscribed_to_cache24)

    def test_tick_updates_window_and_marks_dirty(self):
        coord = self._make_coord()
        coord._dirty["BTCUSDT"] = False
        win = self.mgr.get_window("BTCUSDT")
        self.cache24.on_price_update("BTCUSDT", int(time.time() * 1000), 61234.0)
        self.assertTrue(coord._dirty["BTCUSDT"])
        self.assertIn(61234.0, win.prices)

    def test_concurrent_update_and_read_no_crash(self):
        """A WS thread updates the window while the evaluation reads it."""
        import threading as _t
        coord = self._make_coord()
        stop = _t.Event()
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
                    self.mgr.get_instant_trend("BTCUSDT")
                    self.mgr.get_analyzer("BTCUSDT").calculate_slope_max_min()
                except Exception as e:
                    errors.append(("reader", e))

        threads = [
            _t.Thread(target=writer, name="writer"),
            _t.Thread(target=reader, name="reader_1"),
            _t.Thread(target=reader, name="reader_2"),
        ]
        for th in threads:
            th.start()
        time.sleep(1.0)
        stop.set()
        for th in threads:
            th.join(timeout=5)
        self.assertEqual(errors, [], f"concurrency errors: {errors[:3]}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
