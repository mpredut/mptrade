"""rtrade _trend_too_strong / _followup_force / market_exit_allowed: the trend filter that
makes the spread bot stand aside when the asset is clearly trending (|gradient_recent| >
K*epsilon) AND when the trend signal is unavailable (fail-CLOSED on no data), plus a kill
switch. A fake cacheManager is injected into sys.modules (imported lazily in the functions)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import rtrade


class TrendFilterTest(unittest.TestCase):
    def setUp(self):
        self._en = rtrade.RTRADE_TREND_FILTER_ENABLED
        self._k = rtrade.RTRADE_TREND_FILTER_K
        self._cm = sys.modules.get("cacheManager")
        rtrade.RTRADE_TREND_FILTER_ENABLED = True
        rtrade.RTRADE_TREND_FILTER_K = 2.0

    def tearDown(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = self._en
        rtrade.RTRADE_TREND_FILTER_K = self._k
        if self._cm is not None:
            sys.modules["cacheManager"] = self._cm
        else:
            sys.modules.pop("cacheManager", None)

    def _fake_cm(self, dyn):
        m = types.ModuleType("cacheManager")

        class Mgr:
            def get_instant_trend_for_window(self, sym, w):
                return dyn
        m.get_short_trend_manager = lambda: Mgr()
        sys.modules["cacheManager"] = m

    def test_disabled_returns_false(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = False
        self._fake_cm({"gradient_recent": 100.0, "epsilon": 0.001})   # ar fi trend f. clar
        self.assertFalse(rtrade._trend_too_strong("TAOUSDC"))

    def test_snapshot_strength_cases(self):
        cases = (
            ("strong", {"gradient_recent": 0.5, "epsilon": 0.1}, True),
            ("weak", {"gradient_recent": 0.15, "epsilon": 0.1}, False),
            ("unavailable", None, True),   # no data -> stand aside (fail-closed)
            ("flat", {"gradient_recent": 0.0, "epsilon": 0.0}, False),
        )
        for label, snapshot, expected in cases:
            with self.subTest(case=label):
                self._fake_cm(snapshot)
                self.assertEqual(rtrade._trend_too_strong("TAOUSDC"), expected)

    def test_exception_fails_closed(self):
        m = types.ModuleType("cacheManager")

        def boom():
            raise RuntimeError("cm down")
        m.get_short_trend_manager = boom
        sys.modules["cacheManager"] = m
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))          # An error -> stand aside (fail-closed).

class FollowupForceTest(unittest.TestCase):
    """Follow-up (flip after a fill): force=market ONLY if the trend is not adverse. Adverse:
    A SELL into a decline / a BUY into a rise -> False (a patient limit, it does not dump into the market)."""
    def setUp(self):
        self._en = rtrade.RTRADE_TREND_FILTER_ENABLED
        self._k = rtrade.RTRADE_TREND_FILTER_K
        self._cm = sys.modules.get("cacheManager")
        rtrade.RTRADE_TREND_FILTER_ENABLED = True
        rtrade.RTRADE_TREND_FILTER_K = 2.0

    def tearDown(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = self._en
        rtrade.RTRADE_TREND_FILTER_K = self._k
        if self._cm is not None:
            sys.modules["cacheManager"] = self._cm
        else:
            sys.modules.pop("cacheManager", None)

    def _fake_cm(self, dyn):
        m = types.ModuleType("cacheManager")

        class Mgr:
            def get_instant_trend_for_window(self, sym, w):
                return dyn
        m.get_short_trend_manager = lambda: Mgr()
        sys.modules["cacheManager"] = m

    def test_directional_force_policy(self):
        cases = (
            ("sell-down", "SELL", -0.5, False),
            ("sell-up", "SELL", 0.5, True),
            ("buy-up", "BUY", 0.5, False),
            ("buy-down", "BUY", -0.5, True),
            ("weak-sell", "SELL", 0.15, True),
        )
        for label, side, gradient, expected in cases:
            with self.subTest(case=label):
                self._fake_cm({"gradient_recent": gradient, "epsilon": 0.1})
                self.assertEqual(rtrade._followup_force("TAOUSDC", side), expected)

    def test_disabled_forces_but_no_data_uses_patient_limit(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = False
        self._fake_cm({"gradient_recent": -0.5, "epsilon": 0.1})
        self.assertTrue(rtrade._followup_force("TAOUSDC", "SELL"))    # kill switch off -> as before
        rtrade.RTRADE_TREND_FILTER_ENABLED = True
        self._fake_cm(None)
        self.assertFalse(rtrade._followup_force("TAOUSDC", "SELL"))   # no data -> patient limit (fail-closed)


class MarketRegimeTest(unittest.TestCase):
    def setUp(self):
        self._en = rtrade.RTRADE_TREND_FILTER_ENABLED
        self._k = rtrade.RTRADE_TREND_FILTER_K
        self._cm = sys.modules.get("cacheManager")
        rtrade.RTRADE_TREND_FILTER_ENABLED = True
        rtrade.RTRADE_TREND_FILTER_K = 2.0

    def tearDown(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = self._en
        rtrade.RTRADE_TREND_FILTER_K = self._k
        if self._cm is not None:
            sys.modules["cacheManager"] = self._cm
        else:
            sys.modules.pop("cacheManager", None)

    def _fake_cm(self, dyn):
        m = types.ModuleType("cacheManager")
        class Mgr:
            def get_instant_trend_for_window(self, _sym, _window):
                return dyn
        m.get_short_trend_manager = lambda: Mgr()
        sys.modules["cacheManager"] = m

    def test_regimes_use_signed_gradient_and_adaptive_noise(self):
        for gradient, expected in ((0.5, "bull"), (-0.5, "bear"),
                                   (0.15, "sideways")):
            with self.subTest(expected=expected):
                self._fake_cm({"gradient_recent": gradient, "epsilon": 0.1})
                self.assertEqual(
                    rtrade._market_regime_decision("TAOUSDC").regime, expected)

    def test_missing_signal_is_unknown(self):
        self._fake_cm(None)
        self.assertEqual(
            rtrade._market_regime_decision("TAOUSDC").regime, "unknown")

    def test_trend_too_strong_stands_aside_on_no_data_but_trades_when_flat(self):
        # No trend data (regime "unknown") must fail CLOSED -> stand aside, not trade
        # blind. A data outage must never be read as "flat".
        self._fake_cm(None)
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))
        # A CONFIRMED flat (sideways) market is the spread bot's home -> run.
        self._fake_cm({"gradient_recent": 0.15, "epsilon": 0.1})
        self.assertFalse(rtrade._trend_too_strong("TAOUSDC"))
        # A clear trend -> stand aside (adverse selection).
        for gradient in (0.5, -0.5):
            self._fake_cm({"gradient_recent": gradient, "epsilon": 0.1})
            self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))

    def test_followup_force_uses_patient_limit_on_no_data(self):
        # No trend data (regime "unknown") -> patient limit (False), never a blind
        # market flip.
        self._fake_cm(None)
        self.assertFalse(rtrade._followup_force("TAOUSDC", "SELL"))
        self.assertFalse(rtrade._followup_force("TAOUSDC", "BUY"))
        # Confirmed flat -> an immediate market flip is still permitted.
        self._fake_cm({"gradient_recent": 0.15, "epsilon": 0.1})
        self.assertTrue(rtrade._followup_force("TAOUSDC", "SELL"))
        # Adverse directional -> patient limit (pre-existing behavior preserved).
        self._fake_cm({"gradient_recent": -0.5, "epsilon": 0.1})   # bear
        self.assertFalse(rtrade._followup_force("TAOUSDC", "SELL"))


class MarketExitModeTest(unittest.TestCase):
    """emergency_only: a discretionary / adverse hard stop WAITS (the round holds); only a
    real loss >= the emergency threshold still forces a market exit."""

    def test_emergency_only_holds_until_emergency_even_when_regime_is_adverse(self):
        import types
        from unittest.mock import patch
        fake = types.SimpleNamespace(symbol="TAOUSDC")
        adverse = types.SimpleNamespace(
            regime="bear", strength=5.0, reason="test", adverse_to=lambda side: True)
        thr = rtrade.RTRADE_EMERGENCY_HARD_STOP_PCT
        with (patch.object(rtrade, "RTRADE_DYNAMIC_MARKET_EXIT_MODE", "emergency_only"),
              patch.object(rtrade, "_market_regime_decision", return_value=adverse)):
            # Below the emergency loss: HOLD (do not stop out on a maybe-false drop),
            # even though the regime reads adverse.
            self.assertFalse(
                rtrade._LivePairVenue.market_exit_allowed(fake, "LONG", thr - 0.01, "hard_stop"))
            # At/above the emergency loss: MARKET exit (catastrophe backstop preserved).
            self.assertTrue(
                rtrade._LivePairVenue.market_exit_allowed(fake, "LONG", thr, "emergency"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
