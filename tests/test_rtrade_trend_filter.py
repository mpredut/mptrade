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


class ReaderProcessTrendTest(unittest.TestCase):
    """End to end with the REAL trend manager, as in production: the cacheManager writer
    computes and publishes the dynamic window, and rtrade's process holds an unstarted
    reader. The mocks above replaced the manager, so they could not see that a reader
    without Cache24 always answered None and kept rtrade standing aside for weeks."""

    WINDOW = 900.0

    def setUp(self):
        import json
        import tempfile
        from unittest import mock

        import cacheManager as cm

        self.cm = cm
        self.json = json
        self.tmp = tempfile.TemporaryDirectory()
        self.reader = cm.CachePriceShortTrendManager(
            ["TAOUSDC"], os.path.join(self.tmp.name, "trend_reader.json"))
        self._patches = [
            mock.patch.object(rtrade, "RTRADE_TREND_FILTER_ENABLED", True),
            mock.patch.object(rtrade, "RTRADE_TREND_FILTER_K", 2.0),
            mock.patch.object(rtrade, "RTRADE_TREND_WINDOW_SEC", self.WINDOW),
            mock.patch.object(cm, "CM_PUBLISHED_TREND_WINDOWS_SEC", [self.WINDOW]),
            mock.patch.object(cm, "get_short_trend_manager", return_value=self.reader),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        self.tmp.cleanup()

    def _publish(self, step, *, age_sec=0.0):
        """Let a writer compute the window from ticks and publish it for the reader."""
        import time

        class FakeCache24:
            def __init__(self, entries):
                self.entries = entries

            def get_recent_entries(self, _symbol, last_seconds):
                cutoff = (time.time() - last_seconds) * 1000
                return [e for e in self.entries if e[0] >= cutoff]

        now_ms = int(time.time() * 1000)
        # One tick per second over the window; a small alternating wiggle is the noise.
        entries = [[now_ms - (899 - i) * 1000, 300.0 + step * i + (0.05 if i % 2 else -0.05)]
                   for i in range(900)]
        writer = self.cm.CachePriceShortTrendManager(
            ["TAOUSDC"], os.path.join(self.tmp.name, "trend_writer.json"), writer=True)
        writer._cache24_managers = {"TAOUSDC": FakeCache24(entries)}
        published = writer._published_dynamic_windows("TAOUSDC")
        for dyn in published.values():
            dyn["ts"] -= age_sec
        with open(self.reader.filename, "w", encoding="utf-8") as fh:
            self.json.dump({"TAOUSDC": {"dynamic": published}}, fh)
        return published

    def test_reader_uses_the_published_window(self):
        published = self._publish(step=0.0)
        self.assertIn("900", published)
        self.assertIsNotNone(self.reader.get_instant_trend_for_window("TAOUSDC", self.WINDOW))
        # A sideways market is exactly when the spread bot should run.
        self.assertFalse(rtrade._trend_too_strong("TAOUSDC"))

    def test_strong_trend_still_stands_aside(self):
        self._publish(step=0.2)
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))

    def test_stale_or_missing_publication_fails_closed(self):
        stale = self.cm.CachePriceShortTrendManager.TREND_STALE_SEC + 5
        self._publish(step=0.0, age_sec=stale)
        self.assertIsNone(self.reader.get_instant_trend_for_window("TAOUSDC", self.WINDOW))
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))
        os.remove(self.reader.filename)
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
