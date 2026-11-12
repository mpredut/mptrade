"""rtrade _trend_too_strong / _followup_force / market_exit_allowed: the trend filter that
makes the spread bot stand aside when the asset is clearly trending (|gradient_recent| >
K*epsilon) AND when the trend signal is unavailable (fail-CLOSED on no data), plus a kill
switch. A fake cacheManager is injected into sys.modules (imported lazily in the functions)."""

import cacheManager as cm
import json
import os
import rtrade
import sys
import time
import types
import unittest
from tests.conftest import TempDirTestCase
from unittest.mock import patch




class _RTradeTrendBase(unittest.TestCase):
    def setUp(self):
        self._en = rtrade.RTRADE_TREND_FILTER_ENABLED
        self._k = rtrade.RTRADE_TREND_FILTER_K
        self._flip = getattr(rtrade, "RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED", False)
        self._cm = sys.modules.get("cacheManager")
        self._rtrade_cm = rtrade.cm
        rtrade.RTRADE_TREND_FILTER_ENABLED = True
        rtrade.RTRADE_TREND_FILTER_K = 2.0
        rtrade.RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED = True

    def tearDown(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = self._en
        rtrade.RTRADE_TREND_FILTER_K = self._k
        rtrade.RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED = self._flip
        rtrade.cm = self._rtrade_cm
        if self._cm is not None:
            sys.modules["cacheManager"] = self._cm
        else:
            sys.modules.pop("cacheManager", None)

    def _fake_cm(self, dyn):
        m = types.ModuleType("cacheManager")
        m.get_short_trend_manager = lambda: types.SimpleNamespace(
            get_short_trend_for_window=lambda _sym, _window: dyn
        )
        sys.modules["cacheManager"] = m
        rtrade.cm = m


class TrendFilterTest(_RTradeTrendBase):
    def test_disabled_returns_false(self):
        rtrade.RTRADE_TREND_FILTER_ENABLED = False
        self._fake_cm({"gradient_recent": 100.0, "epsilon": 0.001})   # would be a very clear trend
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
        rtrade.cm = m
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))          # An error -> stand aside (fail-closed).


class FollowupForceTest(_RTradeTrendBase):
    """Follow-up (flip after a fill): force=market ONLY if the trend is not adverse. Adverse:
    A SELL into a decline / a BUY into a rise -> False (a patient limit, it does not dump into the market)."""
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

    def test_market_flip_disabled_always_uses_patient_limit(self):
        rtrade.RTRADE_FOLLOWUP_MARKET_FLIP_ENABLED = False
        self._fake_cm({"gradient_recent": 0.5, "epsilon": 0.1})
        self.assertFalse(rtrade._followup_force("TAOUSDC", "SELL"))
        self._fake_cm({"gradient_recent": 0.15, "epsilon": 0.1})
        self.assertFalse(rtrade._followup_force("TAOUSDC", "SELL"))
        self.assertFalse(rtrade._followup_force("TAOUSDC", "BUY"))


class MarketRegimeTest(_RTradeTrendBase):
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


class ReaderProcessTrendTest(TempDirTestCase):
    """End to end with the REAL trend manager, as in production: the cacheManager writer
    computes and publishes the dynamic window, and rtrade's process holds an unstarted
    reader. The mocks above replaced the manager, so they could not see that a reader
    without Cache24 always answered None and kept rtrade standing aside for weeks."""

    WINDOW = 900.0

    def setUp(self):
        super().setUp()
        self.cm = cm
        self.json = json
        self.reader = cm.CacheShortTrendManager(
            ["TAOUSDC"], os.path.join(self.tmp, "trend_reader.json"))
        self._patches = [
            patch.object(rtrade, "RTRADE_TREND_FILTER_ENABLED", True),
            patch.object(rtrade, "RTRADE_TREND_FILTER_K", 2.0),
            patch.object(rtrade, "RTRADE_TREND_WINDOW_SEC", self.WINDOW),
            patch.object(cm, "CM_PUBLISHED_TREND_WINDOWS_SEC", [self.WINDOW]),
            patch.object(cm, "get_short_trend_manager", return_value=self.reader),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        super().tearDown()

    def _publish(self, step, *, age_sec=0.0):
        """Let a writer compute the window from ticks and publish it for the reader."""
        now_ms = int(time.time() * 1000)
        # One tick per second over the window; a small alternating wiggle is the noise.
        entries = [[now_ms - (899 - i) * 1000, 300.0 + step * i + (0.05 if i % 2 else -0.05)]
                   for i in range(900)]
        writer = self.cm.CacheShortTrendManager(
            ["TAOUSDC"], os.path.join(self.tmp, "trend_writer.json"), data_sync_mode="producer")
        writer._cache24_managers = {
            "TAOUSDC": types.SimpleNamespace(
                entries=entries,
                get_recent_entries=lambda _symbol, last_seconds: [
                    e for e in entries if e[0] >= (time.time() - last_seconds) * 1000
                ],
            )
        }
        published = writer._published_dynamic_windows("TAOUSDC")
        for dyn in published.values():
            dyn["ts"] -= age_sec
        with open(self.reader.filename, "w", encoding="utf-8") as fh:
            self.json.dump({"TAOUSDC": {"dynamic": published}}, fh)
        return published

    def test_reader_uses_the_published_window(self):
        published = self._publish(step=0.0)
        self.assertIn("900", published)
        self.assertIsNotNone(self.reader.get_short_trend_for_window("TAOUSDC", self.WINDOW))
        # A sideways market is exactly when the spread bot should run.
        self.assertFalse(rtrade._trend_too_strong("TAOUSDC"))

    def test_strong_trend_still_stands_aside(self):
        self._publish(step=0.2)
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))

    def test_stale_or_missing_publication_fails_closed(self):
        stale = self.cm.CacheShortTrendManager.TREND_STALE_SEC + 5
        self._publish(step=0.0, age_sec=stale)
        self.assertIsNone(self.reader.get_short_trend_for_window("TAOUSDC", self.WINDOW))
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))
        os.remove(self.reader.filename)
        self.assertTrue(rtrade._trend_too_strong("TAOUSDC"))

    def test_stale_in_memory_live_window_does_not_shadow_fresh_publication(self):
        stale_ts = time.time() - (self.cm.CacheShortTrendManager.TREND_STALE_SEC + 10.0)
        self.reader._persist_live_window(
            "TAOUSDC", "900",
            {
                "final_trend": 0,
                "growth_coefficient": 0.0,
                "slope_full": 0.0,
                "gradient_recent": 0.0,
                "epsilon": 0.05,
                "n_samples": 100,
                "window_seconds": 900.0,
                "sample_rate_sec": 1.0,
                "current_price": 270.0,
                "ts": stale_ts,
            },
        )
        self._publish(step=0.0, age_sec=0.0)
        dyn = self.reader.get_short_trend_for_window("TAOUSDC", self.WINDOW)
        self.assertIsNotNone(dyn)
        self.assertFalse(rtrade._trend_too_strong("TAOUSDC"))


class RTradeLiveTickStreamTest(TempDirTestCase):
    """Verify rtrade._init_live_price_tick_stream serves sub-millisecond RAM ticks via AF_UNIX and falls back to disk when RAM ages out."""

    def test_rtrade_live_tick_stream_prefers_ram_and_falls_back_to_disk(self):
        import tempfile
        import cacheprice as cp_mod

        disk_cp_path = os.path.join(self.tmp, "cache_current_price.json")
        now_ms = int(time.time() * 1000)
        with open(disk_cp_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": {"TAOUSDC": [[now_ms, 250.0]]},
                    "fetchtime": {"TAOUSDC": now_ms / 1000.0},
                },
                fh,
            )

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="mptr_rt_") as sock_dir:
            sock_path = os.path.join(sock_dir, "current_price_ticks.sock")
            producer = cm.CacheCurrentPriceManager(
                sync_ts=9999,
                symbols=["TAOUSDC"],
                filename=os.path.join(self.tmp, "prod_cp.json"),
                ws_manager=None,
                read_only=False,
                data_sync_mode=cm.CacheManagerInterface.MODE_PRODUCER,
            )
            reader = cm.CacheCurrentPriceManager(
                sync_ts=9999,
                symbols=["TAOUSDC"],
                filename=disk_cp_path,
                ws_manager=None,
                read_only=True,
                data_sync_mode=cm.CacheManagerInterface.MODE_DISK_READER,
            )
            try:
                self.assertEqual(producer.start_tick_fanout_server(sock_path), sock_path)
                live_ts_ms = int(time.time() * 1000)
                producer.push_price("TAOUSDC", 288.75, ts_ms=live_ts_ms)

                def _fake_get_cpm(**kw):
                    if kw.get("data_sync_mode") == "live_reader":
                        reader.data_sync_mode = cm.CacheManagerInterface.MODE_LIVE_READER
                    return reader

                orig_start = reader.start_local_tick_stream
                with (
                    patch.object(rtrade, "get_current_price_manager", side_effect=_fake_get_cpm),
                    patch.object(reader, "start_local_tick_stream", side_effect=lambda p=None: orig_start(sock_path)),
                ):
                    started = rtrade._init_live_price_tick_stream("TAOUSDC")
                self.assertIs(started, reader)
                self.assertEqual(reader.data_sync_mode, cm.CacheManagerInterface.MODE_LIVE_READER)

                deadline = time.monotonic() + 1.0
                while (
                    (obs := reader.get_cached_price_observation("TAOUSDC", 3.0)) is None
                    or abs(obs[1] - 288.75) > 1e-6
                ) and time.monotonic() < deadline:
                    time.sleep(0.005)

                # RAM tick (288.75) wins over disk file (250.0).
                px_ram = cp_mod.CacheCurrentPriceManager.read_ssot_current_price_candidates(
                    ("TAOUSDC",),
                    disk_cp_path,
                    max_age_sec=3.0,
                    use_memory_manager=True,
                    memory_manager=reader,
                )
                self.assertAlmostEqual(px_ram, 288.75, places=6)

                # Once 10s elapse with no new AF_UNIX ticks (> 3.0s max_age_sec), fallback reads fresh disk price (277.50).
                future_s = time.time() + 10.0
                fresh_disk_ms = int((future_s - 0.5) * 1000)
                with open(disk_cp_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "items": {"TAOUSDC": [[fresh_disk_ms, 277.50]]},
                            "fetchtime": {"TAOUSDC": fresh_disk_ms / 1000.0},
                        },
                        fh,
                    )
                with patch.object(cp_mod, "_now_time", return_value=future_s):
                    px_disk = cp_mod.CacheCurrentPriceManager.read_ssot_current_price_candidates(
                        ("TAOUSDC",),
                        disk_cp_path,
                        max_age_sec=3.0,
                        now_s=future_s,
                        use_memory_manager=True,
                        memory_manager=reader,
                    )
                self.assertAlmostEqual(px_disk, 277.50, places=6)
            finally:
                reader.shutdown(timeout=1.0)
                producer.shutdown(timeout=1.0)
