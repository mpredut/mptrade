import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import rtrade  # noqa: E402


class RTradeHeartbeatTest(unittest.TestCase):
    def setUp(self):
        self._last = rtrade._rtrade_heartbeat_last

    def tearDown(self):
        rtrade._rtrade_heartbeat_last = self._last

    def test_heartbeat_is_written_and_throttled_by_loop_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rtrade.heartbeat")
            with (
                patch.object(rtrade, "_RTRADE_HEARTBEAT_PATH", path),
                patch.object(rtrade, "_RTRADE_HEARTBEAT_INTERVAL_SEC", 30.0),
                patch.object(rtrade.time, "monotonic", side_effect=(100.0, 110.0, 131.0)),
            ):
                rtrade._rtrade_heartbeat_last = float("-inf")
                rtrade._touch_rtrade_heartbeat()
                self.assertTrue(os.path.exists(path))
                self.assertEqual(rtrade._rtrade_heartbeat_last, 100.0)

                rtrade._touch_rtrade_heartbeat()
                self.assertEqual(rtrade._rtrade_heartbeat_last, 100.0)

                rtrade._touch_rtrade_heartbeat()
                self.assertEqual(rtrade._rtrade_heartbeat_last, 131.0)

    def test_process_manifest_uses_dedicated_heartbeat(self):
        line = next(
            row for row in (ROOT / "procs.conf").read_text(encoding="utf-8").splitlines()
            if row.startswith("rtrade.py|")
        )
        parts = line.split("|")
        self.assertEqual(parts[0], "rtrade.py")
        self.assertEqual(parts[1], "$ROOT")
        self.assertEqual(parts[3], "rtrade")
        hb_targets = [f.strip() for f in parts[4].split(",")]
        self.assertIn("cachedb/rtrade.heartbeat", hb_targets)
        self.assertEqual(parts[5], "180")
        self.assertEqual(parts[6], "fleet")

    def test_rtrade_imports_sys_and_defines_recovery_timeout(self):
        self.assertTrue(hasattr(rtrade, "sys"), "rtrade must import sys for sys.exit")
        self.assertTrue(callable(rtrade.sys.exit))
        self.assertTrue(hasattr(rtrade, "_RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC"))
        self.assertLess(
            rtrade._RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC, 180.0,
            "Startup recovery timeout must be strictly less than orchestrator 180s threshold"
        )

    def test_venue_pair_recovery_blocked_management(self):
        venue = rtrade._LivePairVenue.__new__(rtrade._LivePairVenue)
        venue._recovery_blocked_pairs = set()
        venue.recovery_blocked = False
        venue._set_pair_recovery_blocked("pair-1", True)
        self.assertTrue(venue.recovery_blocked)
        self.assertTrue(venue.pair_recovery_blocked("pair-1"))
        venue._set_pair_recovery_blocked("pair-1", False)
        self.assertFalse(venue.recovery_blocked)
        self.assertFalse(venue.pair_recovery_blocked("pair-1"))

    def test_coordinator_loop_exits_on_venue_recovery_blocked_timeout(self):
        from unittest.mock import MagicMock
        bot = rtrade.TradingBot.__new__(rtrade.TradingBot)
        bot.symbol = "TAOUSDC"
        bot.DEFAULT_ADJUSTMENT_PERCENT = 0.005
        mock_venue = MagicMock()
        mock_venue.recovery_blocked = True
        mock_venue.executor.open_orders.return_value = []
        mock_pair_store = MagicMock()
        mock_pair_store.active.return_value = []

        start_time = 1000.0
        timeout = rtrade._RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC
        monotonic_times = [
            start_time,                  # loop enter: recovery_blocked_since init
            start_time + 1.0,            # loop tick 1
            start_time + timeout + 5.0,  # loop tick 2: timeout exceeded -> sys.exit(1)
        ]
        with (
            patch.object(rtrade.time, "monotonic", side_effect=monotonic_times),
            patch.object(rtrade.time, "sleep", return_value=None),
            patch.object(rtrade.sys, "exit") as mock_exit,
            patch.object(rtrade, "_LivePairVenue", return_value=mock_venue),
            patch.object(rtrade, "RTradePairStore", return_value=mock_pair_store),
        ):
            mock_exit.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                bot._run_coordinator_forever()
            mock_exit.assert_called_once_with(1)

    def test_startup_recovery_terminal_record_clears_venue_blockage(self):
        from unittest.mock import MagicMock
        bot = rtrade.TradingBot.__new__(rtrade.TradingBot)
        bot.symbol = "TAOUSDC"
        bot.DEFAULT_ADJUSTMENT_PERCENT = 0.005
        mock_venue = MagicMock()
        mock_venue.recovery_blocked = False
        mock_pair_store = MagicMock()
        record = {
            "pair_id": "test_pair_1",
            "qty": 1.0,
            "start_side": "SELL",
            "state": {"phase": "quoting", "tickets": []},
            "intents": {
                "limit:SELL": {
                    "recovery_state": "absence_confirmed_no_reuse"
                }
            }
        }
        mock_pair_store.active.return_value = [record]
        mock_venue.merge_checkpoint_intents.return_value = record["state"]
        mock_venue.executor.open_orders.return_value = []

        with (
            patch.object(rtrade, "_LivePairVenue", return_value=mock_venue),
            patch.object(rtrade, "RTradePairStore", return_value=mock_pair_store),
            patch.object(rtrade.time, "sleep", side_effect=StopIteration),
        ):
            try:
                bot._run_coordinator_forever()
            except StopIteration:
                pass
            mock_pair_store.checkpoint.assert_any_call("test_pair_1", record["state"], terminal=True)
            mock_venue._set_pair_recovery_blocked.assert_called_with("test_pair_1", False)


if __name__ == "__main__":
    unittest.main()

