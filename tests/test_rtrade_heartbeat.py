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


if __name__ == "__main__":
    unittest.main()
