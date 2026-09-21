"""watchdogfor_anomaly: logurile DEV/backtest (backtest_cycle.log, refresh_dev.log,
backtest*.log) are excluded from the scan — tracebacks there come from the test
machine (backtest pilot on runner.py), NOT LIVE fleet problems. Without the
exclusion the anomaly watchdog falsely alerted about dev failures."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_tools"))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import watchdogfor_anomaly as w


class DevLogExclusionTest(unittest.TestCase):
    def test_dev_and_live_log_classification(self):
        cases = (
            (True, "logs/backtest_cycle.log"),
            (True, "logs/refresh_dev.log"),
            (True, "logs/trigger_backtest_dev.log"),
            (True, "logs/backtest_pilot.log"),
            # Ad-hoc `python -c` / `python -m` one-liner logs (diagnostics/probes,
            # tagged [unknown]) are not supervised bots — excluded from fleet alerts.
            (True, "logger/-c_2026-09-05.log"),
            (True, "logger/-m_2026-09-05.log"),
            (False, "logs/monitortrades.log"),
            (False, "logs/tradeall.log"),
            (False, "kraken/kraken_bot.log"),
            (False, "hyperliquid/dn_bot.log"),
        )
        for expected, path in cases:
            with self.subTest(path=path):
                self.assertEqual(w._is_dev_log(path), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
