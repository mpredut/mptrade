"""Tests for offline/research/backtest_ranges.py and live config # BACKTEST annotations."""
import configparser
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from offline.research.backtest_ranges import scan_backtest_ranges  # noqa: E402

_INI_CONFIGS = [os.path.join(ROOT, "instruments.conf")]


class TestScanBacktestRanges(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="backtest_ranges_")
        self._tmp = os.path.join(self._tmpdir, "backtest_ranges.conf")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, content):
        with open(self._tmp, "w", encoding="utf-8") as f:
            f.write(content)

    def test_supported_assignment_styles(self):
        cases = (
            (
                "ini",
                "# BACKTEST: 5.0, 6.0, 7.0, 8.0, 9.0\nmt.gain = 7.0\n",
                {"mt.gain": ["5.0", "6.0", "7.0", "8.0", "9.0"]},
            ),
            (
                "env",
                "# BACKTEST: 3, 4, 5.1, 6.5, 8\nTRADEALL_SLOPE_EXTREME_THRESHOLD=5.1\n",
                {"TRADEALL_SLOPE_EXTREME_THRESHOLD": ["3", "4", "5.1", "6.5", "8"]},
            ),
        )
        for label, content, expected in cases:
            with self.subTest(style=label):
                self._write(content)
                self.assertEqual(scan_backtest_ranges(self._tmp), expected)

    def test_intervening_line_breaks_association(self):
        for label, separator in (("blank", "\n"), ("comment", "# another explanatory comment\n")):
            with self.subTest(separator=label):
                self._write("# BACKTEST: 1, 2, 3\n" + separator + "mt.gain = 7.0\n")
                self.assertEqual(scan_backtest_ranges(self._tmp), {})

    def test_multiple_annotations_in_same_file(self):
        self._write(
            "# BACKTEST: 5.0, 7.0, 9.0\n"
            "mt.gain = 7.0\n"
            "mt.lost = 3.3\n"
            "# BACKTEST: 2.3, 3.3, 4.3\n"
            "mt.lost2 = 3.3\n"
        )
        result = scan_backtest_ranges(self._tmp)
        self.assertEqual(result, {"mt.gain": ["5.0", "7.0", "9.0"], "mt.lost2": ["2.3", "3.3", "4.3"]})

    def test_ini_sections_disambiguate_same_key_name(self):
        """instruments.conf reuses mt.gain in every [NAME] section —
        without prefixing, the second section would silently overwrite the first."""
        self._write(
            "[BINANCE_BTC]\n"
            "# BACKTEST: 5.0, 7.0, 8.0, 9.0\n"
            "mt.gain = 7.0\n"
            "\n"
            "[BINANCE_TAO]\n"
            "# BACKTEST: 7.5, 9.2, 10.5, 12.0\n"
            "mt.gain = 9.2\n"
        )
        result = scan_backtest_ranges(self._tmp)
        self.assertEqual(result, {
            "BINANCE_BTC.mt.gain": ["5.0", "7.0", "8.0", "9.0"],
            "BINANCE_TAO.mt.gain": ["7.5", "9.2", "10.5", "12.0"],
        })

    def test_file_without_sections_keeps_bare_key(self):
        self._write(
            "# BACKTEST: 3, 4, 5\n"
            "SOME_KEY=4\n"
        )
        self.assertEqual(scan_backtest_ranges(self._tmp), {"SOME_KEY": ["3", "4", "5"]})

    def test_missing_or_unannotated_returns_empty(self):
        cases = (
            ("missing", os.path.join(self._tmpdir, "does_not_exist.conf"), None),
            ("unannotated", self._tmp, "mt.gain = 7.0\nmt.lost = 3.3\n"),
        )
        for label, path, content in cases:
            with self.subTest(case=label):
                if content is not None:
                    self._write(content)
                self.assertEqual(scan_backtest_ranges(path), {})


class TestBacktestAnnotations(unittest.TestCase):
    def _current_ini_value(self, path, full_key):
        section, key = full_key.split(".", 1)
        cp = configparser.ConfigParser()
        cp.read(path)
        return float(cp[section][key])

    def test_annotations_well_formed_and_live_in_range(self):
        seen_any = False
        for path in _INI_CONFIGS:
            ranges = scan_backtest_ranges(path)
            for full_key, grid in ranges.items():
                seen_any = True
                with self.subTest(config=os.path.basename(path), key=full_key):
                    self.assertGreaterEqual(
                        len(grid), 2, f"{full_key}: grid has < 2 values: {grid}"
                    )
                    try:
                        vals = [float(v) for v in grid]
                    except ValueError as e:
                        self.fail(f"{full_key}: grid has non-numeric values ({grid}): {e}")

                    current = self._current_ini_value(path, full_key)
                    lo, hi = min(vals), max(vals)
                    self.assertTrue(
                        lo <= current <= hi,
                        f"{full_key}: LIVE value {current} left grid interval [{lo}, {hi}] {grid} — stale grid?",
                    )
        self.assertTrue(seen_any, "no # BACKTEST annotation found — is scan or config broken?")


if __name__ == "__main__":
    unittest.main()
