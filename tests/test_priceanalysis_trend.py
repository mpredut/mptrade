import json, os, sys, tempfile, time, unittest

os.environ.setdefault("MPLBACKEND", "Agg")   # No GUI backend when importing matplotlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import patch
import priceAnalysis as pa


def _uniform_series(days=20, step_sec=300.0, slope_per_day=1.0, base=100.0, end=None):
    end = end if end is not None else time.time()
    ts = np.arange(end - days * 86400, end, step_sec)
    pr = base + (ts - ts[0]) / 86400.0 * slope_per_day
    return ts, pr


class TestTimeBasedTrend(unittest.TestCase):
    def test_trend_detection_scenarios(self):
        with self.subTest(msg="uniform_uptrend"):
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            self.assertIsNotNone(r)
            self.assertEqual(r["direction"], "up")

        with self.subTest(msg="uniform_downtrend"):
            ts, pr = _uniform_series(days=20, slope_per_day=-1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            self.assertEqual(r["direction"], "down")

        with self.subTest(msg="window_is_time_based_not_point_count"):
            # DIFFERENT density (1min vs 10min) must give the same direction:
            # proof that the window is measured in time, not in number of points.
            ts_a, pr_a = _uniform_series(days=15, step_sec=60.0, slope_per_day=2.0)
            ts_b, pr_b = _uniform_series(days=15, step_sec=600.0, slope_per_day=2.0)
            ra = pa.detect_long_term_trend(ts_a, pr_a, window_hours=12, step_hours=6)
            rb = pa.detect_long_term_trend(ts_b, pr_b, window_hours=12, step_hours=6)
            self.assertEqual(ra["direction"], rb["direction"])
            # the duration in days is comparable although the density differs 10x
            self.assertAlmostEqual(ra["duration_seconds"] / 86400,
                                   rb["duration_seconds"] / 86400, delta=1.0)

        with self.subTest(msg="gap_stops_trend"):
            # an UP trend with a 10-day gap in the middle -> the trend stops at the gap,
            # it does not claim continuity across the gap.
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            keep = ~((ts > ts[0] + 5 * 86400) & (ts < ts[0] + 15 * 86400))
            r = pa.detect_long_term_trend(ts[keep], pr[keep], window_hours=16,
                                          step_hours=8, noise_tolerance=2)
            self.assertIsNotNone(r)
            # start after the gap (with the padding tolerance (noise+1)*window)
            self.assertGreaterEqual(r["start_timestamp"],
                                    ts[0] + 15 * 86400 - (2 + 1) * 16 * 3600 - 1)

        with self.subTest(msg="insufficient_recent_data_returns_none"):
            # only 2 recent points -> the recent window is below min_points -> None
            ts = np.array([time.time() - 100, time.time() - 50])
            pr = np.array([100.0, 101.0])
            self.assertIsNone(pa.detect_long_term_trend(ts, pr, window_hours=16))

        with self.subTest(msg="blocks_are_index_pairs"):
            ts, pr = _uniform_series(days=20, slope_per_day=1.0)
            r = pa.detect_long_term_trend(ts, pr, window_hours=16, step_hours=8)
            for lo, hi in r["blocks"]:
                self.assertTrue(0 <= lo < hi <= len(ts))


class TestTrendPersistence(unittest.TestCase):
    def test_persistence_behaviors(self):
        with self.subTest(msg="write_all_trends_uses_the_shared_atomic_json_writer"):
            payload = {
                "BTCUSDC": {
                    "direction": "up",
                    "start_timestamp": 1,
                    "duration_seconds": 3600,
                    "estimated_future_hours": 2,
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "trends.json")
                with patch.object(pa, "atomic_write_json") as writer:
                    self.assertEqual(pa.write_all_trends(payload, filename), payload)
    
            writer.assert_called_once_with(filename, payload, indent=2)

        with self.subTest(msg="failed_atomic_write_preserves_the_previous_generation"):
            previous = {"BTCUSDC": {"direction": "down"}}
            payload = {
                "BTCUSDC": {
                    "direction": "up",
                    "start_timestamp": 1,
                    "duration_seconds": 3600,
                    "estimated_future_hours": 2,
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "trends.json")
                with open(filename, "w", encoding="utf-8") as handle:
                    json.dump(previous, handle)
                with patch.object(pa, "atomic_write_json", side_effect=OSError("disk full")):
                    self.assertEqual(pa.write_all_trends(payload, filename), payload)
                with open(filename, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), previous)


class TestBoundedPlotIndices(unittest.TestCase):
    def test_bounded_indices_scenarios(self):
        with self.subTest(msg="short_series_is_unchanged"):
            np.testing.assert_array_equal(
                pa._bounded_plot_indices(4, max_points=5),
                np.array([0, 1, 2, 3]),
            )

        with self.subTest(msg="long_series_is_bounded_and_keeps_endpoints"):
            indices = pa._bounded_plot_indices(43_200, max_points=5_000)
            self.assertEqual(len(indices), 5_000)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 43_199)
            self.assertTrue(np.all(np.diff(indices) > 0))

        with self.subTest(msg="empty_series"):
            self.assertEqual(pa._bounded_plot_indices(0, max_points=5_000).size, 0)


if __name__ == "__main__":
    unittest.main()
