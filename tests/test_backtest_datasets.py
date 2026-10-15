import json
import tempfile
import unittest
from pathlib import Path

from offline.backtests.datasets import (
    align_previous_values,
    dataset_metadata,
    drop_incomplete_last_bar,
    load_dataset,
    merge_datasets,
    save_dataset,
    validate_dataset,
)
from offline.backtests.execution import (
    ExecutionModel,
    FeeModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.hype_candidates import (
    financial_priority_candidates,
    hype_240_candidates,
)


def _rows():
    return [
        {"timestamp": 0, "open": 10, "high": 12, "low": 9, "close": 11},
        {"timestamp": 3600, "open": 11, "high": 13, "low": 10, "close": 12},
    ]


class OhlcDatasetContractTest(unittest.TestCase):
    def test_drops_only_the_incomplete_intraday_bar(self):
        rows = _rows()
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=60, now_timestamp=7199,
            ),
            rows[:1],
        )
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=60, now_timestamp=7200,
            ),
            rows,
        )

    def test_drops_multiple_incomplete_yahoo_tail_rows(self):
        rows = _rows() + [
            {"timestamp": 3900, "open": 12, "high": 13, "low": 11, "close": 12},
        ]
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=60, now_timestamp=3700,
            ),
            rows[:1],
        )

    def test_daily_bar_uses_regular_session_end(self):
        rows = [{"timestamp": 100, "open": 10, "high": 12, "low": 9, "close": 11}]
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=1440, now_timestamp=150,
                regular_session_start=100, regular_session_end=200,
            ),
            [],
        )
        self.assertEqual(
            drop_incomplete_last_bar(
                rows, interval_minutes=1440, now_timestamp=200,
                regular_session_start=100, regular_session_end=200,
            ),
            rows,
        )

    def test_asof_alignment_never_uses_a_future_fx_value(self):
        fx = [
            {"timestamp": 10, "close": 0.20},
            {"timestamp": 20, "close": 0.25},
        ]
        self.assertEqual(
            align_previous_values([10, 19, 20, 30], fx),
            [0.20, 0.20, 0.25, 0.25],
        )
        with self.assertRaisesRegex(ValueError, "extend the FX dataset"):
            align_previous_values([9], fx)

    def test_round_trip_is_canonical_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            save_dataset(_rows(), path)
            loaded = load_dataset(path)
            self.assertEqual(loaded, validate_dataset(_rows(), interval_minutes=60))
            metadata = dataset_metadata(loaded, interval_minutes=60)
            self.assertEqual(metadata["bars"], 2)
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_merge_overlapping_windows_preserves_history_and_newer_value(self):
        older = _rows()
        newer = [
            {**_rows()[1], "close": 12.5, "high": 13.5},
            {"timestamp": 7200, "open": 12.5, "high": 14, "low": 12, "close": 13},
        ]
        merged = merge_datasets(older, newer)

        self.assertEqual([row["timestamp"] for row in merged], [0, 3600, 7200])
        self.assertEqual(merged[1]["close"], 12.5)

    def test_rejects_gap_duplicate_and_invalid_ohlc(self):
        with self.assertRaisesRegex(ValueError, "cadence"):
            validate_dataset([_rows()[0], {**_rows()[1], "timestamp": 7200}], interval_minutes=60)
        with self.assertRaisesRegex(ValueError, "duplicat"):
            validate_dataset([_rows()[0], {**_rows()[1], "timestamp": 0}])
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_dataset([{"timestamp": 0, "open": 10, "high": 9, "low": 8, "close": 10}])

    def test_versioned_hype_manifest_matches_frozen_files(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "offline/research/hype_dataset"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for interval, expected in manifest["datasets"].items():
            records = load_dataset(directory / expected["file"])
            actual = dataset_metadata(records, interval_minutes=int(interval))
            for key in ("sha256", "bars", "start_utc", "end_utc"):
                self.assertEqual(actual[key], expected[key])

    def test_versioned_t212_manifest_matches_frozen_files(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "offline/research/t212_dataset"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for expected in manifest["datasets"].values():
            records = load_dataset(directory / expected["file"])
            actual = dataset_metadata(records)
            for key in ("sha256", "bars", "start_utc", "end_utc"):
                self.assertEqual(actual[key], expected[key])


class ExecutionModelTest(unittest.TestCase):
    def test_fee_model_separates_limit_and_market_costs(self):
        fees = FeeModel(limit_fee_pct=0.16, market_fee_pct=0.26)
        self.assertEqual(fees.rate_pct(market=False), 0.16)
        self.assertEqual(fees.rate_pct(market=True), 0.26)

        for invalid in (-0.01, float("nan"), 100.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    FeeModel(limit_fee_pct=invalid, market_fee_pct=0.26)

    def test_spread_changes_limit_touch_and_market_slippage_is_adverse(self):
        plain = ExecutionModel()
        stressed = ExecutionModel(spread_bps=20, market_slippage_bps=30)
        self.assertTrue(plain.limit_touched("buy", high=101, low=99.95, limit=100))
        self.assertFalse(stressed.limit_touched("buy", high=101, low=99.95, limit=100))
        self.assertAlmostEqual(stressed.market_price("buy", 100), 100.4)
        self.assertAlmostEqual(stressed.market_price("sell", 100), 99.6)

    def test_partial_fill_uses_original_quantity_and_prorates_amount(self):
        order = {"side": "BUY", "kind": "DCA", "qty": 10.0, "amount": 100.0}
        first, qty1, done1 = split_order_fill(
            order, quantity_key="qty", amount_key="amount", ratio=0.4,
        )
        second, qty2, done2 = split_order_fill(
            order, quantity_key="qty", amount_key="amount", ratio=0.4,
        )
        third, qty3, done3 = split_order_fill(
            order, quantity_key="qty", amount_key="amount", ratio=0.4,
        )
        self.assertEqual((qty1, qty2, qty3), (4.0, 4.0, 2.0))
        self.assertEqual((done1, done2, done3), (False, False, True))
        self.assertAlmostEqual(first["amount"] + second["amount"] + third["amount"], 100)
        self.assertEqual(first["kind"], "DCA")
        self.assertEqual(second["kind"], "DCA_PARTIAL")

    def test_worst_case_selects_lower_return(self):
        result = choose_intrabar_scenario(
            ExecutionModel(intrabar_policy="worst_case"),
            lambda model: {
                "return_pct": 2.0 if model.intrabar_policy == "sell_first" else -1.0,
                "max_drawdown_pct": 3.0, "fills": 1, "ambiguous_bars": 1,
            },
        )
        self.assertEqual(result["intrabar_policy_selected"], "buy_first")
        self.assertEqual(result["return_pct"], -1.0)
        self.assertEqual(set(result["intrabar_scenarios"]), {"buy_first", "sell_first"})

    def test_invalid_assumptions_are_rejected(self):
        with self.assertRaises(ValueError):
            ExecutionModel(partial_fill_ratio=0)
        with self.assertRaises(ValueError):
            ExecutionModel(spread_bps=-1)


class HypeCandidatesTest(unittest.TestCase):
    def test_priority_set_is_unique_fixed_and_starts_with_live(self):
        candidates = financial_priority_candidates()
        names = [candidate.name for candidate in candidates]

        self.assertEqual(names[0], "live")
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            names,
            [
                "live", "reentry4", "tp4", "dca15", "dca_progressive025", "dca_vol_m1",
                "A_trail", "B_dcabrake", "overlay650t8",
                "trail_profit_floor_sl18", "trail_profit_floor_sl125",
            ],
        )

    def test_tp4_and_dca15_are_one_factor_candidates(self):
        by_name = {
            candidate.name: candidate
            for candidate in financial_priority_candidates()
        }

        self.assertEqual(by_name["tp4"].overrides, {"takeprofit_pct": 4.0})
        self.assertEqual(by_name["reentry4"].overrides, {"reentry_drop_pct": 4.0})
        self.assertEqual(by_name["dca15"].overrides, {"dca_drop_pct": 1.5})
        self.assertEqual(
            by_name["dca_progressive025"].overrides,
            {"dca_spacing_growth_pct": 0.25},
        )
        self.assertEqual(
            by_name["dca_vol_m1"].overrides,
            {
                "dca_vol_scale_k": -1.0,
                "dca_vol_ref": 2.0,
                "dca_vol_interval": 240,
            },
        )
        self.assertEqual(
            by_name["trail_profit_floor_sl18"].overrides,
            {"tp_trail_profit_floor_pct": 1.0, "stop_loss_pct": 18.0},
        )
        self.assertEqual(
            by_name["trail_profit_floor_sl125"].overrides,
            {"tp_trail_profit_floor_pct": 1.0},
        )

    def test_walk_forward_registry_keeps_historical_report_names(self):
        self.assertEqual(
            [candidate.name for candidate in hype_240_candidates()],
            [
                "live", "overlay_orig", "overlay650t8", "A_adaptive_trail",
                "B_dca_brake", "tp_4", "dca_drop_1_5",
            ],
        )


if __name__ == "__main__":
    unittest.main()
