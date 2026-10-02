#!/usr/bin/env python3
"""Compare two fixed spot profiles on the same finite cash account, offline only.

Example: python -m offline.runners.spot_profile_review --baseline bc40dd7:hyperliquid/config.env
  --candidate 1eaa0ef:hyperliquid/config.env --cash 1107 --output /tmp/spot-review.json
The account value is an explicit scenario, not a verified production balance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from dotenv import dotenv_values
from kraken import replay
from strategies.spot_engine import StratParams
from offline.backtests.datasets import dataset_sha256, load_dataset, validate_dataset
from offline.backtests.evaluation import evaluate_segment, to_ohlc
from offline.backtests.financial_benchmark import (
    BenchmarkScenario, aggregate_financial_windows, default_scenarios,
)
from offline.backtests.execution import FeeModel
from offline.backtests.promotion import evaluate_dual_promotion
from offline.backtests.walk_forward import walk_forward_splits
from offline.runners.kraken_financial_benchmark import DEFAULT_DATASET, DEFAULT_MANIFEST, ROOT


def revision_params(spec: str) -> StratParams:
    text = subprocess.run(["git", "show", spec], check=True, capture_output=True,
                          text=True).stdout
    env = dict(dotenv_values(stream=io.StringIO(text), interpolate=False))
    with patch.dict(os.environ, env, clear=True):
        return StratParams.from_env()


def review(baseline: str, candidate: str, cash: float) -> dict:
    records = validate_dataset(load_dataset(DEFAULT_DATASET), interval_minutes=240)
    digest = dataset_sha256(records)
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    if digest != manifest["datasets"]["240"]["sha256"]:
        raise ValueError("dataset does not match the frozen manifest")
    before, after = revision_params(baseline), revision_params(candidate)
    profiles = {"previous": before, "current": after,
                "current_with_previous_stop": replace(after, stop_loss_pct=before.stop_loss_pct),
                "current_with_overlay": replace(after, trend_overlay=True,
                                                trend_topup=after.effective_entry_amount())}
    central, stress = default_scenarios()
    scenarios = (central, stress, BenchmarkScenario(
        "published_spot_fee_sensitivity", "Published base spot fees; fills remain uncalibrated",
        FeeModel(limit_fee_pct=0.04, market_fee_pct=0.07), central.execution))
    folds = walk_forward_splits(len(records), train_size=720, validation_size=180,
                                test_size=90, step_size=90)
    reports = {}
    with patch.object(replay._strat, "log", lambda *_args: None):
        for name, params in profiles.items():
            report = {"strategy_params": asdict(params), "initial_capital_usd": cash,
                      "dataset": {"sha256": digest, "bars": len(records)},
                      "walk_forward": {"interval_minutes": 240, "train": 720,
                                       "validation": 180, "test": 90, "step": 90, "warmup": 40},
                      "scenarios": {}}
            for scenario in scenarios:
                def run(ohlc, warmup=()):
                    return replay.run_replay(ohlc, params, bar_minutes=240, warmup_ohlc=warmup,
                                             execution=scenario.execution, fee_model=scenario.fees,
                                             initial_cash=cash)
                windows = []
                for index, fold in enumerate(folds, 1):
                    segment = evaluate_segment(
                        records[fold.test], lambda bars, warm, _context: run(bars, warm),
                        warmup_records=records[fold.test.start - 40:fold.test.start])
                    metrics = segment["metrics"]
                    windows.append({"key": f"240m/fold-{index:02d}", "bars": segment["bars"],
                                    "start_utc": segment["start_utc"], "end_utc": segment["end_utc"],
                                    "buy_hold_return_pct": segment["buy_hold_return_pct"],
                                    **{k: metrics[k] for k in ("return_pct", "max_drawdown_pct", "cycles", "fills")},
                                    "metrics": metrics})
                report["scenarios"][scenario.name] = {
                    "assumptions": scenario.as_dict(), "windows": windows,
                    "aggregate": aggregate_financial_windows(windows, initial_capital=cash),
                    "continuous": run(to_ohlc(records[folds[0].test.start:folds[-1].test.stop]),
                                      to_ohlc(records[folds[0].test.start - 40:folds[0].test.start])),
                }
            reports[name] = report
    return {"baseline_config": baseline, "candidate_config": candidate,
            "code": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("strategies/spot_dca.py", "kraken/replay.py", "offline/runners/spot_profile_review.py")},
            "cash_scenario": cash, "profiles": reports,
            "gates": {"current_vs_previous": evaluate_dual_promotion(reports["previous"], reports["current"]),
                      "stop_change_only": evaluate_dual_promotion(reports["current_with_previous_stop"], reports["current"]),
                      "overlay_vs_current": evaluate_dual_promotion(reports["current"], reports["current_with_overlay"])},
            "limitations": [
                "Retrospective fixed-profile comparison on reused data, not an untouched holdout or live proof.",
                "Same corrected engine for all profiles; parameters, not historical engine bugs, are compared.",
                "Cash and pending BUY fee reserves are enforced; live fee currency, precision and minimums are not calibrated.",
                "OHLC decisions once per 4h bar; live checks run more often. Gap exits can exceed the STOP threshold.",
                "Fold state/cash resets; the separate continuous path does not reset between folds.",
                "Published spot fee sensitivity: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees; account discounts unknown.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cash", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.baseline, args.candidate, args.cash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, profile in result["profiles"].items():
        for scenario, values in profile["scenarios"].items():
            agg, continuous = values["aggregate"], values["continuous"]
            print(name, scenario, json.dumps({
                "mean_return_pct": agg["mean_return_pct"], "worst_dd_pct": agg["worst_max_drawdown_pct"],
                "exposure_pct": agg["mean_exposure_pct"], "continuous_return_pct": continuous["return_pct"],
                "continuous_dd_pct": continuous["max_drawdown_pct"],
                "refused_buys": sum(w["metrics"]["funding"]["refused_buys"] for w in values["windows"])}))
    print("Promotion:", {name: gate["promote"] for name, gate in result["gates"].items()})
    print("Report:", args.output)


if __name__ == "__main__":
    main()
