#!/usr/bin/env python3
"""HYPE financial benchmark: fixed configuration, central + stress, OOS TEST only."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
KRAKEN_DIR = ROOT / "kraken"
sys.path.insert(0, str(KRAKEN_DIR))
sys.path.insert(0, str(ROOT))

from kraken_common import load_dotenv  # noqa: E402
import replay as kraken_replay  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402
from offline.backtests.datasets import (  # noqa: E402
    dataset_sha256,
    load_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import evaluate_segment, iso_utc  # noqa: E402
from offline.backtests.financial_benchmark import (  # noqa: E402
    aggregate_financial_windows,
    default_scenarios,
)
from offline.backtests.promotion import evaluate_dual_promotion  # noqa: E402
from offline.backtests.walk_forward import walk_forward_splits  # noqa: E402


DEFAULT_DATASET = (
    ROOT / "offline" / "research" / "hype_dataset" / "HYPEUSDC_240m_hlspot.csv"
)
DEFAULT_MANIFEST = DEFAULT_DATASET.parent / "manifest.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        dirty = bool(status.strip())
        return {"commit": commit, "worktree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "worktree_dirty": None}


def _load_expected_dataset(manifest_path: Path, interval: int) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = (manifest.get("datasets") or {}).get(str(interval))
    if not expected:
        raise ValueError(f"the manifest does not declare the {interval}m dataset")
    return {"manifest": manifest, "expected": expected}


def strat_params_from_dict(values: dict) -> StratParams:
    """Build StratParams from a stored report, dropping fields that no longer exist
    on the dataclass. Versioned baselines can carry parameters removed in later
    refactors (e.g. trend_confirm_bars); ignoring them lets an old artefact still
    reproduce against the current engine instead of raising a TypeError."""
    valid = {f.name for f in dataclasses.fields(StratParams)}
    dropped = sorted(set(values) - valid)
    if dropped:
        print(f"[params] ignoring parameters removed from StratParams: {dropped}")
    return StratParams(**{k: v for k, v in values.items() if k in valid})


def _load_params(args) -> StratParams:
    load_dotenv(args.env_file)
    load_dotenv(args.config_file)
    if not args.params_report:
        return StratParams.from_env()
    with args.params_report.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("strategy_params", payload)
    if not isinstance(values, dict):
        raise ValueError("--params-report must contain strategy_params or an object")
    return strat_params_from_dict(values)


def _scenario_windows(records: list[dict], params: StratParams, scenario, *,
                      interval: int, train: int, validation: int, test: int,
                      step: int, warmup: int) -> list[dict]:
    folds = walk_forward_splits(
        len(records), train_size=train, validation_size=validation,
        test_size=test, step_size=step,
    )
    if not folds:
        raise ValueError("the dataset is too small for the walk-forward schema")

    original_log = kraken_replay._strat.log
    kraken_replay._strat.log = lambda *_args, **_kwargs: None
    windows = []
    try:
        for fold_index, fold in enumerate(folds, start=1):
            test_records = records[fold.test]
            warmup_records = records[max(0, fold.test.start - warmup):fold.test.start]
            segment = evaluate_segment(
                test_records,
                lambda ohlc, warmup_ohlc, _context: kraken_replay.run_replay(
                    ohlc, params, bar_minutes=interval, warmup_ohlc=warmup_ohlc,
                    execution=scenario.execution, fee_model=scenario.fees,
                ),
                warmup_records=warmup_records,
            )
            metrics = segment["metrics"]
            windows.append({
                "key": f"{interval}m/fold-{fold_index:02d}",
                "fold": fold_index,
                "start_utc": segment["start_utc"],
                "end_utc": segment["end_utc"],
                "bars": segment["bars"],
                "return_pct": metrics["return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "buy_hold_return_pct": segment["buy_hold_return_pct"],
                "cycles": metrics["cycles"],
                "fills": metrics["fills"],
                "metrics": metrics,
            })
    finally:
        kraken_replay._strat.log = original_log
    return windows


def build_report(
    args,
    *,
    params: StratParams | None = None,
    candidate_name: str | None = None,
) -> dict:
    interval = 240
    params = params or _load_params(args)
    records = validate_dataset(load_dataset(args.dataset), interval_minutes=interval)
    digest = dataset_sha256(records)
    manifest_info = _load_expected_dataset(args.manifest, interval)
    expected = manifest_info["expected"]
    if digest != expected.get("sha256"):
        raise ValueError(
            f"hash dataset diferit: {digest} != {expected.get('sha256')}"
        )
    if len(records) != int(expected.get("bars", -1)):
        raise ValueError(f"different bar count: {len(records)} != {expected.get('bars')}")

    scenarios = {}
    for scenario in default_scenarios():
        windows = _scenario_windows(
            records, params, scenario, interval=interval,
            train=args.train, validation=args.validation, test=args.test,
            step=args.step, warmup=args.warmup,
        )
        scenarios[scenario.name] = {
            "assumptions": scenario.as_dict(),
            "aggregate": aggregate_financial_windows(
                windows, initial_capital=params.effective_max_budget(),
                regime_threshold_pct=args.regime_threshold,
            ),
            "windows": windows,
        }

    report = {
        "schema_version": 1,
        "benchmark": "HYPE base-v2 financial OOS",
        "candidate_name": candidate_name or args.name,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code": _git_state(),
        "strategy_params": dataclasses.asdict(params),
        "initial_capital_usd": params.effective_max_budget(),
        "dataset": {
            "file": _display_path(args.dataset),
            "manifest": _display_path(args.manifest),
            "sha256": digest,
            "bars": len(records),
            "start_utc": iso_utc(int(records[0]["timestamp"])),
            "end_utc": iso_utc(int(records[-1]["timestamp"])),
            "venue": manifest_info["manifest"].get("source", {}).get("venue"),
            "market": manifest_info["manifest"].get("source", {}).get("market"),
            "role": "cross-venue price-path proxy; not Kraken execution",
        },
        "walk_forward": {
            "interval_minutes": interval,
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
            "step": args.step,
            "warmup": args.warmup,
            "state": "reset for every TEST window",
            "parameter_selection": "none; fixed profile",
            "shuffle": False,
        },
        "scenarios": scenarios,
        "limitations": [
            "central and stress assumptions are provisional, not calibrated from Kraken fills",
            "price path is Hyperliquid HYPE/USDC, not Kraken HYPE/USD execution",
            "OHLC cannot reconstruct queue position, latency or tick path",
            "every TEST window resets state and capital; sum_reset is not compounded live equity",
        ],
    }
    compare_to = getattr(args, "compare_to", None)
    if compare_to:
        with compare_to.open(encoding="utf-8") as handle:
            baseline = json.load(handle)
        report["promotion_gate"] = evaluate_dual_promotion(baseline, report)
    return report


def _format(value, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"


def markdown_report(report: dict) -> str:
    lines = [
        "# HYPE financial benchmark",
        "",
        f"Candidate: `{report['candidate_name']}`",
        "",
        "This is a reproducible OOS benchmark, not a promise of profit.",
        "The dataset is a Hyperliquid proxy; the central/stress costs are still uncalibrated.",
        "",
        "| Scenario | Mean/fold % | Mean USD/fold | Sum reset USD | Worst % | Worst DD % | Buy&hold mean % | CVaR 95% | Exposure % | Positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scenario in report["scenarios"].items():
        aggregate = scenario["aggregate"]
        lines.append(
            f"| `{name}` | {_format(aggregate['mean_return_pct'], signed=True)} | "
            f"{_format(aggregate['mean_net_pnl_usd_per_window'], signed=True)} | "
            f"{_format(aggregate['sum_reset_net_pnl_usd'], signed=True)} | "
            f"{_format(aggregate['worst_return_pct'], signed=True)} | "
            f"{_format(aggregate['worst_max_drawdown_pct'])} | "
            f"{_format(aggregate['mean_buy_hold_return_pct'], signed=True)} | "
            f"{_format(aggregate['worst_cvar_95_pct'], signed=True)} | "
            f"{_format(aggregate['mean_exposure_pct'])} | "
            f"{aggregate['positive_windows']}/{aggregate['window_count']} |"
        )
    lines.extend(["", "## Regimes", ""])
    for name, scenario in report["scenarios"].items():
        lines.extend([
            f"### {name}", "",
            "| Regime | Windows | Strategy mean % | Buy & hold mean % | Worst % |",
            "|---|---:|---:|---:|---:|",
        ])
        for regime, values in scenario["aggregate"]["regimes"].items():
            lines.append(
                f"| {regime} | {values['windows']} | "
                f"{_format(values['mean_return_pct'], signed=True)} | "
                f"{_format(values['mean_buy_hold_return_pct'], signed=True)} | "
                f"{_format(values['worst_return_pct'], signed=True)} |"
            )
        lines.append("")
    if "promotion_gate" in report:
        paths = "/".join(report["promotion_gate"]["promotion_paths"])
        verdict = f"PROMOTE {paths}" if paths else "DO NOT PROMOTE"
        lines.extend(["## Promotion gate", "", f"Verdict: **{verdict}**", ""])
    lines.extend([
        "## Interpretation", "",
        "- `Mean USD/fold` uses the same initial budget in every TEST window.",
        "- `Sum reset USD` adds up the folds; it is not compounded equity and does not carry the position across windows.",
        f"- The {next(iter(report['scenarios'].values()))['aggregate']['total_test_bars']} TEST bars mean "
        f"{next(iter(report['scenarios'].values()))['aggregate']['total_test_bars'] * report['walk_forward']['interval_minutes'] / 1440:.0f} days without overlap.",
        "- Final calibration requires real Kraken distributions for spread, slippage and fee tier.",
        "",
    ])
    return "\n".join(lines)


def _projection(report: dict) -> dict:
    strategy_params = dict(report.get("strategy_params") or {})
    # The new fields, off by default, stay compatible with older baseline artefacts;
    # normalisation avoids regenerating them when the decisions are identical.
    strategy_params.setdefault("dca_spacing_growth_pct", 0.0)
    strategy_params.setdefault("dca_vol_scale_k", 0.0)
    strategy_params.setdefault("dca_vol_ref", 2.0)
    strategy_params.setdefault("dca_vol_interval", 240)
    strategy_params.setdefault("tp_trail_profit_floor_pct", 0.0)
    strategy_params.setdefault("tp_regime_gate", False)
    # The common classifier superseded the overlay-only confirmation field. Its
    # shared warm-up is inactive while all regime gates are disabled.
    strategy_params.pop("trend_confirm_bars", None)
    strategy_params.setdefault("regime_min_samples", 20)
    # Percentage-sizing fields were added after the baseline artefact. Normalise
    # their disabled all-zero state so metadata-only differences do not fail.
    strategy_params.setdefault("total_budget", 0.0)
    strategy_params.setdefault("alloc_pct", 0.0)
    strategy_params.setdefault("entry_pct", 0.0)
    strategy_params.setdefault("dca_pct", 0.0)
    return {
        "schema_version": report.get("schema_version"),
        "strategy_params": strategy_params,
        "initial_capital_usd": report.get("initial_capital_usd"),
        "dataset": {
            key: report.get("dataset", {}).get(key)
            for key in ("sha256", "bars", "venue", "market", "role")
        },
        "walk_forward": report.get("walk_forward"),
        "scenarios": report.get("scenarios"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=KRAKEN_DIR / ".env")
    parser.add_argument("--config-file", type=Path, default=KRAKEN_DIR / "config.env")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--params-report", type=Path)
    parser.add_argument("--name", default="base_v2_live")
    parser.add_argument("--train", type=int, default=720)
    parser.add_argument("--validation", type=int, default=180)
    parser.add_argument("--test", type=int, default=90)
    parser.add_argument("--step", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--regime-threshold", type=float, default=3.0)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    for name in ("train", "validation", "test", "step"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0 or args.regime_threshold < 0:
        parser.error("warmup and regime-threshold cannot be negative")
    args.dataset = args.dataset.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.params_report:
        args.params_report = args.params_report.expanduser().resolve()
    if args.compare_to:
        args.compare_to = args.compare_to.expanduser().resolve()

    report = build_report(args)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.expanduser().resolve() if args.output
        else ROOT / "offline" / "results" / "hype_financial" / f"{args.name}_{stamp}.json"
    )
    markdown = (
        args.markdown.expanduser().resolve() if args.markdown
        else output.with_suffix(".md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write(markdown_report(report))

    print(f"Financial benchmark JSON: {output}")
    print(f"Financial benchmark Markdown: {markdown}")
    for name, scenario in report["scenarios"].items():
        aggregate = scenario["aggregate"]
        print(
            f"{name}: mean={aggregate['mean_return_pct']:+.3f}% "
            f"meanUSD={aggregate['mean_net_pnl_usd_per_window']:+.2f} "
            f"worst={aggregate['worst_return_pct']:+.3f}% "
            f"worstDD={aggregate['worst_max_drawdown_pct']:.3f}%"
        )

    if args.verify:
        with args.verify.expanduser().resolve().open(encoding="utf-8") as handle:
            expected = json.load(handle)
        if _projection(expected) != _projection(report):
            print("VERIFY FAILED: the benchmark differs from the versioned artefact", file=sys.stderr)
            return 1
        print("VERIFY OK: benchmark reproductibil")
    if "promotion_gate" in report:
        print(f"Promotion: {'PASS' if report['promotion_gate']['promote'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
