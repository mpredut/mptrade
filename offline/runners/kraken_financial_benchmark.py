#!/usr/bin/env python3
"""HYPE financial benchmark: fixed configuration, central + stress, OOS TEST only."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
KRAKEN_DIR = ROOT / "kraken"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import (  # noqa: E402
    exact_nonneg_int,
    exact_pos_int,
    load_layered_env_config,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from state_io import atomic_write_json, atomic_write_text, read_json_file  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402
from strategies import spot_replay as kraken_replay  # noqa: E402
from strategies.spot_replay import align_spot_interval_params  # noqa: E402
from offline.backtests.comparison import format_number  # noqa: E402
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
from offline.backtests.promotion import _assumption_signature, evaluate_dual_promotion  # noqa: E402
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
    manifest = read_json_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("the manifest must be a JSON object")
    datasets_map = manifest.get("datasets")
    expected = datasets_map.get(str(interval)) if isinstance(datasets_map, dict) else None
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"the manifest does not declare the {interval}m dataset")
    return {"manifest": manifest, "expected": expected}


def strat_params_from_dict(values: dict) -> StratParams:
    """Build StratParams from a stored report, dropping fields that no longer exist."""
    raw = dict(values) if isinstance(values, dict) else {}
    return StratParams.from_dict(raw, defaults={"sl_mhdsl_enabled": False}, warn_dropped=False)


def _load_params(args) -> StratParams:
    if args.params_report:
        payload = read_json_file(args.params_report)
        values = payload.get("strategy_params", payload) if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise ValueError("--params-report must contain strategy_params or an object")
        return strat_params_from_dict(values)
    cfg = load_layered_env_config(args.config_file, args.env_file, fallback_prefixes=())
    return StratParams.from_env(cfg)


def _scenario_windows(records: list[dict], params: StratParams, scenario, *,
                      interval: int, train: int, validation: int, test: int,
                      step: int, warmup: int) -> list[dict]:
    folds = walk_forward_splits(
        len(records), train_size=train, validation_size=validation,
        test_size=test, step_size=step,
    )
    if not folds:
        raise ValueError("the dataset is too small for the walk-forward schema")

    aligned_params = align_spot_interval_params(params, interval)
    adapter = kraken_replay.make_spot_replay_adapter(
        aligned_params,
        bar_minutes=interval,
        execution=scenario.execution,
        fee_model=scenario.fees,
    )
    windows = []
    for fold_index, fold in enumerate(folds, start=1):
        test_records = records[fold.test]
        warmup_records = records[max(0, fold.test.start - warmup):fold.test.start]
        segment = evaluate_segment(
            test_records,
            adapter,
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
            f"different dataset hash: {digest} != {expected.get('sha256')}"
        )
    expected_bars = safe_int(expected.get("bars"), default=-1)
    if len(records) != expected_bars:
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
                windows, initial_capital=params.allocated_budget(),
                regime_threshold_pct=args.regime_threshold,
            ),
            "windows": windows,
        }

    report = {
        "schema_version": 1,
        "benchmark": "HYPE base-v2 financial OOS",
        "candidate_name": candidate_name or getattr(args, "name", "base_v2_live"),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code": _git_state(),
        "strategy_params": dataclasses.asdict(params),
        "initial_capital_usd": params.allocated_budget(),
        "dataset": {
            "file": _display_path(args.dataset),
            "manifest": _display_path(args.manifest),
            "sha256": digest,
            "bars": len(records),
            "start_utc": iso_utc(safe_int(records[0]["timestamp"], 0)),
            "end_utc": iso_utc(safe_int(records[-1]["timestamp"], 0)),
            "venue": (manifest_info["manifest"].get("source") if isinstance(manifest_info["manifest"].get("source"), dict) else {}).get("venue"),
            "market": (manifest_info["manifest"].get("source") if isinstance(manifest_info["manifest"].get("source"), dict) else {}).get("market"),
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
        baseline = read_json_file(compare_to)
        report["promotion_gate"] = evaluate_dual_promotion(baseline, report)
    return report


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
            f"| `{name}` | {format_number(aggregate['mean_return_pct'], signed=True)} | "
            f"{format_number(aggregate['mean_net_pnl_usd_per_window'], signed=True)} | "
            f"{format_number(aggregate['sum_reset_net_pnl_usd'], signed=True)} | "
            f"{format_number(aggregate['worst_return_pct'], signed=True)} | "
            f"{format_number(aggregate['worst_max_drawdown_pct'])} | "
            f"{format_number(aggregate['mean_buy_hold_return_pct'], signed=True)} | "
            f"{format_number(aggregate['worst_cvar_95_pct'], signed=True)} | "
            f"{format_number(aggregate['mean_exposure_pct'])} | "
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
                f"{format_number(values['mean_return_pct'], signed=True)} | "
                f"{format_number(values['mean_buy_hold_return_pct'], signed=True)} | "
                f"{format_number(values['worst_return_pct'], signed=True)} |"
            )
        lines.append("")
    if "promotion_gate" in report:
        paths = "/".join(report["promotion_gate"]["promotion_paths"])
        verdict = f"PROMOTE {paths}" if paths else "DO NOT PROMOTE"
        lines.extend(["## Promotion gate", "", f"Verdict: **{verdict}**", ""])
    first_scen = next(iter((report.get("scenarios") or {}).values()), None)
    raw_tb = ((first_scen or {}).get("aggregate") or {}).get("total_test_bars") if isinstance(first_scen, dict) and isinstance(first_scen.get("aggregate"), dict) else 0
    total_bars = safe_int(raw_tb, default=0)
    wf_dict = report.get("walk_forward") if isinstance(report.get("walk_forward"), dict) else {}
    raw_wfi = wf_dict.get("interval_minutes", 240.0)
    wf_interval = safe_pos_float(raw_wfi, default=240.0) or 240.0
    wf_step = safe_int(wf_dict.get("step"), default=0)
    wf_test = safe_int(wf_dict.get("test"), default=0)
    overlap_note = (
        "with overlapping windows"
        if (wf_step > 0 and wf_test > 0 and wf_step < wf_test)
        else "without overlap"
    )
    lines.extend([
        "## Interpretation", "",
        "- `Mean USD/fold` uses the same initial budget in every TEST window.",
        "- `Sum reset USD` adds up the folds; it is not compounded equity and does not carry the position across windows.",
        f"- The {total_bars} TEST bars mean "
        f"{total_bars * wf_interval / 1440:.0f} days {overlap_note}.",
        "- Final calibration requires real Kraken distributions for spread, slippage and fee tier.",
        "",
    ])
    return "\n".join(lines)


def _projection(report: dict) -> dict:
    rep = report if isinstance(report, dict) else {}
    raw_sp = rep.get("strategy_params")
    raw_params = dict(raw_sp) if isinstance(raw_sp, dict) else {}
    strategy_params = dataclasses.asdict(strat_params_from_dict(raw_params))
    raw_ds = rep.get("dataset")
    ds_map = raw_ds if isinstance(raw_ds, dict) else {}
    raw_scen = rep.get("scenarios")
    norm_scenarios = (
        {
            s_name: {
                **s_val,
                "assumptions": _assumption_signature(s_val.get("assumptions")),
            }
            if isinstance(s_val, dict)
            else s_val
            for s_name, s_val in raw_scen.items()
        }
        if isinstance(raw_scen, dict)
        else raw_scen
    )
    return {
        "schema_version": rep.get("schema_version"),
        "strategy_params": strategy_params,
        "initial_capital_usd": rep.get("initial_capital_usd"),
        "dataset": {
            key: ds_map.get(key)
            for key in ("sha256", "bars", "venue", "market", "role")
        },
        "walk_forward": rep.get("walk_forward"),
        "scenarios": norm_scenarios,
    }


def validate_financial_benchmark_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("train", "validation", "test", "step"):
        if exact_pos_int(getattr(args, name), default=None) is None:
            parser.error(f"--{name} must be positive")
    if (
        exact_nonneg_int(args.warmup, default=None) is None
        or safe_nonneg_float(args.regime_threshold, default=None) is None
    ):
        parser.error("warmup and regime-threshold cannot be negative")


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    validate_financial_benchmark_args(parser, args)
    if not isinstance(args.name, str) or not args.name.strip() or "/" in args.name or "\\" in args.name or ".." in args.name:
        parser.error(f"invalid --name: {args.name!r}")
    args.dataset = args.dataset.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.params_report:
        args.params_report = args.params_report.expanduser().resolve()
    if args.compare_to:
        args.compare_to = args.compare_to.expanduser().resolve()

    try:
        report = build_report(args)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            args.output.expanduser().resolve() if args.output
            else ROOT / "offline" / "results" / "hype_financial" / f"{args.name.strip()}_{stamp}.json"
        )
        markdown = (
            args.markdown.expanduser().resolve() if args.markdown
            else output.with_suffix(".md")
        )
        atomic_write_json(output, report, indent=2, ensure_ascii=True)
        atomic_write_text(markdown, markdown_report(report))

        print(f"Financial benchmark JSON: {output}")
        print(f"Financial benchmark Markdown: {markdown}")
        for name, scenario in report["scenarios"].items():
            aggregate = scenario["aggregate"]
            print(
                f"{name}: mean={format_number(aggregate.get('mean_return_pct'), signed=True)}% "
                f"meanUSD={format_number(aggregate.get('mean_net_pnl_usd_per_window'), signed=True)} "
                f"worst={format_number(aggregate.get('worst_return_pct'), signed=True)}% "
                f"worstDD={format_number(aggregate.get('worst_max_drawdown_pct'))}%"
            )

        if args.verify:
            expected = read_json_file(args.verify.expanduser().resolve())
            if _projection(expected) != _projection(report):
                print("VERIFY FAILED: the benchmark differs from the versioned artefact", file=sys.stderr)
                return 1
            print("VERIFY OK: reproducible benchmark")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if "promotion_gate" in report:
        print(f"Promotion: {'PASS' if report['promotion_gate']['promote'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
