#!/usr/bin/env python3
"""Compare the pre-registered HYPE set in the central + stress financial scenarios.

The runner searches no parameters and changes no live configuration. It starts from
the versioned baseline parameters, reproduces the live profile and applies to every
candidate the same dataset, the same windows and the same promotion gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNNER_DIR))

import kraken_financial_benchmark as benchmark  # noqa: E402
from offline.backtests.hype_candidates import (  # noqa: E402
    financial_priority_candidates,
)
from offline.backtests.promotion import evaluate_dual_promotion  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402


DEFAULT_BASELINE = (
    ROOT / "offline" / "research" / "hype_dataset" / "financial_baseline_v1.json"
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_summary(report: dict) -> dict:
    return {
        name: values["aggregate"]
        for name, values in report["scenarios"].items()
    }


def build_comparison(args) -> dict:
    baseline_report = _load_json(args.baseline)
    base_params = benchmark.strat_params_from_dict(baseline_report["strategy_params"])
    candidates = financial_priority_candidates()
    selected = set(args.candidate or [])
    if selected:
        known = {candidate.name for candidate in candidates}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"unknown candidates: {', '.join(unknown)}")
        candidates = [
            candidate for candidate in candidates
            if candidate.name == "live" or candidate.name in selected
        ]

    results = {}
    live_report = None
    for candidate in candidates:
        params = dataclasses.replace(base_params, **candidate.overrides)
        report = benchmark.build_report(
            args, params=params, candidate_name=candidate.name,
        )
        if candidate.name == "live":
            if benchmark._projection(report) != benchmark._projection(baseline_report):
                raise ValueError(
                    "the regenerated live profile differs from the versioned baseline"
                )
            live_report = report
        gate = evaluate_dual_promotion(baseline_report, report)
        results[candidate.name] = {
            "description": candidate.description,
            "overrides": candidate.overrides,
            "strategy_params": dataclasses.asdict(params),
            "scenarios": _scenario_summary(report),
            "promotion_gate": gate,
        }

    if live_report is None:
        raise ValueError("the candidate set must contain the live profile")
    return {
        "schema_version": 1,
        "benchmark": "HYPE financial candidate comparison",
        "candidate_set": "hype-priority-v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code": live_report["code"],
        "baseline_file": benchmark._display_path(args.baseline),
        "dataset": live_report["dataset"],
        "walk_forward": live_report["walk_forward"],
        "candidates": results,
        "promotable": sorted(
            name for name, result in results.items()
            if name != "live" and result["promotion_gate"]["promote"]
        ),
    }


def _fmt(value: float, *, signed: bool = False) -> str:
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def markdown_report(report: dict) -> str:
    lines = [
        "# HYPE financial candidate comparison", "",
        "Pre-registered set; the same 31 OOS windows and the same costs for all.",
        "No candidate is enabled by this runner.", "",
        "| Candidate | Scenario | Mean % | Δ mean pp | Worst % | Δ worst pp | "
        "Δ DD pp | Return W/T/L | Calmar Δ | DD W/T/L | Return/Risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, candidate in report["candidates"].items():
        for scenario_name, aggregate in candidate["scenarios"].items():
            dual = candidate["promotion_gate"]
            gate = dual["return_gate"]["scenarios"][scenario_name]
            risk = dual["defensive_gate"]["scenarios"][scenario_name]
            wtl = gate["wins_ties_losses"]
            dd_wtl = risk["drawdown_wins_ties_losses"]
            calmar_delta = risk["median_calmar_improvement_ratio"]
            lines.append(
                f"| `{name}` | `{scenario_name}` | "
                f"{_fmt(aggregate['mean_return_pct'], signed=True)} | "
                f"{_fmt(gate['mean_return_delta_pp'], signed=True)} | "
                f"{_fmt(aggregate['worst_return_pct'], signed=True)} | "
                f"{_fmt(gate['worst_return_delta_pp'], signed=True)} | "
                f"{_fmt(gate['worst_drawdown_delta_pp'], signed=True)} | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                f"{_fmt(calmar_delta * 100.0, signed=True) + '%' if calmar_delta is not None else 'n/a'} | "
                f"{dd_wtl['wins']}/{dd_wtl['ties']}/{dd_wtl['losses']} | "
                f"{'PASS' if gate['passed'] else 'FAIL'}/"
                f"{'PASS' if risk['passed'] else 'FAIL'} |"
            )
    lines.extend([
        "", "## Verdict", "",
        "Promotable: " + (
            ", ".join(f"`{name}`" for name in report["promotable"])
            if report["promotable"] else "no candidate"
        ), "",
        "The result is screening on a Hyperliquid proxy; Kraken execution calibration "
        "and the forward shadow remain separate gates.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--dataset", type=Path, default=benchmark.DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=benchmark.DEFAULT_MANIFEST)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--train", type=int, default=720)
    parser.add_argument("--validation", type=int, default=180)
    parser.add_argument("--test", type=int, default=90)
    parser.add_argument("--step", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--regime-threshold", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for name in ("train", "validation", "test", "step"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0 or args.regime_threshold < 0:
        parser.error("warmup and regime-threshold cannot be negative")
    args.baseline = args.baseline.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()

    try:
        report = build_comparison(args)
    except ValueError as error:
        parser.error(str(error))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.expanduser().resolve() if args.output else
        ROOT / "offline" / "results" / "hype_financial" /
        f"candidate_comparison_{stamp}.json"
    )
    markdown = (
        args.markdown.expanduser().resolve() if args.markdown else
        output.with_suffix(".md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write(markdown_report(report))

    print(f"Financial comparison JSON: {output}")
    print(f"Financial comparison Markdown: {markdown}")
    print(f"Promotable: {', '.join(report['promotable']) or '-'}")
    for name, candidate in report["candidates"].items():
        central = candidate["scenarios"]["central"]
        stress = candidate["scenarios"]["stress"]
        print(
            f"{name}: central={central['mean_return_pct']:+.3f}% "
            f"stress={stress['mean_return_pct']:+.3f}% "
            f"gate={'/'.join(candidate['promotion_gate']['promotion_paths']) or 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
