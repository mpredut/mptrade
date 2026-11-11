#!/usr/bin/env python3
"""Compare two financial reports and apply the conservative gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from botcore import safe_float  # noqa: E402
from offline.backtests.comparison import add_path_cli_args, resolve_existing_report_path  # noqa: E402
from offline.backtests.promotion import evaluate_dual_promotion  # noqa: E402
from state_io import atomic_write_json, read_json_file  # noqa: E402


def _fmt_calmar_pct(calmar) -> str:
    cf = safe_float(calmar, default=None)
    return f"{cf * 100:+.1f}%" if cf is not None else "n/a"


def _fmt_pp(val) -> str:
    vf = safe_float(val, default=None)
    return f"{vf:+.3f}pp" if vf is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = add_path_cli_args(
        argparse.ArgumentParser(description=__doc__), "baseline", "candidate", "--output"
    )
    args = parser.parse_args(argv)
    try:
        baseline_path = resolve_existing_report_path(args.baseline, "baseline")
        candidate_path = resolve_existing_report_path(args.candidate, "candidate")
        baseline = read_json_file(baseline_path)
        candidate = read_json_file(candidate_path)
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError("baseline and candidate reports must be JSON objects")
        result = evaluate_dual_promotion(baseline, candidate)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    paths = "/".join(result["promotion_paths"])
    print(f"PROMOTE {paths}" if paths else "DO NOT PROMOTE")
    for name, scenario in result["return_gate"]["scenarios"].items():
        deltas = scenario["wins_ties_losses"]
        risk = result["defensive_gate"]["scenarios"][name]
        dd = risk["drawdown_wins_ties_losses"]
        calmar_text = _fmt_calmar_pct(risk["median_calmar_improvement_ratio"])
        print(
            f"{name}: return={'PASS' if scenario['passed'] else 'FAIL'} "
            f"defensive={'PASS' if risk['passed'] else 'FAIL'} "
            f"mean_d={_fmt_pp(scenario.get('mean_return_delta_pp'))} "
            f"worst_d={_fmt_pp(scenario.get('worst_return_delta_pp'))} "
            f"dd_d={_fmt_pp(scenario.get('worst_drawdown_delta_pp'))} "
            f"returnW/T/L={deltas['wins']}/{deltas['ties']}/{deltas['losses']} "
            f"calmar_d={calmar_text}"
        )
        print(f"  ddW/T/L={dd['wins']}/{dd['ties']}/{dd['losses']}")
    if args.output:
        try:
            atomic_write_json(args.output.expanduser().resolve(), result, indent=2, ensure_ascii=True)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    return 0 if result["promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
