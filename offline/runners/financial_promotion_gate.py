#!/usr/bin/env python3
"""Compare two financial reports and apply the conservative gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from offline.backtests.promotion import evaluate_dual_promotion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.baseline.expanduser().resolve().open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    with args.candidate.expanduser().resolve().open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("baseline and candidate reports must be JSON objects")
    result = evaluate_dual_promotion(baseline, candidate)
    paths = "/".join(result["promotion_paths"])
    print(f"PROMOTE {paths}" if paths else "DO NOT PROMOTE")
    for name, scenario in result["return_gate"]["scenarios"].items():
        deltas = scenario["wins_ties_losses"]
        risk = result["defensive_gate"]["scenarios"][name]
        dd = risk["drawdown_wins_ties_losses"]
        calmar = risk["median_calmar_improvement_ratio"]
        calmar_text = f"{calmar * 100:+.1f}%" if calmar is not None else "n/a"
        print(
            f"{name}: return={'PASS' if scenario['passed'] else 'FAIL'} "
            f"defensive={'PASS' if risk['passed'] else 'FAIL'} "
            f"meanΔ={scenario['mean_return_delta_pp']:+.3f}pp "
            f"worstΔ={scenario['worst_return_delta_pp']:+.3f}pp "
            f"ddΔ={scenario['worst_drawdown_delta_pp']:+.3f}pp "
            f"returnW/T/L={deltas['wins']}/{deltas['ties']}/{deltas['losses']} "
            f"calmarΔ={calmar_text}"
        )
        print(f"  ddW/T/L={dd['wins']}/{dd['ties']}/{dd['losses']}")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return 0 if result["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
