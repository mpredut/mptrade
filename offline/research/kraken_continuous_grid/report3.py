#!/usr/bin/env python3
"""Aggregate results3.jsonl (finite 3900 USD cash, continuous history) into REPORT3.md."""
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float
from offline.research.kraken_continuous_grid import grid2
from offline.research.kraken_continuous_grid.report import (
    _bind_report_generator,
    _finalize_markdown_report,
    _fmt_num,
    _load_grid_results_by_cfg,
    _resolve_exp_dir,
    _risk_adjusted_score as _ret_dd_ratio,
)


def _generate_report_with_status(exp_dir: str | None = None) -> tuple[list[str], bool]:
    exp = _resolve_exp_dir(exp_dir)
    results3_path = os.path.join(exp, "results3.jsonl")
    valid_rows, errors, by, assets = _load_grid_results_by_cfg(results3_path, spec_map=grid2.ALL)

    L = ["# Finite-cash continuous runs (3900 USD, no refill after losses)\n",
         "Return and max drawdown as % of 3900 USD over the whole history per asset; "
         "`neg` = assets with a loss; `min cash` = lowest free cash seen (median over assets).\n"]
    for fee in ("0.26", "0.4"):
        L.append(f"\n## Fee {fee}% per leg\n")
        L.append("| config | beat live | neg | ret med % | ret min % | ret max % | DD med % | DD max % | "
                 "ret/DD med | refused buys med | min cash med |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        items = []
        for cfg, d in by.items():
            if not assets or not all(a in d for a in assets):
                continue
            fee_dicts = {
                a: d[a].get(fee) if isinstance(d[a].get(fee), dict) else (d[a].get("0.40") if fee == "0.4" else None)
                for a in assets
            }
            if any(not isinstance(fee_dicts[a], dict) for a in assets):
                continue
            ret = [safe_float(fee_dicts[a].get("ret"), default=None) for a in assets]
            dd = [safe_float(fee_dicts[a].get("dd"), default=None) for a in assets]
            if any(x is None for x in ret + dd):
                continue
            live_cfg = by.get("live", {})
            live = []
            for a in assets:
                l_row = live_cfg.get(a, {}) if isinstance(live_cfg.get(a), dict) else {}
                l_fee = l_row.get(fee) if isinstance(l_row.get(fee), dict) else (l_row.get("0.40") if fee == "0.4" else None)
                live.append(safe_float(l_fee.get("ret"), default=None) if isinstance(l_fee, dict) else None)
            refused = [safe_float(fee_dicts[a].get("refused"), default=0.0) for a in assets]
            min_cash = [safe_float(fee_dicts[a].get("min_cash"), default=3900.0) for a in assets]
            score = st.median(_ret_dd_ratio(r, x, dd_floor=1.0) for r, x in zip(ret, dd)) if ret else 0.0
            score_val = safe_float(score, default=None)
            items.append((score_val, cfg, ret, dd, live, refused, min_cash))
        for score, cfg, ret, dd, live, refused, min_cash in sorted(
            items,
            key=lambda x: (x[0] is not None, x[0] if x[0] is not None else -math.inf, x[1]),
            reverse=True,
        ):
            if not ret:
                continue
            live_pairs = [(r, l) for r, l in zip(ret, live) if l is not None]
            beat_str = f"{sum(r > l + 1e-9 for r, l in live_pairs)}/{len(live_pairs)}" if live_pairs else "-/-"
            L.append(f"| {cfg} | {beat_str} | "
                     f"{sum(r < 0 for r in ret)} | {st.median(ret):+.1f} | {min(ret):+.1f} | {max(ret):+.1f} | "
                     f"{st.median(dd):.1f} | {max(dd):.1f} | {_fmt_num(score, '+.2f')} | {st.median(refused):.0f} | {st.median(min_cash):.0f} |")

    L.append("\n## Per asset, fee 0.26% (return %, DD % in brackets)\n")
    cfgs = [c for c in ("live", "old_live_pre0923", "sl0", "ov650_t8", "ov650_t8_brake_gateoff",
                        "ov650_t8_sl0", "ov1000_t8_gateoff", "ov2000_t10") if c in by]
    L.append("| asset | " + " | ".join(cfgs) + " |")
    L.append("|" + "---|" * (len(cfgs) + 1))
    for a in assets:
        cells = []
        for c in cfgs:
            d026 = by.get(c, {}).get(a, {}).get("0.26")
            if (
                isinstance(d026, dict)
                and (r_v := safe_float(d026.get("ret"), default=None)) is not None
                and (d_v := safe_float(d026.get("dd"), default=None)) is not None
            ):
                cells.append(f"{r_v:+.1f} ({d_v:.0f})")
            else:
                cells.append("- / -")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    has_valid = bool(valid_rows)
    report3_path = os.path.join(exp, "REPORT3.md")
    lines = _finalize_markdown_report(
        L,
        errors,
        report3_path,
        write_file=has_valid or not os.path.exists(report3_path),
    )
    return lines, has_valid


generate_report = _bind_report_generator(_generate_report_with_status)


if __name__ == "__main__":
    lines, has_valid = _generate_report_with_status()
    print("\n".join(lines))
    if not has_valid:
        sys.exit(1)
