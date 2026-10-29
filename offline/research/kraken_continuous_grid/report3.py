#!/usr/bin/env python3
"""Aggregate results3.jsonl (finite 3900 USD cash, continuous history) into REPORT3.md."""
import json
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float

_safe_f = safe_float

try:
    from .report import _load_jsonl_dicts
except ImportError:
    from offline.research.kraken_continuous_grid.report import _load_jsonl_dicts


def generate_report(exp_dir: str | None = None) -> list[str]:
    exp = exp_dir or os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
    os.makedirs(exp, exist_ok=True)
    results3_path = os.path.join(exp, "results3.jsonl")
    rows = _load_jsonl_dicts(results3_path)
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r and "cfg" in r and "asset" in r]
    by = defaultdict(dict)
    for r in rows:
        by[r["cfg"]][r["asset"]] = r
    assets = sorted({r["asset"] for r in rows})

    L = ["# Finite-cash continuous runs (3900 USD, no refill after losses)\n",
         "Return and max drawdown as % of 3900 USD over the whole history per asset; "
         "`neg` = assets with a loss; `min cash` = lowest free cash seen (median over assets).\n"]
    for fee in ("0.26", "0.4"):
        L.append(f"\n## Fee {fee}% per leg\n")
        L.append("| config | beat live | neg | ret med % | ret min % | ret max % | DD med % | DD max % | "
                 "ret/DD med | refused buys med |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        items = []
        for cfg, d in by.items():
            if not all(a in d for a in assets):
                continue
            if any(not isinstance(d[a].get(fee), dict) for a in assets):
                continue
            ret = [safe_float(d[a][fee].get("ret"), default=float("nan")) for a in assets]
            dd = [safe_float(d[a][fee].get("dd"), default=float("nan")) for a in assets]
            if not all(safe_float(x, default=None) is not None for x in ret + dd):
                continue
            live_cfg = by.get("live", {})
            live = [safe_float((live_cfg.get(a, {}).get(fee) or {}).get("ret"), default=float("-inf")) for a in assets]
            refused = [safe_float(d[a][fee].get("refused"), default=0.0) for a in assets]
            score = st.median(r / max(x, 1.0) for r, x in zip(ret, dd)) if ret else 0.0
            items.append((safe_float(score, default=float("-inf")), cfg, ret, dd, live, refused))
        for score, cfg, ret, dd, live, refused in sorted(items, reverse=True):
            if not ret:
                continue
            L.append(f"| {cfg} | {sum(r > l for r, l in zip(ret, live))}/{len(assets)} | "
                     f"{sum(r < 0 for r in ret)} | {st.median(ret):+.1f} | {min(ret):+.1f} | {max(ret):+.1f} | "
                     f"{st.median(dd):.1f} | {max(dd):.1f} | {score:+.2f} | {st.median(refused):.0f} |")

    L.append("\n## Per asset, fee 0.26% (return %, DD % in brackets)\n")
    cfgs = [c for c in ("live", "old_live_pre0923", "sl0", "ov650_t8", "ov650_t8_brake_gateoff",
                        "ov650_t8_sl0", "ov1000_t8_gateoff", "ov2000_t10") if c in by]
    L.append("| asset | " + " | ".join(cfgs) + " |")
    L.append("|" + "---|" * (len(cfgs) + 1))
    for a in assets:
        cells = []
        for c in cfgs:
            d026 = by.get(c, {}).get(a, {}).get("0.26")
            if isinstance(d026, dict) and "ret" in d026 and "dd" in d026:
                cells.append(f"{safe_float(d026.get('ret'), 0.0):+.1f} ({safe_float(d026.get('dd'), 0.0):.0f})")
            else:
                cells.append("- / -")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    if errors:
        L.append("\n## Errors\n")
        for e in errors[:20]:
            err_lines = str(e.get("error") or "").strip().splitlines()
            last_err = err_lines[-1] if err_lines else "unknown"
            L.append(f"- {e.get('asset', '?')} {e.get('cfg', '?')}: `{last_err}`")

    with open(os.path.join(exp, "REPORT3.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines))
