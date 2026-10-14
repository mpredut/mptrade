#!/usr/bin/env python3
"""Aggregate results3.jsonl (finite 3900 USD cash, continuous history) into REPORT3.md."""
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def generate_report(exp_dir: str | None = None) -> list[str]:
    exp = exp_dir or os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
    os.makedirs(exp, exist_ok=True)
    results3_path = os.path.join(exp, "results3.jsonl")
    rows = []
    if os.path.exists(results3_path):
        with open(results3_path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
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
            if any(fee not in d[a] for a in assets):
                continue
            ret = [d[a][fee]["ret"] for a in assets]
            dd = [d[a][fee]["dd"] for a in assets]
            live_cfg = by.get("live", {})
            live = [live_cfg.get(a, {}).get(fee, {}).get("ret", float("-inf")) for a in assets]
            refused = [d[a][fee].get("refused", 0) for a in assets]
            score = st.median(r / max(x, 1.0) for r, x in zip(ret, dd)) if ret else 0.0
            items.append((score, cfg, ret, dd, live, refused))
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
        L.append(f"| {a} | " + " | ".join(
            f"{by[c][a]['0.26']['ret']:+.1f} ({by[c][a]['0.26']['dd']:.0f})"
            if (a in by.get(c, {}) and "0.26" in by[c][a]) else "- / -"
            for c in cfgs) + " |")

    if errors:
        L.append("\n## Errors\n")
        for e in errors[:20]:
            L.append(f"- {e.get('asset', '?')} {e.get('cfg', '?')}: `{e.get('error', '').strip().splitlines()[-1]}`")

    with open(os.path.join(exp, "REPORT3.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines))
