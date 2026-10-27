#!/usr/bin/env python3
"""Aggregate results3.jsonl (finite 3900 USD cash, continuous history) into REPORT3.md."""
import json
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _safe_f(val, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _load_jsonl_dicts(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


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
            ret = [_safe_f(d[a][fee].get("ret"), float("nan")) for a in assets]
            dd = [_safe_f(d[a][fee].get("dd"), float("nan")) for a in assets]
            if not all(math.isfinite(x) for x in ret + dd):
                continue
            live_cfg = by.get("live", {})
            live = [_safe_f((live_cfg.get(a, {}).get(fee) or {}).get("ret"), float("-inf")) for a in assets]
            refused = [_safe_f(d[a][fee].get("refused"), 0.0) for a in assets]
            score = st.median(r / max(x, 1.0) for r, x in zip(ret, dd)) if ret else 0.0
            items.append((score if math.isfinite(score) else float("-inf"), cfg, ret, dd, live, refused))
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
            err_lines = str(e.get("error") or "").strip().splitlines()
            last_err = err_lines[-1] if err_lines else "unknown"
            L.append(f"- {e.get('asset', '?')} {e.get('cfg', '?')}: `{last_err}`")

    with open(os.path.join(exp, "REPORT3.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines))
