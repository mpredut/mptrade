#!/usr/bin/env python3
"""Aggregate results2.jsonl (wave 2) into REPORT2.md."""
import json
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float

try:
    from .report import _load_jsonl_dicts
except ImportError:
    from offline.research.kraken_continuous_grid.report import _load_jsonl_dicts


def generate_report(exp_dir: str | None = None) -> list[str]:
    exp = exp_dir or os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
    os.makedirs(exp, exist_ok=True)
    results2_path = os.path.join(exp, "results2.jsonl")
    rows = _load_jsonl_dicts(results2_path)
    results_path = os.path.join(exp, "results.jsonl")
    wave1 = _load_jsonl_dicts(results_path)
    errors = [r for r in rows if r.get("kind") == "error"]
    L = ["# Overnight wave 2: selection, fine overlay grid, fee stress\n"]

    # ---------------------------------------------------------------- selection
    sel = [r for r in rows if r.get("kind") == "select" and r.get("steps")]
    L.append("## Walk-forward selection (365-day lookback, 90-day out-of-sample steps)\n")
    L.append("Each step ranks the candidate set by trailing return/max(DD,5) and runs the winner on the "
             "next 90 days. `sum` = sum of out-of-sample 90-day returns (% of budget); `beat live` = "
             "steps where the pick beat `live`.\n")
    cands = (
        list(sel[0]["steps"][0]["oos"])
        if (
            sel
            and isinstance(sel[0].get("steps"), list)
            and sel[0]["steps"]
            and isinstance(sel[0]["steps"][0], dict)
            and isinstance(sel[0]["steps"][0].get("oos"), dict)
        )
        else []
    )
    L.append("| asset | steps | selector sum | live sum | best static (sum) | selector rank | beat live | picks |")
    L.append("|---|---|---|---|---|---|---|---|")
    tot = defaultdict(float)
    for r in sorted(sel, key=lambda r: str(r.get("asset", ""))):
        s = [x for x in (r.get("steps") or []) if isinstance(x, dict)]
        if not s:
            continue
        sums = {
            c: sum(safe_float((x.get("oos") if isinstance(x.get("oos"), dict) else {}).get(c), 0.0) for x in s)
            for c in cands
        }
        sel_sum = sum(safe_float(x.get("pick_ret"), 0.0) for x in s)
        for c, v in sums.items():
            tot[c] += v
        tot["__selector__"] += sel_sum
        best = max(sums, key=sums.get) if sums else "-"
        best_sum = sums[best] if best in sums else 0.0
        rank = 1 + sum(v > sel_sum for v in sums.values())
        picks = defaultdict(int)
        for x in s:
            picks[x.get("pick", "-")] += 1
        top = ", ".join(f"{k}:{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1])[:3])
        live_sum = sums.get("live", 0.0)
        beat_cnt = sum(
            safe_float(x.get("pick_ret"), 0.0)
            > safe_float((x.get("oos") if isinstance(x.get("oos"), dict) else {}).get("live"), float("-inf"))
            for x in s
        )
        L.append(f"| {r['asset']} | {len(s)} | {sel_sum:+.1f} | {live_sum:+.1f} | {best} ({best_sum:+.1f}) | "
                 f"{rank}/{len(cands) + 1} | {beat_cnt}/{len(s)} | {top} |")
    if sel:
        L.append("\n**Sum over all assets and steps** (higher is better):\n")
        L.append("| strategy | OOS sum % |")
        L.append("|---|---|")
        for c, v in sorted(tot.items(), key=lambda kv: -kv[1]):
            L.append(f"| {'SELECTOR' if c == '__selector__' else c} | {v:+.1f} |")

    # ---------------------------------------------------------------- fine grid + fee
    views = [r for r in rows if r.get("kind") == "views" and "cfg" in r and "fee" in r and "asset" in r]
    by = defaultdict(dict)
    for r in views:
        by[(r["cfg"], r["fee"])][r["asset"]] = r
    for r in wave1:
        if "error" not in r and "cfg" in r and "asset" in r:
            by.setdefault((r["cfg"], "std"), {}).setdefault(r["asset"], r)
    assets = sorted({r["asset"] for r in views})

    def summ(key):
        d = by[key]
        if not assets or not all(a in d and isinstance(d[a].get("full"), dict) for a in assets):
            return None
        live = by.get(("live", "std"), {})
        full = [safe_float(d[a]["full"].get("ret"), float("nan")) for a in assets]
        dd = [safe_float(d[a]["full"].get("dd"), float("nan")) for a in assets]
        if not all(safe_float(x, default=None) is not None for x in full + dd):
            return None
        w90_means = []
        w90pos_vals = []
        for a in assets:
            w90_valid = [
                v for x in (d[a].get("w90") or [])
                if isinstance(x, (list, tuple)) and x and (v := safe_float(x[0], default=None)) is not None
            ]
            if w90_valid:
                w90_means.append(st.mean(w90_valid))
                w90pos_vals.append(100.0 * sum(v > 0 for v in w90_valid) / len(w90_valid))
        beat_live = sum(
            safe_float(d[a]["full"].get("ret"), float("-inf"))
            > safe_float((live.get(a, {}).get("full") or {}).get("ret") if isinstance(live.get(a, {}).get("full"), dict) else None, float("-inf"))
            for a in assets
        )
        return {"cfg": key[0], "fee": key[1],
                "beat": beat_live,
                "full_med": st.median(full) if full else 0.0, "full_min": min(full) if full else 0.0,
                "dd_med": st.median(dd) if dd else 0.0, "dd_max": max(dd) if dd else 0.0,
                "calmar": st.median(f / max(x, 1.0) for f, x in zip(full, dd)) if full else 0.0,
                "w90": st.mean(w90_means) if w90_means else float("nan"),
                "w90pos": st.mean(w90pos_vals) if w90pos_vals else float("nan")}

    def table(items, title):
        L.append(f"\n## {title}\n")
        L.append("| config | fee | beat live (full) | full med % | full min % | DD med | DD max | calmar med | w90 mean | w90 pos% |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in items:
            L.append(f"| {s['cfg']} | {s['fee']} | {s['beat']}/{len(assets)} | {s['full_med']:+.1f} | {s['full_min']:+.1f} | "
                     f"{s['dd_med']:.1f} | {s['dd_max']:.1f} | {s['calmar']:+.2f} | {s['w90']:+.2f} | {s['w90pos']:.0f} |")

    fine = [s for s in (summ(k) for k in by if k[1] == "std") if s]
    table(sorted(fine, key=lambda s: -(safe_float(s.get("calmar"), default=float("-inf"))))[:40], "Fine grid + wave-1 configs, top 40 by median return/DD")
    fee = []
    for k in by:
        if k[1] == "stress":
            a, b = summ((k[0], "std")), summ(k)
            if a and b:
                fee += [a, b]
    table(fee, "Fee stress: 0.26% vs 0.40% per leg")

    if errors:
        L.append("\n## Errors\n")
        for e in errors[:20]:
            err_lines = str(e.get("error") or "").strip().splitlines()
            last_err = err_lines[-1] if err_lines else "unknown"
            L.append(f"- {e.get('args', '?')}: `{last_err}`")
    with open(os.path.join(exp, "REPORT2.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines[:50]))
