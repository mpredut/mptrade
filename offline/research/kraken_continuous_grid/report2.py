#!/usr/bin/env python3
"""Aggregate results2.jsonl (wave 2) into REPORT2.md."""
import math
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float
from state_io import read_jsonl_dicts
from offline.research.kraken_continuous_grid import grid2
from offline.research.kraken_continuous_grid.report import (
    _append_markdown_table_header,
    _bind_report_generator,
    _finalize_markdown_report,
    _fmt_num,
    _is_error_row,
    _matches_known_spec,
    _rank_key,
    _resolve_exp_dir,
)


def _generate_report_with_status(exp_dir: str | None = None) -> tuple[list[str], bool]:
    exp = _resolve_exp_dir(exp_dir)
    results2_path = os.path.join(exp, "results2.jsonl")
    rows = read_jsonl_dicts(results2_path)
    results_path = os.path.join(exp, "results.jsonl")
    wave1 = read_jsonl_dicts(results_path)
    errors = [r for r in rows if _is_error_row(r)]
    L = ["# Overnight wave 2: selection, fine overlay grid, fee stress\n"]

    # ---------------------------------------------------------------- selection
    sel_by_asset: dict[str, dict] = {}
    for r in rows:
        if (
            r.get("kind") == "select"
            and not _is_error_row(r)
            and isinstance(r.get("asset"), str)
            and r["asset"].strip()
            and isinstance(r.get("steps"), list)
            and r["steps"]
        ):
            sel_by_asset[r["asset"]] = r
    sel = list(sel_by_asset.values())
    L.extend([
        "## Walk-forward selection (365-day lookback, 90-day out-of-sample steps)\n",
        "Each step ranks the candidate set by trailing return/max(DD,5) and runs the winner on the "
        "next 90 days. `sum` = sum of out-of-sample 90-day returns (% of budget); `beat live` = "
        "steps where the pick beat `live`.\n",
    ])
    cands = list(
        dict.fromkeys(
            c
            for r in sel
            for x in (r.get("steps") or [])
            if isinstance(x, dict) and isinstance(x.get("oos"), dict)
            for c in x["oos"]
            if isinstance(c, str) and c.strip()
        )
    )
    _append_markdown_table_header(
        L,
        None,
        "| asset | steps | selector sum | live sum | best static (sum) | selector rank | beat live | picks |",
        8,
    )
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
            (pick_r > live_r + 1e-9)
            if (pick_r := safe_float(x.get("pick_ret"), default=None)) is not None
            and isinstance(x.get("oos"), dict)
            and (live_r := safe_float(x["oos"].get("live"), default=None)) is not None
            else False
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
    views = [
        r for r in rows
        if r.get("kind") == "views"
        and not _is_error_row(r)
        and isinstance(r.get("cfg"), str) and r["cfg"].strip()
        and isinstance(r.get("fee"), str) and r["fee"].strip()
        and isinstance(r.get("asset"), str) and r["asset"].strip()
        and _matches_known_spec(r, grid2.ALL)
    ]
    wave1_valid = [
        r for r in wave1
        if not _is_error_row(r)
        and isinstance(r.get("cfg"), str) and r["cfg"].strip()
        and isinstance(r.get("asset"), str) and r["asset"].strip()
        and _matches_known_spec(r, grid2.ALL)
    ]
    by = defaultdict(dict)
    for r in views:
        by[(r["cfg"], r["fee"])][r["asset"]] = r
    wave1_dedup: dict[tuple[str, str], dict] = {}
    for r in wave1_valid:
        wave1_dedup[(r["cfg"], r["asset"])] = r
    for (cfg_k, asset_k), r in wave1_dedup.items():
        by.setdefault((cfg_k, "std"), {}).setdefault(asset_k, r)
    assets = sorted({r["asset"] for r in views} | {r["asset"] for r in wave1_valid})

    def summ(key):
        d = by[key]
        if not assets or not all(a in d and isinstance(d[a].get("full"), dict) for a in assets):
            return None
        live = by.get(("live", "std"), {})
        full = [safe_float(d[a]["full"].get("ret"), math.nan) for a in assets]
        dd = [safe_float(d[a]["full"].get("dd"), math.nan) for a in assets]
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
            (c_ret > live_ret + 1e-9)
            if (c_ret := safe_float(d[a]["full"].get("ret"), default=None)) is not None
            and isinstance(live.get(a, {}).get("full"), dict)
            and (live_ret := safe_float(live[a]["full"].get("ret"), default=None)) is not None
            else False
            for a in assets
        )
        return {"cfg": key[0], "fee": key[1],
                "beat": beat_live,
                "full_med": st.median(full) if full else 0.0, "full_min": min(full) if full else 0.0,
                "dd_med": st.median(dd) if dd else 0.0, "dd_max": max(dd) if dd else 0.0,
                "calmar": st.median(f / max(x, 1.0) for f, x in zip(full, dd)) if full else 0.0,
                "w90": st.mean(w90_means) if w90_means else None,
                "w90pos": st.mean(w90pos_vals) if w90pos_vals else None}

    def table(items, title):
        _append_markdown_table_header(
            L,
            title,
            "| config | fee | beat live (full) | full med % | full min % | DD med | DD max | calmar med | w90 mean | w90 pos% |",
            10,
        )
        for s in items:
            L.append(
                f"| {s['cfg']} | {s['fee']} | {s['beat']}/{len(assets)} | "
                f"{_fmt_num(s['full_med'], '+.1f')} | {_fmt_num(s['full_min'], '+.1f')} | "
                f"{_fmt_num(s['dd_med'], '.1f')} | {_fmt_num(s['dd_max'], '.1f')} | "
                f"{_fmt_num(s['calmar'], '+.2f')} | {_fmt_num(s['w90'], '+.2f')} | {_fmt_num(s['w90pos'], '.0f')} |"
            )

    fine = [s for s in (summ(k) for k in by if k[1] == "std") if s]
    table(
        sorted(fine, key=lambda s: _rank_key(s, calmar_key="calmar"), reverse=True)[:40],
        "Fine grid + wave-1 configs, top 40 by median return/DD",
    )
    fee = []
    for k in by:
        if k[1] == "stress":
            a, b = summ((k[0], "std")), summ(k)
            if a and b:
                fee += [a, b]
    table(fee, "Fee stress: 0.26% vs 0.40% per leg")

    has_valid = bool(sel or views)
    report2_path = os.path.join(exp, "REPORT2.md")
    lines = _finalize_markdown_report(
        L,
        errors,
        report2_path,
        label_fn=lambda e: str(e.get("args", "?")),
        write_file=has_valid or not os.path.exists(report2_path),
    )
    return lines, has_valid


generate_report = _bind_report_generator(_generate_report_with_status)


if __name__ == "__main__":
    lines, has_valid = _generate_report_with_status()
    print("\n".join(lines[:50]))
    if not has_valid:
        sys.exit(1)
