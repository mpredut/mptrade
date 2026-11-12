#!/usr/bin/env python3
"""Summarise results.jsonl from grid.py into REPORT.md."""
from collections import defaultdict
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import load_local_module, optional_env, safe_float
from offline.research.kraken_continuous_grid.report import (
    _append_markdown_table_header,
    _finalize_markdown_report,
    _fmt_num as _fmt_cell,
    load_grid_results_partitioned,
)

EXP = optional_env("GRID_OUT") or os.path.join(ROOT, "offline", "results", "t212_continuous_grid")
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}


def g(r, itv, k):
    sub = r.get(itv) if isinstance(r, dict) else None
    if not isinstance(sub, dict):
        return None
    val = sub.get(k)
    fval = safe_float(val, default=None)
    if fval is None:
        return None
    return val if (isinstance(val, int) and not isinstance(val, bool)) else fval


def generate_report(exp_dir=None):
    exp = exp_dir or optional_env("GRID_OUT") or EXP
    results_path = os.path.join(exp, "results.jsonl")
    if not os.path.exists(results_path):
        return f"# Trading 212 grid\n\nNo results found at {results_path}.\n"

    grid_mod = sys.modules.get("offline.research.t212_continuous_grid.grid") or load_local_module(
        os.path.dirname(os.path.abspath(__file__)),
        "grid",
        module_key="offline.research.t212_continuous_grid.grid",
    )
    profiles = getattr(grid_mod, "PROFILES", None)
    variants_map = getattr(grid_mod, "VARIANTS", None)
    params_fn = getattr(grid_mod, "params_for", None)
    spec_map = (
        {(p, v): params_fn(p, v) for p in profiles for v in variants_map}
        if isinstance(profiles, dict) and isinstance(variants_map, dict) and callable(params_fn)
        else None
    )
    valid_rows, errors = load_grid_results_partitioned(
        results_path,
        ("profile", "variant", "asset"),
        spec_map=spec_map,
        spec_key=("profile", "variant"),
    )
    valid_rows = [
        r for r in valid_rows
        if g(r, "1h", "ret") is not None or g(r, "1d", "ret") is not None
    ]

    by = defaultdict(dict)  # (profile, variant) -> asset -> row
    for r in valid_rows:
        by[(r.get("profile"), r.get("variant"))][r.get("asset")] = r

    L = [
        "# Trading 212 grid\n",
        f"{len(valid_rows)} runs, {len(errors)} errors. Returns and drawdowns in % of the profile budget; "
        "1h = last ~730 days, 1d = ~10 years, continuous; w90 = fresh 90-day windows on 1h.\n",
    ]

    for profile, home in HOME.items():
        variants = sorted({v for (p, v) in by if p == profile and v is not None})
        live = by.get((profile, "live"), {})
        _append_markdown_table_header(
            L,
            f"Profile `{profile}` on its own asset ({home})",
            "| variant | 1h ret | 1h DD | 1h in market % | 1h cycles | 1d ret | 1d DD | w90 mean | w90 pos % |",
            9,
        )
        items = []
        for v in variants:
            r = by[(profile, v)].get(home)
            if not r:
                continue
            w = r.get("w90") or []
            w_rets = [
                safe_float(x[0]) for x in w
                if isinstance(x, (list, tuple)) and len(x) > 0 and safe_float(x[0], default=None) is not None
            ]
            wm = st.fmean(w_rets) if w_rets else None
            wp = (100.0 * st.fmean(ret > 0 for ret in w_rets)) if w_rets else None
            ret_1h = g(r, "1h", "ret")
            ret_1d = g(r, "1d", "ret")
            sort_key = (
                ret_1h is None,
                -(ret_1h if ret_1h is not None else 0.0),
                ret_1d is None,
                -(ret_1d if ret_1d is not None else 0.0),
            )
            items.append((sort_key, v, r, wm, wp))
        for _, v, r, wm, wp in sorted(items, key=lambda x: x[0]):
            L.append(
                f"| {v} | {_fmt_cell(g(r,'1h','ret'))} | {_fmt_cell(g(r,'1h','dd'))} | "
                f"{_fmt_cell(g(r,'1h','expo'))} | {_fmt_cell(g(r,'1h','cyc'))} | "
                f"{_fmt_cell(g(r,'1d','ret'))} | {_fmt_cell(g(r,'1d','dd'))} | "
                f"{_fmt_cell(wm, '+.2f')} | {_fmt_cell(wp, '.0f')} |"
            )

        _append_markdown_table_header(
            L,
            f"Profile `{profile}` on the other assets (generalisation)",
            "| variant | beats live 1h | positive 1h | 1h median ret | 1h median DD | beats live 1d | positive 1d | 1d median ret | 1d median DD |",
            9,
        )
        items = []
        for v in variants:
            d = {a: r for a, r in by[(profile, v)].items() if a != home}
            h = [(a, g(r, "1h", "ret"), g(r, "1h", "dd")) for a, r in d.items() if g(r, "1h", "ret") is not None]
            dd = [(a, g(r, "1d", "ret"), g(r, "1d", "dd")) for a, r in d.items() if g(r, "1d", "ret") is not None]
            if not h and not dd:
                continue

            bl1 = sum(
                (lr := g(live.get(x[0]) or {}, "1h", "ret")) is not None and x[1] is not None and x[1] > (lr + 1e-9)
                for x in h
            )
            bl2 = sum(
                (lr := g(live.get(x[0]) or {}, "1d", "ret")) is not None and x[1] is not None and x[1] > (lr + 1e-9)
                for x in dd
            )
            h_rets = [x[1] for x in h if x[1] is not None]
            h_dds = [x[2] for x in h if x[2] is not None]
            dd_rets = [x[1] for x in dd if x[1] is not None]
            dd_dds = [x[2] for x in dd if x[2] is not None]

            items.append((
                st.median(h_rets) if h_rets else 0.0,
                v,
                len(h),
                bl1,
                sum(x > 0 for x in h_rets),
                st.median(h_dds) if h_dds else 0.0,
                len(dd),
                bl2,
                sum(x > 0 for x in dd_rets),
                st.median(dd_rets) if dd_rets else 0.0,
                st.median(dd_dds) if dd_dds else 0.0,
            ))
        for med, v, n, bl1, pos1, dd1, n2, bl2, pos2, med2, dd2 in sorted(
            items, key=lambda x: (x[2] == 0, -x[0], x[6] == 0, -x[9])
        ):
            L.append(f"| {v} | {bl1}/{n} | {pos1}/{n} | {med:+.1f} | {dd1:.1f} | {bl2}/{n2} | {pos2}/{n2} | {med2:+.1f} | {dd2:.1f} |")

    report_path = os.path.join(exp, "REPORT.md")
    _finalize_markdown_report(
        L,
        errors,
        report_path,
        limit=15,
        label_fn=lambda e: f"{e.get('profile')} {e.get('variant')} {e.get('asset')}",
        write_file=bool(valid_rows) or not os.path.exists(report_path),
    )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rep = generate_report()
    print(rep)
    if "No results found at " in rep or "\n0 runs," in rep:
        sys.exit(1)
