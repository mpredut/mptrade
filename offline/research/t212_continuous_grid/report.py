#!/usr/bin/env python3
"""Summarise results.jsonl from grid.py into REPORT.md."""
from collections import defaultdict
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float

REPO = ROOT
EXP = os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid")
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}


def g(r, itv, k):
    sub = r.get(itv) if isinstance(r, dict) else None
    if not isinstance(sub, dict):
        return None
    val = sub.get(k)
    fval = safe_float(val, default=None)
    if fval is None:
        return None
    return int(fval) if isinstance(val, int) else fval


def generate_report(exp_dir=None):
    exp = exp_dir or os.environ.get("GRID_OUT") or EXP
    results_path = os.path.join(exp, "results.jsonl")
    if not os.path.exists(results_path):
        return f"# Trading 212 grid\n\nNo results found at {results_path}.\n"

    from state_io import read_jsonl_dicts
    rows = read_jsonl_dicts(results_path)

    errors = [r for r in rows if "error" in r]
    valid_rows = [r for r in rows if "error" not in r]

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
        L.append(f"\n## Profile `{profile}` on its own asset ({home})\n")
        L.append("| variant | 1h ret | 1h DD | 1h in market % | 1h cycles | 1d ret | 1d DD | w90 mean | w90 pos % |")
        L.append("|---|---|---|---|---|---|---|---|---|")
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
            sort_key = ret_1h if ret_1h is not None else -1e9
            items.append((sort_key, v, r, wm, wp))
        for _, v, r, wm, wp in sorted(items, key=lambda x: -x[0]):
            L.append(
                f"| {v} | {g(r,'1h','ret')} | {g(r,'1h','dd')} | {g(r,'1h','expo')} | {g(r,'1h','cyc')} | "
                f"{g(r,'1d','ret')} | {g(r,'1d','dd')} | "
                f"{'' if wm is None else f'{wm:+.2f}'} | {'' if wp is None else f'{wp:.0f}'} |"
            )

        L.append(f"\n## Profile `{profile}` on the other assets (generalisation)\n")
        L.append("| variant | beats live 1h | positive 1h | 1h median ret | 1h median DD | beats live 1d | positive 1d | 1d median ret | 1d median DD |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        items = []
        for v in variants:
            d = {a: r for a, r in by[(profile, v)].items() if a != home}
            h = [(a, g(r, "1h", "ret"), g(r, "1h", "dd")) for a, r in d.items() if g(r, "1h", "ret") is not None]
            dd = [(a, g(r, "1d", "ret"), g(r, "1d", "dd")) for a, r in d.items() if g(r, "1d", "ret") is not None]
            if not h and not dd:
                continue

            def _beats_live(asset, ret, itv):
                live_row = live.get(asset) or {}
                live_ret = g(live_row, itv, "ret")
                if live_ret is None or ret is None:
                    return False
                return ret > (live_ret + 1e-9)

            bl1 = sum(_beats_live(x[0], x[1], "1h") for x in h)
            bl2 = sum(_beats_live(x[0], x[1], "1d") for x in dd)
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
            items, key=lambda x: (x[2] == 0, -x[0], -x[9])
        ):
            L.append(f"| {v} | {bl1}/{n} | {pos1}/{n} | {med:+.1f} | {dd1:.1f} | {bl2}/{n2} | {pos2}/{n2} | {med2:+.1f} | {dd2:.1f} |")

    if errors:
        L.append("\n## Errors\n")
        for e in errors[:15]:
            err_line = str(e.get("error") or "").strip().splitlines()
            last_err = err_line[-1] if err_line else "unknown"
            L.append(f"- {e.get('profile')} {e.get('variant')} {e.get('asset')}: `{last_err}`")

    report_content = "\n".join(L) + "\n"
    from state_io import atomic_write_text
    atomic_write_text(os.path.join(exp, "REPORT.md"), report_content)
    return report_content


if __name__ == "__main__":
    rep = generate_report()
    print(rep)
