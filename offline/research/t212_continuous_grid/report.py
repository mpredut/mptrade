#!/usr/bin/env python3
"""Summarise results.jsonl from grid.py into REPORT.md."""
import json, os, statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid")
rows = [json.loads(l) for l in open(os.path.join(EXP, "results.jsonl"))]
errors = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}
by = defaultdict(dict)                       # (profile, variant) -> asset -> row
for r in rows:
    by[(r["profile"], r["variant"])][r["asset"]] = r

L = ["# Trading 212 grid\n",
     f"{len(rows)} runs, {len(errors)} errors. Returns and drawdowns in % of the profile budget; "
     "1h = last ~730 days, 1d = ~10 years, continuous; w90 = fresh 90-day windows on 1h.\n"]


def g(r, itv, k):
    return (r.get(itv) or {}).get(k)


for profile, home in HOME.items():
    variants = sorted({v for (p, v) in by if p == profile})
    live = by[(profile, "live")]
    L.append(f"\n## Profile `{profile}` on its own asset ({home})\n")
    L.append("| variant | 1h ret | 1h DD | 1h in market % | 1h cycles | 1d ret | 1d DD | w90 mean | w90 pos % |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    items = []
    for v in variants:
        r = by[(profile, v)].get(home)
        if not r:
            continue
        w = r.get("w90") or []
        wm = st.mean(x[0] for x in w) if w else None
        wp = 100 * sum(x[0] > 0 for x in w) / len(w) if w else None
        items.append((g(r, "1h", "ret") if g(r, "1h", "ret") is not None else -1e9, v, r, wm, wp))
    for _, v, r, wm, wp in sorted(items, key=lambda x: -x[0]):
        L.append(f"| {v} | {g(r,'1h','ret')} | {g(r,'1h','dd')} | {g(r,'1h','expo')} | {g(r,'1h','cyc')} | "
                 f"{g(r,'1d','ret')} | {g(r,'1d','dd')} | "
                 f"{'' if wm is None else f'{wm:+.2f}'} | {'' if wp is None else f'{wp:.0f}'} |")

    L.append(f"\n## Profile `{profile}` on the other assets (generalisation)\n")
    L.append("| variant | beats live 1h | positive 1h | 1h median ret | 1h median DD | beats live 1d | positive 1d | 1d median ret | 1d median DD |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    items = []
    for v in variants:
        d = {a: r for a, r in by[(profile, v)].items() if a != home}
        h = [(a, g(r, "1h", "ret"), g(r, "1h", "dd")) for a, r in d.items() if g(r, "1h", "ret") is not None]
        dd = [(a, g(r, "1d", "ret"), g(r, "1d", "dd")) for a, r in d.items() if g(r, "1d", "ret") is not None]
        if not h:
            continue
        bl1 = sum(x[1] > g(live[x[0]], "1h", "ret") + 1e-9 for x in h)
        bl2 = sum(x[1] > g(live[x[0]], "1d", "ret") + 1e-9 for x in dd)
        items.append((st.median(x[1] for x in h), v, len(h), bl1, sum(x[1] > 0 for x in h),
                      st.median(x[2] for x in h), len(dd), bl2, sum(x[1] > 0 for x in dd),
                      st.median(x[1] for x in dd) if dd else 0, st.median(x[2] for x in dd) if dd else 0))
    for med, v, n, bl1, pos1, dd1, n2, bl2, pos2, med2, dd2 in sorted(items, key=lambda x: -x[0]):
        L.append(f"| {v} | {bl1}/{n} | {pos1}/{n} | {med:+.1f} | {dd1:.1f} | {bl2}/{n2} | {pos2}/{n2} | {med2:+.1f} | {dd2:.1f} |")

if errors:
    L.append("\n## Errors\n")
    for e in errors[:15]:
        L.append(f"- {e['profile']} {e['variant']} {e['asset']}: `{e['error'].strip().splitlines()[-1]}`")
open(os.path.join(EXP, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
