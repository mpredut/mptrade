#!/usr/bin/env python3
"""Aggregate results.jsonl from grid.py into REPORT.md (plain markdown tables)."""
import json, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
rows = [json.loads(l) for l in open(os.path.join(EXP, "results.jsonl"))]
errors = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]
by = defaultdict(dict)
for r in rows:
    by[r["cfg"]][r["asset"]] = r
assets = sorted({r["asset"] for r in rows})
cfgs = list(by)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * (len(xs) - 1))))] if xs else float("nan")


def win_stats(r, key):
    w = r[key]
    rets = [x[0] for x in w]
    return {"mean": st.mean(rets), "pos": 100 * sum(x > 0 for x in rets) / len(rets),
            "p10": q(rets, 0.10),
            "down": st.mean([x[0] for x in w if x[2] < -10] or [float("nan")]),
            "up": st.mean([x[0] for x in w if x[2] > 10] or [float("nan")])}


summary = []
for c in cfgs:
    if not all(a in by[c] for a in assets):
        continue
    live = by["live"]
    full = [by[c][a]["full"]["ret"] for a in assets]
    dd = [by[c][a]["full"]["dd"] for a in assets]
    expo = [by[c][a]["full"]["expo"] for a in assets]
    beat_full = sum(by[c][a]["full"]["ret"] > live[a]["full"]["ret"] + 1e-9 for a in assets)
    w90 = [win_stats(by[c][a], "w90") for a in assets]
    w30 = [win_stats(by[c][a], "w30") for a in assets]
    beat_w90 = sum(win_stats(by[c][a], "w90")["mean"] > win_stats(live[a], "w90")["mean"] + 1e-9
                   for a in assets)
    years = [y["ret"] for a in assets for y in by[c][a]["year"].values()]
    summary.append({
        "cfg": c, "beat_full": beat_full, "beat_w90": beat_w90,
        "full_med": st.median(full), "full_mean": st.mean(full), "full_min": min(full),
        "dd_med": st.median(dd), "dd_max": max(dd), "expo_med": st.median(expo),
        "w90_mean": st.mean(x["mean"] for x in w90), "w90_pos": st.mean(x["pos"] for x in w90),
        "w90_p10": st.mean(x["p10"] for x in w90), "w90_down": st.mean(x["down"] for x in w90),
        "w30_mean": st.mean(x["mean"] for x in w30), "w30_pos": st.mean(x["pos"] for x in w30),
        "yr_pos": 100 * sum(y > 0 for y in years) / len(years), "yr_worst": min(years),
        "yr_med": st.median(years),
        # Return per unit of drawdown on the continuous run, median over assets.
        "calmar_med": st.median(f / max(d, 1.0) for f, d in zip(full, dd)),
    })

L = []
L.append("# Overnight Kraken spot-DCA grid\n")
L.append(f"{len(rows)} runs, {len(assets)} assets ({', '.join(assets)}), {len(summary)} complete configs, "
         f"{len(errors)} errors. 4h bars, fee 0.26%/leg, returns as % of the 3900 USD cycle budget.\n")
L.append("Views: **full** = one continuous run over the whole history (state carries across cycles, "
         "as live); **year** = continuous per calendar year; **w90/w30** = fresh-state rolling "
         "windows of 90/30 days, step 30 days.\n")
L.append("`beat_full`/`beat_w90` = on how many assets the config beats `live`.\n")


def table(items, title):
    L.append(f"\n## {title}\n")
    L.append("| config | beat_full | beat_w90 | full med % | full min % | full DD med | DD max | "
             "expo med % | calmar med | w90 mean | w90 pos% | w90 p10 | w90 down | w30 mean | "
             "year pos% | worst year | year med |")
    L.append("|" + "---|" * 17)
    for s in items:
        L.append(f"| {s['cfg']} | {s['beat_full']}/{len(assets)} | {s['beat_w90']}/{len(assets)} | "
                 f"{s['full_med']:+.1f} | {s['full_min']:+.1f} | {s['dd_med']:.1f} | {s['dd_max']:.1f} | "
                 f"{s['expo_med']:.0f} | {s['calmar_med']:+.2f} | {s['w90_mean']:+.2f} | {s['w90_pos']:.0f} | "
                 f"{s['w90_p10']:+.2f} | {s['w90_down']:+.2f} | {s['w30_mean']:+.2f} | "
                 f"{s['yr_pos']:.0f} | {s['yr_worst']:+.1f} | {s['yr_med']:+.1f} |")


table(sorted(summary, key=lambda s: -s["calmar_med"]), "All configs, ranked by median full-run return/drawdown")
table([s for s in summary if s["cfg"] in ("live", "old_live_pre0923") or s["cfg"].startswith("abl_")],
      "23 Sep promotion: live vs pre-promotion vs one-change-reverted ablations")

L.append("\n## Continuous full-run return % per asset (DD % in brackets)\n")
key_cfgs = [c for c in ("live", "old_live_pre0923", "ov350_t6", "ov650_t8", "ov1000_t8", "ov2000_t8",
                        "ov650_t8_sl12.5", "peak2.2", "reentry0", "sl0", "tp_classic") if c in by]
L.append("| asset | buy&hold % | " + " | ".join(key_cfgs) + " |")
L.append("|" + "---|" * (len(key_cfgs) + 2))
for a in assets:
    bh = by["live"][a]["full"]["bh"]
    L.append(f"| {a} | {bh:+.0f} | " + " | ".join(
        f"{by[c][a]['full']['ret']:+.1f} ({by[c][a]['full']['dd']:.0f})" if a in by[c] else "-"
        for c in key_cfgs) + " |")

L.append("\n## Per-year continuous return %, live vs ov650_t8\n")
L.append("| asset | year | B&H % | live | ov650_t8 | old_live |")
L.append("|---|---|---|---|---|---|")
for a in assets:
    for y, r in sorted(by["live"][a]["year"].items()):
        o = by.get("ov650_t8", {}).get(a, {}).get("year", {}).get(y)
        p = by.get("old_live_pre0923", {}).get(a, {}).get("year", {}).get(y)
        L.append(f"| {a} | {y} | {r['bh']:+.0f} | {r['ret']:+.1f} | "
                 f"{o['ret']:+.1f} | {p['ret']:+.1f} |" if o and p else f"| {a} | {y} | {r['bh']:+.0f} | {r['ret']:+.1f} | - | - |")

if errors:
    L.append("\n## Errors\n")
    for e in errors[:20]:
        L.append(f"- {e['asset']} {e['cfg']}: `{e['error'].strip().splitlines()[-1]}`")

open(os.path.join(EXP, "REPORT.md"), "w").write("\n".join(L) + "\n")
json.dump(summary, open(os.path.join(EXP, "summary.json"), "w"), indent=1)
print("\n".join(L[:60]))
