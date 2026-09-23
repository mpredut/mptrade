#!/usr/bin/env python3
"""Aggregate results2.jsonl (wave 2) into REPORT2.md."""
import json, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
rows = [json.loads(l) for l in open(os.path.join(EXP, "results2.jsonl"))]
wave1 = [json.loads(l) for l in open(os.path.join(EXP, "results.jsonl"))]
errors = [r for r in rows if r.get("kind") == "error"]
L = ["# Overnight wave 2: selection, fine overlay grid, fee stress\n"]

# ---------------------------------------------------------------- selection
sel = [r for r in rows if r.get("kind") == "select" and r["steps"]]
L.append("## Walk-forward selection (365-day lookback, 90-day out-of-sample steps)\n")
L.append("Each step ranks the candidate set by trailing return/max(DD,5) and runs the winner on the "
         "next 90 days. `sum` = sum of out-of-sample 90-day returns (% of budget); `beat live` = "
         "steps where the pick beat `live`.\n")
cands = list(sel[0]["steps"][0]["oos"]) if sel else []
L.append("| asset | steps | selector sum | live sum | best static (sum) | selector rank | beat live | picks |")
L.append("|---|---|---|---|---|---|---|---|")
tot = defaultdict(float)
for r in sorted(sel, key=lambda r: r["asset"]):
    s = r["steps"]
    sums = {c: sum(x["oos"][c] for x in s) for c in cands}
    sel_sum = sum(x["pick_ret"] for x in s)
    for c, v in sums.items():
        tot[c] += v
    tot["__selector__"] += sel_sum
    best = max(sums, key=sums.get)
    rank = 1 + sum(v > sel_sum for v in sums.values())
    picks = defaultdict(int)
    for x in s:
        picks[x["pick"]] += 1
    top = ", ".join(f"{k}:{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1])[:3])
    L.append(f"| {r['asset']} | {len(s)} | {sel_sum:+.1f} | {sums['live']:+.1f} | {best} ({sums[best]:+.1f}) | "
             f"{rank}/{len(cands) + 1} | {sum(x['pick_ret'] > x['oos']['live'] for x in s)}/{len(s)} | {top} |")
if sel:
    L.append("\n**Sum over all assets and steps** (higher is better):\n")
    L.append("| strategy | OOS sum % |")
    L.append("|---|---|")
    for c, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        L.append(f"| {'SELECTOR' if c == '__selector__' else c} | {v:+.1f} |")

# ---------------------------------------------------------------- fine grid + fee
views = [r for r in rows if r.get("kind") == "views"]
by = defaultdict(dict)
for r in views:
    by[(r["cfg"], r["fee"])][r["asset"]] = r
for r in wave1:
    if "error" not in r:
        by.setdefault((r["cfg"], "std"), {}).setdefault(r["asset"], r)
assets = sorted({r["asset"] for r in views})


def summ(key):
    d = by[key]
    if not assets or not all(a in d for a in assets):
        return None
    live = by[("live", "std")]
    full = [d[a]["full"]["ret"] for a in assets]
    dd = [d[a]["full"]["dd"] for a in assets]
    w90 = [st.mean(x[0] for x in d[a]["w90"]) for a in assets]
    w90pos = [100 * sum(x[0] > 0 for x in d[a]["w90"]) / len(d[a]["w90"]) for a in assets]
    return {"cfg": key[0], "fee": key[1],
            "beat": sum(d[a]["full"]["ret"] > live[a]["full"]["ret"] for a in assets),
            "full_med": st.median(full), "full_min": min(full), "dd_med": st.median(dd), "dd_max": max(dd),
            "calmar": st.median(f / max(x, 1.0) for f, x in zip(full, dd)),
            "w90": st.mean(w90), "w90pos": st.mean(w90pos)}


def table(items, title):
    L.append(f"\n## {title}\n")
    L.append("| config | fee | beat live (full) | full med % | full min % | DD med | DD max | calmar med | w90 mean | w90 pos% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in items:
        L.append(f"| {s['cfg']} | {s['fee']} | {s['beat']}/{len(assets)} | {s['full_med']:+.1f} | {s['full_min']:+.1f} | "
                 f"{s['dd_med']:.1f} | {s['dd_max']:.1f} | {s['calmar']:+.2f} | {s['w90']:+.2f} | {s['w90pos']:.0f} |")


fine = [s for s in (summ(k) for k in by if k[1] == "std") if s]
table(sorted(fine, key=lambda s: -s["calmar"])[:40], "Fine grid + wave-1 configs, top 40 by median return/DD")
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
        L.append(f"- {e['args']}: `{e['error'].strip().splitlines()[-1]}`")
open(os.path.join(EXP, "REPORT2.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:50]))
