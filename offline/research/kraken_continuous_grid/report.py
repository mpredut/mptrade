#!/usr/bin/env python3
"""Aggregate results.jsonl from grid.py into REPORT.md (plain markdown tables)."""
import json
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * (len(xs) - 1))))] if xs else float("nan")


def win_stats(r, key):
    w = r.get(key) or []
    valid = [
        x for x in w
        if isinstance(x, (list, tuple)) and len(x) >= 3 and x[0] is not None and x[2] is not None
    ]
    rets = [float(x[0]) for x in valid]
    if not rets:
        return {"mean": float("nan"), "pos": float("nan"), "p10": float("nan"),
                "down": float("nan"), "up": float("nan")}
    down_vals = [float(x[0]) for x in valid if float(x[2]) < -10]
    up_vals = [float(x[0]) for x in valid if float(x[2]) > 10]
    return {"mean": st.mean(rets), "pos": 100.0 * sum(x > 0 for x in rets) / len(rets),
            "p10": q(rets, 0.10),
            "down": st.mean(down_vals) if down_vals else float("nan"),
            "up": st.mean(up_vals) if up_vals else float("nan")}


def generate_report(exp_dir: str | None = None) -> list[str]:
    exp = exp_dir or os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
    os.makedirs(exp, exist_ok=True)
    results_path = os.path.join(exp, "results.jsonl")
    rows = _load_jsonl_dicts(results_path)
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r and "cfg" in r and "asset" in r]
    by = defaultdict(dict)
    for r in rows:
        by[r["cfg"]][r["asset"]] = r
    assets = sorted({r["asset"] for r in rows})
    cfgs = list(by)

    summary = []
    for c in cfgs:
        if not all(a in by[c] for a in assets):
            continue
        live = by.get("live", {})
        full = [by[c][a]["full"]["ret"] for a in assets]
        dd = [by[c][a]["full"]["dd"] for a in assets]
        expo = [by[c][a]["full"]["expo"] for a in assets]
        beat_full = sum(by[c][a]["full"]["ret"] > live.get(a, {}).get("full", {}).get("ret", float("-inf")) + 1e-9 for a in assets)
        w90 = [win_stats(by[c][a], "w90") for a in assets]
        w30 = [win_stats(by[c][a], "w30") for a in assets]
        beat_w90 = sum(
            (win_stats(by[c][a], "w90")["mean"] > win_stats(live.get(a, {}), "w90")["mean"] + 1e-9)
            if math.isfinite(win_stats(by[c][a], "w90")["mean"]) and math.isfinite(win_stats(live.get(a, {}), "w90")["mean"])
            else False
            for a in assets
        )
        years = [y["ret"] for a in assets for y in by[c][a].get("year", {}).values() if "ret" in y]
        w90_means = [x["mean"] for x in w90 if math.isfinite(x["mean"])]
        w90_pos = [x["pos"] for x in w90 if math.isfinite(x["pos"])]
        w90_p10 = [x["p10"] for x in w90 if math.isfinite(x["p10"])]
        w90_down = [x["down"] for x in w90 if math.isfinite(x["down"])]
        w30_means = [x["mean"] for x in w30 if math.isfinite(x["mean"])]
        w30_pos = [x["pos"] for x in w30 if math.isfinite(x["pos"])]
        summary.append({
            "cfg": c, "beat_full": beat_full, "beat_w90": beat_w90,
            "full_med": st.median(full) if full else 0.0,
            "full_mean": st.mean(full) if full else 0.0,
            "full_min": min(full) if full else 0.0,
            "dd_med": st.median(dd) if dd else 0.0,
            "dd_max": max(dd) if dd else 0.0,
            "expo_med": st.median(expo) if expo else 0.0,
            "w90_mean": st.mean(w90_means) if w90_means else float("nan"),
            "w90_pos": st.mean(w90_pos) if w90_pos else float("nan"),
            "w90_p10": st.mean(w90_p10) if w90_p10 else float("nan"),
            "w90_down": st.mean(w90_down) if w90_down else float("nan"),
            "w30_mean": st.mean(w30_means) if w30_means else float("nan"),
            "w30_pos": st.mean(w30_pos) if w30_pos else float("nan"),
            "yr_pos": (100.0 * sum(y > 0 for y in years) / len(years)) if years else float("nan"),
            "yr_worst": min(years) if years else float("nan"),
            "yr_med": st.median(years) if years else float("nan"),
            "calmar_med": st.median(f / max(d, 1.0) for f, d in zip(full, dd)) if full else 0.0,
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
        bh = by.get("live", {}).get(a, {}).get("full", {}).get("bh", 0.0)
        L.append(f"| {a} | {bh:+.0f} | " + " | ".join(
            f"{by[c][a]['full']['ret']:+.1f} ({by[c][a]['full']['dd']:.0f})" if (a in by.get(c, {}) and "full" in by[c][a]) else "-"
            for c in key_cfgs) + " |")

    L.append("\n## Per-year continuous return %, live vs ov650_t8\n")
    L.append("| asset | year | B&H % | live | ov650_t8 | old_live |")
    L.append("|---|---|---|---|---|---|")
    for a in assets:
        live_years = by.get("live", {}).get(a, {}).get("year", {})
        for y, r in sorted(live_years.items()):
            o = by.get("ov650_t8", {}).get(a, {}).get("year", {}).get(y)
            p = by.get("old_live_pre0923", {}).get(a, {}).get("year", {}).get(y)
            L.append(f"| {a} | {y} | {r.get('bh', 0):+.0f} | {r.get('ret', 0):+.1f} | "
                     f"{o['ret']:+.1f} | {p['ret']:+.1f} |" if (o and p and "ret" in o and "ret" in p) else f"| {a} | {y} | {r.get('bh', 0):+.0f} | {r.get('ret', 0):+.1f} | - | - |")

    if errors:
        L.append("\n## Errors\n")
        for e in errors[:20]:
            err_lines = str(e.get("error") or "").strip().splitlines()
            last_err = err_lines[-1] if err_lines else "unknown"
            L.append(f"- {e.get('asset', '?')} {e.get('cfg', '?')}: `{last_err}`")

    with open(os.path.join(exp, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(exp, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines[:60]))
