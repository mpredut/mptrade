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


def _safe_f(val, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def q(xs, p):
    if p is None or isinstance(p, bool):
        return float("nan")
    try:
        pf = float(p)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    if not math.isfinite(pf):
        return float("nan")
    pf = min(1.0, max(0.0, pf))
    clean = sorted(
        float(x) for x in (xs or [])
        if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
    )
    return clean[min(len(clean) - 1, max(0, int(pf * (len(clean) - 1))))] if clean else float("nan")


def win_stats(r, key):
    w = r.get(key) or []
    valid = []
    for x in w:
        if not isinstance(x, (list, tuple)) or len(x) < 3:
            continue
        if x[0] is None or x[2] is None or isinstance(x[0], bool) or isinstance(x[2], bool):
            continue
        try:
            r0 = float(x[0])
            r2 = float(x[2])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(r0) and math.isfinite(r2):
            valid.append((r0, r2))
    rets = [x[0] for x in valid]
    if not rets:
        return {"mean": float("nan"), "pos": float("nan"), "p10": float("nan"),
                "down": float("nan"), "up": float("nan")}
    down_vals = [x[0] for x in valid if x[1] < -10]
    up_vals = [x[0] for x in valid if x[1] > 10]
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
        if not all(a in by[c] and isinstance(by[c][a].get("full"), dict) for a in assets):
            continue
        live = by.get("live", {})
        full = [_safe_f(by[c][a]["full"].get("ret"), 0.0) for a in assets]
        dd = [_safe_f(by[c][a]["full"].get("dd"), 0.0) for a in assets]
        expo = [_safe_f(by[c][a]["full"].get("expo"), 0.0) for a in assets]
        beat_full = sum(
            _safe_f(by[c][a]["full"].get("ret"), 0.0)
            > _safe_f(
                (live.get(a, {}).get("full") if isinstance(live.get(a, {}).get("full"), dict) else {}).get("ret"),
                float("-inf"),
            ) + 1e-9
            for a in assets
        )
        w90 = [win_stats(by[c][a], "w90") for a in assets]
        w30 = [win_stats(by[c][a], "w30") for a in assets]
        beat_w90 = sum(
            (win_stats(by[c][a], "w90")["mean"] > win_stats(live.get(a, {}), "w90")["mean"] + 1e-9)
            if math.isfinite(win_stats(by[c][a], "w90")["mean"]) and math.isfinite(win_stats(live.get(a, {}), "w90")["mean"])
            else False
            for a in assets
        )
        years = [
            _safe_f(y.get("ret"), float("nan"))
            for a in assets
            for y in (by[c][a].get("year") or {}).values()
            if isinstance(y, dict) and math.isfinite(_safe_f(y.get("ret"), float("nan")))
        ]
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
        live_full = by.get("live", {}).get(a, {}).get("full")
        bh = _safe_f(live_full.get("bh") if isinstance(live_full, dict) else 0.0, 0.0)
        cells = []
        for c in key_cfgs:
            full_d = by.get(c, {}).get(a, {}).get("full")
            if isinstance(full_d, dict) and "ret" in full_d and "dd" in full_d:
                cells.append(f"{_safe_f(full_d.get('ret'), 0.0):+.1f} ({_safe_f(full_d.get('dd'), 0.0):.0f})")
            else:
                cells.append("-")
        L.append(f"| {a} | {bh:+.0f} | " + " | ".join(cells) + " |")

    L.append("\n## Per-year continuous return %, live vs ov650_t8\n")
    L.append("| asset | year | B&H % | live | ov650_t8 | old_live |")
    L.append("|---|---|---|---|---|---|")
    for a in assets:
        live_years = by.get("live", {}).get(a, {}).get("year")
        if not isinstance(live_years, dict):
            continue
        for y, r in sorted(live_years.items()):
            if not isinstance(r, dict):
                continue
            o_yr = by.get("ov650_t8", {}).get(a, {}).get("year")
            p_yr = by.get("old_live_pre0923", {}).get(a, {}).get("year")
            o = o_yr.get(y) if isinstance(o_yr, dict) else None
            p = p_yr.get(y) if isinstance(p_yr, dict) else None
            r_bh = _safe_f(r.get("bh"), 0.0)
            r_ret = _safe_f(r.get("ret"), 0.0)
            if isinstance(o, dict) and isinstance(p, dict) and "ret" in o and "ret" in p:
                L.append(
                    f"| {a} | {y} | {r_bh:+.0f} | {r_ret:+.1f} | "
                    f"{_safe_f(o.get('ret'), 0.0):+.1f} | {_safe_f(p.get('ret'), 0.0):+.1f} |"
                )
            else:
                L.append(f"| {a} | {y} | {r_bh:+.0f} | {r_ret:+.1f} | - | - |")

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
