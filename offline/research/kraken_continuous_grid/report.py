#!/usr/bin/env python3
"""Aggregate results.jsonl from grid.py into REPORT.md (plain markdown tables)."""
import math
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import load_local_module, optional_env, safe_float, safe_nonneg_int
from state_io import atomic_write_json, atomic_write_text, read_jsonl_dicts


def q(xs, p):
    pf = safe_float(p, default=None)
    if pf is None:
        return math.nan
    pf = min(1.0, max(0.0, pf))
    clean = sorted(
        f for x in (xs or [])
        if (f := safe_float(x, default=None)) is not None
    )
    if not clean:
        return math.nan
    idx = min(len(clean) - 1, max(0, safe_nonneg_int(math.floor(pf * (len(clean) - 1)), default=0)))
    return clean[idx]


def win_stats(r, key):
    w = r.get(key) or []
    valid = []
    for x in w:
        if not isinstance(x, (list, tuple)) or len(x) < 3:
            continue
        r0 = safe_float(x[0], default=None)
        r2 = safe_float(x[2], default=None)
        if r0 is not None and r2 is not None:
            valid.append((r0, r2))
    rets = [x[0] for x in valid]
    if not rets:
        return {"mean": math.nan, "pos": math.nan, "p10": math.nan,
                "down": math.nan, "up": math.nan}
    down_vals = [x[0] for x in valid if x[1] < -10]
    up_vals = [x[0] for x in valid if x[1] > 10]
    return {"mean": st.fmean(rets), "pos": 100.0 * st.fmean(x > 0 for x in rets),
            "p10": q(rets, 0.10),
            "down": st.fmean(down_vals) if down_vals else math.nan,
            "up": st.fmean(up_vals) if up_vals else math.nan}


def _fmt_num(val, spec: str | None = None) -> str:
    if val is None or isinstance(val, bool):
        return "-"
    fv = safe_float(val, default=None)
    if fv is None:
        return "-"
    if spec is not None:
        return format(fv, spec)
    return str(val if isinstance(val, int) else fv)


def _resolve_exp_dir(exp_dir: str | None = None) -> str:
    exp = exp_dir or optional_env("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
    os.makedirs(exp, exist_ok=True)
    return exp


def _format_errors_section(errors: list[dict], *, label_fn=None, limit: int = 20) -> list[str]:
    if not errors:
        return []
    lines = ["\n## Errors\n"]
    for e in errors[:limit]:
        err_lines = str(e.get("error") or "").strip().splitlines()
        last_err = err_lines[-1] if err_lines else "unknown"
        label = label_fn(e) if label_fn is not None else f"{e.get('asset', '?')} {e.get('cfg', '?')}"
        lines.append(f"- {label}: `{last_err}`")
    return lines


def _append_markdown_table_header(
    lines: list[str], title: str | None, header_row: str, num_cols: int
) -> None:
    """Append an optional markdown H2 title, a table header row, and its column separator row."""
    if title:
        lines.append(f"\n## {title}\n")
    lines.append(header_row)
    lines.append("|" + "---|" * max(1, int(num_cols)))


def _finalize_markdown_report(
    lines: list[str],
    errors: list[dict],
    out_path: str,
    *,
    label_fn=None,
    limit: int = 20,
) -> list[str]:
    """Append formatted error section, write markdown report to ``out_path``, and return ``lines``."""
    lines.extend(_format_errors_section(errors, label_fn=label_fn, limit=limit))
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return lines


def _risk_adjusted_score(ret_pct: float, dd_pct: float, dd_floor: float = 5.0) -> float:
    """Monotone risk-adjusted return score: penalises larger drawdowns even when return < 0."""
    r = safe_float(ret_pct, 0.0)
    d = max(safe_float(dd_pct, 0.0), dd_floor)
    if r >= 0.0:
        return r / d
    return (r * d) / (dd_floor * dd_floor)


def _is_error_row(r: dict) -> bool:
    return "error" in r or r.get("kind") == "error"


def _matches_known_spec(
    r: dict,
    spec_map: dict | None,
    *,
    spec_key: str | tuple[str, ...] = "cfg",
) -> bool:
    if not spec_map:
        return True
    cfg_name = (
        tuple(r.get(k) for k in spec_key)
        if isinstance(spec_key, tuple)
        else r.get(spec_key)
    )
    if cfg_name not in spec_map:
        return True
    params, peak = spec_map[cfg_name]
    if "params_repr" in r and r.get("params_repr") != repr(params):
        return False
    if "peak" in r and bool(r.get("peak")) != bool(peak):
        return False
    return True


def load_grid_results_partitioned(
    results_path: str,
    required_str_keys: tuple[str, ...],
    *,
    spec_map: dict | None = None,
    spec_key: str | tuple[str, ...] = "cfg",
) -> tuple[list[dict], list[dict]]:
    """Load JSONL grid rows and partition into ``(valid_rows, error_rows)``."""
    rows = read_jsonl_dicts(results_path)
    errors = [r for r in rows if _is_error_row(r)]
    valid_rows = [
        r
        for r in rows
        if not _is_error_row(r)
        and all(isinstance(r.get(k), str) and r[k].strip() for k in required_str_keys)
        and _matches_known_spec(r, spec_map, spec_key=spec_key)
    ]
    return valid_rows, errors


def _load_grid_results_by_cfg(
    results_path: str, *, spec_map: dict | None = None
) -> tuple[list[dict], list[dict], dict[str, dict[str, dict]], list[str]]:
    valid_rows, errors = load_grid_results_partitioned(results_path, ("cfg", "asset"), spec_map=spec_map)
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in valid_rows:
        by[r["cfg"]][r["asset"]] = r
    assets = sorted({r["asset"] for r in valid_rows})
    return valid_rows, errors, by, assets


def _rank_key(s: dict, calmar_key: str = "calmar_med") -> tuple[bool, float, float]:
    full_med = safe_float(s.get("full_med"), default=-math.inf)
    calmar_med = safe_float(s.get(calmar_key), default=-math.inf)
    dd_med = safe_float(s.get("dd_med"), default=math.inf)
    if full_med >= 0.0:
        return (True, calmar_med, -dd_med)
    return (False, full_med, -dd_med)


def generate_report(exp_dir: str | None = None) -> list[str]:
    exp = _resolve_exp_dir(exp_dir)
    results_path = os.path.join(exp, "results.jsonl")
    grid_mod = sys.modules.get("offline.research.kraken_continuous_grid.grid") or load_local_module(
        os.path.dirname(os.path.abspath(__file__)),
        "grid",
        module_key="offline.research.kraken_continuous_grid.grid",
    )
    rows, errors, by, assets = _load_grid_results_by_cfg(
        results_path, spec_map=getattr(grid_mod, "CONFIGS", None)
    )
    cfgs = list(by)

    summary = []
    for c in cfgs:
        if not assets or not all(
            a in by[c]
            and isinstance(by[c][a].get("full"), dict)
            and safe_float(by[c][a]["full"].get("ret"), default=None) is not None
            and safe_float(by[c][a]["full"].get("dd"), default=None) is not None
            for a in assets
        ):
            continue
        live = by.get("live", {})
        full = [safe_float(by[c][a]["full"].get("ret"), 0.0) for a in assets]
        dd = [safe_float(by[c][a]["full"].get("dd"), 0.0) for a in assets]
        expo = [safe_float(by[c][a]["full"].get("expo"), 0.0) for a in assets]
        beat_full = sum(
            (c_ret > live_ret + 1e-9)
            if (c_ret := safe_float(by[c][a]["full"].get("ret"), default=None)) is not None
            and isinstance(live.get(a, {}).get("full"), dict)
            and (live_ret := safe_float(live[a]["full"].get("ret"), default=None)) is not None
            else False
            for a in assets
        )
        w90 = [win_stats(by[c][a], "w90") for a in assets]
        w30 = [win_stats(by[c][a], "w30") for a in assets]
        beat_w90 = sum(
            (w90_m > live_w90_m + 1e-9)
            if (w90_m := safe_float(w90_item.get("mean"), default=None)) is not None
            and (live_w90_m := safe_float(win_stats(live.get(a, {}), "w90").get("mean"), default=None)) is not None
            else False
            for a, w90_item in zip(assets, w90)
        )
        years = [
            yr_ret
            for a in assets
            for y in (by[c][a].get("year") or {}).values()
            if isinstance(y, dict) and (yr_ret := safe_float(y.get("ret"), default=None)) is not None
        ]
        w90_means = [m for x in w90 if (m := safe_float(x.get("mean"), default=None)) is not None]
        w90_pos = [m for x in w90 if (m := safe_float(x.get("pos"), default=None)) is not None]
        w90_p10 = [m for x in w90 if (m := safe_float(x.get("p10"), default=None)) is not None]
        w90_down = [m for x in w90 if (m := safe_float(x.get("down"), default=None)) is not None]
        w30_means = [m for x in w30 if (m := safe_float(x.get("mean"), default=None)) is not None]
        w30_pos = [m for x in w30 if (m := safe_float(x.get("pos"), default=None)) is not None]
        summary.append({
            "cfg": c, "beat_full": beat_full, "beat_w90": beat_w90,
            "full_med": st.median(full) if full else 0.0,
            "full_mean": st.mean(full) if full else 0.0,
            "full_min": min(full) if full else 0.0,
            "dd_med": st.median(dd) if dd else 0.0,
            "dd_max": max(dd) if dd else 0.0,
            "expo_med": st.median(expo) if expo else 0.0,
            "w90_mean": st.mean(w90_means) if w90_means else None,
            "w90_pos": st.mean(w90_pos) if w90_pos else None,
            "w90_p10": st.mean(w90_p10) if w90_p10 else None,
            "w90_down": st.mean(w90_down) if w90_down else None,
            "w30_mean": st.mean(w30_means) if w30_means else None,
            "w30_pos": st.mean(w30_pos) if w30_pos else None,
            "yr_pos": (100.0 * st.fmean(y > 0 for y in years)) if years else None,
            "yr_worst": min(years) if years else None,
            "yr_med": st.median(years) if years else None,
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
        _append_markdown_table_header(
            L,
            title,
            "| config | beat_full | beat_w90 | full med % | full min % | full DD med | DD max | "
            "expo med % | calmar med | w90 mean | w90 pos% | w90 p10 | w90 down | w30 mean | "
            "year pos% | worst year | year med |",
            17,
        )
        for s in items:
            L.append(
                f"| {s['cfg']} | {s['beat_full']}/{len(assets)} | {s['beat_w90']}/{len(assets)} | "
                f"{_fmt_num(s['full_med'], '+.1f')} | {_fmt_num(s['full_min'], '+.1f')} | "
                f"{_fmt_num(s['dd_med'], '.1f')} | {_fmt_num(s['dd_max'], '.1f')} | "
                f"{_fmt_num(s['expo_med'], '.0f')} | {_fmt_num(s['calmar_med'], '+.2f')} | "
                f"{_fmt_num(s['w90_mean'], '+.2f')} | {_fmt_num(s['w90_pos'], '.0f')} | "
                f"{_fmt_num(s['w90_p10'], '+.2f')} | {_fmt_num(s['w90_down'], '+.2f')} | "
                f"{_fmt_num(s['w30_mean'], '+.2f')} | {_fmt_num(s['yr_pos'], '.0f')} | "
                f"{_fmt_num(s['yr_worst'], '+.1f')} | {_fmt_num(s['yr_med'], '+.1f')} |"
            )

    table(sorted(summary, key=_rank_key, reverse=True), "All configs, ranked by median full-run return/drawdown")
    table([s for s in summary if s["cfg"] in ("live", "old_live_pre0923") or s["cfg"].startswith("abl_")],
          "23 Sep promotion: live vs pre-promotion vs one-change-reverted ablations")

    L.append("\n## Continuous full-run return % per asset (DD % in brackets)\n")
    key_cfgs = [c for c in ("live", "old_live_pre0923", "ov350_t6", "ov650_t8", "ov1000_t8", "ov2000_t8",
                            "ov650_t8_sl12.5", "peak2.2", "reentry0", "sl0", "tp_classic") if c in by]
    L.append("| asset | buy&hold % | " + " | ".join(key_cfgs) + " |")
    L.append("|" + "---|" * (len(key_cfgs) + 2))
    for a in assets:
        bh_val = None
        for cfg_name in ("live", *cfgs):
            cand_full = by.get(cfg_name, {}).get(a, {}).get("full")
            if isinstance(cand_full, dict):
                bh_val = safe_float(cand_full.get("bh"), default=None)
                if bh_val is not None:
                    break
        bh = bh_val if bh_val is not None else 0.0
        cells = []
        for c in key_cfgs:
            full_d = by.get(c, {}).get(a, {}).get("full")
            ret_v = safe_float(full_d.get("ret"), default=None) if isinstance(full_d, dict) else None
            dd_v = safe_float(full_d.get("dd"), default=None) if isinstance(full_d, dict) else None
            if ret_v is not None and dd_v is not None:
                cells.append(f"{ret_v:+.1f} ({dd_v:.0f})")
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
            r_bh = safe_float(r.get("bh"), 0.0)
            r_ret = safe_float(r.get("ret"), 0.0)
            if isinstance(o, dict) and isinstance(p, dict) and "ret" in o and "ret" in p:
                L.append(
                    f"| {a} | {y} | {r_bh:+.0f} | {r_ret:+.1f} | "
                    f"{safe_float(o.get('ret'), 0.0):+.1f} | {safe_float(p.get('ret'), 0.0):+.1f} |"
                )
            else:
                L.append(f"| {a} | {y} | {r_bh:+.0f} | {r_ret:+.1f} | - | - |")

    _finalize_markdown_report(L, errors, os.path.join(exp, "REPORT.md"))
    atomic_write_json(os.path.join(exp, "summary.json"), summary, indent=1)
    return L


if __name__ == "__main__":
    lines = generate_report()
    print("\n".join(lines[:60]))
