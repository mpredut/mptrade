"""Pure standalone Unified NAV history tracking, drawdown analytics, and daily NAV guard.

Single Source of Truth (SSOT) for portfolio-wide multi-venue NAV curve persistence, cashflow-adjusted
drawdown calculation (daily, rolling 24h, 7d/30d MaxDD, 30d daily Sharpe), and asymmetric daily NAV
drawdown protection strictly for autonomous ``by=AI`` BUY intents.

Core evaluation functions operate purely in memory (0 ms, zero network/broker/disk I/O) so they can
be invoked independently from the gateway, offline attribution CLI, backtests, or unit tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from typing import Any, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    normalize_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    NAV_HISTORY_STATE_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    resolve_cachedb_filepath,
)
from state_io import (
    generation_file_lock,
    read_json_dict,
    save_json_state,
)
from .portfolio_valuation import compute_venue_nav_breakdown

DEFAULT_DAILY_NAV_DD_MODE: str = "shadow"
DEFAULT_DAILY_NAV_DD_LIMIT_PCT: float = 5.0
DEFAULT_NAV_SAMPLE_INTERVAL_SEC: float = 600.0
DEFAULT_NAV_RETENTION_DAYS: float = 90.0
DEFAULT_CASHFLOW_JUMP_PCT: float = 4.0
_CASHFLOW_CASH_SHARE_MIN: float = 0.70
_VALID_NAV_DD_MODES: frozenset[str] = frozenset({"off", "shadow", "enforce"})


def normalize_nav_dd_mode(raw: Any, default: str = DEFAULT_DAILY_NAV_DD_MODE) -> str:
    txt = (clean_scalar_str(raw) or "").lower()
    if txt in _VALID_NAV_DD_MODES:
        return txt
    return default if default in _VALID_NAV_DD_MODES else DEFAULT_DAILY_NAV_DD_MODE


def _utc_day_str(ts_s: float) -> str:
    return datetime.fromtimestamp(max(0.0, ts_s), timezone.utc).strftime("%Y-%m-%d")


def _extract_point_cash_usd(point: Mapping[str, Any]) -> float:
    return (
        safe_nonneg_float(point.get("free_cash_usd"), default=0.0)
        + safe_nonneg_float(point.get("locked_quote_usd"), default=0.0)
    )


def _normalize_venues_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in sorted(raw.items()):
        vk = clean_scalar_str(k)
        vv = safe_pos_float(v, default=None)
        if vk and vv is not None:
            out[vk.lower()] = round(vv, 2)
    return out


def _normalize_nav_point(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    ts_s = safe_pos_float(raw.get("ts"), default=None)
    nav_usd = safe_pos_float(raw.get("nav_usd"), default=None)
    if ts_s is None or nav_usd is None:
        return None
    pt: dict[str, Any] = {
        "ts": round(ts_s, 3),
        "nav_usd": round(nav_usd, 2),
        "invested_usd": round(safe_nonneg_float(raw.get("invested_usd"), default=0.0), 2),
        "free_cash_usd": round(safe_nonneg_float(raw.get("free_cash_usd"), default=0.0), 2),
        "locked_quote_usd": round(safe_nonneg_float(raw.get("locked_quote_usd"), default=0.0), 2),
    }
    if venues := _normalize_venues_map(raw.get("venues_usd")):
        pt["venues_usd"] = venues
    if _parse_bool_flag(raw.get("external_cashflow"), False):
        pt["external_cashflow"] = True
        if (cf_d := safe_float(raw.get("cashflow_delta_usd"), default=None)) is not None:
            pt["cashflow_delta_usd"] = round(cf_d, 2)
    if _parse_bool_flag(raw.get("venue_set_changed"), False):
        pt["venue_set_changed"] = True
    return pt


def _detect_non_market_step(
    prev: Mapping[str, Any],
    cur: Mapping[str, Any],
    *,
    cashflow_jump_pct: float = DEFAULT_CASHFLOW_JUMP_PCT,
) -> tuple[bool, bool, float]:
    """Return ``(is_external_cashflow, is_venue_set_change, delta_nav_usd)`` for a single NAV interval."""
    prev_nav = safe_pos_float(prev.get("nav_usd"), default=0.0) or 0.0
    cur_nav = safe_pos_float(cur.get("nav_usd"), default=0.0) or 0.0
    if prev_nav <= 0.0 or cur_nav <= 0.0:
        return False, False, 0.0
    delta_nav = cur_nav - prev_nav
    jump_pct = (abs(delta_nav) / prev_nav) * 100.0
    thresh = safe_pos_float(cashflow_jump_pct, default=DEFAULT_CASHFLOW_JUMP_PCT) or DEFAULT_CASHFLOW_JUMP_PCT
    if jump_pct < thresh:
        return False, False, delta_nav

    prev_v = set((prev.get("venues_usd") or {}).keys()) if isinstance(prev.get("venues_usd"), Mapping) else set()
    cur_v = set((cur.get("venues_usd") or {}).keys()) if isinstance(cur.get("venues_usd"), Mapping) else set()
    if prev_v and cur_v and prev_v != cur_v:
        return False, True, delta_nav

    delta_cash = _extract_point_cash_usd(cur) - _extract_point_cash_usd(prev)
    if (delta_cash * delta_nav) > 0.0 and abs(delta_cash) >= _CASHFLOW_CASH_SHARE_MIN * abs(delta_nav):
        return True, False, delta_nav
    return False, False, delta_nav


def build_nav_history_point(
    venue_balances_state: Any,
    *,
    live_prices: Any = None,
    now_ts: Optional[float] = None,
    scope: str = "all",
) -> Optional[dict[str, Any]]:
    """Build a canonical multi-venue NAV point from ``venue_balances_state`` via ``compute_venue_nav_breakdown``."""
    if not isinstance(venue_balances_state, Mapping) or not venue_balances_state:
        return None
    nav = compute_venue_nav_breakdown(
        venue_balances_state=dict(venue_balances_state),
        live_prices=live_prices,
        scope=scope,
    )
    total_usd = safe_pos_float(nav.get("portfolio_total_usd"), default=None)
    if total_usd is None:
        return None
    eff_now = resolve_now_ts(now_ts)
    pt: dict[str, Any] = {
        "ts": round(eff_now, 3),
        "nav_usd": round(total_usd, 2),
        "invested_usd": round(safe_nonneg_float(nav.get("invested_usd"), default=0.0), 2),
        "free_cash_usd": round(safe_nonneg_float(nav.get("free_cash_usd"), default=0.0), 2),
        "locked_quote_usd": round(safe_nonneg_float(nav.get("locked_quote_usd"), default=0.0), 2),
    }
    if venues := _normalize_venues_map(nav.get("venues_usd")):
        pt["venues_usd"] = venues
    return pt


def append_nav_history_point(
    history: list[dict[str, Any]],
    point: Mapping[str, Any],
    *,
    min_interval_sec: float = DEFAULT_NAV_SAMPLE_INTERVAL_SEC,
    retention_days: float = DEFAULT_NAV_RETENTION_DAYS,
    cashflow_jump_pct: float = DEFAULT_CASHFLOW_JUMP_PCT,
) -> tuple[list[dict[str, Any]], bool]:
    """Append ``point`` to bounded NAV history respecting spacing, retention, and cashflow tagging."""
    norm_new = _normalize_nav_point(point)
    cleaned: list[dict[str, Any]] = []
    for item in (history if isinstance(history, list) else []):
        if (norm_it := _normalize_nav_point(item)) is not None:
            cleaned.append(norm_it)
    cleaned.sort(key=lambda r: r["ts"])
    if norm_new is None:
        return cleaned, False

    min_gap = safe_nonneg_float(min_interval_sec, default=DEFAULT_NAV_SAMPLE_INTERVAL_SEC)
    ret_days = safe_pos_float(retention_days, default=DEFAULT_NAV_RETENTION_DAYS) or DEFAULT_NAV_RETENTION_DAYS
    cutoff = norm_new["ts"] - (ret_days * 86400.0)
    cleaned = [p for p in cleaned if p["ts"] >= cutoff]

    if cleaned:
        last = cleaned[-1]
        if norm_new["ts"] < last["ts"]:
            return cleaned, False
        if min_gap > 0.0 and (norm_new["ts"] - last["ts"]) < min_gap:
            return cleaned, False
        is_cf, is_vs, delta_nav = _detect_non_market_step(
            last, norm_new, cashflow_jump_pct=cashflow_jump_pct
        )
        if is_cf:
            norm_new["external_cashflow"] = True
            norm_new["cashflow_delta_usd"] = round(delta_nav, 2)
        elif is_vs:
            norm_new["venue_set_changed"] = True

    cleaned.append(norm_new)
    return cleaned, True


def evaluate_nav_drawdown_metrics(
    history: list[dict[str, Any]],
    *,
    now_ts: Optional[float] = None,
    current_point: Optional[Mapping[str, Any]] = None,
    cashflow_jump_pct: float = DEFAULT_CASHFLOW_JUMP_PCT,
) -> dict[str, Any]:
    """Compute cashflow-adjusted daily/24h/7d/30d NAV drawdown and 30d daily Sharpe from NAV history."""
    points: list[dict[str, Any]] = []
    for item in (history if isinstance(history, list) else []):
        if (norm_it := _normalize_nav_point(item)) is not None:
            points.append(norm_it)
    points.sort(key=lambda r: r["ts"])

    cur_norm = _normalize_nav_point(current_point)
    if cur_norm is not None:
        if not points or cur_norm["ts"] > points[-1]["ts"]:
            points.append(cur_norm)
        elif cur_norm["ts"] == points[-1]["ts"]:
            points[-1] = cur_norm

    eff_now = resolve_now_ts(now_ts) if now_ts is not None else (points[-1]["ts"] if points else resolve_now_ts(None))
    empty: dict[str, Any] = {
        "current_nav_usd": 0.0,
        "utc_day": _utc_day_str(eff_now),
        "sample_count": 0,
        "days_covered": 0.0,
        "daily_nav_return_pct": 0.0,
        "daily_nav_drawdown_pct": 0.0,
        "rolling_24h_drawdown_pct": 0.0,
        "max_drawdown_7d_pct": 0.0,
        "max_drawdown_30d_pct": 0.0,
        "daily_sharpe_30d": None,
        "external_cashflow_events": 0,
        "venues_usd": {},
    }
    if not points:
        return empty

    # Build time-weighted cashflow-adjusted index (1.0 at start) so external deposits/withdrawals never distort DD
    adj_series: list[tuple[float, str, float, dict[str, Any]]] = []
    cur_idx = 1.0
    cf_events = 0
    for i, pt in enumerate(points):
        if i > 0:
            prev = points[i - 1]
            flagged_cf = bool(pt.get("external_cashflow")) or bool(pt.get("venue_set_changed"))
            if not flagged_cf:
                det_cf, det_vs, _ = _detect_non_market_step(prev, pt, cashflow_jump_pct=cashflow_jump_pct)
                flagged_cf = det_cf or det_vs
            if flagged_cf:
                cf_events += 1
            elif prev["nav_usd"] > 0.0:
                cur_idx *= pt["nav_usd"] / prev["nav_usd"]
        adj_series.append((pt["ts"], _utc_day_str(pt["ts"]), cur_idx, pt))

    latest_ts, latest_day, latest_idx, latest_pt = adj_series[-1]
    ref_now = max(eff_now, latest_ts)

    # UTC day baseline: prefer the closing point immediately before today's UTC midnight, else today's first point
    day_open_idx = latest_idx
    day_peak_idx = latest_idx
    found_today = False
    for _, p_day, p_idx, _ in adj_series:
        if p_day < latest_day:
            day_open_idx = p_idx
        elif p_day == latest_day:
            if not found_today:
                if day_open_idx == latest_idx and adj_series[0][1] == latest_day:
                    day_open_idx = p_idx
                day_peak_idx = max(day_open_idx, p_idx)
                found_today = True
            else:
                if p_idx > day_peak_idx:
                    day_peak_idx = p_idx

    daily_ret_pct = ((latest_idx - day_open_idx) / day_open_idx) * 100.0 if day_open_idx > 0.0 else 0.0
    daily_dd_pct = ((latest_idx - day_peak_idx) / day_peak_idx) * 100.0 if day_peak_idx > 0.0 else 0.0

    cutoff_24h = ref_now - 86400.0
    peak_24h = max((p_idx for p_ts, _, p_idx, _ in adj_series if p_ts >= cutoff_24h), default=latest_idx)
    rolling_24h_dd_pct = ((latest_idx - peak_24h) / peak_24h) * 100.0 if peak_24h > 0.0 else 0.0

    def _window_max_dd(days: float) -> float:
        cut = ref_now - (days * 86400.0)
        run_peak = 0.0
        worst_dd = 0.0
        for p_ts, _, p_idx, _ in adj_series:
            if p_ts < cut:
                continue
            if p_idx > run_peak:
                run_peak = p_idx
            if run_peak > 0.0:
                dd = ((p_idx - run_peak) / run_peak) * 100.0
                if dd < worst_dd:
                    worst_dd = dd
        return round(worst_dd, 2)

    # Annualized daily Sharpe over UTC daily closes within last 30 days
    cut_30d = ref_now - (30.0 * 86400.0)
    day_closes: dict[str, float] = {}
    for p_ts, p_day, p_idx, _ in adj_series:
        if p_ts >= cut_30d:
            day_closes[p_day] = p_idx
    ordered_closes = [day_closes[d] for d in sorted(day_closes.keys())]
    daily_rets = [
        (ordered_closes[k] - ordered_closes[k - 1]) / ordered_closes[k - 1]
        for k in range(1, len(ordered_closes))
        if ordered_closes[k - 1] > 0.0
    ]
    sharpe_30d: Optional[float] = None
    if len(daily_rets) >= 3:
        mean_r = sum(daily_rets) / len(daily_rets)
        var_r = sum((r - mean_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
        std_r = math.sqrt(max(0.0, var_r))
        if std_r > 1e-12:
            sharpe_30d = round((mean_r / std_r) * math.sqrt(365.0), 2)

    span_days = max(0.0, (latest_ts - adj_series[0][0]) / 86400.0)
    return {
        "current_nav_usd": round(latest_pt["nav_usd"], 2),
        "utc_day": latest_day,
        "sample_count": len(adj_series),
        "days_covered": round(span_days, 2),
        "daily_nav_return_pct": round(daily_ret_pct, 2),
        "daily_nav_drawdown_pct": round(min(0.0, daily_dd_pct), 2),
        "rolling_24h_drawdown_pct": round(min(0.0, rolling_24h_dd_pct), 2),
        "max_drawdown_7d_pct": _window_max_dd(7.0),
        "max_drawdown_30d_pct": _window_max_dd(30.0),
        "daily_sharpe_30d": sharpe_30d,
        "external_cashflow_events": cf_events,
        "venues_usd": dict(latest_pt.get("venues_usd") or {}),
    }


def evaluate_daily_nav_drawdown_guard(
    metrics: Any,
    *,
    decision: str,
    mode: str = DEFAULT_DAILY_NAV_DD_MODE,
    limit_pct: float = DEFAULT_DAILY_NAV_DD_LIMIT_PCT,
) -> tuple[bool, str, dict[str, Any]]:
    """Asymmetric daily Unified NAV drawdown guard strictly for ``by=AI`` BUY intents.

    Returns ``(allowed, reason, guard_metrics)``.
    - ``SELL`` exits are NEVER blocked.
    - ``mode == 'off'`` or ``limit_pct <= 0`` or fewer than 2 NAV samples always passes.
    - When triggered, caller enforces downgrade to shadow when ``mode == 'enforce'``, or logs a
      non-blocking observational warning when ``mode == 'shadow'``.
    """
    eff_mode = normalize_nav_dd_mode(mode, default=DEFAULT_DAILY_NAV_DD_MODE)
    eff_limit = safe_nonneg_float(limit_pct, default=DEFAULT_DAILY_NAV_DD_LIMIT_PCT)
    side = normalize_side(decision)
    if eff_mode == "off" or eff_limit <= 0.0 or side != "BUY":
        return True, "", {}
    if not isinstance(metrics, Mapping):
        return True, "", {}
    samples = safe_nonneg_int(metrics.get("sample_count"), default=0)
    if samples < 2:
        return True, "", {}
    daily_dd = safe_float(metrics.get("daily_nav_drawdown_pct"), default=0.0) or 0.0
    g_metrics: dict[str, Any] = {
        "daily_nav_drawdown_pct": round(daily_dd, 2),
        "daily_nav_return_pct": round(safe_float(metrics.get("daily_nav_return_pct"), default=0.0) or 0.0, 2),
        "limit_pct": round(eff_limit, 2),
        "guard_mode": eff_mode,
        "current_nav_usd": round(safe_nonneg_float(metrics.get("current_nav_usd"), default=0.0), 2),
    }
    if daily_dd <= -eff_limit:
        return (
            False,
            f"ai_daily_nav_drawdown_guard(dd={daily_dd:.2f}%<=-{eff_limit:.2f}%, mode={eff_mode})",
            g_metrics,
        )
    return True, "", g_metrics


def _nav_history_filepath(workspace_dir: str) -> str:
    return str(resolve_cachedb_filepath(os.path.join(workspace_dir, "cachedb", NAV_HISTORY_STATE_FILENAME)))


def load_nav_history_state(workspace_dir: str) -> list[dict[str, Any]]:
    """Load persisted NAV history points from ``cachedb/state/nav_history_state.json``."""
    path = _nav_history_filepath(workspace_dir)
    doc = read_json_dict(path, default={})
    raw_pts = doc.get("points") if isinstance(doc, dict) else None
    if not isinstance(raw_pts, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_pts:
        if (norm := _normalize_nav_point(item)) is not None:
            out.append(norm)
    out.sort(key=lambda r: r["ts"])
    return out


def _load_workspace_balances_and_prices(workspace_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    vb_path = str(resolve_cachedb_filepath(os.path.join(workspace_dir, "cachedb", VENUE_BALANCES_STATE_FILENAME)))
    cp_path = str(resolve_cachedb_filepath(os.path.join(workspace_dir, "cachedb", CACHE_CURRENT_PRICE_FILENAME)))
    vb = read_json_dict(vb_path, default={})
    cp = read_json_dict(cp_path, default={})
    return (vb if isinstance(vb, dict) else {}), (cp if isinstance(cp, dict) else {})


def record_nav_history_snapshot(
    workspace_dir: str,
    *,
    venue_balances_state: Optional[Mapping[str, Any]] = None,
    live_prices: Optional[Mapping[str, Any]] = None,
    now_ts: Optional[float] = None,
    min_interval_sec: float = DEFAULT_NAV_SAMPLE_INTERVAL_SEC,
    retention_days: float = DEFAULT_NAV_RETENTION_DAYS,
    cashflow_jump_pct: float = DEFAULT_CASHFLOW_JUMP_PCT,
) -> dict[str, Any]:
    """Transactionally sample and persist one Unified NAV point and return evaluated drawdown metrics."""
    eff_now = resolve_now_ts(now_ts)
    vb_map = venue_balances_state
    cp_map = live_prices
    if vb_map is None or cp_map is None:
        disk_vb, disk_cp = _load_workspace_balances_and_prices(workspace_dir)
        if vb_map is None:
            vb_map = disk_vb
        if cp_map is None:
            cp_map = disk_cp

    point = build_nav_history_point(vb_map, live_prices=cp_map, now_ts=eff_now, scope="all")
    nav_path = _nav_history_filepath(workspace_dir)
    if point is None:
        return evaluate_nav_drawdown_metrics(load_nav_history_state(workspace_dir), now_ts=eff_now)

    with generation_file_lock(nav_path):
        existing = load_nav_history_state(workspace_dir)
        updated, appended = append_nav_history_point(
            existing,
            point,
            min_interval_sec=min_interval_sec,
            retention_days=retention_days,
            cashflow_jump_pct=cashflow_jump_pct,
        )
        if appended:
            save_json_state(
                nav_path,
                {
                    "version": 1,
                    "updated_ts": round(eff_now, 3),
                    "points": updated,
                },
                indent=2,
            )
    return evaluate_nav_drawdown_metrics(
        updated,
        now_ts=eff_now,
        current_point=None if appended else point,
        cashflow_jump_pct=cashflow_jump_pct,
    )


def load_and_evaluate_nav_drawdown(
    workspace_dir: str,
    *,
    now_ts: Optional[float] = None,
    include_live_point: bool = True,
) -> dict[str, Any]:
    """Read persisted NAV history (plus optional current mark-to-market snapshot) without mutating disk."""
    eff_now = resolve_now_ts(now_ts)
    history = load_nav_history_state(workspace_dir)
    cur_pt: Optional[dict[str, Any]] = None
    if include_live_point:
        vb_map, cp_map = _load_workspace_balances_and_prices(workspace_dir)
        cur_pt = build_nav_history_point(vb_map, live_prices=cp_map, now_ts=eff_now, scope="all")
    return evaluate_nav_drawdown_metrics(history, now_ts=eff_now, current_point=cur_pt)


def format_nav_drawdown_summary_line(metrics: Any) -> str:
    """Format a compact single-line ASCII summary of Unified NAV & drawdown metrics."""
    if not isinstance(metrics, Mapping):
        return ""
    nav_u = safe_pos_float(metrics.get("current_nav_usd"), default=0.0) or 0.0
    if nav_u <= 0.0:
        return ""
    sh = safe_float(metrics.get("daily_sharpe_30d"), default=None)
    sh_s = f"{sh:.2f}" if sh is not None else "n/a"
    cf_n = safe_nonneg_int(metrics.get("external_cashflow_events"), default=0)
    return (
        f"Unified NAV=${nav_u:,.2f} (samples={safe_nonneg_int(metrics.get('sample_count'), 0)}, "
        f"span={safe_nonneg_float(metrics.get('days_covered'), 0.0):.1f}d): "
        f"day_ret={(safe_float(metrics.get('daily_nav_return_pct'), 0.0) or 0.0):+.2f}%, "
        f"day_dd={(safe_float(metrics.get('daily_nav_drawdown_pct'), 0.0) or 0.0):+.2f}%, "
        f"24h_dd={(safe_float(metrics.get('rolling_24h_drawdown_pct'), 0.0) or 0.0):+.2f}%, "
        f"7d_maxdd={(safe_float(metrics.get('max_drawdown_7d_pct'), 0.0) or 0.0):+.2f}%, "
        f"30d_maxdd={(safe_float(metrics.get('max_drawdown_30d_pct'), 0.0) or 0.0):+.2f}%, "
        f"sharpe_30d={sh_s}, cashflow_adj={cf_n}"
    )


__all__ = [
    "DEFAULT_CASHFLOW_JUMP_PCT",
    "DEFAULT_DAILY_NAV_DD_LIMIT_PCT",
    "DEFAULT_DAILY_NAV_DD_MODE",
    "DEFAULT_NAV_RETENTION_DAYS",
    "DEFAULT_NAV_SAMPLE_INTERVAL_SEC",
    "append_nav_history_point",
    "build_nav_history_point",
    "evaluate_daily_nav_drawdown_guard",
    "evaluate_nav_drawdown_metrics",
    "format_nav_drawdown_summary_line",
    "load_and_evaluate_nav_drawdown",
    "load_nav_history_state",
    "normalize_nav_dd_mode",
    "record_nav_history_snapshot",
]
