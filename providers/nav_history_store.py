"""providers/nav_history_store.py -- Layer 3 persistence & workspace loader facade for Unified NAV history.

Delegates ``nav_history_state.json`` I/O to Layer 2.5 ``NavHistoryStateStore`` and balance/price loading
to ``providers/portfolio_state_store.py`` + ``state_io``. Pure drawdown/Sharpe math lives exclusively in
``providers/nav_drawdown.py`` (zero disk I/O).
"""
from __future__ import annotations

import math
import os
from typing import Any, Mapping, Optional

from botcore import resolve_now_ts, safe_pos_float
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    canonical_base_asset,
    ensure_quote_suffix,
    resolve_cachedb_filepath,
)
from state_io import read_json_dict
from state_stores.nav_history_state import NavHistoryStateStore
from state_stores.venue_fee_maintenance_state import VenueFeeMaintenanceStateStore

from .nav_drawdown import (
    DEFAULT_CASHFLOW_JUMP_PCT,
    DEFAULT_NAV_RETENTION_DAYS,
    DEFAULT_NAV_SAMPLE_INTERVAL_SEC,
    _normalize_nav_point,
    _normalize_nav_scope,
    append_nav_history_point,
    build_nav_history_point,
    evaluate_nav_drawdown_metrics,
)
from .portfolio_state_store import load_venue_balance_caches, load_venue_balances_state_dict


def _nav_history_filepath(workspace_dir: str) -> str:
    return NavHistoryStateStore.resolve_workspace_path(workspace_dir)


def load_nav_history_state(workspace_dir: str) -> list[dict[str, Any]]:
    """Load persisted NAV history points via ``NavHistoryStateStore``."""
    doc = NavHistoryStateStore.read_raw_doc(workspace_dir)
    raw_pts = doc.get("points") if isinstance(doc, dict) else None
    if not isinstance(raw_pts, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_pts:
        if (norm := _normalize_nav_point(item)) is not None:
            out.append(norm)
    out.sort(key=lambda r: r["ts"])
    return out


def _load_workspace_balances_and_prices(
    workspace_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_dir = os.path.join(workspace_dir, "cachedb")
    bal_caches = load_venue_balance_caches(cache_dir)
    vb = load_venue_balances_state_dict(cache_dir)
    cp_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, CACHE_CURRENT_PRICE_FILENAME)))
    cp = read_json_dict(cp_path, default={}, recover_previous=True)
    cp_out = dict(cp) if isinstance(cp, dict) else {}
    # Seed Binance fee-reserve BNB price from venue_fee_maintenance_state.json when BNB is not tracked
    # in cache_current_price.json so automatic BNB fee top-ups from USDC never look like NAV drawdown.
    try:
        items_view = cp_out.get("items") if isinstance(cp_out.get("items"), dict) else cp_out
        if not any(isinstance(k, str) and canonical_base_asset(k) == "BNB" for k in items_view):
            fee_state = VenueFeeMaintenanceStateStore.read_state(cache_dir)
            bn_fee = (fee_state.get("venues") or {}).get("binance") if isinstance(fee_state, dict) else None
            bnb_px = safe_pos_float(bn_fee.get("bnb_price_usdc") if isinstance(bn_fee, dict) else None, default=None)
            if bnb_px is not None:
                bnb_ts_s = safe_pos_float(
                    (bn_fee.get("ts") if isinstance(bn_fee, dict) else None)
                    or (fee_state.get("ts") if isinstance(fee_state, dict) else None),
                    default=None,
                ) or resolve_now_ts()
                bnb_ts_ms = math.trunc(bnb_ts_s * 1000.0)
                bnb_sym = ensure_quote_suffix("BNB", "USDC")
                if isinstance(cp_out.get("items"), dict):
                    items_copy = dict(cp_out["items"])
                    items_copy[bnb_sym] = [[bnb_ts_ms, bnb_px]]
                    cp_out["items"] = items_copy
                else:
                    cp_out[bnb_sym] = [[bnb_ts_ms, bnb_px]]
    except Exception:
        pass
    return bal_caches, vb, cp_out


def record_nav_history_snapshot(
    workspace_dir: str,
    *,
    venue_balances_state: Optional[Mapping[str, Any]] = None,
    venue_balance_caches: Optional[Mapping[str, Any]] = None,
    live_prices: Optional[Mapping[str, Any]] = None,
    now_ts: Optional[float] = None,
    min_interval_sec: float = DEFAULT_NAV_SAMPLE_INTERVAL_SEC,
    retention_days: float = DEFAULT_NAV_RETENTION_DAYS,
    cashflow_jump_pct: float = DEFAULT_CASHFLOW_JUMP_PCT,
) -> dict[str, Any]:
    """Transactionally sample and persist one Unified NAV point and return evaluated drawdown metrics."""
    eff_now = resolve_now_ts(now_ts)
    vb_map = venue_balances_state
    bc_map = venue_balance_caches
    cp_map = live_prices
    if (vb_map is None and bc_map is None) or cp_map is None:
        disk_bc, disk_vb, disk_cp = _load_workspace_balances_and_prices(workspace_dir)
        if bc_map is None and vb_map is None:
            bc_map = disk_bc
            vb_map = disk_vb
        if cp_map is None:
            cp_map = disk_cp

    point = build_nav_history_point(
        vb_map,
        venue_balance_caches=bc_map,
        live_prices=cp_map,
        now_ts=eff_now,
        scope="all",
    )
    nav_path = _nav_history_filepath(workspace_dir)
    if point is None:
        return evaluate_nav_drawdown_metrics(load_nav_history_state(workspace_dir), now_ts=eff_now)

    with NavHistoryStateStore.generation_lock(nav_path):
        existing = load_nav_history_state(workspace_dir)
        updated, appended = append_nav_history_point(
            existing,
            point,
            min_interval_sec=min_interval_sec,
            retention_days=retention_days,
            cashflow_jump_pct=cashflow_jump_pct,
        )
        if appended:
            NavHistoryStateStore.save_points_doc(nav_path, updated, updated_ts=eff_now, indent=2)
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
    scope: str = "all",
    current_point: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Read persisted NAV history (plus optional current mark-to-market snapshot) without mutating disk."""
    eff_now = resolve_now_ts(now_ts)
    scope_norm = _normalize_nav_scope(scope)
    history = load_nav_history_state(workspace_dir)
    cur_pt: Optional[Mapping[str, Any]] = current_point
    if cur_pt is None and include_live_point:
        bc_map, vb_map, cp_map = _load_workspace_balances_and_prices(workspace_dir)
        cur_pt = build_nav_history_point(
            vb_map,
            venue_balance_caches=bc_map,
            live_prices=cp_map,
            now_ts=eff_now,
            scope=scope_norm,
        )
    return evaluate_nav_drawdown_metrics(history, now_ts=eff_now, current_point=cur_pt, scope=scope_norm)


__all__ = [
    "load_and_evaluate_nav_drawdown",
    "load_nav_history_state",
    "record_nav_history_snapshot",
]
