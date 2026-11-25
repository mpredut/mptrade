"""providers/ai_attribution_store.py -- Layer 3 workspace loader facade for AI vs. BOT attribution reports.

Reads local cachedb/logger artifacts (intents, prices, OrderLedger/audit JSONL, reconciler logs,
NAV history, venue balances, and persisted fee specs) and delegates all evaluation math to pure
``providers/ai_attribution.py`` (zero disk I/O).
"""
from __future__ import annotations

from datetime import datetime, timezone
import glob
import os
import re
from typing import Any, Optional

from botcore import clean_upper_scalar, resolve_now_ts, safe_nonneg_float, safe_pos_float
from cachepaths import (
    AUTO_ORDER_INTENTS_STATE_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    resolve_cachedb_filepath,
)
from state_io import read_json_dict, read_jsonl_dicts, read_text_lines

from .ai_attribution import evaluate_ai_attribution
from .ai_survival_guard import extract_live_prices_from_cache_payload
from .execution_audit import iter_audit_jsonl_paths
from .intent_state_store import load_durable_intents_state
from .nav_drawdown import build_nav_history_point
from .nav_history_store import load_nav_history_state
from .order_ledger import OrderLedgerReader
from .portfolio_state_store import load_venue_balance_caches, load_venue_balances_state_dict
from .venue_fee_maintenance import read_persisted_venue_fee_spec

_LOG_DAY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _workspace_cachedb_path(workspace_dir: str, filename: str) -> str:
    return str(resolve_cachedb_filepath(os.path.join(workspace_dir, "cachedb", filename)))


def _load_live_prices_from_workspace(workspace_dir: str) -> dict[str, float]:
    price_path = _workspace_cachedb_path(workspace_dir, CACHE_CURRENT_PRICE_FILENAME)
    raw = read_json_dict(price_path, default={}, recover_previous=True)
    return extract_live_prices_from_cache_payload(raw)


def _load_reconciler_log_lines(workspace_dir: str, *, min_day: str, max_day: str) -> list[str]:
    log_dir = os.path.join(workspace_dir, "logger")
    if not os.path.isdir(log_dir):
        return []
    paths = sorted(glob.glob(os.path.join(glob.escape(log_dir), "autonomous_ai_reconciler*.log")))
    lines: list[str] = []
    for lp in paths:
        bn = os.path.basename(lp)
        m_day = _LOG_DAY_RE.search(bn)
        if m_day:
            file_day = m_day.group(1)
            if (min_day and file_day < min_day) or (max_day and file_day > max_day):
                continue
        lines.extend(read_text_lines(lp, strip=True, skip_empty=True))
    return lines


def load_and_evaluate_ai_attribution(
    workspace_dir: str,
    *,
    lookback_days: float = 7.0,
    scope: str = "all",
    now_ts: Optional[float] = None,
) -> dict[str, Any]:
    """Load local workspace intents, execution audits, live price cache, and reconciler logs and evaluate attribution."""
    eff_now = resolve_now_ts(now_ts)
    lb_days = safe_pos_float(lookback_days, default=7.0) or 7.0
    cutoff = max(0.0, eff_now - (lb_days * 86400.0))
    min_day = datetime.fromtimestamp(max(0.0, cutoff - 86400.0), timezone.utc).strftime("%Y-%m-%d")
    max_day = datetime.fromtimestamp(eff_now + 86400.0, timezone.utc).strftime("%Y-%m-%d")

    intents_path = _workspace_cachedb_path(workspace_dir, AUTO_ORDER_INTENTS_STATE_FILENAME)
    intents = load_durable_intents_state(intents_path, read_only=True)
    live_prices = _load_live_prices_from_workspace(workspace_dir)

    cache_dir = os.path.join(workspace_dir, "cachedb")
    audit_dir = os.path.join(workspace_dir, "logger", "execution_audit")
    ledger_reader = OrderLedgerReader(cachedb_dir=cache_dir)
    ledger_records, has_ledger = ledger_reader.iter_audit_compat_records(min_day=min_day, max_day=max_day)
    if has_ledger:
        audit_records = ledger_records
    else:
        audit_records = [
            rec
            for ap in iter_audit_jsonl_paths(audit_dir, min_day=min_day, max_day=max_day)
            for rec in read_jsonl_dicts(ap)
        ]

    log_lines = _load_reconciler_log_lines(workspace_dir, min_day=min_day, max_day=max_day)
    eval_path = _workspace_cachedb_path(workspace_dir, "auto_cryptotrade_executor_eval.json")
    crypto_eval = read_json_dict(eval_path, default={})
    nav_history = load_nav_history_state(workspace_dir)
    bal_caches = load_venue_balance_caches(cache_dir)
    vb_state = load_venue_balances_state_dict(cache_dir)
    cur_nav_pt = build_nav_history_point(
        vb_state,
        venue_balance_caches=bal_caches,
        live_prices=live_prices,
        now_ts=eff_now,
        scope="all",
    )

    def _persisted_rt_fee_bps(venue: str, symbol: str, order_type: str) -> Optional[float]:
        spec = read_persisted_venue_fee_spec(venue, symbol, cache_dir=cache_dir, now_ts=eff_now)
        if not isinstance(spec, dict) or not spec:
            return None
        leg_key = "taker_fee_bps" if (clean_upper_scalar(order_type) or "LIMIT") == "MARKET" else "maker_fee_bps"
        leg_bps = safe_nonneg_float(spec.get(leg_key), default=None)
        return round(2.0 * leg_bps, 4) if leg_bps is not None else None

    return evaluate_ai_attribution(
        intents,
        now_ts=eff_now,
        lookback_days=lb_days,
        scope=scope,
        live_prices=live_prices,
        audit_records=audit_records,
        reconciler_log_lines=log_lines,
        crypto_eval_state=crypto_eval,
        nav_history=nav_history,
        current_nav_point=cur_nav_pt,
        fee_resolver=_persisted_rt_fee_bps,
    )


__all__ = ["load_and_evaluate_ai_attribution"]
