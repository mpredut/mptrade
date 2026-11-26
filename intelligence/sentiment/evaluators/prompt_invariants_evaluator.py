"""Canonical SSOT deterministic mathematical and structural invariant evaluator for AI executor telemetry & prompts.

Pure, zero-I/O evaluator running in ~1 ms over ``(snapshot, prompt_text)`` for both:
- Continuous inline runtime validation in ``AutonomousAIReconciler`` before every LLM call
- CLI/CI verification via ``verify_tools/audit_prompt_invariants.py`` and ``tests/test_prompt_invariants.py``
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

from botcore import (
    clean_scalar_str,
    clamp_float,
    fmt_signed,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import canonical_base_asset, has_equity_suffix, matches_asset_scope, normalize_venue_name
from instrument_registry import casefold_get
from intelligence.sentiment.evaluators.price_series_evaluator import (
    stale_closed_equity_metric_keys,
)
from providers.ai_survival_guard import (
    _collect_canceled_order_ids,
    _extract_intent_exec_price,
    _extract_intent_exec_ts,
    _is_filled_ai_intent,
    build_normalized_filled_ai_intent_row,
    match_filled_intent_rows_lifo_in_window,
)
from providers.intent_state import extract_intent_side
from providers.venue_metadata import ProviderVenueMetadataMixin

_NAV_LINE_RE = re.compile(
    r"(?:crypto|equities|total)_nav=\$([0-9,]+\.\d+),\s*invested=\$([0-9,]+\.\d+),\s*"
    r"free_cash=\$([0-9,]+\.\d+)[^,]*(?:,\s*locked_quote=\$([0-9,]+\.\d+)[^,]*)?,\s*"
    r"venues\(([^)]+)\)"
)
_VENUE_VAL_RE = re.compile(r"([a-z0-9_]+)=\$([0-9,]+\.\d+)")
_SEC3C_INST_RE = re.compile(r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+):(.*)$", re.MULTILINE)
_FREE_QUOTE_BD_RE = re.compile(r"free_([A-Z]+)=\$([0-9,]+\.?\d*)\s*\(([^)]+)\)")
_BD_ITEM_RE = re.compile(r"([A-Z0-9]+)=\$([0-9,]+\.?\d*)")
_SEC3_SYM_RE = re.compile(
    r"^\s*[\*-]\s*([A-Za-z0-9_]+)(?:\s*(?:\[[^\]]*\]|\([^)]*\)))*:\s*Price=\$?([0-9,]+(?:\.[0-9]+)?)(.*)$",
    re.MULTILINE,
)
_SEC3B_SYM_RE = re.compile(r"^\s*[\*-]\s*([A-Za-z0-9_]+)(?:\s*(?:\[[^\]]*\]|\([^)]*\)))*:(.*)$", re.MULTILINE)
_UNIFIED_SUBLINE_RE = re.compile(r"^\s{2,}-\s*(Trend|Book|Flow):\s*(.*)$")
_RANGE_RE = re.compile(
    r"(24h|7d)_range=\[([0-9.]+)(?:\.\.|,\s*)([0-9.]+)\]\s*(?:\(pos=|pos_(?:24h|7d)=)([0-9.]+)%"
)
_ARB_VS_RE = re.compile(r"arb_vs_([A-Za-z0-9_]+)=([+-]?[0-9.]+)bps")
_VOL_PROFILE_RE = re.compile(
    r"vol_profile\(POC=([0-9.]+)\[([+-]?[0-9.]+)%\].*?VAL=([0-9.]+)\[([+-]?[0-9.]+)%\],\s*VAH=([0-9.]+)\[([+-]?[0-9.]+)%\](?:,\s*va_pos=([0-9.-]+)%)?"
)
_SESSION_VWAP_RE = re.compile(
    r"session_vwap=\$([0-9,]+(?:\.[0-9]+)?)\s*\(([+-]?[0-9.]+)%(?:,\s*z=([+-]?[0-9.]+))?"
)
_OB_IMB_RE = re.compile(
    r"ob_imbalance=([0-9.]+)\s*\(bid_depth=\$([0-9.,]+[KMBkmb]?),\s*ask_depth=\$([0-9.,]+[KMBkmb]?)\)"
)
_VENUE_SPEC_ROW_RE = re.compile(
    r"^\s*\|\s*([a-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\|[^|]*\|\s*([^|]+?)\s*\|[^|]*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|$",
    re.MULTILINE,
)
_FEE_PAIR_RE = re.compile(r"\(([0-9.]+)/([0-9.]+)\)")
_OPEN_ORDER_ROW_RE = re.compile(
    r"^\s*\|\s*([a-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*(BUY|SELL)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|[^|]*\|(?:\s*([^|]+?)\s*\|)?\s*([^|]+?)\s*\|$",
    re.MULTILINE,
)
_DIST_PCT_CELL_RE = re.compile(r"^([+-]?[0-9.]+)%$")
_FILLED_REM_RE = re.compile(r"^([0-9.]+)\(rem=([0-9.]+)\)$")
_PLAN_RR_RE = re.compile(
    r"([0-9.]+)(?:\(([+-]?[0-9.]+)%\))?/([0-9.]+)(?:\(([+-]?[0-9.]+)%\))?\s+RR(?:@limit)?=([0-9.]+)x"
)
_FRONT_RUN_RR_RE = re.compile(r"front_run=([0-9.]+)\(RR=([0-9.]+)x\)")
_TRIGGER_WALL_TRAPPED_RE = re.compile(r"\b([A-Za-z0-9_]+)\s+wall_trapped\(front_run=([0-9.]+),RR=([0-9.]+)x\)")
_NOTE_WALL_PX_RE = re.compile(
    r"\b(?:DEEP_PASSIVE_BID|BEHIND_BID_WALL|AT_BID_WALL|FAR_ABOVE_ASK_WALL|BEHIND_ASK_WALL|AT_ASK_WALL)\(wall=([0-9.]+)"
)
_NOTE_FRONT_RUN_PX_RE = re.compile(r"\bfront_run=([0-9.]+)")
_ACTIVE_PLAN_LINE_RE = re.compile(
    r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+)\s+(BUY|SELL)@([0-9.]+)\s*\(([^)]+)\):\s*(.+)$",
    re.MULTILINE,
)
_ACTIVE_PLAN_SPOT_RE = re.compile(r"\bspot=([0-9.]+)")
_ACTIVE_PLAN_TGT_RE = re.compile(r"\btarget=([0-9.]+)(?:\s*\(([+-]?[0-9.]+)%\s*(vs_spot|vs_limit)\))?")
_ACTIVE_PLAN_INV_RE = re.compile(r"\binvalidation=([0-9.]+)(?:\s*\(([+-]?[0-9.]+)%\s*(vs_spot|vs_limit)\))?")
_ACTIVE_PLAN_RR_RE = re.compile(r"\bRR(?:@(spot|limit))?=([0-9.]+)x\b")
_VENUE_CASH_BD_RE = re.compile(r"\b([a-z0-9_]+)\s+([A-Z]+)=\$([0-9,]+\.?\d*)\s*\(([^)]+)\)")
_VENUE_FREE_CASH_ITEM_RE = re.compile(r"\b([a-z0-9_]+)\s+[A-Z]+=\$([0-9,]+\.?\d*)")
_VENUE_LOCKED_QUOTE_RE = re.compile(r"\[locked_quote_usd=\$([0-9,]+\.?\d*)\]")
_PARENS_RE = re.compile(r"\([^)]*\)")
_COMPACT_USD_MULT: dict[str, float] = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
_OB_MID_RE = re.compile(r"ob_mid=([0-9.]+)")
_BID_WALL_RE = re.compile(r"(?:bid_wall|est_bid_zone)=([0-9.]+)\s*\(")
_ASK_WALL_RE = re.compile(r"(?:ask_wall|est_ask_zone)=([0-9.]+)\s*\(")
_LIMITS_RE = re.compile(r"limits\(([^)]+)\)")
_LIMIT_NUM_RE = re.compile(r"\b(min_qty|step|tick)=([0-9.eE+-]+)")
_LIMIT_CAP_RE = re.compile(r"\b(max_buy|max_sell|replace_max_sell)=\$([0-9,.]+)")
_PX_FIELD_RE = re.compile(r"\bpx=([0-9.]+)")
_FREE_BASE_VAL_RE = re.compile(r"\bfree_[A-Za-z0-9_]+=([0-9.]+)\s*\(\$([0-9,.]+)\s*USD\)")
_LOCKED_BASE_QTY_USD_RE = re.compile(r"\blocked_base_qty=([0-9.]+)\s*\(\$([0-9,.]+)\s*USD\)")
_AVG_COST_PNL_RE = re.compile(r"\bavg_cost=([0-9.]+)\s*\(pnl_vs_cost=([+-]?[0-9.]+)%\)")
_PEAK_DD_RE = re.compile(r"\bpeak_px=([0-9.]+)\s*\(dd_from_peak=([+-]?[0-9.]+)%\)")
_CYCLE_ANCHOR_LINE_RE = re.compile(
    r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+):\s*qty=([0-9.]+),\s*cost=\$([0-9,.]+),\s*avg_cost=([0-9.]+)\b(.*)$",
    re.MULTILINE,
)
_PRE_FREE_PCT_RE = re.compile(r"\bfree_cash=\$([0-9,]+\.\d+)\s*\(([0-9.]+)%\)")
_PRE_LOCKED_PCT_RE = re.compile(r"\blocked_quote=\$([0-9,]+\.\d+)\s*\(([0-9.]+)%\)")
_PRE_VAR_PCT_RE = re.compile(r"\bintraday_var_95=\$([0-9,]+\.\d+)\s*\(([0-9.]+)%\)")
_PRE_CVAR_PCT_RE = re.compile(r"\bcvar_975=\$([0-9,]+\.\d+)\s*\(([0-9.]+)%\)")
_POS_LAST_BUY_RE = re.compile(r"\blast_buy_px=([0-9.]+)\s*\((?:pnl|chg)_vs_last_buy=([+-]?[0-9.]+)%")
_POS_LAST_SELL_RE = re.compile(r"\blast_sell_px=([0-9.]+)\s*\(chg_vs_last_sell=([+-]?[0-9.]+)%")
_POS_LIFO_AVG_RE = re.compile(r"\blifo_avg=([0-9.]+)\s*\(pnl_vs_lifo=([+-]?[0-9.]+)%")
_POS_OPEN_LOTS_RE = re.compile(r"\bopen_lifo_lots=\[([^\]]+)\]")
_POS_LOT_ITEM_RE = re.compile(r"([0-9.]+)@([0-9.]+)\(([+-]?[0-9.]+)%\)")
_CYC_PNL_RE = re.compile(r"\bpnl_vs_cost=([+-]?[0-9.]+)%")
_TAPE_15M_RE = re.compile(r"tape_15m=\[([^\]]+)\]\s*15m_range=\[([0-9.]+),\s*([0-9.]+)\]")
_TAPE_PT_RE = re.compile(r"(-?\d+m|now):([0-9.]+)")
_SHORT_H_RE = re.compile(r"(?:^|[\s|,(])(1m|3m|5m|15m)=([+-][0-9.]+)%")
_SWING_WAVE_RE = re.compile(
    r"swing_wave\(low=([0-9.]+),\s*high=([0-9.]+),\s*rally_low=([+-]?[0-9.]+)%,\s*pullback_high=([+-]?[0-9.]+)%"
    r"(?:,\s*major_low=([0-9.]+)\(rally=([+-]?[0-9.]+)%\))?"
    r"(?:,\s*major_high=([0-9.]+)\(pb=([+-]?[0-9.]+)%\))?\)"
)
_SESSIONS_BLOCK_RE = re.compile(r"sessions\(([^)]+)\)")
_SESSION_ITEM_RE = re.compile(r"(asia|ldn|ny)=\[([0-9.]+),\s*([0-9.]+)\]")
_SPOT_VOL_15M_RE = re.compile(r"\bspot_vol_15m=\$([0-9.,]+[KMBkmb]?)")
_SPOT_VOL_1H_RE = re.compile(r"\bspot_vol_1h=\$([0-9.,]+[KMBkmb]?)")
_AGG_SPOT_CVD_VOL_RE = re.compile(
    r"\bagg_spot_cvd_15m=([+-]?)\$([0-9.,]+[KMBkmb]?)\[[^,\]]+,\s*vol=\$([0-9.,]+[KMBkmb]?)\]"
)
_EXEC_FILL_RE = re.compile(r"\b(BUY|SELL)\s+LIMIT\s+(FILLED|PARTIALLY_FILLED)\s*\(([^)]+)\)")
_EXEC_KV_RE = re.compile(r"([a-z_]+)=([+-]?\$?[0-9,.]*[0-9])")
_EQ_CROSS_ROW_RE = re.compile(
    r"^\s*\|\s*([A-Z0-9_]+_EQ)\s*\|\s*\$([0-9.]+)\s*\|[^|]+\|[^|]+\|[^|]+\|\s*([+-][0-9.]+)%\s*\|\s*\$([0-9.]+)\s*\|",
    re.MULTILINE,
)
_BREADTH_ETH_BTC_RE = re.compile(r"\bETH/BTC=([0-9.]+).*?\bETH=\$([0-9,.]+)")
_LH_SWING_YM_RE = re.compile(r"\b(\d\d-\d\d):")
_SEC4_BW_ROW_RE = re.compile(
    r"^\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*\$[0-9,.]+\s*\|\s*([+-]?[0-9.]+%|-)\s*\|\s*([+-]?[0-9.]+%|-)\s*\|\s*([+-]?[0-9.]+%|-)\s*\|\s*([+-]?[0-9.]+%|-)\s*\|[^|]+\|\s*([^|]+)\|",
    re.MULTILINE,
)
_RANGE_EXT_RE = re.compile(
    r"(24h|7d)_range=\[([0-9.]+),\s*([0-9.]+)\]\s*pos_(?:24h|7d)=([0-9.]+)%\s*"
    r"\(dd_high=([+-]?[0-9.]+)%,\s*bounce_low=([+-]?[0-9.]+)%,\s*span=([0-9.]+)%\)"
)
_RANGE_15M_RE = re.compile(r"15m_range=\[([0-9.]+),\s*([0-9.]+)\]")
_BOTTOM_6H_RE = re.compile(
    r"bottom_6h\(prior_low=([0-9.]+),\s*1h_low=([0-9.]+)(?:,\s*(higher_low_pct|reclaim(?:_pct)?)=([+-]?[0-9.]+)%)?\)"
)
_HORIZON_CHG_RE = re.compile(r"(?:^|[\s|,(])(15m|1h|6h|24h|7d)=([+-][0-9.]+)%")
_VS_BTC_BLOCK_RE = re.compile(r"vs_btc\(([^)]+)\)")
_OB_SPREAD_BBO_RE = re.compile(r"spread=([0-9.]+)bps\s*\(bid=([0-9.]+)/ask=([0-9.]+)\)")
_WALL_DIST_RE = re.compile(
    r"(bid_wall|est_bid_zone|ask_wall|est_ask_zone)=([0-9.]+)\s*\([^)]*dist=([+-]?[0-9.]+)%\)"
)
_XSTOCK_REF_RE = re.compile(
    r"(?:SPCXxUSD=\$([0-9.]+).*?)?SPCX=\$([0-9.]+).*?xstock_vs_underlying=([+-]?[0-9.]+)%"
)
_ADVISOR_BTC_DELTA_RE = re.compile(r"live_btc_flow_delta=\(([^)]*(?:\([^)]*\)[^)]*)*)\)")
_ADV_TR_RE = re.compile(r"\btaker_ratio=([0-9.]+)")
_ADV_BTC_POC_RE = re.compile(r"\bBTC=\$([0-9,]+\.?\d*)(?:\(POC=\$([0-9,]+\.?\d*)\))?")
_TAKER_RATIO_VOL_RE = re.compile(
    r"\btaker_ratio=([0-9.]+)\s*\(buy_v=\$([0-9.,]+[KMBkmb]?)/sell_v=\$([0-9.,]+[KMBkmb]?)\)"
)
_DEPTH_1PCT_RE = re.compile(
    r"\bdepth_1pct=\(bid=\$([0-9.,]+[KMBkmb]?)/ask=\$([0-9.,]+[KMBkmb]?)"
)
_CVD_15M_RE = re.compile(
    r"\bcvd_15m=([+-]?[0-9.,]+)\s+[A-Za-z0-9_]+\s+\(([+-]?)\$([0-9.,]+[KMBkmb]?)\)"
)
_NET_BUY_EDGE_RE = re.compile(r"\bnet_buy_edge=([+-]?[0-9.]+)bps\[maker\]")
_CAL_PLANS_RE = re.compile(r"\bplans=(\d+)\s*\(([^)]+)\)")
_CAL_SIDE_RE = re.compile(r"\b(?:BUY|SELL):(\d+)hit/(\d+)inv/(\d+)early/(\d+)open\b")
_CAL_HIT_RE = re.compile(r"\bhit=(\d+)%")
_CAL_LIM_RE = re.compile(r"\bLIM\s+(\d+)/(\d+)\[(\d+)%\]")
_CAL_OFFSET_BLOCK_RE = re.compile(r"\bfill_by_offset\(([^)]+)\)")
_CAL_OFFSET_ITEM_RE = re.compile(r":(\d+)%\[(\d+)/(\d+)\]")
_BREADTH_ADV_RE = re.compile(r"\badvance_ratio=([0-9.]+).*?\badvancing=(\d+)/(\d+)")
_BREADTH_VOL_SHARE_RE = re.compile(r"\bBTC_vol_share=([0-9.]+)%,\s*Alt_vol_share=([0-9.]+)%")
_LIQ_CLUSTER_ITEM_RE = re.compile(
    r"\b(long_liq|short_liq|deep_long|deep_short)=\$([0-9.,]+)\[\$[^,]+,\s*([+-]?[0-9.]+)%\]"
)
_INTENT_STATUS_ID_RE = re.compile(
    r"\b(FILLED|PARTIALLY_FILLED|OPEN_RESTING|CANCELED|CANCELLED|EXPIRED)\b[^\n]*\bid=([A-Za-z0-9_-]+)"
)
_TERMINAL_ORDER_STATUS_TOKS: frozenset[str] = frozenset({"FILLED", "CANCELED", "CANCELLED", "EXPIRED"})
_WARM_OMITTED_LINE_RE = re.compile(
    r"^- Unchanged in thread \(active from prior turn\):\s*(.+)$",
    re.MULTILINE,
)
_WARM_OMITTED_HEADER_SPECS: tuple[tuple[str, str], ...] = (
    ("Calendar(", "Macro Economic Calendar & Session Window:"),
    ("US_Equities(", "Cross-Market US Equities Telemetry"),
    ("Scorecard(", "AI Decision Scorecard ("),
    ("Calibration(", "AI Execution & Plan Calibration ("),
    ("Cycle_Anchors(", "Active Cycle Cost & Drawdown Anchors"),
    ("Venue_Specs(", "Venue Specs & Fees"),
    ("Fear&Greed(", "- Fear & Greed:"),
    ("ETF_Flows(", "- US Spot Crypto ETF Net Flows"),
    ("Long_Horizon(", "Multi-Month & Annual Structure"),
    ("xStocks(", "Kraken Tokenized Equities / xStocks"),
    ("HL_Summary(", "Hyperliquid Spot/Perp Equity Summary:"),
    ("News(", "- Live Breaking News Feed:"),
)
_LH_ROW_RE = re.compile(
    r"^\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*(?:[+-]?[0-9.]+%|-)\s*\|\s*(?:[+-]?[0-9.]+%|-)\s*\|\s*(?:[+-]?[0-9.]+%|-)\s*\|\s*"
    r"\[\$([0-9.]+[KMBkmb]?)\.\.\$([0-9.]+[KMBkmb]?)\]\s*\(([0-9.]+)%\)\s*\|"
    r"(?:\s*\[\$([0-9.]+[KMBkmb]?)\.\.\$([0-9.]+[KMBkmb]?)\]\s*\(([0-9.]+)%\)\s*\|)?"
    r"[^|]+\|[^|]+\|([^|]*)now:([0-9.]+[KMBkmb]?)\s*\|$",
    re.MULTILINE,
)
_SEC2_CRYPTO_BW_ROW_RE = re.compile(r"^\s*\|\s*([A-Z0-9]+)\s*\|\s*\$([0-9,]+\.?\d*)\s*\|", re.MULTILINE)
_SCORECARD_OUTCOME_RE = re.compile(
    r"AI Decision Scorecard \([^,]+,\s*(?:last\s*(\d+)/(\d+)|n=(\d+))\)(?:\s*\[[^\]]+\])?:\s*(\d+)W/(\d+)BE/(\d+)L\b"
)
_VENUE_LIVE_REC_RE = re.compile(
    r"Total\s+([A-Za-z0-9]+)\s+Portfolio Value:\s*\$[0-9,]+\.\d+\s*[A-Z]+\s*\([^)]*live_reconciled=\$([0-9,]+\.\d+)"
)
_RECENT_FILL_LINE_RE = re.compile(
    r"^\s*\*\s+([A-Za-z0-9_]+)\s+Last Action=[^:]+:(.*)$",
    re.MULTILINE,
)
_RO_LAST_BUY_RE = re.compile(r"\bLast BUY=(?:[0-9.]+\s*@\s*)?([0-9.]+)\b")
_RO_LAST_SELL_RE = re.compile(r"\bLast SELL=(?:[0-9.]+\s*@\s*)?([0-9.]+)\b")
_RO_AVG_COST_RE = re.compile(r"\bavg_cost=([0-9.]+)\b")
_RO_LIFO_AVG_RE = re.compile(r"\blifo_avg=([0-9.]+)\b")
_RO_LIFO_PNL_RE = re.compile(r"Sub-Cycle LIFO Realized PnL=([+-]?)\$([0-9,]+\.\d+)")
_POS_LIFO_PNL_RE = re.compile(r"\blifo_realized_pnl=([+-]?)\$([0-9,]+\.\d+)")
_DOMAIN_AGE_PAIR_RE = re.compile(
    r"\b(prices|orderbook)_newest=([0-9.]+)([smh])[^\n]*?\b\1_oldest=([0-9.]+)([smh])"
)
_AGE_UNIT_SEC: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}
_DVOL_RE = re.compile(r"\b(BTC|ETH)_DVOL=([0-9.]+)%")
_OPTIONS_RV_VRP_RE = re.compile(r"\b(BTC|ETH)_Options\(([^)]+)\)")
_RV_ITEM_RE = re.compile(r"\bRV=([0-9.]+)%")
_VRP_ITEM_RE = re.compile(r"\bVRP=([+-]?[0-9.]+)%")
_BENCH_EQ_PX_RE = re.compile(r"\b(SPY|QQQ)=\$([0-9,]+\.?\d*)")
_VP_SEC_LVN_RE = re.compile(
    r"vol_profile\(POC=([0-9.]+)\[[^\]]+\],\s*sec_POC=([0-9.]+),\s*LVN=([0-9.]+)"
)
_SIBLING_HDR_RE = re.compile(
    r"^\s*\*\s+([A-Za-z0-9_]+)\s+\(sibling venue:\s*([A-Za-z0-9_]+)=\$([0-9,]+\.?\d*)\s*\[([+-]?[0-9.]+)%\]\):\s*Price=\$?([0-9,]+\.?\d*)",
    re.MULTILINE,
)
_KALMAN_PAIR_RE = re.compile(
    r"\bkalman_vel=([+-]?[0-9.]+)[^\n]*?\bkinematics\([^)]*\bkalman_z=([+-]?[0-9.]+)\)"
)
_NUMERIC_HYGIENE_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("NONE_LITERAL", re.compile(r"(?:=None\b|\|\s*None\s*\||\(None\)|\[None\])")),
    ("NAN_OR_INF", re.compile(r"(?:[=:(,\[\s\$])(?:nan|NaN|[+-]?inf|Inf)\b")),
    ("DOUBLE_SIGN", re.compile(r"(?:[+-]{2}\d|\$\s*[+-]{2}|\$-[+])")),
    ("NEG_ZERO", re.compile(r"(?:-0\.0+%|-\$0\.00\b)")),
    ("SCI_NOTATION", re.compile(r"(?:[=:(,\[\s|])\d+(?:\.\d+)?[eE][+-]\d+\b")),
    ("DOUBLE_UNIT", re.compile(r"(?:%%|\$\$)")),
    ("RAW_PYTHON_DICT", re.compile(r"\{'[A-Za-z0-9_]+':")),
)


@dataclass(frozen=True)
class InvariantViolation:
    code: str
    subject: str
    detail: str

    def format_line(self) -> str:
        return f"[{self.code}] {self.subject}: {self.detail}"


def _parse_money(raw: str) -> Optional[float]:
    s = str(raw or "").strip().replace(",", "")
    if not s:
        return None
    mult = _COMPACT_USD_MULT.get(s[-1].upper(), 1.0)
    num_s = s[:-1] if mult > 1.0 else s
    val = safe_float(num_s, default=None)
    return (val * mult) if val is not None else None


def _check_range_pos(
    violations: list[InvariantViolation],
    code: str,
    sym: str,
    label: str,
    px: float,
    lo: Optional[float],
    hi: Optional[float],
    pos: Optional[float],
    tol_pct: float,
) -> None:
    if lo is None or hi is None or pos is None or hi <= lo or px <= 0.0:
        return
    expected = clamp_float(((px - lo) / (hi - lo)) * 100.0, 0.0, 100.0)
    if abs(pos - expected) > tol_pct:
        violations.append(
            InvariantViolation(
                code=code,
                subject=f"{sym}:{label}",
                detail=f"reported pos={pos:.2f}% vs expected {expected:.2f}% (px={px:g}, range=[{lo:g}..{hi:g}])",
            )
        )


def audit_snapshot_invariants(
    snapshot: Any,
    *,
    non_tradable_symbols: Collection[str] = (),
    nav_tol_usd: float = 0.05,
    arb_tol_bps: float = 0.3,
    range_tol_pct: float = 1.0,
    breakdown_tol_frac: float = 0.05,
    gateway_min_usd: float = 0.0,
) -> list[InvariantViolation]:
    """Audit mathematical and structural invariants on a ``TelemetrySensorSnapshot`` or serialized dict."""
    violations: list[InvariantViolation] = []
    non_trad_u = {str(s).strip().upper() for s in non_tradable_symbols if str(s).strip()}

    if isinstance(snapshot, Mapping):
        pt = snapshot.get("prices_and_trends") or {}
        micro = snapshot.get("microstructure") or {}
        ps = snapshot.get("portfolio_summary") or {}
    else:
        pt = getattr(snapshot, "prices_and_trends", None) or {}
        micro = getattr(snapshot, "microstructure", None) or {}
        ps = getattr(snapshot, "portfolio_summary", None) or {}

    if not isinstance(pt, Mapping):
        pt = {}
    if not isinstance(micro, Mapping):
        micro = {}
    if not isinstance(ps, Mapping):
        ps = {}

    # 1. Portfolio NAV conservation
    pr = ps.get("portfolio_risk_engine") if isinstance(ps.get("portfolio_risk_engine"), Mapping) else ps.get("portfolio_risk")
    if isinstance(pr, Mapping):
        nav = safe_nonneg_float(
            pr.get("unified_nav_usd")
            or pr.get("crypto_nav_usd")
            or pr.get("equities_nav_usd")
            or pr.get("total_nav_usd"),
            default=None,
        )
        inv = safe_nonneg_float(
            pr.get("invested_exposure_usd") if pr.get("invested_exposure_usd") is not None else pr.get("invested_usd"),
            default=None,
        )
        free_c = safe_nonneg_float(
            pr.get("global_free_cash_usd") if pr.get("global_free_cash_usd") is not None else pr.get("free_cash_usd"),
            default=None,
        )
        locked_q = safe_nonneg_float(pr.get("locked_quote_usd"), default=0.0)
        if nav is not None and inv is not None and free_c is not None:
            comp_sum = inv + free_c + (locked_q or 0.0)
            if abs(nav - comp_sum) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "NAV_CONSERVATION",
                        "portfolio_risk",
                        f"nav={nav:.2f} != invested({inv:.2f}) + free_cash({free_c:.2f}) + locked({locked_q:.2f}) = {comp_sum:.2f}",
                    )
                )
        v_map = (
            pr.get("venue_nav_breakdown_usd")
            if isinstance(pr.get("venue_nav_breakdown_usd"), Mapping)
            else pr.get("venue_nav_usd")
        )
        if nav is not None and isinstance(v_map, Mapping) and v_map:
            v_sum = sum(safe_nonneg_float(v, default=0.0) or 0.0 for v in v_map.values())
            if abs(nav - v_sum) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "NAV_CONSERVATION",
                        "portfolio_risk.venues",
                        f"nav={nav:.2f} != sum(venues)={v_sum:.2f}",
                    )
                )

    # 2 & 3. Non-tradable ghost exclusion & free_quote_breakdown reconciliation
    tpos = ps.get("tradable_positions")
    if isinstance(tpos, Sequence) and not isinstance(tpos, (str, bytes)):
        seen_vs: set[tuple[str, str]] = set()
        for pos in tpos:
            if not isinstance(pos, Mapping):
                continue
            ven = normalize_venue_name(pos.get("venue"))
            sym = str(pos.get("symbol") or "").strip()
            sym_u = sym.upper()
            if sym_u and sym_u in non_trad_u:
                violations.append(
                    InvariantViolation(
                        "NO_NON_TRADABLE_GHOSTS",
                        f"{ven}:{sym}",
                        "disabled/observe_only registry symbol leaked into tradable_positions",
                    )
                )
            if sym_u and (ven, sym_u) in seen_vs:
                violations.append(
                    InvariantViolation(
                        "DUPLICATE_TRADABLE_POSITION",
                        f"{ven}:{sym}",
                        "duplicate (venue, symbol) entry in tradable_positions",
                    )
                )
            seen_vs.add((ven, sym_u))
            fq = safe_nonneg_float(pos.get("free_quote"), default=None)
            bd = pos.get("free_quote_breakdown")
            if fq is not None and isinstance(bd, Mapping) and bd:
                bd_sum = sum(safe_nonneg_float(v, default=0.0) or 0.0 for v in bd.values())
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "FREE_QUOTE_BREAKDOWN_CONSISTENCY",
                            f"{ven}:{sym}",
                            f"free_quote={fq:.2f} diverges from sum(breakdown)={bd_sum:.2f}",
                        )
                    )

    # 4. Cross-venue arbitrage symmetry
    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        peer = clean_scalar_str(minfo.get("cross_venue_peer"))
        arb_a = safe_float(minfo.get("cross_venue_arb_bps"), default=None)
        if not peer or arb_a is None:
            continue
        peer_m = casefold_get(micro, peer)
        if not isinstance(peer_m, Mapping):
            continue
        peer_back = clean_scalar_str(peer_m.get("cross_venue_peer"))
        arb_b = safe_float(peer_m.get("cross_venue_arb_bps"), default=None)
        if peer_back and peer_back.upper() == str(sym).upper() and arb_b is not None:
            if str(sym).upper() < str(peer).upper() and abs(arb_a + arb_b) > arb_tol_bps:
                violations.append(
                    InvariantViolation(
                        "CROSS_VENUE_ARB_SYMMETRY",
                        f"{sym}<->{peer}",
                        f"arb({sym})={fmt_signed(arb_a, 1)}bps vs arb({peer})={fmt_signed(arb_b, 1)}bps (sum={fmt_signed(arb_a + arb_b, 1)}bps)",
                    )
                )

    # 5, 6 & 7. Price range math, swing wave bounds, closed-equity hygiene, and orderbook wall ordering
    for sym, pinfo in pt.items():
        if not isinstance(pinfo, Mapping):
            continue
        px = safe_pos_float(pinfo.get("price"), default=None)
        if px is None:
            continue
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "24h",
            px,
            safe_pos_float(pinfo.get("low_24h"), default=None),
            safe_pos_float(pinfo.get("high_24h"), default=None),
            safe_float(pinfo.get("pos_24h_pct"), default=None),
            range_tol_pct,
        )
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "7d",
            px,
            safe_pos_float(pinfo.get("low_7d"), default=None),
            safe_pos_float(pinfo.get("high_7d"), default=None),
            safe_float(pinfo.get("pos_7d_pct"), default=None),
            range_tol_pct,
        )
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "15m",
            px,
            safe_pos_float(pinfo.get("low_15m"), default=None),
            safe_pos_float(pinfo.get("high_15m"), default=None),
            safe_float(pinfo.get("pos_15m_pct"), default=None),
            range_tol_pct,
        )
        maj_hi = safe_pos_float(
            pinfo.get("major_swing_high_px") if pinfo.get("major_swing_high_px") is not None else pinfo.get("major_high"),
            default=None,
        )
        maj_lo = safe_pos_float(
            pinfo.get("major_swing_low_px") if pinfo.get("major_swing_low_px") is not None else pinfo.get("major_low"),
            default=None,
        )
        if maj_hi is not None and px > maj_hi * 1.001:
            violations.append(
                InvariantViolation(
                    "SWING_WAVE_EXTREMES",
                    str(sym),
                    f"live price={px:g} exceeds major_high={maj_hi:g}",
                )
            )
        if maj_lo is not None and px < maj_lo * 0.999:
            violations.append(
                InvariantViolation(
                    "SWING_WAVE_EXTREMES",
                    str(sym),
                    f"live price={px:g} below major_low={maj_lo:g}",
                )
            )
        if has_equity_suffix(str(sym)):
            if leaked_eq := [k_s for k_s in stale_closed_equity_metric_keys(pinfo.get("quote_age_min")) if pinfo.get(k_s) is not None]:
                violations.append(
                    InvariantViolation(
                        "CLOSED_EQUITY_STALE_METRICS",
                        str(sym),
                        f"quote_age_min={pinfo.get('quote_age_min')} leaked stale keys: {', '.join(leaked_eq)}",
                    )
                )

    cmc_bw = (ps.get("cross_market_context") or {}) if isinstance(ps.get("cross_market_context"), Mapping) else {}
    eq_bw_map = cmc_bw.get("equity_bellwethers") if isinstance(cmc_bw.get("equity_bellwethers"), Mapping) else {}
    for bw_sym, bw_info in eq_bw_map.items():
        if isinstance(bw_info, Mapping) and has_equity_suffix(str(bw_sym)):
            if leaked_bw := [k_s for k_s in stale_closed_equity_metric_keys(bw_info.get("quote_age_min")) if bw_info.get(k_s) is not None]:
                violations.append(
                    InvariantViolation(
                        "CLOSED_EQUITY_STALE_METRICS",
                        f"bellwether:{bw_sym}",
                        f"quote_age_min={bw_info.get('quote_age_min')} leaked stale keys: {', '.join(leaked_bw)}",
                    )
                )

    gw_min_snap = safe_nonneg_float(gateway_min_usd, default=0.0) or 0.0
    if gw_min_snap > 0.0:
        for pos in ps.get("tradable_positions") or ():
            if not isinstance(pos, Mapping):
                continue
            p_ven = normalize_venue_name(pos.get("venue")) or str(pos.get("venue") or "").strip().lower()
            p_sym = str(pos.get("symbol") or "").strip()
            for cap_k in ("effective_max_buy_usd", "effective_max_sell_usd"):
                cap_v = safe_nonneg_float(pos.get(cap_k), default=None)
                if cap_v is not None and 0.0 < cap_v < gw_min_snap - 1e-6:
                    violations.append(
                        InvariantViolation(
                            "SNAPSHOT_SUB_GATEWAY_MIN_CAP",
                            f"{p_ven}:{p_sym}:{cap_k}",
                            f"{cap_k}=${cap_v:.2f} is below gateway_min_usd=${gw_min_snap:.2f}",
                        )
                    )

    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        if has_equity_suffix(str(sym)):
            for px_k in ("best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px"):
                if (pv := safe_pos_float(minfo.get(px_k), default=None)) is not None:
                    frac = f"{pv:.8f}".rstrip("0").partition(".")[2]
                    if len(frac) > 4:
                        violations.append(
                            InvariantViolation(
                                "EQUITY_OB_PRECISION",
                                f"{sym}:{px_k}",
                                f"equity orderbook price {px_k}={pv} exceeds 4 decimal places",
                            )
                        )
        ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=None)
        if ob_mid is None:
            continue
        bid_w = safe_pos_float(minfo.get("bid_wall_px"), default=None)
        ask_w = safe_pos_float(minfo.get("ask_wall_px"), default=None)
        if bid_w is not None and bid_w > ob_mid * 1.001:
            violations.append(
                InvariantViolation(
                    "ORDERBOOK_WALL_ORDERING",
                    str(sym),
                    f"bid_wall_px={bid_w:g} > ob_mid={ob_mid:g}",
                )
            )
        if ask_w is not None and ask_w < ob_mid * 0.999:
            violations.append(
                InvariantViolation(
                    "ORDERBOOK_WALL_ORDERING",
                    str(sym),
                    f"ask_wall_px={ask_w:g} < ob_mid={ob_mid:g}",
                )
            )

    _audit_snapshot_positions_and_cycles(violations, ps, pt)
    _audit_snapshot_price_geometry(violations, pt)
    _audit_snapshot_micro_spread_and_walls(violations, micro)
    return violations


def _audit_snapshot_positions_and_cycles(
    violations: list[InvariantViolation],
    ps: Mapping[str, Any],
    pt: Mapping[str, Any],
) -> None:
    """Validate position valuation/PnL/peak math, cycle avg_cost parity, open-order dist%, and xStock premium."""
    pos_avg_by_vs: dict[tuple[str, str], float] = {}
    tpos = ps.get("tradable_positions")
    if isinstance(tpos, Sequence) and not isinstance(tpos, (str, bytes)):
        for pos in tpos:
            if not isinstance(pos, Mapping):
                continue
            ven = normalize_venue_name(pos.get("venue")) or str(pos.get("venue") or "").strip().lower()
            sym = str(pos.get("symbol") or "").strip()
            px = safe_pos_float(pos.get("price"), default=None)
            if not sym or px is None:
                continue
            fb = safe_nonneg_float(pos.get("free_base"), default=None)
            val = safe_nonneg_float(pos.get("value_usd"), default=None)
            if fb is not None and val is not None and abs(fb * px - val) > max(0.05, val * 0.005):
                violations.append(
                    InvariantViolation(
                        "POSITION_VALUE_AND_ANCHOR_MATH",
                        f"{ven}:{sym}:value_usd",
                        f"free_base({fb:g})*px({px:g})={fb * px:.2f} != value_usd={val:.2f}",
                    )
                )
            lb_q_s = safe_nonneg_float(pos.get("locked_base_qty"), default=None)
            lb_u_s = safe_nonneg_float(pos.get("locked_base_usd"), default=None)
            if lb_q_s is not None and lb_u_s is not None and abs(lb_q_s * px - lb_u_s) > max(0.05, lb_u_s * 0.005):
                violations.append(
                    InvariantViolation(
                        "POSITION_VALUE_AND_ANCHOR_MATH",
                        f"{ven}:{sym}:locked_base_usd",
                        f"locked_base_qty({lb_q_s:g})*px({px:g})={lb_q_s * px:.2f} != locked_base_usd={lb_u_s:.2f}",
                    )
                )
            avg_b = safe_pos_float(pos.get("active_avg_buy_px"), default=None)
            if avg_b is not None:
                pos_avg_by_vs[(ven, sym.upper())] = avg_b
                pnl_b = safe_float(
                    pos.get("pnl_vs_avg_pct") if pos.get("pnl_vs_avg_pct") is not None else pos.get("pnl_vs_avg_buy_pct"),
                    default=None,
                )
                if pnl_b is not None:
                    exp_p = ((px - avg_b) / avg_b) * 100.0
                    if abs(exp_p - pnl_b) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "POSITION_VALUE_AND_ANCHOR_MATH",
                                f"{ven}:{sym}:pnl_vs_avg",
                                f"reported pnl={fmt_signed(pnl_b, 2)}% != expected {fmt_signed(exp_p, 2)}% (px={px:g}, avg={avg_b:g})",
                            )
                        )
            pk = safe_pos_float(
                pos.get("trailing_peak") if pos.get("trailing_peak") is not None else pos.get("pos_peak"),
                default=None,
            )
            if pk is not None:
                if px > pk * 1.001:
                    violations.append(
                        InvariantViolation(
                            "POSITION_VALUE_AND_ANCHOR_MATH",
                            f"{ven}:{sym}:peak",
                            f"live price={px:g} exceeds peak={pk:g}",
                        )
                    )
                dd_pk = safe_float(pos.get("dd_from_peak_pct"), default=None)
                if dd_pk is not None:
                    exp_dd = ((px - pk) / pk) * 100.0
                    if abs(exp_dd - dd_pk) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "POSITION_VALUE_AND_ANCHOR_MATH",
                                f"{ven}:{sym}:dd_from_peak",
                                f"reported dd={fmt_signed(dd_pk, 2)}% != expected {fmt_signed(exp_dd, 2)}% (px={px:g}, peak={pk:g})",
                            )
                        )

    for cyc_k, cmap in ps.items():
        if not (isinstance(cyc_k, str) and cyc_k.endswith("_cycles") and isinstance(cmap, Mapping)):
            continue
        ven_name = normalize_venue_name(cyc_k[:-7])
        if not ven_name:
            continue
        for csym, cdata in cmap.items():
            if not isinstance(cdata, Mapping):
                continue
            c_avg = safe_pos_float(cdata.get("avg_cost_px"), default=None)
            p_avg = pos_avg_by_vs.get((ven_name, str(csym).strip().upper()))
            if c_avg is not None and p_avg is not None and abs(c_avg - p_avg) > max(1e-4, p_avg * 1e-4):
                violations.append(
                    InvariantViolation(
                        "POSITION_VALUE_AND_ANCHOR_MATH",
                        f"{ven_name}:{csym}:cycle_avg_cost",
                        f"cycle avg_cost_px={c_avg:g} != position active_avg_buy_px={p_avg:g}",
                    )
                )

    raw_oo = ps.get("open_limit_orders")
    if not isinstance(raw_oo, Sequence) or isinstance(raw_oo, (str, bytes)):
        raw_oo = ps.get("open_orders") if isinstance(ps.get("open_orders"), Sequence) else ()
    for oo in raw_oo:
        if not isinstance(oo, Mapping):
            continue
        sym_o = str(oo.get("symbol") or "").strip()
        px_o = safe_pos_float(oo.get("price"), default=None)
        dist_o = safe_float(oo.get("dist_from_spot_pct"), default=None)
        p_ref = casefold_get(pt, sym_o) if sym_o else None
        spot_p = safe_pos_float(p_ref.get("price"), default=None) if isinstance(p_ref, Mapping) else None
        if px_o is not None and spot_p is not None and dist_o is not None:
            exp_d = ((px_o - spot_p) / spot_p) * 100.0
            if abs(exp_d - dist_o) > 0.20:
                violations.append(
                    InvariantViolation(
                        "OPEN_ORDER_DIST_MATH",
                        f"{normalize_venue_name(oo.get('venue'))}:{sym_o}",
                        f"reported dist={fmt_signed(dist_o, 2)}% != expected {fmt_signed(exp_d, 2)}% (limit={px_o:g}, spot={spot_p:g})",
                    )
                )

    xs = next(
        (v for k, v in ps.items() if isinstance(k, str) and k.endswith("_xstocks") and isinstance(v, Mapping)),
        None,
    )
    if isinstance(xs, Mapping) and (rep_prem := safe_float(xs.get("xstock_vs_underlying_pct"), default=None)) is not None:
        u_px = safe_pos_float(xs.get("underlying_px"), default=None)
        pair_s = str(xs.get("listed_pair") or "SPCXxUSD").strip()
        x_ref = casefold_get(pt, pair_s)
        x_px = (
            safe_pos_float(x_ref.get("price"), default=None)
            if isinstance(x_ref, Mapping)
            else safe_pos_float(xs.get("xstock_px"), default=None)
        )
        if x_px is not None and u_px is not None:
            exp_x = ((x_px - u_px) / u_px) * 100.0
            if abs(exp_x - rep_prem) > 0.15:
                violations.append(
                    InvariantViolation(
                        "XSTOCK_PREMIUM_MATH",
                        pair_s,
                        f"reported xstock_vs_underlying={fmt_signed(rep_prem, 2)}% != expected {fmt_signed(exp_x, 2)}% (x={x_px:g}, u={u_px:g})",
                    )
                )


def _audit_snapshot_price_geometry(
    violations: list[InvariantViolation],
    pt: Mapping[str, Any],
) -> None:
    """Validate vs_btc deltas, range bounds/dd_high/bounce_low/span, swing bounds, bottom_6h, and VWAP z sign."""
    btc_ref = casefold_get(pt, "BTCUSDC")
    btc_map = btc_ref if isinstance(btc_ref, Mapping) else {}
    for sym, pinfo in pt.items():
        if not isinstance(pinfo, Mapping):
            continue
        sym_s = str(sym)
        px = safe_pos_float(pinfo.get("price"), default=None)
        if px is None:
            continue
        if sym_s.upper() != "BTCUSDC" and btc_map:
            for hor in ("15m", "1h", "6h", "24h", "7d"):
                vb = safe_float(pinfo.get(f"vs_btc_{hor}_pct"), default=None)
                sc = safe_float(pinfo.get(f"chg_{hor}_pct"), default=None)
                bc = safe_float(btc_map.get(f"chg_{hor}_pct"), default=None)
                if vb is not None and sc is not None and bc is not None and abs(vb - (sc - bc)) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "VS_BTC_MATH",
                            f"{sym_s}:{hor}",
                            f"reported vs_btc={fmt_signed(vb, 2)}% != sym({fmt_signed(sc, 2)}%)-btc({fmt_signed(bc, 2)}%)={fmt_signed(sc - bc, 2)}%",
                        )
                    )
        for tag in ("15m", "24h", "7d"):
            lo = safe_pos_float(pinfo.get(f"low_{tag}"), default=None)
            hi = safe_pos_float(pinfo.get(f"high_{tag}"), default=None)
            if lo is None or hi is None:
                continue
            if lo > hi * 1.0001 or px < lo * 0.999 or px > hi * 1.001:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}",
                        f"px={px:g} inconsistent with {tag} range [{lo:g}, {hi:g}]",
                    )
                )
            dd_h = safe_float(pinfo.get(f"dd_{tag}_high_pct"), default=None)
            if dd_h is not None and abs(dd_h - ((px - hi) / hi) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:dd_high",
                        f"reported dd_high={fmt_signed(dd_h, 2)}% != expected {fmt_signed(((px - hi) / hi) * 100.0, 2)}%",
                    )
                )
            bn_l = safe_float(pinfo.get(f"bounce_{tag}_low_pct"), default=None)
            if bn_l is not None and abs(bn_l - ((px - lo) / lo) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:bounce_low",
                        f"reported bounce_low={fmt_signed(bn_l, 2)}% != expected {fmt_signed(((px - lo) / lo) * 100.0, 2)}%",
                    )
                )
            sp = safe_float(pinfo.get(f"span_{tag}_pct"), default=None)
            if sp is not None and abs(sp - ((hi - lo) / lo) * 100.0) > 0.25:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:span",
                        f"reported span={sp:.2f}% != expected {((hi - lo) / lo) * 100.0:.2f}%",
                    )
                )
        sw_l = safe_pos_float(pinfo.get("swing_low_px"), default=None)
        sw_h = safe_pos_float(pinfo.get("swing_high_px"), default=None)
        if sw_l is not None and sw_h is not None and (sw_l > sw_h * 1.0001 or px < sw_l * 0.999 or px > sw_h * 1.001):
            violations.append(
                InvariantViolation(
                    "RANGE_BOUNDS_AND_EXTREMES_MATH",
                    f"{sym_s}:swing_wave",
                    f"px={px:g} outside swing_wave=[{sw_l:g}, {sw_h:g}]",
                )
            )
        if b6 := clean_scalar_str(pinfo.get("bottom_structure_6h")):
            p6l = safe_pos_float(pinfo.get("prior_6h_low_px"), default=None)
            r1l = safe_pos_float(pinfo.get("recent_1h_low_px"), default=None)
            l15 = safe_pos_float(pinfo.get("low_15m"), default=None)
            l24 = safe_pos_float(pinfo.get("low_24h"), default=None)
            if r1l is not None and l15 is not None and r1l > l15 * 1.0005:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"recent_1h_low_px={r1l:g} > low_15m={l15:g}",
                    )
                )
            if p6l is not None and r1l is not None and l24 is not None and min(p6l, r1l) < l24 * 0.999:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"min(prior_6h_low={p6l:g}, 1h_low={r1l:g}) < low_24h={l24:g}",
                    )
                )
            if b6 == "HIGHER_LOW_DOUBLE_BOTTOM" and p6l is not None and r1l is not None and r1l < p6l * 0.9995:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"HIGHER_LOW_DOUBLE_BOTTOM but 1h_low={r1l:g} < prior_6h_low={p6l:g}",
                    )
                )
        dv = safe_float(pinfo.get("dist_vs_session_vwap_pct"), default=None)
        vz = safe_float(pinfo.get("session_vwap_z"), default=None)
        if dv is not None and vz is not None and dv * vz < -0.02:
            violations.append(
                InvariantViolation(
                    "SESSION_VWAP_SIGN_MATH",
                    sym_s,
                    f"dist_vs_session_vwap_pct={fmt_signed(dv, 2)}% has opposite sign to session_vwap_z={fmt_signed(vz, 2)}",
                )
            )


def _audit_snapshot_micro_spread_and_walls(
    violations: list[InvariantViolation],
    micro: Mapping[str, Any],
) -> None:
    """Validate non-crossed BBO, spread_bps formula, and wall distance percentages vs ob_mid."""
    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        sym_s = str(sym)
        bb = safe_pos_float(minfo.get("best_bid"), default=None)
        ba = safe_pos_float(minfo.get("best_ask"), default=None)
        if bb is not None and ba is not None:
            if bb > ba * 1.00001:
                violations.append(
                    InvariantViolation(
                        "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                        f"{sym_s}:bbo",
                        f"crossed book best_bid={bb:g} > best_ask={ba:g}",
                    )
                )
            sp_bps = safe_float(minfo.get("spread_bps"), default=None)
            if sp_bps is not None:
                exp_sp = ((ba - bb) / ((ba + bb) / 2.0)) * 10000.0
                if abs(exp_sp - sp_bps) > max(1.0, abs(exp_sp) * 0.05):
                    violations.append(
                        InvariantViolation(
                            "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                            f"{sym_s}:spread_bps",
                            f"reported spread={sp_bps:.2f}bps != expected {exp_sp:.2f}bps (bid={bb:g}, ask={ba:g})",
                        )
                    )
        mid = safe_pos_float(minfo.get("ob_mid_price"), default=None)
        if mid is not None:
            for wk in ("bid_wall", "ask_wall"):
                wpx = safe_pos_float(minfo.get(f"{wk}_px"), default=None)
                wdist = safe_float(minfo.get(f"{wk}_dist_pct"), default=None)
                if wpx is not None and wdist is not None:
                    exp_wd = ((wpx - mid) / mid) * 100.0
                    if abs(exp_wd - wdist) > 0.25:
                        violations.append(
                            InvariantViolation(
                                "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                                f"{sym_s}:{wk}_dist",
                                f"reported {wk}_dist={fmt_signed(wdist, 2)}% != expected {fmt_signed(exp_wd, 2)}% (wall={wpx:g}, mid={mid:g})",
                            )
                        )


def audit_scorecard_temporal_invariants(
    intents: Sequence[Mapping[str, Any]],
    *,
    t_early: float,
    t_late: float,
    scope: str = "crypto",
    lookback_hours: float = 48.0,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> list[InvariantViolation]:
    """Verify LIFO closed pairs inside ``[t_late - lookback, t_early]`` stay immutable when no new fills occur."""
    if t_late <= t_early:
        return []
    items = [it for it in intents if isinstance(it, Mapping)]
    canceled = _collect_canceled_order_ids(items)

    def _rows_up_to(ts_limit: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for it in items:
            if not _is_filled_ai_intent(it, canceled):
                continue
            sym = clean_scalar_str(it.get("symbol")) or ""
            ven = normalize_venue_name(clean_scalar_str(it.get("venue")) or "")
            if scope and scope.strip().lower() != "all" and not matches_asset_scope(sym, ven, scope):
                continue
            side = extract_intent_side(it)
            if side not in ("BUY", "SELL"):
                continue
            ts_s = _extract_intent_exec_ts(it)
            px = _extract_intent_exec_price(it)
            if ts_s <= 0.0 or ts_s > ts_limit + 60.0 or px <= 0.0:
                continue
            rows.append(
                build_normalized_filled_ai_intent_row(
                    it,
                    venue=ven,
                    symbol=sym,
                    side=side,
                    ts_s=ts_s,
                    px=px,
                    fee_resolver=fee_resolver,
                )
            )
        return rows

    rows_early = _rows_up_to(t_early)
    rows_late = _rows_up_to(t_late)
    if len(rows_late) != len(rows_early):
        return []

    lb_sec = max(3600.0, (safe_pos_float(lookback_hours, default=48.0) or 48.0) * 3600.0)
    cut_early = t_early - lb_sec
    cut_late = t_late - lb_sec
    c_early, _, _ = match_filled_intent_rows_lifo_in_window(rows_early, cutoff_ts=cut_early)
    c_late, _, _ = match_filled_intent_rows_lifo_in_window(rows_late, cutoff_ts=cut_late)

    def _sig(pair: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[str, str, str, float, float, float]:
        kind, er, xr = pair
        return (
            kind,
            str(er.get("venue") or ""),
            str(er.get("symbol") or ""),
            round(safe_float(er.get("px"), default=0.0) or 0.0, 6),
            round(safe_float(xr.get("px"), default=0.0) or 0.0, 6),
            round(safe_float(xr.get("ts"), default=0.0) or 0.0, 3),
        )

    early_overlapping = sorted(_sig(p) for p in c_early if (safe_float(p[2].get("ts"), default=0.0) or 0.0) >= cut_late)
    late_sigs = sorted(_sig(p) for p in c_late)
    if early_overlapping != late_sigs:
        return [
            InvariantViolation(
                "SCORECARD_TEMPORAL_STABILITY",
                f"{scope}:{lookback_hours:g}h",
                f"closed pairs mutated without new fills: early={early_overlapping} vs late={late_sigs}",
            )
        ]
    return []


def audit_rendered_prompt_text(
    prompt_text: str,
    *,
    non_tradable_symbols: Collection[str] = (),
    nav_tol_usd: float = 0.05,
    arb_tol_bps: float = 0.3,
    range_tol_pct: float = 1.2,
    breakdown_tol_frac: float = 0.05,
    gateway_min_usd: float = 0.0,
) -> list[InvariantViolation]:
    """Audit mathematical and structural invariants directly on a rendered LLM prompt string."""
    violations: list[InvariantViolation] = []
    non_trad_u = {str(s).strip().upper() for s in non_tradable_symbols if str(s).strip()}
    gw_min_f = safe_nonneg_float(gateway_min_usd, default=0.0) or 0.0

    # 1. Section 1 NAV conservation
    nav_invested: Optional[float] = None
    nav_free_cash: Optional[float] = None
    nav_locked_quote: Optional[float] = None
    if (m_nav := _NAV_LINE_RE.search(prompt_text)) is not None:
        nav = _parse_money(m_nav.group(1))
        inv = _parse_money(m_nav.group(2))
        free_c = _parse_money(m_nav.group(3))
        locked_q = _parse_money(m_nav.group(4)) if m_nav.group(4) is not None else 0.0
        nav_invested, nav_free_cash = inv, free_c
        nav_locked_quote = locked_q if m_nav.group(4) is not None else None
        if nav is not None and inv is not None and free_c is not None and locked_q is not None:
            comp = inv + free_c + locked_q
            if abs(nav - comp) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "PROMPT_NAV_CONSERVATION",
                        "Section3C.PortfolioRisk",
                        f"nav={nav:.2f} != invested+free+locked={comp:.2f}",
                    )
                )
        v_sum = sum(_parse_money(v_s) or 0.0 for _, v_s in _VENUE_VAL_RE.findall(m_nav.group(5)))
        if nav is not None and v_sum > 0.0 and abs(nav - v_sum) > nav_tol_usd:
            violations.append(
                InvariantViolation(
                    "PROMPT_NAV_CONSERVATION",
                    "Section3C.Venues",
                    f"nav={nav:.2f} != sum(venues)={v_sum:.2f}",
                )
            )
        if nav is not None and nav > 0.0:
            ln_start = prompt_text.rfind("\n", 0, m_nav.start()) + 1
            ln_end = prompt_text.find("\n", m_nav.end())
            nav_line = prompt_text[ln_start:] if ln_end < 0 else prompt_text[ln_start:ln_end]
            for lbl_p, pat_p, tol_p in (
                ("free_cash_pct", _PRE_FREE_PCT_RE, 0.2),
                ("locked_quote_pct", _PRE_LOCKED_PCT_RE, 0.2),
                ("intraday_var_95_pct", _PRE_VAR_PCT_RE, 0.1),
                ("cvar_975_pct", _PRE_CVAR_PCT_RE, 0.1),
            ):
                if (m_pp := pat_p.search(nav_line)) is not None:
                    amt_p = _parse_money(m_pp.group(1))
                    rep_p = safe_float(m_pp.group(2), default=None)
                    if amt_p is not None and rep_p is not None:
                        exp_p = (amt_p / nav) * 100.0
                        if abs(rep_p - exp_p) > tol_p:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_PRE_PERCENT_MATH",
                                    f"Section1.{lbl_p}",
                                    f"reported {lbl_p}={rep_p:.2f}% != {amt_p:.2f}/{nav:.2f}={exp_p:.2f}%",
                                )
                            )
            m_vp_l, m_cv_l = _PRE_VAR_PCT_RE.search(nav_line), _PRE_CVAR_PCT_RE.search(nav_line)
            if m_vp_l is not None and m_cv_l is not None:
                v_u, cv_u = _parse_money(m_vp_l.group(1)), _parse_money(m_cv_l.group(1))
                if v_u is not None and cv_u is not None and cv_u + 0.05 < v_u:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_PRE_PERCENT_MATH",
                            "Section1.cvar_vs_var",
                            f"cvar_975=${cv_u:.2f} < intraday_var_95=${v_u:.2f}",
                        )
                    )

    # 2 & 3. Section 1 live tradable instruments: ghost check + breakdown sum + precision/caps + position math
    px_by_vs: dict[tuple[str, str], float] = {}
    avg_by_vs: dict[tuple[str, str], float] = {}
    locked_base_by_vs: dict[tuple[str, str], float] = {}
    sum_pos_invested = 0.0
    pos_count = 0
    for m_inst in _SEC3C_INST_RE.finditer(prompt_text):
        ven, sym, rest = m_inst.group(1), m_inst.group(2), m_inst.group(3)
        if "free_" not in rest or "limits(" not in rest:
            continue
        pos_count += 1
        m_px_row = _PX_FIELD_RE.search(rest)
        px_r = safe_pos_float(m_px_row.group(1), default=None) if m_px_row else None
        val_u: Optional[float] = None
        lb_u_row: Optional[float] = None
        if (m_fbv := _FREE_BASE_VAL_RE.search(rest)) is not None:
            fb_q = safe_nonneg_float(m_fbv.group(1), default=None)
            val_u = _parse_money(m_fbv.group(2))
            if val_u is not None:
                sum_pos_invested += val_u
            if px_r is not None and fb_q is not None and val_u is not None and abs(fb_q * px_r - val_u) > max(0.05, val_u * 0.005):
                violations.append(
                    InvariantViolation(
                        "PROMPT_POSITION_MATH",
                        f"{ven}:{sym}:value_usd",
                        f"free_base({fb_q:g})*px({px_r:g})={fb_q * px_r:.2f} != reported ${val_u:.2f}",
                    )
                )
        if (m_lbu := _LOCKED_BASE_QTY_USD_RE.search(rest)) is not None:
            lb_q_r = safe_nonneg_float(m_lbu.group(1), default=None)
            lb_u_row = _parse_money(m_lbu.group(2))
            if lb_q_r is not None:
                locked_base_by_vs[(ven.lower(), sym.upper())] = lb_q_r
            if lb_u_row is not None:
                sum_pos_invested += lb_u_row
            if px_r is not None and lb_q_r is not None and lb_u_row is not None and abs(lb_q_r * px_r - lb_u_row) > max(0.05, lb_u_row * 0.005):
                violations.append(
                    InvariantViolation(
                        "PROMPT_POSITION_MATH",
                        f"{ven}:{sym}:locked_base_usd",
                        f"locked_base_qty({lb_q_r:g})*px({px_r:g})={lb_q_r * px_r:.2f} != reported ${lb_u_row:.2f}",
                    )
                )
        if px_r is not None:
            px_by_vs[(ven.lower(), sym.upper())] = px_r
            if (m_avg := _AVG_COST_PNL_RE.search(rest)) is not None:
                avg_v = safe_pos_float(m_avg.group(1), default=None)
                rep_pnl = safe_float(m_avg.group(2), default=None)
                if avg_v is not None:
                    avg_by_vs[(ven.lower(), sym.upper())] = avg_v
                    if rep_pnl is not None:
                        exp_pnl = ((px_r - avg_v) / avg_v) * 100.0
                        if abs(exp_pnl - rep_pnl) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_MATH",
                                    f"{ven}:{sym}:pnl_vs_cost",
                                    f"reported pnl_vs_cost={fmt_signed(rep_pnl, 2)}% != expected {fmt_signed(exp_pnl, 2)}% (px={px_r:g}, avg={avg_v:g})",
                                )
                            )
            if (m_pk := _PEAK_DD_RE.search(rest)) is not None:
                pk_v = safe_pos_float(m_pk.group(1), default=None)
                rep_dd = safe_float(m_pk.group(2), default=None)
                if pk_v is not None:
                    if px_r > pk_v * 1.001:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_POSITION_MATH",
                                f"{ven}:{sym}:peak_px",
                                f"px={px_r:g} exceeds peak_px={pk_v:g}",
                            )
                        )
                    if rep_dd is not None:
                        exp_dd = ((px_r - pk_v) / pk_v) * 100.0
                        if abs(exp_dd - rep_dd) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_MATH",
                                    f"{ven}:{sym}:dd_from_peak",
                                    f"reported dd_from_peak={fmt_signed(rep_dd, 2)}% != expected {fmt_signed(exp_dd, 2)}% (px={px_r:g}, peak={pk_v:g})",
                                )
                            )
            for anc_k, anc_pat in (
                ("last_buy_px", _POS_LAST_BUY_RE),
                ("last_sell_px", _POS_LAST_SELL_RE),
                ("lifo_avg", _POS_LIFO_AVG_RE),
            ):
                if (m_anc := anc_pat.search(rest)) is not None:
                    anc_px = safe_pos_float(m_anc.group(1), default=None)
                    rep_ap = safe_float(m_anc.group(2), default=None)
                    if anc_px is not None and rep_ap is not None:
                        exp_ap = ((px_r - anc_px) / anc_px) * 100.0
                        if abs(exp_ap - rep_ap) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                                    f"{ven}:{sym}:{anc_k}",
                                    f"reported {anc_k} delta={fmt_signed(rep_ap, 2)}% != expected {fmt_signed(exp_ap, 2)}% (px={px_r:g}, anc={anc_px:g})",
                                )
                            )
            if (m_lots := _POS_OPEN_LOTS_RE.search(rest)) is not None:
                parsed_lots: list[tuple[float, float]] = []
                for m_lt in _POS_LOT_ITEM_RE.finditer(m_lots.group(1)):
                    lq = safe_pos_float(m_lt.group(1), default=None)
                    lp = safe_pos_float(m_lt.group(2), default=None)
                    lr = safe_float(m_lt.group(3), default=None)
                    if lq is not None and lp is not None:
                        parsed_lots.append((lq, lp))
                        if lr is not None:
                            exp_lr = ((px_r - lp) / lp) * 100.0
                            if abs(exp_lr - lr) > 0.15:
                                violations.append(
                                    InvariantViolation(
                                        "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                                        f"{ven}:{sym}:lifo_lot",
                                        f"open_lifo_lots {lq:g}@{lp:g} pnl={fmt_signed(lr, 2)}% != expected {fmt_signed(exp_lr, 2)}%",
                                    )
                                )
                if 0 < len(parsed_lots) < 4 and (m_la := _POS_LIFO_AVG_RE.search(rest)) is not None:
                    rep_la = safe_pos_float(m_la.group(1), default=None)
                    tot_lq = sum(q for q, _ in parsed_lots)
                    if rep_la is not None and tot_lq > 0.0:
                        wavg_la = sum(q * p for q, p in parsed_lots) / tot_lq
                        if abs(wavg_la - rep_la) / rep_la > 0.005:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                                    f"{ven}:{sym}:lifo_avg_weighted",
                                    f"reported lifo_avg={rep_la:g} != weighted(open_lifo_lots)={wavg_la:g}",
                                )
                            )
        if sym.upper() in non_trad_u:
            violations.append(
                InvariantViolation(
                    "PROMPT_NO_NON_TRADABLE_GHOSTS",
                    f"{ven}:{sym}",
                    "disabled/observe_only symbol rendered in Live Tradable Spot Instruments",
                )
            )
        for m_bd in _FREE_QUOTE_BD_RE.finditer(rest):
            q_name = m_bd.group(1)
            fq = _parse_money(m_bd.group(2))
            bd_vals = [_parse_money(v_s) for _, v_s in _BD_ITEM_RE.findall(m_bd.group(3))]
            bd_nums = [v for v in bd_vals if v is not None]
            if fq is not None and bd_nums:
                bd_sum = sum(bd_nums)
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FREE_QUOTE_BREAKDOWN",
                            f"{ven}:{sym}:{q_name}",
                            f"free_{q_name}={fq:.2f} diverges from breakdown sum={bd_sum:.2f}",
                        )
                    )
        if (m_lim := _LIMITS_RE.search(rest)) is not None:
            lim_s = m_lim.group(1)
            lim_nums = {k: safe_float(v, default=None) for k, v in _LIMIT_NUM_RE.findall(lim_s)}
            for k_lim, v_lim in lim_nums.items():
                if v_lim is not None and v_lim <= 0.0:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VENUE_PRECISION",
                            f"{ven}:{sym}:{k_lim}",
                            f"non-positive {k_lim}={v_lim}",
                        )
                    )
            if ProviderVenueMetadataMixin.prefers_static_precision(ven):
                px_c = px_by_vs.get((ven.lower(), sym.upper()), 0.0)
                _, _, _, _, exp_step, exp_tick, _ = (
                    ProviderVenueMetadataMixin.resolve_venue_default_fee_and_precision(ven, px_c)
                )
                for k_lim, exp_v in (("min_qty", exp_step), ("step", exp_step), ("tick", exp_tick)):
                    got_v = lim_nums.get(k_lim)
                    if got_v is not None and abs(got_v - exp_v) > 1e-9:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VENUE_PRECISION",
                                f"{ven}:{sym}:{k_lim}",
                                f"reported {k_lim}={got_v:g} != static venue precision {exp_v:g}",
                            )
                        )
            for cap_k, cap_s in _LIMIT_CAP_RE.findall(lim_s):
                cap_v = _parse_money(cap_s)
                if cap_v is not None and gw_min_f > 0.0 and 0.0 < cap_v < gw_min_f - 1e-6:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_SUB_GATEWAY_MIN_CAP",
                            f"{ven}:{sym}:{cap_k}",
                            f"{cap_k}=${cap_v:.2f} is below gateway_min_usd=${gw_min_f:.2f}",
                        )
                    )
                if cap_k == "max_sell" and cap_v is not None and val_u is not None and cap_v > val_u + max(0.05, val_u * 0.005):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                            f"{ven}:{sym}:max_sell",
                            f"limits max_sell=${cap_v:.2f} exceeds free value_usd=${val_u:.2f}",
                        )
                    )
                if cap_k == "replace_max_sell" and cap_v is not None and val_u is not None:
                    tot_pos_u = val_u + (lb_u_row or 0.0)
                    if cap_v > tot_pos_u + max(0.05, tot_pos_u * 0.005):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                                f"{ven}:{sym}:replace_max_sell",
                                f"limits replace_max_sell=${cap_v:.2f} exceeds total position value=${tot_pos_u:.2f}",
                            )
                        )

    if pos_count > 0 and nav_invested is not None and sum_pos_invested > nav_invested + max(0.25, nav_invested * 0.005):
        violations.append(
            InvariantViolation(
                "PROMPT_SECTION1_SUM_CONSERVATION",
                "Section1.InvestedSum",
                f"sum(tradable_positions)={sum_pos_invested:.2f} exceeds portfolio invested={nav_invested:.2f}",
            )
        )

    maker_bps_by_sym: dict[str, float] = {}
    for m_vspec in _VENUE_SPEC_ROW_RE.finditer(prompt_text):
        ven, sym, fee_cell = m_vspec.group(1), m_vspec.group(2), m_vspec.group(3)
        if (m_fp := _FEE_PAIR_RE.search(fee_cell)) is not None:
            if (mk_v := safe_nonneg_float(m_fp.group(1), default=None)) is not None:
                maker_bps_by_sym[sym.upper()] = mk_v
        lim_nums = {
            "min_qty": safe_float(m_vspec.group(4), default=None),
            "step": safe_float(m_vspec.group(5), default=None),
            "tick": safe_float(m_vspec.group(6), default=None),
        }
        for k_lim, v_lim in lim_nums.items():
            if v_lim is not None and v_lim <= 0.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_VENUE_PRECISION",
                        f"{ven}:{sym}:{k_lim}",
                        f"non-positive {k_lim}={v_lim}",
                    )
                )
        if ProviderVenueMetadataMixin.prefers_static_precision(ven):
            px_c = px_by_vs.get((ven.lower(), sym.upper()), 0.0)
            _, _, _, _, exp_step, exp_tick, _ = (
                ProviderVenueMetadataMixin.resolve_venue_default_fee_and_precision(ven, px_c)
            )
            for k_lim, exp_v in (("min_qty", exp_step), ("step", exp_step), ("tick", exp_tick)):
                got_v = lim_nums.get(k_lim)
                if got_v is not None and abs(got_v - exp_v) > 1e-9:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VENUE_PRECISION",
                            f"{ven}:{sym}:{k_lim}",
                            f"reported {k_lim}={got_v:g} != static venue precision {exp_v:g}",
                        )
                    )

    for m_cyc in _CYCLE_ANCHOR_LINE_RE.finditer(prompt_text):
        c_ven, c_sym = m_cyc.group(1).lower(), m_cyc.group(2).upper()
        c_qty = safe_pos_float(m_cyc.group(3), default=None)
        c_cost = _parse_money(m_cyc.group(4))
        c_avg = safe_pos_float(m_cyc.group(5), default=None)
        c_tail = m_cyc.group(6) or ""
        p_avg = avg_by_vs.get((c_ven, c_sym))
        if c_avg is not None and p_avg is not None and abs(c_avg - p_avg) > max(1e-4, p_avg * 1e-4):
            violations.append(
                InvariantViolation(
                    "PROMPT_CYCLE_ANCHOR_PARITY",
                    f"{c_ven}:{c_sym}",
                    f"cycle avg_cost={c_avg:g} != position avg_cost={p_avg:g}",
                )
            )
        if c_qty is not None and c_avg is not None and c_cost is not None:
            exp_cost = c_qty * c_avg
            if abs(exp_cost - c_cost) > max(0.05, exp_cost * 0.002):
                violations.append(
                    InvariantViolation(
                        "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                        f"{c_ven}:{c_sym}:cycle_cost",
                        f"cycle cost=${c_cost:.2f} != qty({c_qty:g})*avg_cost({c_avg:g})=${exp_cost:.2f}",
                    )
                )
        if c_avg is not None and (m_cpnl := _CYC_PNL_RE.search(c_tail)) is not None:
            px_c = px_by_vs.get((c_ven, c_sym))
            rep_cpnl = safe_float(m_cpnl.group(1), default=None)
            if px_c is not None and rep_cpnl is not None:
                exp_cpnl = ((px_c - c_avg) / c_avg) * 100.0
                if abs(exp_cpnl - rep_cpnl) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_POSITION_ANCHORS_AND_CYCLE_MATH",
                            f"{c_ven}:{c_sym}:cycle_pnl",
                            f"cycle pnl_vs_cost={fmt_signed(rep_cpnl, 2)}% != expected {fmt_signed(exp_cpnl, 2)}%",
                        )
                    )

    open_order_rows: list[tuple[str, str, str, float, Optional[float], str]] = []
    open_order_ids: dict[str, str] = {}
    buy_open_notional = 0.0
    sell_open_qty_by_vs: dict[tuple[str, str], float] = {}
    for m_oo in _OPEN_ORDER_ROW_RE.finditer(prompt_text):
        ven_o, sym_o, side_o = m_oo.group(1), m_oo.group(2), m_oo.group(3)
        px_o = safe_pos_float(m_oo.group(4), default=None)
        qty_o = safe_pos_float(m_oo.group(5), default=None)
        filled_cell = (m_oo.group(6) or "").strip()
        oid_cell = (m_oo.group(7) or "").strip()
        dist_cell = (m_oo.group(8) or "").strip()
        plan_cell = (m_oo.group(9) or "").strip()
        note_cell = (m_oo.group(10) or "").strip()
        if oid_cell and oid_cell != "-":
            open_order_ids[oid_cell] = f"{ven_o}:{sym_o}"
        rem_qty_o = qty_o
        if qty_o is not None and (m_rem := _FILLED_REM_RE.match(filled_cell)) is not None:
            f_v = safe_nonneg_float(m_rem.group(1), default=None)
            r_v = safe_nonneg_float(m_rem.group(2), default=None)
            if r_v is not None:
                rem_qty_o = r_v
            if f_v is not None and r_v is not None and abs((f_v + r_v) - qty_o) > max(1e-6, qty_o * 1e-4):
                violations.append(
                    InvariantViolation(
                        "PROMPT_OPEN_ORDER_REM_MATH",
                        f"{ven_o}:{sym_o}",
                        f"filled({f_v:g})+rem({r_v:g}) != qty({qty_o:g})",
                    )
                )
        if side_o == "BUY" and px_o is not None and rem_qty_o is not None:
            buy_open_notional += px_o * rem_qty_o
        elif side_o == "SELL" and rem_qty_o is not None:
            sk_vs = (ven_o.lower(), sym_o.upper())
            sell_open_qty_by_vs[sk_vs] = sell_open_qty_by_vs.get(sk_vs, 0.0) + rem_qty_o
        dist_v: Optional[float] = None
        if px_o is not None:
            if (m_dc := _DIST_PCT_CELL_RE.match(dist_cell)) is not None:
                dist_v = safe_float(m_dc.group(1), default=None)
            open_order_rows.append((ven_o, sym_o, side_o, px_o, dist_v, note_cell))
        if px_o is not None and (m_prr := _PLAN_RR_RE.search(plan_cell)) is not None:
            t_v = safe_pos_float(m_prr.group(1), default=None)
            t_pct = safe_float(m_prr.group(2), default=None) if m_prr.group(2) is not None else None
            i_v = safe_pos_float(m_prr.group(3), default=None)
            i_pct = safe_float(m_prr.group(4), default=None) if m_prr.group(4) is not None else None
            rep_rr = safe_nonneg_float(m_prr.group(5), default=None)
            for lbl_p, lvl_v, rep_p in (("target", t_v, t_pct), ("invalidation", i_v, i_pct)):
                if lvl_v is not None and rep_p is not None and px_o > 0.0:
                    exp_p = ((lvl_v - px_o) / px_o) * 100.0
                    if abs(rep_p - exp_p) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OPEN_ORDER_RR_MATH",
                                f"{ven_o}:{sym_o}:{lbl_p}",
                                f"plan {lbl_p} dist={fmt_signed(rep_p, 2)}% != expected {fmt_signed(exp_p, 2)}% (px={px_o}, {lbl_p}={lvl_v})",
                            )
                        )
            if t_v is not None and i_v is not None and rep_rr is not None:
                rew = (t_v - px_o) if side_o == "BUY" else (px_o - t_v)
                rsk = (px_o - i_v) if side_o == "BUY" else (i_v - px_o)
                if rsk > 0.0:
                    exp_rr = max(0.0, rew / rsk)
                    if abs(rep_rr - exp_rr) > 0.02:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OPEN_ORDER_RR_MATH",
                                f"{ven_o}:{sym_o}",
                                f"plan RR@limit={rep_rr:.2f}x != expected {exp_rr:.2f}x (px={px_o}, tgt={t_v}, inv={i_v})",
                            )
                        )
                if (m_fr := _FRONT_RUN_RR_RE.search(note_cell)) is not None:
                    fr_px = safe_pos_float(m_fr.group(1), default=None)
                    rep_fr_rr = safe_nonneg_float(m_fr.group(2), default=None)
                    if fr_px is not None and rep_fr_rr is not None:
                        rew_fr = (t_v - fr_px) if side_o == "BUY" else (fr_px - t_v)
                        rsk_fr = (fr_px - i_v) if side_o == "BUY" else (i_v - fr_px)
                        if rsk_fr > 0.0:
                            exp_fr_rr = max(0.0, rew_fr / rsk_fr)
                            if abs(rep_fr_rr - exp_fr_rr) > 0.02:
                                violations.append(
                                    InvariantViolation(
                                        "PROMPT_OPEN_ORDER_RR_MATH",
                                        f"{ven_o}:{sym_o}:front_run",
                                        f"front_run RR={rep_fr_rr:.2f}x != expected {exp_fr_rr:.2f}x (fr={fr_px}, tgt={t_v}, inv={i_v})",
                                    )
                                )

    if nav_locked_quote is not None and buy_open_notional > nav_locked_quote * 1.02 + 1.0:
        violations.append(
            InvariantViolation(
                "PROMPT_SECTION1_SUM_CONSERVATION",
                "Section1.OpenBuyVsLockedQuote",
                f"sum(open BUY notional)={buy_open_notional:.2f} exceeds locked_quote={nav_locked_quote:.2f}",
            )
        )
    for (s_ven, s_sym), sell_rem_q in sell_open_qty_by_vs.items():
        if (s_ven, s_sym) in locked_base_by_vs:
            lb_ref = locked_base_by_vs[(s_ven, s_sym)]
            if sell_rem_q > lb_ref * 1.02 + 1e-6:
                violations.append(
                    InvariantViolation(
                        "PROMPT_SECTION1_SUM_CONSERVATION",
                        f"Section1.OpenSellVsLockedBase:{s_ven}:{s_sym}",
                        f"sum(open SELL rem_qty)={sell_rem_q:g} exceeds locked_base_qty={lb_ref:g}",
                    )
                )

    for m_ap in _ACTIVE_PLAN_LINE_RE.finditer(prompt_text):
        ven_p, sym_p, side_p = m_ap.group(1), m_ap.group(2), m_ap.group(3)
        entry_p = safe_pos_float(m_ap.group(4), default=None)
        meta_p, body_p = m_ap.group(5) or "", m_ap.group(6) or ""
        spot_m = _ACTIVE_PLAN_SPOT_RE.search(meta_p)
        spot_p = safe_pos_float(spot_m.group(1), default=None) if spot_m else None
        is_resting_p = "RESTING" in meta_p.upper()
        anc_p = entry_p if is_resting_p else spot_p
        m_tgt = _ACTIVE_PLAN_TGT_RE.search(body_p)
        m_inv = _ACTIVE_PLAN_INV_RE.search(body_p)
        tgt_p = safe_pos_float(m_tgt.group(1), default=None) if m_tgt else None
        inv_p = safe_pos_float(m_inv.group(1), default=None) if m_inv else None
        if anc_p is not None and anc_p > 0.0:
            for lbl_k, m_lvl, lvl_px in (("target", m_tgt, tgt_p), ("invalidation", m_inv, inv_p)):
                if m_lvl is not None and m_lvl.group(2) is not None and lvl_px is not None:
                    rep_pct = safe_float(m_lvl.group(2), default=None)
                    exp_pct = ((lvl_px - anc_p) / anc_p) * 100.0
                    if rep_pct is not None and abs(rep_pct - exp_pct) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_ACTIVE_PLAN_RR_MATH",
                                f"{ven_p}:{sym_p}:{lbl_k}",
                                f"reported {lbl_k} dist={fmt_signed(rep_pct, 2)}% != expected {fmt_signed(exp_pct, 2)}% (px={lvl_px}, anc={anc_p})",
                            )
                        )
            if (m_rrp := _ACTIVE_PLAN_RR_RE.search(body_p)) is not None and any(
                tag in body_p for tag in ("[INVALIDATED]", "[TARGET_HIT]", "[CLOSED_EARLY]")
            ):
                violations.append(
                    InvariantViolation(
                        "PROMPT_ACTIVE_PLAN_RR_MATH",
                        f"{ven_p}:{sym_p}:terminal_rr",
                        f"terminal plan must not display forward {m_rrp.group(0)}",
                    )
                )
            elif tgt_p is not None and inv_p is not None and m_rrp is not None:
                rr_anc_tag = m_rrp.group(1)
                exp_anc_tag = "limit" if is_resting_p else "spot"
                if rr_anc_tag is not None and rr_anc_tag != exp_anc_tag:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_ACTIVE_PLAN_RR_MATH",
                            f"{ven_p}:{sym_p}:rr_anchor",
                            f"active plan RR@{rr_anc_tag} mismatches order_state (expected RR@{exp_anc_tag})",
                        )
                    )
                rep_rrp = safe_nonneg_float(m_rrp.group(2), default=None)
                rew_p = (tgt_p - anc_p) if side_p == "BUY" else (anc_p - tgt_p)
                rsk_p = (anc_p - inv_p) if side_p == "BUY" else (inv_p - anc_p)
                if rep_rrp is not None and rsk_p > 0.0:
                    exp_rrp = max(0.0, rew_p / rsk_p)
                    if abs(rep_rrp - exp_rrp) > 0.02:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_ACTIVE_PLAN_RR_MATH",
                                f"{ven_p}:{sym_p}",
                                f"active plan RR@{exp_anc_tag}={rep_rrp:.2f}x != expected {exp_rrp:.2f}x (anc={anc_p}, tgt={tgt_p}, inv={inv_p})",
                            )
                        )

    for ln_cash in prompt_text.splitlines():
        if "Available Free Quote Cash per Venue:" not in ln_cash:
            continue
        after_c = ln_cash.split("Available Free Quote Cash per Venue:", 1)[1]
        for m_cbd in _VENUE_CASH_BD_RE.finditer(after_c):
            ven_c, q_name = m_cbd.group(1), m_cbd.group(2)
            fq = _parse_money(m_cbd.group(3))
            bd_vals = [_parse_money(v_s) for _, v_s in _BD_ITEM_RE.findall(m_cbd.group(4))]
            bd_nums = [v for v in bd_vals if v is not None]
            if fq is not None and bd_nums:
                bd_sum = sum(bd_nums)
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FREE_QUOTE_BREAKDOWN",
                            f"{ven_c}:{q_name}",
                            f"{q_name}={fq:.2f} diverges from breakdown sum={bd_sum:.2f}",
                        )
                    )
        stripped_c = _PARENS_RE.sub("", after_c)
        ven_free_vals = [_parse_money(v_s) for _, v_s in _VENUE_FREE_CASH_ITEM_RE.findall(stripped_c)]
        ven_free_nums = [v for v in ven_free_vals if v is not None]
        if nav_free_cash is not None and ven_free_nums:
            ven_free_sum = sum(ven_free_nums)
            if abs(ven_free_sum - nav_free_cash) > max(0.10, nav_tol_usd):
                violations.append(
                    InvariantViolation(
                        "PROMPT_SECTION1_SUM_CONSERVATION",
                        "Section1.FreeCashSum",
                        f"sum(venue free cash)={ven_free_sum:.2f} != portfolio free_cash={nav_free_cash:.2f}",
                    )
                )
        ven_lock_vals = [_parse_money(v_s) for v_s in _VENUE_LOCKED_QUOTE_RE.findall(stripped_c)]
        ven_lock_nums = [v for v in ven_lock_vals if v is not None]
        if nav_locked_quote is not None and ven_lock_nums:
            ven_lock_sum = sum(ven_lock_nums)
            if abs(ven_lock_sum - nav_locked_quote) > max(0.10, nav_tol_usd):
                violations.append(
                    InvariantViolation(
                        "PROMPT_SECTION1_SUM_CONSERVATION",
                        "Section1.LockedQuoteSum",
                        f"sum(venue locked_quote_usd)={ven_lock_sum:.2f} != portfolio locked_quote={nav_locked_quote:.2f}",
                    )
                )

    # 4 & 5. Collect both single-line and unified multi-line symbol cards (Trend / Book / Flow)
    sec3_items: list[tuple[str, float, str]] = []
    sec3b_items: list[tuple[str, str]] = []
    prompt_lines = prompt_text.splitlines()
    cur_card_sym: Optional[str] = None
    cur_card_px: Optional[float] = None
    cur_trend_parts: list[str] = []
    cur_micro_parts: list[str] = []

    def _flush_card() -> None:
        if cur_card_sym is None:
            return
        if cur_card_px is not None and cur_card_px > 0.0 and cur_trend_parts:
            sec3_items.append((cur_card_sym, cur_card_px, " | ".join(cur_trend_parts)))
        if cur_micro_parts:
            sec3b_items.append((cur_card_sym, ", ".join(cur_micro_parts)))

    for raw_ln in prompt_lines:
        if (m_sub := _UNIFIED_SUBLINE_RE.match(raw_ln)) is not None and cur_card_sym is not None:
            sub_kind, sub_body = m_sub.group(1), m_sub.group(2)
            if sub_kind == "Trend":
                cur_trend_parts.append(sub_body)
            else:
                cur_micro_parts.append(sub_body)
            continue
        _flush_card()
        cur_card_sym = None
        cur_card_px = None
        cur_trend_parts = []
        cur_micro_parts = []
        if (m_s3 := _SEC3_SYM_RE.match(raw_ln)) is not None:
            sym = m_s3.group(1)
            px = _parse_money(m_s3.group(2))
            rest = m_s3.group(3)
            cur_card_sym = sym
            cur_card_px = px
            if rest.strip():
                cur_trend_parts.append(rest)
            continue
        if (m_3b := _SEC3B_SYM_RE.match(raw_ln)) is not None:
            sym, rest = m_3b.group(1), m_3b.group(2)
            cur_card_sym = sym
            if rest.strip():
                cur_micro_parts.append(rest)
    _flush_card()

    # 4. Section 3 range pos math, vol_profile order/deltas, and session_vwap delta
    poc_by_sym: dict[str, float] = {}
    for sym, px, rest in sec3_items:
        for m_rng in _RANGE_RE.finditer(rest):
            lbl = m_rng.group(1)
            lo = safe_pos_float(m_rng.group(2), default=None)
            hi = safe_pos_float(m_rng.group(3), default=None)
            pos = safe_float(m_rng.group(4), default=None)
            _check_range_pos(violations, "PROMPT_RANGE_POS_MATH", sym, lbl, px, lo, hi, pos, range_tol_pct)
        if (m_vp := _VOL_PROFILE_RE.search(rest)) is not None:
            poc = safe_pos_float(m_vp.group(1), default=None)
            poc_d = safe_float(m_vp.group(2), default=None)
            val = safe_pos_float(m_vp.group(3), default=None)
            val_d = safe_float(m_vp.group(4), default=None)
            vah = safe_pos_float(m_vp.group(5), default=None)
            vah_d = safe_float(m_vp.group(6), default=None)
            if poc is not None:
                poc_by_sym[sym.upper()] = poc
            if poc is not None and val is not None and vah is not None:
                if not (val <= poc <= vah):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VOL_PROFILE_ORDER",
                            sym,
                            f"vol_profile out of order: VAL={val:g}, POC={poc:g}, VAH={vah:g}",
                        )
                    )
                for vp_lbl, lv, rep_d in (("POC", poc, poc_d), ("VAL", val, val_d), ("VAH", vah, vah_d)):
                    if rep_d is not None:
                        exp_d = ((px - lv) / lv) * 100.0
                        if abs(exp_d - rep_d) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_VOL_PROFILE_DELTA",
                                    f"{sym}:{vp_lbl}",
                                    f"reported {fmt_signed(rep_d, 2)}% vs expected {fmt_signed(exp_d, 2)}% (px={px:g}, {vp_lbl}={lv:g})",
                                )
                            )
                rep_vp = safe_float(m_vp.group(7), default=None) if m_vp.group(7) is not None else None
                if rep_vp is not None and vah > val:
                    exp_vp = ((px - val) / (vah - val)) * 100.0
                    if abs(exp_vp - rep_vp) > 1.0:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VA_POS_AND_LIQ_MATH",
                                f"{sym}:va_pos",
                                f"reported va_pos={rep_vp:.1f}% != expected {exp_vp:.1f}% (px={px:g}, VAL={val:g}, VAH={vah:g})",
                            )
                        )
        if (m_vw := _SESSION_VWAP_RE.search(rest)) is not None:
            vwap_px = _parse_money(m_vw.group(1))
            rep_vd = safe_float(m_vw.group(2), default=None)
            rep_vz = safe_float(m_vw.group(3), default=None) if m_vw.group(3) is not None else None
            if vwap_px is not None and vwap_px > 0.0 and rep_vd is not None:
                exp_vd = ((px - vwap_px) / vwap_px) * 100.0
                if abs(exp_vd - rep_vd) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_SESSION_VWAP_DELTA",
                            sym,
                            f"reported {fmt_signed(rep_vd, 2)}% vs expected {fmt_signed(exp_vd, 2)}% (px={px:g}, vwap={vwap_px:g})",
                        )
                    )
            if rep_vd is not None and rep_vz is not None and rep_vd * rep_vz < -0.02:
                violations.append(
                    InvariantViolation(
                        "PROMPT_SESSION_VWAP_DELTA",
                        f"{sym}:z_sign",
                        f"session_vwap delta={fmt_signed(rep_vd, 2)}% has opposite sign to z={fmt_signed(rep_vz, 2)}",
                    )
                )

    # 5. Cross-venue arb symmetry, OB imbalance math, and wall ordering
    arb_pairs: dict[tuple[str, str], float] = {}
    book_walls_by_sym: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for sym, rest in sec3b_items:
        if (m_arb := _ARB_VS_RE.search(rest)) is not None:
            peer = m_arb.group(1)
            val = safe_float(m_arb.group(2), default=None)
            if val is not None:
                arb_pairs[(sym.upper(), peer.upper())] = val
        if (m_imb := _OB_IMB_RE.search(rest)) is not None:
            rep_imb = safe_float(m_imb.group(1), default=None)
            bd_v = _parse_money(m_imb.group(2))
            ad_v = _parse_money(m_imb.group(3))
            if rep_imb is not None and bd_v is not None and ad_v is not None and (bd_v + ad_v) > 0.0:
                exp_imb = bd_v / (bd_v + ad_v)
                if abs(exp_imb - rep_imb) > 0.015:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_IMBALANCE_MATH",
                            sym,
                            f"reported ob_imbalance={rep_imb:.3f} vs expected {exp_imb:.3f} (bid=${bd_v:.1f}, ask=${ad_v:.1f})",
                        )
                    )
        bw_px: Optional[float] = None
        aw_px: Optional[float] = None
        if (m_bw := _BID_WALL_RE.search(rest)) is not None:
            bw_px = safe_pos_float(m_bw.group(1), default=None)
        if (m_aw := _ASK_WALL_RE.search(rest)) is not None:
            aw_px = safe_pos_float(m_aw.group(1), default=None)
        if bw_px is not None or aw_px is not None:
            book_walls_by_sym[sym.upper()] = (bw_px, aw_px)
        if (m_mid := _OB_MID_RE.search(rest)) is not None:
            ob_mid = safe_pos_float(m_mid.group(1), default=None)
            if ob_mid is not None:
                if bw_px is not None and bw_px > ob_mid * 1.001:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_WALL_ORDERING",
                            sym,
                            f"bid_wall={bw_px:g} > ob_mid={ob_mid:g}",
                        )
                    )
                if aw_px is not None and aw_px < ob_mid * 0.999:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_WALL_ORDERING",
                            sym,
                            f"ask_wall={aw_px:g} < ob_mid={ob_mid:g}",
                        )
                    )
    for (s_a, s_b), v_a in arb_pairs.items():
        if s_a < s_b and (s_b, s_a) in arb_pairs:
            v_b = arb_pairs[(s_b, s_a)]
            if abs(v_a + v_b) > arb_tol_bps:
                violations.append(
                    InvariantViolation(
                        "PROMPT_CROSS_VENUE_ARB_SYMMETRY",
                        f"{s_a}<->{s_b}",
                        f"arb_vs_{s_b}={fmt_signed(v_a, 1)}bps vs arb_vs_{s_a}={fmt_signed(v_b, 1)}bps",
                    )
                )

    _audit_prompt_cross_section_math(
        violations,
        prompt_text,
        sec3_items,
        sec3b_items,
        px_by_vs,
        open_order_rows,
        book_walls_by_sym=book_walls_by_sym,
        poc_by_sym=poc_by_sym,
        arb_pairs=arb_pairs,
        maker_bps_by_sym=maker_bps_by_sym,
        arb_tol_bps=arb_tol_bps,
        open_order_ids=open_order_ids,
    )
    _audit_prompt_calibration_and_breadth(violations, prompt_text)
    _audit_prompt_section1_parity_and_structure(violations, prompt_text, sec3_items)
    _audit_prompt_numeric_hygiene(violations, prompt_text)
    return violations


def _audit_prompt_cross_section_math(
    violations: list[InvariantViolation],
    prompt_text: str,
    sec3_items: Sequence[tuple[str, float, str]],
    sec3b_items: Sequence[tuple[str, str]],
    px_by_vs: Mapping[tuple[str, str], float],
    open_order_rows: Sequence[tuple[str, str, str, float, Optional[float], str]],
    *,
    book_walls_by_sym: Mapping[str, tuple[Optional[float], Optional[float]]] = (),  # type: ignore[assignment]
    poc_by_sym: Mapping[str, float] = (),  # type: ignore[assignment]
    arb_pairs: Mapping[tuple[str, str], float] = (),  # type: ignore[assignment]
    maker_bps_by_sym: Mapping[str, float] = (),  # type: ignore[assignment]
    arb_tol_bps: float = 0.3,
    open_order_ids: Mapping[str, str] = (),  # type: ignore[assignment]
) -> None:
    """Validate rendered range extremes, bottom_6h, vs_btc, OB spread/wall/depth/flow/edge, open orders, advisor & xStock."""
    sec3_px_by_sym: dict[str, float] = {}
    hor_chg_by_sym: dict[str, dict[str, float]] = {}
    for sym, px, rest in sec3_items:
        sym_u = sym.upper()
        sec3_px_by_sym[sym_u] = px
        h_map: dict[str, float] = {}
        for m_hc in _HORIZON_CHG_RE.finditer(rest.split("vs_btc(", 1)[0]):
            if (hv := safe_float(m_hc.group(2), default=None)) is not None:
                h_map.setdefault(m_hc.group(1), hv)
        hor_chg_by_sym[sym_u] = h_map

        l24_found: Optional[float] = None
        for m_re in _RANGE_EXT_RE.finditer(rest):
            lbl = m_re.group(1)
            lo = safe_pos_float(m_re.group(2), default=None)
            hi = safe_pos_float(m_re.group(3), default=None)
            dd_h = safe_float(m_re.group(5), default=None)
            bn_l = safe_float(m_re.group(6), default=None)
            sp = safe_float(m_re.group(7), default=None)
            if lo is None or hi is None:
                continue
            if lbl == "24h":
                l24_found = lo
            if lo > hi * 1.0001 or px < lo * 0.999 or px > hi * 1.001:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}",
                        f"px={px:g} inconsistent with {lbl}_range=[{lo:g}, {hi:g}]",
                    )
                )
            if dd_h is not None and abs(dd_h - ((px - hi) / hi) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:dd_high",
                        f"reported dd_high={fmt_signed(dd_h, 2)}% != expected {fmt_signed(((px - hi) / hi) * 100.0, 2)}%",
                    )
                )
            if bn_l is not None and abs(bn_l - ((px - lo) / lo) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:bounce_low",
                        f"reported bounce_low={fmt_signed(bn_l, 2)}% != expected {fmt_signed(((px - lo) / lo) * 100.0, 2)}%",
                    )
                )
            if sp is not None and abs(sp - ((hi - lo) / lo) * 100.0) > 0.25:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:span",
                        f"reported span={sp:.2f}% != expected {((hi - lo) / lo) * 100.0:.2f}%",
                    )
                )

        if (m_b6 := _BOTTOM_6H_RE.search(rest)) is not None:
            p6h = safe_pos_float(m_b6.group(1), default=None)
            r1h = safe_pos_float(m_b6.group(2), default=None)
            b_lbl, b_pct = m_b6.group(3), safe_float(m_b6.group(4), default=None) if m_b6.group(4) else None
            m_r15 = _RANGE_15M_RE.search(rest)
            l15 = safe_pos_float(m_r15.group(1), default=None) if m_r15 else None
            if r1h is not None and l15 is not None and r1h > l15 * 1.0005:
                violations.append(
                    InvariantViolation(
                        "PROMPT_BOTTOM_STRUCTURE_MATH",
                        sym,
                        f"bottom_6h 1h_low={r1h:g} > 15m_range low={l15:g}",
                    )
                )
            if p6h is not None and r1h is not None and l24_found is not None and min(p6h, r1h) < l24_found * 0.999:
                violations.append(
                    InvariantViolation(
                        "PROMPT_BOTTOM_STRUCTURE_MATH",
                        sym,
                        f"bottom_6h min({p6h:g}, {r1h:g}) < low_24h={l24_found:g}",
                    )
                )
            if p6h is not None and r1h is not None and b_lbl and b_pct is not None:
                num_px = r1h if b_lbl == "higher_low_pct" else px
                exp_bp = ((num_px - p6h) / p6h) * 100.0
                if abs(exp_bp - b_pct) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_BOTTOM_STRUCTURE_MATH",
                            f"{sym}:{b_lbl}",
                            f"reported {b_lbl}={fmt_signed(b_pct, 2)}% != expected {fmt_signed(exp_bp, 2)}%",
                        )
                    )
        if (m_tp := _TAPE_15M_RE.search(rest)) is not None:
            lo15_t = safe_pos_float(m_tp.group(2), default=None)
            hi15_t = safe_pos_float(m_tp.group(3), default=None)
            t_pts = {k: safe_pos_float(v, default=None) for k, v in _TAPE_PT_RE.findall(m_tp.group(1))}
            if lo15_t is not None and hi15_t is not None:
                for tk, tv in t_pts.items():
                    if tv is not None and (tv < lo15_t * 0.999 or tv > hi15_t * 1.001):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                                f"{sym}:tape_{tk}",
                                f"tape_15m {tk}={tv:g} outside 15m_range=[{lo15_t:g}, {hi15_t:g}]",
                            )
                        )
            short_h_map: dict[str, float] = {}
            for m_sh in _SHORT_H_RE.finditer(rest.split("vs_btc(", 1)[0]):
                if (sv := safe_float(m_sh.group(2), default=None)) is not None:
                    short_h_map.setdefault(m_sh.group(1), sv)
            for h_lbl in ("1m", "3m", "5m", "15m"):
                t0_px = t_pts.get(f"-{h_lbl}")
                rep_sh = short_h_map.get(h_lbl)
                if t0_px is not None and rep_sh is not None:
                    exp_sh = ((px - t0_px) / t0_px) * 100.0
                    if abs(exp_sh - rep_sh) > 0.20:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                                f"{sym}:tape_{h_lbl}",
                                f"reported {h_lbl}={fmt_signed(rep_sh, 2)}% != tape_15m delta {fmt_signed(exp_sh, 2)}% (t0={t0_px:g}, now={px:g})",
                            )
                        )
        if (m_sw := _SWING_WAVE_RE.search(rest)) is not None:
            sw_checks = [
                ("rally_low", safe_pos_float(m_sw.group(1), default=None), safe_float(m_sw.group(3), default=None)),
                ("pullback_high", safe_pos_float(m_sw.group(2), default=None), safe_float(m_sw.group(4), default=None)),
            ]
            if m_sw.group(5) and m_sw.group(6):
                sw_checks.append(("major_low", safe_pos_float(m_sw.group(5), default=None), safe_float(m_sw.group(6), default=None)))
            if m_sw.group(7) and m_sw.group(8):
                sw_checks.append(("major_high", safe_pos_float(m_sw.group(7), default=None), safe_float(m_sw.group(8), default=None)))
            for sw_lbl, sw_ref, sw_rep in sw_checks:
                if sw_ref is not None and sw_rep is not None:
                    exp_sw = ((px - sw_ref) / sw_ref) * 100.0
                    if abs(exp_sw - sw_rep) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                                f"{sym}:swing_{sw_lbl}",
                                f"swing_wave {sw_lbl}={fmt_signed(sw_rep, 2)}% != expected {fmt_signed(exp_sw, 2)}% (px={px:g}, ref={sw_ref:g})",
                            )
                        )
        if (m_sb := _SESSIONS_BLOCK_RE.search(rest)) is not None and (m_r24 := _RANGE_EXT_RE.search(rest)) is not None and m_r24.group(1) == "24h":
            lo24_s = safe_pos_float(m_r24.group(2), default=None)
            hi24_s = safe_pos_float(m_r24.group(3), default=None)
            if lo24_s is not None and hi24_s is not None:
                for sn, slo_s, shi_s in _SESSION_ITEM_RE.findall(m_sb.group(1)):
                    slo_f, shi_f = safe_pos_float(slo_s, default=None), safe_pos_float(shi_s, default=None)
                    if slo_f is not None and shi_f is not None and (slo_f > shi_f or slo_f < lo24_s * 0.999 or shi_f > hi24_s * 1.001):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                                f"{sym}:session_{sn}",
                                f"session {sn}=[{slo_f:g}, {shi_f:g}] inconsistent with 24h_range=[{lo24_s:g}, {hi24_s:g}]",
                            )
                        )

    for (p_ven, p_sym_u), p_px in px_by_vs.items():
        if (s3_px := sec3_px_by_sym.get(p_sym_u)) is not None and s3_px > 0.0:
            if abs(p_px - s3_px) / s3_px > 0.005:
                violations.append(
                    InvariantViolation(
                        "PROMPT_SEC1_SEC3_PRICE_PARITY",
                        f"{p_ven}:{p_sym_u}",
                        f"Section 1 px={p_px:g} diverges >0.5% from Section 3 Price={s3_px:g}",
                    )
                )

    btc_h = hor_chg_by_sym.get("BTCUSDC", {})
    if btc_h:
        for sym, _, rest in sec3_items:
            sym_u = sym.upper()
            if sym_u == "BTCUSDC" or (m_vb := _VS_BTC_BLOCK_RE.search(rest)) is None:
                continue
            sym_h = hor_chg_by_sym.get(sym_u, {})
            for m_vh in _HORIZON_CHG_RE.finditer(m_vb.group(1)):
                hor = m_vh.group(1)
                rep_vb = safe_float(m_vh.group(2), default=None)
                if rep_vb is not None and hor in sym_h and hor in btc_h:
                    exp_vb = sym_h[hor] - btc_h[hor]
                    if abs(rep_vb - exp_vb) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VS_BTC_MATH",
                                f"{sym}:{hor}",
                                f"reported vs_btc {hor}={fmt_signed(rep_vb, 2)}% != sym({fmt_signed(sym_h[hor], 2)}%)-btc({fmt_signed(btc_h[hor], 2)}%)={fmt_signed(exp_vb, 2)}%",
                            )
                        )

    tr_by_sym: dict[str, float] = {}
    edge_by_sym: dict[str, float] = {}
    for sym, rest in sec3b_items:
        sym_u = sym.upper()
        if (m_sp := _OB_SPREAD_BBO_RE.search(rest)) is not None:
            rep_sp = safe_float(m_sp.group(1), default=None)
            bb = safe_pos_float(m_sp.group(2), default=None)
            ba = safe_pos_float(m_sp.group(3), default=None)
            if bb is not None and ba is not None:
                if bb > ba * 1.00001:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_SPREAD_AND_WALL_DIST",
                            f"{sym}:bbo",
                            f"crossed book bid={bb:g} > ask={ba:g}",
                        )
                    )
                if rep_sp is not None:
                    exp_sp = ((ba - bb) / ((ba + bb) / 2.0)) * 10000.0
                    if abs(exp_sp - rep_sp) > max(1.0, abs(exp_sp) * 0.05):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_SPREAD_AND_WALL_DIST",
                                f"{sym}:spread",
                                f"reported spread={rep_sp:.2f}bps != expected {exp_sp:.2f}bps (bid={bb:g}, ask={ba:g})",
                            )
                        )
        if (m_mid := _OB_MID_RE.search(rest)) is not None and (ob_mid := safe_pos_float(m_mid.group(1), default=None)) is not None:
            for m_wd in _WALL_DIST_RE.finditer(rest):
                w_lbl = m_wd.group(1)
                w_px = safe_pos_float(m_wd.group(2), default=None)
                rep_wd = safe_float(m_wd.group(3), default=None)
                if w_px is not None and rep_wd is not None:
                    exp_wd = ((w_px - ob_mid) / ob_mid) * 100.0
                    if abs(exp_wd - rep_wd) > 0.25:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_SPREAD_AND_WALL_DIST",
                                f"{sym}:{w_lbl}_dist",
                                f"reported {w_lbl} dist={fmt_signed(rep_wd, 2)}% != expected {fmt_signed(exp_wd, 2)}% (wall={w_px:g}, mid={ob_mid:g})",
                            )
                        )
            for m_lq in _LIQ_CLUSTER_ITEM_RE.finditer(rest):
                lq_lbl = m_lq.group(1)
                lq_px = _parse_money(m_lq.group(2))
                rep_ld = safe_float(m_lq.group(3), default=None)
                if lq_px is not None and lq_px > 0.0 and rep_ld is not None:
                    exp_ld = ((lq_px - ob_mid) / ob_mid) * 100.0
                    if abs(exp_ld - rep_ld) > 0.25:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VA_POS_AND_LIQ_MATH",
                                f"{sym}:{lq_lbl}",
                                f"reported {lq_lbl} dist={fmt_signed(rep_ld, 2)}% != expected {fmt_signed(exp_ld, 2)}% (liq={lq_px:g}, mid={ob_mid:g})",
                            )
                        )
        if (m_tr := _TAKER_RATIO_VOL_RE.search(rest)) is not None:
            rep_tr = safe_float(m_tr.group(1), default=None)
            bv = _parse_money(m_tr.group(2))
            sv = _parse_money(m_tr.group(3))
            if rep_tr is not None:
                tr_by_sym[sym_u] = rep_tr
            if rep_tr is not None and bv is not None and sv is not None and sv > 0.0:
                exp_tr = bv / sv
                if abs(rep_tr - exp_tr) > max(0.05, abs(exp_tr) * 0.03):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FLOW_DEPTH_EDGE_MATH",
                            f"{sym}:taker_ratio",
                            f"reported taker_ratio={rep_tr:.4f} != buy_v/sell_v={exp_tr:.4f}",
                        )
                    )
        if (m_imb := _OB_IMB_RE.search(rest)) is not None and (m_d1 := _DEPTH_1PCT_RE.search(rest)) is not None:
            bd_tot, ad_tot = _parse_money(m_imb.group(2)), _parse_money(m_imb.group(3))
            d1_b, d1_a = _parse_money(m_d1.group(1)), _parse_money(m_d1.group(2))
            if bd_tot is not None and d1_b is not None and d1_b > bd_tot * 1.05 + 1.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_FLOW_DEPTH_EDGE_MATH",
                        f"{sym}:depth_1pct_bid",
                        f"depth_1pct bid=${d1_b:.1f} exceeds total bid_depth=${bd_tot:.1f}",
                    )
                )
            if ad_tot is not None and d1_a is not None and d1_a > ad_tot * 1.05 + 1.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_FLOW_DEPTH_EDGE_MATH",
                        f"{sym}:depth_1pct_ask",
                        f"depth_1pct ask=${d1_a:.1f} exceeds total ask_depth=${ad_tot:.1f}",
                    )
                )
        if (m_cvd := _CVD_15M_RE.search(rest)) is not None and (px_s := sec3_px_by_sym.get(sym_u)) is not None:
            base_q = safe_float(m_cvd.group(1).replace(",", ""), default=None)
            usd_sgn = -1.0 if m_cvd.group(2) == "-" else 1.0
            usd_mag = _parse_money(m_cvd.group(3))
            if base_q is not None and usd_mag is not None and usd_mag > 0.0:
                usd_signed = usd_sgn * usd_mag
                if (base_q * usd_signed < -1.0) or abs(abs(base_q) * px_s - usd_mag) > max(50.0, usd_mag * 0.03):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FLOW_DEPTH_EDGE_MATH",
                            f"{sym}:cvd_15m",
                            f"cvd_15m base({base_q:g})*px({px_s:g})={base_q * px_s:.1f} != reported {fmt_signed(usd_signed, 1)} USD",
                        )
                    )
        if (m_nbe := _NET_BUY_EDGE_RE.search(rest)) is not None:
            if (nbe_v := safe_float(m_nbe.group(1), default=None)) is not None:
                edge_by_sym[sym_u] = nbe_v
        if (m_v15 := _SPOT_VOL_15M_RE.search(rest)) is not None and (m_v1h := _SPOT_VOL_1H_RE.search(rest)) is not None:
            v15_f, v1h_f = _parse_money(m_v15.group(1)), _parse_money(m_v1h.group(1))
            if v15_f is not None and v1h_f is not None and v15_f > v1h_f * 1.05 + 1.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                        f"{sym}:spot_vol",
                        f"spot_vol_15m=${v15_f:.1f} exceeds spot_vol_1h=${v1h_f:.1f}",
                    )
                )
        if (m_acv := _AGG_SPOT_CVD_VOL_RE.search(rest)) is not None:
            cvd_mag = _parse_money(m_acv.group(2))
            agg_vol = _parse_money(m_acv.group(3))
            if cvd_mag is not None and agg_vol is not None and cvd_mag > agg_vol * 1.02 + 1.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_TAPE_SWING_AND_FLOW_BOUNDS",
                        f"{sym}:agg_spot_cvd_vs_vol",
                        f"|agg_spot_cvd_15m|=${cvd_mag:.1f} exceeds agg_spot_vol=${agg_vol:.1f}",
                    )
                )

    for (s_a, s_b), arb_ab in arb_pairs.items():
        if s_a in edge_by_sym:
            if s_a < s_b and s_b in edge_by_sym and abs(edge_by_sym[s_a] + edge_by_sym[s_b]) > arb_tol_bps:
                violations.append(
                    InvariantViolation(
                        "PROMPT_FLOW_DEPTH_EDGE_MATH",
                        f"{s_a}<->{s_b}:net_buy_edge",
                        f"net_buy_edge({s_a})={fmt_signed(edge_by_sym[s_a], 1)}bps vs net_buy_edge({s_b})={fmt_signed(edge_by_sym[s_b], 1)}bps",
                    )
                )
            if s_a in maker_bps_by_sym and s_b in maker_bps_by_sym:
                exp_edge = (maker_bps_by_sym[s_b] - maker_bps_by_sym[s_a]) - arb_ab
                if abs(edge_by_sym[s_a] - exp_edge) > arb_tol_bps:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FLOW_DEPTH_EDGE_MATH",
                            f"{s_a}:net_buy_edge",
                            f"reported net_buy_edge={fmt_signed(edge_by_sym[s_a], 1)}bps != fee_diff({fmt_signed(maker_bps_by_sym[s_b]-maker_bps_by_sym[s_a], 1)})-arb({fmt_signed(arb_ab, 1)})={fmt_signed(exp_edge, 1)}bps",
                        )
                    )

    for ven_o, sym_o, side_o, px_o, rep_dist, note_cell in open_order_rows:
        sym_ou = sym_o.upper()
        spot_p = sec3_px_by_sym.get(sym_ou) or px_by_vs.get((ven_o.lower(), sym_ou))
        if spot_p is not None and spot_p > 0.0 and rep_dist is not None:
            exp_d = ((px_o - spot_p) / spot_p) * 100.0
            if abs(exp_d - rep_dist) > 0.20:
                violations.append(
                    InvariantViolation(
                        "PROMPT_OPEN_ORDER_DIST_MATH",
                        f"{ven_o}:{sym_o}",
                        f"reported Dist={fmt_signed(rep_dist, 2)}% != expected {fmt_signed(exp_d, 2)}% (limit={px_o:g}, spot={spot_p:g})",
                    )
                )
        if (m_nw := _NOTE_WALL_PX_RE.search(note_cell)) is not None and sym_ou in book_walls_by_sym:
            nw_px = safe_pos_float(m_nw.group(1), default=None)
            bw_ref, aw_ref = book_walls_by_sym[sym_ou]
            book_w = bw_ref if side_o == "BUY" else aw_ref
            if nw_px is not None and book_w is not None and abs(nw_px - book_w) / book_w > 0.005:
                violations.append(
                    InvariantViolation(
                        "PROMPT_OPEN_ORDER_WALL_BOOK_MATH",
                        f"{ven_o}:{sym_o}:wall",
                        f"open-order Note wall={nw_px:g} diverges from Section 3 Book wall={book_w:g}",
                    )
                )
        if (m_frp := _NOTE_FRONT_RUN_PX_RE.search(note_cell)) is not None and spot_p is not None and spot_p > 0.0:
            fr_px = safe_pos_float(m_frp.group(1), default=None)
            if fr_px is not None:
                bad_buy = side_o == "BUY" and (fr_px <= px_o or fr_px > spot_p * 1.0005)
                bad_sell = side_o == "SELL" and (fr_px >= px_o or fr_px < spot_p * 0.9995)
                if bad_buy or bad_sell:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OPEN_ORDER_WALL_BOOK_MATH",
                            f"{ven_o}:{sym_o}:front_run",
                            f"{side_o} front_run={fr_px:g} out of passive range (limit={px_o:g}, spot={spot_p:g})",
                        )
                    )

    sec0_head = prompt_text.split("### 1.", 1)[0]
    if "Trigger:" in sec0_head:
        oo_fr_by_sym_px: dict[tuple[str, float], float] = {}
        for _, sym_o, _, _, _, note_cell in open_order_rows:
            if (m_fr_o := _FRONT_RUN_RR_RE.search(note_cell)) is not None:
                fpx_o = safe_pos_float(m_fr_o.group(1), default=None)
                frr_o = safe_nonneg_float(m_fr_o.group(2), default=None)
                if fpx_o is not None and frr_o is not None:
                    oo_fr_by_sym_px[(sym_o.upper(), round(fpx_o, 6))] = frr_o
        for m_tw in _TRIGGER_WALL_TRAPPED_RE.finditer(sec0_head):
            t_sym, t_px = m_tw.group(1).upper(), safe_pos_float(m_tw.group(2), default=None)
            t_rr = safe_nonneg_float(m_tw.group(3), default=None)
            if t_px is not None and t_rr is not None:
                oo_rr = oo_fr_by_sym_px.get((t_sym, round(t_px, 6)))
                if oo_rr is not None and abs(t_rr - oo_rr) > 0.02:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OPEN_ORDER_RR_MATH",
                            f"Section0.{t_sym}:wall_trapped",
                            f"Section 0 trigger front_run RR={t_rr:.2f}x != Section 1 table RR={oo_rr:.2f}x (fr={t_px:g})",
                        )
                    )

    if (m_adv := _ADVISOR_BTC_DELTA_RE.search(prompt_text)) is not None:
        adv_body = m_adv.group(1)
        if (m_bp := _ADV_BTC_POC_RE.search(adv_body)) is not None:
            adv_btc = _parse_money(m_bp.group(1))
            adv_poc = _parse_money(m_bp.group(2)) if m_bp.group(2) else None
            sec3_btc = sec3_px_by_sym.get("BTCUSDC")
            sec3_poc = poc_by_sym.get("BTCUSDC")
            if adv_btc is not None and sec3_btc is not None and abs(adv_btc - sec3_btc) / sec3_btc > 0.0015:
                violations.append(
                    InvariantViolation(
                        "PROMPT_ADVISOR_BTC_PARITY",
                        "Section2.Advisor.BTC",
                        f"advisor live_btc_flow_delta BTC=${adv_btc:.2f} != Section 3 BTCUSDC=${sec3_btc:.2f}",
                    )
                )
            if adv_poc is not None and sec3_poc is not None and abs(adv_poc - sec3_poc) / sec3_poc > 0.0015:
                violations.append(
                    InvariantViolation(
                        "PROMPT_ADVISOR_BTC_PARITY",
                        "Section2.Advisor.POC",
                        f"advisor live_btc_flow_delta POC=${adv_poc:.2f} != Section 3 BTCUSDC POC=${sec3_poc:.2f}",
                    )
                )
        if (m_atr := _ADV_TR_RE.search(adv_body)) is not None:
            adv_tr = safe_float(m_atr.group(1), default=None)
            sec3_tr = tr_by_sym.get("BTCUSDC")
            if adv_tr is not None and sec3_tr is not None and abs(adv_tr - sec3_tr) > 0.03:
                violations.append(
                    InvariantViolation(
                        "PROMPT_ADVISOR_BTC_PARITY",
                        "Section2.Advisor.taker_ratio",
                        f"advisor taker_ratio={adv_tr:.2f} != Section 3 BTCUSDC taker_ratio={sec3_tr:.2f}",
                    )
                )

    for m_xs in _XSTOCK_REF_RE.finditer(prompt_text):
        x_px = (
            safe_pos_float(m_xs.group(1), default=None)
            if m_xs.group(1) is not None
            else (
                casefold_get(sec3_px_by_sym, "SPCXxUSD")
                or next((p for (_, s_u), p in px_by_vs.items() if s_u.casefold() == "SPCXxUSD".casefold()), None)
            )
        )
        u_px = safe_pos_float(m_xs.group(2), default=None)
        rep_prem = safe_float(m_xs.group(3), default=None)
        if x_px is not None and u_px is not None and rep_prem is not None:
            exp_prem = ((x_px - u_px) / u_px) * 100.0
            if abs(exp_prem - rep_prem) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_XSTOCK_PREMIUM_MATH",
                        "SPCXxUSD",
                        f"reported xstock_vs_underlying={fmt_signed(rep_prem, 2)}% != expected {fmt_signed(exp_prem, 2)}% (x={x_px:g}, u={u_px:g})",
                    )
                )

    # Order lifecycle consistency across Section 1 (no ghost open order also marked FILLED/CANCELED; no status contradiction per order ID)
    sec1_body = prompt_text.split("### 2.", 1)[0]
    seen_oid_statuses: dict[str, set[str]] = {}
    for m_is in _INTENT_STATUS_ID_RE.finditer(sec1_body):
        st_tok, oid_tok = m_is.group(1), m_is.group(2)
        seen_oid_statuses.setdefault(oid_tok, set()).add(st_tok)
        if oid_tok in open_order_ids and st_tok in _TERMINAL_ORDER_STATUS_TOKS:
            violations.append(
                InvariantViolation(
                    "PROMPT_ORDER_LIFECYCLE_CONSISTENCY",
                    f"{open_order_ids[oid_tok]}:id={oid_tok}",
                    f"order id={oid_tok} appears BOTH in open_limit_orders AND marked {st_tok}",
                )
            )
    for oid_tok, st_set in seen_oid_statuses.items():
        if len(st_set) > 1:
            violations.append(
                InvariantViolation(
                    "PROMPT_ORDER_LIFECYCLE_CONSISTENCY",
                    f"id={oid_tok}",
                    f"order id={oid_tok} has conflicting statuses across Section 1: {sorted(st_set)}",
                )
            )

    # Warm-turn delta elision truthfulness: no block declared omitted in Section 0 may still be rendered in full
    if (m_om := _WARM_OMITTED_LINE_RE.search(prompt_text)) is not None:
        om_s = m_om.group(1)
        body_after_om = prompt_text[m_om.end():]
        for om_tok, hdr_str in _WARM_OMITTED_HEADER_SPECS:
            if om_tok in om_s and hdr_str in body_after_om:
                violations.append(
                    InvariantViolation(
                        "PROMPT_WARM_ELISION_TRUTHFULNESS",
                        om_tok.rstrip("("),
                        f"Section 0 declares {om_tok}...) omitted, yet prompt body still renders {hdr_str!r}",
                    )
                )

    # Multi-Month & Annual Structure (Long Horizon) range position math and live-price parity
    sec2_crypto_bw_px: dict[str, float] = {}
    if "Cross-Market Live Crypto Telemetry" in prompt_text:
        cm_seg = prompt_text.split("Cross-Market Live Crypto Telemetry", 1)[1].split("### 3.", 1)[0]
        for m_cbw in _SEC2_CRYPTO_BW_ROW_RE.finditer(cm_seg):
            if (cbw_px := _parse_money(m_cbw.group(2))) is not None and cbw_px > 0.0:
                sec2_crypto_bw_px[m_cbw.group(1).upper()] = cbw_px
    for m_lh in _LH_ROW_RE.finditer(prompt_text):
        lh_sym = m_lh.group(1)
        lh_sym_u = lh_sym.upper()
        lo52, hi52, p52 = _parse_money(m_lh.group(2)), _parse_money(m_lh.group(3)), safe_float(m_lh.group(4), default=None)
        lo_h = _parse_money(m_lh.group(5)) if m_lh.group(5) else None
        hi_h = _parse_money(m_lh.group(6)) if m_lh.group(6) else None
        p_h = safe_float(m_lh.group(7), default=None) if m_lh.group(7) else None
        swings_pre = m_lh.group(8) or ""
        now_lh = _parse_money(m_lh.group(9))
        if lo52 is not None and hi52 is not None and lo_h is not None and hi_h is not None:
            if lo_h > lo52 * 1.005 or hi_h < hi52 * 0.995:
                violations.append(
                    InvariantViolation(
                        "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                        f"{lh_sym}:lh_nesting",
                        f"52w range [{lo52:g}..{hi52:g}] extends outside multi-year range [{lo_h:g}..{hi_h:g}]",
                    )
                )
        sw_yms = _LH_SWING_YM_RE.findall(swings_pre)
        if sw_yms != sorted(sw_yms):
            violations.append(
                InvariantViolation(
                    "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                    f"{lh_sym}:lh_swings_order",
                    f"non-chronological long-horizon swings: {sw_yms}",
                )
            )
        if now_lh is not None and now_lh > 0.0:
            if lo52 is not None and hi52 is not None and p52 is not None and hi52 > lo52:
                exp_52 = clamp_float(((now_lh - lo52) / (hi52 - lo52)) * 100.0, 0.0, 100.0)
                if abs(exp_52 - p52) > 1.5:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_LONG_HORIZON_MATH",
                            f"{lh_sym}:Pos52w",
                            f"reported Pos52w={p52:.1f}% != expected {exp_52:.1f}% (now={now_lh:g}, [{lo52:g}..{hi52:g}])",
                        )
                    )
            if lo_h is not None and hi_h is not None and p_h is not None and hi_h > lo_h:
                exp_h = clamp_float(((now_lh - lo_h) / (hi_h - lo_h)) * 100.0, 0.0, 100.0)
                if abs(exp_h - p_h) > 1.5:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_LONG_HORIZON_MATH",
                            f"{lh_sym}:PosHorizon",
                            f"reported Pos={p_h:.1f}% != expected {exp_h:.1f}% (now={now_lh:g}, [{lo_h:g}..{hi_h:g}])",
                        )
                    )
            live_ref = sec3_px_by_sym.get(lh_sym_u) or sec2_crypto_bw_px.get(canonical_base_asset(lh_sym))
            if live_ref is not None and live_ref > 0.0 and abs(now_lh - live_ref) / live_ref > 0.01:
                violations.append(
                    InvariantViolation(
                        "PROMPT_LONG_HORIZON_MATH",
                        f"{lh_sym}:now_px",
                        f"Long_Horizon now={now_lh:g} diverges from live price {live_ref:g}",
                    )
                )

    # Execution fill notional/arr_bps, Cross-Market US Equities ExtVsRTH, Breadth ETH/BTC, and Section 4 vs_btc parity
    for m_ex in _EXEC_FILL_RE.finditer(sec1_body):
        ex_side, ex_st, ex_body = m_ex.group(1), m_ex.group(2), m_ex.group(3)
        kv = dict(_EXEC_KV_RE.findall(ex_body))
        if ex_st == "FILLED" and "fill_px" in kv and "qty" in kv and "notional" in kv:
            fp_v = _parse_money(kv["fill_px"].lstrip("$"))
            q_v = safe_pos_float(kv["qty"].replace(",", ""), default=None)
            n_v = _parse_money(kv["notional"].lstrip("$"))
            if fp_v is not None and q_v is not None and n_v is not None and abs(fp_v * q_v - n_v) > max(0.50, n_v * 0.005):
                violations.append(
                    InvariantViolation(
                        "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                        f"Exec.{ex_side}.notional",
                        f"fill_px({fp_v:g})*qty({q_v:g})={fp_v * q_v:.2f} != notional=${n_v:.2f}",
                    )
                )
        if "fill_px" in kv and "ref_px" in kv and "arr_bps" in kv:
            fp_v = _parse_money(kv["fill_px"].lstrip("$"))
            rp_v = _parse_money(kv["ref_px"].lstrip("$"))
            ab_v = safe_float(kv["arr_bps"].replace(",", ""), default=None)
            if fp_v is not None and rp_v is not None and rp_v > 0.0 and ab_v is not None:
                sgn_ex = 1.0 if ex_side == "BUY" else -1.0
                exp_ab = sgn_ex * ((fp_v - rp_v) / rp_v) * 10000.0
                if abs(exp_ab - ab_v) > 1.0:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                            f"Exec.{ex_side}.arr_bps",
                            f"reported arr_bps={fmt_signed(ab_v, 2)} != expected {fmt_signed(exp_ab, 2)} (fill={fp_v:g}, ref={rp_v:g})",
                        )
                    )
    for m_eqx in _EQ_CROSS_ROW_RE.finditer(prompt_text):
        eq_s = m_eqx.group(1)
        px_e = safe_pos_float(m_eqx.group(2), default=None)
        rep_ext = safe_float(m_eqx.group(3), default=None)
        cl_e = safe_pos_float(m_eqx.group(4), default=None)
        if px_e is not None and rep_ext is not None and cl_e is not None:
            exp_ext = ((px_e - cl_e) / cl_e) * 100.0
            if abs(exp_ext - rep_ext) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                        f"{eq_s}:ExtVsRTH",
                        f"reported ExtVsRTH={fmt_signed(rep_ext, 2)}% != expected {fmt_signed(exp_ext, 2)}% (px={px_e:g}, close={cl_e:g})",
                    )
                )
    if (m_eb := _BREADTH_ETH_BTC_RE.search(prompt_text)) is not None and (btc_live := sec3_px_by_sym.get("BTCUSDC")) is not None:
        rep_eb = safe_pos_float(m_eb.group(1), default=None)
        eth_br = _parse_money(m_eb.group(2))
        if rep_eb is not None and eth_br is not None and btc_live > 0.0:
            exp_eb = eth_br / btc_live
            if abs(rep_eb - exp_eb) / exp_eb > 0.03:
                violations.append(
                    InvariantViolation(
                        "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                        "MarketBreadth.ETH_BTC",
                        f"reported ETH/BTC={rep_eb:.5f} diverges >3% from ETH({eth_br:.2f})/BTC({btc_live:.2f})={exp_eb:.5f}",
                    )
                )
    if btc_h and "### 4. MARKET BELLWETHERS" in prompt_text:
        sec4_body = prompt_text.split("### 4. MARKET BELLWETHERS", 1)[1]
        for m_bw4 in _SEC4_BW_ROW_RE.finditer(sec4_body):
            bw_s, c1h_s, c24_s, ctx_s = m_bw4.group(1), m_bw4.group(3), m_bw4.group(5), m_bw4.group(6)
            if (m_vb4 := _VS_BTC_BLOCK_RE.search(ctx_s)) is not None:
                for h_k, c_s in (("1h", c1h_s), ("24h", c24_s)):
                    if c_s == "-" or h_k not in btc_h:
                        continue
                    c_v = safe_float(c_s.rstrip("%"), default=None)
                    for m_vh4 in _HORIZON_CHG_RE.finditer(m_vb4.group(1)):
                        if m_vh4.group(1) == h_k and c_v is not None:
                            rep_v4 = safe_float(m_vh4.group(2), default=None)
                            if rep_v4 is not None:
                                exp_v4 = c_v - btc_h[h_k]
                                if abs(rep_v4 - exp_v4) > 0.15:
                                    violations.append(
                                        InvariantViolation(
                                            "PROMPT_EXEC_AND_CROSS_MARKET_MATH",
                                            f"Section4.{bw_s}:vs_btc_{h_k}",
                                            f"bellwether vs_btc {h_k}={fmt_signed(rep_v4, 2)}% != {fmt_signed(c_v, 2)}%-{fmt_signed(btc_h[h_k], 2)}%={fmt_signed(exp_v4, 2)}%",
                                        )
                                    )


def _audit_prompt_calibration_and_breadth(
    violations: list[InvariantViolation],
    prompt_text: str,
) -> None:
    """Validate AI Execution & Plan Calibration arithmetic and Market Breadth ratio/share bounds."""
    for ln in prompt_text.splitlines():
        if "AI Execution & Plan Calibration" in ln:
            if (m_pl := _CAL_PLANS_RE.search(ln)) is not None:
                rep_plans = safe_nonneg_int(m_pl.group(1), default=None)
                inner = m_pl.group(2)
                side_rows = list(_CAL_SIDE_RE.finditer(inner))
                if rep_plans is not None and side_rows:
                    tot_h = tot_i = tot_e = tot_o = 0
                    for m_s in side_rows:
                        tot_h += safe_nonneg_int(m_s.group(1), default=0)
                        tot_i += safe_nonneg_int(m_s.group(2), default=0)
                        tot_e += safe_nonneg_int(m_s.group(3), default=0)
                        tot_o += safe_nonneg_int(m_s.group(4), default=0)
                    sum_p = tot_h + tot_i + tot_e + tot_o
                    if sum_p != rep_plans:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_CALIBRATION_MATH",
                                "Calibration.plans",
                                f"reported plans={rep_plans} != sum(hit+inv+early+open)={sum_p}",
                            )
                        )
                    if (m_hit := _CAL_HIT_RE.search(inner)) is not None:
                        rep_hit = safe_nonneg_float(m_hit.group(1), default=None)
                        resolved = tot_h + tot_i
                        if rep_hit is not None and (resolved <= 0 or abs(rep_hit - (tot_h / resolved) * 100.0) > 1.0):
                            exp_h = (tot_h / resolved) * 100.0 if resolved > 0 else 0.0
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_CALIBRATION_MATH",
                                    "Calibration.hit_rate",
                                    f"reported hit={rep_hit:.0f}% != expected {exp_h:.1f}% ({tot_h}/{resolved})",
                                )
                            )
            for m_lim in _CAL_LIM_RE.finditer(ln):
                f_n = safe_nonneg_int(m_lim.group(1), default=0)
                t_n = safe_nonneg_int(m_lim.group(2), default=0)
                rep_p = safe_nonneg_float(m_lim.group(3), default=None)
                if t_n <= 0 or f_n > t_n or (rep_p is not None and abs(rep_p - (f_n / t_n) * 100.0) > 1.0):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_CALIBRATION_MATH",
                            "Calibration.venue_lim_rate",
                            f"LIM {f_n}/{t_n}[{rep_p}%] inconsistent",
                        )
                    )
            if (m_ob := _CAL_OFFSET_BLOCK_RE.search(ln)) is not None:
                for m_oi in _CAL_OFFSET_ITEM_RE.finditer(m_ob.group(1)):
                    rep_p = safe_nonneg_float(m_oi.group(1), default=None)
                    f_n = safe_nonneg_int(m_oi.group(2), default=0)
                    t_n = safe_nonneg_int(m_oi.group(3), default=0)
                    if t_n <= 0 or f_n > t_n or (rep_p is not None and abs(rep_p - (f_n / t_n) * 100.0) > 1.0):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_CALIBRATION_MATH",
                                "Calibration.fill_by_offset",
                                f"fill_by_offset {rep_p}%[{f_n}/{t_n}] inconsistent",
                            )
                        )
        if "Market Breadth:" in ln:
            if (m_ba := _BREADTH_ADV_RE.search(ln)) is not None:
                ar = safe_float(m_ba.group(1), default=None)
                adv = safe_nonneg_int(m_ba.group(2), default=0)
                tot = safe_nonneg_int(m_ba.group(3), default=0)
                if ar is not None and (tot <= 0 or adv > tot or abs(ar - (adv / tot)) > 0.002):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_BREADTH_MATH",
                            "MarketBreadth.advance_ratio",
                            f"advance_ratio={ar:.4f} != advancing({adv}/{tot})={adv / tot if tot > 0 else 0.0:.4f}",
                        )
                    )
            if (m_vs := _BREADTH_VOL_SHARE_RE.search(ln)) is not None:
                btc_s = safe_float(m_vs.group(1), default=None)
                alt_s = safe_float(m_vs.group(2), default=None)
                if btc_s is not None and alt_s is not None and (btc_s < 0.0 or alt_s < 0.0 or (btc_s + alt_s) > 100.5):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_BREADTH_MATH",
                            "MarketBreadth.vol_share",
                            f"BTC_vol_share={btc_s:.2f}% + Alt_vol_share={alt_s:.2f}% = {btc_s + alt_s:.2f}% > 100%",
                        )
                    )


def _audit_prompt_section1_parity_and_structure(
    violations: list[InvariantViolation],
    prompt_text: str,
    sec3_items: Sequence[tuple[str, float, str]],
) -> None:
    """Validate Scorecard/Binance/RecentFills parity and Sections 0/2/3/4 domain-age, options VRP, VP LVN, Kalman & disjointness."""
    sec1_body = prompt_text.split("### 2.", 1)[0]

    # 1. Scorecard W + BE + L == n_shown (and last K/N obeys K <= N)
    if (m_sc := _SCORECARD_OUTCOME_RE.search(sec1_body)) is not None:
        last_k = safe_nonneg_int(m_sc.group(1), default=None) if m_sc.group(1) else None
        last_n = safe_nonneg_int(m_sc.group(2), default=None) if m_sc.group(2) else None
        exact_n = safe_nonneg_int(m_sc.group(3), default=None) if m_sc.group(3) else None
        w_n = safe_nonneg_int(m_sc.group(4), default=0)
        be_n = safe_nonneg_int(m_sc.group(5), default=0)
        l_n = safe_nonneg_int(m_sc.group(6), default=0)
        exp_n = last_k if last_k is not None else exact_n
        if exp_n is not None and (w_n + be_n + l_n) != exp_n:
            violations.append(
                InvariantViolation(
                    "PROMPT_SECTION1_PARITY_AND_SCORECARD_MATH",
                    "Scorecard.outcome_sum",
                    f"{w_n}W+{be_n}BE+{l_n}L={w_n + be_n + l_n} != reported count={exp_n}",
                )
            )
        if last_k is not None and last_n is not None and last_k > last_n:
            violations.append(
                InvariantViolation(
                    "PROMPT_SECTION1_PARITY_AND_SCORECARD_MATH",
                    "Scorecard.window_bounds",
                    f"last {last_k}/{last_n} has window count > total count",
                )
            )

    # 2. Total <Venue> Portfolio Value live_reconciled vs Portfolio Risk Engine venues(<venue>=...)
    if (m_nav := _NAV_LINE_RE.search(sec1_body)) is not None:
        v_map_s1 = {vk.lower(): _parse_money(vv) for vk, vv in _VENUE_VAL_RE.findall(m_nav.group(5))}
        for m_vlr in _VENUE_LIVE_REC_RE.finditer(sec1_body):
            ven_lbl = m_vlr.group(1).lower()
            ven_amt = v_map_s1.get(ven_lbl)
            live_amt = _parse_money(m_vlr.group(2))
            if ven_amt is not None and live_amt is not None and abs(ven_amt - live_amt) > 0.10:
                violations.append(
                    InvariantViolation(
                        "PROMPT_SECTION1_PARITY_AND_SCORECARD_MATH",
                        f"Section1.VenueReconciled:{ven_lbl}",
                        f"venues({ven_lbl})=${ven_amt:.2f} != Total {m_vlr.group(1)} live_reconciled=${live_amt:.2f}",
                    )
                )

    # 3. Recent_Fills vs Live Tradable Spot Instruments anchor parity (unique-venue symbols)
    pos_anchors_by_sym: dict[str, dict[str, float]] = {}
    pos_sym_counts: dict[str, int] = {}
    for m_inst in _SEC3C_INST_RE.finditer(sec1_body):
        sym_i, rest_i = m_inst.group(2).upper(), m_inst.group(3)
        if "free_" not in rest_i or "limits(" not in rest_i:
            continue
        pos_sym_counts[sym_i] = pos_sym_counts.get(sym_i, 0) + 1
        anc_d: dict[str, float] = {}
        for k_a, pat_a in (
            ("avg_cost", _AVG_COST_PNL_RE),
            ("last_buy_px", _POS_LAST_BUY_RE),
            ("last_sell_px", _POS_LAST_SELL_RE),
            ("lifo_avg", _POS_LIFO_AVG_RE),
        ):
            if (m_a := pat_a.search(rest_i)) is not None:
                if (av := safe_pos_float(m_a.group(1), default=None)) is not None:
                    anc_d[k_a] = av
        if (m_lp := _POS_LIFO_PNL_RE.search(rest_i)) is not None:
            if (lp_mag := _parse_money(m_lp.group(2))) is not None:
                anc_d["lifo_pnl"] = (-lp_mag) if m_lp.group(1) == "-" else lp_mag
        pos_anchors_by_sym[sym_i] = anc_d

    for m_ro in _RECENT_FILL_LINE_RE.finditer(sec1_body):
        sym_r, body_r = m_ro.group(1).upper(), m_ro.group(2)
        if pos_sym_counts.get(sym_r, 0) != 1:
            continue
        p_anc = pos_anchors_by_sym.get(sym_r) or {}
        for fld_lbl, ro_pat, pos_k in (
            ("last_buy_px", _RO_LAST_BUY_RE, "last_buy_px"),
            ("last_sell_px", _RO_LAST_SELL_RE, "last_sell_px"),
            ("avg_cost", _RO_AVG_COST_RE, "avg_cost"),
            ("lifo_avg", _RO_LIFO_AVG_RE, "lifo_avg"),
        ):
            if pos_k in p_anc and (m_rf := ro_pat.search(body_r)) is not None:
                rv = safe_pos_float(m_rf.group(1), default=None)
                pv = p_anc[pos_k]
                if rv is not None and abs(rv - pv) > max(1e-4, pv * 1e-4):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_SECTION1_PARITY_AND_SCORECARD_MATH",
                            f"RecentFillsParity:{sym_r}:{fld_lbl}",
                            f"Recent_Fills {fld_lbl}={rv:g} != Tradable Position {fld_lbl}={pv:g}",
                        )
                    )
        if "lifo_pnl" in p_anc and (m_rlp := _RO_LIFO_PNL_RE.search(body_r)) is not None:
            if (rlp_mag := _parse_money(m_rlp.group(2))) is not None:
                rlp_v = (-rlp_mag) if m_rlp.group(1) == "-" else rlp_mag
                if abs(rlp_v - p_anc["lifo_pnl"]) > 0.05:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_SECTION1_PARITY_AND_SCORECARD_MATH",
                            f"RecentFillsParity:{sym_r}:lifo_pnl",
                            f"Recent_Fills lifo_pnl={rlp_v:.2f} != Tradable Position lifo_pnl={p_anc['lifo_pnl']:.2f}",
                        )
                    )

    # 4. Section 0 Domain Age Ordering (newest <= oldest)
    for m_da in _DOMAIN_AGE_PAIR_RE.finditer(sec1_body):
        dom_n = m_da.group(1)
        new_s = safe_nonneg_float(m_da.group(2), default=None)
        old_s = safe_nonneg_float(m_da.group(4), default=None)
        if new_s is not None and old_s is not None:
            new_sec = new_s * _AGE_UNIT_SEC.get(m_da.group(3), 1.0)
            old_sec = old_s * _AGE_UNIT_SEC.get(m_da.group(5), 1.0)
            if new_sec > old_sec + 2.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                        f"Section0.DomainAgeOrder:{dom_n}",
                        f"{dom_n}_newest={m_da.group(2)}{m_da.group(3)} > {dom_n}_oldest={m_da.group(4)}{m_da.group(5)}",
                    )
                )

    # 5. Section 2 Options VRP Identity (VRP == DVOL - RV) and SPY/QQQ vs Long_Horizon now: parity
    dvol_by_coin: dict[str, float] = {}
    for m_dv in _DVOL_RE.finditer(prompt_text):
        if (dv_v := safe_pos_float(m_dv.group(2), default=None)) is not None:
            dvol_by_coin[m_dv.group(1)] = dv_v
    for m_op in _OPTIONS_RV_VRP_RE.finditer(prompt_text):
        coin_o, body_o = m_op.group(1), m_op.group(2)
        if coin_o in dvol_by_coin:
            m_rv, m_vrp = _RV_ITEM_RE.search(body_o), _VRP_ITEM_RE.search(body_o)
            if m_rv is not None and m_vrp is not None:
                rv_v = safe_pos_float(m_rv.group(1), default=None)
                vrp_v = safe_float(m_vrp.group(1), default=None)
                if rv_v is not None and vrp_v is not None:
                    exp_vrp = dvol_by_coin[coin_o] - rv_v
                    if abs(exp_vrp - vrp_v) > 0.3:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                                f"Options.{coin_o}_VRP",
                                f"{coin_o}_Options VRP={fmt_signed(vrp_v, 1)}% != DVOL({dvol_by_coin[coin_o]:.1f}%)-RV({rv_v:.1f}%)={fmt_signed(exp_vrp, 1)}%",
                            )
                        )
    bench_eq_px: dict[str, float] = {}
    for m_beq in _BENCH_EQ_PX_RE.finditer(prompt_text):
        if (bp_v := _parse_money(m_beq.group(2))) is not None and bp_v > 0.0:
            bench_eq_px.setdefault(m_beq.group(1), bp_v)
    if bench_eq_px:
        for m_lh in _LH_ROW_RE.finditer(prompt_text):
            lh_s = m_lh.group(1).upper()
            if lh_s in bench_eq_px:
                now_lh = _parse_money(m_lh.group(9))
                ref_b = bench_eq_px[lh_s]
                if now_lh is not None and abs(now_lh - ref_b) / ref_b > 0.01:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                            f"LongHorizon.{lh_s}_vs_Sec2",
                            f"Long_Horizon {lh_s} now={now_lh:g} diverges >1% from Section 2 {lh_s}=${ref_b:g}",
                        )
                    )

    # 6. Section 3 vol_profile LVN ordering, Kalman vel/z sign parity, sibling venue header math, and Sec3/Sec4 disjointness
    seen_s3: set[str] = set()
    for sym, _, rest in sec3_items:
        sym_u = sym.upper()
        if sym_u in seen_s3:
            violations.append(
                InvariantViolation(
                    "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                    f"Section3.Duplicate:{sym}",
                    f"symbol {sym} rendered more than once in Section 3",
                )
            )
        seen_s3.add(sym_u)
        if (m_lvn := _VP_SEC_LVN_RE.search(rest)) is not None:
            poc_v = safe_pos_float(m_lvn.group(1), default=None)
            sec_v = safe_pos_float(m_lvn.group(2), default=None)
            lvn_v = safe_pos_float(m_lvn.group(3), default=None)
            if poc_v is not None and sec_v is not None and lvn_v is not None:
                lo_p, hi_p = min(poc_v, sec_v), max(poc_v, sec_v)
                if abs(sec_v - poc_v) <= 1e-12 or lvn_v < lo_p * 0.999 or lvn_v > hi_p * 1.001:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                            f"{sym}:vol_profile_LVN",
                            f"vol_profile LVN={lvn_v:g} inconsistent with [POC={poc_v:g}, sec_POC={sec_v:g}]",
                        )
                    )
        if (m_km := _KALMAN_PAIR_RE.search(rest)) is not None:
            kv = safe_float(m_km.group(1), default=None)
            kz = safe_float(m_km.group(2), default=None)
            if kv is not None and kz is not None and kv * kz < -1e-4:
                violations.append(
                    InvariantViolation(
                        "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                        f"{sym}:kalman_sign",
                        f"kalman_vel={kv:g} has opposite sign to kalman_z={kz:g}",
                    )
                )

    for m_sib in _SIBLING_HDR_RE.finditer(prompt_text):
        main_s, sib_s = m_sib.group(1), m_sib.group(2)
        sib_px = _parse_money(m_sib.group(3))
        rep_sp = safe_float(m_sib.group(4), default=None)
        main_px = _parse_money(m_sib.group(5))
        if sib_px is not None and main_px is not None and main_px > 0.0 and rep_sp is not None:
            exp_sp = ((sib_px - main_px) / main_px) * 100.0
            if abs(exp_sp - rep_sp) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                        f"{main_s}:sibling_header:{sib_s}",
                        f"sibling header delta={fmt_signed(rep_sp, 2)}% != expected {fmt_signed(exp_sp, 2)}% ({sib_s}={sib_px:g}, {main_s}={main_px:g})",
                    )
                )

    if "### 4. MARKET BELLWETHERS" in prompt_text:
        sec4_body = prompt_text.split("### 4. MARKET BELLWETHERS", 1)[1]
        seen_s4: set[str] = set()
        for m_bw4 in _SEC4_BW_ROW_RE.finditer(sec4_body):
            bw_s = m_bw4.group(1)
            bw_u = bw_s.upper()
            if bw_u in seen_s4:
                violations.append(
                    InvariantViolation(
                        "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                        f"Section4.Duplicate:{bw_s}",
                        f"bellwether {bw_s} rendered more than once in Section 4",
                    )
                )
            seen_s4.add(bw_u)
            if bw_u in seen_s3:
                violations.append(
                    InvariantViolation(
                        "PROMPT_STRUCTURE_AND_OPTIONS_MATH",
                        f"Section3_4_Overlap:{bw_s}",
                        f"symbol {bw_s} rendered in BOTH Section 3 tradable cards AND Section 4 observe-only bellwethers",
                    )
                )


def _audit_prompt_numeric_hygiene(
    violations: list[InvariantViolation],
    prompt_text: str,
) -> None:
    """Detect leaked Python None, NaN/Inf, double signs, negative zeros, scientific notation, or double units."""
    body = prompt_text.split("LIVE TELEMETRY SNAPSHOT:", 1)[1] if "LIVE TELEMETRY SNAPSHOT:" in prompt_text else prompt_text
    in_news = False
    for raw_ln in body.splitlines():
        s_ln = raw_ln.strip()
        if s_ln.startswith("- Live Breaking News Feed"):
            in_news = True
            continue
        if in_news:
            if s_ln.startswith("* "):
                continue
            in_news = False
        if s_ln.startswith("```") or s_ln.startswith('"'):
            continue
        for kind, pat in _NUMERIC_HYGIENE_SPECS:
            if (m_h := pat.search(raw_ln)) is not None:
                violations.append(
                    InvariantViolation(
                        "PROMPT_NUMERIC_HYGIENE",
                        kind,
                        f"malformed numeric token {m_h.group(0).strip()!r} in: {s_ln[:120]}",
                    )
                )
                break


def evaluate_live_prompt_invariants(
    snapshot: Any,
    prompt_text: str,
    *,
    resume_prompt_text: Optional[str] = None,
    non_tradable_symbols: Collection[str] = (),
    gateway_min_usd: float = 0.0,
) -> list[InvariantViolation]:
    """Run deduplicated snapshot + full/warm rendered prompt invariant validation for continuous runtime checks."""
    out: list[InvariantViolation] = []
    seen: set[tuple[str, str, str]] = set()

    def _extend(items: Sequence[InvariantViolation]) -> None:
        for v in items:
            sig = (v.code, v.subject, v.detail)
            if sig not in seen:
                seen.add(sig)
                out.append(v)

    _extend(
        audit_snapshot_invariants(
            snapshot,
            non_tradable_symbols=non_tradable_symbols,
            gateway_min_usd=gateway_min_usd,
        )
    )
    if prompt_text:
        _extend(
            audit_rendered_prompt_text(
                prompt_text,
                non_tradable_symbols=non_tradable_symbols,
                gateway_min_usd=gateway_min_usd,
            )
        )
    if resume_prompt_text and resume_prompt_text != prompt_text:
        _extend(
            audit_rendered_prompt_text(
                resume_prompt_text,
                non_tradable_symbols=non_tradable_symbols,
                gateway_min_usd=gateway_min_usd,
            )
        )
    return out


__all__ = [
    "InvariantViolation",
    "audit_rendered_prompt_text",
    "audit_scorecard_temporal_invariants",
    "audit_snapshot_invariants",
    "evaluate_live_prompt_invariants",
]
