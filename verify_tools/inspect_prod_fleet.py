#!/usr/bin/env python3
"""inspect_prod_fleet.py -- Clean, robust inspection of PROD live fleet state via SSH.

Queries mptrade-prod cachedb states and returns structured JSON or human-readable tables:
- Cash balances (free and locked) across Binance, Kraken, Hyperliquid, Trading212
- Active topic conversation thread state (steps, turns, rotations, summaries)
- Resting limit orders and asset positions
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from account_cache_health import (
    DEFAULT_REQUIRED_VENUE_STREAMS,
    SUPPORTED_HEALTH_VENUES,
    inspect_all_venues_health_snapshot,
)
from botcore import (
    extract_current_price_item,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    is_live_network_backfill_enabled,
    safe_float,
)
from cacheaccount import CacheOrderOpenManager
from cachepaths import (
    CACHE_ACCOUNT_HEALTH_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    LLM_TOPIC_THREADS_FILENAME,
    PORTFOLIO_POSITIONS_STATE_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    cache_path,
    is_equity_venue,
)
from instrument_registry import is_symbol_market_open, symbols_for
from providers.order_ohlc_utils import extract_orig_qty
from providers.portfolio_state_store import VENUE_BALANCE_CACHE_FILENAMES
from providers.portfolio_valuation import compose_unified_portfolio_state
from state_io import encode_json_payload, parse_json_payload, read_json_file


_PROD_ROOT = "/home/predut/mptrade"


def _to_prod_remote_path(path: str) -> str:
    norm = os.path.abspath(path).replace("\\", "/")
    root_norm = os.path.abspath(_ROOT_DIR).replace("\\", "/")
    if norm == root_norm or norm.startswith(root_norm + "/"):
        rel = norm[len(root_norm):].lstrip("/")
        return f"{_PROD_ROOT}/{rel}" if rel else _PROD_ROOT
    if "/cachedb/" in norm:
        tail = norm.split("/cachedb/", 1)[1]
        return f"{_PROD_ROOT}/cachedb/{tail}"
    if "/statedb/" in norm:
        tail = norm.split("/statedb/", 1)[1]
        return f"{_PROD_ROOT}/statedb/{tail}"
    return path


def ssh_cat(remote_path: str) -> dict | list | str | None:
    prod_path = _to_prod_remote_path(remote_path)
    cmd = ["ssh", "-o", "BatchMode=yes", "mptrade-prod", f"cat {prod_path}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        try:
            return parse_json_payload(res.stdout)
        except Exception:
            return res.stdout
    if is_live_network_backfill_enabled() and os.path.isfile(prod_path):
        return read_json_file(prod_path, default=None)
    return None


def inspect_fleet(json_output: bool = False):
    bal_caches: dict[str, dict] = {}
    for v_key, bal_fname in VENUE_BALANCE_CACHE_FILENAMES.items():
        raw_bc = (
            ssh_cat(cache_path(f"account/{bal_fname}"))
            or ssh_cat(cache_path(f"orders/{bal_fname}"))
            or ssh_cat(cache_path(bal_fname))
        )
        if isinstance(raw_bc, dict) and isinstance(raw_bc.get("items"), dict) and raw_bc.get("items"):
            bal_caches[v_key] = raw_bc
    raw_positions = (
        ssh_cat(f"statedb/portfolio/{PORTFOLIO_POSITIONS_STATE_FILENAME}")
        or ssh_cat(cache_path(f"portfolio/{PORTFOLIO_POSITIONS_STATE_FILENAME}"))
        or ssh_cat(cache_path(f"state/{PORTFOLIO_POSITIONS_STATE_FILENAME}"))
        or ssh_cat(cache_path(PORTFOLIO_POSITIONS_STATE_FILENAME))
    )
    pos_state = raw_positions if isinstance(raw_positions, dict) else {}
    if bal_caches or pos_state:
        balances = compose_unified_portfolio_state(bal_caches, pos_state)
    else:
        raw_balances = (
            ssh_cat(f"statedb/portfolio/{VENUE_BALANCES_STATE_FILENAME}")
            or ssh_cat(cache_path(f"portfolio/{VENUE_BALANCES_STATE_FILENAME}"))
            or ssh_cat(cache_path(f"state/{VENUE_BALANCES_STATE_FILENAME}"))
            or ssh_cat(cache_path(VENUE_BALANCES_STATE_FILENAME))
        )
        balances = raw_balances if isinstance(raw_balances, dict) else {}
    raw_topic = (
        ssh_cat(f"statedb/llm/{LLM_TOPIC_THREADS_FILENAME}")
        or ssh_cat(cache_path(f"llm/{LLM_TOPIC_THREADS_FILENAME}"))
        or ssh_cat(cache_path(f"system/{LLM_TOPIC_THREADS_FILENAME}"))
        or ssh_cat(cache_path(LLM_TOPIC_THREADS_FILENAME))
    )
    topic_data = raw_topic if isinstance(raw_topic, dict) else {}
    open_orders_by_venue: dict[str, dict] = {}
    for v_key, open_fname in CacheOrderOpenManager.VENUE_FILENAMES.items():
        raw_oo = (
            ssh_cat(cache_path(f"account/{open_fname}"))
            or ssh_cat(cache_path(f"orders/{open_fname}"))
            or ssh_cat(cache_path(open_fname))
        )
        if isinstance(raw_oo, dict):
            open_orders_by_venue[v_key] = raw_oo

    # 1. Balances & Positions
    cash = {k: v for k, v in balances.items() if isinstance(k, str) and k.startswith("cash:") and isinstance(v, dict)}
    positions = {k: v for k, v in balances.items() if isinstance(k, str) and k.startswith("pos:") and isinstance(v, dict)}

    total_free_cash = sum(safe_float(v.get("free"), 0.0) or 0.0 for v in cash.values())
    total_locked_cash = sum(safe_float(v.get("locked_quote_usd"), 0.0) or 0.0 for v in cash.values())
    total_free_pos_val = sum(safe_float(v.get("value_usd"), 0.0) or 0.0 for v in positions.values())
    total_locked_pos_val = sum(safe_float(v.get("locked_base_usd"), 0.0) or 0.0 for v in positions.values())

    # Unified Portfolio Value (Cash + Assets)
    total_unified_val = total_free_cash + total_locked_cash + total_free_pos_val + total_locked_pos_val
    cash_reserves = total_free_cash + total_locked_cash
    cash_pct = (cash_reserves / total_unified_val * 100.0) if total_unified_val > 0 else 0.0

    # 2. Topic State
    raw_auto_topic = topic_data.get("auto_cryptotrade_executor")
    auto_topic = raw_auto_topic if isinstance(raw_auto_topic, dict) else {}
    raw_history = auto_topic.get("rolling_history")
    history = raw_history if isinstance(raw_history, list) else []
    latest_summary = (
        history[-1].get("summary", "")
        if history and isinstance(history[-1], dict)
        else ""
    )

    # 3. Active Resting Orders from CacheOrderOpenManager (cache_order_open_<venue>.json)
    active_orders = []
    seen_order_keys: set[tuple[str, str, str]] = set()
    def _append_active_order(o: dict, venue: str, sym: str, default_status: object = None) -> None:
        oid = extract_order_id(o)
        if oid:
            dedup_key = (venue, sym, oid)
            if dedup_key in seen_order_keys:
                return
            seen_order_keys.add(dedup_key)
        active_orders.append({
            "venue": venue,
            "symbol": sym,
            "side": extract_order_side(o) or o.get("side"),
            "price": extract_order_price(o, default=0.0) or 0.0,
            "orig_qty": extract_orig_qty(o, default=0.0) or 0.0,
            "status": o.get("status") or default_status,
        })

    for v_key, oo_doc in open_orders_by_venue.items():
        items = oo_doc.get("items") if isinstance(oo_doc, dict) else None
        if isinstance(items, dict):
            for sym, olist in items.items():
                if isinstance(olist, list):
                    for o in olist:
                        if isinstance(o, dict):
                            _append_active_order(o, v_key, str(o.get("symbol") or sym).strip(), "NEW")

    # 4. Market Telemetry, Sentiment & Health (Optional Enrichment)
    raw_sentiment = ssh_cat(cache_path("macro/sentiment_eval.json"))
    sentiment_data = raw_sentiment if isinstance(raw_sentiment, dict) else {}
    raw_breadth = ssh_cat(cache_path("macro/market_breadth_collect.json"))
    breadth_data = raw_breadth if isinstance(raw_breadth, dict) else {}
    raw_health = ssh_cat(cache_path("system/fleet_health_status.json"))
    health_data = raw_health if isinstance(raw_health, dict) else {}
    raw_acct_health = (
        ssh_cat(cache_path(f"account/{CACHE_ACCOUNT_HEALTH_FILENAME}"))
        or ssh_cat(cache_path(CACHE_ACCOUNT_HEALTH_FILENAME))
    )
    acct_health_summary: dict[str, dict] = {}
    if isinstance(raw_acct_health, dict) and raw_acct_health:
        now_s = time.time()
        req_by_venue = {}
        for v_key in SUPPORTED_HEALTH_VENUES:
            if not is_equity_venue(v_key):
                req_by_venue[v_key] = ("filled_orders", "open_orders", "balances")
            else:
                v_syms = list(symbols_for(v_key))
                req_by_venue[v_key] = (
                    ("filled_orders", "open_orders", "balances")
                    if (not v_syms or any(is_symbol_market_open(s, now_s) for s in v_syms))
                    else DEFAULT_REQUIRED_VENUE_STREAMS
                )
        acct_statuses = inspect_all_venues_health_snapshot(
            raw_acct_health,
            now_ms=round(now_s * 1000),
            required_streams_by_venue=req_by_venue,
        )
        for v_key, st in acct_statuses.items():
            acct_health_summary[v_key] = {
                "ready": st.ready,
                "reason": st.reason,
                "filled_orders_age_sec": round(st.filled_orders_age_sec, 1) if st.filled_orders_age_sec is not None else None,
                "open_orders_age_sec": round(st.open_orders_age_sec, 1) if st.open_orders_age_sec is not None else None,
                "balances_age_sec": round(st.balances_age_sec, 1) if st.balances_age_sec is not None else None,
                "trades_age_sec": round(st.trades_age_sec, 1) if st.trades_age_sec is not None else None,
            }
    raw_prices = ssh_cat(cache_path(f"prices/{CACHE_CURRENT_PRICE_FILENAME}")) or ssh_cat(cache_path(CACHE_CURRENT_PRICE_FILENAME))
    prices_data = raw_prices.get("items", {}) if isinstance(raw_prices, dict) and isinstance(raw_prices.get("items"), dict) else {}

    market_telemetry = {}
    for sym in ("BTCUSDC", "TAOUSDC", "HYPEUSDC"):
        ob = ssh_cat(cache_path(f"telemetry/orderbook_depth_{sym}.json"))
        whale = ssh_cat(cache_path(f"telemetry/whale_snapshot_{sym}.json"))
        px_val = extract_current_price_item(prices_data.get(sym))

        entry_dict = {}
        if px_val is not None:
            entry_dict["price"] = px_val
        if isinstance(ob, dict):
            entry_dict["orderbook"] = {
                "mid_price": ob.get("mid_price"),
                "bid_depth_usd": ob.get("bid_depth_usd"),
                "ask_depth_usd": ob.get("ask_depth_usd"),
                "imbalance_ratio": ob.get("imbalance_ratio"),
                "largest_bid_wall_usd": ob.get("largest_bid_wall_usd"),
                "largest_bid_wall_price": ob.get("largest_bid_wall_price"),
                "largest_ask_wall_usd": ob.get("largest_ask_wall_usd"),
                "largest_ask_wall_price": ob.get("largest_ask_wall_price"),
            }
        if isinstance(whale, dict):
            entry_dict["derivatives"] = {
                "top_traders_long_pct": whale.get("top_traders_long_pct"),
                "top_traders_long_ratio": whale.get("top_traders_long_ratio"),
                "open_interest_usd": whale.get("open_interest_usd"),
                "taker_ratio_5m": whale.get("taker_ratio_5m"),
                "taker_ratio_1h": whale.get("taker_ratio_1h"),
                "cvd_5m_usd": whale.get("cvd_5m_usd"),
                "cvd_1h_usd": whale.get("cvd_1h_usd"),
                "cvd_4h_usd": whale.get("cvd_4h_usd"),
            }
        if entry_dict:
            market_telemetry[sym] = entry_dict

    result = {
        "unified_nav_usd": round(total_unified_val, 2),
        "total_free_cash_usd": round(total_free_cash, 2),
        "total_locked_cash_usd": round(total_locked_cash, 2),
        "total_cash_reserves_usd": round(cash_reserves, 2),
        "total_free_assets_usd": round(total_free_pos_val, 2),
        "total_locked_assets_usd": round(total_locked_pos_val, 2),
        "cash_breakdown": cash,
        "positions_breakdown": positions,
        "active_orders": active_orders,
        "topic_state": {
            "conversation_id": auto_topic.get("conversation_id"),
            "step_count": auto_topic.get("step_count"),
            "turn_count": auto_topic.get("turn_count"),
            "rotation_count": auto_topic.get("rotation_count"),
            "compaction_count": auto_topic.get("compaction_count"),
            "latest_summary": latest_summary
        }
    }
    if market_telemetry:
        result["market_telemetry"] = market_telemetry
    if sentiment_data:
        result["sentiment"] = {
            "market_bias": sentiment_data.get("market_bias"),
            "confidence": sentiment_data.get("confidence"),
            "recommended_action": sentiment_data.get("recommended_action"),
            "summary": sentiment_data.get("summary"),
        }
    if health_data:
        result["fleet_health"] = {
            "fleet_healthy": health_data.get("fleet_healthy"),
            "running_processes": health_data.get("running_processes"),
            "total_processes": health_data.get("total_processes"),
            "stale_processes": health_data.get("stale_processes"),
        }
    if acct_health_summary:
        result["account_cache_health"] = acct_health_summary

    if json_output:
        sys.stdout.write(encode_json_payload(result, indent=2) + "\n")
        return result

    print("=== CASH BALANCES ===")
    for k, v in cash.items():
        f = safe_float(v.get("free"), 0.0) or 0.0
        l = safe_float(v.get("locked_quote_usd"), 0.0) or 0.0
        print(f"  {k}: Free=${f:,.2f} | Locked=${l:,.2f}")
    print(f"  --> TOTAL FREE CASH: ${total_free_cash:,.2f} USD")
    print(f"  --> TOTAL LOCKED CASH: ${total_locked_cash:,.2f} USD")
    print(f"  --> TOTAL CASH RESERVES: ${cash_reserves:,.2f} USD ({cash_pct:.1f}%)")

    print("\n=== TOTAL ESTIMATED PORTFOLIO VALUE ===")
    print(f"  Unified Value (Assets + Cash): ${total_unified_val:,.2f} USD")

    print(
        "\n=== TOPIC ROTATION STATE ===\n"
        "  Topic: auto_cryptotrade_executor\n"
        f"  Rotation Count: #{auto_topic.get('rotation_count')}\n"
        f"  Step Count: {auto_topic.get('step_count')} / 240\n"
        f"  Turn Count: {auto_topic.get('turn_count')}\n"
        f"  Compaction Count: {auto_topic.get('compaction_count')}\n"
        f"  Active Conversation ID: {auto_topic.get('conversation_id')}"
    )

    print("\n=== ASSET POSITIONS ===")
    for k, v in positions.items():
        val = safe_float(v.get("value_usd"), 0.0) or 0.0
        l_usd = safe_float(v.get("locked_base_usd"), 0.0) or 0.0
        print(f"  {k}: Free=${val:,.2f} | Locked Base=${l_usd:,.2f} | Total=${val + l_usd:,.2f}")

    if active_orders:
        print(f"\n=== ACTIVE RESTING ORDERS IN CACHE ({len(active_orders)}) ===")
        for o in active_orders:
            print(f"  [{o['venue']}] {o['side']} {o['symbol']} @ ${o['price']:,.2f} (qty={o['orig_qty']})")

    if market_telemetry:
        print("\n=== MARKET TELEMETRY & ORDERBOOK DEPTH ===")
        for sym, d in market_telemetry.items():
            px_str = f"${d['price']:,.2f}" if "price" in d else "N/A"
            ob = d.get("orderbook", {})
            deriv = d.get("derivatives", {})
            bids_str = f"${ob.get('bid_depth_usd', 0)/1e3:,.1f}K" if ob.get("bid_depth_usd") else "N/A"
            asks_str = f"${ob.get('ask_depth_usd', 0)/1e3:,.1f}K" if ob.get("ask_depth_usd") else "N/A"
            imb_str = f"{ob.get('imbalance_ratio', 0):.3f}" if ob.get("imbalance_ratio") is not None else "N/A"
            wall_str = (
                f"${ob.get('largest_bid_wall_usd', 0)/1e3:,.1f}K @ ${(ob.get('largest_bid_wall_price') or 0):,.2f}"
                if ob.get("largest_bid_wall_usd")
                else "N/A"
            )
            cvd5 = f"${deriv.get('cvd_5m_usd', 0)/1e3:+,.1f}K" if deriv.get("cvd_5m_usd") is not None else "N/A"
            cvd1h = f"${deriv.get('cvd_1h_usd', 0)/1e6:+,.2f}M" if deriv.get("cvd_1h_usd") is not None else "N/A"
            cvd4h = f"${deriv.get('cvd_4h_usd', 0)/1e6:+,.2f}M" if deriv.get("cvd_4h_usd") is not None else "N/A"
            tr5 = f"{deriv.get('taker_ratio_5m', 0):.2f}" if deriv.get("taker_ratio_5m") is not None else "N/A"
            long_pct = f"{deriv.get('top_traders_long_pct', 0)*100:.1f}%" if deriv.get("top_traders_long_pct") else "N/A"
            print(f"  {sym}: Px={px_str} | Imb={imb_str} (Bids={bids_str} Asks={asks_str}) | Wall={wall_str} | CVD 5m={cvd5} (tr={tr5}) 1h={cvd1h} 4h={cvd4h} | Long={long_pct}")

    if sentiment_data or health_data:
        print("\n=== MACRO SENTIMENT & FLEET HEALTH ===")
        if sentiment_data:
            print(f"  Sentiment: {sentiment_data.get('market_bias')} | Conf={sentiment_data.get('confidence')} | Action={sentiment_data.get('recommended_action')}")
        if breadth_data and isinstance(breadth_data.get("snapshot"), dict):
            snap = breadth_data["snapshot"]
            print(f"  Breadth: {snap.get('advancing_pct', 0)*100:.1f}% advancing | Regime={breadth_data.get('regime')}")
        if health_data:
            print(f"  Fleet Health: {health_data.get('running_processes')}/{health_data.get('total_processes')} processes running | Healthy={health_data.get('fleet_healthy')}")

    if acct_health_summary:
        print("\n=== ACCOUNT CACHE HEALTH ===")
        for v_key, st in acct_health_summary.items():
            f_age = f"{st['filled_orders_age_sec']:.1f}s" if st.get("filled_orders_age_sec") is not None else "n/a"
            o_age = f"{st['open_orders_age_sec']:.1f}s" if st.get("open_orders_age_sec") is not None else "n/a"
            b_age = f"{st['balances_age_sec']:.1f}s" if st.get("balances_age_sec") is not None else "n/a"
            t_age = f"{st['trades_age_sec']:.1f}s" if st.get("trades_age_sec") is not None else "n/a"
            status_lbl = "READY" if st.get("ready") else f"FAIL ({st.get('reason')})"
            print(f"  {v_key:12} {status_lbl} | filled={f_age} open={o_age} bal={b_age} trades={t_age}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PROD live fleet state via SSH")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()
    inspect_fleet(json_output=args.json)
