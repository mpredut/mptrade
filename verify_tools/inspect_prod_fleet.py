#!/usr/bin/env python3
"""inspect_prod_fleet.py — Clean, robust inspection of PROD live fleet state via SSH.

Queries mptrade-prod cachedb states and returns structured JSON or human-readable tables:
- Cash balances (free and locked) across Binance, Kraken, Hyperliquid, Trading212
- Active topic conversation thread state (steps, turns, rotations, summaries)
- Resting limit orders and asset positions
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import safe_float
from cachepaths import cache_path
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
)
from state_io import read_json_file


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
    return path


def ssh_cat(remote_path: str) -> dict | list | str | None:
    prod_path = _to_prod_remote_path(remote_path)
    cmd = ["ssh", "-o", "BatchMode=yes", "mptrade-prod", f"cat {prod_path}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        try:
            return json.loads(res.stdout)
        except Exception:
            return res.stdout
    return None


def inspect_fleet(json_output: bool = False):
    raw_balances = ssh_cat(cache_path("venue_balances_state.json"))
    balances = raw_balances if isinstance(raw_balances, dict) else {}
    raw_topic = ssh_cat(cache_path("system/llm_topic_threads.json")) or ssh_cat(cache_path("llm_topic_threads.json"))
    topic_data = raw_topic if isinstance(raw_topic, dict) else {}
    raw_orders = ssh_cat(cache_path("cache_order.json"))
    orders_data = raw_orders if isinstance(raw_orders, dict) else {}

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

    # 3. Active Resting Orders from venue_balances_state (open_orders:*) and cache_order fallback
    active_orders = []
    seen_order_keys: set[tuple[str, str, str]] = set()
    has_binance_open_orders = False
    open_order_keys = [k for k in balances.keys() if isinstance(k, str) and k.startswith("open_orders:")]
    for k in open_order_keys:
        olist = balances.get(k)
        if isinstance(olist, list):
            parts = k.split(":", 2)
            default_venue = (parts[1] if len(parts) > 1 else "binance").strip().lower() or "binance"
            default_sym = parts[2] if len(parts) > 2 else ""
            for o in olist:
                if isinstance(o, dict):
                    venue = str(o.get("venue") or default_venue).strip().lower() or default_venue
                    sym = str(o.get("symbol") or default_sym).strip()
                    oid = extract_order_id(o)
                    if venue == "binance":
                        has_binance_open_orders = True
                    if oid:
                        dedup_key = (venue, sym, oid)
                        if dedup_key in seen_order_keys:
                            continue
                        seen_order_keys.add(dedup_key)
                    active_orders.append({
                        "venue": venue,
                        "symbol": sym,
                        "side": extract_order_side(o) or o.get("side"),
                        "price": extract_order_price(o, default=0.0) or 0.0,
                        "orig_qty": extract_orig_qty(o, default=0.0) or 0.0,
                        "status": o.get("status") or "NEW",
                    })
    if not has_binance_open_orders and isinstance(orders_data, dict) and "items" in orders_data:
        items = orders_data["items"]
        if isinstance(items, dict):
            for sym, olist in items.items():
                if isinstance(olist, list):
                    for o in olist:
                        if isinstance(o, dict) and o.get("status") in ("NEW", "OPEN", "PARTIALLY_FILLED"):
                            sym_s = str(o.get("symbol") or sym).strip()
                            oid = extract_order_id(o)
                            if oid:
                                dedup_key = ("binance", sym_s, oid)
                                if dedup_key in seen_order_keys:
                                    continue
                                seen_order_keys.add(dedup_key)
                            active_orders.append({
                                "venue": "binance",
                                "symbol": sym_s,
                                "side": extract_order_side(o) or o.get("side"),
                                "price": extract_order_price(o, default=0.0) or 0.0,
                                "orig_qty": extract_orig_qty(o, default=0.0) or 0.0,
                                "status": o.get("status"),
                            })

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

    if json_output:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
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

    print("\n=== TOPIC ROTATION STATE ===")
    print(f"  Topic: auto_cryptotrade_executor")
    print(f"  Rotation Count: #{auto_topic.get('rotation_count')}")
    print(f"  Step Count: {auto_topic.get('step_count')} / 240")
    print(f"  Turn Count: {auto_topic.get('turn_count')}")
    print(f"  Compaction Count: {auto_topic.get('compaction_count')}")
    print(f"  Active Conversation ID: {auto_topic.get('conversation_id')}")

    print("\n=== ASSET POSITIONS ===")
    for k, v in positions.items():
        val = safe_float(v.get("value_usd"), 0.0) or 0.0
        l_usd = safe_float(v.get("locked_base_usd"), 0.0) or 0.0
        print(f"  {k}: Free=${val:,.2f} | Locked Base=${l_usd:,.2f} | Total=${val + l_usd:,.2f}")

    if active_orders:
        print(f"\n=== ACTIVE RESTING ORDERS IN CACHE ({len(active_orders)}) ===")
        for o in active_orders:
            print(f"  [{o['venue']}] {o['side']} {o['symbol']} @ ${o['price']:,.2f} (qty={o['orig_qty']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PROD live fleet state via SSH")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()
    inspect_fleet(json_output=args.json)
