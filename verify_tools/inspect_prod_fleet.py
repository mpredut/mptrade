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
import subprocess
import sys


def ssh_cat(remote_path: str) -> dict | list | str | None:
    cmd = ["ssh", "mptrade-prod", f"cat {remote_path}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        try:
            return json.loads(res.stdout)
        except Exception:
            return res.stdout
    return None


def inspect_fleet(json_output: bool = False):
    balances = ssh_cat("/home/predut/mptrade/cachedb/state/venue_balances_state.json") or {}
    topic_data = ssh_cat("/home/predut/mptrade/cachedb/system/llm_topic_threads.json") or {}
    orders_data = ssh_cat("/home/predut/mptrade/cachedb/orders/cache_order.json") or {}

    # 1. Balances & Positions
    cash = {k: v for k, v in balances.items() if k.startswith("cash:")}
    positions = {k: v for k, v in balances.items() if k.startswith("pos:")}

    total_free_cash = sum(v.get("free", 0.0) for v in cash.values())
    total_locked_cash = sum(v.get("locked_quote_usd", 0.0) for v in cash.values())
    total_free_pos_val = sum(v.get("value_usd", 0.0) for v in positions.values())
    total_locked_pos_val = sum(v.get("locked_base_usd", 0.0) for v in positions.values())

    # Unified Portfolio Value (Cash + Assets)
    total_unified_val = total_free_cash + total_locked_cash + total_free_pos_val + total_locked_pos_val

    # 2. Topic State
    auto_topic = topic_data.get("auto_cryptotrade_executor", {}) if isinstance(topic_data, dict) else {}
    history = auto_topic.get("rolling_history") or []
    latest_summary = history[-1].get("summary", "") if history else ""

    # 3. Active Resting Orders from cache_order
    active_orders = []
    if isinstance(orders_data, dict) and "items" in orders_data:
        items = orders_data["items"]
        if isinstance(items, dict):
            for sym, olist in items.items():
                if isinstance(olist, list):
                    for o in olist:
                        if isinstance(o, dict) and o.get("status") in ("NEW", "OPEN", "PARTIALLY_FILLED"):
                            active_orders.append({
                                "venue": "binance",
                                "symbol": o.get("symbol", sym),
                                "side": o.get("side"),
                                "price": float(o.get("price", 0.0)),
                                "orig_qty": float(o.get("origQty", 0.0)),
                                "status": o.get("status"),
                            })

    result = {
        "unified_nav_usd": round(total_unified_val, 2),
        "total_free_cash_usd": round(total_free_cash, 2),
        "total_locked_cash_usd": round(total_locked_cash, 2),
        "total_cash_reserves_usd": round(total_free_cash + total_locked_cash, 2),
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
        print(json.dumps(result, indent=2))
        return

    print("=== CASH BALANCES ===")
    for k, v in cash.items():
        f = v.get("free", 0.0)
        l = v.get("locked_quote_usd", 0.0)
        print(f"  {k}: Free=${f:,.2f} | Locked=${l:,.2f}")
    print(f"  --> TOTAL FREE CASH: ${total_free_cash:,.2f} USD")
    print(f"  --> TOTAL LOCKED CASH: ${total_locked_cash:,.2f} USD")
    print(f"  --> TOTAL CASH RESERVES: ${total_free_cash + total_locked_cash:,.2f} USD ({(total_free_cash + total_locked_cash)/total_unified_val*100:.1f}%)")

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
        val = v.get("value_usd", 0.0)
        l_usd = v.get("locked_base_usd", 0.0)
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
