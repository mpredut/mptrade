#!/usr/bin/env python3
"""LIVE fleet-wide P&L report: realized P&L (average cost from real fills) plus unrealized
P&L (current position at current price). Sources: Binance get_filled_orders (direct API,
not the incomplete cache) and Kraken/HL state. Binance uses a 120-day window, so older
cost basis is approximate. Run with: ./myenv/bin/python verify_tools/pnl_report.py"""
from collections import defaultdict
import math
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    safe_float,
    safe_pos_float,
)
from order_retry import get_accepted_order_owner_map
from providers.market_api import api as mkt
from providers.order_id_context import resolve_order_owner, resolve_priority_bot_owner
from providers.portfolio_state_store import resolve_kraken_trade_cache_position
from instrument_registry import discover_tradable_specs, symbols_for, venue_for_symbol
from state_io import read_json_dict

WINDOW_D = 120


def _fetch_provider_fills(prov, symbol: str, start_ms: int) -> list[dict]:
    if prov is None:
        return []
    raw_fills = None
    if callable(getattr(prov, "fetch_remote_trades_with_client", None)):
        raw_fills = prov.fetch_remote_trades_with_client(symbol, start_ms, strict=False)
    if raw_fills is None and callable(getattr(prov, "fetch_remote_trades", None)):
        raw_fills = prov.fetch_remote_trades(symbol, start_ms)
    return [f for f in (raw_fills or []) if isinstance(f, dict)]


def venue_symbol_pnl(symbol: str, provider_name: str | None = None) -> dict:
    st = math.trunc((time.time() - WINDOW_D * 24 * 3600) * 1000)
    venue = provider_name or venue_for_symbol(symbol, default="binance")
    prov = mkt.provider_by_name(venue)
    fills = _fetch_provider_fills(prov, symbol, st)
    fills.sort(key=lambda t: extract_timestamp_ms(t, default=0) or 0)
    qty = cost = realized = 0.0
    buys = sells = 0
    for t in fills:
        p = extract_order_price(t)
        q = extract_order_qty(t)
        if p <= 0 or q <= 0:
            continue
        side = extract_order_side(t)
        if side == "BUY":
            qty += q; cost += q * p; buys += 1
        elif side == "SELL":
            sells += 1
            if qty > 1e-12:
                m = min(q, qty); avg = cost / qty
                realized += (p - avg) * m; cost -= avg * m; qty -= m
    try:
        cur = safe_pos_float(mkt.get_current_price(symbol), default=0.0)
    except Exception:
        cur = 0.0
    unreal = (qty * cur - cost) if (qty > 1e-9 and cur > 0) else 0.0
    return dict(realized=realized, unreal=unreal, pos=qty, cur=cur, n=len(fills), buys=buys, sells=sells)


def binance_symbol(symbol):
    return venue_symbol_pnl(symbol, provider_name="binance")


# -- PER-BOT ATTRIBUTION (from get_all_orders / fills, by clientOrderId or ledger owner) --
# Separate PER-BOT activity (RT_/TA_/MT_/AG_/AI_) from MANUAL activity (and_/web_/x-).
# realized_own covers only each bot's own round trip: min(buy_qty, sell_qty) times
# (avg_sell - avg_buy). For bots that exchange inventory (tradeall buys TAO while
# rtrade sells it), net_qty shows which one accumulates versus distributes.


def _bot(cid, order=None, owner_map=None):
    owner = resolve_order_owner(cid) if cid else "unspecified"
    if owner in ("unspecified", "unknown") and isinstance(order, dict):
        owner = resolve_priority_bot_owner(order, owner_map=owner_map)
    if owner == "manual":
        return "MANUAL(app/web)"
    return owner


def per_bot(symbol, provider_name: str | None = None):
    venue = provider_name or venue_for_symbol(symbol, default="binance")
    try:
        prov = mkt.provider_by_name(venue)
        if prov is not None and callable(getattr(prov, "fetch_recent_orders", None)):
            orders = prov.fetch_recent_orders(symbol, limit=1000) or []
            src_kind = "orders"
        else:
            st = math.trunc((time.time() - WINDOW_D * 24 * 3600) * 1000)
            orders = _fetch_provider_fills(prov, symbol, st)
            src_kind = "fills"
    except Exception as e:  # noqa: BLE001
        print(f"  {symbol} [{venue}]: ERROR {e}"); return
    owner_map = get_accepted_order_owner_map() or {}
    g = defaultdict(lambda: {"bq": 0.0, "bv": 0.0, "sq": 0.0, "sv": 0.0})
    for o in orders:
        if not isinstance(o, dict):
            continue
        eq = safe_pos_float(o.get("executedQty", 0), default=0.0) or extract_order_qty(o)
        if eq <= 0:
            continue
        cqq = safe_pos_float(o.get("cummulativeQuoteQty", 0), default=0.0)
        if cqq <= 0:
            op = extract_order_price(o)
            cqq = eq * op if op > 0 else 0.0
        if cqq <= 0:
            continue
        a = g[_bot(o.get("clientOrderId", ""), order=o, owner_map=owner_map)]
        side = extract_order_side(o)
        if side in ("BUY", "SELL"):
            pfx = "b" if side == "BUY" else "s"
            a[f"{pfx}q"] += eq
            a[f"{pfx}v"] += cqq
    print(f"\n  {symbol} [{venue}] (the last {len(orders)} {src_kind}):")
    print(f"    {'source':<16}{'buy$':>9}{'sell$':>9}{'net_qty':>12}{'realized_own':>14}")
    for b, a in sorted(g.items(), key=lambda x: -(x[1]["bq"] + x[1]["sq"])):
        avg_b = a["bv"] / a["bq"] if a["bq"] else 0.0
        avg_s = a["sv"] / a["sq"] if a["sq"] else 0.0
        matched = min(a["bq"], a["sq"])
        realized_own = matched * (avg_s - avg_b) if (avg_b and avg_s) else 0.0
        print(f"    {b:<16}{a['bv']:>9.0f}{a['sv']:>9.0f}{a['bq'] - a['sq']:>+12.4f}{realized_own:>+14.2f}")


def _active_crypto_specs():
    specs = discover_tradable_specs(asset_scope="crypto", only_workspace_conf=True)
    if isinstance(specs, dict) and specs:
        return list(specs.values())
    return [type("_Spec", (), {"provider": venue_for_symbol(s, default="binance"), "symbol": s})() for s in symbols_for("binance")]


def main():
    print(f"\n=== FLEET P&L REPORT ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"{'source':<24}{'realized':>11}{'unrealized':>12}{'position':>12}")
    tot_r = tot_u = 0.0
    cachedb_dir = os.path.join(ROOT, "cachedb")
    active_specs = _active_crypto_specs()

    for spec in active_specs:
        venue = str(getattr(spec, "provider", "") or "binance").strip().lower()
        sym = str(getattr(spec, "symbol", "") or "").strip()
        if not sym:
            continue
        label = f"{venue.capitalize()} {sym}"
        try:
            k_pos = resolve_kraken_trade_cache_position(sym, cachedb_dir)
            if isinstance(k_pos, dict):
                cur = safe_pos_float(mkt.get_current_price(sym), default=0.0) or 0.0
                k_qty = safe_float(k_pos.get("qty"), 0.0)
                k_cost = safe_float(k_pos.get("cost"), 0.0)
                kr = safe_float(k_pos.get("realized_net"), 0.0)
                ku = (k_qty * cur - k_cost) if (k_qty > 1e-9 and k_cost > 0.0 and cur > 0.0) else 0.0
                tot_r += kr; tot_u += ku
                print(
                    f"{label:<24}{kr:>+11.2f}{ku:>+12.2f}{k_qty:>10.4f}  "
                    f"(cache avg cost {k_cost / k_qty if k_qty else 0:.2f}, cur ~{cur:.2f})"
                )
                continue
            d = venue_symbol_pnl(sym, provider_name=venue)
            tot_r += d["realized"]; tot_u += d["unreal"]
            print(
                f"{label:<24}{d['realized']:>+11.2f}{d['unreal']:>+12.2f}"
                f"{d['pos']:>10.4f}  ({d['n']} fills {WINDOW_D}d, {d['buys']}b/{d['sells']}s @ {d['cur']})"
            )
        except Exception as e:
            print(f"{label:<24} ERROR: {e}")

    # Optional archived Hyperliquid delta-neutral funding record (retired Aug 2026)
    dn_path = os.path.join(ROOT, "hyperliquid/.state_dn_HYPE.json")
    if os.path.exists(dn_path):
        try:
            h = read_json_dict(dn_path, default=None, recover_previous=True)
            if isinstance(h, dict):
                h_fund = safe_float(h.get("funding_accrued"), 0.0)
                h_fees = safe_float(h.get("fees_paid"), 0.0)
                h_spot = safe_float(h.get("spot_qty"), 0.0)
                hl = h_fund - h_fees
                tot_r += hl
                print(f"{'HL dn_bot (archived)':<24}{hl:>+11.2f}{0.0:>+12.2f}{h_spot:>10.4f}  (funding {h_fund:.2f} - fee {h_fees:.2f})")
        except Exception as e:
            print(f"{'HL dn_bot (archived)':<24} ERROR: {e}")

    print("-" * 59)
    print(f"{'TOTAL':<24}{tot_r:>+11.2f}{tot_u:>+12.2f}   -> NET {tot_r+tot_u:+.2f} USD")
    print(
        f"\nNote: remote-fill realized figures cover a {WINDOW_D}-day window (older cost basis is approximated).\n"
        "Exchange fee rebates/BNB discounts may differ slightly from gross fill prices."
    )

    print("\n=== ATTRIBUTION PER BOT (clientOrderId / ledger; realized_own = only its own round trips) ===")
    for spec in active_specs:
        venue = str(getattr(spec, "provider", "") or "binance").strip().lower()
        sym = str(getattr(spec, "symbol", "") or "").strip()
        if sym:
            per_bot(sym, provider_name=venue)
    print(
        "Note: bots that trade inventory between themselves (tradeall buys / rtrade sells the same TAO)\n"
        "have no cleanly separable realized — net_qty shows who accumulates (+) versus who distributes (-)."
    )


if __name__ == "__main__":
    main()
