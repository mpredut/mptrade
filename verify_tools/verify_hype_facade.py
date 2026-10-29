#!/usr/bin/env python3
"""verify_hype_facade.py — DRY-RUN sanity check of the HYPE facade (Hyperliquid SPOT).

Places ZERO real orders. Through the market_api facade (the `mkt` singleton) it checks:
  - symbol ROUTING: HYPEUSDC -> Hyperliquid; BTCUSDC/TAOUSDC -> Binance;
    bare assets BTC/TAO -> Binance (behaviour-preserving default); HYPE -> Hyperliquid.
  - get_current_price(HYPEUSDC) -> a sane SPOT price (>0).
  - get_price_history(HYPEUSDC) -> a fine-grained ascending series, prices >0.
  - free_balance(HYPE) -> the free SPOT balance (total minus hold).
  - get_orders(HYPEUSDC, BUY/SELL) -> normalised {side,price,qty,timestamp}; SPOT only.
  - get_position_stats (avg_buy/net) reproduced from the facade's orders.
  - the DECISION monitor_price_and_trade would take for HYPE (without placing it).
  - place_order(HYPEUSDC, SELL) is DRY (HL_LIVE_ORDERS off) -> None, no order sent.

Run it locally with the venv that has the HL SDK:
  ../.venv/bin/python verify_hype_facade.py
Exits non-zero if anything is clearly unhealthy (wrong routing, invalid price, real order).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from providers.market_api import api as mkt          # noqa: E402  (facada singleton)
import symbols as sym                        # noqa: E402

HYPE = sym.hypesymbol                        # "HYPEUSDC"
MAXAGE = 17 * 24 * 3600
failures = []


from botcore import safe_float

_f = safe_float


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(f"{name}: {detail}")


def main():
    failures.clear()
    print("==== RUTARE pe symbol (Binance ramane default) ====")
    check("HYPEUSDC -> Hyperliquid", mkt.provider_name_for(HYPE) == "Hyperliquid",
          mkt.provider_name_for(HYPE))
    check("BTCUSDC  -> Binance", mkt.provider_name_for(sym.btcsymbol) == "Binance",
          mkt.provider_name_for(sym.btcsymbol))
    check("TAOUSDC  -> Binance", mkt.provider_name_for(sym.taosymbol) == "Binance",
          mkt.provider_name_for(sym.taosymbol))
    check("asset HYPE -> Hyperliquid", mkt.provider_name_for("HYPE") == "Hyperliquid",
          mkt.provider_name_for("HYPE"))
    check("asset BTC  -> Binance (default)", mkt.provider_name_for("BTC") == "Binance",
          mkt.provider_name_for("BTC"))
    check("asset TAO  -> Binance (default)", mkt.provider_name_for("TAO") == "Binance",
          mkt.provider_name_for("TAO"))

    print("\n==== HYPE MARKET DATA (public, no key) ====")
    price = mkt.get_current_price(HYPE)
    price_f = _f(price, 0.0)
    print(f"  get_current_price(HYPEUSDC) = {price}")
    check("valid SPOT price (>0)", price_f > 0.0,
          repr(price))

    hist = mkt.get_price_history(HYPE, 3)
    hist_dicts = [p for p in (hist or []) if isinstance(p, dict)] if isinstance(hist, list) else []
    n = len(hist_dicts)
    print(f"  get_price_history(HYPEUSDC, 3h): {n} puncte")
    if hist_dicts:
        print(f"    first={hist_dicts[0]}  last={hist_dicts[-1]}")
    asc = bool(hist_dicts) and all(
        _f(hist_dicts[i].get("timestamp"), 0.0) <= _f(hist_dicts[i + 1].get("timestamp"), 0.0)
        for i in range(len(hist_dicts) - 1)
    )
    pos = bool(hist_dicts) and all(_f(p.get("price"), 0.0) > 0.0 for p in hist_dicts)
    check("the history is granular, ascending, with prices>0", bool(hist_dicts) and asc and pos,
          f"n={n} asc={asc} pos={pos}")
    if price_f > 0.0 and hist_dicts:
        last = _f(hist_dicts[-1].get("price"), 0.0)
        near = last > 0.0 and abs(last - price_f) / price_f < 0.10
        check("the last close ~ the current price (<10%)", near,
              f"close={last} price={price}")

    print("\n==== CONT SPOT HYPE (citire) ====")
    free_hype = mkt.free_balance_for("hyperliquid", "HYPE")
    free_usdc = mkt.free_balance_for("hyperliquid", "USDC")
    print(f"  free_balance(HYPE) = {free_hype}")
    print("  free_balance(USDC) = "
          f"{free_usdc}  (provider explicit: hyperliquid)")
    check("free_balance(HYPE) numeric >=0", _f(free_hype, -1.0) >= 0.0,
          repr(free_hype))

    buys = mkt.get_orders(HYPE, "BUY", MAXAGE) or []
    sells = mkt.get_orders(HYPE, "SELL", MAXAGE) or []
    print(f"  get_orders BUY={len(buys)} SELL={len(sells)} (ultimele {MAXAGE//86400} zile, SPOT)")
    shape_ok = all(isinstance(o, dict) and set(("side", "price", "qty", "timestamp")) <= set(o) for o in (buys + sells))
    sides_ok = all(isinstance(o, dict) and o.get("side") == "BUY" for o in buys) and all(isinstance(o, dict) and o.get("side") == "SELL" for o in sells)
    check("orders normalised to {side,price,qty,timestamp}", shape_ok)
    check("side filtrat corect (BUY/SELL)", sides_ok)
    for o in (buys[:2] + sells[:2]):
        if isinstance(o, dict):
            print(f"    {o.get('side')} px={o.get('price')} qty={o.get('qty')} ts={o.get('timestamp')}")

    print("\n==== POZITIE (logica get_position_stats) ====")
    valid_buys = [
        o for o in buys
        if isinstance(o, dict) and _f(o.get("qty"), 0.0) > 0.0 and _f(o.get("price"), 0.0) > 0.0
    ]
    valid_sells = [
        o for o in sells
        if isinstance(o, dict) and _f(o.get("qty"), 0.0) > 0.0
    ]
    tbq = sum(_f(o.get("qty"), 0.0) for o in valid_buys)
    tbv = sum(_f(o.get("price"), 0.0) * _f(o.get("qty"), 0.0) for o in valid_buys)
    tsq = sum(_f(o.get("qty"), 0.0) for o in valid_sells)
    avg_buy = tbv / tbq if tbq else 0.0
    net = tbq - tsq
    print(f"  buy_qty={tbq:.6f} sell_qty={tsq:.6f} net_qty={net:.6f} avg_buy={avg_buy:.4f}")
    check("avg_buy is plausible (0 or positive and within price range)",
          avg_buy == 0.0 or (price_f > 0.0 and 0.2 * price_f <= avg_buy <= 5 * price_f),
          f"avg_buy={avg_buy} price={price}")

    print("\n==== A simulated DECISION (like monitor_price_and_trade, WITHOUT placing) ====")
    if valid_buys and price_f > 0.0:
        last_buy = _f(sorted(valid_buys, key=lambda x: _f(x.get("timestamp"), 0.0), reverse=True)[0].get("price"), 0.0)
        if last_buy > 0:
            inc = (price_f - last_buy) / last_buy
            print(f"  last buy={last_buy}  price={price}  change={inc*100:+.2f}%")
            gain, lost = 0.092, 0.049
            if inc >= 0.17:
                decision = "HARD-TP (sell a fraction, regardless of the trend)"
            elif inc > gain:
                decision = "gain above threshold -> sell ONLY if the trend is NOT up"
            elif (last_buy - price_f) / last_buy > lost:
                decision = "loss above threshold -> sell ONLY if the trend is NOT up"
            else:
                decision = "nothing interesting"
            print(f"  -> decision: {decision}")
        else:
            print("  (last buy price invalid — no sell decision)")
    else:
        print("  (no recent SPOT buys or no price — no sell decision)")

    print("\n==== ORDER GATE: place_order must be DRY ====")
    saved_live_orders = os.environ.pop("HL_LIVE_ORDERS", None)   # asiguram DRY pt test
    try:
        res = mkt.place_order(HYPE, "SELL", price_f if price_f > 0.0 else 1.0, 0.0001)
    finally:
        if saved_live_orders is not None:
            os.environ["HL_LIVE_ORDERS"] = saved_live_orders
    check("place_order(HYPEUSDC) DRY -> None (no real order)", res is None, repr(res))

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} probleme:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — facada HYPE (Hyperliquid spot) e sanatoasa in DRY-RUN.")
    sys.exit(0)


if __name__ == "__main__":
    main()
