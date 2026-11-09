from botcore import safe_float


# ------------------------------
# CACHE LOCAL
# ------------------------------
balances_cache = {}
orders_cache = {}
trades_cache = []
_MAX_TRADES_CACHE = 1000
_MAX_ORDERS_CACHE = 5000


# ------------------------------
# MESSAGE-HANDLING FUNCTIONS
# ------------------------------
def handle_execution_report(data):
    if not isinstance(data, dict):
        return
    symbol = data.get("s", "")
    order_id = data.get("i")
    status = data.get("X", "")
    event_type = data.get("x", "")
    side = data.get("S", "")

    if order_id is not None and not isinstance(order_id, bool):
        orders_cache[order_id] = {
            "symbol": symbol,
            "side": side,
            "type": data.get("o", ""),
            "status": status,
            "price": safe_float(data.get("p"), 0.0),
            "origQty": safe_float(data.get("q"), 0.0),
            "executedQty": safe_float(data.get("z"), 0.0),
            "updateTime": data.get("E"),
        }
        if len(orders_cache) > _MAX_ORDERS_CACHE:
            for k in list(orders_cache.keys())[:-_MAX_ORDERS_CACHE]:
                orders_cache.pop(k, None)

    if event_type == "TRADE":
        trade = {
            "tradeId": data.get("t"),
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "price": safe_float(data.get("L", data.get("p")), 0.0),
            "qty": safe_float(data.get("l", data.get("z")), 0.0),
            "time": data.get("T", data.get("E")),
        }
        trades_cache.append(trade)
        if len(trades_cache) > _MAX_TRADES_CACHE:
            del trades_cache[:-_MAX_TRADES_CACHE]
        print(f"[TRADE] {symbol} {side} {trade['qty']} @ {trade['price']}")
    else:
        print(f"[ORDER UPDATE] {symbol} → {status}")

def handle_balance_update(asset, delta, timestamp):
    if not isinstance(asset, str) or not asset.strip():
        return
    asset = asset.strip()
    delta_f = safe_float(delta, 0.0)
    prev_f = safe_float(balances_cache.get(asset, 0.0), 0.0)
    balances_cache[asset] = prev_f + delta_f
    print(f"[BALANCE UPDATE] {asset}: {delta_f:+f} (at {timestamp})")

def handle_account_position(balances):
    if not isinstance(balances, (list, tuple)):
        return
    updated = 0
    for bal in balances:
        if not isinstance(bal, dict):
            continue
        raw_asset = bal["a"] if "a" in bal else bal.get("asset", "")
        if not isinstance(raw_asset, str) or not raw_asset.strip():
            continue
        asset = raw_asset.strip()
        free = safe_float(bal.get("f", bal.get("free", 0)), 0.0)
        locked = safe_float(bal.get("l", bal.get("locked", 0)), 0.0)
        balances_cache[asset] = free + locked
        updated += 1
        #print(f"asset {asset}, {balances_cache[asset]}")
    print(f"[ACCOUNT UPDATE] {updated} assets updated")

