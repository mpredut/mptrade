import threading
import time

from botcore import safe_float
from . import bapi as api   # package-local Binance API facade


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

# ------------------------------
# POLLING — WebSocket replacement
# ------------------------------

_prev_orders = {}  # orderId -> status
_prev_balances = {}  # asset -> total

def _poll_loop(handler=None, interval_sec=47):
    """
    Poll the Binance API for orders and balances.

    Detect changes and invoke the handler through the same interface previously
    used by the WebSocket implementation.
    """
    global _prev_orders, _prev_balances

    # Initialize the baseline state.
    try:
        account = api.client.get_account()
        for bal in account["balances"]:
            asset = bal["asset"]
            total = float(bal["free"]) + float(bal["locked"])
            _prev_balances[asset] = total
        handle_account_position(account["balances"])
    except Exception as e:
        print(f"[POLLING] Error initialising balances: {e}")

    while True:
        try:
            # --- Check open orders across all symbols. ---
            open_orders = api.client.get_open_orders()
            current_order_ids = set()

            for order in open_orders:
                oid = order["orderId"]
                current_order_ids.add(oid)
                prev_status = _prev_orders.get(oid, {}).get("status")
                curr_status = order["status"]

                if prev_status != curr_status:
                    # Simulate an executionReport event.
                    fake_event = {
                        "e": "executionReport",
                        "s": order["symbol"],
                        "i": oid,
                        "X": curr_status,
                        "x": "TRADE" if curr_status == "FILLED" else "NEW",
                        "S": order["side"],
                        "o": order["type"],
                        "p": order["price"],
                        "q": order["origQty"],
                        "z": order["executedQty"],
                        "E": order["updateTime"],
                        "t": oid,
                        "L": order["price"],
                        "l": order["executedQty"],
                        "T": order["updateTime"],
                    }
                    handle_execution_report(fake_event)
                    if handler:
                        try:
                            handler("executionReport", fake_event)
                        except Exception as e:
                            print(f"⚠️ Error in handler: {e}")

                _prev_orders[oid] = {"status": curr_status}

            # --- Check balances. ---
            account = api.client.get_account()
            changed_balances = []
            for bal in account["balances"]:
                asset = bal["asset"]
                total = float(bal["free"]) + float(bal["locked"])
                if abs(total - _prev_balances.get(asset, 0)) > 1e-8:
                    delta = total - _prev_balances.get(asset, 0)
                    handle_balance_update(asset, delta, int(time.time() * 1000))
                    changed_balances.append(bal)
                    _prev_balances[asset] = total

                    if handler:
                        fake_balance_event = {
                            "e": "balanceUpdate",
                            "a": asset,
                            "d": delta,
                            "T": int(time.time() * 1000)
                        }
                        try:
                            handler("balanceUpdate", fake_balance_event)
                        except Exception as e:
                            print(f"⚠️ Error in the balanceUpdate handler: {e}")

            # Optional periodic debugging.
            # print(f"Orders: {len(orders_cache)}, Trades: {len(trades_cache)}, Balances: {len(balances_cache)}")

        except Exception as e:
            print(f"[POLLING] General error: {e}")

        time.sleep(interval_sec)


# ------------------------------
# startWSevents — preserve the previous interface
# ------------------------------
_ws_started = False

def startWSevents(handler=None, interval_sec=47):
    global _ws_started
    if _ws_started:
        print("⚠️ Polling already started! Ignoring the duplicate call.")
        return
    _ws_started = True

    poll_thread = threading.Thread(
        target=_poll_loop,
        name="poll_loop",
        args=(handler, interval_sec),
        daemon=True
    )
    poll_thread.start()
    print(f"🚀 Binance polling started (interval {interval_sec}s) - replacing the WebSocket\n")
