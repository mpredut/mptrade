import threading
import time

from . import bapi as api   # package-local Binance API facade


# ------------------------------
# CACHE LOCAL
# ------------------------------
balances_cache = {}
orders_cache = {}
trades_cache = []

# ------------------------------
# MESSAGE-HANDLING FUNCTIONS
# ------------------------------
def handle_execution_report(data):
    symbol = data["s"]
    order_id = data["i"]
    status = data["X"]
    event_type = data["x"]

    orders_cache[order_id] = {
        "symbol": symbol,
        "side": data["S"],
        "type": data["o"],
        "status": status,
        "price": float(data["p"]),
        "origQty": float(data["q"]),
        "executedQty": float(data["z"]),
        "updateTime": data["E"]
    }

    if event_type == "TRADE":
        trade = {
            "tradeId": data["t"],
            "orderId": order_id,
            "symbol": symbol,
            "side": data["S"],
            "price": float(data["L"]),
            "qty": float(data["l"]),
            "time": data["T"]
        }
        trades_cache.append(trade)
        print(f"[TRADE] {symbol} {data['S']} {trade['qty']} @ {trade['price']}")
    else:
        print(f"[ORDER UPDATE] {symbol} → {status}")

def handle_balance_update(asset, delta, timestamp):
    balances_cache[asset] = balances_cache.get(asset, 0) + delta
    print(f"[BALANCE UPDATE] {asset}: {delta:+f} (at {timestamp})")

def handle_account_position(balances):
    for bal in balances:
        asset = bal["a"] if "a" in bal else bal.get("asset", "")
        free = float(bal.get("f", bal.get("free", 0)))
        locked = float(bal.get("l", bal.get("locked", 0)))
        balances_cache[asset] = free + locked
        #print(f"asset {asset}, {balances_cache[asset]}")
    print(f"[ACCOUNT UPDATE] {len(balances)} assets updated")

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
