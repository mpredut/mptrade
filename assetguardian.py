from __future__ import annotations
import os
import hashlib
import math
import time

from binance_api import bapi as api
from providers.market_api import api as mkt   # Single guarded proxy (Instrument.place).
from providers.execution_audit import ExecutionAudit
from providers.quantity import resolve_assets

import cacheManager as cm
import symbols as sym
from instrument_registry import symbols_for

# Load tunable parameters from the versioned, secret-free config before reading
# environment variables below. load_dotenv does not overwrite real environment.
from botcore import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))

REQUIRED_CONFIG_KEYS = (
    "AG_CHECK_INTERVAL_SEC",
    "AG_REFERENCE_MINUTES_BACK",
    "AG_BUY_USE_CASH_RATIO",
    "AG_BUY_TIERS",
    "AG_SELL_TIERS",
    "AG_SELL_REARM_GROWTH_PCT",
    "AG_ORDER_MAX_AGE_SEC",
    "AG_RECOVERY_RESET_PCT",
    "AG_NEAR_TRIGGER_SEC",
    "AG_ACTIVE_TRIGGER_SEC",
    "AG_NEAR_TRIGGER_DISTANCE_PCT",
    "AG_TREND_DEFER_MAX_SEC",
    "AG_ORDER_MISSING_CONFIRMATIONS",
)


def _validate_required_config_presence():
    """Abort startup with the complete list of missing mandatory settings."""
    missing = [
        name for name in REQUIRED_CONFIG_KEYS
        if os.environ.get(name) is None or not str(os.environ[name]).strip()
    ]
    if missing:
        raise ValueError(
            "Missing or empty mandatory AssetGuardian settings: "
            + ", ".join(missing))


def _required_config(name):
    """Return one mandatory AG setting or abort startup on missing/empty input."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing or empty mandatory AssetGuardian setting: {name}")
    return str(raw).strip()


def _required_float_config(name):
    raw = _required_config(name)
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid numeric AssetGuardian setting: {name}={raw!r}") from exc


def _required_int_config(name):
    value = _required_float_config(name)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(
            f"Invalid integer AssetGuardian setting: {name}={value!r}")
    return int(value)


# Every financial/operational parameter is mandatory in config.env.
# There are deliberately no hidden code defaults: a missing key stops startup.
_validate_required_config_presence()
CHECK_INTERVAL_SECONDS = _required_float_config("AG_CHECK_INTERVAL_SEC")
ASSET_REFERENCE_MINUTES_BACK_DEFAULT = _required_float_config(
    "AG_REFERENCE_MINUTES_BACK")
BUY_USE_CASH_RATIO = _required_float_config("AG_BUY_USE_CASH_RATIO")
BUY_TIERS_RAW = _required_config("AG_BUY_TIERS")
SELL_TIERS_RAW = _required_config("AG_SELL_TIERS")
SELL_REARM_GROWTH_PERCENT = _required_float_config(
    "AG_SELL_REARM_GROWTH_PCT")
ORDER_MAX_AGE_SECONDS = _required_float_config("AG_ORDER_MAX_AGE_SEC")
RECOVERY_RESET_PERCENT = _required_float_config("AG_RECOVERY_RESET_PCT")
NEAR_TRIGGER_SECONDS = _required_float_config("AG_NEAR_TRIGGER_SEC")
ACTIVE_TRIGGER_SECONDS = _required_float_config("AG_ACTIVE_TRIGGER_SEC")
NEAR_TRIGGER_DISTANCE_PCT = _required_float_config(
    "AG_NEAR_TRIGGER_DISTANCE_PCT")
TREND_DEFER_MAX_SECONDS = _required_float_config("AG_TREND_DEFER_MAX_SEC")


def _parse_tiers(raw):
    tiers = []
    for item in str(raw).split(","):
        threshold, allocation = item.strip().split(":", 1)
        tiers.append((float(threshold), float(allocation)))
    tiers.sort()
    return tuple(tiers)


BUY_TIERS = _parse_tiers(BUY_TIERS_RAW)
SELL_TIERS = _parse_tiers(SELL_TIERS_RAW)
# Only the shallowest buy tier (the first, smallest drawdown) respects the historical
# price anti-chasing guard, so it will not chase a shallow dip that is still near a
# recent high (as happened on the 241 TAO buy). Deeper tiers keep bypassing it: a large
# drawdown is a genuine entry regardless of the recent-sell reference window.
GUARDED_BUY_THRESHOLD = min((t for t, _ in BUY_TIERS), default=None)
TRACKED_SYMBOLS = tuple(symbols_for("binance", "assetguardian"))
LEGACY_BUY_SYMBOL = sym.btcsymbol
ORDER_MISSING_CONFIRMATIONS = _required_int_config(
    "AG_ORDER_MISSING_CONFIRMATIONS")
STATE = AssetGuardianState()
_last_evaluation = {
    symbol: {"growth": None, "drawdown": None, "pending_tier": False}
    for symbol in TRACKED_SYMBOLS
}
_local_price_samples = {symbol: [] for symbol in TRACKED_SYMBOLS}


def _validate_config():
    if not math.isfinite(CHECK_INTERVAL_SECONDS) or CHECK_INTERVAL_SECONDS <= 0:
        raise ValueError("AG_CHECK_INTERVAL_SEC must be > 0")
    if (not math.isfinite(ASSET_REFERENCE_MINUTES_BACK_DEFAULT)
            or ASSET_REFERENCE_MINUTES_BACK_DEFAULT <= 0):
        raise ValueError("AG_REFERENCE_MINUTES_BACK must be > 0")
    if not math.isfinite(BUY_USE_CASH_RATIO) or not 0 < BUY_USE_CASH_RATIO <= 1:
        raise ValueError("AG_BUY_USE_CASH_RATIO must be in (0, 1]")
    if not BUY_TIERS or any(
            not math.isfinite(threshold) or threshold <= 0
            or not math.isfinite(allocation) or allocation <= 0
            for threshold, allocation in BUY_TIERS):
        raise ValueError("AG_BUY_TIERS must contain positive threshold:allocation pairs")
    if len({threshold for threshold, _ in BUY_TIERS}) != len(BUY_TIERS):
        raise ValueError("AG_BUY_TIERS contains duplicate thresholds")
    if sum(allocation for _, allocation in BUY_TIERS) > 1.0 + 1e-12:
        raise ValueError("suma ponderilor AG_BUY_TIERS depaseste 1")
    if not SELL_TIERS or any(
            not math.isfinite(threshold) or threshold <= 0
            or not math.isfinite(allocation) or allocation <= 0
            for threshold, allocation in SELL_TIERS):
        raise ValueError("AG_SELL_TIERS must contain positive threshold:allocation pairs")
    if len({threshold for threshold, _ in SELL_TIERS}) != len(SELL_TIERS):
        raise ValueError("AG_SELL_TIERS contains duplicate thresholds")
    if not math.isclose(
            sum(allocation for _, allocation in SELL_TIERS), 1.0,
            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the AG_SELL_TIERS weights must sum to exactly 1")
    if (not math.isfinite(SELL_REARM_GROWTH_PERCENT)
            or SELL_REARM_GROWTH_PERCENT <= 0
            or SELL_REARM_GROWTH_PERCENT >= SELL_TIERS[0][0]):
        raise ValueError(
            "AG_SELL_REARM_GROWTH_PCT must be > 0 and below the first SELL threshold")
    if not math.isfinite(ORDER_MAX_AGE_SECONDS) or ORDER_MAX_AGE_SECONDS <= 0:
        raise ValueError("AG_ORDER_MAX_AGE_SEC must be finite and > 0")
    if ORDER_MISSING_CONFIRMATIONS <= 0:
        raise ValueError("AG_ORDER_MISSING_CONFIRMATIONS must be an integer and > 0")
    # An empty explicit role selection disables evaluations without adding assets.
    for value, name in (
            (RECOVERY_RESET_PERCENT, "AG_RECOVERY_RESET_PCT"),
            (NEAR_TRIGGER_SECONDS, "AG_NEAR_TRIGGER_SEC"),
            (ACTIVE_TRIGGER_SECONDS, "AG_ACTIVE_TRIGGER_SEC"),
            (NEAR_TRIGGER_DISTANCE_PCT, "AG_NEAR_TRIGGER_DISTANCE_PCT"),
            (TREND_DEFER_MAX_SECONDS, "AG_TREND_DEFER_MAX_SEC")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")


_validate_config()


ORDER_LIFECYCLE = mkt.tracked_order_lifecycle(
    provider_name="binance",
    venue="Binance",
    missing_confirmations=ORDER_MISSING_CONFIRMATIONS,
    retry_on_lookup_error=True,
    max_age_seconds=ORDER_MAX_AGE_SECONDS,
    audit=ExecutionAudit(),
    clock=lambda: time.time(),
)


def _finite_float(raw, *, positive=False):
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


def _price_row(row):
    """Normalize a Price24 row to ``(timestamp_seconds, positive_price)``."""
    if isinstance(row, dict):
        raw_ts = row.get("timestamp")
        raw_price = row.get("price")
    elif isinstance(row, (list, tuple)) and len(row) >= 2:
        raw_ts, raw_price = row[0], row[1]
    else:
        return None
    timestamp = _finite_float(raw_ts, positive=True)
    price = _finite_float(raw_price, positive=True)
    if timestamp is None or price is None:
        return None
    if timestamp > 100_000_000_000:  # Price24 persists Binance timestamps in ms.
        timestamp /= 1000.0
    return timestamp, price


def _read_symbol_price_rows(symbol):
    """Copy one symbol's shared 24h price history under the manager lock."""
    try:
        managers = cm.get_cache_manager("Price24", symbols=list(TRACKED_SYMBOLS))
        manager = managers.get(symbol) if isinstance(managers, dict) else None
        if manager is None:
            print(f"ERROR Price24 manager missing for {symbol}")
            return []
        with manager.lock:
            rows = list(manager.cache.get(symbol, []))
        print(f"[DEBUG] {symbol} Price24 rows loaded: {len(rows)}")
        return rows
    except Exception as e:
        print(f"ERROR reading Price24 cache for {symbol}: {e}")
        return []


def _get_symbol_window_extrema(symbol, current_price,
                               minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    """Return validated per-symbol price minimum/maximum for the rolling window."""
    minutes_back = _finite_float(minutes_back, positive=True)
    current_price = _finite_float(current_price, positive=True)
    if minutes_back is None or current_price is None:
        print(f"[{symbol}] Invalid price/cache window; skip evaluation.")
        return None
    now_ts = float(time.time())
    target_ts = now_ts - minutes_back * 60.0

    local = _local_price_samples.setdefault(symbol, [])
    local.append([now_ts, current_price])
    local[:] = [row for row in local
                if (parsed := _price_row(row)) is not None
                and target_ts <= parsed[0] <= now_ts]

    normalized = []
    seen = set()
    for raw in _read_symbol_price_rows(symbol) + list(local):
        parsed = _price_row(raw)
        if parsed is None:
            continue
        timestamp, price = parsed
        if not target_ts <= timestamp <= now_ts:
            continue
        key = (timestamp, price)
        if key not in seen:
            seen.add(key)
            normalized.append({"timestamp": timestamp, "price": price})

    print(f"[DEBUG] {symbol} candidate price rows: {len(normalized)}")
    if len(normalized) < 2:
        print(f"[{symbol}] Insufficient per-asset price baseline; skip evaluation.")
        return None
    minimum = min(normalized, key=lambda row: row["price"])
    maximum = max(normalized, key=lambda row: row["price"])
    print(f"[DEBUG] {symbol} chosen MIN/MAX: {minimum}/{maximum}")
    return minimum, maximum


def sell_asset(symbol, qty, current_price, client_order_id, tier_threshold):
    """Submit one explicit SELL tier and return the native order response."""
    try:
        base_asset, _ = resolve_assets(symbol)
        provider = mkt.provider_by_name("binance")
        if provider is None:
            print(f"[{symbol}] Binance provider unavailable; skip SELL fail-closed.")
            return None
        free_qty = _finite_float(provider.free_balance(base_asset), positive=True)
        requested_qty = _finite_float(qty, positive=True)
        price = _finite_float(current_price, positive=True)
    except Exception as e:
        print(f"ERROR preparing SELL for {symbol}: {e}")
        return None
    if free_qty is None:
        print(f"[{symbol}] No valid free {base_asset} balance; skip SELL.")
        return None
    if requested_qty is None or price is None:
        print(f"[{symbol}] Invalid SELL price/quantity; skip SELL.")
        return None
    qty_to_sell = min(requested_qty, free_qty)
    if qty_to_sell <= 0:
        print(f"[{symbol}] SELL tier has no reconciled free quantity.")
        return None
    try:
        order = mkt.place(
            symbol, "SELL", price, qty_to_sell,
            force=False, smart=False, caller_owns_retry=True,
            bypass_quantity_policy=True,
            wait_for_trend=False,
            client_order_id=client_order_id,
            motivation=f"assetguardian_growth_exit_tier_{tier_threshold:g}")
        if order:
            print(
                f" SELL tier submitted: {symbol} requested={requested_qty:.8f} "
                f"free={free_qty:.8f} submitted_cap={qty_to_sell:.8f} "
                f"clientOrderId={client_order_id}")
            return order
        print(f" SELL tier not accepted: {symbol} qty={qty_to_sell}")
        return None
    except Exception as e:
        print(f"ERROR selling {symbol}: {e}")
        return None


def buy_with_all_cash(buy_symbol, cash_ratio=BUY_USE_CASH_RATIO,
                      cash_amount=None, client_order_id=None,
                      respect_guard=False):
    try:
        _, quote_asset = resolve_assets(buy_symbol)
        cash_ratio = _finite_float(cash_ratio, positive=True)
        if cash_ratio is None or cash_ratio > 1:
            print(f" Invalid cash ratio for buy: {cash_ratio}")
            return False
        free_cash = _finite_float(
            mkt.provider_by_name("binance").free_balance(quote_asset),
            positive=True,
        )
        current_price = _finite_float(api.get_current_price(buy_symbol), positive=True)
    except Exception as e:
        print(f"ERROR preparing buy for {buy_symbol}: {e}")
        return False
    print(
        f"[DEBUG] buy check symbol={buy_symbol}, quote={quote_asset}, "
        f"free_cash={free_cash}, current_price={current_price}"
    )

    if free_cash is None:
        print(f" No valid available {quote_asset} balance; skip buy fail-closed.")
        return False
    if current_price is None:
        print(f" Invalid current price for {buy_symbol}.")
        return False

    requested_cash = (free_cash * cash_ratio if cash_amount is None
                      else _finite_float(cash_amount, positive=True))
    if requested_cash is None:
        print(" Invalid requested cash amount for buy.")
        return False
    cash_to_use = min(requested_cash, free_cash * BUY_USE_CASH_RATIO)
    qty = cash_to_use / current_price
    if qty <= 0:
        print(" Computed qty <= 0. Skip buy.")
        return False

    print(
        f" BUY trigger active -> symbol={buy_symbol}, using {cash_to_use:.6f} {quote_asset} "
        f"(safe cap {cash_ratio*100:.2f}% of free cash), qty={qty:.8f}"
    )
    try:
        # respect_guard=True keeps the historical-price anti-chasing guard active
        # (bypass_profit_reference=False), so a shallow-dip tier will not buy far above
        # the recent sell reference. Deeper tiers pass respect_guard=False and bypass it.
        order = mkt.place(
            buy_symbol, "BUY", current_price, qty,
            force=False, smart=False, caller_owns_retry=True,
            bypass_profit_reference=not respect_guard,
            wait_for_trend=False,
            client_order_id=client_order_id,
            motivation="assetguardian_drawdown_buy")
        if order:
            print(f" BUY safe-order sent: {buy_symbol}, qty={qty:.8f}")
            return order
        print(f" BUY safe-order failed: {buy_symbol}, qty={qty:.8f}")
        return None
    except Exception as e:
        print(f"ERROR buying {buy_symbol}: {e}")
        return None


def _completed_tier_values(state):
    completed = set()
    for raw in state.get("completed_tiers", []):
        value = _finite_float(raw, positive=True)
        if value is not None:
            completed.add(value)
    return completed


def _load_state_root():
    """Load v3 BUY/SELL state and conservatively migrate older BUY-only data."""
    raw = STATE.load()
    if not isinstance(raw, dict):
        raw = {}
    symbol_rows = raw.get("symbols")
    if isinstance(symbol_rows, dict):
        normalized = {}
        for symbol, value in symbol_rows.items():
            if not isinstance(value, dict):
                continue
            if "buy" in value or "sell" in value:
                buy = value.get("buy")
                sell = value.get("sell")
                row = {}
                if isinstance(buy, dict) and buy:
                    row["buy"] = dict(buy)
                if isinstance(sell, dict) and sell:
                    row["sell"] = dict(sell)
            else:
                # Version 2 stored the BUY campaign directly under the symbol.
                row = {"buy": dict(value)} if value else {}
            if row:
                normalized[str(symbol)] = row
        return {
            "version": 3,
            "symbols": normalized,
        }
    legacy_keys = {"peak_value", "peak_ts", "initial_cash", "completed_tiers"}
    if legacy_keys.intersection(raw):
        return {
            "version": 3,
            "symbols": {LEGACY_BUY_SYMBOL: {"buy": dict(raw)}},
        }
    return {"version": 3, "symbols": {}}


def _symbol_campaign(symbol, kind):
    row = _load_state_root()["symbols"].get(symbol, {})
    campaign = row.get(kind, {}) if isinstance(row, dict) else {}
    return dict(campaign) if isinstance(campaign, dict) else {}


def _save_symbol_campaign(symbol, kind, campaign):
    if kind not in {"buy", "sell"}:
        raise ValueError(f"campanie AssetGuardian necunoscuta: {kind}")
    root = _load_state_root()
    symbols_state = dict(root["symbols"])
    row = dict(symbols_state.get(symbol, {}))
    if campaign:
        row[kind] = dict(campaign)
    else:
        row.pop(kind, None)
    if row:
        symbols_state[symbol] = row
    else:
        symbols_state.pop(symbol, None)
    STATE.save({"version": 3, "symbols": symbols_state})


def _trend_defer_ready(symbol, kind, side, threshold, state):
    """Non-blocking trend deferral, re-evaluated by the normal Guardian loop."""
    state = dict(state)
    now = float(time.time())
    defer = state.get("trend_defer")
    if not isinstance(defer, dict):
        defer = {}
    same_intent = (
        str(defer.get("side") or "").upper() == str(side).upper()
        and _finite_float(defer.get("threshold"), positive=True) == float(threshold)
    )
    started_at = (_finite_float(defer.get("started_at"), positive=True)
                  if same_intent else None)
    try:
        should_wait = bool(
            cm.get_short_trend_manager().should_wait(side, symbol))
    except Exception as exc:
        print(f"[{symbol}] Trend defer unavailable; continue submit: {exc}")
        should_wait = False

    if should_wait:
        if started_at is None:
            started_at = now
            state["trend_defer"] = {
                "side": str(side).upper(),
                "threshold": float(threshold),
                "started_at": started_at,
            }
            _save_symbol_campaign(symbol, kind, state)
        elapsed = max(0.0, now - started_at)
        if elapsed < TREND_DEFER_MAX_SECONDS:
            print(
                f"[{symbol}] {side} tier {float(threshold):g}% deferred by "
                f"short trend: {elapsed:.1f}/{TREND_DEFER_MAX_SECONDS:.1f}s; "
                "no blocking sleep.")
            return False, state
        print(
            f"[{symbol}] {side} trend defer reached "
            f"{TREND_DEFER_MAX_SECONDS:.1f}s; submit only if signal remains valid.")

    if "trend_defer" in state:
        state.pop("trend_defer", None)
        _save_symbol_campaign(symbol, kind, state)
    return True, state


def _clear_inactive_trend_defer(symbol, kind, state):
    """Reset the 180s clock when its tier signal is no longer crossed."""
    if not isinstance(state, dict) or "trend_defer" not in state:
        return state
    state = dict(state)
    state.pop("trend_defer", None)
    _save_symbol_campaign(symbol, kind, state)
    return state


def _campaign_tier(symbol, drawdown_abs, maximum_row, free_cash):
    """Select one crossed, incomplete BUY tier for one symbol campaign."""
    state = _symbol_campaign(symbol, "buy")
    if drawdown_abs < RECOVERY_RESET_PERCENT:
        if state:
            _save_symbol_campaign(symbol, "buy", {})
        return None, {}

    peak_price = _finite_float(maximum_row.get("price"), positive=True)
    peak_ts = _finite_float(maximum_row.get("timestamp"), positive=True)
    stored_peak = _finite_float(state.get("peak_price"), positive=True)
    if (peak_price is None or peak_ts is None
            or _finite_float(free_cash, positive=True) is None):
        return None, state
    if not state:
        state = {
            "peak_price": peak_price,
            "peak_ts": peak_ts,
            "initial_cash": free_cash,
            "completed_tiers": [],
            "spent_quote_by_tier": {},
            "total_spent_quote": 0.0,
            "attempts_by_tier": {},
            "terminal_orders": [],
        }
        _save_symbol_campaign(symbol, "buy", state)
    elif stored_peak is None:
        # A legacy global campaign has no meaningful per-asset peak. Preserve
        # its completed tiers so migration cannot repeat an already accepted
        # order, then attach the first valid per-asset peak conservatively.
        state = dict(state)
        state["peak_price"] = peak_price
        state["peak_ts"] = peak_ts
        if _finite_float(state.get("initial_cash"), positive=True) is None:
            state["initial_cash"] = free_cash
        state["completed_tiers"] = sorted(_completed_tier_values(state))
        _save_symbol_campaign(symbol, "buy", state)
    elif peak_price > stored_peak:
        # A higher peak within the same unrecovered campaign changes the
        # reference, but must never clear completed tiers or replenish budget.
        state = dict(state)
        state["peak_price"] = peak_price
        state["peak_ts"] = peak_ts
        _save_symbol_campaign(symbol, "buy", state)

    completed = _completed_tier_values(state)
    for threshold, allocation in BUY_TIERS:
        if drawdown_abs >= threshold and threshold not in completed:
            return (threshold, allocation), state
    return None, state


def _complete_campaign_tier(symbol, kind, state, threshold):
    completed = _completed_tier_values(state)
    completed.add(float(threshold))
    state = dict(state)
    state["completed_tiers"] = sorted(completed)
    _save_symbol_campaign(symbol, kind, state)


def _tier_key(threshold):
    return f"{float(threshold):g}"


def _buy_remaining_cash(state, threshold, allocation):
    initial_cash = _finite_float(state.get("initial_cash"), positive=True)
    if initial_cash is None:
        return None, None, None
    target_cash = initial_cash * BUY_USE_CASH_RATIO * float(allocation)
    prior_spent = _finite_float(
        dict(state.get("spent_quote_by_tier") or {}).get(
            _tier_key(threshold))) or 0.0
    return max(0.0, target_cash - prior_spent), target_cash, prior_spent


def _sell_client_order_id(symbol, state, threshold, attempt):
    raw = (
        f"assetguardian:sell:{symbol}:{state.get('trough_ts')}:"
        f"{float(threshold):g}:{int(attempt)}")
    return "AGS" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:29]


def _buy_client_order_id(symbol, state, threshold, attempt):
    raw = (
        f"assetguardian:buy:{symbol}:{state.get('peak_ts')}:"
        f"{float(threshold):g}:{int(attempt)}")
    return "AGB" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:29]


def _record_terminal_buy(symbol, state, status):
    """Atomically apply one terminal BUY result and acknowledge generic pending."""
    pending = dict(state.get("pending") or {})
    threshold = _finite_float(pending.get("threshold"), positive=True)
    if threshold is None:
        print(f"[{symbol}] Corrupt BUY terminal pending; preserve fail-closed.")
        return state

    key = _tier_key(threshold)
    spent_by_tier = dict(state.get("spent_quote_by_tier") or {})
    prior_spent = _finite_float(spent_by_tier.get(key)) or 0.0
    executed_quote = max(0.0, float(status.cost) + float(status.fee))
    spent_by_tier[key] = prior_spent + executed_quote

    state = dict(state)
    state["spent_quote_by_tier"] = spent_by_tier
    state["total_spent_quote"] = (
        (_finite_float(state.get("total_spent_quote")) or 0.0)
        + executed_quote)
    if status.status == "closed" and status.filled_qty > 0:
        completed = _completed_tier_values(state)
        completed.add(threshold)
        state["completed_tiers"] = sorted(completed)

    terminal = list(state.get("terminal_orders") or [])[-19:]
    terminal.append({
        "order_id": pending.get("order_id"),
        "client_order_id": pending.get("client_order_id"),
        "threshold": threshold,
        "status": status.status,
        "filled_qty": status.filled_qty,
        "cost": status.cost,
        "fee": status.fee,
        "terminal_ts": time.time(),
    })
    state["terminal_orders"] = terminal
    state.pop("pending", None)
    _save_symbol_campaign(symbol, "buy", state)
    print(
        f"[{symbol}] BUY tier terminal: threshold={threshold:g}% "
        f"status={status.status} orderId={pending.get('order_id')} "
        f"filled={status.filled_qty:.8f} cost={status.cost:.8f} "
        f"fee={status.fee:.8f}")
    return state


def _record_terminal_sell(symbol, state, status):
    """Apply one terminal venue status without inferring acceptance as a fill."""
    pending = dict(state.get("pending") or {})
    threshold = _finite_float(pending.get("threshold"), positive=True)
    if threshold is None:
        print(f"[{symbol}] Corrupt SELL pending tier; preserve fail-closed.")
        return state

    key = _tier_key(threshold)
    filled_by_tier = dict(state.get("filled_qty_by_tier") or {})
    prior_filled = _finite_float(filled_by_tier.get(key)) or 0.0
    filled_qty = _finite_float(status.filled_qty) or 0.0
    filled_by_tier[key] = prior_filled + max(0.0, filled_qty)

    state = dict(state)
    state["filled_qty_by_tier"] = filled_by_tier
    state["total_filled_qty"] = (
        (_finite_float(state.get("total_filled_qty")) or 0.0)
        + max(0.0, filled_qty))
    if status.status == "closed" and filled_qty > 0:
        completed = _completed_tier_values(state)
        completed.add(threshold)
        state["completed_tiers"] = sorted(completed)

    terminal = list(state.get("terminal_orders") or [])[-19:]
    terminal.append({
        "order_id": pending.get("order_id"),
        "client_order_id": pending.get("client_order_id"),
        "threshold": threshold,
        "status": status.status,
        "filled_qty": filled_qty,
        "cost": _finite_float(status.cost) or 0.0,
        "fee": _finite_float(status.fee) or 0.0,
        "terminal_ts": time.time(),
    })
    state["terminal_orders"] = terminal
    state.pop("pending", None)
    _save_symbol_campaign(symbol, "sell", state)
    print(
        f"[{symbol}] SELL tier terminal: threshold={threshold:g}% "
        f"status={status.status} orderId={pending.get('order_id')} "
        f"filled={filled_qty:.8f} cost={float(status.cost):.8f} "
        f"fee={float(status.fee):.8f}")
    return state


def _campaign_pending_persistor(symbol, kind, state):
    """Embed generic tracked pending state in one atomic campaign save."""
    if kind not in {"buy", "sell"}:
        raise ValueError(f"invalid tracked campaign kind: {kind}")
    box = {"state": dict(state)}

    def persist(pending):
        updated = dict(box["state"])
        if pending is None:
            updated.pop("pending", None)
        else:
            updated["pending"] = dict(pending)
        box["state"] = updated
        _save_symbol_campaign(symbol, kind, updated)

    return box, persist


def _reuse_unaccepted_attempt(symbol, kind, state, pending):
    """Release one unverified submit without consuming its deterministic ID.

    The next strategy evaluation increments the counter back to the same attempt,
    producing the same client_order_id.  This gives AssetGuardian at-least-once
    submit semantics while retaining venue-side idempotency where supported.
    """
    threshold = _finite_float(pending.get("threshold"), positive=True)
    try:
        attempt = int(pending.get("attempt"))
    except (TypeError, ValueError, OverflowError):
        attempt = 0
    if threshold is None or attempt <= 0:
        print(f"[{symbol}] Cannot reuse corrupt {kind.upper()} attempt; preserve counters.")
        return state
    key = _tier_key(threshold)
    attempts = dict(state.get("attempts_by_tier") or {})
    attempts[key] = max(0, attempt - 1)
    state = dict(state)
    state["attempts_by_tier"] = attempts
    _save_symbol_campaign(symbol, kind, state)
    return state


def _normalize_legacy_sell_pending(symbol, pending):
    """Add generic identity fields to pending rows created before the refactor."""
    normalized = dict(pending)
    normalized.setdefault("intent_id", normalized.get("client_order_id"))
    normalized.setdefault("symbol", symbol)
    normalized.setdefault("side", "SELL")
    normalized.setdefault("kind", "ASSET_GUARDIAN_SELL_TIER")
    normalized.setdefault("requested_price", None)
    return normalized


def _normalize_legacy_buy_pending(symbol, pending):
    normalized = dict(pending)
    normalized.setdefault("intent_id", normalized.get("client_order_id"))
    normalized.setdefault("symbol", symbol)
    normalized.setdefault("side", "BUY")
    normalized.setdefault("kind", "ASSET_GUARDIAN_BUY_TIER")
    normalized.setdefault("requested_price", None)
    return normalized


def _reconcile_pending_sell(symbol, state):
    """Adapt generic tracked-order truth to the SELL campaign accounting."""
    pending = state.get("pending")
    if not isinstance(pending, dict) or not pending:
        return state, "none"
    pending = _normalize_legacy_sell_pending(symbol, pending)
    box, persist = _campaign_pending_persistor(symbol, "sell", state)
    if pending != state.get("pending"):
        persist(pending)
    try:
        result = ORDER_LIFECYCLE.reconcile(pending, persist=persist)
    except Exception as exc:
        print(f"[{symbol}] Invalid/corrupt SELL pending; block fail-closed: {exc}")
        return box["state"], "active"
    state = box["state"]
    if result.outcome == "terminal":
        return _record_terminal_sell(symbol, state, result.status), "terminal"
    if result.outcome in {"absent", "retryable"}:
        state = _reuse_unaccepted_attempt(symbol, "sell", state, result.intent)
        truth = ("confirmed absent" if result.outcome == "absent"
                 else "lookup unavailable")
        print(
            f"[{symbol}] SELL intent {truth}; release for same-ID, revalidated retry "
            "without completing tier.")
        return state, "retryable"
    status_label = result.status.status if result.status is not None else "unknown"
    print(
        f"[{symbol}] SELL tier pending: status={status_label} "
        f"orderId={result.intent.get('order_id')} "
        f"filled={_finite_float(result.intent.get('filled_qty')) or 0.0:.8f}")
    return state, "active"


def _reconcile_pending_buy(symbol, state):
    """Adapt generic tracked-order truth to BUY campaign accounting."""
    pending = state.get("pending")
    if not isinstance(pending, dict) or not pending:
        return state, "none"
    pending = _normalize_legacy_buy_pending(symbol, pending)
    box, persist = _campaign_pending_persistor(symbol, "buy", state)
    if pending != state.get("pending"):
        persist(pending)
    try:
        result = ORDER_LIFECYCLE.reconcile(pending, persist=persist)
    except Exception as exc:
        print(f"[{symbol}] Invalid/corrupt BUY pending; block fail-closed: {exc}")
        return box["state"], "active"
    state = box["state"]
    if result.outcome == "terminal":
        return _record_terminal_buy(symbol, state, result.status), "terminal"
    if result.outcome in {"absent", "retryable"}:
        state = _reuse_unaccepted_attempt(symbol, "buy", state, result.intent)
        truth = ("confirmed absent" if result.outcome == "absent"
                 else "lookup unavailable")
        print(
            f"[{symbol}] BUY intent {truth}; release for same-ID, revalidated retry "
            "without completing tier.")
        return state, "retryable"
    status_label = result.status.status if result.status is not None else "unknown"
    print(
        f"[{symbol}] BUY tier pending: status={status_label} "
        f"orderId={result.intent.get('order_id')} "
        f"filled={_finite_float(result.intent.get('filled_qty')) or 0.0:.8f}")
    return state, "active"


def _active_sell_orders(provider, symbol):
    """Return active SELL orders, or ``None`` when venue truth is unavailable."""
    if provider is None:
        return None
    try:
        orders = provider.open_orders(symbol) or []
    except Exception as exc:
        print(f"[{symbol}] Open-order reconciliation unavailable: {exc}")
        return None
    return [
        order for order in orders
        if isinstance(order, dict)
        and str(order.get("side") or "").upper() == "SELL"
    ]


def _sell_campaign_tier(symbol, current_price, minimum_row, free_qty, provider):
    """Select one incomplete SELL tier against a frozen campaign trough."""
    state = _symbol_campaign(symbol, "sell")
    new_campaign = not bool(state)
    current_price = _finite_float(current_price, positive=True)
    free_qty = _finite_float(free_qty)
    if current_price is None or free_qty is None or free_qty < 0:
        return None, state, None
    if free_qty == 0:
        open_sells = _active_sell_orders(provider, symbol)
        if open_sells is None or open_sells:
            print(f"[{symbol}] Zero free qty with unknown/active SELL orders; preserve state.")
            return None, state, None
        if state:
            _save_symbol_campaign(symbol, "sell", {})
        return None, {}, None

    if state:
        trough_price = _finite_float(state.get("trough_price"), positive=True)
        trough_ts = _finite_float(state.get("trough_ts"), positive=True)
        initial_qty = _finite_float(state.get("initial_qty"), positive=True)
        if trough_price is None or trough_ts is None or initial_qty is None:
            print(f"[{symbol}] Invalid SELL campaign; clear without submitting.")
            _save_symbol_campaign(symbol, "sell", {})
            return None, {}, None
        growth = ((current_price - trough_price) / trough_price) * 100.0
        if growth <= SELL_REARM_GROWTH_PERCENT:
            print(
                f"[{symbol}] SELL campaign rearmed: growth={growth:.4f}% <= "
                f"{SELL_REARM_GROWTH_PERCENT:.4f}% against frozen trough "
                f"{trough_price:.8f}.")
            _save_symbol_campaign(symbol, "sell", {})
            return None, {}, growth
        expected_remaining = max(
            0.0,
            initial_qty - (_finite_float(state.get("total_filled_qty")) or 0.0),
        )
        balance_tolerance = max(1e-12, initial_qty * 1e-6)
        if free_qty > expected_remaining + balance_tolerance:
            print(
                f"[{symbol}] Free balance increased from expected "
                f"{expected_remaining:.8f} to {free_qty:.8f}; rearm SELL campaign "
                "for the reconciled larger position.")
            _save_symbol_campaign(symbol, "sell", {})
            return None, {}, growth
    else:
        trough_price = _finite_float(minimum_row.get("price"), positive=True)
        trough_ts = _finite_float(minimum_row.get("timestamp"), positive=True)
        if trough_price is None or trough_ts is None:
            return None, {}, None
        growth = ((current_price - trough_price) / trough_price) * 100.0
        if growth < SELL_TIERS[0][0]:
            return None, {}, growth
        state = {
            "trough_price": trough_price,
            "trough_ts": trough_ts,
            "initial_qty": free_qty,
            "created_at": time.time(),
            "completed_tiers": [],
            "filled_qty_by_tier": {},
            "total_filled_qty": 0.0,
            "attempts_by_tier": {},
            "terminal_orders": [],
        }
    open_sells = _active_sell_orders(provider, symbol)
    if open_sells is None:
        print(f"[{symbol}] Cannot verify open SELL orders; do not submit tier.")
        return None, ({} if new_campaign else state), growth
    if open_sells:
        print(
            f"[{symbol}] Existing SELL order blocks AssetGuardian tier: "
            f"{[order.get('orderId') for order in open_sells]}")
        return None, ({} if new_campaign else state), growth
    if new_campaign:
        _save_symbol_campaign(symbol, "sell", state)

    completed = _completed_tier_values(state)
    initial_qty = _finite_float(state.get("initial_qty"), positive=True)
    filled_by_tier = dict(state.get("filled_qty_by_tier") or {})
    for threshold, allocation in SELL_TIERS:
        if growth < threshold or threshold in completed:
            continue
        target_qty = initial_qty * allocation
        already_filled = _finite_float(
            filled_by_tier.get(_tier_key(threshold))) or 0.0
        remaining_qty = max(0.0, target_qty - already_filled)
        if remaining_qty <= 0:
            _complete_campaign_tier(symbol, "sell", state, threshold)
            return None, _symbol_campaign(symbol, "sell"), growth
        return (threshold, allocation, remaining_qty), state, growth
    return None, state, growth


def _submit_sell_tier(symbol, current_price, tier, state):
    threshold, allocation, qty = tier
    key = _tier_key(threshold)
    attempts = dict(state.get("attempts_by_tier") or {})
    attempt = int(attempts.get(key) or 0) + 1
    attempts[key] = attempt
    client_order_id = _sell_client_order_id(symbol, state, threshold, attempt)
    state = dict(state)
    state["attempts_by_tier"] = attempts
    intent = ORDER_LIFECYCLE.new_intent(
        intent_id=client_order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side="SELL",
        requested_qty=qty,
        requested_price=current_price,
        kind="ASSET_GUARDIAN_SELL_TIER",
        attempt=attempt,
        metadata={"threshold": threshold, "allocation": allocation},
    )
    box, persist = _campaign_pending_persistor(symbol, "sell", state)
    try:
        result = ORDER_LIFECYCLE.submit(
            intent,
            persist=persist,
            submit=lambda: sell_asset(
                symbol, qty=qty, current_price=current_price,
                client_order_id=client_order_id, tier_threshold=threshold),
        )
    except Exception as exc:
        print(f"[{symbol}] SELL tracked submit failed before/around submit: {exc}")
        return False
    # Known order ID means venue acceptance was observed.  An ambiguous response
    # remains pending and is recovered by client ID in a later evaluator cycle.
    return result.order_known


def _submit_buy_tier(symbol, current_price, cash_amount, tier, state):
    threshold, allocation = tier
    # Only the first/shallowest dip tier respects the anti-chasing guard.
    respect_guard = (GUARDED_BUY_THRESHOLD is not None
                     and threshold == GUARDED_BUY_THRESHOLD)
    key = _tier_key(threshold)
    attempts = dict(state.get("attempts_by_tier") or {})
    attempt = int(attempts.get(key) or 0) + 1
    attempts[key] = attempt
    client_order_id = _buy_client_order_id(symbol, state, threshold, attempt)
    qty = _finite_float(cash_amount, positive=True) / _finite_float(
        current_price, positive=True)
    state = dict(state)
    state["attempts_by_tier"] = attempts
    intent = ORDER_LIFECYCLE.new_intent(
        intent_id=client_order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side="BUY",
        requested_qty=qty,
        requested_price=current_price,
        kind="ASSET_GUARDIAN_BUY_TIER",
        attempt=attempt,
        metadata={
            "threshold": threshold,
            "allocation": allocation,
            "requested_cash": cash_amount,
        },
    )
    _box, persist = _campaign_pending_persistor(symbol, "buy", state)
    try:
        result = ORDER_LIFECYCLE.submit(
            intent,
            persist=persist,
            submit=lambda: buy_with_all_cash(
                buy_symbol=symbol,
                cash_amount=cash_amount,
                client_order_id=client_order_id,
                respect_guard=respect_guard,
            ),
        )
    except Exception as exc:
        print(f"[{symbol}] BUY tracked submit failed before/around submit: {exc}")
        return False
    return result.order_known


def _next_check_seconds():
    if any(state.get("pending_tier") for state in _last_evaluation.values()):
        return min(CHECK_INTERVAL_SECONDS, ACTIVE_TRIGGER_SECONDS)
    root = _load_state_root()
    if any(
            any(isinstance(row.get(kind, {}).get("pending"), dict)
                for kind in ("buy", "sell"))
            for row in root["symbols"].values() if isinstance(row, dict)):
        return min(CHECK_INTERVAL_SECONDS, ACTIVE_TRIGGER_SECONDS)
    for symbol, evaluation in _last_evaluation.items():
        drawdown = evaluation.get("drawdown")
        row = root["symbols"].get(symbol, {})
        buy_completed = _completed_tier_values(row.get("buy", {}))
        buy_next = next((threshold for threshold, _ in BUY_TIERS
                         if threshold not in buy_completed), None)
        if (drawdown is not None and buy_next is not None
                and buy_next - max(0.0, -drawdown) <= NEAR_TRIGGER_DISTANCE_PCT):
            return min(CHECK_INTERVAL_SECONDS, NEAR_TRIGGER_SECONDS)
        growth = evaluation.get("growth")
        sell_completed = _completed_tier_values(row.get("sell", {}))
        sell_next = next((threshold for threshold, _ in SELL_TIERS
                          if threshold not in sell_completed), None)
        if (growth is not None and sell_next is not None
                and sell_next - max(0.0, growth) <= NEAR_TRIGGER_DISTANCE_PCT):
            return min(CHECK_INTERVAL_SECONDS, NEAR_TRIGGER_SECONDS)
    return CHECK_INTERVAL_SECONDS


def evaluate_symbol(symbol, minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    """Evaluate and possibly submit exactly one asset-specific SELL or BUY."""
    evaluation = _last_evaluation.setdefault(
        symbol, {"growth": None, "drawdown": None, "pending_tier": False})
    evaluation.update(growth=None, drawdown=None, pending_tier=False)
    if symbol not in TRACKED_SYMBOLS:
        print(f"[{symbol}] Invalid AssetGuardian symbol; skip evaluation.")
        return False

    # Reconcile external truth before requiring fresh signal data. A stale/missing
    # price must not prevent terminalizing an already submitted order, and a BUY
    # recovery must never clear a pending intent before venue reconciliation.
    for kind, reconcile in (
            ("sell", _reconcile_pending_sell),
            ("buy", _reconcile_pending_buy)):
        campaign = _symbol_campaign(symbol, kind)
        if not campaign.get("pending"):
            continue
        _campaign, pending_outcome = reconcile(symbol, campaign)
        if pending_outcome == "active":
            evaluation["pending_tier"] = True
            return False
        if pending_outcome == "terminal":
            # One-cycle cooldown separates terminal/cancel accounting from a new
            # signal and prevents cancel-partial from immediate resubmission.
            return False

    current_price = _finite_float(api.get_current_price(symbol), positive=True)
    if current_price is None:
        print(f"[{symbol}] Current price unavailable; skip evaluation.")
        return False
    extrema = _get_symbol_window_extrema(
        symbol, current_price, minutes_back=minutes_back)

    if not extrema:
        print(f"[{symbol}] No per-asset baseline for the last {minutes_back}m.")
        return False

    minimum_row, maximum_row = extrema
    minimum_price = _finite_float(minimum_row.get("price"), positive=True)
    maximum_price = _finite_float(maximum_row.get("price"), positive=True)
    if minimum_price is None or maximum_price is None:
        print(f"[{symbol}] Invalid per-asset baseline.")
        return False

    rolling_growth_percent = ((current_price - minimum_price) / minimum_price) * 100.0
    drawdown_percent = ((current_price - maximum_price) / maximum_price) * 100.0
    evaluation["drawdown"] = drawdown_percent
    if drawdown_percent > -RECOVERY_RESET_PERCENT and _symbol_campaign(symbol, "buy"):
        print(f"[{symbol}] Drawdown recovered below {RECOVERY_RESET_PERCENT:.2f}%; "
              "rearm BUY campaign.")
        _save_symbol_campaign(symbol, "buy", {})

    provider = mkt.provider_by_name("binance")
    try:
        base_asset, _ = resolve_assets(symbol)
        free_base_qty = (provider.free_balance(base_asset)
                         if provider is not None else None)
    except Exception as exc:
        print(f"[{symbol}] SELL balance unavailable; skip SELL fail-closed: {exc}")
        free_base_qty = None
    sell_tier, sell_state, effective_growth = _sell_campaign_tier(
        symbol, current_price, minimum_row, free_base_qty, provider)
    if sell_tier is None:
        sell_state = _clear_inactive_trend_defer(
            symbol, "sell", sell_state)
    evaluation["growth"] = (
        rolling_growth_percent if effective_growth is None else effective_growth)
    frozen_trough = _finite_float(sell_state.get("trough_price"), positive=True)
    print(
        f"[{symbol}] price={current_price:.8f}, window_min/max="
        f"{minimum_price:.8f}/{maximum_price:.8f}, minutes_back={minutes_back:.1f}, "
        f"rolling_growth_from_min={rolling_growth_percent:.4f}%, "
        f"sell_growth={evaluation['growth']:.4f}%, "
        f"frozen_sell_trough={frozen_trough}, "
        f"drawdown_from_max={drawdown_percent:.4f}%"
    )

    if sell_tier is not None:
        threshold, allocation, qty = sell_tier
        print(
            f"[{symbol}] SELL tier +{threshold:.2f}% allocation={allocation:.3f} "
            f"target_remaining={qty:.8f} base."
        )
        evaluation["pending_tier"] = True
        ready, sell_state = _trend_defer_ready(
            symbol, "sell", "SELL", threshold, sell_state)
        if not ready:
            return False
        accepted = _submit_sell_tier(
            symbol, current_price=current_price, tier=sell_tier, state=sell_state)
        if accepted:
            evaluation["pending_tier"] = bool(
                _symbol_campaign(symbol, "sell").get("pending"))
            return True
        return False

    if drawdown_percent > -BUY_TIERS[0][0]:
        _clear_inactive_trend_defer(
            symbol, "buy", _symbol_campaign(symbol, "buy"))
    else:
        print(
            f"[{symbol}] Drawdown threshold reached ({drawdown_percent:.4f}% "
            f"<= -{BUY_TIERS[0][0]:.4f}%). Checking staged BUY..."
        )
        provider = mkt.provider_by_name("binance")
        free_cash = (_finite_float(provider.free_balance("USDC"), positive=True)
                     if provider else None)
        if free_cash is None:
            print(f"[{symbol}] No valid USDC balance for staged BUY.")
            return False
        tier, campaign = _campaign_tier(
            symbol, abs(drawdown_percent), maximum_row, free_cash)
        if tier is None:
            print(f"[{symbol}] No uncompleted BUY tier at current drawdown.")
            return False
        threshold, allocation = tier
        evaluation["pending_tier"] = True
        initial_cash = _finite_float(campaign.get("initial_cash"), positive=True)
        if initial_cash is None:
            print(f"[{symbol}] Invalid campaign initial cash; reset fail-closed.")
            _save_symbol_campaign(symbol, "buy", {})
            return False
        cash_amount, target_cash, prior_spent = _buy_remaining_cash(
            campaign, threshold, allocation)
        if cash_amount is None:
            print(f"[{symbol}] Invalid BUY tier budget; preserve fail-closed.")
            return False
        if cash_amount <= 0:
            _complete_campaign_tier(symbol, "buy", campaign, threshold)
            print(
                f"[{symbol}] BUY tier -{threshold:.2f}% already funded by "
                f"terminal partials ({prior_spent:.6f} quote); mark complete.")
            return False
        print(f"[{symbol}] BUY tier -{threshold:.2f}% allocation={allocation:.3f} "
              f"cash_remaining={cash_amount:.6f}/{target_cash:.6f} USDC")
        ready, campaign = _trend_defer_ready(
            symbol, "buy", "BUY", threshold, campaign)
        if not ready:
            return False
        if _submit_buy_tier(
                symbol, current_price, cash_amount, tier, campaign):
            evaluation["pending_tier"] = True
            return True

    return False


def evaluate_and_maybe_sell_or_buy(
    minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT,
    symbols=None,
):
    """Evaluate configured assets independently, accepting at most one order/cycle."""
    selected = tuple(symbols) if symbols is not None else TRACKED_SYMBOLS
    for symbol in selected:
        if evaluate_symbol(symbol, minutes_back=minutes_back):
            return True
    return False


def run_forever():
    print(
        f" Started. check_interval={CHECK_INTERVAL_SECONDS}s, "
        f"sell_tiers={SELL_TIERS}, sell_rearm={SELL_REARM_GROWTH_PERCENT}%, "
        f"order_max_age={ORDER_MAX_AGE_SECONDS}s, "
        f"trend_defer_max={TREND_DEFER_MAX_SECONDS}s, "
        f"buy_tiers={BUY_TIERS}, "
        f"minutes_back={ASSET_REFERENCE_MINUTES_BACK_DEFAULT}m, "
        f"symbols={TRACKED_SYMBOLS}"
    )
    while True:
        try:
            evaluate_and_maybe_sell_or_buy(minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT)
        except Exception as e:
            print(f" Runtime ERROR: {e}")
        sleep_seconds = _next_check_seconds()
        # Flushing at the cycle boundary publishes all preceding lines when stdout
        # is redirected by flota_start.sh, preventing a quiet bot from appearing stale.
        print(f"[DEBUG] sleep {sleep_seconds}s before next cycle", flush=True)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_forever()
"""Minimal atomic state store for Assetguardian tranches."""
import os
from state_io import atomic_write_json, load_json_state

from lock import FileLock


class AssetGuardianState:
    def __init__(self, path=None, *, fail_closed=True):
        root = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(root, "cachedb", "assetguardian_state.json")
        self.lock_path = self.path + ".lock"
        self.fail_closed = fail_closed

    def load(self):
        with FileLock(self.lock_path):
            return load_json_state(
                self.path, default_factory=dict, fail_closed=self.fail_closed,
                label="AssetGuardian",
            )

    def save(self, value):
        with FileLock(self.lock_path):
            directory = os.path.dirname(self.path)
            os.makedirs(directory, exist_ok=True)
            atomic_write_json(
                self.path, value, sort_keys=True, separators=(",", ":"),
            )
