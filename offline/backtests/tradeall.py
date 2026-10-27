#!/usr/bin/env python3
"""
offline.backtests.tradeall — replays the price history already saved
(cachedb/cache_price_{symbol}.jsonl, ~11 months) through EXACTLY the same model of
decizie ca tradeall.py (PriceWindow / WindowAnalyzer / TrendState / logic /
logic_small), with place_order_smart replaced by a stub that simulates
execution instead of hitting the Binance network. It writes ONLY into a separate
logger/backtest/<run_id>/ — NEVER in the live logs used by
tradeall.py real (vezi plan A5).

Rulare:
    python3 -m offline.backtests.tradeall --symbol BTCUSDC --start 2026-06-01 --speed fast
    python3 -m offline.backtests.tradeall --symbol BTCUSDC --start 2026-06-01 --end 2026-06-08 --speed real

Visualisation (while running or afterwards): in another terminal,
    python -m visuals.tradeall_observe --backtest-dir logger/backtest/<run_id> --symbols BTCUSDC
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import tradeall as ta  # We reuse PriceWindow/WindowAnalyzer/TrendState/logic (A5) — the model is not reimplemented.
# 23 Jul: SimClock extracted into providers/replay_clock.py (shared with
# monitortrades — see UNIFIED_BACKTEST_PLAN.md §7/§8), passed to
# TrendState(now_fn=...) so that fast-forward is correct (A5). The
# `_SimClock` alias is kept for any existing external reference (offline/research/*.py).
from providers.replay_clock import SimClock as _SimClock  # noqa: F401
from offline.research.monitortrades_backtest.replay_trend_source import (  # noqa: E402
    DynamicReplayTrendSource,
)


def _sanitize(value):
    return str(value).replace("|", "/").replace("\n", " ") if value is not None else ""


FEE_PCT = 0.1   # comision spot Binance ~0.1% per leg (taker)
# marime standard de pozitie pt simulare (kalman-primary + benchmark buy&hold) —
# 21 Jul: replaces ta.api.quantities[symbol] (removed from bapi.py, it was only
# a placeholder always trimmed by the weight limit in live, but here we really do need
# a REAL, fixed NUMBER, so that variants can be compared honestly on the same volume.
BACKTEST_NOTIONAL_USD = 1000.0


class BacktestBroker:
    """Stub for po.place_order_smart: it simulates execution (no network),
    writes into its own backtest folder, the same pipe format as the live
    order_outcomes (Step A2) — so that tradeall_observe.py can render it identically.
    It also keeps P&L ACCOUNTING: position, average cost, realised, fees."""
    def __init__(self, out_dir, clock, trend_gate=None, trend_gate_timeout_sec=None,
                 trend_gate_max_wait_sec=3600.0):
        self.clock = clock
        self.path = os.path.join(out_dir, "order_outcomes.log")
        self.n_buy = self.n_sell = 0
        self.pos_qty = 0.0
        self.pos_cost = 0.0       # Total cost of the current position (excluding fees).
        self.realized = 0.0       # realised profit/loss (excluding fees)
        self.fees = 0.0
        self.last_price = None
        # 29 Jul (user's idea): a "wait for a better price" gate. FAITHFUL to live
        # (bapi_placeorder.__place_order -> _maybe_wait_trend -> wait_for_favorable_entry):
        # the gate does NOT block the order, it only DELAYS it until a favourable tick
        # (or until max_wait, then places it anyway). The P&L effect is a better
        # a better entry/exit, NOT avoided trades. Deferred orders sit in
        # self.pending and execute on future ticks (process_pending, called
        # from the main loop). trend_gate=None => OFF (the old behaviour).
        self.trend_gate = trend_gate
        self.trend_gate_timeout_sec = trend_gate_timeout_sec
        _max_wait = 3600.0
        if trend_gate_max_wait_sec is not None and not isinstance(trend_gate_max_wait_sec, bool):
            try:
                _mw = float(trend_gate_max_wait_sec)
                if math.isfinite(_mw) and _mw >= 0.0:
                    _max_wait = _mw
            except (TypeError, ValueError):
                pass
        self.trend_gate_max_wait_sec = _max_wait
        self.pending = []          # [{side, symbol, qty_arg, motivation, deadline}]
        self.n_waited = 0          # How many orders were delayed (and then executed).
        self._log_f = None

    def write_outcome_row(self, cols):
        line = "|".join(_sanitize(c) for c in cols) + "\n"
        if self._log_f is not None and not self._log_f.closed:
            self._log_f.write(line)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    def _execute(self, order_type, symbol, price, qty=None, motivation=None):
        """The real P&L accounting (no gate) — used directly or after waiting."""
        order_type = str(order_type or "").strip().upper()
        if order_type not in ("BUY", "SELL"):
            return None
        if isinstance(price, bool):
            return None
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 0.0:
            return None
        if qty is None:
            qty = BACKTEST_NOTIONAL_USD / price
        if isinstance(qty, bool):
            return None
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(qty) or qty <= 0.0:
            return None
        self.last_price = price
        if order_type == "BUY":
            self.n_buy += 1
            self.pos_qty += qty
            self.pos_cost += qty * price
            self.fees += qty * price * FEE_PCT / 100
        else:
            if self.pos_qty <= 1e-12:
                return None            # Nothing to sell (spot) — refuse, as in reality.
            sell_q = min(qty, self.pos_qty)
            avg = self.pos_cost / self.pos_qty
            self.realized += (price - avg) * sell_q
            self.fees += sell_q * price * FEE_PCT / 100
            self.pos_cost -= avg * sell_q
            self.pos_qty -= sell_q
            if self.pos_qty <= 1e-9:
                self.pos_qty = 0.0
                self.pos_cost = 0.0
            self.n_sell += 1
        cols = [self.clock(), symbol, order_type, price, qty, "executed", "", "backtest", motivation]
        self.write_outcome_row(cols)
        return {"orderId": -1, "backtest": True}   # A truthy object, as in logic()'s "if order:".

    def place_order_smart(self, order_type, symbol, price, qty=None, motivation=None, **kwargs):
        order_type = str(order_type or "").strip().upper()
        if order_type not in ("BUY", "SELL"):
            return None
        if order_type != "BUY" and self.pos_qty <= 1e-12:
            return None
        # The gate applies UNCONDITIONALLY (force=True included) — user (29 Jul, "the
        # is always welcome"), consistent with live. If the direction is NOT favourable yet,
        # It DELAYS (it does not block): the order is recorded as pending and executed on the
        # first favourable tick (or when max_wait expires). As in live, the caller
        # (tradeall) gets a truthy result immediately -> it marks the fire confirmed,
        # exactly as after a wait_for_favorable_entry that ends in a placement.
        if (self.trend_gate is not None
                and self.trend_gate.should_wait(symbol, order_type, self.trend_gate_timeout_sec)):
            self.pending.append({
                "side": order_type, "symbol": symbol, "qty_arg": qty,
                "motivation": motivation, "deadline": self.clock() + self.trend_gate_max_wait_sec})
            self.n_waited += 1
            return {"orderId": -1, "backtest": True, "pending": True}   # Truthy, as in live after the wait.
        return self._execute(order_type, symbol, price, qty, motivation)

    def process_pending(self, symbol, ts, price):
        """Execute the delayed orders that became favourable OR expired
        (max_wait), at the current tick's price. Called on EVERY tick of the loop."""
        if not self.pending:
            return
        still = []
        for o in self.pending:
            if o["symbol"] != symbol:
                still.append(o)
                continue
            favorable = (
                not self.trend_gate.should_wait(symbol, o["side"], self.trend_gate_timeout_sec)
                if self.trend_gate is not None else True
            )
            expired = ts >= o["deadline"]
            if favorable or expired:
                self._execute(o["side"], symbol, price, o["qty_arg"], o["motivation"])
            else:
                still.append(o)
        self.pending = still

    def sell_all(self, symbol, price, motivation):
        if self.pos_qty <= 1e-12:
            return None
        return self.place_order_smart("SELL", symbol, price, self.pos_qty, motivation=motivation)

    def pnl_summary(self):
        m2m = 0.0
        if self.pos_qty > 1e-12 and self.last_price:
            m2m = (self.last_price - self.pos_cost / self.pos_qty) * self.pos_qty
        return {"buys": self.n_buy, "sells": self.n_sell,
                "realized": round(self.realized, 2), "fees": round(self.fees, 2),
                "open_qty": round(self.pos_qty, 6), "mark_to_market": round(m2m, 2),
                "net_total": round(self.realized + m2m - self.fees, 2),
                "n_waited": self.n_waited}


def make_decision_logger(out_dir, clock, file_handle=None):
    path = os.path.join(out_dir, "tradeall_decisions.log")

    def _log_decision(symbol, event, **fields):
        try:
            cols = [clock(), symbol, event, fields.get("state", ""), fields.get("old_state", ""),
                    fields.get("price", ""), fields.get("prev_confirm_count", "")]
            line = "|".join(_sanitize(c) for c in cols) + "\n"
            if file_handle is not None and not file_handle.closed:
                file_handle.write(line)
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as e:
            print(f"[tradeall_backtest] log_decision error: {e}")
    return _log_decision


_TICK_SLICE_CACHE = {}
_FILE_TICKS_CACHE = {}


def _iter_jsonl_ticks(path, symbol, start_ts, end_ts):
    try:
        st = os.stat(path)
        with open(path, "r", encoding="utf-8") as f:
            head_line = f.readline()
        sig = (head_line, st.st_mtime_ns if st.st_size < 65536 else 0)
    except OSError:
        sig = ("", 0)
    file_key = (path, symbol, sig)
    cache_key = (file_key, start_ts, end_ts)
    cached = _TICK_SLICE_CACHE.get(cache_key)
    if cached is not None:
        yield from cached
        return

    full_ticks = _FILE_TICKS_CACHE.get(file_key)
    if full_ticks is None:
        full_ticks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict) or rec.get("s") != symbol:
                    continue
                item = rec.get("i")
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                if isinstance(item[0], bool) or isinstance(item[1], bool):
                    continue
                try:
                    ts_ms, price = float(item[0]), float(item[1])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(ts_ms) and math.isfinite(price) and price > 0.0):
                    continue
                full_ticks.append((ts_ms / 1000.0, price))
        if len(_FILE_TICKS_CACHE) >= 4:
            _FILE_TICKS_CACHE.clear()
        _FILE_TICKS_CACHE[file_key] = full_ticks

    ticks = [
        (ts, price)
        for ts, price in full_ticks
        if ts >= start_ts and (end_ts is None or ts <= end_ts)
    ]
    if len(_TICK_SLICE_CACHE) >= 8:
        _TICK_SLICE_CACHE.clear()
    _TICK_SLICE_CACHE[cache_key] = ticks
    yield from ticks


def load_ticks_history(symbol, start_ts, end_ts):
    """Reads cache_sparse_price_{symbol}.jsonl — a long history, SPARSE."""
    path = os.path.join(ROOT, "cachedb", f"cache_sparse_price_{symbol}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path} (the price history for {symbol})")
    yield from _iter_jsonl_ticks(path, symbol, start_ts, end_ts)


def load_ticks_cache24(symbol, start_ts, end_ts, filename=None):
    """Read cache_24_price_{symbol}.json (or cache_365_price_{symbol}.jsonl
    if a filename is given) — a DENSE resolution (~1s/tick, as in live)."""
    if filename:
        path = filename
    else:
        path = os.path.join(ROOT, "cachedb", f"cache_24_price_{symbol}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path} (cache24 for {symbol})")
    if str(path).endswith(".jsonl"):
        yield from _iter_jsonl_ticks(path, symbol, start_ts, end_ts)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("items", {}).get(symbol, []) if isinstance(data, dict) else []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        if isinstance(entry[0], bool) or isinstance(entry[1], bool):
            continue
        try:
            ts_ms, price = float(entry[0]), float(entry[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(ts_ms) and math.isfinite(price) and price > 0.0):
            continue
        ts = ts_ms / 1000.0
        if ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        yield ts, price


def run_backtest(symbol, start_ts, end_ts, speed, run_id, source, cache24_file=None, quiet=False,
                 kalman_primary=False, threshold_provider=None, trend_gate_timeout_sec=None,
                 trend_gate_max_wait_sec=3600.0):
    """threshold_provider is OPTIONAL (default None = the OLD behaviour, unchanged:
    uses the fixed ta.PRICE_CHANGE_THRESHOLD_EUR/_BIG_EUR). If given, it is a callable
    threshold_provider(window_small, window_big) -> (thr_small, thr_big), called on
    EVERY tick BEFORE check_price_change() — that way an experiment can test
    DYNAMIC thresholds (e.g. adaptive on volatility) without copying the whole loop
    (23 Jul: offline/research/tradeall_adaptive_thresholds/ had done exactly that before, with
    the risk that its own loop would silently diverge from this, the "official" one, over time)."""
    safe_run_id = str(run_id or "").strip().replace("/", "_").replace("\\", "_")
    if not safe_run_id or safe_run_id in (".", ".."):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    out_dir = os.path.join(ROOT, "logger", "backtest", safe_run_id)
    # 23 Jul: clean the folder BEFORE running — several files here
    # (order_outcomes.log, tradeall_decisions.log, tradeall_shadow.log,
    # tradeall_price_samples.log) are written with open(..., "a") (append), so without
    # this, a SECOND run with the SAME run_id would silently mix the old history with
    # the new one. Until now, every caller had to remember to do the
    # shutil.rmtree() itself (offline/research/tradeall_trigger_gate/experiment_quality_
    # signal_v2.py did it by hand) — moved here once, for every caller.
    import shutil as _shutil
    _shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    price_path = os.path.join(out_dir, "tradeall_price_samples.log")

    if quiet:
        # tradeall.logic()/check_price_change() print() on every tick, mirrored onto
        # disk (log.py) — on DENSE data (cache24, tens of thousands of ticks) that dominates the
        # run time. log.disable_print() suppresses print() globally (OUR messages use
        # sys.stderr.write, which stays visible).
        try:
            import log
            log.disable_print()
        except Exception:
            pass

    # 29 Jul (user's idea): a "wait for a favourable moment" gate for the
    # BUY/SELL retry intents — an isolated DynamicReplayTrendSource (it does not touch
    # cache_short_trend.json), fed tick by tick below, in the main loop.
    trend_gate = DynamicReplayTrendSource([symbol]) if trend_gate_timeout_sec is not None else None

    clock = _SimClock()
    broker = BacktestBroker(out_dir, clock, trend_gate=trend_gate,
                            trend_gate_timeout_sec=trend_gate_timeout_sec,
                            trend_gate_max_wait_sec=trend_gate_max_wait_sec)
    # 30 Jul: tradeall._fire_order now uses mkt.place(symbol, side, price, qty, ...)
    # (a single proxy), not po.place_order_smart(side, symbol, ...). We stub ta.mkt.place with a
    # an adapter that swaps the order (symbol,side)->(side,symbol) and calls the simulated broker
    # -> it does NOT touch the network. (We also patch ta.po.place_order_smart as a safety net for
    # any residual path.)
    orig_place_smart = getattr(getattr(ta, "po", None), "place_order_smart", None)
    orig_mkt_place = getattr(ta.mkt, "place", None)
    orig_mkt_cancel = getattr(ta.mkt, "cancel_order", None)
    orig_api_cancel_expired = getattr(getattr(ta, "api", None), "cancel_expired_orders", None)
    orig_log_decision = getattr(ta, "log_decision", None)
    orig_kalman_ref = getattr(ta, "_kalman_ref", None)
    orig_gate_log = getattr(ta, "GATE_OUTCOME_LOG", None)
    _bt_env_keys = ("BACKTEST_MODE", "REPLAY_MODE", "DISABLE_EXTERNAL_NOTIFICATIONS", "DISABLE_GEMINI_LLM")
    _orig_bt_env = {k: os.environ.get(k) for k in _bt_env_keys}
    for _k in _bt_env_keys:
        os.environ[_k] = "1"

    try:
        ta.mkt.cancel_order = lambda *a, **kw: True
        if hasattr(ta, "api"):
            ta.api.cancel_expired_orders = lambda *a, **kw: None
        if kalman_primary:
            # PRIMARY KALMAN MODE: the old model only LOGS (its orders are not
            # executes); the broker is driven exclusively by the Kalman transitions.
            _old_attempts = {"n": 0}
            def _journal_only(order_type, symbol_, price_, qty_=None, motivation=None, **kw):
                _old_attempts["n"] += 1
                return None
            if hasattr(ta, "po"):
                ta.po.place_order_smart = _journal_only
            ta.mkt.place = lambda symbol_, side_, price_, qty_=None, **kw: _journal_only(
                side_, symbol_, price_, qty_, **kw)
        else:
            if hasattr(ta, "po"):
                ta.po.place_order_smart = broker.place_order_smart           # A stub — it does NOT touch the network.
            ta.mkt.place = lambda symbol_, side_, price_, qty_=None, **kw: broker.place_order_smart(
                side_, symbol_, price_, qty_, **kw)
        decision_path = os.path.join(out_dir, "tradeall_decisions.log")

        window_small = ta.PriceWindow(symbol, 300, sample_rate_sec=ta.TIME_SLEEP_GET_PRICE,
                                       window_seconds=ta.WINDOW_SECONDS_SMALL)
        window_big = ta.PriceWindow(symbol, 300, sample_rate_sec=ta.TIME_SLEEP_GET_PRICE,
                                     window_seconds=ta.WINDOW_SECONDS_BIG)
        analyzer_small = ta.WindowAnalyzer(window_small)
        analyzer_big = ta.WindowAnalyzer(window_big)
        trend_state = ta.TrendState(max_duration_seconds=2.5 * 60 * 60, expiration_trend_time=2.7 * 60,
                                     fresh_trend_time=3.7 * 60, now_fn=clock)
        trend_state_big = ta.TrendState(max_duration_seconds=3 * 60 * 60, expiration_trend_time=2.7 * 60,
                                         fresh_trend_time=3.7 * 60, now_fn=clock)
    
        # SHADOW (observational, the 17 Jul plan): the same objects as live, with the simulated
        # clock; a FLAT journal in the run's folder (the backtest monitor draws it).
        import shadow_signals
        shadow = shadow_signals.KalmanStateTracker(
            journal=shadow_signals.KalmanJournal(fixed_path=os.path.join(out_dir, "tradeall_shadow.log")),
            now_fn=clock,
            compute_vol=(speed != "fast"))
        # The KALMAN GATE is in the backtest too (parity with live), but with the block journal
        # redirected into the run's folder — NEVER into the live order_outcomes (A5).
        ta._kalman_ref = shadow
        def _bt_gate_log(symbol_, side, price_, qty, outcome, reason, motivation):
            cols = [clock(), symbol_, side, price_, qty, outcome, reason, "backtest", motivation]
            broker.write_outcome_row(cols)
        ta.GATE_OUTCOME_LOG = _bt_gate_log
    
        if source == "cache24":
            tick_source = load_ticks_cache24(symbol, start_ts, end_ts, filename=cache24_file)
        else:
            tick_source = load_ticks_history(symbol, start_ts, end_ts)
    
        prev_ts = None
        n = 0
        first_price = None
        prev_ktrend = 0
        state_interval = 5000 if speed == "fast" else 100
        last_state = None
        slope = slope_big = gradient = _gr = 0.0
        fast_kalman_only = bool(kalman_primary and speed == "fast" and quiet)
        with open(price_path, "a", encoding="utf-8") as price_f, \
             open(broker.path, "a", encoding="utf-8") as outcome_f, \
             open(decision_path, "a", encoding="utf-8") as decision_f:
            broker._log_f = outcome_f
            ta.log_decision = make_decision_logger(out_dir, clock, file_handle=decision_f)
            for ts, price in tick_source:
                clock.ts = ts   # The simulated clock = the time of the REPLAYED tick, not the real clock (A5).
                if speed == "real" and prev_ts is not None:
                    time.sleep(max(0.0, ts - prev_ts))
                dt = ts - prev_ts if prev_ts is not None else None
                prev_ts = ts
    
                if dt and dt > 0:
                    window_small.set_sample_rate(dt)
                    window_big.set_sample_rate(dt)
                window_small.process_price(price)
                window_big.process_price(price)
                if trend_gate is not None:
                    trend_gate.advance(symbol, ts, price)
                    broker.process_pending(symbol, ts, price)   # Execute delayed orders that became favourable.
                price_f.write(f"{ts}|{symbol}|{price}\n")
    
                if not fast_kalman_only:
                    if threshold_provider is not None:
                        thr_small, thr_big = threshold_provider(window_small, window_big)
                    else:
                        thr_small, thr_big = ta.PRICE_CHANGE_THRESHOLD_EUR, ta.PRICE_CHANGE_THRESHOLD_BIG_EUR
    
                    slope, _pos = analyzer_small.check_price_change(thr_small)
                    gradient, _gc, _sf, _gr = window_small.get_short_trend()
                    ta.logic_small("SMALL", True, symbol, gradient, slope, trend_state, price,
                                   price_window=window_small)
    
                    slope_big, _price_diff = analyzer_big.check_price_change(thr_big)
                    ta.logic("BIG", True, symbol, gradient, slope_big, trend_state_big, price)
    
                # SHADOW: the same call as in the live TrendCoordinator.evaluate, with a simulated clock.
                need_eps = (
                    ts - shadow._last_fed.get(symbol, -1e18) >= shadow_signals.KALMAN_SAMPLE_SEC
                    and len(shadow._fed_prices.get(symbol, ())) < 4
                )
                shadow_fields = shadow.update(
                    symbol,
                    ts,
                    price,
                    epsilon=window_small.get_noise_epsilon() if need_eps else 0.0,
                    big_prices=window_big.prices,
                    big_sample_rate=window_big.sample_rate_sec,
                )
    
                if first_price is None:
                    first_price = price
                # 23 Jul: broker.last_price is updated UNCONDITIONALLY, on every tick — not
                # only in kalman_primary mode. Bug found today: without it, mark_to_market()
                # uses the price of the LAST TRADE (not the current market price) for any
                # position left OPEN at the end in normal mode (model_actual) — a BUY
                # at 100 followed by a rise to 150 without another trade reported mark_to_market=$0
                # instead of $50 (verified in isolation: net_total -$0.10 versus the correct +$49.90).
                broker.last_price = price
                if kalman_primary:
                    ktrend = shadow_fields.get("kalman_trend", prev_ktrend)
                    if ktrend != prev_ktrend:
                        if ktrend == 1:
                            broker.place_order_smart("BUY", symbol, price, BACKTEST_NOTIONAL_USD / price,
                                                      motivation="kalman_up")
                        elif ktrend == -1:
                            broker.sell_all(symbol, price, motivation="kalman_down")
                        prev_ktrend = ktrend
    
                n += 1
                if speed != "fast" and n % 100 == 0:
                    # Simulated analysis state — read by tradeall_observe.py (chart hover),
                    # the same content as live's cache_short_trend.json.
                    try:
                        last_state = {symbol: {
                            "current_price": price, "final_trend": gradient,
                            "gradient_recent": _gr, "slope_small": slope, "slope_big": slope_big,
                            "epsilon": window_small.get_noise_epsilon(), "ts": ts,
                            **shadow_fields,
                        }}
                        with open(os.path.join(out_dir, "analysis_state.json"), "w", encoding="utf-8") as sf:
                            json.dump(last_state, sf)
                    except Exception:
                        pass
                if n % 5000 == 0:
                    # sys.stderr.write, not print(): in --quiet mode, print() is suppressed globally
                    # (log.disable_print()) — OUR progress messages still have to be visible.
                    sys.stderr.write(f"[tradeall_backtest] {n} ticks, the last {datetime.fromtimestamp(ts)} "
                                     f"(BUY {broker.n_buy} / SELL {broker.n_sell})\n")
            broker._log_f = None
        if n > 0:
            try:
                if fast_kalman_only:
                    slope, _ = analyzer_small.check_price_change(ta.PRICE_CHANGE_THRESHOLD_EUR)
                    gradient, _, _, _gr = window_small.get_short_trend()
                    slope_big, _ = analyzer_big.check_price_change(ta.PRICE_CHANGE_THRESHOLD_BIG_EUR)
                if not shadow.compute_vol and shadow_fields:
                    v1h = shadow_signals.vol_1h_pct(window_big.prices, window_big.sample_rate_sec)
                    adapt_re, adapt_dca = shadow_signals.adaptive_thresholds(v1h)
                    shadow_fields = {
                        **shadow_fields,
                        "vol_1h_pct": v1h,
                        "adapt_reentry_pct": adapt_re,
                        "adapt_dca_pct": adapt_dca,
                    }
                last_state = {symbol: {
                    "current_price": price, "final_trend": gradient,
                    "gradient_recent": _gr, "slope_small": slope, "slope_big": slope_big,
                    "epsilon": window_small.get_noise_epsilon(), "ts": ts,
                    **shadow_fields,
                }}
                with open(os.path.join(out_dir, "analysis_state.json"), "w", encoding="utf-8") as sf:
                    json.dump(last_state, sf)
            except Exception:
                pass
    
        pnl = broker.pnl_summary()
        if first_price and broker.last_price:
            # benchmark: buy & hold on the same standard quantity, the same interval
            bh_qty = BACKTEST_NOTIONAL_USD / first_price if first_price else 0
            pnl["buy_hold_net"] = round((broker.last_price - first_price) * bh_qty
                                         - 2 * bh_qty * first_price * FEE_PCT / 100, 2)
        pnl["mode"] = "kalman_primary" if kalman_primary else "model_actual"
        try:
            with open(os.path.join(out_dir, "pnl.json"), "w", encoding="utf-8") as pf:
                json.dump(pnl, pf, indent=1)
        except OSError:
            pass
        sys.stderr.write(f"[tradeall_backtest] P&L: {pnl}\n")
        sys.stderr.write(f"[tradeall_backtest] DONE: {n} ticks, BUY={broker.n_buy} SELL={broker.n_sell}\n")
        sys.stderr.write(f"[tradeall_backtest] results in: {out_dir}\n")
        sys.stderr.write(f"[tradeall_backtest] view: "
                          f"python -m visuals.tradeall_observe --backtest-dir {out_dir} --symbols {symbol}\n")
    finally:
        if orig_place_smart is not None and hasattr(ta, "po"):
            ta.po.place_order_smart = orig_place_smart
        if orig_mkt_place is not None:
            ta.mkt.place = orig_mkt_place
        if orig_mkt_cancel is not None:
            ta.mkt.cancel_order = orig_mkt_cancel
        if orig_api_cancel_expired is not None and hasattr(ta, "api"):
            ta.api.cancel_expired_orders = orig_api_cancel_expired
        if orig_log_decision is not None:
            ta.log_decision = orig_log_decision
        ta._kalman_ref = orig_kalman_ref
        if orig_gate_log is not None:
            ta.GATE_OUTCOME_LOG = orig_gate_log
        for _k, _v in _orig_bt_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
        if quiet:
            try:
                import log
                log.enable_print()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD — simulation start date")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: up to the end of saved data)")
    p.add_argument("--speed", choices=["real", "fast"], default="fast",
                   help="real = honour the historical intervals; fast = no waiting (the default)")
    p.add_argument("--run-id", default=None, help="default: <symbol>_<start>_<timestamp>")
    p.add_argument("--source", choices=["history", "cache24"], default="history",
                   help="history = cache_sparse_price_*.jsonl (~11 months, but SPARSE, see the caveat in the plan); "
                        "cache24 = cache_24_price_*.json, a DENSE resolution (~1s) but only what was retained "
                        "that cache (by default the last ~24h, or a file with long retention if "
                        "tradeall_price_archiver.py is already running)")
    p.add_argument("--cache24-file", default=None,
                   help="an explicit path to a cache24 file (e.g. cache_365_price_BTCUSDC.jsonl); "
                        "default: cachedb/cache_24_price_<symbol>.json")
    p.add_argument("--quiet", action="store_true",
                   help="suppress noisy print() calls from tradeall.logic() (much faster on "
                        "dense/long data); own progress messages remain visible")
    p.add_argument("--kalman-primary", action="store_true",
                   help="Kalman drives (BUY on ->UP, SELL likewise on ->DOWN); the old model only "
                        "journals. For an A/B on P&L against the normal run.")
    p.add_argument("--trend-gate-timeout-sec", type=float, default=None,
                   help="29 Jul: a 'wait for a better price' gate for BUY/SELL (faithful to live: "
                        "it DELAYS until a favourable tick, it does NOT block). A SHORT dynamic "
                        "trend window. Default None = OFF (the old behaviour).")
    p.add_argument("--trend-gate-max-wait-sec", type=float, default=3600.0,
                   help="how long a delayed order waits at most before being placed anyway "
                        "(default 3600 = the live value).")
    args = p.parse_args()

    start_ts = datetime.strptime(args.start, "%Y-%m-%d").timestamp()
    end_ts = datetime.strptime(args.end, "%Y-%m-%d").timestamp() if args.end else None
    run_id = args.run_id or f"{args.symbol}_{args.start}_{int(time.time())}"

    run_backtest(args.symbol, start_ts, end_ts, args.speed, run_id, args.source, args.cache24_file,
                 args.quiet, kalman_primary=args.kalman_primary,
                 trend_gate_timeout_sec=args.trend_gate_timeout_sec,
                 trend_gate_max_wait_sec=args.trend_gate_max_wait_sec)


if __name__ == "__main__":
    main()
