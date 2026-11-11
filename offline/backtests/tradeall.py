#!/usr/bin/env python3
"""
offline.backtests.tradeall -- replays the price history already saved
(cachedb/cache_price_{symbol}.jsonl, ~11 months) through EXACTLY the same decision
model as tradeall.py (PriceWindow / WindowAnalyzer / TrendState / logic /
logic_small), with place_order_smart replaced by a stub that simulates
execution instead of hitting the Binance network. It writes ONLY into a separate
logger/backtest/<run_id>/ -- NEVER in the live logs used by
live tradeall.py (see plan A5).

Usage:
    python3 -m offline.backtests.tradeall --symbol BTCUSDC --start 2026-06-01 --speed fast
    python3 -m offline.backtests.tradeall --symbol BTCUSDC --start 2026-06-01 --end 2026-06-08 --speed real

Visualisation (while running or afterwards): in another terminal,
    python -m visuals.tradeall_observe --backtest-dir logger/backtest/<run_id> --symbols BTCUSDC
"""
import argparse
from collections import deque
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import order_lifecycle as _ol_mod
import tradeall as ta  # We reuse PriceWindow/WindowAnalyzer/TrendState/logic (A5) -- the model is not reimplemented.
# 23 Jul: SimClock extracted into providers/replay_clock.py (shared with
# monitortrades -- see UNIFIED_BACKTEST_PLAN.md sec. 7/8), passed to
# TrendState(now_fn=...) so that fast-forward is correct (A5).
from botcore import lru_dict_put, resolve_now_ts, safe_int, safe_nonneg_float, safe_pos_float, sanitize_pipe_field
from providers.base import normalize_side
from providers.replay_provider import SimClock, backtest_isolation_context  # noqa: F401
from state_io import append_text_line, atomic_write_json, iter_jsonl_dicts, read_json_dict
from cachepaths import cache_path, cache_symbol_variants, legacy_filenames_for, migrate_legacy_filename_if_needed
from instrument_registry import casefold_get
from offline.backtests.evaluation import parse_cli_float
from offline.research.monitortrades_backtest.replay_trend_source import (  # noqa: E402
    DynamicReplayTrendSource,
)
import shadow_signals


FEE_PCT = 0.1   # Binance spot fee ~0.1% per leg (taker)
# Standard position size for simulation (kalman-primary + buy&hold benchmark) --
# 21 Jul: replaces ta.api.quantities[symbol] (removed from bapi.py, it was only
# a placeholder always trimmed by the weight limit in live, but here we really do need
# a REAL, fixed NUMBER, so that variants can be compared honestly on the same volume.
BACKTEST_NOTIONAL_USD = 1000.0


class BacktestBroker:
    """Stub for po.place_order_smart: it simulates execution (no network),
    writes into its own backtest folder, the same pipe format as the live
    order_outcomes (Step A2) -- so that tradeall_observe.py can render it identically.
    It also keeps P&L ACCOUNTING: position, average cost, realised, fees."""
    def __init__(self, out_dir, clock, trend_gate=None, trend_gate_timeout_sec=None,
                 trend_gate_max_wait_sec=3600.0, max_quote_budget=None):
        self.clock = clock
        self.path = os.path.join(out_dir, "order_outcomes.log")
        self.n_buy = self.n_sell = 0
        self.pos_qty = 0.0
        self.pos_cost = 0.0       # Total cost of the current position (excluding fees).
        self.peak_deployed_cost = 0.0
        self.max_quote_budget = safe_pos_float(max_quote_budget, default=None) if max_quote_budget is not None else None
        self.realized = 0.0       # realised profit/loss (excluding fees)
        self.fees = 0.0
        self.last_price = None
        self.current_tick_price = None
        # 29 Jul (user's idea): a "wait for a better price" gate. FAITHFUL to live
        # (bapi_placeorder.__place_order -> _maybe_wait_trend -> wait_for_favorable_entry):
        # the gate does NOT block the order, it only DELAYS it until a favourable tick
        # (or until max_wait, then places it anyway). The P&L effect is a better
        # entry/exit, NOT avoided trades. Deferred orders sit in
        # self.pending and execute on future ticks (process_pending, called
        # from the main loop). trend_gate=None => OFF (the old behaviour).
        self.trend_gate = trend_gate
        self.trend_gate_timeout_sec = trend_gate_timeout_sec
        _mw = safe_nonneg_float(trend_gate_max_wait_sec, default=None)
        self.trend_gate_max_wait_sec = _mw if _mw is not None else 3600.0
        self.pending = []          # [{side, symbol, qty_arg, motivation, deadline}]
        self.n_waited = 0          # How many orders were delayed (and then executed).
        self._next_oid = 0
        self._outcome_buf: list[str] | None = None

    def _alloc_order_id(self) -> int:
        self._next_oid += 1
        return self._next_oid

    def flush_outcomes(self) -> None:
        if self._outcome_buf:
            append_text_line(self.path, "".join(self._outcome_buf))
            self._outcome_buf.clear()

    def write_outcome_row(self, cols):
        line = "|".join(sanitize_pipe_field(c) for c in cols) + "\n"
        if self._outcome_buf is not None:
            self._outcome_buf.append(line)
            if len(self._outcome_buf) >= 256:
                self.flush_outcomes()
        else:
            append_text_line(self.path, line)

    def _execute(self, order_type, symbol, price, qty=None, motivation=None):
        """The real P&L accounting (no gate) -- used directly or after waiting."""
        order_type = normalize_side(order_type)
        if order_type not in ("BUY", "SELL"):
            return None
        req_price = safe_pos_float(price, default=None)
        if req_price is None:
            return None
        exec_price = safe_pos_float(self.current_tick_price, default=None) or req_price
        if qty is None:
            qty = BACKTEST_NOTIONAL_USD / exec_price
        qty = safe_pos_float(qty, default=None)
        if qty is None:
            return None
        self.last_price = exec_price
        if order_type == "BUY":
            if self.max_quote_budget is not None:
                rem_budget = self.max_quote_budget - self.pos_cost
                if rem_budget <= 1e-9:
                    return None
                qty = min(qty, rem_budget / exec_price)
                if qty <= 1e-12:
                    return None
            exec_qty = qty
            self.n_buy += 1
            self.pos_qty += qty
            self.pos_cost += qty * exec_price
            self.peak_deployed_cost = max(self.peak_deployed_cost, self.pos_cost)
            self.fees += qty * exec_price * FEE_PCT / 100
        else:
            if self.pos_qty <= 1e-12:
                return None            # Nothing to sell (spot) -- refuse, as in reality.
            sell_q = min(qty, self.pos_qty)
            exec_qty = sell_q
            avg = self.pos_cost / self.pos_qty
            self.realized += (exec_price - avg) * sell_q
            self.fees += sell_q * exec_price * FEE_PCT / 100
            self.pos_cost -= avg * sell_q
            self.pos_qty -= sell_q
            if self.pos_qty <= 1e-9:
                self.pos_qty = 0.0
                self.pos_cost = 0.0
            self.n_sell += 1
        oid = self._alloc_order_id()
        cols = [self.clock(), symbol, order_type, exec_price, exec_qty, "executed", "", "backtest", motivation]
        self.write_outcome_row(cols)
        return {"orderId": oid, "backtest": True}   # Positive int orderId for live track_and_place_order parity.

    def place_order_smart(self, order_type, symbol, price, qty=None, motivation=None, **kwargs):
        order_type = normalize_side(order_type)
        if order_type not in ("BUY", "SELL"):
            return None
        if order_type != "BUY" and self.pos_qty <= 1e-12:
            return None
        # The gate applies UNCONDITIONALLY (force=True included) -- user (29 Jul, "the
        # is always welcome"), consistent with live. If the direction is NOT favourable yet,
        # It DELAYS (it does not block): the order is recorded as pending and executed on the
        # first favourable tick (or when max_wait expires). As in live, the caller
        # (tradeall) gets a truthy result immediately -> it marks the fire confirmed,
        # exactly as after a wait_for_favorable_entry that ends in a placement.
        if (self.trend_gate is not None
                and self.trend_gate.should_wait(symbol, order_type, self.trend_gate_timeout_sec)):
            oid = self._alloc_order_id()
            self.pending.append({
                "order_id": oid, "side": order_type, "symbol": symbol, "qty_arg": qty,
                "motivation": motivation, "deadline": self.clock() + self.trend_gate_max_wait_sec})
            self.n_waited += 1
            return {"orderId": oid, "backtest": True, "pending": True}
        return self._execute(order_type, symbol, price, qty, motivation)

    def cancel_order(self, symbol, order_id=None, *args, **kwargs) -> bool:
        """Cancel a pending delayed order in the simulated broker."""
        if order_id is None:
            self.pending = [o for o in self.pending if o.get("symbol") != symbol]
        else:
            self.pending = [
                o for o in self.pending
                if not (o.get("symbol") == symbol and str(o.get("order_id")) == str(order_id))
            ]
        return True

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
        self.pending = [o for o in self.pending if not (o["symbol"] == symbol and o["side"] == "BUY")]
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


def make_decision_logger(out_dir, clock, decision_buf: list[str] | None = None):
    path = os.path.join(out_dir, "tradeall_decisions.log")

    def _log_decision(symbol, event, **fields):
        try:
            cols = [clock(), symbol, event, fields.get("state", ""), fields.get("old_state", ""),
                    fields.get("price", ""), fields.get("prev_confirm_count", "")]
            line = "|".join(sanitize_pipe_field(c) for c in cols) + "\n"
            if decision_buf is not None:
                decision_buf.append(line)
                if len(decision_buf) >= 256:
                    append_text_line(path, "".join(decision_buf))
                    decision_buf.clear()
            else:
                append_text_line(path, line)
        except OSError as e:
            print(f"[tradeall_backtest] log_decision error: {e}")
    return _log_decision


_TICK_SLICE_CACHE = {}
_FILE_TICKS_CACHE = {}


def _parse_tick_pair(item):
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    ts_ms = safe_pos_float(item[0], default=None)
    price = safe_pos_float(item[1], default=None)
    if ts_ms is None or price is None:
        return None
    return ts_ms / 1000.0, price


def _resolve_existing_cache_file(path):
    migrate_legacy_filename_if_needed(path)
    if not os.path.exists(path):
        for legacy_path in legacy_filenames_for(path):
            if os.path.exists(legacy_path):
                return legacy_path
    return path


_MAX_CACHEABLE_FILE_BYTES = 16 * 1024 * 1024
_MAX_CACHEABLE_TICKS = 250_000


def _iter_jsonl_ticks(path, symbol, start_ts, end_ts):
    try:
        st = os.stat(path)
        sig = (st.st_ino, st.st_size, st.st_mtime_ns)
    except OSError:
        sig = (0, 0, 0)
    file_key = (path, symbol, sig)
    cache_key = (file_key, start_ts, end_ts)
    cached = _TICK_SLICE_CACHE.get(cache_key)
    if cached is not None:
        yield from cached
        return

    sym_variants = set(cache_symbol_variants(symbol, include_original=True))
    if sig[1] > _MAX_CACHEABLE_FILE_BYTES:
        for rec in iter_jsonl_dicts(path):
            if rec.get("s") not in sym_variants:
                continue
            parsed = _parse_tick_pair(rec.get("i"))
            if parsed is None:
                continue
            ts, price = parsed
            if (start_ts is None or ts >= start_ts) and (end_ts is None or ts <= end_ts):
                yield ts, price
        return

    full_ticks = _FILE_TICKS_CACHE.get(file_key)
    if full_ticks is None:
        full_ticks = [
            parsed
            for rec in iter_jsonl_dicts(path)
            if rec.get("s") in sym_variants and (parsed := _parse_tick_pair(rec.get("i"))) is not None
        ]
        if len(full_ticks) <= _MAX_CACHEABLE_TICKS:
            lru_dict_put(_FILE_TICKS_CACHE, file_key, full_ticks, max_entries=4)

    ticks = [
        (ts, price)
        for ts, price in full_ticks
        if (start_ts is None or ts >= start_ts) and (end_ts is None or ts <= end_ts)
    ]
    if len(ticks) <= _MAX_CACHEABLE_TICKS:
        lru_dict_put(_TICK_SLICE_CACHE, cache_key, ticks, max_entries=8)
    yield from ticks


def load_ticks_history(symbol, start_ts, end_ts):
    """Reads cache_sparse_price_{symbol}.jsonl -- a long history, SPARSE."""
    path = _resolve_existing_cache_file(cache_path(f"cache_sparse_price_{symbol}.jsonl"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path} (the price history for {symbol})")
    yield from _iter_jsonl_ticks(path, symbol, start_ts, end_ts)


def load_ticks_cache24(symbol, start_ts, end_ts, filename=None):
    """Read cache_24_price_{symbol}.json (or cache_365_price_{symbol}.jsonl
    if a filename is given) -- a DENSE resolution (~1s/tick, as in live)."""
    if filename:
        path = filename
    else:
        path = cache_path(f"cache_24_price_{symbol}.json")
        migrate_legacy_filename_if_needed(path)
        if not os.path.exists(path) and not os.path.exists(f"{path}.previous"):
            alt_jsonl = _resolve_existing_cache_file(cache_path(f"cache_365_price_{symbol}.jsonl"))
            if os.path.exists(alt_jsonl):
                path = alt_jsonl
    if str(path).endswith(".jsonl"):
        path = _resolve_existing_cache_file(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {path} (cache24 for {symbol})")
        yield from _iter_jsonl_ticks(path, symbol, start_ts, end_ts)
        return
    if not os.path.exists(path) and not os.path.exists(f"{path}.previous"):
        raise FileNotFoundError(f"Cannot find {path} (cache24 for {symbol})")
    data = read_json_dict(path, default={}, recover_previous=True)
    items_map = data.get("items") if isinstance(data, dict) else None
    entries = []
    if isinstance(items_map, dict):
        for cand in cache_symbol_variants(symbol, include_original=True):
            cand_val = casefold_get(items_map, cand)
            if isinstance(cand_val, list) and cand_val:
                entries = cand_val
                break
    for entry in entries:
        parsed = _parse_tick_pair(entry)
        if parsed is None:
            continue
        ts, price = parsed
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        yield ts, price


def run_backtest(symbol, start_ts, end_ts, speed, run_id, source, cache24_file=None, quiet=False,
                 kalman_primary=False, trend_gate_timeout_sec=None,
                 trend_gate_max_wait_sec=3600.0):
    """Run offline replay backtest for tradeall.py on historical or dense cache24 ticks."""
    safe_run_id = str(run_id or "").strip().replace("/", "_").replace("\\", "_")
    if not safe_run_id or safe_run_id in (".", ".."):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    out_dir = os.path.join(ROOT, "logger", "backtest", safe_run_id)
    # 23 Jul: clean the folder BEFORE running -- several files here
    # (order_outcomes.log, tradeall_decisions.log, tradeall_shadow.log,
    # tradeall_price_samples.log) are written via append, so without
    # this, a SECOND run with the SAME run_id would silently mix the old history with
    # the new one. Until now, every caller had to remember to do the
    # shutil.rmtree() itself (offline/research/tradeall_trigger_gate/experiment_quality_
    # signal_v2.py did it by hand) -- moved here once, for every caller.
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    price_path = os.path.join(out_dir, "tradeall_price_samples.log")

    timeout_f = safe_pos_float(trend_gate_timeout_sec, default=None) if trend_gate_timeout_sec is not None else None
    trend_gate = (
        DynamicReplayTrendSource([symbol], max_timeout_sec=max(660.0, timeout_f))
        if timeout_f is not None
        else None
    )
    clock = SimClock()
    broker = BacktestBroker(out_dir, clock, trend_gate=trend_gate,
                            trend_gate_timeout_sec=timeout_f,
                            trend_gate_max_wait_sec=trend_gate_max_wait_sec)
    _MISSING = object()
    orig_mkt_place = getattr(ta.mkt, "place", _MISSING)
    orig_mkt_cancel = getattr(ta.mkt, "cancel_order", _MISSING)
    orig_mkt_tick_size = getattr(ta.mkt, "get_symbol_tick_size", _MISSING)
    orig_mkt_price_tick_size = getattr(ta.mkt, "price_tick_size", _MISSING)
    orig_api_cancel_expired = getattr(getattr(ta, "api", None), "cancel_expired_orders", _MISSING)
    orig_log_decision = getattr(ta, "log_decision", _MISSING)
    orig_kalman_ref = getattr(ta, "_kalman_ref", _MISSING)
    orig_gate_log = getattr(ta, "GATE_OUTCOME_LOG", _MISSING)
    orig_fire_symbols = getattr(ta, "TRADEALL_FIRE_SYMBOLS", _MISSING)
    orig_symbol_oids = (
        {k: list(v) for k, v in ta._symbol_order_ids.items()}
        if isinstance(getattr(ta, "_symbol_order_ids", None), dict)
        else None
    )
    orig_tick_cache = (
        dict(ta._SYMBOL_TICK_SIZE_CACHE)
        if isinstance(getattr(ta, "_SYMBOL_TICK_SIZE_CACHE", None), dict)
        else None
    )
    orig_ol_tick_cache = (
        dict(_ol_mod._SYMBOL_TICK_SIZE_CACHE)
        if isinstance(getattr(_ol_mod, "_SYMBOL_TICK_SIZE_CACHE", None), dict)
        else None
    )
    order_retry_mod = getattr(ta, "order_retry", None)
    orig_retry_get_ids = getattr(order_retry_mod, "get_accepted_order_ids_for_owner", _MISSING) if order_retry_mod is not None else _MISSING
    orig_retry_resolve = getattr(order_retry_mod, "resolve_accepted_order", _MISSING) if order_retry_mod is not None else _MISSING

    with backtest_isolation_context(quiet=quiet):
        try:
            ta.mkt.cancel_order = broker.cancel_order
            ta.mkt.get_symbol_tick_size = lambda *a, **kw: None
            ta.mkt.price_tick_size = lambda *a, **kw: 0.0
            if hasattr(ta, "api"):
                ta.api.cancel_expired_orders = lambda *a, **kw: None
            if order_retry_mod is not None:
                if orig_retry_get_ids is not _MISSING:
                    order_retry_mod.get_accepted_order_ids_for_owner = lambda *a, **kw: set()
                if orig_retry_resolve is not _MISSING:
                    order_retry_mod.resolve_accepted_order = lambda *a, **kw: None
            if orig_symbol_oids is not None:
                ta._symbol_order_ids.clear()
            if orig_tick_cache is not None:
                ta._SYMBOL_TICK_SIZE_CACHE.clear()
            if orig_ol_tick_cache is not None:
                _ol_mod._SYMBOL_TICK_SIZE_CACHE.clear()
            if isinstance(orig_fire_symbols, (set, frozenset)):
                ta.TRADEALL_FIRE_SYMBOLS = set(orig_fire_symbols) | {symbol}
            elif isinstance(orig_fire_symbols, (list, tuple)):
                ta.TRADEALL_FIRE_SYMBOLS = list(dict.fromkeys([*orig_fire_symbols, symbol]))
            if kalman_primary:
                _old_attempts = {"n": 0}
                def _journal_only(order_type, symbol_, price_, qty_=None, motivation=None, **kw):
                    _old_attempts["n"] += 1
                    return None
                ta.mkt.place = lambda symbol_, side_, price_, qty_=None, **kw: _journal_only(
                    side_, symbol_, price_, qty_, **kw)
            else:
                ta.mkt.place = lambda symbol_, side_, price_, qty_=None, **kw: broker.place_order_smart(
                    side_, symbol_, price_, qty_, **kw)
            decision_path = os.path.join(out_dir, "tradeall_decisions.log")

            window_small = ta.PriceWindow(symbol, 300, sample_rate_sec=ta.TIME_SLEEP_GET_PRICE,
                                           window_seconds=ta.WINDOW_SECONDS_SMALL)
            window_big = ta.PriceWindow(symbol, 300, sample_rate_sec=ta.TIME_SLEEP_GET_PRICE,
                                         window_seconds=ta.WINDOW_SECONDS_BIG)
            analyzer_small = ta.WindowAnalyzer(window_small)
            analyzer_big = ta.WindowAnalyzer(window_big)
            trend_state = ta.TrendState(expiration_trend_time=2.7 * 60,
                                         fresh_trend_time=3.7 * 60, now_fn=clock)
            trend_state_big = ta.TrendState(expiration_trend_time=2.7 * 60,
                                             fresh_trend_time=3.7 * 60, now_fn=clock)

            shadow = shadow_signals.KalmanStateTracker(
                journal=shadow_signals.KalmanJournal(fixed_path=os.path.join(out_dir, "tradeall_shadow.log")),
                now_fn=clock,
                compute_vol=(speed != "fast"))
            ta._kalman_ref = shadow
            ta.GATE_OUTCOME_LOG = lambda symbol_, side, price_, qty, outcome, reason, motivation: (
                broker.write_outcome_row([clock(), symbol_, side, price_, qty, outcome, reason, "backtest", motivation])
            )

            tick_source = (
                load_ticks_cache24(symbol, start_ts, end_ts, filename=cache24_file)
                if source == "cache24"
                else load_ticks_history(symbol, start_ts, end_ts)
            )

            prev_ts = None
            dt_history: deque[float] = deque(maxlen=5)
            n = 0
            first_price = None
            prev_ktrend = 0
            slope = slope_big = gradient = _gc = _sf = _gr = 0.0
            fast_kalman_only = bool(kalman_primary and speed == "fast" and quiet)
            def _persist_analysis_state(cur_px, cur_ts, cur_shadow):
                try:
                    state_obj = {symbol: {
                        "current_price": cur_px, "final_trend": gradient,
                        "growth_coefficient": _gc, "slope_full": _sf,
                        "gradient_recent": _gr, "slope_small": slope, "slope_big": slope_big,
                        "epsilon": window_small.get_noise_epsilon(), "ts": cur_ts,
                        **cur_shadow,
                    }}
                    atomic_write_json(os.path.join(out_dir, "analysis_state.json"), state_obj)
                except Exception:
                    pass

            price_buf: list[str] = []
            decision_buf: list[str] = []
            broker._outcome_buf = []
            ta.log_decision = make_decision_logger(out_dir, clock, decision_buf=decision_buf)
            try:
                for path_init in (price_path, broker.path, decision_path):
                    append_text_line(path_init, "")
                    try:
                        os.truncate(path_init, 0)
                    except OSError:
                        pass
                for ts, price in tick_source:
                    clock.ts = ts
                    broker.current_tick_price = price
                    if speed == "real" and prev_ts is not None:
                        time.sleep(max(0.0, ts - prev_ts))
                    dt = ts - prev_ts if prev_ts is not None else None
                    prev_ts = ts

                    if dt and dt > 0:
                        dt_history.append(dt)
                        eff_dt = statistics.median(dt_history)
                        window_small.set_sample_rate(eff_dt)
                        window_big.set_sample_rate(eff_dt)
                    window_small.process_price(price)
                    window_big.process_price(price)
                    if trend_gate is not None:
                        trend_gate.advance(symbol, ts, price)
                        broker.process_pending(symbol, ts, price)
                    price_buf.append(f"{ts}|{symbol}|{price}\n")
                    if speed != "fast" or len(price_buf) >= 2000:
                        append_text_line(price_path, "".join(price_buf))
                        price_buf.clear()

                    if not fast_kalman_only:
                        thr_small, thr_big = ta.PRICE_CHANGE_THRESHOLD_EUR, ta.PRICE_CHANGE_THRESHOLD_BIG_EUR
                        slope, _pos = analyzer_small.check_price_change(thr_small)
                        gradient, _gc, _sf, _gr = window_small.get_short_trend()
                        small_placed = ta.logic_small(
                            "SMALL", True, symbol, gradient, slope, trend_state, price,
                            price_window=window_small,
                        )
                        slope_big, _price_diff = analyzer_big.check_price_change(thr_big)
                        ta.logic("BIG", not bool(small_placed), symbol, gradient, slope_big, trend_state_big, price)

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
                    broker.last_price = price
                    if kalman_primary:
                        ktrend = shadow_fields.get("kalman_trend", prev_ktrend)
                        if ktrend != prev_ktrend:
                            if ktrend == 1:
                                broker.pending = [
                                    o for o in broker.pending
                                    if not (o.get("symbol") == symbol and o.get("side") == "SELL")
                                ]
                                has_pending_buy = any(
                                    o.get("symbol") == symbol and o.get("side") == "BUY"
                                    for o in broker.pending
                                )
                                if broker.pos_qty <= 1e-12 and not has_pending_buy:
                                    broker.place_order_smart(
                                        "BUY", symbol, price, BACKTEST_NOTIONAL_USD / price,
                                        motivation="kalman_up",
                                    )
                            elif ktrend == -1:
                                broker.sell_all(symbol, price, motivation="kalman_down")
                            prev_ktrend = ktrend

                    n += 1
                    if speed != "fast" and n % 100 == 0:
                        _persist_analysis_state(price, ts, shadow_fields)
                    if n % 5000 == 0:
                        sys.stderr.write(f"[tradeall_backtest] {n} ticks, the last {datetime.fromtimestamp(ts, tz=timezone.utc)} "
                                         f"(BUY {broker.n_buy} / SELL {broker.n_sell})\n")
            finally:
                broker.current_tick_price = None
                if price_buf:
                    append_text_line(price_path, "".join(price_buf))
                    price_buf.clear()
                if decision_buf:
                    append_text_line(decision_path, "".join(decision_buf))
                    decision_buf.clear()
                broker.flush_outcomes()
                broker._outcome_buf = None
            if n > 0:
                try:
                    if fast_kalman_only:
                        slope, _ = analyzer_small.check_price_change(ta.PRICE_CHANGE_THRESHOLD_EUR)
                        gradient, _gc, _sf, _gr = window_small.get_short_trend()
                        slope_big, _ = analyzer_big.check_price_change(ta.PRICE_CHANGE_THRESHOLD_BIG_EUR)
                    if not shadow.compute_vol and shadow_fields:
                        v1h = shadow_signals.calculate_volatility_1h(window_big.prices, window_big.sample_rate_sec)
                        adapt_re, adapt_dca = shadow_signals.adaptive_thresholds(v1h)
                        shadow_fields = {
                            **shadow_fields,
                            "vol_1h_pct": v1h,
                            "adapt_reentry_pct": adapt_re,
                            "adapt_dca_pct": adapt_dca,
                        }
                except Exception:
                    pass
                _persist_analysis_state(price, ts, shadow_fields)

            pnl = broker.pnl_summary()
            if first_price and broker.last_price:
                bh_qty = BACKTEST_NOTIONAL_USD / first_price if first_price else 0
                pnl["buy_hold_net"] = round((broker.last_price - first_price) * bh_qty
                                             - (first_price + broker.last_price) * bh_qty * FEE_PCT / 100, 2)
            pnl["mode"] = "kalman_primary" if kalman_primary else "model_actual"
            try:
                atomic_write_json(os.path.join(out_dir, "pnl.json"), pnl, indent=1)
            except OSError:
                pass
            sys.stderr.write(f"[tradeall_backtest] P&L: {pnl}\n")
            sys.stderr.write(f"[tradeall_backtest] DONE: {n} ticks, BUY={broker.n_buy} SELL={broker.n_sell}\n")
            sys.stderr.write(f"[tradeall_backtest] results in: {out_dir}\n")
            sys.stderr.write(f"[tradeall_backtest] view: "
                              f"python -m visuals.tradeall_observe --backtest-dir {out_dir} --symbols {symbol}\n")
        finally:
            def _restore_attr(obj, attr_name, orig_val):
                if orig_val is _MISSING:
                    try:
                        delattr(obj, attr_name)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, attr_name, orig_val)

            _restore_attr(ta.mkt, "place", orig_mkt_place)
            _restore_attr(ta.mkt, "cancel_order", orig_mkt_cancel)
            _restore_attr(ta.mkt, "get_symbol_tick_size", orig_mkt_tick_size)
            _restore_attr(ta.mkt, "price_tick_size", orig_mkt_price_tick_size)
            if hasattr(ta, "api"):
                _restore_attr(ta.api, "cancel_expired_orders", orig_api_cancel_expired)
            if order_retry_mod is not None:
                _restore_attr(order_retry_mod, "get_accepted_order_ids_for_owner", orig_retry_get_ids)
                _restore_attr(order_retry_mod, "resolve_accepted_order", orig_retry_resolve)
            if orig_symbol_oids is not None:
                ta._symbol_order_ids.clear()
                ta._symbol_order_ids.update(orig_symbol_oids)
            if orig_tick_cache is not None:
                ta._SYMBOL_TICK_SIZE_CACHE.clear()
                ta._SYMBOL_TICK_SIZE_CACHE.update(orig_tick_cache)
            if orig_ol_tick_cache is not None:
                _ol_mod._SYMBOL_TICK_SIZE_CACHE.clear()
                _ol_mod._SYMBOL_TICK_SIZE_CACHE.update(orig_ol_tick_cache)
            _restore_attr(ta, "TRADEALL_FIRE_SYMBOLS", orig_fire_symbols)
            _restore_attr(ta, "log_decision", orig_log_decision)
            _restore_attr(ta, "_kalman_ref", orig_kalman_ref)
            _restore_attr(ta, "GATE_OUTCOME_LOG", orig_gate_log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD -- simulation start date")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: up to the end of saved data)")
    p.add_argument("--speed", choices=["real", "fast"], default="fast",
                   help="real = honour the historical intervals; fast = no waiting (the default)")
    p.add_argument("--run-id", default=None, help="default: <symbol>_<start>_<timestamp>")
    p.add_argument("--source", choices=["history", "cache24"], default="history",
                   help="history = cache_sparse_price_*.jsonl (~11 months, but SPARSE, see the caveat in the plan); "
                        "cache24 = cache_24_price_*.json or cache_365_price_*.jsonl, DENSE resolution (~1s) "
                        "persisted by cacheManager.py")
    p.add_argument("--cache24-file", default=None,
                   help="an explicit path to a cache24 file (e.g. cache_365_price_BTCUSDC.jsonl); "
                        "default: cachedb/cache_24_price_<symbol>.json")
    p.add_argument("--quiet", action="store_true",
                   help="suppress noisy print() calls from tradeall.logic() (much faster on "
                        "dense/long data); own progress messages remain visible")
    p.add_argument("--kalman-primary", action="store_true",
                   help="Kalman drives (BUY on ->UP, SELL likewise on ->DOWN); the old model only "
                        "journals. For an A/B on P&L against the normal run.")
    p.add_argument("--trend-gate-timeout-sec", type=parse_cli_float, default=None,
                   help="29 Jul: a 'wait for a better price' gate for BUY/SELL (faithful to live: "
                        "it DELAYS until a favourable tick, it does NOT block). A SHORT dynamic "
                        "trend window. Default None = OFF (the old behaviour).")
    p.add_argument("--trend-gate-max-wait-sec", type=parse_cli_float, default=3600.0,
                   help="how long a delayed order waits at most before being placed anyway "
                        "(default 3600 = the live value).")
    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    if (
        not symbol
        or symbol.startswith("-")
        or ".." in symbol
        or "/" in symbol
        or "\\" in symbol
        or any(ch.isspace() for ch in symbol)
    ):
        p.error(f"Invalid --symbol: {args.symbol!r}")
    if args.run_id is not None:
        clean_run_id = args.run_id.strip()
        if not clean_run_id or ".." in clean_run_id or "/" in clean_run_id or "\\" in clean_run_id:
            p.error(f"Invalid --run-id: {args.run_id!r}")
    else:
        clean_run_id = None
    if args.trend_gate_timeout_sec is not None and safe_nonneg_float(args.trend_gate_timeout_sec, default=None) is None:
        p.error("--trend-gate-timeout-sec must be >= 0")
    if safe_nonneg_float(args.trend_gate_max_wait_sec, default=None) is None:
        p.error("--trend-gate-max-wait-sec must be >= 0")

    try:
        start_ts = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        p.error(f"Invalid --start date (expected YYYY-MM-DD): {args.start!r}")
    end_ts = None
    if args.end:
        try:
            end_ts = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() + 86399.999
        except ValueError:
            p.error(f"Invalid --end date (expected YYYY-MM-DD): {args.end!r}")
        if end_ts < start_ts:
            p.error("--end date cannot be earlier than --start date")
    run_id = clean_run_id or f"{symbol}_{args.start}_{safe_int(resolve_now_ts(), 0)}"

    run_backtest(symbol, start_ts, end_ts, args.speed, run_id, args.source, args.cache24_file,
                 args.quiet, kalman_primary=args.kalman_primary,
                 trend_gate_timeout_sec=args.trend_gate_timeout_sec,
                 trend_gate_max_wait_sec=args.trend_gate_max_wait_sec)


if __name__ == "__main__":
    main()
