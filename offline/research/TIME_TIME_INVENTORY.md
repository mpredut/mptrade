# A `time.time()` inventory -- the fleet plus the bots (23 Jul 2026)

Step 0 of the plan in `UNIFIED_BACKTEST_PLAN.md`: find ALL references to
`time.time()` in the fleet (tradeall/monitortrades/rtrade/assetguardian) and the bots
(kraken/hyperliquid/212trading), classified by what removing them would mean --
using the time that arrives WITH THE PRICE (the tick's or bar's timestamp), or
injected from outside (a `Clock`, like `_SimClock` in offline/backtests/tradeall.py).

Legend:
- [HIGH] **DECISION** -- it affects WHAT is traded and WHEN. It must be injected for
  a faithful backtest.
- [MED] **LIVE INFRA** -- cache, rate limit or REAL polling towards the exchange. There is no
  sense in it becoming "simulated time" -- it has nothing to simulate (it does not exist in a
  backtest, where the network is never touched).
- [LOW] **ALREADY SOLVED** -- the code containing it is already REPLACED entirely in the
  backtest (monkeypatched or bypassed), never called there.
- [WARN] **NOT just injection** -- the dependency on REAL time runs deeper than
  a parameter (a blocking loop on a threading.Event, a synchronous wait on the network).

---

## THE FLEET

### `tradeall.py` (4 occurrences, but only 1 is a real problem)

| Line | Context | Category | Note |
|---|---|---|---|
| 107 | `log_decision()`: `cols=[time.time(), ...]` -- writes to the decision journal | [LOW] | `offline/backtests/tradeall.py` replaces the WHOLE FUNCTION (`ta.log_decision = make_decision_logger(out_dir, clock)`), so this one NEVER runs in a backtest. |
| 613 | `handle_symbol()`: `"ts": time.time()` in the returned snapshot | [HIGH] | It feeds `shadow.update(symbol, snapshot["ts"], ...)`, and Kalman uses `dt = ts - last_ts` to scale the process noise. **Bypassed today** (the backtest does not call `handle_symbol()`, it builds its own flow) -- but if the REAL `handle_symbol()` code ever runs on a replay (the unification plan), this `ts` MUST come from the replayed price timestamp rather than the wall clock, otherwise Kalman computes dt wrongly (mixing real time with historical data). |
| 711 | `TrendCoordinator._is_due()`: `self._last_eval[symbol] = time.time()` | [HIGH] | It gates WHEN a symbol is re-evaluated (min/max interval throttling). Bypassed today (the backtest does not call `evaluate()` through the coordinator). Simple to inject -- a `now_fn` instead of calling `time.time()` directly. |
| 768 | `TrendCoordinator.run()`: `now = time.time()` in the main loop | [WARN] | **NOT just injection** -- the line ABOVE is `self._event.wait(timeout=self.max_interval)`, a REAL wait on a `threading.Event`. To run on a fast-forward replay, this loop would have to be REPLACED (not merely given an injected clock), otherwise a 329-day backtest would literally take 329 days. |

**tradeall.py conclusion**: today, NOTHING in the table above blocks the
backtest (everything that matters is already bypassed or replaced). It becomes relevant ONLY
if the unification plan gets as far as reusing `handle_symbol()`/
`TrendCoordinator` itself (not a separate loop) -- in which case line 768 is
the real obstacle (a redesign, not a parameter), 613 is simple (a parameter), and 107/711
are already solved by the substitution pattern.

### `monitortrades.py` (2 occurrences, both DECISION)

| Line | Context | Category | Note |
|---|---|---|---|
| 294 | `get_relevant_trade()`: `current_time_s = int(time.time())` | [HIGH] | `can_trade = current_time_s - trade_time < threshold_s` -- it blocks a new trade if the last one was too recent. Directly on the decision path. |
| 459 | `monitor_price_and_trade()`: `current_time_s = int(time.time())` | [HIGH] | The HARD-TP cooldown (`current_time_s - _hard_tp_last.get(symbol,0) >= hard_tp_cd`) plus the "trades too recent" window (MT_ALL_TRADES_BLOCK_SEC). |

**monitortrades.py conclusion**: EXACTLY 2 points to inject, both simple
(arithmetic comparisons on an int, no blocking loop involved) --
the most tractable module in the whole fleet for Phase 1, confirming the choice in
`UNIFIED_BACKTEST_PLAN.md` sec. 7.

### `rtrade.py` (0 direct occurrences -- a special case)

Grep confirms: NOT ONE `time.time()`. Rtrade never reads "now" anywhere
directly -- its notion of time is delegated entirely to the API responses:
`api.check_order_filled_by_time("BUY", symbol, time_back_in_seconds=WAIT_FOR_ORDER)`
asks the EXCHANGE "was it filled in the last X seconds?", rather than comparing a
local `time.time()` against a remembered timestamp. That means making rtrade
testable on a replay is NOT a matter of injecting a Clock -- the RESPONSES of those
API calls (`check_order_filled`, `check_order_filled_by_time`,
`cancel_order`) would have to be simulated in a fake broker, like `BacktestBroker`. It confirms one more
reason (besides the concurrent BUY/SELL threads, already noted in the plan) that
rtrade is a different challenge, NOT merely "the same pattern in another file" -- it stays
justifiably Phase 2.

### `assetguardian.py` (1 occurrence, DECISION)

| Line | Context | Category | Note |
|---|---|---|---|
| 51 | `_get_symbol_window_extrema()`: `now_ts = float(time.time())` | [HIGH] | `target_ts = now_ts - minutes_back*60` -- the per-asset window for the low and the high. SELL uses `AG_SELL_TIERS` and freezes the campaign low at the first tranche; BUY uses the per-asset high.

---

## THE BOTS (position-based: kraken, hyperliquid, 212trading)

The SAME pattern repeated in all three: when an order is placed, it records
`"ts": time.time()`; later, `age = (time.time() - ts) / 60` decides whether the
order has sat too long (order TTL, reprice or cancel). Plus similar
cooldowns (`buy_backoff_until`, `_dca_gate_until`, `cooldown_until`). Once
the injection pattern is designed for ONE of them (I recommend kraken/strategy.py, the most
investigated today), the other two align almost mechanically -- they ARE structurally
identical, not 3 different problems.

### `kraken/strategy.py` -- 4 occurrences, all DECISION

| Line | Context | Note |
|---|---|---|
| 212, 219 | `"ts": time.time()` when an order is placed (`open_orders`) | Used at line 263 for the order TTL (`STRAT_ORDER_TTL_MIN`, reprice or cancel). |
| 263 | `age = (time.time() - o.get("ts",0)) / 60` | The decision to reprice or cancel an unfilled order. |
| 442 | `self._shadow_prices.append((time.time(), price))` | It feeds `_shadow_vol_1h()` and therefore the ADAPTIVE re-entry threshold, PROMOTED TO REAL MONEY in this session (`STRAT_REENTRY_ADAPTIVE=true`). The most important entry in the whole bot inventory -- any future backtest of the REAL strategy (not `kraken/backtest.py::simulate()`, which is a different paradigm over OHLC bars) must inject the time correctly here, otherwise the computed volatility is false. |

**A methodological note**: `kraken/backtest.py::simulate()` (today's "position" engine)
does NOT use `kraken/strategy.py` at all -- it is a separate reimplementation over
bare OHLC (already documented in `UNIFIED_BACKTEST_PLAN.md` sec. 1). The rows
above matter ONLY if the plan evolves towards "the REAL strategy code
runs on the replay" (the unified facade, sec. 6 of the plan) -- it changes nothing in
today's `simulate()`.

### `hyperliquid/strategy.py` plus `delta_neutral.py` plus `signals.py` -- 7 DECISION occurrences

The same pattern (a ts on the order plus an age on reading) in `strategy.py:164,170,227`.
`delta_neutral.py` adds: `opened_ts`/`opened_at` (the age of the DN position),
`cooldown_until` (anti-thrash between rebalances) -- lines 272,315,487,495.
`signals.py:62` -- generic staleness (`if time.time()-ts > max_age`). Without
a backtest engine of its own today (unlike kraken) -- it would need
a new one, following the Kraken pattern, if DN is ever chosen for testing.

### `212trading/strategy.py` plus `market_data.py` -- 10 DECISION occurrences

The same order-TTL pattern (lines 301,308,323,329,531) plus specific
cooldowns (`buy_backoff_until`:313,677; `locked_zero_until`:343,473;
`_dca_gate_until`:744,748) plus staleness on market data
(`market_data.py:115,125` -- `age_sec`/`series_age`). Without a backtest engine
of its own today.

### Live infra (there is no sense in these becoming "simulated time")

- `kraken_cachemanager.py` (109,119,190), `kraken_client.py` (54,61),
  `kraken_xstock_watch.py` (97), `hl_client.py` (237),
  `212trading/order_manager.py` (71,73), `hyperliquid/dn_bot.py` (44) --
  they are all either (a) parameters for REAL calls to the exchange API
  (a lookback window, a local cache TTL), or (b) a SYNCHRONOUS wait loop
  on a real network response. None of them exists "during"
  a backtest (which never touches the network) -- there is nothing to inject them WITH.

### Test files (`hyperliquid/test_dn.py`, `212trading/test_launch_detect.py`)

They use `time.time()` to build fixtures (not production code).
If `delta_neutral.py` and the `212trading` code receive an injectable Clock,
these tests could move to a fake clock in turn instead of
`time.time() - X` -- an improvement in test determinism, but NOT something that
blocks the backtest plan (they are tests, not code that runs inside a backtest).

---

## Summary -- what really has to be done in Phase 1 (tradeall plus monitortrades)

| Module | Real DECISION occurrences to inject today | Complexity |
|---|---|---|
| `tradeall.py` | 0 (everything that matters is already bypassed in the backtest) -- it becomes 2 (613 simple, 768 a redesign) ONLY if `handle_symbol`/`TrendCoordinator` are reused directly | Small today, medium if it grows |
| `monitortrades.py` | 2 (lines 294, 459) | Small -- 2 arithmetic comparisons |

Conclusion: **monitortrades.py is in fact simpler to inject than
tradeall.py** in the strict sense (2 clear points, no blocking loops) -- but
tradeall.py already has the replay infrastructure (PriceWindow/TrendState with
`now_fn`) built and validated today, merely not exposed generically. The two
remain the right candidates for Phase 1, for complementary reasons.
