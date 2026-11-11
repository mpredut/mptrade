# Plan: a unified backtest framework -- FOR DISCUSSION, not implemented

An answer to the question from the session: "I would like a single backtest for all
the modules, where I just set the parameter or module to test; the ranges should come from
the module's config, perhaps through a structured comment above each
parameter -- or do you have a more elegant idea, with less code and information that is
less duplicated?"

Short conclusion: YES to the idea of uniformity, but NOT through a single
simulation engine, and NOT through parseable comments in every config file.
The recommendation: 2 engines (already separate, and they stay separate) plus 1 generic CLI above them
plus 1 SINGLE declarative file holding the ranges (not comments scattered across N
different config formats). Details below.

---

## 1. Why NOT a single simulation engine

From what I found while working on #1/#2 today, there already are, de facto, TWO
irreconcilable simulation paradigms in this repository:

| | **Fleet** (tradeall.py, monitortrades.py) | **Position bots** (kraken, hyperliquid, t212) |
|---|---|---|
| Existing engine | `offline.backtests.tradeall.run_backtest()` | `kraken/backtest.py::simulate()` |
| Time unit | TICK (a continuous price, ~1-7 min per tick from the archive) | OHLC BAR (1h/4h/1d) |
| State | `PriceWindow`/`TrendState`/`WindowAnalyzer` -- sliding windows, a continuous trend | `qty/cost/dca/last_open` -- a discrete DCA/TP/SL state machine |
| Decision | slope/gradient vs a threshold, over a window | price vs average*(1+/-%), on the bar close |
| Symbol/pair | multi-symbol, coordinated (`TrendCoordinator`) | ONE symbol per bot instance |

Forcing the two into the SAME engine would mean either (a) turning OHLC bars
into pseudo-ticks (losing fidelity: the real kraken strategy evaluates
on the bar close, not on every tick), or (b) a single file full of
`if bot_type == "fleet": ... else: ...` which would become EXACTLY the kind of
unclear code that the unification is trying to avoid. Today's two engines
are already CORRECT and validated (kraken/backtest.py now also has the re-entry
barrier, after today's merge) -- the problem is not that there are two, it is that
they have no shared facade above them.

**Recommendation: keep 2 engines, unify only the LAYER ABOVE them**
(CLI, grid generation, reporting) -- see sec. 3.

---

## 2. Why NOT structured, parseable comments in the config

The idea (a comment above each parameter, in a format the backtest can
parse, e.g. `# SWEEP: 0.5,1.0,1.5,2.0,2.5`) is attractive
at first sight, but it has 3 concrete problems, ALREADY observed in this repository:

1. **3 different config formats, today**: `.env` (KEY=VALUE, `tradeall_config.env`
   and so on), `monitortrades.conf` (its own format, `key = value`), `instruments.conf`
   (INI, `[NAME]` sections). A comment parser would have to know how to
   read all 3 -- the exact opposite of "less code, more uniform".
2. **Not everything worth testing already has an env var**: 3 of the Kalman constants
   (`CONF_ENTER`, `MIN_VEL_PCT_MIN`, `GAP_RESET_SEC`, see
   `BACKTEST_CANDIDATES.md`) are hardcoded, with NO config line where
   a comment could be attached. A mechanism based on "a comment above
   the parameter in config" only covers them once you extract them first.
3. **Comments drift silently** -- in this very session I found and
   fixed TWO stale comments (`tradeall.py`: "SHADOW observational" when
   it was in fact initiating real orders; `instruments_config.py`: claiming consumers
   that did not exist). A comment that is BOTH documentation AND parseable
   configuration inherits exactly the same fragility -- nothing guarantees it
   stays in sync with the real value next to it.

**The proposed alternative: a SINGLE declarative file**, not comments
scattered across N formats. See `offline/research/BACKTEST_CANDIDATES.md` (already written
today) -- I would extend EXACTLY that file (not a new one) with a machine-readable block
(YAML/JSON in a fenced code block, or a `.json` sidecar next to it) that
contains, for each table row, exactly what a runner needs in order to
know what to run:

```yaml
# an illustrative example, NOT implemented yet
- id: mt_btc_gain
  module: monitortrades
  engine: fleet
  target: {file: instruments.conf, section: BINANCE_BTC, key: mt.gain}
  range: {min: 5.0, max: 9.0, values: 5}   # -> 5,6,7,8,9
- id: kraken_dca_drop
  module: kraken
  engine: position
  target: {file: kraken/config.env, key: STRAT_DCA_DROP_PCT}
  range: {min: 0.5, max: 2.0, values: 5}   # -> 0.5,0.75,1.0,1.5,2.0
```

Why this is simpler than comments-in-config:
- **ONE place, one format** -- not 3 parsers for 3 config formats.
- **It covers what has not been extracted yet** (`target` may be missing or empty for an idea
  that is not configurable yet -- the runner knows to refuse or warn clearly, rather than
  guessing from an absent comment).
- **It does not duplicate the information twice "in code"** -- the config file stays
  100% clean (only the LIVE value, as it is today); the test range lives WHERE
  the candidate list already lives today, merely structured instead of prose.
- Editable by hand just as easily as today's markdown table (it stays in
  the SAME file, only with a data block next to the prose).

---

## 3. What "a single backtest, I just set the parameter" would mean concretely

A thin, generic CLI above the 2 engines plus the range file above:

```
python3 offline/research/backtest_runner.py --param mt_btc_gain
python3 offline/research/backtest_runner.py --param kraken_dca_drop --symbol HYPEUSD
```

What it would do (schematically, still a plan, not code):
1. Look `--param` up in the declarative block inside `BACKTEST_CANDIDATES.md`.
2. From `engine: fleet|position` it knows which of the 2 engines to use.
3. Generate the grid (`min/max/values`, or an explicit list) -- at most 5 values,
   the same generation logic for BOTH engines (that is the genuinely
   "unified" part: generating the grid, running the loop and reporting the results
   in a shared table, NOT the simulation itself).
4. For each value in the grid, call a SMALL, engine-specific adapter:
   - the `fleet` adapter: build a suitable `threshold_provider`/monkeypatch
     and call `offline.backtests.tradeall.run_backtest(..., threshold_provider=...)`
     (the hook added today for #2).
   - the `position` adapter: set the key in the `P` dict and call
     `kraken.backtest.simulate(ohlc, P, ...)` (extended today for #1).
5. Collect `pnl.json` from each run and print a comparison table
   (value | net_total | buy_hold | cycles/trades | maxDD) -- the same
   format for any parameter, whichever engine it uses.

New code required: 1 small CLI (the sweep loop plus the table) plus 2 small adapters (fleet,
position) -- the rest (the engines, the hooks) has existed since today. It is not a
rewrite, it is a facade over what exists.

---

## 4. Fleet vs bots -- a direct answer to the question in the message

Yes, a SEPARATE backtest for the fleet (tradeall/monitortrades) and for the position bots
(kraken/hyperliquid/t212) -- but NOT two separate CLIs for the user, rather
TWO ADAPTERS under the SAME CLI (`--param X` picks the right engine automatically through
the `engine:` field in the declaration). From your perspective as a user, it stays
"a single backtest, I just set the parameter" -- the real separation (fleet vs position) is
an implementation detail that is hidden, not something you have to choose by hand.

---

## 5. What this plan does NOT solve (limits reported honestly)

- Rtrade and assetguardian still have NO backtest engine at all (unlike
  kraken/tradeall) -- they would need a third adapter, or
  one of the 2 existing ones has to be extended, once it is decided which paradigm
  fits better (rtrade looks closer to "position" -- DCA-like -- although
  with concurrent BUY and SELL, something no engine models today).
- Comparability BETWEEN different modules (e.g. "which is better, a 7% gain
  on BTC or a K of 2.0 on Kraken re-entry") makes no direct sense -- each
  parameter is compared only against ITS OWN variants, not across modules. The plan above
  does not try to solve that (nor should it).
- The proposed declarative file (sec. 2) still demands human discipline to be updated
  when a live value changes -- it reduces the risk of drift (one single place,
  not N comments), but it does not remove it entirely. There could be a simple test
  that checks that every `target.key` in the declaration really exists in
  the config file it refers to (at least avoiding typos and deleted keys).

---

## Questions for discussion -- DECIDED 28 Jul (user)

1. The declarative file (sec. 2): **a block inside `BACKTEST_CANDIDATES.md`** (not a YAML/JSON
   sidecar) -- one single place, next to the prose table, without duplication. It is useful
   mainly as fuel for the CLI (sec. 3); until then it stays a structured
   registry, not a necessity.
2. rtrade/assetguardian: **out of scope for now.** rtrade needs a new engine (concurrent
   BUY/SELL on the same symbol, modelled by no engine today); assetguardian
   has little backtest value. We will include them once the adapter pattern is
   validated on the 2 from phase 1.
3. The CLI (sec. 3): **later**, after another 1-2 cases (e.g. #15-16) -- we stay on
   individual scripts so that the adapter emerges from experience rather than from design
   a priori.

---

## 9. A 2-machine architecture (DECIDED 28 Jul, user) -- backtest on "dev", apply on production

Motivation: running the backtests on the PRODUCTION machine competes for CPU with
live trading (exactly what made ~20-90 min/cycle prohibitive for "several
times a day"). Moving them to a mirror machine (**"dev"**) solves that and was
the real bottleneck for periodisation. Three components:

**A. The dev machine runs the backtests.** They read `cachedb/cache_price_{symbol}.jsonl`
(the price archive) plus the LIVE values from config (the baseline). So dev needs
both fresh -> **sync production->dev through git** (user's decision: git, not
ssh/rsync -- it gives an audit trail for free, and it is reversible). dev pulls the archive and config.

**B. Writing back is a PROPOSAL GATE, not a direct write** (the guardrails
split across the 2 machines):
- **dev** runs the grid plus the confirmation over 2 windows, producing only
  the *confirmed winner* per key (the clean signal), and proposes it (a git commit).
- **production** receives the proposal and APPLIES it: averaging with the real live value
  (authoritative there), a 7-day rate limit, an audit, and writes the config. dev has NO right
  to write directly -- the same 5 guardrails remain, only the CPU moves.

**C. `watchdogfor_cacheandconfig.py`** (user's decision: ONE single .py for cache AND
config -- extend the existing watchdogfor_cache.py, NOT a separate file). Besides
cache staleness (today), it looks through the config files and restarts
the owning process when a config has changed recently. It generalises
`scheduled_pilot._restart_monitortrades()`, plus a bonus: **it also catches
manual edits** (today you edit a config and have to remember to restart). 3
requirements for it to be robust:
1. **Detection on a content hash, NOT on mtime** (the same content-based pattern
   already used for the cache) -- otherwise touching a file without a real change
   causes false restarts.
2. **A config->process map** -- note that some configs have SEVERAL consumers:
   `instruments.conf` is read by monitortrades.py AND tradeall.py, so one change means
   restarting both.
3. **Debounce / atomic write** -- it restarts only after a complete write, a
   single time.

To be implemented once the dev connection details are available. Until then:
backtest candidates that run 100% locally (one-off, dry run), see sec. 8/#15-16.

---

## 6. Observation 23 Jul (after #1/#2): does the market API have a unified interface
that can give both "live now" and "the simulated state at moment X"?

The user's question: do the exchanges have a unified API from which you take either the
real LIVE state now, or the SIMULATED state at a moment X -- that would be an important step
towards a consistent backtest?

Answer: YES, that is the right direction, and it is PARTLY true here already, but in two
separate pieces that have not yet been joined:

- `providers/market_api.py` (the `mkt` facade) already unifies LIVE **across
  exchanges** (Binance/Kraken/Hyperliquid/T212 answer the same calls:
  `get_current_price`, `get_orders`, `free_balance`).
- `offline/backtests/tradeall.py`'s `_SimClock` plus the historical tick iterator
  already unifies LIVE versus HISTORICAL **for time**, but ONLY for tradeall.py, and NOT
  through the facade -- it is a separate loop that rebuilds `PriceWindow`/
  `TrendState` directly from historical data, bypassing `TrendCoordinator`/
  cacheManager entirely (the REAL path by which tradeall.py obtains prices today).

What does NOT exist yet: the `mkt` facade itself having a "replay mode" -- that is,
`mkt.get_current_price(symbol)` being able to answer either "now" or "at
the simulated timestamp T", through the SAME call. If it existed, the REAL code of
the bots (not a separate reimplementation like `kraken/backtest.py::simulate()`)
could run unchanged against history -- removing entirely the risk of
drift between "what the real bot does" and "what the backtest simulates" (exactly
the problem found today in #1: the re-entry barrier was missing from the simulation because
the simulation was a COPY, not the real code).

An honest limit: this unification only solves the "what the market said" side --
you still need a separate simulated broker (like today's `BacktestBroker`/`simulate()`
engines) in order to decide "would this order have filled at that historical
price" -- that stays a DIFFERENT, complementary mechanism, one that does not
disappear once the price/time source is unified.

---

## 7. User request: the fleet (tradeall/monitortrades/rtrade/assetguardian) must
be UNIFORM in its price source (the cache, not live) and its time (from the timestamp
of the price, or the simulated scale of the backtest) -- where do we start?

I pick 2 modules for PHASE 1 (not all 4 at once), on the criterion of "the least effort
times the greatest immediate value":

### PHASE 1: `tradeall.py` (formalise what already exists) plus `monitortrades.py` (new)

**`tradeall.py` -- already ~70% there.** `TrendState`/`PriceWindow` already accept
an injectable `now_fn` (that is EXACTLY the "time comes from the
simulation" mechanism being asked for) and `offline/backtests/tradeall.py` already refeeds `PriceWindow`
with historical prices instead of live ones. What is missing today: the mechanism is ad hoc,
written once in `offline/backtests/tradeall.py`, not reusable from elsewhere
(today's `threshold_provider` hook is a first step towards generalising it, but
the price source and the clock stay "sewn into" the `run_backtest()` loop, rather than a
separate, reusable component). Phase 1 here means extracting `_SimClock` plus
the historical tick loading into a small, separate component
(`PriceReplaySource`?), WITHOUT changing tradeall.py itself (it is already injectable
enough).

**`monitortrades.py` -- 0% today, but the greatest value.** There is NO
backtest for it at all, and `BACKTEST_CANDIDATES.md` identified gain/lost per
symbol (`instruments.conf`) as the most valuable UNTESTED candidate in the whole
inventory (#4-5, HIGH priority).

**23 Jul, CONFIRMED (not merely speculated) -- the injection seam ALREADY EXISTS, complete:**
- `Instrument.__init__(..., api=None)` -- if `api` is not given, it falls back to
  the live singleton (`_default_api`); if it IS given, `self._provider =
  self._api.provider_by_name(provider)` uses THAT api. Every method
  (`price()`, `orders()`, `free()`) delegates to `self._provider`.
- `instruments_config.load_instruments(path=None, api=None)` and
  `load_for(consumer, path=None, api=None, ...)` ALREADY propagate this `api`
  onwards to every `Instrument` built from `instruments.conf`.
- Conclusion: `monitortrades.py` needs NO change AT ALL at the lines where
  it reads price and orders (`inst.price()`, `inst.orders(...)`, `load_for("mt")`)
  -- only a different `MarketApi` (a REPLAY one) has to be built and injected when
  a backtest starts. That was in fact exactly the purpose this facade
  was designed for ("Phase 2a/2b" in `market_api.py`'s docstring -- someone
  in an earlier session had already planned this kind of extension).
- `MarketDataProvider` already has a `get_price_history(symbol, lookback_h)` stub
  -- but, checked today: the 2 REAL implementations that exist (Hyperliquid,
  Kraken) are LIVE-ONLY ("the last N hours from time.time() NOW", hitting the real
  network every time) -- fine for backfill when a bot starts, UNUSABLE
  as a replay source (they do not read from a local cache and accept no arbitrary moment T
  in the past). T212 and Binance do not even implement it (they return None).

**Only ONE new piece remains to be written**: `ReplayMarketDataProvider` (implementing
`MarketDataProvider`, reads from `cache_price_{symbol}.jsonl`/`cache_24price_*`,
keeping an internal cursor/clock that advances with every read) plus injecting the
2 `time.time()` calls in `monitortrades.py` (`get_relevant_trade`,
`monitor_price_and_trade`) through a `now_fn` defaulting to `time.time`, tied to
the SAME clock that the new provider advances -- that way the time really does come "from
the time of the price obtained", as the message asked, not from a separate simulated clock.

The effort is much smaller than it first looked: 1 new file (the replay
provider) plus a minimal clock injection in monitortrades.py -- NOT a rewrite of
the price and order paths, which already work through `api` injection.

**Why NOT rtrade/assetguardian in phase 1:**
- `rtrade.py` runs BUY and SELL on SEPARATE THREADS, concurrently, on
  the SAME symbol -- no engine today (fleet or position) models that;
  it would require a new design, not merely price/time injection.
- `assetguardian.py` evaluates once every ~54s on an AGGREGATE portfolio
  value (the "AssetValue" cache), not on the price of one symbol -- its source
  of "truth" is a different kind of cache from the price one; injecting
  time and price is simpler there, but the backtest value is smaller
  (it is "practically stopped" on a rise already, see `BACKTEST_CANDIDATES.md` sec. exclusions).

They remain PHASE 2, once the pattern (an injectable price source plus an injectable clock)
is validated on the 2 from phase 1.

### What a unified price source plus clock would mean concretely (schematic, still a plan)

Two SMALL components, reusable between tradeall and monitortrades:

- **`Clock`**: an object with one method, `now() -> float`. Default = `time.time`
  (live behaviour, unchanged). In replay, `now()` returns the timestamp of
  the LAST price read from the source below -- not a simulated clock that advances
  independently, exactly as the message asked ("time should come from the time of the price
  obtained") -- that is already the `_SimClock` pattern from `offline/backtests/tradeall.py`,
  merely generalised so it is not tied to a single file.
- **`PriceSource`**: an object with one method, `get_price(symbol) -> float`.
  Default = today's live path (mkt/cacheManager, unchanged). In replay, it
  reads sequentially from `cache_price_{symbol}.jsonl`/`cache_24price_*.json`,
  advancing the associated `Clock` on every read.

Both modules (tradeall, monitortrades) would receive these 2 objects through
injection (a parameter whose default is today's live behaviour), not through
an external monkeypatch -- that is the difference from today's pattern in
`offline/backtests/tradeall.py` (which monkeypatches `ta.po.place_order_smart`
and so on from OUTSIDE) and would make testing more direct and clearer.

### A caution (the same standard as today's extractions)

Any change in `monitortrades.py` itself (not just a harness beside it)
must pass the same test: the default value (without a custom Clock/PriceSource
injected) must reproduce EXACTLY today's behaviour -- verified
numerically, with dedicated tests, before any commit. The decision logic does not
change, only the SOURCE of the input data.

---

## 8. The pilot built on 23 Jul -- status and a real TODO found (not merely theoretical)

The pilot (monitortrades.py, `offline/research/backtest_ranges.py` plus
`offline/research/monitortrades_backtest/scheduled_pilot.py`) is built and validated
manually (dry run, one parameter). During the validation, 2 REAL bugs
(not hypothetical) surfaced in the backtest mechanism itself:

1. **`run_replay_backtest.SYMBOLS` was a hardcoded copy** of `instruments.conf`,
   frozen before we added `mt.buy_budget`/`mt.max_budget` -- any test
   would later run WITHOUT that protection, unnoticed (the fix: read it live,
   on every access).
2. **`is_trend_up()` was reading the LIVE trend cache** (contaminating a historical
   replay with the REAL, current state of the market -- the same backtest, run
   twice, gave different results). Fix APPLIED: neutralised deterministically
   (`return False`) during the backtest.

Fix #2 is a SIMPLIFICATION, not full fidelity -- user's observation (23 Jul,
evening): `priceAnalysis.py` (which feeds the trend signal from
Binance in production) should run IN SYNC with `tradeall.py`/the replay
historically in a backtest, not merely be neutralised. That is: `priceAnalysis.py` would
need the SAME price-plus-clock injection treatment as `monitortrades.py` got
today, computing the trend FROM the replayed history (the same clock, the same ticks),
not from a separate live feed. That would make `is_trend_up()` reflect what the bot
would TRULY have seen at that historical moment, rather than just "no signal".

**It stays an explicit TODO, NOT implemented yet** -- it is a piece of work comparable
in size to what we did today for monitortrades.py (Instrument/MarketDataProvider),
but for priceAnalysis.py, which today has NO injection point at all. Until then,
any backtest that uses is_trend_up() (so any monitortrades backtest)
keeps the "neutral" simplification -- correctly labelled, but incomplete.

**Re-validation after the fixes**: today's `instruments.conf` (buy_budget=250,
max_budget=3500 for TAO) gives STABLE and identical results for max_budget between
1500-5000 (net -$170.83 vs buy&hold -$426.06) -- the instability seen before
the fix (the same max_budget giving +$3016 and then -$5279) was entirely bug
#2, not a real sensitivity to the parameter. No config change needed.

**The scheduler's cadence**: a complete run for 1 parameter (4 values x 2
windows) took ~5 min on this hardware; all 4 pilot parameters would
mean ~20-90 min per cycle -- too slow for "several times a day" without
adjustments (a shorter window for routine runs? rotating through the parameters instead
of all of them at once? to be decided before putting it on cron).

**The pilot was RUN in full (23 Jul, evening)**: 3 of the 4 keys were NOT confirmed (a different
winner between the 2 windows -- the guardrail working correctly, rejected as noise). 1
clearly confirmed: TAO `mt.lost` -- 5.6% won on BOTH windows (edge vs
buy&hold 259.85/329.0, against 84.07/208.21 at the current 4.9%). Applied automatically
(averaged): 4.9 -> 5.25. Every guardrail worked as designed.

**A TODO investigated, NOT implemented (23 Jul, evening -- user request)**: the observation
that `priceAnalysis.py` and `tradeall.py` should be in sync inside a
backtest. Verified by grep: it is NOT `priceAnalysis.py` that writes the trend signal
read by `is_trend_up()` -- it is `tradeall.py` itself (`TrendCoordinator.evaluate()`
writes `gradient_recent`/`final_trend` into `cacheManager.get_short_trend_manager()`,
a singleton with IN-PROCESS memory plus a file fallback on `cache_instant_trend.json`
for any process that has nothing in memory yet -- which explains exactly
the contamination found: the backtest process, being a NEW process, falls back to
the file on disk, which is written LIVE by the real tradeall.py). The correct fix would
mean feeding a `PriceWindow` (the same type already used by
`offline/backtests/tradeall.py`) with the SAME replayed history as `ReplayMarketDataProvider`,
computing a real `gradient_recent` from historical data, published into an
ISOLATED snapshot (not the global file) so that `is_trend_up()` reads it.
Tractable (the pieces already exist), but untested -- deliberately deferred (not at this
hour, without being able to validate it as rigorously as the rest of the session) in favour of
the backtest runs already built and validated, left to run overnight.

---

## 10. The search strategy: OFAT vs a full grid (EXTENSION, 28 Jul -- NOT implemented)

The user's question: do I take the parameters one at a time, or all combinations? An estimate
was given: "4 params x 3 samples = 12 runs".

**An arithmetic correction** (the trap is combinatorial, not linear):
- **One-at-a-time (OFAT / coordinate)** -- you change ONE parameter, the others stay fixed
  at the live value: `4 params x 3 samples = 12`. The figure of 12 is correct ONLY here.
- **A full grid (the Cartesian product)** -- every combination: `3^4 = 81`. With 5 samples
  -> `5^4 = 625`. At ~37s/run (2 windows): OFAT ~9 min/symbol; a grid of 81 ~100
  min/symbol; a grid of 625 ~13h/symbol. It explodes quickly.

| | OFAT | Full grid |
|---|---|---|
| Runs (4p x 3s) | 12 | 81 |
| Does it catch interactions between parameters? | NO | YES |
| Interpretable ("which knob is badly set") | YES | hard (a coupled verdict) |

**Why the interactions matter here (not theoretically):** `max_budget=5000` gave
+$3016 in one configuration and -$5279 in another, on the SAME history -- the parameters
couple strongly. OFAT can find a good winner for parameter A ONLY at the current
value of B; if B changes, A's optimum moves.

**Recommendation (pragmatic, real money):**
1. **Default = OFAT** -- exactly what the pilot does today. Cheap, a clear verdict per parameter,
   and it suits rotation on dev (one parameter per run).
2. **A full grid ONLY on known coupled pairs** (e.g. `gain x lost`, or
   `hardtp x hardtp_fraction`) -- 3x3=9 combinations, cheap, and it catches precisely the
   interaction OFAT misses. A grid over all 4 at once -> avoided (81+).
3. **One confirmation run on the COMPLETE proposed config before applying it** --
   even with OFAT plus independent per-parameter application, you end up at a combination
   never backtested together; confirm it once before writing the config.
4. **Tight ranges (3-4 samples)**, always including the live value in the grid.

**A mitigation that already exists:** the 7-day per-parameter rate limit in
scheduled_pilot means that in production you change ONE parameter at a time anyway (each
confirmed change waits 7 days) -- you do not apply 4 moves simultaneously and blindly. The risk
of an "untested combined config" is naturally bounded.

**A proposed extension (NOT implemented):** a `--grid gain,lost` mode in
scheduled_pilot that takes the Cartesian product ONLY over the given pair (the rest stay
OFAT/fixed). Small, optional, to be added on dev when we implement the dev/prod phase.
