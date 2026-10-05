# ARCHITECTURE — decoupling and providers (reference notes)

A design snapshot (mid-2026). Check the specifics in the code.

## Shared runtime helpers

`botcore.py` is the single source for `.env`, numeric conversions, single-instance
locking, the clock/log and the stdlib HTTP transport (`GET`, JSON, form and generic
methods). `kraken/kraken_common.py`, `hyperliquid/common.py` and
`212trading/ipo_common.py` keep only genuine display or runtime particularities and
re-export the old API for compatibility.

`alertnotifiers.bind_notify()` centralises how the symbol is chosen from the environment.
The per-venue `notify.py` files are thin shims, still needed for the historical
entrypoints; ntfy/email routing remains a single implementation. The same component
applies deduplication and persistent daily budgets across processes (by default ntfy 100
with 20 reserved for urgent alerts; email 40 with 10 reserved). The runtime state lives in
`logs/notification_delivery_state.json` and resets daily at UTC midnight.

## A shared engine, separate entrypoints

Separating the live entrypoint from the offline one does not mean two strategies:

```text
                         shared strategy engine
                        /                      \
live entrypoint ─► real StrategyExecutor   replay entrypoint ─► OHLC executor
  config/secrets     orders/reconciliation     dataset/hash       fill model/report
  loop/heartbeat     persistent state          no private network controlled state
```

The engine, the rules, the parameters and the financial transitions must be imported from
the same module. Only orchestration and capabilities are separated: the offline process
gets no client with trading rights, and the live process contains no dataset selection or
research metrics. A shared renderer can be used by two thin live/offline entrypoints
without duplicating the logic.

## Ownership inventory and execution audit

The system uses no allocation ledger while the accounts are isolated and execution
overlaps are rare. Two read-only tools cover the current need:

- the execution audit: who requested the order, on which venue/symbol, why, what status and fill it got;
- `verify_tools/ownership_inventory.py`: which owner may execute on each
  `venue + account_ref + symbol`, and where configured or running overlaps exist.

The inventory neither reads nor displays keys, blocks no orders and changes nothing live.
An explicit, non-sensitive `account_ref` can be set per owner or through
`ownership.account_ref` on an instrument; the fallback is `<venue>:default`.
Two primary strategies from the same coordinated pipeline are only `INFO`; two
independent execution domains on the same key are a `WARNING`.

```bash
.venv/bin/python verify_tools/ownership_inventory.py
.venv/bin/python verify_tools/ownership_inventory.py --running
.venv/bin/python verify_tools/ownership_inventory.py --running --json
```

The ledger is reconsidered only if frequent trading from several independent processes on
the same balance ever becomes intentional.

## The market/account facade — decoupling from Binance
`providers/market_api.py` is the facade that routes by **symbol** to the providers (the goal:
the trade monitor becomes generic, not Binance-only).
- The `MarketDataProvider` interface: `get_current_price`, `get_price_history`, `free_balance(asset)`,
  `get_orders(symbol, side, since)`, `get_trades`, `open_orders`,
  `place_order(symbol, side, price, qty, **kwargs)`.
- Providers: `BinanceProvider`, `HyperliquidProvider`, `kraken_provider`, `t212_provider`.
- `MarketApi([providers])` picks the first provider whose `supports_symbol(symbol)` matches; if none
  claims it, the **default is the first, Binance** (behaviour-preserving). The `api` singleton.
- `monitortrades` uses the facade for price, trend, balance, orders and `place_order`. Binance
  stays identical (BinanceProvider delegates to `bapi`/`bapi_placeorder`).
- A generic Instrument plus `instruments.conf` (resolved through `provider_by_name`); BTC/TAO on Binance unchanged.

## The spot DCA/trailing engine

`strategies/spot_dca.py` holds the base v2 financial decision and depends only on
`StrategyExecutor`. Kraken injects `KrakenProvider`, and the replay injects the offline
executor; both run the same class. `kraken/strategy.py` remains a shim for the historical
commands. The state directory, the notifier and the venue label are injectable, but the
Kraken fallback keeps exactly the existing state file.

`hl_bot.py` injects `HyperliquidProvider` into the same `spot_dca` engine.
T212, the legacy PERP engine and delta-neutral are not aliases of it: providers may
satisfy the same mechanical contract, but distinct financial strategies stay separate.

`strategies/state_store.py` centralises the financial snapshots for the spot engine and
T212. Writing is atomic (`fsync` followed by `os.replace`); in real mode, corrupt or
unsaveable state stops the decisions, while PAPER may start clean.

The T212 engine keeps an order locally until the venue reports a terminal status,
including after a cancellation request has been accepted. If the cancellation fails or is
still in flight, it places no repricing or TP ladder on top of the possibly active order;
STOP/trailing may send the urgent exit once the cancellations are accepted, but both
orders stay reconciled.

The T212 engine uses `T212Provider` for the whole submit/status/cancel cycle, while
keeping its own financial rules and the Yahoo feed. The position quantity stays anchored
in the portfolio, and price and P&L come from the real cumulative fills only when the
order delta matches the portfolio delta. Partial fills are applied once; if the status is
temporarily unavailable, the order stays tracked. STOP and trailing are MARKET orders; the
replay fills them at the next bar's open and may apply adverse spread and slippage.

`providers/execution_audit.py` is a strictly observational decorator over
`StrategyExecutor`. Every live intent gets an `intent_id`, kept in the order state, and
submit/status/cancel are written as JSONL into `logger/execution_audit/`.
A failure of the audit can neither refuse nor modify an order.

### HYPE on Hyperliquid (SPOT)
`providers/hyperliquid_provider.py`:
- **public** HL price/history (the @index pair, e.g. `@107` = HYPE/USDC);
- `free_balance` is SPOT (`total − hold`); `get_orders`/`get_trades` are SPOT fills
  (`coin == @index`; PERP fills with `coin=HYPE` are EXCLUDED, so DN does not get mixed in);
- it reuses `hyperliquid/hl_client.py` (the SDK) with a **LAZY import** — the fleet does NOT fall over
  if the HL SDK is missing from its venv (Binance unaffected).
- **Separate gates:** `MT_HYPE_ENABLED` claims HYPE in `monitortrades`, while
  `HL_LIVE_ORDERS` lets the provider send orders. The values can be overridden by
  `.env`; the real state is established from the manifest plus the processes plus the
  environment, not from `config.env` alone.
- At the audit of 21 August 2026, the `PAPER-1` incident from the legacy Kraken fallback
  was fixed: the launcher isolates the HL state and separates PAPER from LIVE.
  The process is now stopped and absent from the manifest; the scaled 1,000/600 profile
  can deploy up to 7,000 USDC, above the ~1,024 USDC available balance.
- ⚠ **Spot co-mingling** (see [OPERATIONS.md](OPERATIONS.md) §3): if DN or several owners
  are reactivated, the same HYPE spot balance can be sold by the wrong engine.

## Multi-process Kraken (a replicated cacheManager)
For 2-3 HYPE trading processes on Kraken (the same `HYPEUSD` symbol) on ONE account:
- **`kraken/kraken_cachemanager.py`** is a SEPARATE process (isolation from Binance: Kraken down
  is not Binance down) that keeps the fills in a cache with its own NAMESPACE
  (`cachedb/cache_trade_kraken.json`); `kraken_provider.get_orders` READS from it (a correct
  cross-process profit guard plus a single feed, so the rate limit is fine), falling back to
  `TradesHistory`.
  - **poll** mode (default, ~5s) / **ws** mode (`KRAKEN_CACHE_MODE=ws`, real-time `ownTrades` —
    the code is ready but inactive; for scalping under 5s it needs `websocket-client`).
- **The Kraken nonce is per KEY** and strictly increasing, so each process needs its own key
  pair (`KRAKEN_API_KEY` / `_WS`), otherwise "Invalid nonce". The keys live ONLY in `kraken/.env*`.
- **Balance:** one account means every process sees the same `free_balance` (a risk of over-selling
  the same symbol); mitigated by the weight cap, the cooldown and the exchange's own rejection.
  Extra, only if needed: a balance reservation layer in the shared cache.

## Trailing stop (a shared core plus per-provider adapters)
A CRASH breaker on the manual holdings (NOT alpha): a WIDE threshold (Binance 20-22%, Kraken 15%)
fires only on a sustained collapse. Refactor, June 2026: the logic was duplicated almost
line-for-line across the two `trailing_stop.py` files, so it moved into
`trailing_core.TrailingCore` (written once).
- **`trailing_core.py`** is the state machine (provider-agnostic): warmup -> track the peak ->
  sell at -trail% -> re-buy on a bounce from the low. **`binance_api/trailing_stop.py`** and
  **`kraken/trailing_stop.py`** are thin ADAPTERS (the `TrailingStop`/`KrakenTrailing` classes),
  carrying only their API plus log/notify. They stay **2 files = 2 processes** with separate
  configs and states (deduplication is not the same as a single file).
- **The adapter contract** (duck-typed): `assets()→(key,asset,pair,trail)`, `begin_tick()→bool`,
  `free_qty(asset)`, `price(pair)`, `trend(pair)`, `execute_sell(...)→bool`, `execute_rebuy(...)→bool`,
  plus `log_*` (venue-specific wording). A new provider only implements these methods; the decision
  logic is never rewritten.
- **The state machine** (`_process`, per asset per tick): (1) **warmup** if `min_profit_pct>0` (it does
  not arm until `price≥entry·(1+min%)`, which avoids selling at a loss after a dip right after you
  bought); (2) a pending **re-buy** (a `+bounce%` recovery from the low, skipped if the trend is
  clearly down); (3) below notional -> skip; (4) `price>peak` -> raise the peak;
  (5) `price≤peak·(1−trail%)` -> sell `free·sell_fraction`, re-arm the peak and arm the `rebuy`.
- **Persisted state** (the schema is unchanged by the refactor): `{"<key>": {"peak", "rebuy":{qty,sell_price,low}?, "warmup_at"?}}`.
  Binance uses `cachedb/trailing_state.json` (keyed by symbol), Kraken `kraken/trailing_state.json` (keyed by asset).
  It survives a restart (the peak is not reset).
- **`item_isolation`** (the error model, a genuine difference): Binance `True` = a try per coin plus
  always saving; Kraken `False` = one try for the whole tick, with no save on error.
- **Config**: `*/trailing.conf` — `(KRAKEN_)TRAILING_ENABLED` means LIVE (dry run by default), `_REBUY_*`,
  `_MIN_PROFIT_PCT`; the thresholds and `CHECK_SECONDS` are in the code (Binance 60s, Kraken 120s).
  **Notification**: Kraken calls `notify()` (ntfy plus email, `source=kraken-trail`) on sell and rebuy;
  **Binance does NOT notify** (only the `trail_b.log` log, which is block-buffered, so confirm through
  the state file or `--status`).
- **Tests** (they guarantee the refactor's equivalence): `tests/test_trailing_stop.py`,
  `kraken/test_trailing_kraken.py`. CLI: `--once`, `--status`. Launched from `restart_bots.sh`,
  supervised by `healthcheck.sh --supervise` (see [OPERATIONS.md](OPERATIONS.md)).

## Market Intelligence Architecture
The `intelligence/` framework provides a four-pillar decision and capital protection layer:
1. **Internal Quantitative**: Kalman trend state, linear gradients, mean reversion, and mathematical guards (anti-FOMO parabolic surge guard, Weibull trend exhaustion guard downscaling mature moves past P90, noise floor guard).
2. **External Microstructure**: Whale flow volume prints, orderbook depth imbalance, and liquidation cascade vetoes.
3. **Sentiment & LLM Reasoning**: Fear & Greed contrarian triggers, market breadth, and Google Gemini LLM reasoning (30m macro assessments and pre-flight vetoes on orders >= 1,000 EUR).
4. **Macro Geopolitical Shield**: Real-time Google News RSS screening with Gemini risk scoring to veto BUYs during international conflict or energy supply shocks.

Wired into `order_guard.py` via `check_intelligence_guards()` and configured in `order_guard.conf`. For complete specifications and backtest verification evidence, see [MARKET_INTELLIGENCE.md](MARKET_INTELLIGENCE.md).

## Performance Optimizations and Non-blocking Execution

To maintain sub-millisecond execution loops across high-frequency price analysis and placement guards, the runtime avoids repetitive allocations, heavy Python wrappers, and in-band network calls:

1. **Analytical Closed-Form OLS Regressions (`pricewindow.py`, `priceAnalysis.py`, `hyperliquid/price_analysis.py`)**:
   - Replaced general-purpose library routines (`scipy.stats.linregress` and `numpy.polyfit(..., 1)`) with vectorized, closed-form formulas:
     \[
     \text{slope} = \frac{N \sum xy - \sum x \sum y}{N \sum x^2 - (\sum x)^2}, \quad r = \frac{N \sum xy - \sum x \sum y}{\sqrt{[N \sum x^2 - (\sum x)^2][N \sum y^2 - (\sum y)^2]}}
     \]
   - Evaluates directly in C-speed vector arithmetic. Bit-for-bit numerical parity confirmed against `linregress` across 30,000 randomized test trials ($\Delta < 10^{-11}$).
2. **Dirty-Flag Position Cost Basis Memoization (`monitortrades.py`)**:
   - Instead of recalculating unmemoized cumulative position statistics on every tick, `_pos_stats_cache_by_symbol` caches the result and invalidates strictly on order placements, cancellations, or a 15-second heartbeat timeout.
3. **Analytical Single-Ratio Return Volatility (`intelligence/internal/state/volatility.py`)**:
   - Uses single logarithmic division rather than computing and allocating full difference arrays, achieving zero discrepancy ($0.00\text{e}+00$) with baseline array slicing.
4. **Zero-Latency Non-blocking Snapshot Policy (`allow_network=False`)**:
   - `order_guard.py` evaluates external Pillar 2 (orderbook depth, whale flow, funding rates) and Pillar 4 (geopolitical threat state) guards strictly from in-memory structures or cached local JSON snapshots.
   - Network polling is decoupled into background collectors. If snapshot data is absent, guards fail open safely without blocking order placement.
5. **Single-Placement Intelligence Memoization**:
   - `regime_context._intelligence_decision` caches the evaluated decision for the lifespan of an `Instrument.place` invocation, preventing repeated execution of guards across multiple internal checks.

## Unified Cross-Venue Trade Weight and Quantity Policy (`order_guard.py`)

A centralized, platform-agnostic allocation and quantity-limiting layer evaluated before placing orders across all venues (Binance, Kraken, Hyperliquid):

1. **Canonical Weight Resolution (`resolve_trade_weight`)**:
   - **Tier 1 (Own Trend Gaussian)**: When a coin has a verified long-term trend, reads `priceAnalysis.get_weight_for_cash_permission_at_quant_time` (evaluating Gaussian distribution, Lindy plateau, and 3-zone momentum/exhaustion bounds).
   - **Tier 2 (Cross-Venue Proxy)**: When the symbol has no trend of its own (e.g. `TAOUSDC` or `HYPE` during consolidation or new listings), looks up `<venue>_weight_proxy` (configured to `BTCUSDC` in `order_guard.conf`). If the proxy has an active trend, applies the proxy's Gaussian weight.
   - **Tier 3 (Conservative Chop Fallback)**: If neither the symbol nor proxy has an active trend, falls back to `chop_weight_for(provider_name)` (default `0.03` / 3%).
   - **Fault-Isolation**: Infrastructure downtime (exceptions from `priceAnalysis`) causes BUY submissions to fail closed with `SubmissionRefused("weight_policy_unavailable")`, while SELL orders fail open to balance cap to preserve exit liquidity. Corrupt numeric outputs (NaN or out-of-range weights) raise `invalid_weight_policy_weight`.

2. **Unified Mathematical Quantity Capping (`compute_weight_capped_qty`)**:
   - Enforces 24-hour traded value limits identically across Binance (`bapi_placeorder.apply_weight_limit`) and shared venues (`order_guard.weight_limit`):
     \[
     \text{total\_ref} = \text{traded\_24h} + \text{available\_qty} \times \text{price}
     \]
     \[
     \text{max\_trade\_value} = \text{total\_ref} \times \text{weight}
     \]
     \[
     \text{remaining\_qty} = \frac{\max(0.0, \text{max\_trade\_value} - \text{traded\_24h})}{\text{price}}
     \]
     \[
     \text{adjusted\_qty} = \min(\text{required\_qty}, \text{remaining\_qty})
     \]
   - Eliminates duplicate logic between venue drivers and unifies behavior across the fleet.

## Single Source of Truth for Sizing, Pricing, and Precision (`providers/quantity.py`, `providers/base.py`)

A unified, venue-agnostic execution contract eliminates duplicated formatting and rounding code across individual strategy callers:

1. **Venue Precision Abstraction (`PairPrecision`)**:
   - Encapsulated in the immutable `PairPrecision` dataclass (`price_decimals`, `volume_decimals`, `order_min`, `base_asset`).
   - Every provider implements `pair_precision(symbol)` and standard rounding helpers:
     - `round_quantity(symbol, qty)`: floors quantity to the venue's step/lot size, guaranteeing the order never breaches available account balance.
     - `round_price(symbol, price)`: rounds limit price to the venue's tick size/price decimals, preventing exchange tick-rejection errors.
     - `order_filter_refusal(symbol, side, price, qty)`: preflights minimum order size and notional limits before order dispatch.
     - `min_order_qty(symbol)`: returns minimum allowable order quantity.
   - Delegated transparently by the top-level `MarketApi` facade (`market_api.round_quantity`, `market_api.round_price`).

2. **Autonomous Quantity Decision Flow (`qty=None` Contract)**:
   - Callers/strategies simply declare their trading intent (`side="BUY"` / `"SELL"`), optionally specifying a target or limit price, leaving `qty=None`.
   - `Instrument.place` passes `qty=None` into `decide_quantity(...)`, where `requested` defaults to `+inf`:
     - **`balance_cap`**: reads available free balance from the venue (`quote` currency on BUY, `base` asset on SELL).
     - **`policy_cap`**: applies portfolio allocation and weight bounds.
     - **`fee_cap`**: reserves balance for exchange trading fees.
     - **Raw Sizing**: computes `min(balance_cap, policy_cap, fee_cap)`.
     - **Market Intelligence Scaling**: dynamically modulates BUY sizing via `intelligence.compute_combined_scale(regime_context)` (e.g., scaling down to 0.5x or 0.25x in adverse regime states).
     - **Precision Truncation**: rounds to venue decimal precision via `round_quantity`.
     - **Filter Guard**: validates lot size and notional constraints via `order_filter_refusal`.
   - If a caller supplies an explicit `qty`, it acts strictly as an upper bound (`requested`) and remains bounded by balance and risk guards.

3. **Automatic Price Tick Rounding in `Instrument.place`**:
   - All non-market (limit) orders pass automatically through `provider.round_price` before guards, order persistence, and venue submission. Callers never need manual `round(price, N)` calls.

4. **Order Retry Worker Quantity Synchronization**:
   - In flight intent claims are updated atomically with the downscaled, guard-adjusted quantity before submission, ensuring retries and outbox tracking never retain stale or un-scaled sizes.

## Single Source of Truth for Fleet Registry and Modular Monitoring

1. **Fleet Registry Architecture (`instruments.conf`)**:
   - `instruments.conf` is the solitary source of truth for instrument configuration across the fleet (Binance, Kraken, Hyperliquid, Trading212).
   - Defines symbols, asset pairs, market hours, isolation modes, operational roles (`role.mt`, `role.tradeall_fire`, `role.trailing`, `role.assetguardian`, `role.force_sell`), and per-coin parameters (trailing stop percentages, buy/max budgets).
   - Loaded and validated strictly via `instrument_registry.py` (`load_registry`, `select_instruments`, `symbols_for`, `single_symbol_for`).

2. **Lightweight Compatibility Facade (`symbols.py`)**:
   - `symbols.py` serves strictly as an import-light compatibility facade over `instrument_registry.py`, reading dynamically from `instruments.conf`.
   - Avoids pulling in heavy network clients or creating circular import chains across legacy callers.
   - Dynamic attribute resolution (`__getattr__`) maps any `<coin>symbol` dynamically against the registry.

3. **Active Modular Price Monitoring and State Ownership**:
   - Multi-source price aggregation, caching, and rate-limited polling are modularized in `market_monitor/pricefetcher.py`.
   - 24-hour spike/drop anomaly detection and alert cooldowns reside in `market_monitor/pricechecker.py`.
   - AssetGuardian position tracking and re-arm thresholds are managed exclusively by `AssetGuardianState` in `assetguardian.py`, persisted atomically with fail-closed concurrency locking in `cachedb/assetguardian_state.json`.



