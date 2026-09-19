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

`hl_dca_bot.py` injects `HyperliquidProvider` into the same `spot_dca` engine.
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
