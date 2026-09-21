# Provider-agnostic unification of the strategy engine (Path B)

**Goal:** make the base v2 engine (DCA + TP + trailing, validated live on HYPE) independent
of Kraken and able to run through the same contract on any compatible venue, with a single
conformance suite and without forcing the different T212/HL strategies into the same
algorithm.

**Why B (and not an adapter):** less code (no third abstraction appears) and more testable (a
single parametrised suite across every provider). The engine stays on the strict
`StrategyExecutor` contract; it is not routed through `MarketApi.place`, whose guardrails do
not distinguish between entries and urgent STOP/trailing exits.

## The contract (the single source of truth)

`providers/strategy_executor.py` — the `StrategyExecutor` Protocol plus `OrderStatus`,
`PairPrecision` and `ProviderError`. Mapped against KrakenClient:

| KrakenClient (today) | The agnostic contract | State in providers/ |
|---|---|---|
| `add_order`→txid | `submit_order(...)->order_id` | ✅ Kraken/HL/Binance/T212 |
| `query_orders(txid)` | `order_status(symbol,id)->OrderStatus` | ✅ Kraken/HL/Binance/T212 |
| `cancel_order(txid)` | `cancel_order(symbol,id)` | ✅ Kraken/HL/Binance/T212 |
| `pair_info` | `pair_precision->PairPrecision` | ✅ Kraken/HL/Binance/T212 |
| `balance` | `free_balance(asset)` | ✅ Kraken/HL/Binance/T212 |
| `ohlc_closes` | `ohlc_closes(symbol,interval)` | ✅ Kraken/HL/Binance/T212 |
| ticker/quote | `get_current_price(symbol)` | ✅ Kraken/HL/Binance/T212 |

The real size in `strategy.py`: **8 call sites** (`self.client.*`) plus **6** `except
KrakenError`. The backtest (`replay.py`) uses `MagicMock`, so it is almost untouched.

## Phases (each with a gate)

- **Phase 0 — the contract plus a regression net** ✅
  - `providers/strategy_executor.py` (the contract: 7 methods plus OrderStatus/PairPrecision/ProviderError).
  - `tests/test_kraken_strategy_golden.py` — GOLDEN: the exact trace (14 orders, hash `69fd0a50…`)
    plus the base v2 metrics on HYPE. **It must pass unchanged after the whole refactor.**
- **Phase 1 — Kraken on the contract** ✅ — `kraken_provider` delegates to `kraken_client` plus
  `KrakenError→ProviderError`. 12 conformance tests.
- **Phase 2 — rewire `strategy.py`** ✅ · **the golden is BYTE-IDENTICAL** — 8 call sites moved to
  the contract; 6 `except` clauses moved to `ProviderError`; `kraken_bot` injects
  `KrakenProvider`; `get_current_price` was added to the contract (a `run()` gap caught). A new
  live-path test.
- **Phase 3 — Hyperliquid** ✅ — real `submit_order` (gated by `HL_LIVE_ORDERS`), `order_status`
  (fills), `cancel_order`, `pair_precision` and `ohlc_closes`. 11 conformance tests.
- **Phase 4 — Binance** ✅ — `submit_order`, `order_status` (get_order), `cancel_order`,
  `pair_precision` (filters), `ohlc_closes` (klines). 7 tests. Note on completeness: Binance base
  v2 overlaps with tradeall; `order_status.fee=0` (an approximation, refinable from get_my_trades).
- **Phase 5 — consolidation** ✅ — `tests/test_provider_contract_conformance.py`: a single
  parametrised guard (Kraken/HL/Binance/T212 all satisfy `StrategyExecutor`).
- **Phase 5b — the engine in a neutral namespace** ✅ — the implementation lives in
  `strategies/spot_dca.py`; `kraken/strategy.py` is only a compatibility shim. The state
  directory, the notifier, the source and the venue label are injectable. The replay and the
  tests import the canonical module, with no collisions between the venues' `strategy.py` files.
- **Phase 5c — fidelity plus audit** ✅ — T212 reconciles the real cumulative prices, including
  partial fills; `AuditedStrategyExecutor` adds an `intent_id` and a JSONL journal for
  submit/status/cancel without blocking orders. The autonomous T212 engine now uses the same
  contract for the whole order cycle; STOP/trailing are MARKET, and the replay models them at the
  open with spread and slippage.
- **Phase 6 (deferred; it needs a redesign)** — base v2 is NOT routed directly through
  `MarketApi.place`. If a cross-strategy need arises, an intent-aware decorator is introduced
  separately, one in which STOP/trailing cannot be blocked by trend, cooldown or a cap.

  Re-audit, 20 August 2026: the decision still holds. `MarketApi.place()` is already the guarded
  path for tradeall/rtrade/trailing and returns the result of a one-off placement. `spot_dca`
  instead needs a durable `submit/status/cancel` lifecycle, an `intent_id`, partial fills and
  urgent MARKET exits. Routing through `place()` would layer the cap, the cooldown and the trend
  on top of the strategy's own budget and state machine, and could block STOP/trailing. Unified
  auditing already exists through `AuditedStrategyExecutor`; without a new requirement for a
  global cross-strategy limit, Phase 6 adds no financial or operational value.

  Update, 5 September 2026: a *targeted* slice of the intent-aware guard was added inside the
  engine instead of routing through `place()` — `spot_dca._place` refuses a non-urgent limit
  SELL priced below the average cost (a stale/miscomputed take-profit would otherwise realise a
  loss). Urgent MARKET exits (STOP, trailing) stay exempt, preserving the "STOP/trailing cannot
  be blocked" invariant. This closes the loss-on-mispriced-TP gap for HL/Kraken with no cross-
  strategy coupling; the full Phase 6 (cap/cooldown/trend on the engine) stays deferred.
  Guarded by the byte-identical golden and `tests/test_spot_dca_profit_floor_guard.py`.

## Closing state

Phases 0-5c are merged into `main` at `f5ac673`, validated by the byte-identical golden, the
reproducible financial benchmark and the full suite. There is no remaining contract gap for
the Kraken, Hyperliquid, Binance or Trading212 providers. Phase 6 stays deliberately deferred
and does not block closing the provider-agnostic refactor.

## Safety invariant
The Phase 2 gate is `tests/test_kraken_strategy_golden.py`, and it must pass **byte-identical**.
If it fails, the refactor changed the live base v2 decisions — stop and investigate.
