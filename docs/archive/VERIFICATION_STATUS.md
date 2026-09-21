# Verification status

Last updated: 2026-08-23. This is an observational snapshot, not proof of fills or
future profitability.

## Automated gates

- Complete suite: `1173 passed`, `324 subtests passed` after common Binance filters,
  the direct-order boundary gate and read-only intent health reporting.
- Monitortrades characterization and cooldown gates: `38 passed`.
- Import inventory: `16/16 OK` for active/common modules.
- Instrument/provider routing: PASS. Private Binance balance checks are explicitly
  skipped on local WSL when credentials are unavailable; `None` is not treated as a
  real zero balance.
- Account-facade parity: confirmed differences are zero. Private comparisons are
  inconclusive and skipped when the safe facade reports unavailable account data.

The remaining three test warnings come from the installed `python-binance` dependency
importing deprecated `websockets` compatibility APIs. They are not suppressed because
keeping them visible makes the dependency upgrade debt explicit. The previous Python
3.14 fork warnings were removed by running the cross-process cooldown test with
`spawn`, without weakening its file-lock assertion.

## Order ownership inventory

The static inventory reports three informational overlaps, all within an explicit
coordination domain:

- Binance `BTCUSDC`: monitortrades and tradeall are primary owners; assetguardian and
  trailing are additional protective/portfolio owners.
- Binance `TAOUSDC`: monitortrades, rtrade and tradeall are primary owners;
  assetguardian and trailing are additional owners.
- Kraken `HYPEUSD`: monitortrades is primary and trailing is protective.

These are not currently classified as conflicts by the inventory. They must be
rechecked before changing execution ownership, bypass policy or a bot's live symbol
set.

The direct-order static gate separately inventories low-level venue submit/cancel calls.
Only reviewed mechanics and venue-adapter boundaries are allowlisted; a new production
call site fails the test until its persistence and recovery ownership are reviewed.

`healthcheck.sh` also reports the read-only active-intent index. Index read errors and
persisted `unknown` submissions are alert conditions, but healthcheck has no authority to
write, retry, submit, or cancel an intent.

## Binance exchange filters

Price, limit quantity, market quantity and notional validation now share
`providers/binance_filters.py`. Exchange `PRICE_FILTER`, `LOT_SIZE`,
`MARKET_LOT_SIZE`, `MIN_NOTIONAL` and `NOTIONAL` remain distinct from the versioned
business minimum `PLACE_ORDER_MIN_NOTIONAL`.

## Production snapshot

PROD was deployed at `main@42ee397`; `binance.service`, `monitortrades.py` and
`rtrade.py` were active after restart, the checkout was clean, and the inspected recent
logs contained no traceback, 429/418 or timeout errors.

## Deferred because they are not quick, behavior-neutral changes

- durable hard-TP intent/cooldown reconciliation;
- fleet-wide typed order outcomes and unknown-submit recovery;
- retry/cancel-replace lifecycle redesign;
- wiring common market-regime decisions into Kraken/Hyperliquid live policies;
- strategy promotion before sufficient forward divergences and real fill calibration.
