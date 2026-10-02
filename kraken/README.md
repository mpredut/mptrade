# Kraken bot — DCA plus take-profit

A port of the `212trading` (Trading 212) strategy onto **Kraken (Spot)**.
The same logic: enter at `market − x%`, buy more on the way down (DCA), sell everything at
`average_price × (1 + TP%)`, then start the cycle again. The symbol is configurable.

## Structure

| File | Role |
|--------|-----|
| `common.py` | utilities: log, `.env`, HTTP (GET plus POST form) |
| `kraken_client.py` | Kraken REST client: **public** (ticker, asset_pairs) plus **private** (balance, add_order, cancel_order, query_orders) with an HMAC-SHA512 signature |
| `market_data.py` | price plus pair availability (the "listing" detector) |
| `notify.py` | ntfy and email notifications (through `alertnotifiers.py` in the root) |
| `strategy.py` | the DCA plus take-profit engine, with **net** P&L (the real Kraken fee) |
| `kraken_bot.py` | the entrypoint, configured from `.env` |
| `.env.example` | a configuration template (copy it to `.env`) |

## Starting it

```bash
cp .env.example .env      # then set KRAKEN_API_KEY / KRAKEN_API_SECRET
python3 kraken_bot.py --find-pair hype   # find the exact pair (public, no keys)
python3 kraken_bot.py --price            # see the price (public)
python3 kraken_bot.py --paper            # run the strategy in PAPER mode (no money)
python3 kraken_bot.py                     # LIVE (when STRAT_EXECUTE=true)
```

## Kraken vs Trading 212 differences (worth remembering)

| | Trading 212 | Kraken |
|---|---|---|
| Auth | Basic (key:secret) | **HMAC-SHA512** signed (validated against the vector in the docs) |
| Price | Yahoo Finance | **Kraken's public ticker** |
| Average position cost | the API provides it (`averagePrice`) | **it does NOT** — we track it ourselves from the fills |
| Status of an executed order | `/orders/{id}` returns **404** | `QueryOrders` **works**, including for closed orders |
| Fee | 0.15% FX conversion x 2 | a **trading fee** of ~0.26% taker / ~0.16% maker (real, as reported by Kraken) |
| Sizing | account currency (RON/EUR) -> USD | directly in the pair's quote currency |

## Symbol availability status (checked today)

- **HYPE** ✅ listed — the pairs `HYPEEUR` and `HYPEUSD`. Ready to run.
- **SPCX** ❌ is NOT on Kraken (no SpaceX / xStock token). Set `KRAKEN_PAIR=...` if and when it
  appears; the bot waits by itself until then (just as SPCX does on T212).

## ⚠ Economics
The Kraken spot fee is ~0.26% taker per trade, so **~0.5% per round trip**.
`STRAT_TAKEPROFIT_PCT` has to exceed that threshold (plus the spread) for a net profit.
**Limit** orders resting in the book are often *maker* (~0.16%), which is cheaper.

## Two symbols at the same time
Run two instances, each with its own pair (with separate `.state_<PAIR>.json` state):
```bash
python3 kraken_bot.py --pair HYPEEUR &
python3 kraken_bot.py --pair SPCXEUR &     # once SPCX exists
```

## Archived experiment: trailing decay v3

`kraken-trail-decay-v3` tested a trailing take-profit that narrowed linearly over time: it
started at `tp_trail_pct` and reached `tp_trail_end_pct` after `tp_trail_decay_steps` bars.
The intent was to give a fresh trend room to run, then protect the profit more aggressively
as the trend aged.

The experiment is not promoted into the live strategy. The implementation was built on the
old `kraken/strategy.py` module, before the shared engine moved into
`strategies/spot_engine.py`, and it has no sufficient financial validation on the current
historical data. The risk-adjusted metrics introduced back then (Sharpe/Sortino and friends)
now live in the shared backtest infrastructure.

For a future retest, the decay must be reimplemented in the shared engine, OFF by default,
with fail-closed validated parameters and compatible persistent state. Promotion happens
only after a full DEV benchmark, walk-forward and a fee/slippage stress test, compared
against the current fixed trailing and passed through the promotion gate.
