# binance — multi-venue trading system

The repository contains the live trading and monitoring system for Binance,
Kraken, and Trading 212, adapters for Hyperliquid, plus the separate replay,
backtest, and validation environment. The repository name is historical: the
architecture is no longer exclusively tied to Binance.

> **Warning:** The code can place real orders. Do not manually start live
> entrypoints and do not run the same fleet in two locations with the same keys.
> Use the manifest and the operational runbook.

## Current architecture

```text
                    config + instruments.conf
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    Binance fleet      independent bots    offline research
  (fleet_supervisor.sh)      (restart_bots.sh)      (offline/, research/)
          │                   │                   │
 tradeall / rtrade       Kraken / T212       replay + backtest
 monitortrades           trailing stops      no live keys
 priceAnalysis           Hyperliquid adapter       │
 cacheManager                  │                   │
          └──────────────┬─────┴───────────────────┘
                         │
             strategies/ + providers/
                         │
        guards → audit → submit/status/cancel
                         │
               persistent state and logs
```

### 1. Orchestration and supervision

`procs.conf` is the single inventory of processes. Each entry has the role:

- `fleet` — processes coordinated by `fleet_supervisor.sh`, launched from the virtual
  environment and supervised by the own loop of the `binance` systemd service;
- `bot` — independent processes launched by `restart_bots.sh` and verified/repaired
  by `healthcheck.sh --supervise`.

`healthcheck.sh` detects both missing processes and live processes with a frozen
heartbeat. `fleet_supervisor.sh` uses `flock`, so that two instances of the fleet
cannot trade simultaneously.

### 2. Main Binance fleet

| Process | Responsibility |
|---|---|
| `cacheManager.py` | market caches, prices, and shared state |
| `priceAnalysis.py` | trend, historical windows, and derived signals |
| `tradeall.py` | main accumulation/trading strategy |
| `rtrade.py` | reaction pipeline and stabilized concurrent execution |
| `monitortrades.py` | generic monitoring through the multi-provider facade |
| `assetguardian.py` | protections and checks on assets |
| `market_alerts.py` | market alerts with cooldown and configurable watchlist |
| `order_retry_worker.py` | single consumer of the persistent order-lifecycle outbox |

`tradeall_observe.py` is an auxiliary observation process. The dense archiver
`tradeall_price_archiver.py` is part of the supervised fleet manifest: upon
restart, it loads the existing JSONL and continues the append, and upon SIGTERM,
it flushes. However, an interruption leaves an unrecoverable gap at WebSocket
resolution, which is why the watchdog restarts it quickly.

### 3. Strategies and execution

- `strategies/spot_dca.py` contains the common spot DCA engine for live and replay;
- `strategies/state_store.py` atomically writes the financial state and treats
  corruption or the inability to save as fail-closed in live;
- `providers/market_api.py` routes operations by symbol to Binance, Kraken,
  Hyperliquid, or Trading 212;
- `providers/execution_audit.py` attaches an `intent_id` and writes the
  submit/status/cancel cycle to `logger/execution_audit/`, without modifying the decision;
- `order_guard.py`, `order_retry.py`, and `order_retry_worker.py` apply guards,
  persistence, and retry reconciliation;
- `order_retry.py` is also the canonical home of `TrackedOrderLifecycle`; strategies
  may retain campaign state while sharing one persist/submit/recover/status/cancel
  implementation. The migration boundary is documented in
  `docs/ORDER_LIFECYCLE_CENTRALIZATION.md`;
- for normal `Instrument.place` calls, the exact client ID is persisted before the
  external submit; provider acceptance is logged as `accepted`, never as a fill;
- shared-outbox deduplication by only `symbol+side` is disabled, so an intent from one
  module cannot overwrite an independent same-side intent from another module;
- the common outbox keeps the venue order ID after acceptance and advances one bounded
  status step per worker cycle: `open`/partial stays tracked without resubmit, a fill
  completes the record, and status failures remain durable for a later poll;
- native `REJECTED`/`EXPIRED` may create a new deterministic client-ID revision for
  only the unfilled remainder. `CANCELED` is terminal and alerted, never blindly
  replayed, because the common layer cannot prove whether cancellation was intentional;
- Trading 212 one-shot IPO orders keep a separate durable lifecycle because that API
  does not provide a usable client-order-ID reconciliation path;
- `trailing_core.py` is the common state machine, and the Binance and Kraken adapters
  keep the venue-specific API, configuration, and state.

Providers unify the mechanics of venue access, not financial strategies. T212 and
Hyperliquid keep their own logic where the execution model differs.

### 4. Independent bots

- `kraken/` — common fills cache, spot bot, xStocks watcher, and trailing stop;
- `212trading/` — a `t212_bot.py` process, with configurable profiles and persistent
  state for orders, partial fills, cancellation, and repricing;
- `binance_api/trailing_stop.py` — trailing/re-buy circuit breaker for Binance;
- `hyperliquid/` — HYPE client and provider. `dn_bot` is disabled in
  `procs.conf`; `hl_dca_bot` is configured outside the manifest, with REAL gates
  active, but is stopped until the DCA reserve is funded. Snapshot from August 21:
  ~1,024 USDC available, zero orders; the 1,000/600 profile requires a minimum of 7,000 USDC.

Kraken uses a separate namespace for the transactions cache. Kraken processes
that send private requests must comply with the key/nonce policy documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### 5. Data, state, and configuration

- `.env`, `*.env` files, and keys are local and are not committed;
- `.env.example` and configurations without secrets describe the configuration contract;
- `cachedb/` contains persistent caches and queues;
- `logs/` and `logger/` directories contain heartbeats, audit, and runtime results;
- `instruments.conf` defines the instruments and their provider;
- `procs.conf` defines only the processes and their supervision mode.

Do not delete state files for a "clean restart": the bot may lose ownership of a
position or may duplicate an existing entry.

### 6. Live versus offline

`offline/` and `research/` are the area for replay, simulations, backtest, and
promoting candidates. The financial engine can be shared with live, but the offline
entrypoint receives a controlled executor and must not have access to private
trading capabilities. Promotion to production requires the active tests and evidence
described in the validation documentation.

## Quick operations

Run the commands from the root of the repository:

| Action | Command |
|---|---|
| Read-only status | `./healthcheck.sh --check` |
| `bot` processes supervision | `./healthcheck.sh --supervise` |
| Fleet start/restart | `sudo systemctl restart binance` |
| Independent bots start | `./restart_bots.sh` |
| Ownership verification | `.venv/bin/python verify_tools/ownership_inventory.py --running` |
| Portfolio snapshot | `.venv/bin/python verify_tools/portfolio_snapshot.py` |
| Controlled deploy | `./deploy_providers.sh` |
| Secrets backup | `./tools/admin/backup_local.sh` / `./tools/admin/backup_remote.sh` |
| Server restoration | `./restore.sh <secrets_folder>` |

After any change in the manifest or configuration:

```bash
./healthcheck.sh --check
git status --short
```

Do not use `pkill -f` manually without checking the pattern; operation scripts have
the necessary order and exceptions for restarting.

## Repository map

| Path | Role |
|---|---|
| `providers/` | contracts and adapters for market data/execution |
| `strategies/` | reusable financial logic and state store |
| `binance_api/` | Binance client and trailing adapter |
| `kraken/` | Kraken integration and processes |
| `212trading/` | Trading 212 engine and integration |
| `hyperliquid/` | Hyperliquid integration; DN stopped in production |
| `forecast/` | trend and survival estimations |
| `verify_tools/` | health, ownership, snapshot, and operational validations |
| `offline/` | runners, simulations, and replay isolated from live |
| `research/` | experiments and results before promotion |
| `tests/` | unit, characterization, and regression tests |
| `docs/` | architecture, strategy, operations, and disaster recovery |
| `systemd/` | units for server and VPN |

## Documentation

- [`docs/README.md`](docs/README.md) — documentation index;
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — runbook, reboot, and diagnostic;
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — contracts, providers, and ownership;
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — financial rules;
- [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md) — backup and recovery;
- [`kraken/README.md`](kraken/README.md) and
  [`hyperliquid/README.md`](hyperliquid/README.md) — per-component details.
