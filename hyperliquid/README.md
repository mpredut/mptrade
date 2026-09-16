# Hyperliquid — integration, operational state and the HYPE strategy

This directory holds three distinct capabilities: the HYPE/USDC spot provider used by the
shared `strategies/spot_dca` engine, the historical directional PERP engine, and the
delta-neutral engine. The existence of code and configuration does not mean a process is
active in production.

## Production state — 21 August 2026

- `hl_dca_bot.py` sits outside `procs.conf`, with the REAL gates configured, but it is
  stopped after the preflight found only ~1,024 USDC for a sizing of 7,000; it is not
  supervised and does not restart automatically;
- the `PAPER-1` incident in the legacy Kraken state was fixed and validated; the LIVE HL
  state is isolated, and the last order was cancelled without a fill;
- `dn_bot.py` and the watcher are stopped and commented out in the manifest;
- the launcher separates the HL state from Kraken and PAPER from LIVE; the PAPER smoke test,
  the reconciliation tests and the first two REAL ticks were validated;
- `monitortrades` for HYPE/Hyperliquid remains disabled through the instrument gate.

The authoritative check is always the combination of:

```bash
rg -n 'hl_dca|dn_bot' procs.conf
ps -ef | grep -E '[h]l_dca_bot|[d]n_bot'
python3 verify_tools/ownership_inventory.py --running
```

## Entrypoints

| File | Market | Engine | State |
|---|---|---|---|
| `hl_dca_bot.py` | HYPE/USDC spot | `strategies.spot_dca` (base v2) | stopped until the DCA reserve is funded |
| `hl_bot.py` | PERP long/short | `hyperliquid/strategy.py` | legacy, unregistered and not started |
| `dn_bot.py` | spot long plus perp short | `delta_neutral.py` | explicitly stopped in the manifest |
| `providers/hyperliquid_provider.py` | spot | the `StrategyExecutor` contract | a lazily imported adapter |
| `hl_client.py` | spot and perp | the Hyperliquid SDK | a wrapper for reads and orders |

`hl_dca_bot.py` uses the same live/replay financial engine as the Kraken bot; the provider
changes the venue, not the strategy's rules.

## Configuration and precedence

The launcher loads `hyperliquid/.env` first, then `hyperliquid/config.env`, and values
already defined are not overwritten. Therefore:

```text
local .env (runtime)  >  versioned config.env  >  the defaults in the code
```

`config.env` and the local overrides now describe the long-term TP5 profile. At the check of
21 August 2026, the effective non-sensitive parameters were:

```text
entry 1,000 USDC | DCA 600 USDC at -2% | cap 10,000 USDC | SL 7%
TP 5% | trend-hold active | adaptive trailing 1.5-8%
```

With `STRAT_MAX_DCA_BUYS=10`, the maximum exposure is `1,000 + 10x600 = 7,000 USDC`, not the
nominal 10,000 cap. The operational snapshot had ~1,023.68 USDC free and zero orders;
activation requires at least 7,000 USDC, 7,200 recommended.

Do not infer the live configuration by reading `config.env` alone. For diagnosis, load the
files in the same order as the launcher and print only the non-sensitive keys. Never commit
`.env` and never print the agent-wallet key.

## Safety gates

For `hl_dca_bot.py`, real money requires all of these at once:

1. the process launched without `--paper`;
2. `STRAT_EXECUTE=true`;
3. `HL_LIVE_ORDERS=true`;
4. a valid agent-wallet key and account;
5. ownership that does not conflict with another process on the same HYPE spot balance.

A missing gate keeps it in PAPER or makes the provider refuse the order. These gates do not
replace operational approval and explicit inclusion in the manifest.

## The spot fee and what it implies for the TP

Spot and PERP have different fee schedules. At the base spot tier, the official Hyperliquid
documentation gives roughly `0.040% maker` and `0.070% taker` per fill; the old values of
`0.015%/0.045%` are for PERP and do not justify a spot TP of `0.5%`. The account tier,
staking, the builder fee and the effective fill type can all change the cost. The backtest
must model LIMIT and MARKET separately, plus a stress scenario.

Source: <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees>.

## The HYPE long-term analysis — 21 August 2026

The offline study used the frozen HYPE/USDC Hyperliquid spot dataset: 3,772 bars of 4h
(~628 days), OOS walk-forward with 15/30/60-day windows, state reset per TEST, partial fills
and worst-case intrabar ordering.

The central scenario used fees of `0.04% LIMIT / 0.07% MARKET`; the stress scenario used
`0.07% / 0.10%`, a 20 bps spread, 30 bps of market slippage and at most a 50% LIMIT fill per
bar. The execution assumptions are conservative but still uncalibrated against real
Hyperliquid fills.

Mean return under stress:

| Variant | 15 days | 30 days | 60 days |
|---|---:|---:|---:|
| the effective adaptive TP5 profile | -0.114% | -0.114% | +0.393% |
| **TP3 + trend-hold + fixed 3% trail** | **+0.098%** | **+0.464%** | **+1.069%** |
| TP5 + fixed 3% trail | -0.002% | +0.018% | +0.526% |

The preferred candidate for **shadow**, not for live, is `long_tp3_trail3`: the same amounts
and SL, the TP armed at 3%, trend-hold active and a fixed 3% trailing. Its stress drawdown
was lower than the effective profile's at every horizon (`5.74/8.78/8.82%` versus
`6.41/9.49/10.09%`). The aggressive TP5/trail3/SL10 profile had the highest mean, but its
stress drawdown rose to `16.7%` over 60 days, so it is not the robust candidate.

No candidate passed the formal promotion gate. `long_tp3_trail3` improved the mean, the tail
and the drawdown, but it did not win consistently enough fold by fold in the 15-day scheme.
The conclusion is **shadow/paper only** until there is forward evidence, not a live change.

The versioned forward test runs through `hyperliquid/shadow_longterm.py`. The runner uses
only public HYPE spot candles, builds `HLClient()` without a key and without an `Exchange`
object, keeps a separate anchor, and compares `current`, `long_tp3_trail3` and `reentry4`. It
does not read the bot's balance or state and cannot place orders. The re-evaluation threshold
remains at least 30 days and 20 decision divergences, not 20 identical snapshots.

```bash
./myenv/bin/python hyperliquid/shadow_longterm.py
```

The confirmation run was executed exclusively on the DEV host `backtest`. The current
temporary artefact is:

```text
/tmp/hl_dev_sweep_20260821.json
```

The artefact is not versioned; the durable figures and assumptions are kept here and in
`chatgpt_agent_work/OPEN_ACTIONS_PROD_FINANCIAL.md`.

## Co-mingling and ownership

The HYPE spot balance is a single pool per wallet. If DN were restarted, its long spot leg
would share the balance with `hl_dca_bot` or `monitortrades`; a SELL of "everything
available" could undo the hedge. Before any activation, use a separate subaccount or wallet,
or demonstrate exclusive ownership. DN being stopped today removes the current runtime
conflict, not the architectural risk on restart.

## Authentication

Hyperliquid uses an EIP-712/ECDSA wallet signature. For automation, use an approved
agent/API wallet rather than the main key:

- `HL_SECRET_KEY` — the agent's private key;
- `HL_ACCOUNT_ADDRESS` — the main account's address.

The secrets stay exclusively in `.env`, are excluded from Git, and must be included in the
backup/disaster recovery procedure.

## Safe commands

```bash
# import/provider, no orders (from the repository root)
cd "$(git rev-parse --show-toplevel)"
myenv/bin/python -m unittest -q tests.test_hyperliquid_provider_executor

# the launcher forced into PAPER; do not add it to the manifest just for a test
cd hyperliquid
../myenv/bin/python hl_dca_bot.py --paper
```

Starting PAPER still creates a persistent process; stop it in a controlled way after the
check. Do not use the commands above as a substitute for the promotion gate.
