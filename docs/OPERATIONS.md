# OPERATIONS — how it works, and the pitfalls (runbook)

The system's "why", and the traps that are invisible from the code. For a full rebuild
see [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md).

## Architecture in brief
- **The Binance fleet** (8 processes): `cacheManager`, `assetguardian`, `priceAnalysis`,
  `tradeall`, `monitortrades`, `rtrade`, `market_alerts`, `order_retry_worker`. Started
  and supervised by `trade_engine/orchestrator.py`, under **systemd `binance`**.
- **Bots running outside the fleet**: `kraken_cachemanager`, `kraken_bot`,
  `kraken_xstock_watch`, `t212_bot`, `kraken/trailing_stop` and
  `binance_api/trailing_stop`. They are started from the `bot` roles in `procs.conf` and
  supervised by `healthcheck.sh --supervise` (cron */5).
- **Hyperliquid**: `dn_bot`/watch are stopped and commented out in the manifest. `hl_dca_bot`
  is stopped, has zero orders, and stays outside `procs.conf`. The prepared
  1,000/600/10,000 profile needs at least 7,000 USDC; the snapshot had ~1,024 USDC available.
- **Market/account facade**: `providers/market_api.py` routes by symbol to
  `BinanceProvider` / `HyperliquidProvider` / `kraken` / `t212`. `monitortrades` uses it.

## The single source of processes: `procs.conf`
Format: `pat | dir | start_cmd | label | hb_log | hb_stale_s | role` (`role=bot|fleet`).
Read by **all of them**: `healthcheck.sh`, `trade_engine/orchestrator.py`, `restart_bots.sh`, `deploy_providers.sh`.
**To add, remove or change a process, edit ONLY `procs.conf`.**

## Supervision — `healthcheck.sh`
- `--check` — a READ-ONLY preview (what it would do, without touching anything). Always safe.
- `--supervise` (cron */5) — restarts `role=bot` processes that are dead OR frozen; `role=fleet` is
  alert-only (fleet_supervisor owns those). Backoff: at most 3 restarts per 30 min, then a crash-loop alert.
- `--alert` — alerts only, no restart.
- **Double detection:** absence (`pgrep`) **and HANG** (a live process whose `hb_log` has not been
  written for `hb_stale_s`). The active heartbeats are the ones declared on each manifest
  line; the commented-out HL entries are neither supervised nor restarted.

## Startup / deploy / backup
- **Startup:** `trade_engine/orchestrator.py` (the fleet, systemd) · `restart_bots.sh` (the bots).
- **Code deploy:** `deploy_providers.sh` — `git pull` -> an **import gate** (it does not restart
  if the facade fails to load) -> fleet restart -> verification.
- **Backup/DR:** `tools/admin/backup_local.sh` (local, derived automatically from `git ls-files`), `tools/admin/backup_remote.sh`
  (encrypted Storj), `restore.sh`. Details in DISASTER_RECOVERY.md.

## On REBOOT — everything comes back on its own
- systemd `binance` (enabled) -> `fleet_supervisor` -> the fleet (after VPN/pia).
- The crontab persists -> `healthcheck --supervise` (*/5) starts the processes declared
  active (`role=bot`) within 5 minutes.
- The commented-out HL entries, and `hl_dca_bot.py` which is absent from `procs.conf`, do **not** restart.
- No manual intervention is needed for the fleet and the bots declared active.

### Enabling HL after funding

1. Confirm at least `7,000 USDC` free; `7,200` is recommended for fees and slippage.
2. Confirm zero HYPE orders and a single owner in `ownership_inventory.py --running`.
3. Run the tests and the backtest on DEV after syncing `cachedb`; never on PROD.
4. Start `hl_dca_bot.py` in a controlled way, then check the detached PID, the LIVE state and
   two consecutive ticks. Do not add the process to the manifest without an explicit decision.

## ⚠ PITFALLS AND LESSONS (read before changing anything)

### 1. Lock leak through fd inheritance (a supervisor disabled "silently")
`trade_engine/orchestrator.py` (`exec 9>fleet_supervisor.lock`) and `healthcheck --supervise`
(`exec 8>/tmp/binance_supervise.lock`) use `flock`. If they start a child with
`nohup … &`, the child **inherits the lock's fd** -> it keeps the lock open after the
script exits -> the next run reports "**already running**" forever, which means supervision
is **silently disabled**.
- **Fix (applied):** `8>&-` / `9>&-` at spawn time (the child no longer inherits the fd).
- **Diagnosis:** `lsof /tmp/binance_supervise.lock` (or `fleet_supervisor.lock`) -> a PID holding `8w`/`9w`.
- **Immediate unblock:** `rm /tmp/binance_supervise.lock` (the next run takes a fresh inode).

### 2. A hang is not a crash (a lesson kept from DN)
While it was active, `dn_bot` could freeze silently (a live process producing no ticks), which
is why `pgrep` alone was not enough. If DN is ever reintroduced, its manifest entries
must carry a heartbeat on `hb_log`/`hb_stale_s` again; they are commented out today.

### 3. ⚠ SPOT co-mingling on Hyperliquid
The HYPE spot balance is a single pool per wallet. If DN is restarted, its LONG spot leg,
`hl_dca_bot` and any `monitortrades` owner would all see the same balance; a SELL of "everything
available" can undo the hedge or another engine's position. At the last check `hl_dca_bot` was
stopped; before any second owner exists, exclusive ownership must be demonstrated, or a
separate subaccount/wallet used. `STRAT_EXECUTE` and `HL_LIVE_ORDERS` are necessary gates, not
proof of ownership and not deploy approval.

### 4. The execute bit is lost on edits from Windows
Editing a `.sh` from Windows/UNC resets it to `644` -> cron's `./script.sh` reports
"Permission denied". **Fix:** `chmod +x x.sh && git update-index --chmod=+x x.sh`.

### 5. `pkill -f` can catch ITSELF
`pkill -f trade_engine/orchestrator.py` run from a command whose own string CONTAINS the pattern kills
its own shell. **Use script files or PIDs**, not inline patterns.

### 6. WSL does NOT reach the server
From WSL, `192.168.0.144` loops back to localhost (VPN routing). **Only Windows**
(plink/pscp) reaches the server. Backups are PULLED from Windows into WSL, not the other way round.

### 7. Quoting through plink -> PowerShell -> bash is fragile
Avoid in inline commands: `$( )`, `<`, `|` (alternation in grep), `\"`, `\$`, parentheses in
`echo`. Put the logic in a **script file** (pscp then run) whenever it is non-trivial.

### 8. Do NOT run the fleet in two places on the same keys
The fleet started SIMULTANEOUSLY locally (WSL `/home/mariusp`) AND on the server (`/home/predut`),
on the SAME live API keys -> **duplicated trades** plus Kraken nonce conflicts. Run the
fleet in ONE place; for local testing use separate keys or a demo account, or stop the
server first. (The guard in `--supervise` refuses to start on `/home/mariusp`, but
`fleet_supervisor`/`restart_bots` have NO such guard — be careful.)

## Quick diagnostics
```bash
./healthcheck.sh --check                 # state of every process (read-only)
./healthcheck.sh                         # full report (processes plus HL/Kraken/T212 accounts)
ps -ef | grep -E '[h]l_dca_bot|[d]n_bot' # HL processes, including those outside the manifest
python3 verify_tools/ownership_inventory.py --running
lsof /tmp/binance_supervise.lock         # who holds the supervise lock (a leak?)
tail -n 5 logs/healthcheck.log           # what the supervisor did (cron)
```
