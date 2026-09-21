# Architecture: OS Orchestrator & Trade Engine (Phase 4)

This document describes the decoupled architecture implemented to separate Linux OS-level administration from the Python MTrade bot execution environment.

## The Principle of Separation
Previously, bash scripts (`fleet_supervisor.sh`, `healthcheck.sh`, `pia_selfheal.sh`) and systemd units were heavily intertwined with Python bots. If the VPN failed, the OS would aggressively kill the trading bots.

The new architecture enforces a strict boundary:
1. **OS Orchestrator**: Maintains the Linux host, VPN, and resource limits.
2. **Trade Engine**: A purely Python-based supervisor that runs bots, handles logging, and watches configurations.

---

## 1. `os_orchestrator/` (OS Admin & Monitoring)
This directory houses all Bash/Python scripts that run as `cron` jobs or one-off tasks.

### Subdirectories:
* **`admin/`**
  * `manage_backups.sh`: Backs up cachedb, logs, and state.
  * `manage_logs.sh`: Rotates and cleans up old `.log` files.
  * `manage_gitautodeploy.sh`: Auto-pulls the latest `main` branch.
  * `make_venv_portable.sh`: Utility to fix venv absolute paths.
  
* **`livecheck/`**
  * `vpn_watchdog.sh`: Monitors `pia.service`. If the VPN drops, it initiates an **Exponential Backoff** recovery (5m -> 10m -> 20m -> up to 2h) to avoid breaking the internet or hitting PIA API limits. While in backoff, the OS routes traffic via the raw ISP, allowing Kraken/HyperLiquid to trade unhindered.
  * `os_resources_watchdog.py`: Monitors process CPU and RAM. Sends an alert if any process consumes 100% CPU for multiple minutes (deadlock detection).
  * `deadman_switch.sh`: A final heartbeat script ensuring the machine itself hasn't frozen.
  
* **`lib/`**
  * `os_notify.sh`: A standalone `curl` wrapper allowing bash scripts to send `ntfy` alerts independently of Python.

---

## 2. `trade_engine/` (MTrade Python Engine)
The centralized supervisor for the Python trading algorithms. 
It runs as a systemd service (`trade_engine.service`) but **does NOT depend** (`Requires=`) on `pia.service`. It stays alive even if the network drops.

* **`orchestrator.py`**
  * Reads `procs.conf` and spawns bots (`kraken_bot.py`, `hl_dca_bot.py`, `monitortrades.py`) via `asyncio.create_subprocess_shell`.
  * **Auto-Restart**: If a bot exits or crashes, the orchestrator instantly restarts it. No polling needed.
  * **Hot-Reload Watchdog**: Every 10 seconds, it checks the MD5 hash of `config.env`. If changed, it gracefully restarts all bots to apply the new config, notifying the user.
  
* **`notification_server.py`** & **`notification_rules.json`**
  * The orchestrator asynchronously reads the `stdout` of all bots. 
  * It passes each log line to the Notification Server, which matches it against regex patterns in `notification_rules.json`.
  * Allows intelligent alert routing (INFO/WARN/CRITICAL) directly from stdout without the bots needing `requests` or `SMTP` libraries.

---

## 3. `verify_tools/` (Manual Diagnostics)
A suite of read-only Python scripts used for manual intervention and state querying.
* `portfolio_snapshot.py`, `check_cache_coherence.py`, `pnl_report.py`, etc.
* Kept purely as tools; they do not interfere with the orchestrator.

## Fallback Polling Flow (The Result)
1. **VPN Drops**: PIA disconnects. Killswitch is disabled, traffic routes through Vodafone.
2. **OS Reaction**: `vpn_watchdog.sh` detects failure, triggers a 5m backoff.
3. **Trade Engine Reaction**: `orchestrator.py` keeps bots running.
4. **Bot Reaction**: 
   * Binance WS drops -> Enters `_mark_unhealthy()` -> Switches to REST Polling (read-only due to IP restrictions).
   * Kraken / HL -> Seamlessly reconnect over Vodafone and continue trading.
5. **Recovery**: `vpn_watchdog.sh` eventually recovers PIA. Routing switches back to `tun0`. Bots automatically reconnect to WS over the Dedicated IP.
