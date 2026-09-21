# Scripts Catalog

This document provides a persistent mapping and explanation of all shell scripts in the repository, organized by directory and purpose. It prevents having to re-analyze what each script does and how they interlock.

## Root Level
Core execution boundaries and entrypoints.

- **`deploy_providers.sh`**
  The main deployment script. Validates python configuration (`--check` mode) or updates the local tree (`git pull --ff-only`), triggers python imports tests, and restarts the processes defined in `procs.conf`. Returns a failure if caching checks fail after deployment.
- **`env_common.sh`**
  A shared environment bootstrapping file. Discovered and sourced by almost every other script to find the correct python `.venv` and export `$PYTHON_BIN`.
- **`fleet_supervisor.sh`** (formerly `flota_start.sh`)
  The continuous execution daemon for the system's core "fleet" role. Used exclusively as `ExecStart=` by `systemd/binance.service`. Supervises processes like CacheManager.
- **`healthcheck.sh`**
  Diagnostic and health monitoring script. Validates the VPN tunnel (`PIA_VPN_IF`), tests real outbound traffic via curl to `api.binance.com`, and confirms fleet operation. If errors occur, it pushes alerts via `ntfy` to the user's phone.
- **`pia_supervisor.sh`** (formerly `pia_start.sh`)
  The continuous execution daemon for the Private Internet Access VPN. Used as `ExecStart=` by `systemd/pia.service`. Applies MTU fixes, configures `gai.conf` (IPv4 precedence), logs in with `piatoken.txt`, and enables the tunnel kill switch.
- **`restart_bots.sh`** (formerly `bots_start.sh`)
  A one-shot executable called by `deploy_providers.sh` (or manually) that signals all running `role=bot` processes listed in `procs.conf` to reload themselves.

## `tools/admin/`
Administrative and disaster recovery tools.

- **`manage_logs.sh`** (merges `rotate_logs.sh` & `logger_retention.sh`)
  Scheduled via `crontab.prod.txt` every hour. Truncates console logs (e.g. `cron.log`, `deploy.log`) if they exceed 50MB and aggressively deletes archived `.log.gz` or dated logs older than 7 days.
- **`git_autodeploy.sh`**
  Automated deployment daemon run by root's cron. Can observe (`shadow` mode) or automatically apply (`on` mode) new commits from `origin/main` (or backtest proposals), restarting `binance.service` safely and applying cooldowns.
- **`pia_selfheal.sh`**
  Emergency disaster recovery script for `pia.service`. Used manually (`--check` or `--force`) to run a recovery ladder (reconnect, restart daemon, relogin, reinstall) when the VPN is hopelessly wedged. It is no longer run from cron to avoid fighting systemd.
- **`manage_backups.sh`**
  Unified script for backup and disaster recovery. Handles local tarball creation, remote uploads (e.g. to Storj) with encryption, and full machine restoration from backups. Replaces the legacy `backup_local.sh`, `backup_remote.sh`, and `restore.sh` scripts.
- **`make_venv_portable.sh`**
  Fixes hardcoded absolute paths inside `.venv/bin/` wrappers when the repository is cloned or moved to a new path.
- **`rename_root.sh`**
  Utility to adjust paths globally across files if the repo name changes.

## `tools/monitoring/`
Observability scripts.

- **`deadman_switch.sh`**
  Cron job that continuously pings a `healthchecks.io` URL to prove the PROD machine is online and scheduling jobs. If this ping stops, `healthchecks.io` sends an alert, ensuring major outages are caught even if local `ntfy` fails.
- **`check_workload.sh`**
  Collects `top`/`ps` metrics of the running bots and dumps a CPU footprint report.
- **`ntfy_check.sh`**
  Simple CLI tool to manually test push notifications.
- **`local_watch_start.sh`**
  Developer utility leveraging `inotifywait` to automatically restart bots when a `.py` file is saved locally.

## `tools/lib/`
Shared utility libraries.

- **`process_control.sh`**
  Bash function library that parses `procs.conf`, matches binary paths via `ps`, and exposes `manifest_pids()` and `stop_manifest_process()` for clean process shutdowns.

## `offline/runners/`
Backtesting and simulation execution scripts.

- **`load_dev_backtest_env.sh`**
  Sets required environment variables like `BACKTEST_PROPOSALS_BRANCH`.
- **`refresh_dev.sh`**
  Triggered via SSH on the PROD server to sync the DEV machine. It `rsync`s the live SQLite cache database from PROD to DEV so backtests run on fresh data.
- **`run_backtest_cycle.sh`** (now includes `publish_proposals.sh`)
  Executes the full end-to-end backtest cycle: runs the simulation pilot in `--propose` mode, checks the results, and publishes them securely to the `backtest-proposals` git branch (via a dedicated git worktree) without dirtying the `main` branch working tree.
- **`trigger_backtest_dev.sh`**
  Initiates `run_backtest_cycle.sh` on the DEV machine remotely.
- **`run_backtest_parallel.sh`** & **`run_financial_baseline.sh`**
  Execution wrappers for historical validation and financial sanity checking of the simulation environment.

## `systemd/`
OS-level provisioning tools.

- **`install_prod.sh`**
  Executed with `sudo`. Copies `binance.service` and `pia.service` to `/etc/systemd/system/`, replacing `@TRADING_HOME@` placeholders. Installs `crontab.prod.txt` (as the local user) and `crontab.root.prod.txt` (as root). Reloads daemons.
- **`pia_settings_mtu.sh`**
  Adjusts interface MTU size.

## `hyperliquid/` & `kraken/`
Provider-specific lifecycle scripts (startup wrappers and emergency closer scripts for different exchanges).
