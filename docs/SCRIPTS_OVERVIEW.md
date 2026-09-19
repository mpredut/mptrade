# Script & Configuration Overview

## Scripts in \	ools/admin/\
- **\manage_backups.sh\**: Unifies backup and restore operations.
  - \local\: Backs up secrets (e.g., \.env\, keys, \cachedb/\, states) to a local directory.
  - \emote\: Performs a local backup, then encrypts and uploads it to Storj using \clone\.
  - \estore <dir>\: Restores secrets and completely rebuilds the trading environment (venv, dependencies, systemd profiles, cron).
- **\manage_logs.sh\**: Cleans up logs and cache based on retention policies defined in \config.env\.
- **\pia_selfheal.sh\**: A manual diagnostics and self-healing tool for the PIA VPN connection.
- **\git_autodeploy.sh\**: Continuously checks the main branch for updates, pulls them, and restarts the processes without rebooting.
- **\ename_root.sh\**: Renames the repository root folder while preserving secrets.
- **\make_venv_portable.sh\**: Rewrites hardcoded absolute paths inside the virtual environment to make it portable.

## Scripts in \	ools/monitoring/\
- **\deadman_switch.sh\**: Pings Healthchecks.io at regular intervals. If it fails to ping, it triggers an alert indicating the system might be down.
- **\
tfy_check.sh\**: A diagnostic script for sending test notifications via ntfy.
- **\local_watch_start.sh\**: Starts the main bot fleet locally (for dev/testing).

## Scripts in \offline/runners/\
- **\un_backtest_cycle.sh\**: Runs the backtest proposals generator, then commits and pushes them to the \acktest-proposals\ branch.
- **\	rigger_backtest_dev.sh\**: Initiates a long backtest sequence in the background on DEV.
- **\efresh_dev.sh\**: Syncs production prices to the dev machine for backtesting.

## Configuration
- **\config.env\**: A consolidated file containing global configuration policies for trading parameters, thresholds, risk limits, and logging. (Merged from legacy \_config.env\ files).
- **\instruments.conf\ / \monitortrades.conf\**: Registry for managing supported trading pairs and their parameters.
