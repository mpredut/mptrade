# Systemd & Orchestration (mptrade)

This directory contains the systemd services and deployment configuration required to run the automated trading fleet on the PROD machine.

## Components

1. **`python_orchestrator.service`**: The main entry point. It runs `python_orchestrator/orchestrator.py`, which is responsible for parsing `procs.conf` and spawning all bots and fleet processes asynchronously. It provides built-in hot-reloading if configurations change.
2. **`pia.service` & `piavpn.service`**: Private Internet Access daemon and supervisor. They use `os_orchestrator/livecheck/pia_supervisor.sh` to enforce the Dedicated IP and provide robust fallback routing if the VPN daemon fails.
3. **`crontab.root.prod.txt`**: Root-level cron jobs (like `vpn_watchdog.sh` for auto-recovery).
4. **`crontab.prod.txt`**: User-level cron jobs (like `manage_gitautodeploy.sh`).

## Documentation

All architectural and disaster recovery documentation has been moved to the `docs/` folder in the root repository.
Please refer to `docs/DISASTER_RECOVERY.md` and `docs/PROXMOX_DR.md` for bare-metal rebuild procedures.

## Installation

To rebuild PROD on a fresh machine:
```bash
sudo env TRADING_ROOT="$PWD" TRADING_USER="$(id -un)" systemd/install_prod.sh
```
