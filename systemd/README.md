# Systemd & Orchestration (mptrade)

This directory contains the systemd services, administration tools, and deployment configuration required to run the automated trading fleet on the PROD machine.

## Components

1. **`python_orchestrator.service`**: The main fleet entry point. It runs `orchestratorTrade/orchestrator.py`, which parses `procs.conf` and spawns all trading bots and background processes asynchronously with process group isolation, hot-reload, and crash-loop backoff.
2. **`pia.service` & `piavpn.service`**: Private Internet Access daemon and supervisor. Managed by `orchestratorOS/livecheck/pia_supervisor.sh` to enforce Dedicated IP binding, MTU clamping, and fallback routing.
3. **`trading-admin`**: Privileged management CLI wrapper (installed to `/usr/local/sbin/trading-admin`) granting scoped passwordless sudo for:
   - `trading-admin status`: Show live status for fleet and VPN services.
   - `trading-admin restart|stop|start`: Control the fleet and VPN services safely.
   - `trading-admin reinstall`: Re-render systemd units and crontabs from a clean git repo.
   - `trading-admin rename <folder>`: Safely rename trading directory and update systemd unit working directories.
   - `trading-admin pin-dns`: Apply DNS cache tuning and restart systemd-resolved without restarting the fleet.
4. **`crontab.root.prod.txt`**: Root-level cron jobs (such as `vpn_watchdog.sh` and `manage_gitautodeploy.sh`).
5. **`crontab.prod.txt`**: Unprivileged trading user cron jobs (such as log pruning, deadman switch, backups, and resource watchdog).
6. **Network & System Tuning**:
   - `resolved-20-trading-cache.conf`: Systemd-resolved DNS cache drop-in.
   - `netplan-99-force-gateway.yaml`: Direct uplink routing bypass for Proxmox hairpinning.
   - `logrotate-pia-daemon.conf`: Caps PIA daemon debug logs to prevent disk exhaustion.
   - `pia_settings_mtu.sh`: Pre-start WireGuard MTU clamp.

## Manual Diagnostics

To manually test the PIA WireGuard tunnel and Binance connectivity on demand without restarting the daemon:
```bash
./orchestratorOS/livecheck/pia_supervisor.sh --check
```

## Installation

To install or reconfigure on the target machine:
```bash
sudo env TRADING_ROOT="$PWD" TRADING_USER="$(id -un)" systemd/install_prod.sh
```

For bare-metal rebuild procedures, see `docs/DISASTER_RECOVERY.md` and `docs/PROXMOX_DR.md`.

