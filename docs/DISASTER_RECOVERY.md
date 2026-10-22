# Disaster Recovery — Full VM/Network and Fleet Rebuild

How to rebuild the trading VM, its networking, VPN state, and fleet orchestrators from scratch.
This complements `README.md` (services/cron) and `PIA.md` (VPN details).

---

## 1. Target State (What "Correct" Looks Like)

Snapshot of a healthy box:

```text
ens18    192.168.0.144/24  MTU 1280   gateway 192.168.0.1 (direct router uplink, static onlink)
wgpia0   10.x.x.x/32       PIA WireGuard tunnel:
                           - Primary exit:  192.109.159.105 (Frankfurt Dedicated IP, dip-y6N4...)
                           - Fallback exit: 85.122.194.79   (Belgium Dedicated IP, dip-y6N4...)
                           - Dynamic exit:  de-frankfurt    (last-resort VPN region before ISP cooldown)
default via 192.168.0.1 dev ens18 proto static onlink        # base uplink (netplan)
DNS: systemd-resolved stub 127.0.0.53; Global EMPTY;
     wgpia0 -> 10.0.0.243 (Domains ~., owns all public lookups); ens18 -> 192.168.0.1
IPv6: enabled but deprioritized (gai.conf); the kill switch blocks it (tunnel is IPv4-only)
sysctl: net.ipv4.ip_forward=0  rp_filter=2  net.ipv6.conf.all.disable_ipv6=0
```

### Verification Checklist (All Must Hold)

```bash
piactl get connectionstate            # Connected
piactl get vpnip                      # 192.109.159.105 (or 85.122.194.79 during temporary failover)
curl -4 -s https://api.ipify.org      # Matches piactl get vpnip
for i in $(seq 10); do getent hosts api.binance.com >/dev/null && echo ok; done   # 10/10
curl -4 -s -o /dev/null -w '%{http_code}\n' https://api.binance.com/api/v3/time   # 200
sudo /usr/local/sbin/trading-admin status                                          # active / active
./orchestratorOS/livecheck/pia_supervisor.sh --check                              # Health Probe: OK
./orchestratorOS/admin/manage_agy.sh status                                       # Daemon status: active (prod-vm)
```

---

## 2. Automated Components (Git + Backups + install_prod.sh)

The entire deployment is reproducible and path-agnostic:

- **Unified Backup & Restore** (`orchestratorOS/admin/manage_backups.sh`):
  - `backup local`: Takes a snapshot of all repo secrets (`.env`, `config.env`, `212trading/.env`, `autodeploy.local.conf`), caches (`cachedb/`), state files, Antigravity CLI credentials & project configs (`~/.gemini/antigravity-cli/antigravity-oauth-token`, `settings.json`, `~/.gemini/config/config.json`, `~/.gemini/config/projects/`), SSH keys & config (`~/.ssh/id_ed25519*`, `id_rsa*`, `config`, `authorized_keys`, `known_hosts`), `rclone` configuration (`~/.config/rclone/rclone.conf`), and `~/piatoken.txt` into `$HOME/mptrade-secrets-backup.tar.gz`.
  - `backup remote`: Uploads encrypted snapshot to Storj.
  - `restore <tarball_or_folder>`: Rebuilds secrets, Antigravity credentials, SSH keys/config, `rclone` config, and project definitions, establishes virtualenv, installs python dependencies, and runs `systemd/install_prod.sh` (which auto-starts `antigravity-cli-daemon.service` authenticated as `prod-vm`).
- **Fleet Orchestration** (`orchestratorTrade/orchestrator.py`):
  - Supervised by `python_orchestrator.service` (`systemd`).
  - Reads `procs.conf` manifest to spawn all bots in isolated process groups.
  - Automatic restart with exponential backoff on crash loops.
  - Hot-reload on config updates (`config.env`, `procs.conf`, `instruments.conf`).
- **Services & Crontab Installation** (`systemd/install_prod.sh`):
  - Installs `python_orchestrator.service`, `pia.service`, `piavpn.service`.
  - Installs `/usr/local/sbin/trading-admin` and validates `/etc/sudoers.d/trading`.
  - Installs DNS drop-in `resolved-20-trading-cache.conf` (tunnel `wgpia0` owns DNS resolution).
  - Installs direct gateway routing drop-in `netplan-99-force-gateway.yaml`.
  - Installs logrotate cap for PIA daemon debug log (`logrotate-pia-daemon.conf`).
  - Installs user-level `antigravity-cli-daemon.service` along with its drop-ins (`antigravity-cli-daemon.service.d/override.conf` waiting on VPN readiness via `wait_for_vpn.sh`, and `antigravity-cli-daemon.service.d/store-fix.conf` running `orchestratorOS/admin/agy_store_fix.py pre/post` to keep the Web UI conversation summary store synchronized), plus the isolated headless bot wrapper `~/.local/bin/agy-bot` (`systemd/agy-bot`, using `~/agy_bot_home` so automated bot prompts never overwrite the human Web UI store), and enables user lingering (`loginctl enable-linger`).
  - Sets up dynamic crontabs (`systemd/crontab.prod.txt` and `systemd/crontab.root.prod.txt`) including `manage_agy.sh`, `autonomous_ai_reconciler.py`, `refresh_dev.sh` (every 30m PROD $\rightarrow$ DEV sync), and `trigger_backtest_dev.sh` (02:00 / 14:00 backtest runs on DEV).
  - DEV machine (`192.168.0.138`) crontab (`systemd/crontab.dev.txt`) maintains automated continuous CI regression testing (`dev_ci_watcher.sh` & `run_dev_regression.sh`).
- **VPN Supervisor** (`orchestratorOS/livecheck/pia_supervisor.sh`):
  - Reads `PIA_USER`, `PIA_PASS`, `PIA_DIP_TOKEN_FRANKFURT`, and `PIA_DIP_TOKEN_BELGIUM` directly from `.env` (fail-fast).
  - Automatically authenticates `piactl`, registers both Dedicated IP tokens (Frankfurt Primary `192.109.159.105` + Belgium Fallback `85.122.194.79`), and configures port forwarding on startup.
  - Implements a multi-tier failover ladder: **Primary Frankfurt DIP** $\rightarrow$ **Fallback Belgium DIP** $\rightarrow$ **Dynamic Frankfurt (`de-frankfurt`)** $\rightarrow$ **ISP outage cooldown**, and automatically probes every 5 minutes (`RESTORE_INTERVAL=300`) while on a fallback region to fail back to Primary Frankfurt DIP as soon as it recovers.
  - Enforces MTU 1280 on uplink and IPv4 preference in `/etc/gai.conf`.
  - Provides instant diagnostic CLI via `--check`.
- **VPN Watchdog** (`orchestratorOS/livecheck/vpn_watchdog.sh`):
  - Runs in root cron (`*/5 * * * *`) to heal connection drops with an on-disk alert spool.

---

## 3. Manual Rebuild Pieces (Off-Repo Requirements)

### 1. Base OS Packages (Fresh Linux Install)

If rebuilding on a fresh Ubuntu/Debian minimal install:
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl net-tools wireguard logrotate
```

### 2. PIA VPN Daemon Installation

PIA is a proprietary client (`/opt/piavpn`). On a bare VM (unless restored from Proxmox PBS/vzdump):
```bash
curl -sSLO https://installers.privateinternetaccess.com/download/pia-linux-3.5.7-08120.run
chmod +x pia-linux-3.5.7-08120.run
sudo ./pia-linux-3.5.7-08120.run --extra-args "--no-sandbox"
```
*(All login credentials and Dedicated IP tokens will be restored automatically from `.env` by `pia_supervisor.sh`).*

### 3. Netplan Uplink (Direct to Router .1)

The VM must route directly to router `192.168.0.1` and not through the Proxmox host (`.2`).
`systemd/install_prod.sh` installs `systemd/netplan-99-force-gateway.yaml` to `/etc/netplan/99-force-gateway.yaml`.
It takes effect cleanly on reboot.

### 4. Binance API IP Whitelist

Both Dedicated IPs — Primary Frankfurt (`192.109.159.105`) and Fallback Belgium (`85.122.194.79`) — are whitelisted on all Binance API keys.
If PIA ever issues a new Dedicated IP token resulting in an IP change:
- Update the API key whitelist in the Binance web console.

### 5. Proxmox Hypervisor (192.168.0.2)

See `docs/PROXMOX_DR.md` for hypervisor network and firewall settings.
- VM 100 must run with **NIC firewall OFF (`firewall=0`)** so PIA's WireGuard TLS handshakes are never dropped.

### 6. Off-Site Backups Notice

Daily backups via `manage_backups.sh local` create `$HOME/mptrade-secrets-backup.tar.gz` and rotate 7 daily copies locally.
For complete disaster resistance against total VM/hypervisor storage loss:
- Maintain Proxmox VE backups (vzdump / PBS) of VM 100, OR
- Periodically copy `~/mptrade-secrets-backup.tar.gz` off-host (e.g., via `scp` or remote storage).

---

## 4. Full Disaster Recovery Rebuild Order

When rebuilding on a fresh VM from scratch:

```bash
# 0. On a bare VM, ensure base packages & PIA are installed (see Section 3.1 & 3.2)

# 1. Clone the repository
git clone https://github.com/mpredut/mptrade.git ~/mptrade
cd ~/mptrade

# 2. Restore secrets and state from backup tarball (or directory)
# (Automatically unpacks secrets, sets up venv, installs deps, renders units/crontabs, enables linger, and starts services)
./orchestratorOS/admin/manage_backups.sh restore ~/mptrade-secrets-backup.tar.gz

# 3. Reboot the VM (applies netplan force-gateway and starts all services cleanly)
sudo reboot
```

After reboot:
```bash
# 4. Verify system health
sudo /usr/local/sbin/trading-admin status
./orchestratorOS/livecheck/pia_supervisor.sh --check
```
