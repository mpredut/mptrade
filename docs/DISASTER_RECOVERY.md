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
  - `backup local`: Takes a full local DR snapshot of all repo secrets (`.env`, `config.env`, `212trading/.env`, `kraken/.env`, `hyperliquid/.env`, `autodeploy.local.conf`), caches (`cachedb/`), state files, Antigravity CLI credentials & project configs (`~/.gemini/antigravity-cli/antigravity-oauth-token`, `settings.json`, `~/.gemini/config/config.json`, `~/.gemini/config/projects/`), SSH keys & config (`~/.ssh/id_ed25519*`, `id_rsa*`, `config`, `authorized_keys`, `known_hosts`), `rclone` configuration (`~/.config/rclone/rclone.conf`, auto-rendered from `STORJ_ACCESS_KEY` / `STORJ_SECRET_KEY` / `STORJ_ENDPOINT` in `.env`), and `~/piatoken.txt` into `$HOME/mptrade-secrets-backup.tar.gz` (plus 7 daily rotated copies).
  - `backup remote` (scheduled daily at `03:30 AM` on PROD): Runs `backup local` first, then builds a dedicated **raw price-only archive** (`$HOME/mptrade-price-data.tar.gz`, containing exclusively `cachedb/cache_365_price_*.jsonl`, `cachedb/cache_24_price_*.json`, `cachedb/cache_sparse_price_*.jsonl`, `cachedb/cache_watcher_all_price.json`, and their `.meta` files — zero `.env` secrets or order/trade state) and uploads both `mptrade-price-data.tar.gz` and `mptrade-price-data-YYYYMMDD.tar.gz` (retaining last 7 days) to the Storj S3 bucket `storj:mptrade-price-data/`.
  - `pull-prices`: Downloads `storj:mptrade-price-data/mptrade-price-data.tar.gz` from Storj and unpacks the raw high-density price archives directly into `cachedb/`.
  - `restore <tarball_or_folder>`: Rebuilds secrets, Antigravity credentials, SSH keys/config, `rclone` config, and project definitions, automatically pulls `mptrade-price-data.tar.gz` from Storj if `cachedb/cache_365_price_*.jsonl` is missing locally, establishes virtualenv, installs python dependencies, and runs `systemd/install_prod.sh` (which auto-starts `antigravity-cli-daemon.service` authenticated as `prod-vm`).
- **Off-Host Daily Pull & Continuous Cache Mirror (DEV + Windows/WSL)**:
  - **DEV (`192.168.0.138`)**: Synced every 30 minutes from PROD via `offline/runners/sync_cache_to.sh dev` (git pull + `rsync` of `cachedb/`).
  - **Windows + WSL (`Ubuntu-24.04`, `/home/mariusp/mptrade`)**:
    - Daily at `03:45 AM`, the Windows Scheduled Task `BinanceBackupPull` executes `C:\Users\mariusp\pull-binance-backup.ps1` (`orchestratorOS/admin/windows/pull-binance-backup.ps1`), which downloads `~/mptrade-secrets-backup.tar.gz` from PROD (`192.168.0.144:32238`) to `C:\Users\mariusp\mptrade-secrets-backup.tar.gz`, copies it into WSL (`/home/mariusp/mptrade-secrets-backup.tar.gz`), and invokes `offline/runners/sync_cache_to.sh wsl --unpack-backup` to unpack the archive into `/home/mariusp/mptrade-secrets-backup` and `/home/mariusp/mptrade`.
    - Every 30 minutes (`*/30 * * * *`), WSL cron executes `offline/runners/sync_cache_to.sh wsl`, which `rsync`s `cachedb/` from `mptrade-prod` and `*.npz` + `dev_regression_status.json` from `mptrade-dev` into `/home/mariusp/mptrade/cachedb/`.
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
  - Sets up dynamic crontabs (`systemd/crontab.prod.txt` and `systemd/crontab.root.prod.txt`) including `manage_agy.sh`, `autonomous_ai_reconciler.py`, `sync_cache_to.sh dev` (every 30m PROD $\rightarrow$ DEV sync), and `run_backtest_cycle.sh remote` (02:00 / 14:00 backtest runs on DEV).
  - DEV machine (`192.168.0.138`) crontab (`systemd/crontab.dev.txt`) maintains automated continuous CI regression testing (`run_dev_regression.sh watch` & `run_dev_regression.sh run`).
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

### 6. Off-Site Backups & Multi-Tier DR Topology

Daily backups at `03:30 AM` via `manage_backups.sh remote` maintain a 3-tier disaster recovery architecture:
1. **PROD Local Full DR Archive**: `$HOME/mptrade-secrets-backup.tar.gz` (+ 7 daily copies on PROD) containing all `.env` secrets, SSH/Antigravity/rclone credentials, and `cachedb/`.
2. **Windows + WSL (`Ubuntu-24.04`) Off-Host Full Mirror**: Pulled daily at `03:45 AM` over SSH by Windows Scheduled Task `BinanceBackupPull` (`pull-binance-backup.ps1`) to `C:\Users\mariusp\mptrade-secrets-backup.tar.gz` and `/home/mariusp/mptrade-secrets-backup.tar.gz`, unpacked into `/home/mariusp/mptrade`, and continuously synced every 30 minutes via `offline/runners/sync_cache_to.sh wsl`.
3. **Storj Cloud Price Archive (`storj:mptrade-price-data/`)**: Holds `mptrade-price-data.tar.gz` (+ 7 daily dated copies) containing **only** raw collected high-density price series (`cache_365_price_*.jsonl`, `cache_24_price_*.json`, `cache_sparse_price_*.jsonl`, `cache_watcher_all_price.json`) with zero secrets exposed in cloud storage.

---

## 4. Full Disaster Recovery Rebuild Order

When rebuilding on a fresh VM from scratch:

```bash
# 0. On a bare VM, ensure base packages & PIA are installed (see Section 3.1 & 3.2)

# 1. Clone the repository
git clone https://github.com/mpredut/mptrade.git ~/mptrade
cd ~/mptrade

# 2. Restore secrets and state from backup tarball (copied from Windows/WSL or local backup)
# (Automatically unpacks secrets + cachedb, pulls 365d price archives from Storj if missing,
#  sets up venv, installs deps, renders units/crontabs, enables linger, and starts services)
./orchestratorOS/admin/manage_backups.sh restore ~/mptrade-secrets-backup.tar.gz

# (Optional) To explicitly pull the latest raw price archive from Storj at any time:
./orchestratorOS/admin/manage_backups.sh pull-prices

# 3. Reboot the VM (applies netplan force-gateway and starts all services cleanly)
sudo reboot
```

After reboot:
```bash
# 4. Verify system health
sudo /usr/local/sbin/trading-admin status
./orchestratorOS/livecheck/pia_supervisor.sh --check
```
