# Disaster Recovery — Full VM/Network and Fleet Rebuild

How to rebuild the trading VM, its networking, VPN state, and fleet orchestrators from scratch.
This complements `README.md` (services/cron) and `PIA.md` (VPN details).

---

## 1. Target State (What "Correct" Looks Like)

Snapshot of a healthy box:

```text
ens18    192.168.0.144/24  MTU 1280   gateway 192.168.0.1 (direct router uplink, static onlink)
wgpia0   10.79.x.x/32      PIA WireGuard tunnel, exit 192.109.159.105 (Dedicated IP)
default via 192.168.0.1 dev ens18 proto static onlink        # base uplink (netplan)
DNS: systemd-resolved stub 127.0.0.53; Global EMPTY;
     wgpia0 -> 10.0.0.243 (Domains ~., owns all public lookups); ens18 -> 192.168.0.1
IPv6: enabled but deprioritized (gai.conf); the kill switch blocks it (tunnel is IPv4-only)
sysctl: net.ipv4.ip_forward=0  rp_filter=2  net.ipv6.conf.all.disable_ipv6=0
```

### Verification Checklist (All Must Hold)

```bash
piactl get connectionstate            # Connected
piactl get vpnip                      # 192.109.159.105 (Binance-whitelisted Dedicated IP)
curl -4 -s https://api.ipify.org      # 192.109.159.105
for i in $(seq 10); do getent hosts api.binance.com >/dev/null && echo ok; done   # 10/10
curl -4 -s -o /dev/null -w '%{http_code}\n' https://api.binance.com/api/v3/time   # 200
sudo /usr/local/sbin/trading-admin status                                          # active / active
./orchestratorOS/livecheck/pia_supervisor.sh --check                              # Health Probe: OK
```

---

## 2. Automated Components (Git + Backups + install_prod.sh)

The entire deployment is reproducible and path-agnostic:

- **Unified Backup & Restore** (`orchestratorOS/admin/manage_backups.sh`):
  - `backup local`: Takes a snapshot of all secrets (`.env`, `212trading/.env`), caches (`cachedb/`), and state files (`lock/trade_cooldown.json`) into `$HOME/mptrade-secrets-backup.tar.gz`. (All PIA credentials and DIP tokens are centralized in `.env`).
  - `backup remote`: Uploads encrypted snapshot to Storj.
  - `restore <tarball_or_folder>`: Rebuilds secrets, establishes virtualenv, installs python dependencies, and runs `systemd/install_prod.sh`.
- **Fleet Orchestration** (`orchestratorTrade/orchestrator.py`):
  - Supervised by `python_orchestrator.service` (`systemd`).
  - Reads `procs.conf` manifest to spawn all bots in isolated process groups.
  - Automatic restart with exponential backoff on crash loops.
  - Hot-reload on config updates (`config.env`, `procs.conf`, `instruments.conf`).
- **Services & Crontab Installation** (`systemd/install_prod.sh`):
  - Installs `python_orchestrator.service`, `pia.service`, `piavpn.service`.
  - Installs `/usr/local/sbin/trading-admin` and validates `/etc/sudoers.d/trading`.
  - Installs DNS drop-in `resolved-20-trading-cache.conf` (tunnel wgpia0 owns DNS resolution).
  - Installs direct gateway routing drop-in `netplan-99-force-gateway.yaml`.
  - Installs logrotate cap for PIA daemon debug log.
  - Installs user-level `antigravity-cli-daemon.service` and enables user lingering (`loginctl enable-linger`).
  - Cleans up and eliminates any legacy `binance.service`.
  - Sets up dynamic crontab (`systemd/crontab.prod.txt` including `manage_agy.sh`) for trading user and root.
- **VPN Supervisor** (`orchestratorOS/livecheck/pia_supervisor.sh`):
  - Reads `PIA_USER`, `PIA_PASS`, and `PIA_DIP_TOKEN_FRANKFURT` directly from `.env` (fail-fast, no persistent token files on disk).
  - Automatically authenticates `piactl`, registers Dedicated IP token, and configures port forwarding on startup.
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

The Dedicated IP (`192.109.159.105`) is whitelisted on all Binance API keys.
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
