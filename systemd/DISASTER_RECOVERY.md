# Disaster recovery — full VM/network rebuild

How to rebuild the trading VM's networking + VPN state from scratch so the fleet leaves
through the PIA **dedicated IP** again. This complements `README.md` (services/cron) and
`PIA.md` (the VPN runbook). It exists because several pieces are NOT in `git` and a couple
live on the Proxmox host, so a bare `install_prod.sh` is not enough on its own.

`DNS_RESILIENCE.md` covers the Hyperliquid-specific DNS cache/retry and stays valid.

## Target state (what "correct" looks like)

Snapshot of a healthy box (2026-09-18), for reference when verifying a rebuild:

```
ens18    192.168.0.144/24  MTU 1280   gateway 192.168.0.1 (direct, static onlink)
wgpia0   10.79.x.x/32      PIA WireGuard tunnel, exit 85.122.194.79 (the DIP)
default via 192.168.0.1 dev ens18 proto static onlink        # base uplink (netplan)
DNS: systemd-resolved stub 127.0.0.53; Global EMPTY;
     wgpia0 -> 10.0.0.243 (Domains ~., owns all public lookups); ens18 -> 192.168.0.1
IPv6: enabled but deprioritized (gai.conf); the kill switch blocks it (tunnel is IPv4-only)
sysctl: net.ipv4.ip_forward=0  rp_filter=2  net.ipv6.conf.all.disable_ipv6=0
```

Green checklist (all must hold):

```bash
piactl get connectionstate            # Connected
piactl get vpnip                      # 85.122.194.79  (== the Binance-whitelisted DIP)
curl -4 -s https://api.ipify.org      # 85.122.194.79
for i in $(seq 10); do getent hosts api.binance.com >/dev/null && echo ok; done   # 10/10
curl -4 -s -o /dev/null -w '%{http_code}\n' https://api.binance.com/api/v3/time   # 200
systemctl is-active pia binance       # active / active
./healthcheck.sh --check              # fleet processes alive (is-active can lie)
```

## What PIA builds automatically — do NOT reproduce it by hand

When `pia connect` succeeds, the daemon creates all of this itself; it is runtime state,
not config to restore. Listed only so nobody tries to "fix" it manually:

- routing tables `piavpnWgrt` (default dev wgpia0) and `piavpnFwdrt` (default dev wgpia0 +
  `blackhole default` = the kill switch);
- policy rules: `50: suppress_prefixlength 1`, `70: fwmark 0x3214 -> piavpnFwdrt`,
  `102: not fwmark 0x3213 -> piavpnWgrt`;
- the nftables kill-switch ruleset;
- the `wgpia0` interface and its address.

`ip rule show` / `ip route show table all` will show these once connected. If they are
missing, the fix is `piactl connect` (or `systemctl restart pia.service`), never a manual
`ip rule add`. A `netplan apply` while PIA is up WIPES these — see the netplan note below.

## Automated (git + install_prod.sh + pia_start.sh)

A `git pull` + `sudo systemd/install_prod.sh` restores everything here:

- **Services/cron/sshd/DNS drop-in/netplan**: `install_prod.sh` renders + installs
  `binance.service`, `pia.service`, `piavpn.service`, `binancedemon.service`, both
  crontabs, `sshd-20-trading.conf`, the resolved drop-in `resolved-20-trading-cache.conf`
  (Global DNS empty -> the tunnel owns resolution), the direct-default netplan file
  `netplan-99-force-gateway.yaml` (installed, not applied -- see section 1), and the
  `logrotate-pia-daemon.conf` cap on PIA's debug log (`/opt/piavpn/var/daemon.log`, kept ON
  in production). It also `systemctl restart systemd-resolved`.
- **PIA connection logic** (`pia_start.sh`, run by `pia.service`): WireGuard protocol,
  `allowlan true` (kill switch must not cut LAN/SSH), derive the dedicated region from
  `piactl get regions` (never hardcoded), connect, health-probe loop. It also, as root,
  before connecting:
  - **clamps the uplink MTU** (`ens18` -> 1280; `PIA_UPLINK_IF`/`PIA_UPLINK_MTU`);
  - **prefers IPv4** by adding `precedence ::ffff:0:0/96 100` to `/etc/gai.conf` (the
    tunnel is IPv4-only, so IPv6-first lookups hit the kill switch).
- **PIA tunnel MTU 1200** (`settings.json`): enforced by `pia_settings_mtu.sh`, wired as an
  `ExecStartPre=-` of `piavpn.service` (idempotent, non-fatal). `piactl` has no `mtu`
  setting, so it lives in the daemon's `settings.json`, which the daemon reads only at
  startup. This is normally a no-op (settings.json is persistent) — it matters after a PIA
  reinstall (self-heal rung 4), which resets it to defaults.
- **Self-healing** (`pia_selfheal.sh`, root crontab): reconnect -> restart daemon ->
  re-register DIP -> reinstall ladder, with an on-disk alert spool.

## Manual / off-repo — the pieces a rebuild MUST redo by hand

### 1. netplan — route the VM DIRECT to the gateway (.1)

The VM must send its uplink straight to `192.168.0.1`, not hairpin through the Proxmox
host (`.2`); otherwise the addKey to the dedicated IP fails. The LAN DHCP has handed out
`.2` as the gateway, so a drop-in overrides it: `use-routes: false` (ignore the DHCP
gateway) + a static default via `.1`.

This is now MIRRORED at `systemd/netplan-99-force-gateway.yaml` and **installed** (0600
root) by `install_prod.sh`, on top of cloud-init's `50-cloud-init.yaml` (`dhcp4: true`,
regenerated automatically). Result once active:

```
default via 192.168.0.1 dev ens18 proto static onlink
```

`install_prod.sh` installs the file but deliberately does NOT `netplan apply` it, because
that would wipe PIA's live policy routing (`piavpnWgrt`) and send traffic direct instead of
through the tunnel. To activate: **reboot** (clean order netplan -> pia.service), or run
`sudo netplan apply` and then `sudo systemctl restart pia.service` to rebuild the tunnel
routing. On the current box the file is already installed and active.

### 2. Binance API IP whitelist — the dedicated IP

The DIP is whitelisted on the Binance API keys. If PIA hands out a NEW dedicated IP (a
re-added token can change it, e.g. `.86 -> .79`), **update the whitelist by hand** or every
signed request gets `-2015`. Nothing automates this. See `PIA.md`.

### 3. Proxmox host (192.168.0.2) — see systemd/PROXMOX_DR.md

The hypervisor's config (host network, firewall, VM definitions, storage) is documented for
rebuild in `PROXMOX_DR.md`. Key points for THIS VM: the host's wired uplink (`vmbr0` static
`.2` -> `.1` over `enp2s0`) must be the default route, and the USB wifi must stay NOT `auto`
so a `linkdown` wifi default cannot shadow it. The VM is a bridge port on `vmbr0` and routes
DIRECTLY to `.1`, so the host needs NO MASQUERADE/forwarding for it — the earlier
hairpin-era iptables cruft was removed. See also `pia-uplink-proxmox` in memory.

## Full rebuild order

1. Proxmox host uplink healthy (wired default, wifi not shadowing it).
2. Fresh trading account + `git clone`; restore secrets/state from backup; install
   venv + PIA under the same paths (`README.md` steps 1-3).
3. Install PIA's DIP token (`~/piatoken.txt`) and log in
   (`piactl login ~/pia.txt`).
4. `sudo env TRADING_ROOT="$PWD" TRADING_USER="$(id -un)" systemd/install_prod.sh` — renders
   and installs the units (incl. the tunnel-MTU `ExecStartPre`), the DNS drop-in, the
   direct-default netplan file, and cron; restarts `piavpn`/`pia`/`binance`.
5. **Reboot** so netplan applies the direct route (`.1`) at boot and `pia.service` then
   connects on the dedicated IP with the correct routing. (Without a reboot: `sudo netplan
   apply`, then `sudo systemctl restart pia.service` to rebuild the tunnel routing.)
6. Confirm the Binance whitelist matches `piactl get vpnip` (section 2).
7. Run the green checklist above.

On the CURRENTLY running box everything in "Automated" is already deployed and live; the
only genuinely manual survivors are the netplan drop-in (section 1) and the Binance
whitelist (section 2).
