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

- **Services/cron/sshd/DNS drop-in**: `install_prod.sh` renders + installs
  `binance.service`, `pia.service`, `piavpn.service`, `binancedemon.service`, both
  crontabs, `sshd-20-trading.conf`, and the resolved drop-in
  `resolved-20-trading-cache.conf` (Global DNS empty -> the tunnel owns resolution). It
  also `systemctl restart systemd-resolved`.
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
host (`.2`); otherwise the addKey to the dedicated IP fails. This is a static default in a
netplan drop-in (`/etc/netplan/99-force-gateway.yaml`, root, 0600). It is NOT yet mirrored
in this repo — MIRROR PENDING (paste `sudo cat /etc/netplan/99-force-gateway.yaml`). The
required outcome is exactly:

```
default via 192.168.0.1 dev ens18 proto static onlink
```

CAUTION: `netplan apply` while PIA is connected wipes PIA's policy routing (`piavpnWgrt`)
and traffic starts leaving direct instead of through the tunnel. After any `netplan apply`,
run `piactl connect` (or `systemctl restart pia.service`) to rebuild it. On a clean reboot
the order is safe (netplan first, then `pia.service`).

### 2. Binance API IP whitelist — the dedicated IP

The DIP is whitelisted on the Binance API keys. If PIA hands out a NEW dedicated IP (a
re-added token can change it, e.g. `.86 -> .79`), **update the whitelist by hand** or every
signed request gets `-2015`. Nothing automates this. See `PIA.md`.

### 3. Proxmox host (192.168.0.2) — not this repo, but part of the path

- The host's wired uplink must be the default route; the wifi hotspot interface
  (`wlx...`) is commented out of `auto` in the host netplan/`/etc/network/interfaces` so a
  `linkdown` wifi default cannot shadow the wired one.
- If the VM is ever routed through the host (hairpin), the host needs a MASQUERADE rule for
  the VM subnet (non-persistent by default). The direct-to-.1 netplan above avoids needing
  this. See `pia-uplink-proxmox` in memory.

## Full rebuild order

1. Proxmox host uplink healthy (wired default, wifi not shadowing it).
2. Fresh trading account + `git clone`; restore secrets/state from backup; install
   venv + PIA under the same paths (`README.md` steps 1-3).
3. Install PIA's DIP token (`~/piatoken_new.txt`, fallback `~/piatoken.txt`) and log in
   (`piactl login ~/pia.txt`).
4. netplan: install the direct-default drop-in (section 1) and `netplan apply` (PIA not yet
   up, so safe).
5. `sudo env TRADING_ROOT="$PWD" TRADING_USER="$(id -un)" systemd/install_prod.sh` — this
   renders/installs the units (incl. the MTU `ExecStartPre`), the DNS drop-in and cron, and
   restarts `piavpn`/`pia`/`binance`.
6. Confirm the Binance whitelist matches `piactl get vpnip` (section 2).
7. Run the green checklist above.

On the CURRENTLY running box everything in "Automated" is already deployed and live; the
only genuinely manual survivors are the netplan drop-in (section 1) and the Binance
whitelist (section 2).
