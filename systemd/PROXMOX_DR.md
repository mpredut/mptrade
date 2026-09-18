# Proxmox host — disaster recovery

Config snapshot of the hypervisor that runs the trading VM, so the node + its VMs can be
rebuilt. Complements `DISASTER_RECOVERY.md` (which covers the VM's networking/VPN/fleet).
Captured 2026-09-19 from the node's root shell.

**Secrets are redacted** (`<REDACTED>`). VM DISK DATA is not here — it comes from backups
(vzdump / PBS), restored separately. This file is the CONFIG needed to recreate the shell.

## Node

- Proxmox `pve-manager/8.4.19`, kernel `6.8.12-4-pve`.
- Host name `server`, management IP `192.168.0.2/24`, gateway `192.168.0.1`.
- Admin access: SSH as `noroot` (root SSH disabled); privileged work via the web UI node
  Shell (root). `noroot` has no passwordless sudo.

## Host network — `/etc/network/interfaces`

Wired uplink only; the USB wifi is defined but intentionally **not** `auto` so it cannot
grab the default route (a `linkdown` wifi default once shadowed the wired one).

```
auto lo
iface lo inet loopback

iface enp2s0 inet manual

auto vmbr0
iface vmbr0 inet static
        address 192.168.0.2/24
        netmask 255.255.255.0
        gateway 192.168.0.1
        bridge-ports enp2s0
        bridge-stp off
        bridge-fd 0

# USB wifi fallback -- DEFINED but NOT `auto` (must stay this way; do not re-enable `auto`).
# iface wlx20235199b316 inet dhcp
#         wpa-ssid HOSPOT
#         wpa-psk <REDACTED>
source /etc/network/interfaces.d/*
```

Note: on the live host the wifi stanza is present as `iface ... inet dhcp` (SSID `HOSPOT`,
psk redacted) with its `auto` line commented — keep it non-auto. An older static wifi
block (SSID `mp`) is fully commented out.

## Firewall (Proxmox, enabled + running)

Datacenter master ON with default-deny inbound; each VM re-states it and allows only SSH
(from the LAN) + ICMP. Outbound is left open (the fleet needs many endpoints and PIA's
kill switch is the real egress control).

`/etc/pve/firewall/cluster.fw`:
```
[OPTIONS]
policy_in: DROP
enable: 1
```

`/etc/pve/firewall/100.fw` and `101.fw` (identical):
```
[OPTIONS]
policy_in: DROP
enable: 1

[RULES]
IN ACCEPT -p tcp -dport 32238 -source 192.168.0.0/24   # SSH management, LAN only
IN ACCEPT -p icmp                                       # ping
# outbound is open by default -> PIA/Binance/DNS work; replies are ESTABLISHED,RELATED
```

Both VMs' NIC has `firewall=1`, so these rules are actually enforced (the per-VM chain is
`tapNi0-IN`). WITHOUT the `firewall=1` NIC checkbox the VM firewall does nothing, so it is
easy to think a VM is protected when it is not. To harden a VM: add the SSH allow rule
FIRST, then tick `firewall=1` on the NIC, with the noVNC console open as a fallback --
ticking it with only `policy_in: DROP` and no SSH rule locks you out.

## Virtual machines

Two VMs, both `onboot: 1`, both on `vmbr0`, disks on ZFS `local-zfs` (`rpool/data`).

- **VM 100 = the TRADING VM** (`192.168.0.144`, MAC `BC:24:11:2D:2F:23`): 4 cores,
  6144 MB (balloon 4096), `x86-64-v2-AES`, `scsi0 local-zfs:vm-100-disk-0 32G`
  (`virtio-scsi-single`, iothread), `net0 virtio ...,firewall=1`, `ostype l26`. Runs the
  live fleet + PIA (see `DISASTER_RECOVERY.md`).
- **VM 101 = `clone-of-100`** (MAC `BC:24:11:05:FA:22`): 4 cores, 4048 MB, `scsi0
  local-zfs:vm-101-disk-0 32G`, `net0 ...,firewall=1`. Runs ONLY scheduled backtests via
  cron -- NOT the live fleet and NOT PIA. This matters: if 101 ever started the fleet/PIA
  it would fight VM 100 for the single dedicated IP `.79` (flapping) and double live
  trades on the same keys. Keep 101 backtest-only.

## Storage — `/etc/pve/storage.cfg`

```
dir: local
        path /var/lib/vz
        content iso,vztmpl,backup

zfspool: local-zfs
        pool rpool/data
        sparse
        content images,rootdir
```

Usage at capture: `local` 48G (~20%), `local-zfs` 72G (~47%). VM disks live on
`local-zfs` (`rpool/data`); ISOs/backups on `local` (`/var/lib/vz`).

## iptables on the host — no manual rules to reproduce

The earlier uplink troubleshooting (VM hairpinning through the host / wifi uplink) left
non-persistent `nat POSTROUTING` MASQUERADE + `FORWARD` rules and `net.ipv4.ip_forward=1`.
Once the VM routes DIRECTLY to `.1` (bridged, not routed through the host) these are unused
and were removed; they are NOT needed on a rebuild. Do NOT re-add MASQUERADE/forwarding for
the VM -- it is a bridge port on `vmbr0`, its traffic is L2-forwarded, not NATed by the
host. The only firewall state that matters is the PVE firewall above (PVE manages its own
`PVEFW-*` chains).

## Rebuild order (bare node)

1. Install Proxmox VE 8.4.x; set host name `server`, root password.
2. `/etc/network/interfaces` as above: `vmbr0` static `.2` gw `.1` bridging `enp2s0`; wifi
   defined but NOT `auto`.
3. Firewall: datacenter `cluster.fw` (enable 1, policy_in DROP); per-VM `.fw` with the
   SSH-LAN + ICMP rules; ensure each VM NIC has `firewall=1`.
4. Storage: `local` (dir) + `local-zfs` (ZFS pool `rpool/data`).
5. Recreate VMs 100/101 from the definitions above (or restore from vzdump/PBS backup),
   with `onboot: 1`.
6. Inside VM 100, follow `DISASTER_RECOVERY.md` to bring the fleet up on the dedicated IP.
7. Confirm VM 101 runs only backtests (no fleet/PIA).
