# PIA VPN — runbook

The Binance bots must leave through PIA's **dedicated** IP, because that is the
address whitelisted on the API keys. If the tunnel drops, or comes up on a
different IP, Binance answers `-2015` to every signed request. This document is
what you need to know when PIA misbehaves.

## Useful commands

`piactl` talks to the daemon over a socket. **Always wrap it in `timeout`**: when
the daemon is wedged, `piactl` blocks forever.

```bash
timeout 8 piactl get connectionstate   # Disconnected | Connecting | Connected | ...
timeout 8 piactl get vpnip             # the tunnel's exit IP   <-- this is what matters
timeout 8 piactl get pubip             # the PHYSICAL link's IP <-- not a bug, see below
timeout 8 piactl get region            # the SELECTED region
timeout 8 piactl get regions           # the AVAILABLE regions (the dedicated one shows here)
timeout 8 piactl get portforward
piactl background enable               # mandatory on a server, see pitfalls
piactl connect / piactl disconnect
piactl dedicatedip add "$HOME/piatoken.txt" # Frankfurt DIP token (or piatoken_frankfurt.txt)
piactl dedicatedip add "$HOME/piatoken_belgia.txt" # Belgium DIP token
piactl login "$HOME/pia.txt"          # account login: username on line 1, password on line 2
piactl set debuglogging true           # creates /opt/piavpn/var/daemon.log (root)
```

### Credentials & Token File Conventions

| File | Alternative Name | Contents / Purpose |
|------|------------------|-------------------|
| `~/pia.txt` | `~/pia_credentials.txt` | PIA account credentials (line 1: `pXXXXXXX`, line 2: password) |
| `~/piatoken.txt` | `~/piatoken_frankfurt.txt` | Frankfurt Dedicated IP token (`192.109.159.105`) |
| `~/piatoken_belgia.txt` | `~/piatoken_belgium.txt` | Belgium Dedicated IP token |

All scripts (`pia_supervisor.sh`, `vpn_watchdog.sh`, `manage_backups.sh`) recognize both primary and alternative names.

#### Optional Centralization in `.env`
You can centrally store PIA secrets in `.env`:
```bash
PIA_USER=pXXXXXXX
PIA_PASS=YourPiaPassword
PIA_DIP_TOKEN_FRANKFURT=YourFrankfurtDedicatedIpToken
PIA_DIP_TOKEN_BELGIUM=YourBelgiumDedicatedIpToken
```
If `.env` contains these variables, `pia_supervisor.sh` will auto-generate `~/pia.txt` and `~/piatoken*.txt` with mode `0600` on startup if they do not already exist on disk.

Quick diagnosis without touching anything:

```bash
./orchestratorOS/livecheck/pia_supervisor.sh --check
```

The REAL check of where traffic leaves (do not trust `connectionstate`):

```bash
curl -s --interface wgpia0 https://ipinfo.io/ip     # must be the dedicated IP
```

## Pitfalls that cost us 34 days

**`pubip` is NOT the IP you leave through.** It is the public IP of the physical
link (the ISP line). It stays on the home address even when the tunnel is perfectly
healthy. For the effective exit address use `vpnip` or `curl --interface tun0`.

**`piactl connect` is SILENTLY ignored without a GUI.** No graphical client runs on
the server, so without `piactl background enable` the daemon accepts the
`connectVPN` RPC, reports success, and stays calmly `Disconnected`.

**`connectionstate = Connected` can lie.** During a flap PIA reports Connected while
tun0/DNS/HTTPS are already dead. The probe has to be bound explicitly to `tun0`.

**Logging out deletes the dedicated IP.** The DIP registration is tied to the
account session. After `piactl logout` (or a reset), `piactl get regions` no longer
contains any `dedicated-*` line, while the region stays set to one that does not
exist -> `Unknown region`, the connection fails, and if anything falls back to
`auto` the tunnel comes up on a pool IP, which means `-2015` at Binance.

**A re-added DIP token can return a DIFFERENT IP.** On 1 Sep 2026: `85.122.194.86`
-> `85.122.194.79`. That is why `pia_start.sh` **no longer hardcodes the region** and
derives it from `piactl get regions` instead. When the IP changes, **the whitelist on
the Binance account must be updated by hand** — nothing automates that.

**The killswitch must be off for automatic direct ISP fallback.** In production, `PIA_KILLSWITCH=off`
is configured in `config.env` and written to `/opt/piavpn/etc/settings.json`. If all VPN regions fail,
`pia_supervisor.sh` disconnects the tunnel so outbound traffic immediately falls back to the physical
uplink (`ens18`) without packet drop, allowing the trading fleet to remain operational.

## Failover Ladder & Exponential Backoff Cooldown

`pia_supervisor.sh` enforces a deterministic region failover ladder:

1. **Primary**: Frankfurt Dedicated IP (`dedicated-de-frankfurt-*`)
2. **Fallback 1**: Belgium Dedicated IP (`dedicated-belgium-*`)
3. **Fallback 2**: Dynamic Frankfurt IP (`de-frankfurt`)
4. **Fallback 3 (All VPN failed)**: Direct Romanian ISP uplink (`ens18`) via `pia disconnect`.

### Exponential Backoff Model
When all VPN regions fail, the supervisor enters a direct Romanian ISP cooldown:
- Starts at **60 seconds (1 minute)**.
- Doubles on each failed recovery attempt: `60s -> 120s -> 240s -> 480s -> 960s -> 1920s -> 3840s -> 7200s`.
- Capped at **7200 seconds (2 hours)** and stays there indefinitely until VPN recovery.
- Resets back to **60 seconds** upon successful VPN recovery (`vpn_healthy`).
- During cooldown, local DNS cache is pre-warmed every 30 seconds for all critical endpoints (Binance, Kraken, Hyperliquid, Trading 212, NTFY).
- NTFY alerts are dispatched for every ladder transition and recovery.

**`python_orchestrator.service` has `Requires=pia.service`.** Because `pia_supervisor.sh` remains active
and manages cooldown internally rather than crashing, `pia.service` stays healthy, preventing systemd
restart thrashing and allowing continuous fleet trading. Check the processes: `./healthcheck.sh --check`.

## The 1 Sep 2026 incident (why pia_selfheal.sh exists)

The full chain, verified in the logs:

1. `pia-daemon` crashed on `ExceptionHandler::GenerateDump sys_pipe failed: Too
   many open files` and stayed behind as a ghost process with a `[pia-openvpn]
   <defunct>` child. Every `piactl` call returned `Timed out after 5 sec`.
2. With no tun0, the killswitch cut all outbound traffic.
3. `pia.service` went into a restart loop — it reached **6975** restarts.
4. The fleet never started at all (see `Requires=` above).
5. **No alert got out**: `healthcheck.sh` and `deadman_switch.sh` were doing their
   job, but every alert path goes through ntfy.sh, that is, through exactly the
   internet that was missing (`278 x "EROARE curl"` in `logs/deadman.log`).

The repair: restarting `piavpn.service` (plus `kill -9` on the daemon) brought the
internet back, but connecting still failed with `AUTH_FAILED` — a PIA account
problem. After a fresh login and a new DIP token, everything came back on
`85.122.194.79`.

## pia_selfheal.sh

Runs from the **root crontab** every 5 minutes (`systemd/crontab.root.prod.txt`) —
as an unprivileged account it would fail in precisely the scenario it exists for, because rungs 3-4
need `systemctl` and `kill` on pia-daemon.

The escalation ladder stops at the first rung that works, and every rung is followed
by a real HTTPS probe through `tun0`:

| # | Action |
|---|--------|
| 1 | `background enable` + `connect` |
| 2 | `disconnect` + `connect` |
| 3 | stop services -> `kill -9` daemon -> start -> re-register DIP -> `set region` -> `connect` |
| 4 | download and reinstall PIA (24h cooldown) |

If `piactl` does not answer at all, rungs 1-2 are skipped from the start.

**Alerts go into a spool on disk** (`/var/lib/pia_selfheal/alert_spool`) and are
delivered retroactively once connectivity returns. That closes the hole from the
incident: the story of the outage reaches the phone even though not a single packet
could leave during it. The real outage length is measured from
`/var/lib/pia_selfheal/outage_since`.

**Boot grace:** during the first 5 minutes of uptime the script does not escalate,
because the VPN is legitimately down while `pia.service` (Restart=always) negotiates
the tunnel.

Tunable through environment variables: `PIA_BOOT_GRACE`, `PIA_CONNECT_WAIT`,
`PIA_FALLBACK_REGION`, `PIA_DIP_TOKEN`, `PIA_VERSION`, `PIA_REINSTALL_COOLDOWN`,
`PIA_SPOOL_MAX_BYTES`.

About rung 4: the PIA installer refuses to run as root and may want interactive
escalation, so it **can legitimately fail** — in that case the script sends an alert
asking for a manual reinstall rather than pretending it succeeded. It never executes
the downloaded file if it is under 20 MB or does not start with `#!` (an HTML error
page never gets run). The URL must be **versioned**: `pia-linux-latest.run` answers
403, and the site's "latest" endpoint returns HTML.

## What happens on reboot

- `piavpn.service`, `pia.service`, `python_orchestrator.service`, `cron.service` — all `enabled`.
- SSH is **socket-activated** on Ubuntu 24.04: `ssh.service` shows as `disabled`, but
  `ssh.socket` is `enabled` and listens on 32238. That is not a problem.
- Both crontabs (the trading account and root) survive the reboot.
- The non-fleet bots have no `@reboot` entry; `healthcheck.sh --supervise` brings
  them back within 3 minutes.
- `pia_selfheal.sh` does not intervene during the first 5 minutes (boot grace).
