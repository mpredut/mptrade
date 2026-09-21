# DNS resilience for the Hyperliquid bot (disaster recovery)

## Problem
`hl_dca_bot` (Hyperliquid) intermittently fails to resolve **`api.hyperliquid.xyz`**
("Failed to resolve" / "Temporary failure in name resolution", Errno -3) during PIA VPN
or resolver blips. Kraken (`api.kraken.com`) and Binance are unaffected — it is that one
endpoint. Each failure makes the bot skip one price/manage tick until it recovers, which
matters when a position is on a trailing stop.

Two independent layers defend against it. **Recreate BOTH on a PROD rebuild.**

## Layer 1 — local DNS cache (systemd-resolved)
The box resolves via systemd-resolved (`/etc/resolv.conf` -> `nameserver 127.0.0.53`).
`/etc/systemd/resolved.conf`:

```ini
[Resolve]
Cache=yes
StaleRetentionSec=4h          # serve a STALE cached answer when the upstream DNS fails
FallbackDNS=1.1.1.1 8.8.8.8 1.0.0.1
```

Apply: `sudo systemctl restart systemd-resolved`
Verify: `resolvectl statistics` (cache active) and `resolvectl query api.hyperliquid.xyz`.
(Configured on PROD since 2026-08-21.)

## Layer 2 — code-level connect-retry (`hyperliquid/hl_client.py`, `_force_timeout`)
The Hyperliquid SDK's `requests` session mounts an `HTTPAdapter` with
`Retry(total=3, connect=3, read=0, status=0, backoff_factor=0.4)`:

- retries **only connection-establishment failures** (DNS / connect), which happen
  BEFORE the request is sent — so a retry can never double-submit an order;
- `read`/`status` stay **0** so a POST that may already have reached the exchange is
  never replayed;
- applied to both the Info (reads) and Exchange (orders) sessions.

Guarded by `tests/test_hl_client_timeout.py::test_connect_only_retries_are_mounted_for_dns_blips`.
No sudo, no config — it ships with the code, so a `git pull` restores it.

## Why both
The cache serves stale answers on an upstream failure and cuts the number of lookups,
but it cannot help when the resolver itself is momentarily unreachable (e.g. during a VPN
reconnect); the code-retry then recovers within the same tick. Together they shrink the
blind windows to near zero. Neither changes trading logic.
