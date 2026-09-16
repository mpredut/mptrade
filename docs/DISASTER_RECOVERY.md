# Disaster Recovery — rebuilding the trading server from scratch

Everything **versioned** (the code, the `procs.conf` manifest, the systemd units,
`requirements.txt`, `systemd/crontab.prod.txt`, the scripts) comes from git. The only thing
that is **NOT in git** (and must not be) are the **secrets** — you keep those in a separate,
off-machine backup.

## What is in git vs what you keep (off-machine)

| In git (automatic on `git clone`) | NOT in git — a separate backup |
|---|---|
| the code, `procs.conf` and the scripts (flota_start, bots_start, healthcheck, restore.sh) | `.env` (root, hyperliquid, kraken, 212trading) |
| `systemd/*.service` (binance, pia) | `keys/apikeys.py` (the Binance keys) |
| `requirements.txt` (the venv dependencies) | `keys/ed25519_*.pem` (the Kraken keys) |
| `systemd/crontab.prod.txt` plus `systemd/install_prod.sh` | (optional) bot state: `.state_*.json`, `cachedb/` |

⚠ **`hyperliquid/.env` holds the HL agent-wallet key** — losing it means you can no longer
sign HL orders. The secrets backup is CRITICAL.

## ⚠ How do the secrets reach an EMPTY VM? (the chicken-and-egg)

A new VM **has no keys at all** and does NOT "pull from WSL" by itself. YOU initiate the
restore (from the dev box, or on the VM using the DR seed). The direction is either **the VM
downloads from Storj** or **you push** the backup to the VM. Never "the VM pulls from WSL".

**🔑 THE DR SEED** — keep these 3 SEPARATELY (a password manager, or paper), not only in the backup:
1. the git repository URL (`git@github.com:mpredut/mptrade.git`) plus GitHub access (an SSH key or an HTTPS token)
2. the **Storj access grant**
3. the **rclone crypt password** (to decrypt the backup)

With those three, a completely empty VM can rebuild itself. (If you put them ONLY in the
backup you get the chicken-and-egg: you need them to download the backup that contains them.)

## Rebuilding on a new machine (Ubuntu) — the steps

```bash
# 0. dependencies
sudo apt update && sudo apt install -y git python3 python3-venv curl unzip

# 1. the code — clone into ANY folder, as ANY user; restore.sh and install_prod.sh derive
#    the checkout path and the account automatically (nothing is hardcoded to a name/user).
#    (HTTPS+token if you have no GitHub key on the VM; or add the key)
git clone git@github.com:mpredut/mptrade.git ~/mptrade && cd ~/mptrade

# 2. BRING the secrets backup onto the VM — choose A or B:
#   (A) STORJ (recommended, no dev box needed): configure rclone with the access grant and the
#       crypt password (from the DR seed), then download and decrypt:
#         ~/bin/rclone copyto storj-crypt:mptrade-secrets-backup.tar.gz ~/bk.tar.gz
#         mkdir bk && tar xzf ~/bk.tar.gz -C bk
#   (B) PUSH from the dev box (interim): on the DEV BOX run
#         scp ~/mptrade-secrets-backup.tar.gz user@NEW_VM:/tmp/
#       then on the VM: mkdir bk && tar xzf /tmp/mptrade-secrets-backup.tar.gz -C bk

# 3. ONE COMMAND — it rebuilds everything (secrets + venv + systemd + cron):
./restore.sh bk/mptrade-secrets-backup

# 4. PIA/VPN (once): install the PIA client and log in, then:
sudo systemctl start pia binance

# 5. verify
./healthcheck.sh --check        # every process should be 'ok'
```

`restore.sh` does: restore the secrets -> create `myenv` and `pip install -r requirements.txt`
-> install the systemd units (and enable them) -> install the crontab. After `systemctl start`,
the **fleet** starts through systemd and the **bots** through the `healthcheck.sh --supervise`
cron (within 5 minutes).

## How to (re)make the secrets backup

Run it on the live machine (it creates the folder and the tar, without touching git):

```bash
~/mptrade/backup_secrets.sh            # -> ~/mptrade-secrets-backup/ plus .tar.gz
# then copy the tarball OFF-machine (USB, private cloud, another machine)
```

Remember: the secrets NEVER go into git (they are in `.gitignore`). Keeping the backup safe
and off-machine is your responsibility.

## A local copy on WSL (interim, until Storj) — a Windows task
The server rebuilds the backup daily (cron 03:30, `backup_secrets.sh`) and keeps **history:
the last 7 dated tarballs** (`mptrade-secrets-backup-YYYYMMDD.tar.gz`) alongside the stable
path `mptrade-secrets-backup.tar.gz` (latest) — so a corruption that makes it into the backup
no longer overwrites the single good copy. A Windows task pulls the latest at 04:00 (WSL does
not reach the server, only Windows does): it downloads **locally** first
(`%USERPROFILE%\mptrade-secrets-backup.tar.gz`, which works even with WSL stopped), then copies
it into WSL as well. The script is versioned:
[`../windows/pull-binance-backup.ps1`](../windows/pull-binance-backup.ps1) (keyless).

Setting it up on a new Windows machine (once):
```powershell
# 1. an SSH key (without a passphrase, for automation) and add the public part on the server
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_binance" -N '""' -C binance-backup-pull -q
# add the contents of id_binance.pub to ~/.ssh/authorized_keys on the server (once, using the password)
# 2. copy the script from the repository onto Windows (adjust the paths if they differ)
copy <repo>\windows\pull-binance-backup.ps1 C:\Users\<user>\pull-binance-backup.ps1
# 3. a daily 04:00 task, with catch-up if the PC was switched off
$a=New-ScheduledTaskAction -Execute powershell.exe -Argument '-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Users\<user>\pull-binance-backup.ps1'
$t=New-ScheduledTaskTrigger -Daily -At 4:00am
$s=New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName BinanceBackupPull -Action $a -Trigger $t -Settings $s -Force
```
The PRIVATE key `id_binance` does NOT go into git (put it in the secrets backup, or regenerate
it and re-add the public part on the server). Once Storj is running, this task becomes optional.

## On reboot (without a rebuild) — everything comes back on its own
- `binance.service` is `enabled`, so systemd starts the fleet after the VPN.
- The crontab persists on disk, so `healthcheck --supervise` (cron */5) starts the bots within 5 minutes.
- Nothing to do by hand.
