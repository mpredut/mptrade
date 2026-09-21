# pull-binance-backup.ps1 — pull the secrets backup from the server to the dev box.
# INTERIM (until off-site Storj). Run by the Windows task "BinanceBackupPull"
# (see docs/DISASTER_RECOVERY.md). SSH-KEY AUTH (no password in the file).
#
# Strategy: download FIRST to a LOCAL Windows path (works even with WSL stopped),
# then also copy into WSL if available. That way the backup always lands somewhere.
# The paths are specific to the dev box — adjust if they differ.
$ErrorActionPreference = 'Stop'
$key      = "$env:USERPROFILE\.ssh\id_binance"
$serverUser = if ($env:TRADING_USER) { $env:TRADING_USER } else { 'predut' }
$wslUser    = if ($env:WSL_USER) { $env:WSL_USER } else { 'mariusp' }
$src        = "${serverUser}@192.168.0.144:/home/${serverUser}/mptrade-secrets-backup.tar.gz"
$dstLocal   = "$env:USERPROFILE\mptrade-secrets-backup.tar.gz"
$dstWsl     = "\\wsl.localhost\ubuntu-24.04\home\${wslUser}\mptrade-secrets-backup.tar.gz"
$stamp      = Get-Date -Format 'yyyy-MM-dd HH:mm'

# 1) download locally (does not depend on WSL)
scp -i "$key" -P 32238 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$src" "$dstLocal"
if ($LASTEXITCODE -ne 0) {
    Write-Host ("{0} FAILED (scp exit {1}) - is the server reachable (VPN)?" -f $stamp, $LASTEXITCODE)
    exit 1
}
$sz = [math]::Round((Get-Item $dstLocal).Length / 1MB, 1)
Write-Host ("{0} OK - downloaded locally ({1} MB): {2}" -f $stamp, $sz, $dstLocal)

# 2) also copy into WSL, if it is running (best-effort; the local copy remains anyway)
try {
    Copy-Item -Path $dstLocal -Destination $dstWsl -Force
    Write-Host ("{0} OK - also copied into WSL: {1}" -f $stamp, $dstWsl)
} catch {
    Write-Host ("{0} WSL unavailable (stopped?) - keeping only the local Windows copy" -f $stamp)
}
