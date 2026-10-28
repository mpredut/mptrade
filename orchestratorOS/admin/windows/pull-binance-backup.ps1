# pull-binance-backup.ps1 — pull the secrets & cache backup from PROD to the local Windows + WSL box,
# then unpack it in WSL Ubuntu-24.04 and sync live cachedb/ files.
# Run by the Windows Scheduled Task "BinanceBackupPull" at 03:45 AM (see docs/DISASTER_RECOVERY.md).
$ErrorActionPreference = 'Stop'
$key        = "$env:USERPROFILE\.ssh\id_binance"
$serverUser = if ($env:TRADING_USER) { $env:TRADING_USER } else { 'predut' }
$wslUser    = if ($env:WSL_USER) { $env:WSL_USER } else { 'mariusp' }
$src        = "${serverUser}@192.168.0.144:/home/${serverUser}/mptrade-secrets-backup.tar.gz"
$dstLocal   = "$env:USERPROFILE\mptrade-secrets-backup.tar.gz"
$dstWsl     = "\\wsl.localhost\Ubuntu-24.04\home\${wslUser}\mptrade-secrets-backup.tar.gz"
$stamp      = Get-Date -Format 'yyyy-MM-dd HH:mm'

# 1) Download locally on Windows (does not depend on WSL)
scp -i "$key" -P 32238 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$src" "$dstLocal"
if ($LASTEXITCODE -ne 0) {
    Write-Host ("{0} FAILED (scp exit {1}) - is the server reachable?" -f $stamp, $LASTEXITCODE)
    exit 1
}
$sz = [math]::Round((Get-Item $dstLocal).Length / 1MB, 1)
Write-Host ("{0} OK - downloaded locally ({1} MB): {2}" -f $stamp, $sz, $dstLocal)

# 2) Copy into WSL Ubuntu-24.04 and unpack into /home/mariusp/mptrade + sync live cachedb
try {
    Copy-Item -Path $dstLocal -Destination $dstWsl -Force
    Write-Host ("{0} OK - copied into WSL: {1}" -f $stamp, $dstWsl)
    wsl.exe -d Ubuntu-24.04 -u $wslUser -e bash -lc "/home/${wslUser}/mptrade/offline/runners/sync_wsl_cache.sh --unpack-backup >> /home/${wslUser}/mptrade/logs/sync_wsl_cache.log 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("{0} WARN - WSL sync_wsl_cache.sh exited with code {1}" -f $stamp, $LASTEXITCODE)
    } else {
        Write-Host ("{0} OK - unpacked and synced cachedb in WSL (/home/{1}/mptrade)" -f $stamp, $wslUser)
    }
} catch {
    Write-Host ("{0} WARN - WSL copy/unpack issue: {1}" -f $stamp, $_.Exception.Message)
}
