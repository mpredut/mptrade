# pull-binance-backup.ps1 — trage backup-ul de secrete de pe server pe dev box.
# INTERIMAR (pana la Storj off-site). Rulat de task-ul Windows "BinanceBackupPull"
# (vezi docs/DISASTER_RECOVERY.md). AUTH CU CHEIE SSH (fara parola in fisier).
#
# Strategie: descarca INTAI intr-o cale LOCALA Windows (merge si cu WSL oprit),
# apoi copiaza si in WSL daca e disponibil. Asa backup-ul aterizeaza mereu undeva.
# Caile sunt specifice dev box-ului — ajusteaza daca difera.
$ErrorActionPreference = 'Stop'
$key      = "$env:USERPROFILE\.ssh\id_binance"
$src      = 'predut@192.168.0.144:/home/predut/mptrade-secrets-backup.tar.gz'
$dstLocal = "$env:USERPROFILE\mptrade-secrets-backup.tar.gz"
$dstWsl   = '\\wsl.localhost\ubuntu-24.04\home\mariusp\mptrade-secrets-backup.tar.gz'
$stamp    = Get-Date -Format 'yyyy-MM-dd HH:mm'

# 1) descarca local (nu depinde de WSL)
scp -i "$key" -P 32238 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$src" "$dstLocal"
if ($LASTEXITCODE -ne 0) {
    Write-Host ("{0} ESUAT (scp exit {1}) - e serverul accesibil (VPN)?" -f $stamp, $LASTEXITCODE)
    exit 1
}
$sz = [math]::Round((Get-Item $dstLocal).Length / 1MB, 1)
Write-Host ("{0} OK - descarcat local ({1} MB): {2}" -f $stamp, $sz, $dstLocal)

# 2) copiaza si in WSL, daca e pornit (best-effort; localul ramane oricum)
try {
    Copy-Item -Path $dstLocal -Destination $dstWsl -Force
    Write-Host ("{0} OK - copiat si in WSL: {1}" -f $stamp, $dstWsl)
} catch {
    Write-Host ("{0} WSL indisponibil (oprit?) - raman doar cu copia locala Windows" -f $stamp)
}
