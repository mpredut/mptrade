ssh -o BatchMode=yes mptrade-prod << 'INNER'
tail -n 20 ~/mptrade/logs/__main___.log 2>/dev/null
INNER
