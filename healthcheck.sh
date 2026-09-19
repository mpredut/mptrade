#!/bin/bash
# healthcheck.sh — supervision plus a consolidated report for the bots/fleet (HL/Kraken/T212).
#
# THE SINGLE SOURCE OF TRUTH: procs.conf (also read by bots_start.sh plus flota_start.sh).
# There are NO hardcoded process lists here any more. DOUBLE detection: absence (pgrep) AND
# a hang (the process is alive but the log is frozen, a heartbeat on mtime) — it replaces dn_watchdog.sh.
#   --supervise  (cron */5): restarts dead/frozen bots (role=bot) with backoff; the fleet is alert-only.
#   --alert      : alert only if something is missing or hung (no restart).
#   --check      : READ-ONLY preview (what --supervise would do) — safe, touches nothing.
#   (no arg)     : the full report (processes plus the HL/Kraken/T212 accounts).
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env_common.sh"
MANIFEST="$ROOT/procs.conf"
HLPY="$PYTHON_BIN"
now=$(date +%s)
PIA_CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
VPN_PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-8}"
# PIA's tunnel interface: wgpia0 under WireGuard (tun0 was OpenVPN). Keep in sync with
# pia_start.sh / pia_selfheal.sh -- probing the wrong interface reports a PERMANENT false
# VPN fault (spamming alerts and masking a real outage).
VPN_IF="${PIA_VPN_IF:-wgpia0}"

pia() { timeout "$PIA_CLI_TIMEOUT" piactl "$@" 2>/dev/null | tr -d '\r'; }

# Read-only financial-intent visibility. The index has no write/submit/cancel API.
intent_state() {
    "$HLPY" - "$ROOT" <<'PY' 2>/dev/null
import sys
sys.path.insert(0, sys.argv[1])
from active_intents import build_active_intent_index
result = build_active_intent_index(sys.argv[1])
unknown = sum(row.get("status") == "unknown" for row in result["intents"])
print(f"{'error' if result['errors'] else 'ok'}|{len(result['intents'])}|{unknown}|{len(result['errors'])}")
PY
}

# Return a concise reason instead of only Connected/DOWN. Every external probe is
# bounded, and HTTPS is explicitly bound to the tunnel ($VPN_IF) so it cannot escape via
# the ISP.
vpn_state() {
    [ "$(pia get connectionstate)" = "Connected" ] || { echo piactl; return; }
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' \
        || { echo "$VPN_IF"; return; }
    resolvectl query -i "$VPN_IF" api.binance.com >/dev/null 2>&1 \
        || { echo dns; return; }
    local binance_ip
    binance_ip=$(getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}')
    [ -n "$binance_ip" ] || binance_ip=$(resolvectl query api.binance.com 2>/dev/null \
        | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    curl -4 --interface "$VPN_IF" \
        --resolve "api.binance.com:443:$binance_ip" \
        --connect-timeout 4 --max-time "$VPN_PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || { echo https; return; }
    echo ok
}

push_ntfy() {
    local title="$1" body="$2"
    [ -n "${TOPIC:-}" ] || return 1
    curl --fail-with-body -sS -m 10 --retry 2 --retry-all-errors \
        -H "Title: $title" -d "$body" "https://ntfy.sh/$TOPIC" >/dev/null
}

# The state of one line: ok | absent | stopped | zombie | hung.
proc_state() {
    local pat="$1" dir="$2" hblog="$3" hbstale="$4"
    local pids states
    pids=$(pgrep -f "$pat") || { echo absent; return; }
    states=$(ps -o state= -p "$(echo "$pids" | paste -sd, -)" 2>/dev/null | tr -d ' ')
    echo "$states" | grep -q T && { echo stopped; return; }
    echo "$states" | grep -q Z && { echo zombie; return; }
    if [ -n "$hblog" ] && [ -n "$hbstale" ]; then
        local lp="$hblog"; case "$hblog" in /*) ;; *) lp="$dir/$hblog";; esac
        if [ -f "$lp" ]; then
            local age=$(( now - $(stat -c %Y "$lp") ))
            [ "$age" -ge "$hbstale" ] && { echo hung; return; }
        fi
    fi
    echo ok
}

# ===== --check MODE: a READ-ONLY preview (it touches nothing) ==============
if [ "$1" = "--check" ]; then
    echo "=== CHECK (read-only) $(date '+%H:%M:%S') — source: $MANIFEST ==="
    vpn=$(vpn_state)
    [ "$vpn" = ok ] && echo "  VPN              ok (piactl + $VPN_IF + DNS + Binance HTTPS)" \
        || echo "  VPN              FAULT ($vpn)"
    intents=$(intent_state || echo "error|0|0|1")
    IFS='|' read -r intent_ok intent_count intent_unknown intent_errors <<EOF
$intents
EOF
    printf '  %-16s %-6s active=%-4s unknown=%-4s errors=%s\n' \
        "INTENTS" "$intent_ok" "$intent_count" "$intent_unknown" "$intent_errors"
    while IFS='|' read -r pat dir cmd label hblog hbstale role; do
        [ -z "$pat" ] && continue
        case "$pat" in \#*) continue;; esac
        dir=$(eval echo "$dir")
        st=$(proc_state "$pat" "$dir" "$hblog" "$hbstale")
        extra=""
        if [ -n "$hblog" ]; then
            lp="$hblog"; case "$hblog" in /*) ;; *) lp="$dir/$hblog";; esac
            [ -f "$lp" ] && extra="(heartbeat ${hblog}: $(( now - $(stat -c %Y "$lp") ))s/${hbstale}s)"
        fi
        act="-"
        [ "$st" != ok ] && { [ "$role" = bot ] && act="RESTART" || act="alert"; }
        printf '  %-16s %-6s %-7s %-10s %s\n' "$label" "$role" "$st" "$act" "$extra"
    done < "$MANIFEST"
    exit 0
fi

# ===== MODE --alert: ntfy alert ONLY, if something is missing or hung ======
if [ "$1" = "--alert" ]; then
    missing=""
    vpn=$(vpn_state)
    [ "$vpn" != ok ] && missing="$missing VPN($vpn)"
    intents=$(intent_state || echo "error|0|0|1")
    IFS='|' read -r intent_ok intent_count intent_unknown intent_errors <<EOF
$intents
EOF
    [ "$intent_ok" != ok ] && missing="$missing INTENT_INDEX(errors=$intent_errors)"
    [ "$intent_unknown" -gt 0 ] && missing="$missing INTENTS_UNKNOWN($intent_unknown)"
    while IFS='|' read -r pat dir cmd label hblog hbstale role; do
        [ -z "$pat" ] && continue
        case "$pat" in \#*) continue;; esac
        dir=$(eval echo "$dir")
        st=$(proc_state "$pat" "$dir" "$hblog" "$hbstale")
        [ "$st" != ok ] && missing="$missing $label($st)"
    done < "$MANIFEST"
    if [ -n "$missing" ]; then
        TOPIC=$(grep -hs NTFY_TOPIC "$ROOT/kraken/config.env" "$ROOT/config.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '" ')
        push_ntfy "Server processes" \
            "Dead/hung:$missing  -> check (./bots_start.sh / flota_start)" \
            || echo "$(date '+%H:%M') ALERT NOT DELIVERED: an ntfy HTTP or network error"
        echo "$(date '+%H:%M') ALERT: $missing"
    else
        echo "$(date '+%H:%M') OK (all processes are running)"
    fi
    exit 0
fi

# ===== --supervise MODE (cron */5): restart dead/frozen bots plus a fleet alert =====
if [ "$1" = "--supervise" ]; then
    # Supervision can start live bots. Enable it explicitly in the production
    # scheduler instead of guessing the environment from a username or path.
    [ "${TRADING_SUPERVISE_ENABLED:-false}" = true ] || {
        echo "$(date '+%H:%M') supervision disabled; set TRADING_SUPERVISE_ENABLED=true"
        exit 0
    }
    exec 8>/tmp/binance_supervise.lock
    flock -n 8 || { echo "$(date '+%H:%M') supervise is already running — skipping (anti-duplication)"; exit 0; }
    SUP=/tmp/binance_sup; mkdir -p "$SUP"; WINDOW=1800; MAX=3
    TOPIC=$(grep -hs NTFY_TOPIC "$ROOT/kraken/config.env" "$ROOT/config.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '" ')
    push(){ push_ntfy "$1" "$2"; }
    alert_miss=""
    vpn=$(vpn_state)
    [ "$vpn" != ok ] && alert_miss=" VPN($vpn)"
    intents=$(intent_state || echo "error|0|0|1")
    IFS='|' read -r intent_ok intent_count intent_unknown intent_errors <<EOF
$intents
EOF
    [ "$intent_ok" != ok ] && alert_miss="$alert_miss INTENT_INDEX(errors=$intent_errors)"
    [ "$intent_unknown" -gt 0 ] && alert_miss="$alert_miss INTENTS_UNKNOWN($intent_unknown)"
    while IFS='|' read -r pat dir cmd label hblog hbstale role; do
        [ -z "$pat" ] && continue
        case "$pat" in \#*) continue;; esac
        dir=$(eval echo "$dir")
        st=$(proc_state "$pat" "$dir" "$hblog" "$hbstale")
        if [ "$st" = ok ]; then
            [ "$role" = bot ] && rm -f "$SUP/$label" "$SUP/$label.esc"   # healthy -> reset backoff
            continue
        fi
        if [ "$role" != bot ]; then          # fleet: alert only (flota_start owns it)
            alert_miss="$alert_miss $label($st)"
            continue
        fi
        # role=bot, state absent|hung
        if [ "$st" = hung ]; then
            echo "$(date '+%H:%M') $label HUNG (an old heartbeat) -> kill"
            pkill -f "$pat" 2>/dev/null; sleep 2; pkill -9 -f "$pat" 2>/dev/null
        fi
        cnt=0; ws=$now
        [ -f "$SUP/$label" ] && read -r cnt ws < "$SUP/$label"
        [ $((now - ws)) -gt $WINDOW ] && { cnt=0; ws=$now; }   # a new window
        if [ "$cnt" -ge "$MAX" ]; then
            [ -f "$SUP/$label.esc" ] || { push "Bot in a CRASH LOOP" "$label ($st) ${cnt}x in 30min — no longer restarting, manual intervention needed"; touch "$SUP/$label.esc"; }
            echo "$(date '+%H:%M') $label CRASH LOOP (not restarting)"; continue
        fi
        # 8>&- : the started bot does NOT inherit fd 8 (the supervise lock) -> no lock leak
        # (otherwise later --supervise runs find the lock held by a bot and skip forever).
        # Background the subshell with closed stdin/redirected stdout so healthcheck does not block.
        ( cd "$dir" && eval "$cmd" ) 8>&- </dev/null >/dev/null 2>&1 &
        cnt=$((cnt + 1)); echo "$cnt $ws" > "$SUP/$label"; rm -f "$SUP/$label.esc"
        push "Bot restarted" "$label ($st) -> RESTARTED (attempt $cnt/$MAX)"
        echo "$(date '+%H:%M') $label RESTARTED ($st, attempt $cnt)"
    done < "$MANIFEST"
    [ -n "$alert_miss" ] && { push "Processes to check" "Dead/hung (not restarted from here):$alert_miss"; echo "$(date '+%H:%M') fleet alert:$alert_miss"; }
    [ -z "$alert_miss" ] && echo "$(date '+%H:%M') supervise: fleet OK"
    exit 0
fi

echo "============ HEALTHCHECK $(date '+%Y-%m-%d %H:%M') ============"
echo "=== PROCESSES (etime = how long they have run) ==="
ps -eo etime,args | grep -E "dn_bot|kraken_bot|kraken_xstock_watch|t212_bot|ipo.py|trailing_stop|cacheManager|priceAnalysis|tradeall|rtrade|monitortrades|market_alerts|run_price_monitor|assetguardian" | grep -v grep

echo "=== ACTIVE INTENTS (READ-ONLY INDEX) ==="
intent_state

echo "=== HYPERLIQUID DN ==="
( cd "$ROOT/hyperliquid" && "$HLPY" dn_bot.py --status 2>&1 | grep -E "SPOT|PERP|DELTA|FUNDING|LICHIDARE|COLATERAL" )

echo "=== KRAKEN ==="
( cd "$ROOT/kraken" && python3 - <<'PY' 2>/dev/null
import sys, os; sys.path.insert(0, ".")
from common import load_dotenv
load_dotenv("config.env"); load_dotenv("config.env")
from kraken_client import KrakenClient
try:
    from kraken_xstock_watch import yahoo_last
except Exception:
    yahoo_last = lambda s: None
c = KrakenClient(osconfig.environ.get("KRAKEN_API_KEY"), osconfig.environ.get("KRAKEN_API_SECRET"))
b = c.balance()
print("  cash ZUSD %.0f + USDC %.0f | HYPE %s @ %s" % (
    float(b.get("ZUSD", 0)), float(b.get("USDC", 0)), b.get("HYPE"), c.last_price("HYPEUSD")))
oo = c.open_orders()
print("  orders: %d %s" % (len(oo), [o.get("descr", {}).get("order") for o in oo.values()]))
sp = float(b.get("SPCXx.T", 0))
if sp:
    px = yahoo_last("SPCX") or 0
    print("  SPCXx.T %.4f @ %.2f -> $%.0f" % (sp, px, sp * px))
PY
)

echo "=== T212 ==="
( cd "$ROOT/212trading" && python3 - <<'PY' 2>/dev/null
import sys, os, time; sys.path.insert(0, ".")
from ipo_common import load_dotenv
load_dotenv("config.env")
from t212_client import T212Client
c = T212Client(osconfig.environ["T212_API_KEY"], osconfig.environ.get("T212_API_SECRET"), env="live")
pf = None
for _ in range(3):
    pf = c.get_portfolio()
    if pf:
        break
    time.sleep(2)
for p in (pf or []):
    if any(s in p.get("ticker", "") for s in ("NVDA", "SPCX")):
        print("  %s qty %s avg %s price %s P&L %s" % (
            p.get("ticker"), p.get("quantity"), p.get("averagePrice"),
            p.get("currentPrice"), p.get("ppl")))
if not pf:
    print("  portfolio unavailable")
PY
)
echo "============ END ============"
