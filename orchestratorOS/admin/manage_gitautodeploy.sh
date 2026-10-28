#!/bin/bash
# manage_gitautodeploy.sh — optional auto-deploy, part of disaster recovery.
# Runs from the ROOT crontab (systemd/crontab.root.prod.txt); executes git as the repo owner via runuser.
# Modes (in autodeploy.local.conf):
#   off    -- do nothing
#   shadow -- (DEFAULT) detect a moved origin, log + alert what it would do; apply NOTHING
#   on     -- pull the fast-forward and `systemctl restart python_orchestrator.service`
set -u
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$ROOT/orchestratorOS/lib/os_notify.sh"

OWNER="$(stat -c %U "$ROOT")"
STATE_DIR="/var/lib/git_autodeploy"
[ "$(id -u)" -ne 0 ] && [ ! -w "$STATE_DIR" ] && STATE_DIR="$ROOT/logs/.git_autodeploy"
LAST_MARK="$STATE_DIR/last_deploy"
SHADOW_MARK="$STATE_DIR/shadow_seen"
last_sha=""; last_ts=0
[ -f "$LAST_MARK" ] && { read -r last_sha last_ts < "$LAST_MARK" 2>/dev/null || true; }

AUTODEPLOY_MODE=shadow
AUTODEPLOY_BRANCH=main
AUTODEPLOY_COOLDOWN=900
AUTODEPLOY_RESTART_ON_LOCAL=false
LOCAL_CONF="$ROOT/autodeploy.local.conf"
[ -r "$LOCAL_CONF" ] && . "$LOCAL_CONF"
case "${AUTODEPLOY_MODE:-shadow}" in
    off) exit 0 ;;
    shadow|on) ;;
    *) AUTODEPLOY_MODE=shadow ;;
esac
BRANCH="${AUTODEPLOY_BRANCH:-main}"
COOLDOWN="${AUTODEPLOY_COOLDOWN:-900}"
RESTART_ON_LOCAL="${AUTODEPLOY_RESTART_ON_LOCAL:-false}"
mkdir -p "$STATE_DIR" 2>/dev/null

log() { echo "$(date '+%F %T') $*"; }
g() {
    if [ "$(id -u)" -eq 0 ]; then
        runuser -u "$OWNER" -- git -C "$ROOT" "$@"
    else
        git -C "$ROOT" "$@"
    fi
}
alert() { send_os_ntfy "$1" "$2" default || true; }

shadow_report_and_exit() {  # $1=action_desc
    log "SHADOW: origin/$BRANCH=$remote_sha (HEAD=$local_sha) -- WOULD $1. Not applied."
    local seen=""; [ -f "$SHADOW_MARK" ] && seen="$(cat "$SHADOW_MARK" 2>/dev/null)"
    if [ "$seen" != "$remote_sha" ]; then
        echo "$remote_sha" > "$SHADOW_MARK"
        alert "autodeploy SHADOW ($(hostname))" \
            "origin/$BRANCH -> ${remote_sha:0:9} is ready. Set AUTODEPLOY_MODE=on in autodeploy.local.conf to apply ($1)."
    fi
    exit 0
}

fetched=0
for _i in 1 2 3; do g fetch --quiet origin "$BRANCH" 2>/dev/null && { fetched=1; break; }; sleep 5; done
[ "$fetched" = 1 ] || { log "fetch of origin/$BRANCH failed after retries; retrying next run"; exit 0; }

local_sha="$(g rev-parse HEAD 2>/dev/null)"
remote_sha="$(g rev-parse "origin/$BRANCH" 2>/dev/null)"
[ -n "$local_sha" ] && [ -n "$remote_sha" ] || { log "cannot resolve SHAs; skipping"; exit 0; }

if [ -n "$(g status --porcelain -uno 2>/dev/null)" ]; then
    log "REFUSED: tracked working tree is dirty"
    exit 0
fi

if [ "$local_sha" = "$remote_sha" ]; then
    case "${RESTART_ON_LOCAL:-false}" in
        1|[tT][rR][uU][eE]|[yY][eE][sS]|[oO][nN])
            [ -n "$last_sha" ] && [ "$last_sha" = "$local_sha" ] && exit 0 ;;
        *) exit 0 ;;
    esac
fi

if [ "$local_sha" != "$remote_sha" ]; then
    if [ "$(g merge-base HEAD "origin/$BRANCH" 2>/dev/null)" != "$local_sha" ]; then
        log "REFUSED: origin/$BRANCH is not a fast-forward of HEAD (diverged)"
        exit 0
    fi
    [ "$AUTODEPLOY_MODE" = shadow ] && shadow_report_and_exit "pull + restart python_orchestrator.service"
    if ! g pull --ff-only --quiet origin "$BRANCH" 2>/dev/null; then
        log "git pull --ff-only failed"
        alert "autodeploy FAILED ($(hostname))" "git pull --ff-only origin/$BRANCH failed; fleet NOT restarted."
        exit 0
    fi
    g ls-files --stage 2>/dev/null | awk '$1 == "100755" {print $4}' | while IFS= read -r _f; do
        [ -f "$ROOT/$_f" ] && [ ! -x "$ROOT/$_f" ] && chmod +x "$ROOT/$_f" 2>/dev/null || true
    done
fi

[ "$AUTODEPLOY_MODE" = shadow ] && shadow_report_and_exit "restart python_orchestrator.service"

if [ -n "$last_sha" ] && [ "$last_sha" = "$remote_sha" ] && [ $(( $(date +%s) - ${last_ts:-0} )) -lt "$COOLDOWN" ]; then
    log "already deployed $remote_sha within the ${COOLDOWN}s cooldown; skipping"
    exit 0
fi

printf '%s %s\n' "$remote_sha" "$(date +%s)" > "$LAST_MARK"
log "deployed $remote_sha; restarting python_orchestrator.service (role=fleet reloads; role=bot reload on next supervise)"
systemctl restart python_orchestrator.service >/dev/null 2>&1
alert "autodeploy ($(hostname))" \
    "Deployed $BRANCH -> ${remote_sha:0:9} and restarted the fleet (python_orchestrator.service). PIA untouched, no reboot."
