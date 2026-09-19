#!/bin/bash
# git_autodeploy.sh -- optional auto-deploy, part of disaster recovery.
#
# From the ROOT crontab (systemd/crontab.root.prod.txt) it checks whether origin/<branch>
# moved (e.g. a push from WSL). Depending on the mode it either just reports what it WOULD
# do (shadow), or fast-forwards the local repo and restarts the fleet (binance.service) so
# the role=fleet processes come back on the new code. It does NOT reboot and does NOT touch
# PIA (the tunnel stays up); the role=bot processes reload on their next supervise cycle.
#
# MODE lives in a LOCAL, gitignored config so toggling it never dirties the tree or needs a
# commit:
#     cp autodeploy.local.conf.example autodeploy.local.conf     # then set AUTODEPLOY_MODE
#   off    -- do nothing.
#   shadow -- (DEFAULT) detect a moved origin, log + alert what it would do; apply NOTHING.
#   on     -- pull the fast-forward and `systemctl restart binance.service`.
#
# Guards (never clobber, never loop):
#   - refuses if the working tree is dirty (uncommitted local changes) -> alert;
#   - refuses if origin/<branch> is not a clean fast-forward of HEAD (divergence) -> alert;
#   - retries the fetch (GitHub is flaky through the PIA tunnel) and does nothing on failure;
#   - in `on`, records the deployed SHA + a cooldown so a bad state cannot loop-restart.
#
# Runs as ROOT (needs systemctl); every git command runs as the repo owner via runuser.
set -u
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"   # cron has a minimal PATH

ROOT="$(cd "$(dirname "$0")" && pwd)"
OWNER="$(stat -c %U "$ROOT")"
STATE_DIR="/var/lib/git_autodeploy"
LAST_MARK="$STATE_DIR/last_deploy"
SHADOW_MARK="$STATE_DIR/shadow_seen"

# --- local config (gitignored; default SHADOW = observe only) ---
AUTODEPLOY_MODE=shadow
AUTODEPLOY_BRANCH=main
AUTODEPLOY_COOLDOWN=900          # min seconds between fleet restarts for the same target
LOCAL_CONF="$ROOT/autodeploy.local.conf"
[ -r "$LOCAL_CONF" ] && . "$LOCAL_CONF"
case "${AUTODEPLOY_MODE:-shadow}" in
    off) exit 0 ;;
    shadow|on) ;;
    *) AUTODEPLOY_MODE=shadow ;;
esac
BRANCH="${AUTODEPLOY_BRANCH:-main}"
COOLDOWN="${AUTODEPLOY_COOLDOWN:-900}"
mkdir -p "$STATE_DIR" 2>/dev/null

log() { echo "$(date '+%F %T') $*"; }
g()   { runuser -u "$OWNER" -- git -C "$ROOT" "$@"; }   # git as the repo owner

alert() {  # best-effort ntfy; delivery failure is fine (informational)
    local topic
    topic=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '" ')
    [ -z "$topic" ] && topic=$(grep -hs -m1 '^NTFY_TOPIC=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '" ')
    [ -n "$topic" ] && curl --fail-with-body -sS -m 10 --retry 1 \
        -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
}

# Fetch with retry (GitHub over the tunnel is intermittently flaky).
fetched=0
for _i in 1 2 3; do g fetch --quiet origin "$BRANCH" 2>/dev/null && { fetched=1; break; }; sleep 5; done
[ "$fetched" = 1 ] || { log "fetch of origin/$BRANCH failed after retries; retrying next run"; exit 0; }

local_sha="$(g rev-parse HEAD 2>/dev/null)"
remote_sha="$(g rev-parse "origin/$BRANCH" 2>/dev/null)"
[ -n "$local_sha" ] && [ -n "$remote_sha" ] || { log "cannot resolve SHAs; skipping"; exit 0; }
[ "$local_sha" = "$remote_sha" ] && exit 0     # up to date

# GUARD: clean working tree (a dirty tree would be clobbered by the pull).
if [ -n "$(g status --porcelain 2>/dev/null)" ]; then
    log "REFUSED: working tree is dirty"
    alert "autodeploy SKIPPED ($(hostname))" \
        "origin/$BRANCH moved but the working tree has uncommitted changes; NOT deploying. Resolve by hand."
    exit 0
fi
# GUARD: fast-forward only (origin must descend from HEAD; no divergence / history rewrite).
if [ "$(g merge-base HEAD "origin/$BRANCH" 2>/dev/null)" != "$local_sha" ]; then
    log "REFUSED: origin/$BRANCH is not a fast-forward of HEAD (diverged)"
    alert "autodeploy SKIPPED ($(hostname))" \
        "origin/$BRANCH diverged from local HEAD (not a fast-forward); NOT deploying. Resolve by hand."
    exit 0
fi

# SHADOW: report what we WOULD do, deduped so we do not alert every run for the same target.
if [ "$AUTODEPLOY_MODE" = shadow ]; then
    log "SHADOW: origin/$BRANCH=$remote_sha (HEAD=$local_sha) -- WOULD pull + restart binance.service. Not applied."
    seen=""; [ -f "$SHADOW_MARK" ] && seen="$(cat "$SHADOW_MARK" 2>/dev/null)"
    if [ "$seen" != "$remote_sha" ]; then
        echo "$remote_sha" > "$SHADOW_MARK"
        alert "autodeploy SHADOW ($(hostname))" \
            "origin/$BRANCH -> ${remote_sha:0:9} is ready. Set AUTODEPLOY_MODE=on in autodeploy.local.conf to apply (pull + restart the fleet)."
    fi
    exit 0
fi

# --- mode=on: apply ---
# GUARD: anti-loop -- do not redeploy the same target within the cooldown.
if [ -f "$LAST_MARK" ]; then
    read -r last_sha last_ts < "$LAST_MARK" 2>/dev/null || true
    if [ "${last_sha:-}" = "$remote_sha" ] && [ $(( $(date +%s) - ${last_ts:-0} )) -lt "$COOLDOWN" ]; then
        log "already deployed $remote_sha within the ${COOLDOWN}s cooldown; skipping"
        exit 0
    fi
fi
if ! g pull --ff-only --quiet origin "$BRANCH" 2>/dev/null; then
    log "git pull --ff-only failed"
    alert "autodeploy FAILED ($(hostname))" "git pull --ff-only origin/$BRANCH failed; fleet NOT restarted."
    exit 0
fi
printf '%s %s\n' "$remote_sha" "$(date +%s)" > "$LAST_MARK"
log "deployed $remote_sha; restarting binance.service (role=fleet reloads; role=bot reload on next supervise)"
systemctl restart binance.service >/dev/null 2>&1
alert "autodeploy ($(hostname))" \
    "Pulled $BRANCH -> ${remote_sha:0:9} and restarted the fleet (binance.service). PIA untouched, no reboot."
