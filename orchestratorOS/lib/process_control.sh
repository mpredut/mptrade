#!/usr/bin/env bash
# Scoped process operations for manifest-driven launchers. Source, do not execute.
# Only Python processes owned by this user in the declared working directory match.
manifest_pids() {
    local pat="$1" dir="$2" pid exe
    dir="$(realpath -e "$dir")" || return 1
    while read -r pid; do
        [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$dir" ] || continue
        exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null)" || continue
        case "${exe##*/}" in python*) ;; *) continue;; esac
        printf '%s\n' "$pid"
    done < <(pgrep -u "$(id -u)" -f -- "$pat" || true)
}

stop_manifest_process() {
    local pat="$1" dir="$2" pid attempt alive
    local -a old=()
    mapfile -t old < <(manifest_pids "$pat" "$dir")
    [ "${#old[@]}" -gt 0 ] || return 0
    kill -TERM "${old[@]}" || return 1
    # Keep the captured PID set: the supervisor may already have spawned replacements.
    for ((attempt=0; attempt<20; attempt++)); do
        alive=0
        for pid in "${old[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 0 ] && return 0
        sleep 0.5
    done
    echo "Refusing replacement: $pat did not stop gracefully; inspect pending writes/orders." >&2
    return 1
}
