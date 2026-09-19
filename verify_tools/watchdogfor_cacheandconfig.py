#!/usr/bin/env python3
"""Watch cache freshness and configuration changes in one short cron task.

Cache checks alert and may restart cacheManager when a cache becomes stale. Config checks
hash file contents and restart owner processes after real edits, guarded by an opt-in kill
switch. healthcheck supervision makes restarts respawn-safe. Per-cache thresholds and alert
cooldown distinguish fast, slow, and event-driven data sources.
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watchdog_common as wc       # infrastructura partajata: env, ntfy/email, state

_ROOT = wc.ROOT                                   # Repository root above verify_tools.
wc.load_env()

# Caches live in cachedb unless BINANCE_CACHE_DIR overrides it.
_CACHE_DIR = Path(os.environ.get("BINANCE_CACHE_DIR", _ROOT / "cachedb"))
STATE_FILE = _ROOT / ".watchdog_state.json"
STALE_MINUTES = wc.required_float_env("WATCHDOG_STALE_MINUTES")
COOLDOWN_MINUTES = wc.required_float_env("WATCHDOG_COOLDOWN_MINUTES")

# Auto-restart applies only when a fast cache written by cacheManager is genuinely stale.
# Cooldown and a rolling-window cap prevent loops; exceeding them requires manual action.
AUTO_RESTART = wc.required_bool_env("WATCHDOG_AUTO_RESTART")
AUTO_RESTART_COOLDOWN_MIN = wc.required_float_env("WATCHDOG_AUTO_RESTART_COOLDOWN_MIN")
AUTO_RESTART_MAX = wc.required_int_env("WATCHDOG_AUTO_RESTART_MAX")
AUTO_RESTART_WINDOW_H = wc.required_float_env("WATCHDOG_AUTO_RESTART_WINDOW_H")
AUTO_RESTART_TARGET = "python cacheManager.py"   # pgrep/pkill pattern; fleet supervision respawns it.

# Config watch restarts an owner only after a content-hash change, avoiding false restarts
# from touching a file. Targets in procs.conf are respawn-safe under healthcheck supervision.
CONFIG_RESTART = wc.required_bool_env("WATCHDOG_CONFIG_RESTART")
CONFIG_RESTART_COOLDOWN_MIN = wc.required_float_env("WATCHDOG_CONFIG_COOLDOWN_MIN")
CONFIG_RESTART_MAX = wc.required_int_env("WATCHDOG_CONFIG_MAX")
CONFIG_RESTART_WINDOW_H = wc.required_float_env("WATCHDOG_CONFIG_WINDOW_H")
# Map root-relative configs to owner-process kill patterns. Shared secret/global configs
# are deliberately excluded because restarting the whole fleet requires a human decision.
_CONFIG_OWNERS = {
    # The restart map currently assigns instruments.conf only to monitortrades.
    # cacheManager and priceAnalysis also read its ``mt`` namespace but are not
    # restarted by this mapping. tradeall/rtrade use their own config files.
    "instruments.conf": ["monitortrades.py"],
    "monitortrades.conf": ["monitortrades.py"],
}

# Slow caches update rarely, while order/trade caches update only on exchange events.
# Give event-driven caches large thresholds to avoid quiet-market false positives; fast
# price caches still detect fleet failure promptly.
_STALE_OVERRIDES = {
    # Long-trend writes only for a significant Mann-Kendall result. A sideways market can
    # legitimately leave it unchanged for hours, so use a 24-hour safety-net threshold.
    "cache_price_long_trend.json": 1440,
    "cache_asset_value.jsonl": 60,
    "cache_T_trend.json": 11520,   # An empirical T per coin: recomputed every 7 days -> a threshold of 8 days.
    # Event-driven content may legitimately remain unchanged for days; use 72 hours.
    "cache_order.json": 4320,
    "cache_trade.jsonl": 4320,
    "cache_trade_kraken.json": 4320,
    # market_alerts, not cacheManager, writes cache_prices_multi about every five minutes.
    # Give it an eight-minute alert-only threshold and never restart cacheManager for it.
    "cache_prices_multi.json": 8,
}

# A fresh fast cache proves the fleet is alive, so stale event-driven fill caches then
# mean only that no fills occurred. Suppress those alerts until a hard 30-day ceiling,
# after which fill tracking itself is suspect. Without proof of life, fail safe and alert.
_EVENT_DRIVEN_CACHES = {
    "cache_order.json", "cache_trade.jsonl", "cache_trade_kraken.json",
    # Retain recognition of explicit legacy/test overrides during the JSONL migration.
    "cache_order.jsonl", "cache_trade.json",
}
_FLEET_ALIVE_CACHES = {"cache_prices_multi.json", "cache_currentprice.json", "cache_instant_trend.json"}
_EVENT_DRIVEN_HARD_CEILING_MIN = 43200   # Fill tracking is suspect after 30 days even if fleet is alive.

# Truly fast one-second price caches use a tighter threshold. Only these trigger automatic
# cacheManager restart because that process writes them. Sparse JSONL history keeps the
# general threshold and additional margin.
_FAST_PRICE_THRESHOLD_MIN = wc.required_float_env("WATCHDOG_FAST_PRICE_MINUTES")


def _is_fast_price_cache(name):
    """Return whether cacheManager writes this fast cache at roughly one-second cadence.

    These receive a tight threshold and may trigger restart. Exclude slower JSONL archives
    and cache_prices_multi, which has its own alert-only threshold.
    """
    if name in ("cache_currentprice.json", "cache_instant_trend.json"):
        return True
    if name.startswith("cache_24price_") and name.endswith(".json"):   # Per-symbol WebSocket/poll cache.
        return True
    return False


def _threshold_for(name):
    """Return the fast, overridden slow/event-driven, or default stale threshold."""
    if _is_fast_price_cache(name):
        return _FAST_PRICE_THRESHOLD_MIN
    return _STALE_OVERRIDES.get(name, STALE_MINUTES)


def _cache_files():
    """List tracked JSON and JSONL cache files, excluding backups, temporaries, and metadata."""
    patterns = ("cache_*.json", "cache_*.jsonl")
    files = {p for pat in patterns for p in _CACHE_DIR.glob(pat)}
    files = {p for p in files if not p.name.endswith((".bak", ".tmp", ".meta"))}
    # Migration deliberately leaves rollback copies beside the active cache. Once the
    # active counterpart exists, do not monitor the frozen legacy file as live state.
    legacy_when_active = {
        "cache_order.jsonl": "cache_order.json",
        "cache_trade.json": "cache_trade.jsonl",
        "cache_asset_value.json": "cache_asset_value.jsonl",
    }
    files = {
        p for p in files
        if not (p.name in legacy_when_active
                and (_CACHE_DIR / legacy_when_active[p.name]).exists())
    }
    return sorted(files)


def _normalize_ts_seconds(value):
    """Normalize millisecond or second fetch timestamps to float seconds."""
    if not isinstance(value, (int, float)) or value <= 0:
        return 0.0
    return value / 1000.0 if value > 1e12 else float(value)


def cache_freshness_seconds(path):
    """Read the newest content freshness timestamp and its source.

    Use fetch time or per-symbol ``ts``. Fall back to mtime only when content has no
    timestamp, because periodically saving frozen data must not make it appear fresh.
    Return zero with a reason for missing or corrupt files.
    """
    p = Path(path)
    if not p.exists():
        return 0.0, f"file {p.name} does not exist"
    newest = 0.0
    if p.name.endswith(".jsonl"):
        # Read only the tail of potentially large JSONL and use its latest complete line.
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                chunk = f.read().decode("utf-8", errors="replace")
            lines = [l for l in chunk.split("\n") if l.strip()]
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                    ts = rec.get("i", [0])[0] if isinstance(rec.get("i"), list) else 0
                    if ts:
                        newest = _normalize_ts_seconds(ts)
                        break
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue   # seek may cut a line; try the preceding complete line.
        except OSError as e:
            return 0.0, f"cache corupt: {e}"
    else:
        try:
            data = json.load(open(p))
            if isinstance(data, dict):
                for v in data.get("fetchtime", {}).values():
                    newest = max(newest, _normalize_ts_seconds(v))
                if newest == 0.0:
                    # Without fetch times, search each symbol's ``ts`` field.
                    for v in data.values():
                        if isinstance(v, dict):
                            newest = max(newest, _normalize_ts_seconds(v.get("ts", 0)))
        except Exception as e:
            return 0.0, f"cache corupt: {e}"
    if newest > 0.0:
        return newest, "continut"
    try:
        return p.stat().st_mtime, "mtime (content without a timestamp)"
    except OSError:
        return 0.0, "mtime indisponibil"


def _do_restart(target=AUTO_RESTART_TARGET):
    """Kill the target for fleet supervision to respawn; isolated for testing."""
    import subprocess
    subprocess.run(["pkill", "-f", target], timeout=10, check=False)
    return True


def _config_hash(path):
    """Return the content SHA-256, or None when the file is absent."""
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def _config_kv(path):
    """Parseaza un *_config.env in {KEY: VALUE} (ignora comentarii/goale)."""
    kv = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.split("#", 1)[0].strip()
                if "=" in ln and not ln.startswith("export"):
                    k, v = ln.split("=", 1)
                    kv[k.strip()] = v.strip()
    except OSError:
        pass
    return kv


def _diff_config(old_kv, new_kv):
    """Lines of 'KEY: old->new' for the keys changed, added or removed."""
    lines = []
    for k in sorted(set(old_kv) | set(new_kv)):
        o, n = old_kv.get(k), new_kv.get(k)
        if o != n:
            lines.append(f"{k}: {o if o is not None else '(new)'} -> {n if n is not None else '(removed)'}")
    return lines


def check_configs_once(now=None):
    """Restart config owners after debounced content changes.

    First observation establishes a baseline. Updating hashes immediately makes each
    change fire once. A kill switch, cooldown, and rolling cap guard restarts.
    Return the restarted process patterns.
    """
    now = now if now is not None else time.time()
    state = wc.load_state(STATE_FILE)
    hashes = state.setdefault("config_hashes", {})

    kv_state = state.setdefault("config_kv", {})
    changed = []
    diffs = {}
    for name in _CONFIG_OWNERS:
        path = str(_ROOT / name)
        h = _config_hash(path)
        if h is None:
            continue
        prev = hashes.get(name)
        new_kv = _config_kv(path)
        prev_kv = kv_state.get(name)
        hashes[name] = h                      # Always update for baseline and debounce.
        kv_state[name] = new_kv
        if prev is not None and h != prev:
            changed.append(name)
            if prev_kv is not None:
                diffs[name] = _diff_config(prev_kv, new_kv)

    if not changed:
        wc.save_state(STATE_FILE, state)
        return []

    owners = sorted({o for n in changed for o in _CONFIG_OWNERS[n]})
    diff_txt = "".join(
        f"\n{n}:\n  " + "\n  ".join(diffs[n]) for n in changed if diffs.get(n)
    )
    note = f"config schimbat: {', '.join(changed)} -> proprietari: {', '.join(owners)}" + diff_txt

    if not CONFIG_RESTART:
        print(f"[watchdog] {note} — WATCHDOG_CONFIG_RESTART=false, only notifying")
        wc.send_ntfy("⚙️ Config changed", note + "\n(auto-restart is OFF; restart by hand if needed)")
        wc.save_state(STATE_FILE, state)
        return []

    hist = [t for t in state.get("config_restart_history", []) if now - t < CONFIG_RESTART_WINDOW_H * 3600]
    if hist and (now - max(hist)) < CONFIG_RESTART_COOLDOWN_MIN * 60:
        print(f"[watchdog] {note} — but in COOLDOWN ({CONFIG_RESTART_COOLDOWN_MIN:.0f}min), not restarting now")
        wc.save_state(STATE_FILE, state)
        return []
    if len(hist) >= CONFIG_RESTART_MAX:
        msg = f"⛔ config-restart CAP reached ({CONFIG_RESTART_MAX} in {CONFIG_RESTART_WINDOW_H:.0f}h). {note}. NOT restarting — check manually."
        print(f"[watchdog] {msg}")
        wc.send_ntfy("⛔ Config-restart plafonat", msg)
        wc.save_state(STATE_FILE, state)
        return []

    for pat in owners:
        _do_restart(pat)
    hist.append(now)
    state["config_restart_history"] = hist
    wc.save_state(STATE_FILE, state)

    msg = (f"🔄 {', '.join(changed)} changed -> restarted {', '.join(owners)} "
           f"(respawned through healthcheck --supervise). Restart {len(hist)}/{CONFIG_RESTART_MAX} in {CONFIG_RESTART_WINDOW_H:.0f}h."
           + diff_txt)
    print(f"[watchdog] {msg}")
    wc.send_ntfy("🔄 Config schimbat -> restart", msg)
    wc.send_email("Config schimbat -> restart proces", msg)
    return owners


def _maybe_auto_restart(stale, now, state):
    """Restart cacheManager for stale fast caches subject to cooldown and rolling cap."""
    if not AUTO_RESTART:
        return False, ""
    # Only fast caches written by cacheManager justify restarting that process.
    critical = [name for (name, _age, _thr, _det) in stale if _is_fast_price_cache(name)]
    if not critical:
        return False, ""
    hist = [t for t in state.get("auto_restart_history", []) if now - t < AUTO_RESTART_WINDOW_H * 3600]
    if hist and (now - max(hist)) < AUTO_RESTART_COOLDOWN_MIN * 60:
        return False, (f"the auto-restart is in COOLDOWN ({AUTO_RESTART_COOLDOWN_MIN:.0f}min since the last one) "
                       f"— only alerting, cacheManager NOT restarted")
    if len(hist) >= AUTO_RESTART_MAX:
        return False, (f"⛔ PLAFON auto-restart atins ({AUTO_RESTART_MAX} in {AUTO_RESTART_WINDOW_H:.0f}h) "
                       f"— MANUAL INTERVENTION needed, no more automatic restarts")
    try:
        _do_restart()
        hist.append(now)
        state["auto_restart_history"] = hist
        return True, (f"🔁 cacheManager RESTARTED automatically (stale cache: {', '.join(critical)}). "
                      f"Restart {len(hist)}/{AUTO_RESTART_MAX} in a window of {AUTO_RESTART_WINDOW_H:.0f}h.")
    except Exception as e:  # noqa: BLE001 — restart failure must not suppress the alert.
        return False, f"the auto-restart FAILED ({e}) — restart the fleet MANUALLY"


def check_once(now=None):
    """Check every cache and return whether a non-cooled-down stale alert was sent."""
    now = now if now is not None else time.time()
    files = _cache_files()
    stale = []
    fleet_alive = False   # A fresh fast cache proves fleet liveness.
    if not files:
        stale.append(("(no cache_*.json)", float("inf"), STALE_MINUTES,
                      f"{_CACHE_DIR} is empty or missing"))
    for p in files:
        freshness, detail = cache_freshness_seconds(p)
        age_min = (now - freshness) / 60.0 if freshness > 0 else float("inf")
        thr = _threshold_for(p.name)
        if p.name in _FLEET_ALIVE_CACHES and age_min <= thr:
            fleet_alive = True
        if age_min > thr:
            stale.append((p.name, age_min, thr, detail))

    # With proven fleet liveness, suppress benign event-cache staleness until its hard
    # ceiling. Without that proof, suppress nothing.
    if fleet_alive:
        suppressed = [s for s in stale
                      if s[0] in _EVENT_DRIVEN_CACHES and s[1] < _EVENT_DRIVEN_HARD_CEILING_MIN]
        if suppressed:
            names = ", ".join(s[0] for s in suppressed)
            print(f"[watchdog] {names} stale but the fleet is alive (fresh price) — benign, not alarming")
        stale = [s for s in stale if s not in suppressed]

    if not stale:
        print(f"[watchdog] OK — {len(files)} cache-uri proaspete")
        return False

    state = wc.load_state(STATE_FILE)

    # Auto-restart has its own guardrails and is independent of alert cooldown.
    restarted, restart_note = _maybe_auto_restart(stale, now, state)
    if restart_note:
        print(f"[watchdog] {restart_note}")

    # Alert cooldown prevents repetition, but an actual restart bypasses it because the
    # operator must know that a process restarted.
    last = state.get("last_alert_ts", 0)
    if (now - last) < COOLDOWN_MINUTES * 60 and not restarted:
        print(f"[watchdog] STALE ({', '.join(s[0] for s in stale)}) but in cooldown — not re-alarming")
        wc.save_state(STATE_FILE, state)   # Persist restart history even without an alert.
        return False

    lines = []
    for name, age_min, thr, detail in stale:
        age_txt = f"{age_min:.0f} min" if age_min != float("inf") else "∞"
        lines.append(f"  • {name}: {age_txt} (threshold {thr:.0f} min) — {detail}")
    title = "⚠️ Cache STALE pe server"
    message = ("Stale caches (cacheManager/priceAnalysis probably stopped):\n"
               + "\n".join(lines))
    message += ("\n\n" + restart_note) if restart_note else "\nCheck the fleet (flota_start) and restart it."
    print(f"[watchdog] ALARM:\n{message}")
    wc.send_ntfy(title, message)
    wc.send_email(title, message)
    state["last_alert_ts"] = now
    wc.save_state(STATE_FILE, state)
    return True


if __name__ == "__main__":
    # Run config watch first. The checks use disjoint state keys so hashes and histories
    # persist correctly across invocations.
    check_configs_once()
    sent = check_once()
    sys.exit(2 if sent else 0)
