#!/usr/bin/env python3
"""agy_store_fix.py - keep Antigravity (agy 1.4.0) conversations visible in the Web UI across restarts/reboots.

Background (see AGY_FINDINGS_20261016.md):
  * The Web UI hides conversations whose summary has source = CORTEX_TRAJECTORY_SOURCE_CLI (17).
  * The daemon forces source=17 on every <cid>.db it finds in ~/.gemini/antigravity-cli/conversations at startup,
    but keeps source=0 for files in ~/.gemini/antigravity/conversations.
  * Conversation CONTENT is only loadable from ~/.gemini/antigravity-cli/conversations.

Layout enforced here:
  real file : ~/.gemini/antigravity/conversations/<cid>.db        (source 0 -> visible)
  hard link : ~/.gemini/antigravity-cli/conversations/<cid>.db    (created AFTER daemon start, so content opens)

Modes (run by the systemd drop-in of antigravity-cli-daemon.service):
  pre    daemon stopped: checkpoint WAL, move/unlink so the real file lives only in the visible dir, and repair the
         summary rows (source cleared, title kept, mtimes normalised, fileModTimes refreshed).
  post   daemon started: wait for its startup scan to finish, then hard-link the files back for content loading.
  check  print a status summary (read-only).
  ui-sync-check --idle-sec N
         read-only: exit 0 when the store diverges from the daemon's in-memory list (new/removed/rewritten
         conversations, e.g. mirrored autonomous topic threads) AND no conversation activity for N seconds,
         i.e. an idle restart is due (driven by `manage_agy.sh ui-sync`). 10=nothing pending, 11=busy,
         12=same divergence survived the previous restart (no restart loop).

Safety: a file is never deleted unless the identical inode/content already exists in the visible dir; anything that
diverges is moved into ~/agy_store_fix_backups/<timestamp>/ instead of being overwritten.
"""
from __future__ import annotations

import collections
import fcntl
import glob
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:   # executed directly by the systemd drop-in with /usr/bin/python3
    sys.path.insert(0, _ROOT_DIR)

from botcore import _parse_bool_flag, safe_nonneg_int, safe_pos_float
from state_io import (  # noqa: E402
    atomic_write_bytes,
    discard_generation_file,
    read_bytes_file,
    read_json_dict,
    read_text_file,
    save_json_state,
)

from orchestratorOS.agy.paths import (  # noqa: E402
    resolve_bot_cli_base,
    resolve_store_lock_path,
    visible_conversations_dir,
)
from orchestratorOS.agy.protobuf import (  # noqa: E402
    enc_fields,
    list_entry_cids,
    summary_annotation_title,
    summary_ui_visible,
)
from orchestratorOS.agy.summary_store import (  # noqa: E402
    checkpoint_db,
    drop_wal_sidecars,
    read_annotation_title,
    read_summary_row,
    upsert_ui_visible_summary,
)
from orchestratorOS.agy.topics import (  # noqa: E402
    get_active_topic_cids,
    get_topic_canonical_titles,
    get_topic_registry_path,
)


HOME = os.environ.get("AGY_FIX_HOME") or os.path.expanduser("~")
CLI = os.path.join(HOME, ".gemini", "antigravity-cli")
D1 = os.path.join(CLI, "conversations")                       # content dir (source forced to 17 by the daemon)
D2 = visible_conversations_dir(CLI)                           # visible dir (source stays 0)
UI_SYNC_STATE = os.path.join(HOME, ".agy_ui_sync_state.json")
ANN = os.path.join(CLI, "annotations")
DB = os.path.join(CLI, "conversation_summaries.db")
PB = os.path.join(CLI, "agyhub_summaries_proto.pb")
LOGS = os.path.join(CLI, "log")
LOCK = resolve_store_lock_path(HOME)
LOG_FILE = os.path.join(HOME, "agy_store_fix.log")
BACKUPS = os.path.join(HOME, "agy_store_fix_backups")
AGY_BIN = os.path.join(HOME, ".local", "bin", "agy")
WEBRTC_FLAG_ORIG = b"remote-control-webrtc-mesh"
WEBRTC_FLAG_PATCHED = b"remote-control-webrtc-mes0"
UNIT = "antigravity-cli-daemon.service"

log = logging.getLogger("agy_store_fix")


# ---------------------------------------------------------------- summary helpers
def fix_summary(raw: bytes, title: str) -> bytes:
    """UI-visible summary (source dropped, title applied) with a stale RUNNING status reset to IDLE."""
    return summary_ui_visible(raw, title, reset_running=True)


def raw_title(raw: bytes) -> str:
    return summary_annotation_title(raw)


def pbtxt_title(cid: str) -> str:
    return read_annotation_title(CLI, cid)


# ---------------------------------------------------------------- file helpers
def checkpoint(path: str) -> bool:
    """Flush the WAL into the main file. Returns False if it could not be done safely."""
    return checkpoint_db(path)


def drop_sidecars(path: str) -> None:
    drop_wal_sidecars(path, force=False)


def same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def backup_file(path: str, stamp: str) -> None:
    dst = os.path.join(BACKUPS, stamp)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(path, os.path.join(dst, os.path.basename(path)))


def unit_active() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def disable_webrtc_mesh(stamp: str) -> str:
    """Disable the Mendel flag 'remote-control-webrtc-mesh' in the agy binary so remote-control
    stays on Boq WebChannel (HTTPS/TCP) and never upgrades to Pion WebRTC DataChannel (UDP),
    whose hardcoded 1228-byte SCTP MTU exceeds the wgpia0 WireGuard MTU (1200) and freezes streams."""
    if not os.path.isfile(AGY_BIN):
        return "missing"
    data = read_bytes_file(AGY_BIN, default=None)
    if data is None:
        log.warning("cannot read %s", AGY_BIN)
        return "error"
    if WEBRTC_FLAG_ORIG not in data:
        return "already_disabled" if WEBRTC_FLAG_PATCHED in data else "flag_not_found"
    try:
        backup_dir = os.path.join(BACKUPS, stamp)
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(AGY_BIN, os.path.join(backup_dir, "agy.orig"))
        patched = data.replace(WEBRTC_FLAG_ORIG, WEBRTC_FLAG_PATCHED)
        tmp_bin = AGY_BIN + ".tmp"
        with open(tmp_bin, "wb") as f:
            f.write(patched)
        shutil.copymode(AGY_BIN, tmp_bin)
        os.replace(tmp_bin, AGY_BIN)
        log.info("pre: patched %s to disable remote-control-webrtc-mesh (backup in %s)", AGY_BIN, backup_dir)
        return "patched"
    except OSError as e:
        log.warning("failed to patch %s: %s", AGY_BIN, e)
        return "error"


def webrtc_mesh_status() -> str:
    if not os.path.isfile(AGY_BIN):
        return "missing"
    data = read_bytes_file(AGY_BIN, default=None)
    if data is None:
        return "unreadable"
    if WEBRTC_FLAG_ORIG in data:
        return "ENABLED (unpatched)"
    if WEBRTC_FLAG_PATCHED in data:
        return "disabled (patched)"
    return "unknown"


# ---------------------------------------------------------------- pre
_as_bool_flag = _parse_bool_flag


def mode_pre(force: bool = False) -> int:
    force = _as_bool_flag(force, False)
    if not force and os.environ.get("AGY_FIX_HOME") is None and unit_active():
        log.error("daemon is active; refusing to run 'pre' (use --force to override)")
        return 2
    os.makedirs(D1, exist_ok=True)
    os.makedirs(D2, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mesh_state = disable_webrtc_mesh(stamp)
    moved = unlinked = diverged = skipped = deleted = 0

    # Detect conversations that existed in D2 from a previous session but whose summary row was
    # deleted in the Web UI (removed from conversation_summaries.db). Move them to backups so
    # they are not resurrected into D1 on post.
    known_cids: set[str] = set()
    if os.path.exists(DB):
        try:
            con = sqlite3.connect(DB, timeout=15)
            try:
                known_cids = {r[0] for r in con.execute("SELECT conversation_id FROM conversation_summaries")}
            finally:
                con.close()
        except sqlite3.Error as e:
            log.warning("cannot read summary IDs from %s: %s", DB, e)

    if known_cids:
        del_dir = os.path.join(BACKUPS, stamp, "deleted_in_ui")
        for p2 in sorted(glob.glob(os.path.join(glob.escape(D2), "*.db"))):
            name = os.path.basename(p2)
            cid = name[:-3]
            if cid in known_cids:
                continue
            p1 = os.path.join(D1, name)
            checkpoint(p2)
            os.makedirs(del_dir, exist_ok=True)
            shutil.move(p2, os.path.join(del_dir, name))
            drop_sidecars(p2)
            if os.path.exists(p1):
                checkpoint(p1)
                try:
                    os.remove(p1)
                except OSError:
                    pass
                drop_sidecars(p1)
            ann_file = os.path.join(ANN, cid + ".pbtxt")
            if os.path.exists(ann_file):
                try:
                    shutil.move(ann_file, os.path.join(del_dir, cid + ".pbtxt"))
                except OSError:
                    pass
            deleted += 1

    for name in sorted(os.listdir(D1)):
        if not name.endswith(".db"):
            continue
        p1, p2 = os.path.join(D1, name), os.path.join(D2, name)
        if os.path.islink(p1):
            log.warning("%s is a symlink; skipped", p1)
            skipped += 1
            continue
        if os.path.exists(p2):
            if same_file(p1, p2):
                if checkpoint(p1):
                    os.remove(p1)
                    drop_sidecars(p1)
                    unlinked += 1
                else:
                    skipped += 1
                continue
            # diverged copies: keep the newer in D2, back up the other one, never overwrite silently
            diverged += 1
            if os.stat(p1).st_mtime_ns > os.stat(p2).st_mtime_ns and checkpoint(p1) and checkpoint(p2):
                backup_file(p2, stamp)
                os.replace(p1, p2)
                drop_sidecars(p1)
            elif checkpoint(p1):
                backup_file(p1, stamp)
                os.remove(p1)
                drop_sidecars(p1)
            else:
                skipped += 1
            continue
        if not checkpoint(p1):
            skipped += 1
            continue
        shutil.move(p1, p2)
        drop_sidecars(p1)
        moved += 1

    # leftover WAL/SHM sidecars on the real files
    for p2 in glob.glob(os.path.join(glob.escape(D2), "*.db")):
        if checkpoint(p2):
            drop_sidecars(p2)

    # normalise mtimes to whole seconds so stored fileModTimes (seconds) always >= disk mtime
    mt = {}
    for p2 in glob.glob(os.path.join(glob.escape(D2), "*.db")):
        cid = os.path.basename(p2)[:-3]
        sec = int(os.stat(p2).st_mtime)
        os.utime(p2, (sec, sec))
        mt[cid] = sec

    rows_fixed = repair_summaries(mt)
    log.info("pre: moved=%d unlinked=%d deleted=%d diverged=%d skipped=%d rows_fixed=%d visible_files=%d webrtc_mesh=%s",
             moved, unlinked, deleted, diverged, skipped, rows_fixed, len(mt), mesh_state)
    return 0


def repair_summaries(mt: dict) -> int:
    if not os.path.exists(DB):
        return 0
    topic_titles = topic_canonical_titles()
    if topic_titles and os.environ.get("AGY_FIX_HOME") is None:
        try:
            bot_base = resolve_bot_cli_base()
            if os.path.isdir(bot_base) and os.path.abspath(bot_base) != os.path.abspath(CLI):
                for cid, ttl in topic_titles.items():
                    if cid not in mt:
                        continue
                    bot_row = read_summary_row(bot_base, cid)
                    ui_row = read_summary_row(CLI, cid)
                    if bot_row is not None and (
                        ui_row is None
                        or safe_nonneg_int(bot_row.get("step_count"), 0)
                        >= safe_nonneg_int(ui_row.get("step_count"), 0)
                    ):
                        upsert_ui_visible_summary(CLI, bot_row, ttl)
        except Exception as e:
            log.warning("pre: failed refreshing topic summaries from bot store: %s", e)
    con = sqlite3.connect(DB, timeout=15)
    try:
        rows = con.execute("SELECT conversation_id, title, raw_summary FROM conversation_summaries").fetchall()
        raws, fixed, orphans = {}, 0, []
        for cid, title, raw in rows:
            raw = bytes(raw or b"")
            if cid in mt:
                ttl = topic_titles.get(cid) or pbtxt_title(cid) or (title or "").strip() or raw_title(raw)
                new = fix_summary(raw, ttl)
                con.execute(
                    "UPDATE conversation_summaries SET title=?, source='', status='CASCADE_RUN_STATUS_IDLE', "
                    "not_fully_idle=0, raw_summary=? WHERE conversation_id=?", (ttl, new, cid))
                raws[cid] = new
                fixed += 1
            else:
                orphans.append(cid)
        if orphans:
            con.execute(
                f"DELETE FROM conversation_summaries WHERE conversation_id IN ({','.join('?' for _ in orphans)})",
                orphans,
            )
            log.info("pre: removed %d orphan summary rows from conversation_summaries.db", len(orphans))
        con.commit()
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
    finally:
        con.close()

    # rebuild agyhub_summaries_proto.pb (and jetbox_summaries_proto.pb if present): Keep only existing mt files.
    out = bytearray()
    for cid, raw in raws.items():
        if cid in mt:
            out += enc_fields([(1, 2, enc_fields([(1, 2, cid.encode()), (2, 2, raw)]))])
    for cid, sec in mt.items():
        out += enc_fields([(2, 2, enc_fields([(1, 2, cid.encode()), (2, 0, sec)]))])
    for pb_path in (PB, os.path.join(CLI, "jetbox_summaries_proto.pb")):
        if pb_path != PB and not os.path.exists(pb_path):
            continue
        atomic_write_bytes(pb_path, bytes(out))
    return fixed


# ---------------------------------------------------------------- post
def log_started_at(path: str) -> float:
    """cli-YYYYMMDD_HHMMSS.log -> epoch seconds (local time); 0 if the name does not match."""
    m = re.search(r"cli-(\d{8}_\d{6})\.log$", path)
    try:
        return time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S")) if m else 0.0
    except ValueError:
        return 0.0


def main_pid() -> str:
    unit = os.environ.get("AGY_FIX_UNIT", UNIT)
    try:
        r = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                           capture_output=True, text=True, timeout=10)
        pid = r.stdout.strip()
        return pid if pid and pid != "0" else ""
    except Exception:
        return ""


def newest_daemon_log_ready(since: float, pid: str = "") -> bool:
    """True once THIS daemon instance (matched by PID, else by log-file start time >= since) logged the end of
    its startup reconciliation. Never matches the log of a previous instance."""
    for lf in glob.glob(os.path.join(glob.escape(LOGS), "cli-*.log")):
        if not pid and log_started_at(lf) < since:
            continue
        txt = read_text_file(lf, default=None, errors="ignore")
        if txt is None:
            continue
        if pid and f"language server process with pid {pid}" not in txt:
            continue
        if "summary store: reconciliation complete" in txt:
            return True
    return False


def mode_post(timeout: int = 50) -> int:
    timeout_s = safe_pos_float(timeout, default=50.0)
    start = time.time()
    deadline = start + timeout_s
    ready = False
    while time.time() < deadline:
        pid = main_pid()
        if newest_daemon_log_ready(start - 15, pid):
            ready = True
            break
        time.sleep(0.2)
    if not ready:
        log.warning("post: daemon startup scan not confirmed within %ds; continuing after a fixed wait", int(timeout_s))
        time.sleep(5)
    time.sleep(0.5)   # link immediately after startup scan so reconnecting Web UI tabs find trajectories in D1
    linked = exists = bad = 0
    for p2 in sorted(glob.glob(os.path.join(glob.escape(D2), "*.db"))):
        p1 = os.path.join(D1, os.path.basename(p2))
        if os.path.lexists(p1):
            if same_file(p1, p2):
                exists += 1
            else:
                log.warning("post: %s exists and is a different file; left untouched", p1)
                bad += 1
            continue
        try:
            os.link(p2, p1)
            linked += 1
        except OSError as e:
            log.warning("post: cannot link %s: %s", p1, e)
            bad += 1
    log.info("post: linked=%d already=%d problems=%d waited=%.0fs", linked, exists, bad, time.time() - start)
    return 0


# ---------------------------------------------------------------- check
def mode_check() -> int:
    d1 = {f for f in os.listdir(D1) if f.endswith(".db")} if os.path.isdir(D1) else set()
    d2 = {f for f in os.listdir(D2) if f.endswith(".db")} if os.path.isdir(D2) else set()
    linked = sum(1 for f in d1 & d2 if same_file(os.path.join(D1, f), os.path.join(D2, f)))
    print(f"visible dir files={len(d2)}  content dir files={len(d1)}  hard-linked={linked}  only-in-content-dir={len(d1 - d2)}  webrtc_mesh={webrtc_mesh_status()}")
    if os.path.exists(DB):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            cnt = collections.Counter(
                ((r[0] or "none")[:8], r[1] or "unset", "titled" if r[2] else "untitled")
                for r in c.execute("select project_id, source, title from conversation_summaries")
            )
        finally:
            c.close()
        for k, v in sorted(cnt.items()):
            print(" ", v, k)
    return 0


# ---------------------------------------------------------------- ui-sync-check
# The daemon keeps its conversation list in memory, discovers .db files only during its startup scan and
# persists that in-memory list to agyhub_summaries_proto.pb. Anything written into the store from outside
# (mirrored autonomous topic threads, rows removed on rotation) becomes visible only after a daemon restart.
# ui-sync-check decides, read-only, whether such an idle-gated restart is due.
UI_SYNC_READY, UI_SYNC_NOTHING, UI_SYNC_BUSY, UI_SYNC_STUCK = 0, 10, 11, 12


def daemon_known_cids() -> set[str]:
    """Conversation IDs the running daemon holds in memory (as persisted in agyhub_summaries_proto.pb)."""
    data = read_bytes_file(PB, default=None)
    return list_entry_cids(data) if data is not None else set()


def summary_row_cids() -> set[str]:
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=15)
        try:
            return {r[0] for r in c.execute("SELECT conversation_id FROM conversation_summaries") if r and r[0]}
        finally:
            c.close()
    except (sqlite3.Error, OSError):
        return set()


def pending_ui_sync() -> list[str]:
    """Divergences between the on-disk store and the daemon's in-memory list, as stable fingerprint entries.

    new:<cid>       summary row (with content on disk) exists, daemon does not know the conversation yet
    gone:<cid>      daemon still lists a conversation whose summary row was removed (deleted/rotated)
    content:<cid>   content file is not hard-linked into the visible dir (new or rewritten since startup)
    """
    rows = summary_row_cids()
    known = daemon_known_cids()
    disk_cids: set[str] = set()
    for d in (D1, D2):
        if os.path.isdir(d):
            disk_cids.update(n[:-3] for n in os.listdir(d) if n.endswith(".db"))
    active_rows = rows & disk_cids
    topic_cids = mirrored_topic_cids()
    entries = [f"new:{cid}" for cid in active_rows - known] + [f"gone:{cid}" for cid in known - active_rows]
    if os.path.isdir(D1):
        for name in os.listdir(D1):
            cid = name[:-3]
            if not name.endswith(".db") or cid not in rows:
                continue
            p1 = os.path.join(D1, name)
            p2 = os.path.join(D2, name)
            if (cid not in known or os.path.exists(p2) or cid in topic_cids) and not same_file(p1, p2):
                try:
                    entries.append(f"content:{cid}:{os.stat(p1).st_ino}")
                except OSError:
                    pass
    return sorted(entries)


TOPIC_REGISTRY = get_topic_registry_path()


def mirrored_topic_cids() -> set[str]:
    """Conversation IDs owned by mirrored background topic threads (excluded from interactive idle detection)."""
    return get_active_topic_cids(TOPIC_REGISTRY)


def topic_canonical_titles() -> dict[str, str]:
    """Canonical permanent titles for topic threads from cachedb/llm_topic_threads.json."""
    return get_topic_canonical_titles(TOPIC_REGISTRY)


def last_store_activity() -> float:
    """Newest mtime of interactive conversation content, active WAL writes, daemon Web UI stream, or brain artifacts."""
    ignored = mirrored_topic_cids()
    newest = 0.0
    if os.path.exists(DB):
        try:
            con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
            try:
                try:
                    rows = con.execute(
                        "SELECT conversation_id FROM conversation_summaries "
                        "WHERE status='CASCADE_RUN_STATUS_RUNNING' OR COALESCE(not_fully_idle, 0) != 0"
                    ).fetchall()
                except sqlite3.Error:
                    rows = con.execute(
                        "SELECT conversation_id FROM conversation_summaries WHERE status='CASCADE_RUN_STATUS_RUNNING'"
                    ).fetchall()
                running_cids = [r[0] for r in rows if r and r[0] and r[0] not in ignored]
            finally:
                con.close()
            now = time.time()
            for rcid in running_cids:
                found_on_disk = False
                for d in (D1, D2):
                    for ext in (".db", ".db-wal"):
                        p = os.path.join(d, f"{rcid}{ext}")
                        try:
                            if os.path.exists(p):
                                found_on_disk = True
                                if (now - os.stat(p).st_mtime) < 900:
                                    return now
                        except OSError:
                            pass
                if not found_on_disk:
                    for sum_p in (DB, DB + "-wal"):
                        try:
                            if os.path.exists(sum_p) and (now - os.stat(sum_p).st_mtime) < 900:
                                return now
                        except OSError:
                            pass
        except sqlite3.Error:
            pass
    for d in (D1, D2):
        if os.path.isdir(d):
            for name in os.listdir(d):
                p = os.path.join(d, name)
                try:
                    cid = name.split(".", 1)[0]
                    if name.endswith(".db-wal") and os.path.getsize(p) > 0 and cid not in ignored:
                        newest = max(newest, os.stat(p).st_mtime)
                    elif name.endswith((".db", ".db-wal")) and cid not in ignored:
                        newest = max(newest, os.stat(p).st_mtime)
                except OSError:
                    pass
    if os.path.isdir(LOGS):
        try:
            cli_logs = sorted(glob.glob(os.path.join(glob.escape(LOGS), "cli-*.log")), key=os.path.getmtime, reverse=True)
            if cli_logs:
                latest_log = cli_logs[0]
                txt = read_text_file(latest_log, default="", errors="ignore")
                conn_idx = txt.rfind("Connection status: Connected")
                if conn_idx >= 0:
                    tail = txt[conn_idx:]
                    recent_tail = "\n".join(tail.splitlines()[-2:])
                    if any(
                        marker in recent_tail
                        for marker in (
                            "SEND_USER_CASCADE_MESSAGE",
                            "streamGenerateContent",
                            "LLM Monitor",
                        )
                    ):
                        newest = max(newest, os.stat(latest_log).st_mtime)
        except OSError:
            pass
    brain = os.path.join(CLI, "brain")
    for root, dirs, files in os.walk(brain):
        if root == brain:
            dirs[:] = [d for d in dirs if d not in ignored]
        elif root.count(os.sep) - brain.count(os.sep) >= 3:
            dirs[:] = []
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def mode_ui_sync_check(idle_sec: int) -> int:
    idle_sec_f = safe_pos_float(idle_sec, default=None)
    if idle_sec_f is None:
        raise ValueError(f"--idle-sec must be > 0, got {idle_sec}")
    if not (os.path.exists(PB) and os.path.exists(DB)):
        log.warning("ui-sync-check: summary store not initialised (%s / %s); skipping", PB, DB)
        return UI_SYNC_NOTHING
    entries = pending_ui_sync()
    if not entries:
        discard_generation_file(UI_SYNC_STATE, include_previous=True)
        return UI_SYNC_NOTHING
    idle_for = time.time() - last_store_activity()
    if idle_for < idle_sec_f:
        log.info("ui-sync-check: %d pending, store active %.0fs ago (< %gs); waiting", len(entries), idle_for, idle_sec_f)
        return UI_SYNC_BUSY
    fingerprint = hashlib.sha1("\n".join(entries).encode()).hexdigest()
    state = read_json_dict(UI_SYNC_STATE, default={}, recover_previous=True)
    if state.get("fingerprint") == fingerprint:
        log.warning("ui-sync-check: same %d pending entries persisted after the last restart; not restarting again: %s",
                    len(entries), entries[:5])
        return UI_SYNC_STUCK
    save_json_state(UI_SYNC_STATE, {"fingerprint": fingerprint, "ts": time.time(), "entries": entries}, indent=1)
    log.info("ui-sync-check: restart due (%d pending, idle %.0fs): %s", len(entries), idle_for, entries[:10])
    return UI_SYNC_READY


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(LOCK)) or ".", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
    lock = open(LOCK, "w")
    try:
        lock_wait_s = 15.0 if mode in ("pre", "post") else 2.0
        lock_deadline = time.time() + lock_wait_s
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= lock_deadline:
                    log.error("another agy_store_fix is running")
                    return 3
                time.sleep(0.1)
        try:
            if mode == "pre":
                return mode_pre(force="--force" in argv)
            if mode == "post":
                return mode_post()
            if mode == "check":
                return mode_check()
            if mode == "ui-sync-check":
                if "--idle-sec" not in argv or argv.index("--idle-sec") + 1 >= len(argv):
                    raise ValueError("ui-sync-check requires --idle-sec <seconds>")
                return mode_ui_sync_check(float(argv[argv.index("--idle-sec") + 1]))
            print(__doc__)
            return 1
        except Exception:
            log.exception("agy_store_fix %s failed", mode)
            return 1
    finally:
        try:
            lock.close()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
