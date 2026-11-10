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
    durable_replace_file,
    read_bytes_file,
    read_json_dict,
    read_tail_jsonl_dicts,
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


# ---------------------------------------------------------------- file helpers
def same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def backup_file(path: str, stamp: str) -> None:
    dst = os.path.join(BACKUPS, stamp)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(path, os.path.join(dst, os.path.basename(path)))


# ---------------------------------------------------------------- pre
def mode_pre(force: bool = False) -> int:
    force = _parse_bool_flag(force, False)
    if not force and os.environ.get("AGY_FIX_HOME") is None:
        try:
            active = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True, timeout=10).stdout.strip() == "active"
        except Exception:
            active = False
        if active:
            log.error("daemon is active; refusing to run 'pre' (use --force to override)")
            return 2
    os.makedirs(D1, exist_ok=True)
    os.makedirs(D2, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.isfile(AGY_BIN):
        mesh_state = "missing"
    elif (agy_data := read_bytes_file(AGY_BIN, default=None)) is None:
        log.warning("cannot read %s", AGY_BIN)
        mesh_state = "error"
    elif WEBRTC_FLAG_ORIG not in agy_data:
        mesh_state = "already_disabled" if WEBRTC_FLAG_PATCHED in agy_data else "flag_not_found"
    else:
        try:
            backup_dir = os.path.join(BACKUPS, stamp)
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(AGY_BIN, os.path.join(backup_dir, "agy.orig"))
            atomic_write_bytes(AGY_BIN, agy_data.replace(WEBRTC_FLAG_ORIG, WEBRTC_FLAG_PATCHED), preserve_mode=True)
            log.info("pre: patched %s to disable remote-control-webrtc-mesh (backup in %s)", AGY_BIN, backup_dir)
            mesh_state = "patched"
        except OSError as e:
            log.warning("failed to patch %s: %s", AGY_BIN, e)
            mesh_state = "error"
    moved = unlinked = diverged = skipped = deleted = 0

    # Detect conversations that existed in D2 from a previous session but whose summary row was
    # deleted in the Web UI (removed from conversation_summaries.db). Move them to backups so
    # they are not resurrected into D1 on post.
    known_cids: set[str] = set()
    if os.path.exists(DB):
        checkpoint_db(DB)
        try:
            con = sqlite3.connect(DB, timeout=15)
            try:
                known_cids = {r[0] for r in con.execute("SELECT conversation_id FROM conversation_summaries")}
            finally:
                con.close()
        except sqlite3.Error as e:
            log.warning("cannot read summary IDs from %s: %s", DB, e)

    topic_cids = mirrored_topic_cids()
    if known_cids:
        del_dir = os.path.join(BACKUPS, stamp, "deleted_in_ui")
        for p2 in sorted(glob.glob(os.path.join(glob.escape(D2), "*.db"))):
            name = os.path.basename(p2)
            cid = name[:-3]
            if cid in known_cids or cid in topic_cids:
                continue
            p1 = os.path.join(D1, name)
            checkpoint_db(p2)
            os.makedirs(del_dir, exist_ok=True)
            shutil.move(p2, os.path.join(del_dir, name))
            drop_wal_sidecars(p2, force=False)
            if os.path.exists(p1):
                checkpoint_db(p1)
                try:
                    os.remove(p1)
                except OSError:
                    pass
                drop_wal_sidecars(p1, force=False)
            ann_file = os.path.join(ANN, cid + ".pbtxt")
            if os.path.exists(ann_file):
                try:
                    shutil.move(ann_file, os.path.join(del_dir, cid + ".pbtxt"))
                except OSError:
                    pass
            shutil.rmtree(os.path.join(CLI, "brain", cid), ignore_errors=True)
            deleted += 1
        for p1 in sorted(glob.glob(os.path.join(glob.escape(D1), "*.db"))):
            if os.path.islink(p1):
                continue
            name = os.path.basename(p1)
            cid = name[:-3]
            if cid in known_cids or cid in topic_cids:
                continue
            checkpoint_db(p1)
            os.makedirs(del_dir, exist_ok=True)
            shutil.move(p1, os.path.join(del_dir, name))
            drop_wal_sidecars(p1, force=False)
            ann_file = os.path.join(ANN, cid + ".pbtxt")
            if os.path.exists(ann_file):
                try:
                    shutil.move(ann_file, os.path.join(del_dir, cid + ".pbtxt"))
                except OSError:
                    pass
            shutil.rmtree(os.path.join(CLI, "brain", cid), ignore_errors=True)
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
                if checkpoint_db(p1):
                    os.remove(p1)
                    drop_wal_sidecars(p1, force=False)
                    unlinked += 1
                else:
                    skipped += 1
                continue
            # diverged copies: keep the newer in D2, back up the other one, never overwrite silently
            diverged += 1
            if os.stat(p1).st_mtime_ns > os.stat(p2).st_mtime_ns and checkpoint_db(p1) and checkpoint_db(p2):
                backup_file(p2, stamp)
                durable_replace_file(p1, p2)
                drop_wal_sidecars(p1, force=False)
            elif checkpoint_db(p1):
                backup_file(p1, stamp)
                os.remove(p1)
                drop_wal_sidecars(p1, force=False)
            else:
                skipped += 1
            continue
        if not checkpoint_db(p1):
            skipped += 1
            continue
        shutil.move(p1, p2)
        drop_wal_sidecars(p1, force=False)
        moved += 1

    # leftover WAL/SHM sidecars on the real files
    for p2 in glob.glob(os.path.join(glob.escape(D2), "*.db")):
        if checkpoint_db(p2):
            drop_wal_sidecars(p2, force=False)

    # normalise mtimes to whole seconds so stored fileModTimes (seconds) always >= disk mtime
    mt = {}
    for p2 in glob.glob(os.path.join(glob.escape(D2), "*.db")):
        cid = os.path.basename(p2)[:-3]
        sec = int(os.stat(p2).st_mtime)
        os.utime(p2, (sec, sec))
        mt[cid] = sec

    rows_fixed = repair_summaries(mt)
    if os.path.isdir(ANN):
        orphan_ann_dir = os.path.join(BACKUPS, stamp, "orphan_annotations")
        for ann_path in sorted(glob.glob(os.path.join(glob.escape(ANN), "*.pbtxt"))):
            ann_name = os.path.basename(ann_path)
            ann_cid = ann_name[:-6]
            if ann_cid not in mt and ann_cid not in topic_cids:
                try:
                    os.makedirs(orphan_ann_dir, exist_ok=True)
                    shutil.move(ann_path, os.path.join(orphan_ann_dir, ann_name))
                except OSError:
                    pass
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
                ttl = topic_titles.get(cid) or read_annotation_title(CLI, cid) or (title or "").strip() or summary_annotation_title(raw)
                new = summary_ui_visible(raw, ttl, reset_running=True)
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
    if os.path.isdir(D1):
        for p1 in sorted(glob.glob(os.path.join(glob.escape(D1), "*.db"))):
            p2 = os.path.join(D2, os.path.basename(p1))
            if os.path.lexists(p2) or os.path.islink(p1):
                continue
            if checkpoint_db(p1):
                drop_wal_sidecars(p1, force=False)
                try:
                    os.link(p1, p2)
                    linked += 1
                except OSError as e:
                    log.warning("post: cannot link %s -> %s: %s", p1, p2, e)
                    bad += 1
    log.info("post: linked=%d already=%d problems=%d waited=%.0fs", linked, exists, bad, time.time() - start)
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


_DEFAULT_TOPIC_REGISTRY = get_topic_registry_path()
TOPIC_REGISTRY = _DEFAULT_TOPIC_REGISTRY


def _effective_topic_registry() -> str:
    return TOPIC_REGISTRY if TOPIC_REGISTRY != _DEFAULT_TOPIC_REGISTRY else get_topic_registry_path()


def mirrored_topic_cids() -> set[str]:
    """Conversation IDs owned by mirrored background topic threads (excluded from interactive idle detection)."""
    return get_active_topic_cids(_effective_topic_registry())


def topic_canonical_titles() -> dict[str, str]:
    """Canonical permanent titles for topic threads from cachedb/llm_topic_threads.json."""
    return get_topic_canonical_titles(_effective_topic_registry())


_GLOG_TS_RE = re.compile(r"^[IWEF](\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?")
_DAEMON_LOG_NOISE_MARKERS = (
    "file_watcher.go",
    "LoadTrajectory",
    "GetSlashCommands",
    "ensure trajectory",
    "Monitor sir_flash_lite SKIPPED",
    "Monitor severe_internal_risks_agy SKIPPED",
)
_DAEMON_ACTIVE_STREAM_MARKERS = (
    "SEND_USER_CASCADE_MESSAGE",
    "streamGenerateContent",
    "LLM Monitor",
    "Monitor agy_tool_safety",
)


def _glog_rel_ts(line: str) -> float | None:
    m = _GLOG_TS_RE.match(line)
    if not m:
        return None
    try:
        from datetime import datetime, timezone
        mo, dy, hh, mm, ss, us_raw = m.groups()
        us = int((us_raw or "0")[:6].ljust(6, "0"))
        return datetime(2024, int(mo), int(dy), int(hh), int(mm), int(ss), us, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


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
                    if name.endswith((".db", ".db-wal")) and cid not in ignored:
                        newest = max(newest, os.stat(p).st_mtime)
                except OSError:
                    pass
    if os.path.isdir(LOGS):
        try:
            cli_logs = sorted(glob.glob(os.path.join(glob.escape(LOGS), "cli-*.log")), key=os.path.getmtime, reverse=True)
            for candidate_log in cli_logs[:10]:
                txt = read_text_file(candidate_log, default="", errors="ignore")
                if not txt or ("Connection status:" not in txt and "summary store: reconciliation complete" not in txt):
                    continue
                conn_idx = txt.rfind("Connection status: Connected")
                if conn_idx >= 0:
                    all_lines = txt[conn_idx:].splitlines()
                    meaningful = [ln for ln in all_lines if not any(n in ln for n in _DAEMON_LOG_NOISE_MARKERS)]
                    recent_lines = meaningful[-2:]
                    matched_line = next(
                        (ln for ln in reversed(recent_lines) if any(mk in ln for mk in _DAEMON_ACTIVE_STREAM_MARKERS)),
                        None,
                    )
                    if matched_line is not None:
                        log_mtime = os.stat(candidate_log).st_mtime
                        t_match = _glog_rel_ts(matched_line)
                        t_last = next((t for ln in reversed(all_lines) if (t := _glog_rel_ts(ln)) is not None), None)
                        if t_match is not None and t_last is not None:
                            delta = t_last - t_match
                            if delta < -86400 * 180:
                                delta += 86400 * 366
                            if 0.0 <= delta <= 86400 * 30:
                                log_mtime -= delta
                        newest = max(newest, log_mtime)
                break
        except OSError:
            pass
    brain = os.path.join(CLI, "brain")
    now_ts = time.time()
    if os.path.isdir(brain):
        try:
            brain_cids = os.listdir(brain)
        except OSError:
            brain_cids = []
        from datetime import datetime
        for cid in brain_cids:
            if cid in ignored:
                continue
            tp = os.path.join(brain, cid, ".system_generated", "logs", "transcript.jsonl")
            try:
                st = os.stat(tp)
            except OSError:
                continue
            if (now_ts - st.st_mtime) > 86400:
                continue
            records = read_tail_jsonl_dicts(tp, limit=80)
            oneshot_until = 0.0
            cron_active = False
            sched_task_id = ""
            sched_cond = "never"
            awaiting_sched_task_id = False
            sched_calls = 0
            latest_user_input_ts = 0.0
            for rec in records:
                step_type = str(rec.get("type") or "")
                content = str(rec.get("content") or "")
                if step_type == "USER_INPUT":
                    sched_calls = 0
                    created_raw = str(rec.get("created_at") or "").strip()
                    if created_raw:
                        try:
                            latest_user_input_ts = max(
                                latest_user_input_ts,
                                datetime.fromisoformat(created_raw.replace("Z", "+00:00")).timestamp(),
                            )
                        except Exception:
                            latest_user_input_ts = max(latest_user_input_ts, st.st_mtime)
                if awaiting_sched_task_id and step_type == "GENERIC":
                    m_tid = re.search(r"background task with task id:\s*(\S+)", content)
                    if m_tid and ("Timer:" in content or "Cron:" in content or "schedule" in content):
                        sched_task_id = m_tid.group(1).strip()
                    awaiting_sched_task_id = False
                if step_type == "SYSTEM_MESSAGE":
                    if "stopped due to server restart" in content:
                        oneshot_until = 0.0
                        cron_active = False
                        sched_task_id = ""
                        sched_calls = 0
                    elif re.search(r"\(iteration\s+\d+\)\s*$", content):
                        cron_active = True
                        m_sender = re.search(r"\bsender=(\S+)", content)
                        if m_sender:
                            sched_task_id = m_sender.group(1).strip()
                    else:
                        m_sender = re.search(r"\bsender=(\S+)", content)
                        sender_id = m_sender.group(1).strip() if m_sender else ""
                        if (
                            not sender_id
                            or not sched_task_id
                            or sender_id == sched_task_id
                            or sched_cond == "any"
                            or sender_id == sched_cond
                        ):
                            oneshot_until = 0.0
                            sched_task_id = ""
                elif step_type == "USER_INPUT" and sched_cond == "any":
                    oneshot_until = 0.0
                    sched_task_id = ""
                for tc in rec.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    name = str(tc.get("name") or "")
                    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                    if name == "manage_task" and str(args.get("Action") or "").strip().strip('"') == "kill":
                        kill_tid = str(args.get("TaskId") or "").strip().strip('"')
                        if not kill_tid or not sched_task_id or kill_tid == sched_task_id or sched_task_id.endswith(f"/{kill_tid}"):
                            oneshot_until = 0.0
                            cron_active = False
                            sched_task_id = ""
                            sched_calls = 0
                    elif name == "schedule":
                        sched_calls += 1
                        created_raw = str(rec.get("created_at") or "").strip()
                        sched_ts = st.st_mtime
                        if created_raw:
                            try:
                                sched_ts = datetime.fromisoformat(created_raw.replace("Z", "+00:00")).timestamp()
                            except Exception:
                                sched_ts = st.st_mtime
                        dur_raw = str(args.get("DurationSeconds") or "").strip().strip('"')
                        cron_raw = str(args.get("CronExpression") or "").strip().strip('"')
                        cond_raw = str(args.get("TimerCondition") or "never").strip().strip('"').lower()
                        dur_s = safe_pos_float(dur_raw, default=None)
                        if dur_s is not None:
                            oneshot_until = max(oneshot_until, sched_ts + dur_s + 120.0)
                            sched_cond = cond_raw or "never"
                            sched_task_id = ""
                            awaiting_sched_task_id = True
                        elif cron_raw:
                            cron_active = True
                            sched_task_id = ""
                            awaiting_sched_task_id = True
            if cron_active or now_ts < oneshot_until:
                is_repeating_loop = sched_calls >= 2
                is_stale_user_input = not cron_active and latest_user_input_ts > 0.0 and (now_ts - latest_user_input_ts) > 1800.0
                if is_repeating_loop or is_stale_user_input:
                    continue
                return now_ts
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


def mode_ui_sync_check(idle_sec: int, min_interval_sec: float = 0.0) -> int:
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
    now = time.time()
    min_itv_f = safe_pos_float(min_interval_sec, default=0.0) or 0.0
    if min_itv_f > 0.0:
        started_ts = 0.0
        if os.path.isdir(LOGS):
            for ts, lf in sorted(
                ((log_started_at(lf), lf) for lf in glob.glob(os.path.join(glob.escape(LOGS), "cli-*.log"))),
                reverse=True,
            ):
                if ts <= 0.0:
                    break
                txt = read_text_file(lf, default=None, errors="ignore")
                if txt and ("Connection status:" in txt or "summary store: reconciliation complete" in txt):
                    started_ts = ts
                    break
        uptime_for = now - started_ts if started_ts > 0.0 else min_itv_f
        if 0.0 <= uptime_for < max(0.0, min_itv_f - 60.0):
            log.info("ui-sync-check: %d pending, daemon uptime %.0fs (< %gs cadence); waiting", len(entries), uptime_for, min_itv_f)
            return UI_SYNC_BUSY
    last_act = last_store_activity()
    now = max(now, time.time())
    idle_for = max(0.0, now - last_act)
    if idle_for < idle_sec_f:
        log.info("ui-sync-check: %d pending, store active %.0fs ago (< %gs); waiting", len(entries), idle_for, idle_sec_f)
        return UI_SYNC_BUSY
    fingerprint = hashlib.sha1("\n".join(entries).encode()).hexdigest()
    state = read_json_dict(UI_SYNC_STATE, default={}, recover_previous=True)
    if state.get("fingerprint") == fingerprint:
        log.warning("ui-sync-check: same %d pending entries persisted after the last restart; not restarting again: %s",
                    len(entries), entries[:5])
        return UI_SYNC_STUCK
    save_json_state(UI_SYNC_STATE, {"fingerprint": fingerprint, "ts": now, "entries": entries}, indent=1)
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
                d1 = {f for f in os.listdir(D1) if f.endswith(".db")} if os.path.isdir(D1) else set()
                d2 = {f for f in os.listdir(D2) if f.endswith(".db")} if os.path.isdir(D2) else set()
                linked = sum(1 for f in d1 & d2 if same_file(os.path.join(D1, f), os.path.join(D2, f)))
                if not os.path.isfile(AGY_BIN):
                    mesh_st = "missing"
                elif (agy_bytes := read_bytes_file(AGY_BIN, default=None)) is None:
                    mesh_st = "unreadable"
                elif WEBRTC_FLAG_ORIG in agy_bytes:
                    mesh_st = "ENABLED (unpatched)"
                elif WEBRTC_FLAG_PATCHED in agy_bytes:
                    mesh_st = "disabled (patched)"
                else:
                    mesh_st = "unknown"
                print(f"visible dir files={len(d2)}  content dir files={len(d1)}  hard-linked={linked}  only-in-content-dir={len(d1 - d2)}  webrtc_mesh={mesh_st}")
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
            if mode == "ui-sync-check":
                if _ROOT_DIR not in sys.path:
                    sys.path.insert(0, _ROOT_DIR)
                from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals
                ssot_itvs = resolve_derived_telemetry_intervals()
                idle_sec = float(ssot_itvs.get("ui_sync_idle_sec", 120.0))
                if "--idle-sec" in argv and argv.index("--idle-sec") + 1 < len(argv):
                    idle_sec = float(argv[argv.index("--idle-sec") + 1])
                min_itv = float(ssot_itvs.get("ui_sync_interval_sec", 720.0))
                if "--min-interval-sec" in argv and argv.index("--min-interval-sec") + 1 < len(argv):
                    min_itv = safe_pos_float(argv[argv.index("--min-interval-sec") + 1], default=0.0) or 0.0
                return mode_ui_sync_check(idle_sec, min_interval_sec=min_itv)
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
