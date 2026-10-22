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

Safety: a file is never deleted unless the identical inode/content already exists in the visible dir; anything that
diverges is moved into ~/agy_store_fix_backups/<timestamp>/ instead of being overwritten.
"""
from __future__ import annotations

import fcntl
import glob
import io
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

HOME = os.environ.get("AGY_FIX_HOME") or os.path.expanduser("~")
CLI = os.path.join(HOME, ".gemini", "antigravity-cli")
D1 = os.path.join(CLI, "conversations")                       # content dir (source forced to 17 by the daemon)
D2 = os.path.join(HOME, ".gemini", "antigravity", "conversations")  # visible dir (source stays 0)
ANN = os.path.join(CLI, "annotations")
DB = os.path.join(CLI, "conversation_summaries.db")
PB = os.path.join(CLI, "agyhub_summaries_proto.pb")
LOGS = os.path.join(CLI, "log")
LOCK = os.path.join(HOME, ".agy_store_fix.lock")
LOG_FILE = os.path.join(HOME, "agy_store_fix.log")
BACKUPS = os.path.join(HOME, "agy_store_fix_backups")
UNIT = "antigravity-cli-daemon.service"

log = logging.getLogger("agy_store_fix")


# ---------------------------------------------------------------- protobuf helpers
def enc_varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def dec_varint(buf: bytes, pos: int):
    val = shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, pos
        shift += 7
    raise ValueError("truncated varint")


def parse_fields(buf: bytes):
    pos, out = 0, []
    while pos < len(buf):
        key, pos = dec_varint(buf, pos)
        tag, wire = key >> 3, key & 7
        if wire == 0:
            v, pos = dec_varint(buf, pos)
        elif wire == 2:
            n, pos = dec_varint(buf, pos)
            v = buf[pos:pos + n]
            pos += n
        elif wire == 1:
            v = buf[pos:pos + 8]
            pos += 8
        elif wire == 5:
            v = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((tag, wire, v))
    return out


def enc_fields(fields) -> bytes:
    out = bytearray()
    for tag, wire, v in fields:
        out += enc_varint((tag << 3) | wire)
        if wire == 0:
            out += enc_varint(int(v))
        elif wire == 2:
            out += enc_varint(len(v)) + bytes(v)
        else:
            out += bytes(v)
    return bytes(out)


def fix_summary(raw: bytes, title: str) -> bytes:
    """Drop field 20 (source) and make sure annotations (15) carry the title; keep other annotation sub-fields."""
    keep, annot_sub = [], []
    for tag, wire, v in parse_fields(raw):
        if tag == 20:
            continue
        if tag == 15 and wire == 2:
            annot_sub = [x for x in parse_fields(v) if x[0] != 1]   # keep e.g. last_user_view_time
            continue
        if tag == 28 and wire == 0 and v == 2:   # stale RUNNING -> IDLE
            keep.append((28, 0, 4))
            continue
        keep.append((tag, wire, v))
    if title:
        annot_sub = [(1, 2, title.encode())] + [x for x in annot_sub if x[0] != 1]
    if annot_sub:
        keep.append((15, 2, enc_fields(annot_sub)))
    keep.sort(key=lambda x: x[0])
    return enc_fields(keep)


def raw_title(raw: bytes) -> str:
    for tag, wire, v in parse_fields(raw):
        if tag == 15 and wire == 2:
            for t, w, x in parse_fields(v):
                if t == 1 and w == 2:
                    return x.decode("utf-8", "ignore").strip()
    return ""


def pbtxt_title(cid: str) -> str:
    try:
        txt = open(os.path.join(ANN, cid + ".pbtxt"), encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""
    m = re.search(r'title:\s*"((?:[^"\\]|\\.)*)"', txt)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\").strip() if m else ""


# ---------------------------------------------------------------- file helpers
def checkpoint(path: str) -> bool:
    """Flush the WAL into the main file. Returns False if it could not be done safely."""
    wal = path + "-wal"
    try:
        if not os.path.exists(wal) or os.path.getsize(wal) == 0:
            return True
        c = sqlite3.connect(path, timeout=15)
        row = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        c.close()
        return bool(row) and row[0] == 0
    except sqlite3.Error as e:
        log.warning("checkpoint failed for %s: %s", path, e)
        return False


def drop_sidecars(path: str) -> None:
    for ext in ("-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            try:
                if ext == "-wal" and os.path.getsize(p) > 0:
                    log.warning("refusing to drop non-empty %s", p)
                    continue
                os.remove(p)
            except OSError as e:
                log.warning("cannot remove %s: %s", p, e)


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


# ---------------------------------------------------------------- pre
def mode_pre(force: bool = False) -> int:
    if not force and os.environ.get("AGY_FIX_HOME") is None and unit_active():
        log.error("daemon is active; refusing to run 'pre' (use --force to override)")
        return 2
    os.makedirs(D2, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    moved = unlinked = diverged = skipped = 0

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

    # leftover WAL on the real files
    for p2 in glob.glob(os.path.join(D2, "*.db")):
        checkpoint(p2)

    # normalise mtimes to whole seconds so stored fileModTimes (seconds) always >= disk mtime
    mt = {}
    for p2 in glob.glob(os.path.join(D2, "*.db")):
        cid = os.path.basename(p2)[:-3]
        sec = int(os.stat(p2).st_mtime)
        os.utime(p2, (sec, sec))
        mt[cid] = sec

    rows_fixed = repair_summaries(mt)
    log.info("pre: moved=%d unlinked=%d diverged=%d skipped=%d rows_fixed=%d visible_files=%d",
             moved, unlinked, diverged, skipped, rows_fixed, len(mt))
    return 0


def repair_summaries(mt: dict) -> int:
    if not os.path.exists(DB):
        return 0
    con = sqlite3.connect(DB, timeout=15)
    rows = con.execute("SELECT conversation_id, title, raw_summary FROM conversation_summaries").fetchall()
    raws, fixed = {}, 0
    for cid, title, raw in rows:
        raw = bytes(raw or b"")
        if cid in mt:
            ttl = (title or "").strip() or raw_title(raw) or pbtxt_title(cid)
            new = fix_summary(raw, ttl)
            con.execute(
                "UPDATE conversation_summaries SET title=?, source='', status='CASCADE_RUN_STATUS_IDLE', "
                "not_fully_idle=0, raw_summary=? WHERE conversation_id=?", (ttl, new, cid))
            raws[cid] = new
            fixed += 1
        else:
            raws[cid] = raw
    con.commit()
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass
    con.close()

    # rebuild agyhub_summaries_proto.pb: tag1 = summaries, tag2 = fileModTimes (seconds). Keep only existing mt files.
    out = bytearray()
    for cid, raw in raws.items():
        if cid in mt:
            out += enc_fields([(1, 2, enc_fields([(1, 2, cid.encode()), (2, 2, raw)]))])
    for cid, sec in mt.items():
        out += enc_fields([(2, 2, enc_fields([(1, 2, cid.encode()), (2, 0, sec)]))])
    tmp = PB + ".tmp"
    with open(tmp, "wb") as f:
        f.write(bytes(out))
    os.replace(tmp, PB)
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
    for lf in glob.glob(os.path.join(LOGS, "cli-*.log")):
        if not pid and log_started_at(lf) < since:
            continue
        try:
            txt = open(lf, errors="ignore").read()
        except OSError:
            continue
        if pid and f"language server process with pid {pid}" not in txt:
            continue
        if "summary store: reconciliation complete" in txt:
            return True
    return False


def mode_post(timeout: int = 50) -> int:
    start = time.time()
    deadline = start + timeout
    ready = False
    while time.time() < deadline:
        pid = main_pid()
        if newest_daemon_log_ready(start - 15, pid):
            ready = True
            break
        time.sleep(1)
    if not ready:
        log.warning("post: daemon startup scan not confirmed within %ds; continuing after a fixed wait", timeout)
        time.sleep(10)
    time.sleep(6)   # let the initial scan/refresh settle before adding files to the watched dir
    linked = exists = bad = 0
    for p2 in sorted(glob.glob(os.path.join(D2, "*.db"))):
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
    print(f"visible dir files={len(d2)}  content dir files={len(d1)}  hard-linked={linked}  only-in-content-dir={len(d1 - d2)}")
    if os.path.exists(DB):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        import collections
        cnt = collections.Counter((r[0][:8], r[1] or "unset", "titled" if r[2] else "untitled") for r in
                                  c.execute("select project_id, source, title from conversation_summaries"))
        for k, v in sorted(cnt.items()):
            print(" ", v, k)
    return 0


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another agy_store_fix is running")
        return 3
    try:
        if mode == "pre":
            return mode_pre(force="--force" in argv)
        if mode == "post":
            return mode_post()
        if mode == "check":
            return mode_check()
        print(__doc__)
        return 1
    except Exception:
        log.exception("agy_store_fix %s failed", mode)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
