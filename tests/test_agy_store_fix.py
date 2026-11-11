"""Unit tests for agy_store_fix ui-sync-check (idle-gated daemon restart decision)."""
import os
import sqlite3
import time

import pytest

from orchestratorOS.admin import agy_store_fix as asf
from datetime import datetime, timezone
import json


def _pb_with_cids(cids):
    out = b""
    for cid in cids:
        entry = asf.enc_fields([(1, 2, cid.encode()), (2, 2, b"\x0a\x01t")])
        out += asf.enc_fields([(1, 2, entry)])
    return out


@pytest.fixture
def store(tmp_path, monkeypatch):
    cli = tmp_path / ".gemini" / "antigravity-cli"
    d1 = cli / "conversations"
    d2 = tmp_path / ".gemini" / "antigravity" / "conversations"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(asf, "CLI", str(cli))
    monkeypatch.setattr(asf, "D1", str(d1))
    monkeypatch.setattr(asf, "D2", str(d2))
    monkeypatch.setattr(asf, "LOGS", str(cli / "log"))
    monkeypatch.setattr(asf, "DB", str(cli / "conversation_summaries.db"))
    monkeypatch.setattr(asf, "PB", str(cli / "agyhub_summaries_proto.pb"))
    monkeypatch.setattr(asf, "UI_SYNC_STATE", str(tmp_path / "ui_sync_state.json"))
    monkeypatch.setattr(asf, "TOPIC_REGISTRY", str(tmp_path / "llm_topic_threads.json"))
    conn = sqlite3.connect(asf.DB)
    conn.execute("CREATE TABLE conversation_summaries (conversation_id text primary key)")
    conn.commit()
    conn.close()
    return cli, d1, d2


def _set_rows(cids):
    conn = sqlite3.connect(asf.DB)
    conn.execute("DELETE FROM conversation_summaries")
    conn.executemany("INSERT INTO conversation_summaries VALUES (?)", [(c,) for c in cids])
    conn.commit()
    conn.close()


def _age_store(d1, seconds):
    old = time.time() - seconds
    for name in os.listdir(d1):
        os.utime(os.path.join(d1, name), (old, old))


def test_visible_dir_is_sibling_of_cli_store():
    assert asf.visible_conversations_dir("/h/.gemini/antigravity-cli") == "/h/.gemini/antigravity/conversations"


def test_nothing_pending_when_store_matches_daemon(store):
    cli, d1, d2 = store
    (d2 / "a.db").write_text("x")
    os.link(d2 / "a.db", d1 / "a.db")
    _set_rows(["a"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["a"]))
    assert asf.pending_ui_sync() == []
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_NOTHING


def test_new_gone_and_rewritten_content_are_pending(store):
    cli, d1, d2 = store
    (d2 / "kept.db").write_text("x")
    os.link(d2 / "kept.db", d1 / "kept.db")
    (d1 / "new.db").write_text("y")          # mirrored, daemon does not know it yet
    (d1 / "rewritten.db").write_text("z")    # known, but content replaced (no longer linked)
    (d2 / "rewritten.db").write_text("old")
    _set_rows(["kept", "new", "rewritten"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["kept", "rewritten", "gone"]))
    kinds = sorted(e.split(":")[0] + ":" + e.split(":")[1] for e in asf.pending_ui_sync())
    assert kinds == ["content:new", "content:rewritten", "gone:gone", "new:new"]


def test_restart_gated_by_idle_and_not_repeated_for_same_divergence(store):
    cli, d1, _ = store
    (d1 / "new.db").write_text("y")
    _set_rows(["new"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))

    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_BUSY      # just written -> active
    _age_store(d1, 1000)
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY     # idle -> restart due
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_STUCK     # same divergence after restart -> no loop

    (d1 / "other.db").write_text("w")                           # a new divergence re-arms the restart
    _set_rows(["new", "other"])
    _age_store(d1, 1000)
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY


def test_invalid_idle_fails_fast(store, tmp_path):
    with pytest.raises(ValueError):
        asf.mode_ui_sync_check(0)
    with pytest.raises(ValueError):
        asf.mode_ui_sync_check(True)

    # Non-dict JSON in UI_SYNC_STATE does not raise AttributeError
    cli, d1, _ = store
    (d1 / "new.db").write_text("y")
    _set_rows(["new"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1000)
    (tmp_path / "ui_sync_state.json").write_text("[]")
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY


def test_mirrored_5min_topic_thread_does_not_block_idle_sync(store, tmp_path):

    cli, d1, _ = store
    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps({"auto_trade_executor": {"conversation_id": "topic-5m"}})
    )
    (d1 / "human.db").write_text("h")
    _age_store(d1, 1200)                         # human session has been idle for 20 minutes
    (d1 / "topic-5m.db").write_text("bot")       # 5-minute bot thread just mirrored now
    (cli / "brain" / "topic-5m").mkdir(parents=True, exist_ok=True)
    (cli / "brain" / "topic-5m" / "step.txt").write_text("bot")
    _set_rows(["human", "topic-5m"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["human"]))

    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY


def test_orphan_summary_row_without_db_file_does_not_trigger_new_sync(store):
    cli, d1, d2 = store
    (d2 / "kept.db").write_text("x")
    os.link(d2 / "kept.db", d1 / "kept.db")
    _set_rows(["kept", "orphan_deleted_in_ui"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["kept"]))
    assert asf.pending_ui_sync() == []
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_NOTHING


def test_stale_running_status_does_not_deadlock_ui_sync_while_fresh_running_blocks(store, tmp_path):

    cli, d1, d2 = store
    conn = sqlite3.connect(asf.DB)
    conn.execute("DROP TABLE conversation_summaries")
    conn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text primary key, title text, source text, status text, "
        "not_fully_idle integer, raw_summary blob)"
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES ('stale-run', 'ntfy-guard: old', '', 'CASCADE_RUN_STATUS_RUNNING', 1, ?)",
        (b"\x0a\x01t",),
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES ('new-topic', 'ntfy-macro: old', '', 'CASCADE_RUN_STATUS_IDLE', 0, ?)",
        (b"\x0a\x01t",),
    )
    conn.commit()
    conn.close()

    (d1 / "stale-run.db").write_text("x")
    (d1 / "new-topic.db").write_text("y")
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["stale-run"]))

    # Freshly modified (<300s) RUNNING conversation blocks ui-sync
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_BUSY

    # Once both files are older than 900s, a stale RUNNING status from a crashed session must not deadlock ui-sync
    _age_store(d1, 1200)
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY

    # An IDLE conversation with not_fully_idle=1 (e.g. sleeping between repeating timers) aged > idle_sec (180s > 120s, < 900s) does not block ui-sync
    conn = sqlite3.connect(asf.DB)
    conn.execute("UPDATE conversation_summaries SET status='CASCADE_RUN_STATUS_IDLE', not_fully_idle=1 WHERE conversation_id='stale-run'")
    conn.commit()
    conn.close()
    _age_store(d1, 180)
    (tmp_path / "ui_sync_state.json").unlink(missing_ok=True)
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY
    _age_store(d1, 1200)

    # And repair_summaries restores canonical topic titles from TOPIC_REGISTRY / TOPIC_SPECS over stale ntfy-* titles
    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps({"macro_sentiment": {"conversation_id": "new-topic"}})
    )
    asf.repair_summaries({"stale-run": 100, "new-topic": 100})
    conn = sqlite3.connect(asf.DB)
    rows = dict(conn.execute("SELECT conversation_id, title FROM conversation_summaries").fetchall())
    conn.close()
    assert rows["new-topic"] == "🧭 [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)"


def test_active_webui_daemon_log_or_new_running_thread_blocks_ui_sync(store):
    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1200)

    # 1. Active Web UI session in daemon log (after 'Connection status: Connected') blocks ui-sync
    log_dir = cli / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cli-20261024_221208.log"
    log_file.write_text(
        "I1024 22:12:13.000000 1 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1024 22:15:00.000000 2 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse\n"
    )
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_BUSY

    # Once the daemon log is also older than 900s, ui-sync is ready
    old = time.time() - 1200
    os.utime(log_file, (old, old))
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY


def test_known_mirrored_topic_relinked_or_new_webui_thread_does_not_trigger_restart(store):
    cli, d1, d2 = store
    # 1. Known permanent topic thread updated in D1 (unlinked from D2) queues content sync, while
    #    brand-new interactive Web UI thread created directly by the running daemon in D1 does not.
    (d1 / "topic-10m.db").write_text("new-turn-content")
    os.link(d1 / "topic-10m.db", d2 / "topic-10m.db")
    (d1 / "webui-chat.db").write_text("user-chat")
    _set_rows(["topic-10m", "webui-chat"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["topic-10m", "webui-chat"]))
    _age_store(d1, 1200)

    assert asf.pending_ui_sync() == []
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_NOTHING

    # 2. When a mirrored topic turn replaces D1/topic-10m.db and drops D2/topic-10m.db, pending_ui_sync detects it
    os.remove(d2 / "topic-10m.db")
    (d1 / "new-topic.db").write_text("first-turn")
    _set_rows(["topic-10m", "webui-chat", "new-topic"])
    with open(asf.TOPIC_REGISTRY, "w", encoding="utf-8") as f:
        f.write(
            '{"auto_cryptotrade_executor": {"conversation_id": "new-topic"}, '
            '"auto_sharetrade_executor": {"conversation_id": "topic-10m"}}'
        )
    log_dir = cli / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cli-20261026_100000.log"
    log_file.write_text(
        "I1026 10:00:01.000000 1 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1026 10:01:00.000000 2 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse\n"
        "I1026 10:05:00.000000 3 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist\n"
        "I1026 10:05:01.000000 4 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels\n"
    )
    assert len(asf.pending_ui_sync()) == 3
    # When daemon just started (< 720s cadence), min_interval_sec defers the restart
    recent_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() - 180))
    recent_log = log_dir / f"cli-{recent_stamp}.log"
    recent_log.write_text("I1102 23:00:00.000000 1 remote_control_v2.go:2056] Connection status: Connected\n")
    assert asf.mode_ui_sync_check(120, min_interval_sec=720.0) == asf.UI_SYNC_BUSY
    # Calling main(["agy_store_fix.py", "ui-sync-check"]) without explicit flags uses SSOT defaults (120s idle, 720s cadence)
    assert asf.main(["agy_store_fix.py", "ui-sync-check"]) == asf.UI_SYNC_BUSY
    os.remove(recent_log)
    # One-shot non-daemon CLI logs (without remote_control_v2 / reconciliation markers) do not reset daemon uptime
    oneshot_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() - 60))
    oneshot_log = log_dir / f"cli-{oneshot_stamp}.log"
    oneshot_log.write_text("I1102 23:02:00.000000 1 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist\n")
    assert asf.mode_ui_sync_check(900, min_interval_sec=720.0) == asf.UI_SYNC_READY


def test_active_schedule_timer_in_interactive_transcript_blocks_ui_sync(store):

    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1500)

    log_dir = cli / "brain" / "interactive-cid" / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tp = log_dir / "transcript.jsonl"
    sched_ts = time.time() - 1000.0  # 16.6m into a 20m (1200s) schedule timer
    sched_iso = datetime.fromtimestamp(sched_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tp.write_text(
        json.dumps(
            {
                "step_index": 10,
                "type": "PLANNER_RESPONSE",
                "created_at": sched_iso,
                "tool_calls": [{"name": "schedule", "args": {"DurationSeconds": "1200", "Prompt": "wake"}}],
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # Unrelated background task SYSTEM_MESSAGE or manage_task kill does not cancel a TimerCondition="never" schedule timer
    tp.write_text(
        json.dumps(
            {
                "step_index": 10,
                "type": "PLANNER_RESPONSE",
                "created_at": sched_iso,
                "tool_calls": [
                    {"name": "schedule", "args": {"DurationSeconds": "1200", "Prompt": "wake", "TimerCondition": "never"}}
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 11,
                "type": "GENERIC",
                "content": "Tool is running as a background task with task id: interactive-cid/task-sched-1\nTask Description: Timer: 20m",
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 12,
                "type": "SYSTEM_MESSAGE",
                "content": "[Message] sender=interactive-cid/task-other-cmd Task finished",
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 13,
                "type": "PLANNER_RESPONSE",
                "tool_calls": [{"name": "manage_task", "args": {"Action": "kill", "TaskId": "task-other-cmd"}}],
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # Once the matching timer fires (SYSTEM_MESSAGE appended) and store is aged past idle_sec, ui-sync proceeds
    tp.write_text(
        tp.read_text()
        + json.dumps(
            {
                "step_index": 14,
                "type": "SYSTEM_MESSAGE",
                "content": "[Message] sender=interactive-cid/task-sched-1 timer fired",
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY


def test_repeating_schedule_loop_does_not_block_ui_sync(store):
    """An autonomous loop with multiple schedule calls must not permanently starve UI sync."""

    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1500)

    log_dir = cli / "brain" / "looping-cid" / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tp = log_dir / "transcript.jsonl"
    sched_ts = time.time() - 300.0  # 5m ago (within 20m timer)
    sched_iso = datetime.fromtimestamp(sched_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tp.write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "created_at": sched_iso,
                "tool_calls": [{"name": "schedule", "args": {"DurationSeconds": "1200", "Prompt": "wake 1"}}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "created_at": sched_iso,
                "tool_calls": [{"name": "schedule", "args": {"DurationSeconds": "1200", "Prompt": "wake 2"}}],
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY


def test_stale_user_input_with_active_timer_does_not_block_ui_sync(store):
    """A conversation where user input occurred >30m ago does not block UI sync even if a timer is active."""

    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1500)

    log_dir = cli / "brain" / "stale-cid" / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tp = log_dir / "transcript.jsonl"
    user_ts = time.time() - 2400.0  # 40m ago (> 30m)
    user_iso = datetime.fromtimestamp(user_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sched_ts = time.time() - 300.0  # 5m ago
    sched_iso = datetime.fromtimestamp(sched_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tp.write_text(
        json.dumps(
            {
                "step_index": 0,
                "type": "USER_INPUT",
                "created_at": user_iso,
                "content": "do recurring check",
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "created_at": sched_iso,
                "tool_calls": [{"name": "schedule", "args": {"DurationSeconds": "1200", "Prompt": "wake"}}],
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY

    # A fresh USER_INPUT after an earlier schedule resets sched_calls so a new timer blocks UI sync
    fresh_user_ts = time.time() - 60.0
    fresh_user_iso = datetime.fromtimestamp(fresh_user_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tp.write_text(
        tp.read_text()
        + json.dumps(
            {
                "step_index": 2,
                "type": "USER_INPUT",
                "created_at": fresh_user_iso,
                "content": "restart timer",
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 3,
                "type": "PLANNER_RESPONSE",
                "created_at": fresh_user_iso,
                "tool_calls": [{"name": "schedule", "args": {"CronExpression": "*/20 * * * *", "Prompt": "wake"}}],
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # Even after the original schedule call scrolls past the 80-record tail, a first-iteration cron wakeup keeps cron_active=True
    tp.write_text(
        json.dumps(
            {
                "step_index": 200,
                "type": "SYSTEM_MESSAGE",
                "content": "[Message] timestamp=2026-11-10T12:17:32Z sender=stale-cid/task-7417 priority=MESSAGE_PRIORITY_HIGH content=Run audit (iteration 1)",
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # Once a cron loop reaches iteration >= 2 (even wrapped in XML <SYSTEM_MESSAGE> tags), it is treated as an autonomous repeating loop and must not starve periodic UI sync
    tp.write_text(
        json.dumps(
            {
                "step_index": 201,
                "type": "SYSTEM_MESSAGE",
                "content": "The following is a <SYSTEM_MESSAGE> not actually sent by the user.\n<SYSTEM_MESSAGE>\n[Message] timestamp=2026-11-10T12:37:32Z sender=stale-cid/task-7417 priority=MESSAGE_PRIORITY_HIGH content=Run audit (iteration 8)\n</SYSTEM_MESSAGE>",
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (sched_ts, sched_ts))
    (cli.parent.parent / "ui_sync_state.json").unlink(missing_ok=True)
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY

    # Likewise, a zombie pre-restart transcript older than the current daemon startup log must not block UI sync
    tp.write_text(
        json.dumps(
            {
                "step_index": 202,
                "type": "SYSTEM_MESSAGE",
                "content": "[Message] timestamp=2026-11-10T12:17:32Z sender=stale-cid/task-7417 priority=MESSAGE_PRIORITY_HIGH content=Run audit (iteration 1)",
            }
        )
        + "\n"
    )
    os.utime(tp, (sched_ts, sched_ts))
    daemon_log_dir = cli / "log"
    daemon_log_dir.mkdir(parents=True, exist_ok=True)
    restart_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(sched_ts + 60.0))
    daemon_log = daemon_log_dir / f"cli-{restart_stamp}.log"
    daemon_log.write_text("I1110 12:18:32.000000 1 remote_control_v2.go:2056] Connection status: Connected\n", encoding="utf-8")
    os.utime(daemon_log, (sched_ts, sched_ts))
    (cli.parent.parent / "ui_sync_state.json").unlink(missing_ok=True)
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY


def test_mode_pre_protects_topic_cids_and_archives_orphan_annotations(store, tmp_path, monkeypatch):

    cli, d1, d2 = store
    ann = cli / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    backups = tmp_path / "backups"
    monkeypatch.setattr(asf, "ANN", str(ann))
    monkeypatch.setattr(asf, "BACKUPS", str(backups))
    monkeypatch.setattr(asf, "AGY_BIN", str(tmp_path / "nonexistent_agy"))
    monkeypatch.setenv("AGY_FIX_HOME", str(tmp_path))

    conn = sqlite3.connect(asf.DB)
    conn.execute("DROP TABLE conversation_summaries")
    conn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text primary key, title text, source text, status text, "
        "not_fully_idle integer, raw_summary blob)"
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES ('kept-ui', 'UI Chat', '', 'CASCADE_RUN_STATUS_IDLE', 0, ?)",
        (b"\x0a\x01t",),
    )
    conn.commit()
    conn.close()

    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps({"auto_cryptotrade_executor": {"conversation_id": "topic-perm"}}),
        encoding="utf-8",
    )
    (d2 / "kept-ui.db").write_text("ui")
    (d2 / "topic-perm.db").write_text("topic")
    (d2 / "deleted-ui.db").write_text("deleted")
    (ann / "kept-ui.pbtxt").write_text('title:"UI Chat"\n')
    (ann / "topic-perm.pbtxt").write_text('title:"Topic"\n')
    (ann / "stale-orphan.pbtxt").write_text('title:"Orphan"\n')

    assert asf.mode_pre(force=True) == 0
    # Kept UI conversation and permanent topic thread stay in D2; UI-deleted conversation is archived
    assert (d2 / "kept-ui.db").exists()
    assert (d2 / "topic-perm.db").exists()
    assert not (d2 / "deleted-ui.db").exists()
    # Orphan annotation without .db is archived into orphan_annotations, while active/topic annotations remain
    assert (ann / "kept-ui.pbtxt").exists()
    assert (ann / "topic-perm.pbtxt").exists()
    assert not (ann / "stale-orphan.pbtxt").exists()
    assert list(backups.rglob("orphan_annotations/stale-orphan.pbtxt"))


def test_last_store_activity_handles_oneshot_cli_logs_and_file_watcher_noise(store):
    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1500)

    log_dir = cli / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    daemon_log = log_dir / "cli-20261110_202010.log"
    noise = "".join(
        f"E1110 21:06:58.{500000 + i:06d} 89 file_watcher.go:194] skipping empty or temp file:\n"
        for i in range(40)
    )
    daemon_log.write_text(
        "I1110 20:20:11.000000 1 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1110 21:06:58.300000 2 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse\n"
        + noise,
        encoding="utf-8",
    )
    # Newer non-daemon CLI log (e.g. `agy remote-control status`) must NOT mask the active daemon stream
    oneshot_log = log_dir / "cli-20261110_210659.log"
    oneshot_log.write_text("I1110 21:06:59.000000 1 status.go:10] Remote-control status queried\n", encoding="utf-8")
    now = time.time()
    os.utime(daemon_log, (now - 1, now - 1))
    os.utime(oneshot_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # If file_watcher.go appends noise 20 minutes AFTER streamGenerateContent (updating daemon_log mtime to now),
    # the intra-log glog timestamp delta recognizes streamGenerateContent happened 20m ago -> UI_SYNC_READY!
    daemon_log.write_text(
        daemon_log.read_text(encoding="utf-8")
        + "E1110 21:26:58.300000 89 file_watcher.go:194] skipping empty or temp file:\n",
        encoding="utf-8",
    )
    os.utime(daemon_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY


def test_rotated_topic_preserves_ghost_trajectory_until_mode_pre_reaps_it(store, tmp_path, monkeypatch):
    from orchestratorOS.agy.protobuf import summary_ui_visible
    from orchestratorOS.agy.topics import topic_spec
    from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui

    cli, d1, d2 = store
    ann = cli / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    backups = tmp_path / "backups"
    monkeypatch.setattr(asf, "ANN", str(ann))
    monkeypatch.setattr(asf, "BACKUPS", str(backups))
    monkeypatch.setattr(asf, "AGY_BIN", str(tmp_path / "nonexistent_agy"))
    monkeypatch.setenv("AGY_FIX_HOME", str(tmp_path))
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(cli))

    old_cid = "6a3d50d3-f1d2-411f-925e-73412ad73434"
    new_cid = "fa9827b7-92b9-40a4-962b-aea3a159aca4"
    topic_title = topic_spec("auto_cryptotrade_executor").title
    raw_old = summary_ui_visible(b"\x0a\x03old", topic_title)

    # Running Web UI daemon currently holds old_cid in memory (agyhub_summaries_proto.pb)
    pb_entry = asf.enc_fields([(1, 2, old_cid.encode()), (2, 2, raw_old)])
    (cli / "agyhub_summaries_proto.pb").write_bytes(asf.enc_fields([(1, 2, pb_entry)]))

    conn = sqlite3.connect(asf.DB)
    conn.execute("DROP TABLE conversation_summaries")
    conn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text primary key, title text, source text, status text, "
        "project_id text, step_count integer, not_fully_idle integer, raw_summary blob)"
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, '', 'CASCADE_RUN_STATUS_IDLE', 'autonommptrade', 120, 0, ?)",
        (old_cid, topic_title, raw_old),
    )
    conn.commit()
    conn.close()

    # Bot store rotates to new_cid with fresh steps
    bot_base = tmp_path / "bot_home" / ".gemini" / "antigravity-cli"
    (bot_base / "conversations").mkdir(parents=True, exist_ok=True)
    bot_conv_db = bot_base / "conversations" / f"{new_cid}.db"
    bconn = sqlite3.connect(str(bot_conv_db))
    bconn.execute("CREATE TABLE trajectory_meta (trajectory_id text primary key, cascade_id text, trajectory_type integer, source integer)")
    bconn.execute("CREATE TABLE steps (idx integer primary key, step_payload blob)")
    bconn.execute("INSERT INTO trajectory_meta VALUES ('traj-1', ?, 4, 17)", (new_cid,))
    bconn.execute("INSERT INTO steps VALUES (0, ?)", (f"turn:{new_cid}:HOLD".encode("utf-8"),))
    bconn.commit()
    bconn.close()

    bot_sum_db = bot_base / "conversation_summaries.db"
    bsconn = sqlite3.connect(str(bot_sum_db))
    bsconn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text primary key, title text, source text, status text, "
        "project_id text, step_count integer, not_fully_idle integer, raw_summary blob)"
    )
    bsconn.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, 'CORTEX_TRAJECTORY_SOURCE_CLI', 'CASCADE_RUN_STATUS_IDLE', 'autonommptrade', 4, 0, ?)",
        (new_cid, topic_title, b"\x0a\x03new"),
    )
    bsconn.commit()
    bsconn.close()

    (tmp_path / "llm_topic_threads.json").write_text(
        json.dumps({"auto_cryptotrade_executor": {"conversation_id": new_cid, "title": topic_title}}),
        encoding="utf-8",
    )

    assert mirror_topic_thread_to_ui(new_cid, topic_title, bot_cli_base=str(bot_base), ui_cli_base=str(cli)) is True

    # Before daemon restart: ghost old_cid.db exists in D1 with cascade_id rewritten to old_cid and annotation preserved!
    ghost_db = d1 / f"{old_cid}.db"
    assert ghost_db.exists()
    assert asf.read_annotation_title(str(cli), old_cid) == topic_title
    gconn = sqlite3.connect(str(ghost_db))
    try:
        assert gconn.execute("SELECT cascade_id FROM trajectory_meta").fetchone()[0] == old_cid
        assert gconn.execute("SELECT step_payload FROM steps WHERE idx=0").fetchone()[0] == f"turn:{old_cid}:HOLD".encode("utf-8")
    finally:
        gconn.close()

    # If the daemon resurrects old_cid into conversation_summaries when opened in Web UI, the next mirror cleans the duplicate row again
    conn = sqlite3.connect(asf.DB)
    conn.execute(
        "INSERT OR REPLACE INTO conversation_summaries VALUES (?, ?, '', 'CASCADE_RUN_STATUS_IDLE', 'autonommptrade', 1, 0, ?)",
        (old_cid, asf.read_annotation_title(str(cli), old_cid), raw_old),
    )
    conn.commit()
    conn.close()
    assert mirror_topic_thread_to_ui(new_cid, topic_title, bot_cli_base=str(bot_base), ui_cli_base=str(cli)) is True

    # Meanwhile pending_ui_sync still queues both gone:<old_cid> and new:<new_cid>, and fresh ghost updates do not block idle check
    pending = asf.pending_ui_sync()
    assert f"gone:{old_cid}" in pending
    assert f"new:{new_cid}" in pending
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY

    # Even if graceful daemon shutdown flushes old_cid back into conversation_summaries right before mode_pre runs,
    # mode_pre purges the stale topic ghost row and promotes only new_cid.db into D2
    conn = sqlite3.connect(asf.DB)
    conn.execute(
        "INSERT OR REPLACE INTO conversation_summaries VALUES (?, ?, '', 'CASCADE_RUN_STATUS_IDLE', 'autonommptrade', 120, 0, ?)",
        (old_cid, topic_title, raw_old),
    )
    conn.commit()
    conn.close()
    assert asf.mode_pre(force=True) == 0
    assert not (d1 / f"{old_cid}.db").exists()
    assert not (d2 / f"{old_cid}.db").exists()
    assert (d2 / f"{new_cid}.db").exists()
    assert asf.daemon_known_cids() == {new_cid}


def test_active_background_command_and_subagent_block_restart_until_completed(store, tmp_path, monkeypatch):

    cli, d1, _ = store
    monkeypatch.setattr(asf, "LOG_FILE", str(tmp_path / "agy_store_fix.log"))
    monkeypatch.setattr(asf, "LOCK", str(tmp_path / "agy_store_fix.lock"))

    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 2400)

    log_dir = cli / "brain" / "backtest-cid" / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tp = log_dir / "transcript.jsonl"

    user_ts = time.time() - 2400.0  # USER_INPUT 40m ago (>30m stale user input must NOT unblock live bg command!)
    user_iso = datetime.fromtimestamp(user_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd_ts = time.time() - 400.0    # background regression launched ~6.6m ago (>120s idle_sec)
    cmd_iso = datetime.fromtimestamp(cmd_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Active background run_command blocks both ui-sync-check and active-check (even after sender=system notice)
    tp.write_text(
        json.dumps({"step_index": 1, "type": "USER_INPUT", "created_at": user_iso, "content": "run regression"})
        + "\n"
        + json.dumps(
            {
                "step_index": 2,
                "type": "GENERIC",
                "status": "RUNNING",
                "created_at": cmd_iso,
                "content": (
                    "Created At: 2026-11-11T11:31:41+02:00\n"
                    "Tool is running as a background task with task id: backtest-cid/task-2216\n"
                    "Task Description: ./offline/runners/trigger_dev_regression.sh --wait\n"
                ),
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 3,
                "type": "SYSTEM_MESSAGE",
                "created_at": cmd_iso,
                "content": "[Message] sender=system priority=MESSAGE_PRIORITY_LOW content=[Notice] background info",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(tp, (cmd_ts, cmd_ts))
    for root, _, files in os.walk(cli / "brain"):
        for f in files:
            os.utime(os.path.join(root, f), (cmd_ts, cmd_ts))

    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY
    assert asf.main(["agy_store_fix.py", "active-check", "--idle-sec", "120"]) == asf.UI_SYNC_BUSY

    # 2. Once task-2216 finishes (SYSTEM_MESSAGE from sender=backtest-cid/task-2216) and stays idle, restart unblocks
    tp.write_text(
        tp.read_text(encoding="utf-8")
        + json.dumps(
            {
                "step_index": 4,
                "type": "SYSTEM_MESSAGE",
                "created_at": cmd_iso,
                "content": "[Message] sender=backtest-cid/task-2216 priority=MESSAGE_PRIORITY_HIGH content=Regression PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(tp, (cmd_ts, cmd_ts))
    assert asf.main(["agy_store_fix.py", "active-check", "--idle-sec", "120"]) == asf.UI_SYNC_READY
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY
    (tmp_path / "ui_sync_state.json").unlink(missing_ok=True)

    # 3. Active invoke_subagent blocks until subagent replies or manage_subagents kills it
    sub_cid = "3afdb20b-94a8-40ae-bac1-3cc33714728e"
    tp.write_text(
        tp.read_text(encoding="utf-8")
        + json.dumps(
            {
                "step_index": 5,
                "type": "GENERIC",
                "status": "DONE",
                "created_at": cmd_iso,
                "content": f'Created the following subagent:\n{{\n  "conversationId":  "{sub_cid}"\n}}\n',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(tp, (cmd_ts, cmd_ts))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY
    assert asf.mode_active_check(120) == asf.UI_SYNC_BUSY

    tp.write_text(
        tp.read_text(encoding="utf-8")
        + json.dumps(
            {
                "step_index": 6,
                "type": "PLANNER_RESPONSE",
                "created_at": cmd_iso,
                "tool_calls": [{"name": "manage_subagents", "args": {"Action": "kill", "ConversationIds": [sub_cid]}}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "step_index": 7,
                "type": "GENERIC",
                "status": "DONE",
                "created_at": cmd_iso,
                "content": (
                    "Created At: 2026-11-11T13:59:12+02:00\n"
                    "Completed At: 2026-11-11T13:59:13+02:00\n"
                    "Showing lines 760-830 of 1085\n"
                    '768: "Tool is running as a background task with task id: other-cid/task-9999\\n"\n'
                    '822: \'Created the following subagent:\\n{"conversationId": "phantom-sub-1"}\'\n'
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(tp, (cmd_ts, cmd_ts))
    assert asf.mode_active_check(120) == asf.UI_SYNC_READY
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY


def test_running_status_with_old_db_but_fresh_transcript_blocks_restart(store):
    cli, d1, _ = store
    conn = sqlite3.connect(asf.DB)
    conn.execute("DROP TABLE conversation_summaries")
    conn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text primary key, status text)"
    )
    conn.execute("INSERT INTO conversation_summaries VALUES ('thinking-chat', 'CASCADE_RUN_STATUS_RUNNING')")
    conn.execute("INSERT INTO conversation_summaries VALUES ('pending-topic', 'CASCADE_RUN_STATUS_IDLE')")
    conn.commit()
    conn.close()

    (d1 / "thinking-chat.db").write_text("old-checkpoint")
    (d1 / "pending-topic.db").write_text("y")
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["thinking-chat"]))
    _age_store(d1, 1500)  # .db has not checkpointed for 25 minutes (> 900s)

    tp_dir = cli / "brain" / "thinking-chat" / ".system_generated" / "logs"
    tp_dir.mkdir(parents=True, exist_ok=True)
    tp = tp_dir / "transcript.jsonl"
    tp.write_text('{"step_index": 1, "type": "USER_INPUT", "content": "deep audit"}\n', encoding="utf-8")
    fresh_turn_ts = time.time() - 300.0  # turn started 5m ago (>120s idle_sec, <900s running bound)
    os.utime(tp, (fresh_turn_ts, fresh_turn_ts))

    assert asf.mode_active_check(120) == asf.UI_SYNC_BUSY
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY


def test_daemon_has_active_child_tasks_cgroup_and_proc_tree(tmp_path):
    proc_root = tmp_path / "proc"
    cg_root = tmp_path / "cgroup"
    cg_dir = cg_root / "user.slice" / "antigravity-cli-daemon.service"
    cg_dir.mkdir(parents=True)
    proc_root.mkdir(parents=True)

    clk_tck = int(os.sysconf("SC_CLK_TCK")) if hasattr(os, "SC_CLK_TCK") else 100
    btime = int(time.time()) - 10000
    (proc_root / "stat").write_text(f"cpu  1 2 3 4\nbtime {btime}\n", encoding="utf-8")

    def write_proc(pid: int, ppid: int, state: str, age_s: int, cmd: str) -> None:
        pdir = proc_root / str(pid)
        pdir.mkdir(parents=True, exist_ok=True)
        start_ticks = max(1, int((time.time() - age_s - btime) * clk_tck))
        rest = ["0"] * 25
        rest[0] = state
        rest[1] = str(ppid)
        rest[19] = str(start_ticks)
        (pdir / "stat").write_text(f"{pid} (proc name) {' '.join(rest)}\n", encoding="utf-8")
        (pdir / "cmdline").write_bytes(cmd.encode("utf-8").replace(b" ", b"\x00") + b"\x00")

    write_proc(500, 1, "S", 5000, "/home/predut/.local/bin/agy remote-control serve")
    (proc_root / "500" / "cgroup").write_text(
        "11:net_cls:/\n0::/user.slice/antigravity-cli-daemon.service\n",
        encoding="utf-8",
    )
    # Benign management helper + zombie + stale (>3600s) task must NOT block
    write_proc(501, 500, "S", 5, "/usr/bin/python3 orchestratorOS/admin/agy_store_fix.py active-check")
    write_proc(502, 500, "Z", 10, "bash -c defunct")
    write_proc(503, 500, "S", 4000, "sleep 99999")
    (cg_dir / "cgroup.procs").write_text("500\n501\n502\n503\n", encoding="utf-8")

    assert asf._daemon_has_active_child_tasks(daemon_pid=500, proc_root=str(proc_root), cgroup_root=str(cg_root)) is False

    # Live background regression child task (<3600s) blocks via cgroup v2 AND via /proc ppid fallback
    write_proc(504, 500, "S", 300, "bash ./offline/runners/run_dev_regression.sh trigger --wait")
    (cg_dir / "cgroup.procs").write_text("500\n504\n", encoding="utf-8")
    assert asf._daemon_has_active_child_tasks(daemon_pid=500, proc_root=str(proc_root), cgroup_root=str(cg_root)) is True
    (cg_dir / "cgroup.procs").unlink()
    assert asf._daemon_has_active_child_tasks(daemon_pid=500, proc_root=str(proc_root), cgroup_root=str(cg_root)) is True


def test_manage_agy_defers_stuck_tunnel_and_cron_update_restarts_when_chats_active(tmp_path):
    import stat
    import subprocess

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manage_agy = os.path.join(root_dir, "orchestratorOS", "admin", "manage_agy.sh")
    fake_home = tmp_path / "home"
    cli_dir = fake_home / ".gemini" / "antigravity-cli"
    (cli_dir / "conversations").mkdir(parents=True)
    (fake_home / ".gemini" / "antigravity" / "conversations").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    calls_log = tmp_path / "agy_calls.log"

    # Populate a fresh interactive conversation (<120s old) so active-check returns UI_SYNC_BUSY (11)
    active_db = cli_dir / "conversations" / "active-chat.db"
    active_db.write_text("live-turn", encoding="utf-8")
    sum_conn = sqlite3.connect(str(cli_dir / "conversation_summaries.db"))
    sum_conn.execute("CREATE TABLE conversation_summaries (conversation_id text primary key, status text)")
    sum_conn.execute("INSERT INTO conversation_summaries VALUES ('active-chat', 'CASCADE_RUN_STATUS_RUNNING')")
    sum_conn.commit()
    sum_conn.close()

    # 3 consecutive Failed before open -> is_tunnel_stuck() == 0
    (cli_dir / "cli.log").write_text(
        "W1111 12:15:00.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n"
        "W1111 12:16:00.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n"
        "W1111 12:17:00.000000 1 remote_control_v2.go:1] Connection status: Failed before open\n",
        encoding="utf-8",
    )

    def make_stub(name: str, body: str) -> None:
        p = fake_bin / name
        p.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    stopped_flag = tmp_path / "daemon_stopped"
    fake_cgroup = tmp_path / "self.cgroup"
    fake_cgroup.write_text("0::/user.slice/cron.service\n", encoding="utf-8")

    make_stub(
        "agy",
        f'echo "$*" >> "{calls_log}"\n'
        'case "${1:-} ${2:-}" in\n'
        '  "--version ") echo "1.4.0" ;;\n'
        f'  "remote-control stop") touch "{stopped_flag}" ;;\n'
        f'  "remote-control start") rm -f "{stopped_flag}" ;;\n'
        '  "remote-control status") echo "Daemon status: active" ;;\n'
        '  "update ") echo "Update successful" ;;\n'
        '  *) exit 0 ;;\n'
        'esac',
    )
    make_stub(
        "systemctl",
        f'if [ "${{2:-}}" = "is-active" ]; then [ ! -f "{stopped_flag}" ]; exit $?; fi\n'
        'if [ "${2:-}" = "show" ] && [ "${4:-}" = "ActiveState" ]; then echo "active"; exit 0; fi\n'
        'exit 1',
    )
    make_stub("curl", "exit 0")
    make_stub("systemd-run", "exit 1")

    env = dict(os.environ)
    env.update(
        {
            "HOME": str(fake_home),
            "AGY_FIX_HOME": str(fake_home),
            "AGY_SELF_CGROUP_FILE": str(fake_cgroup),
            "AGY_SUPPRESS_ALERTS": "1",
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(tmp_path),
            "PYTHON_BIN": os.path.join(root_dir, ".venv", "bin", "python3"),
        }
    )

    # 1. Stuck tunnel while chat is active -> restart MUST be deferred (zero `remote-control stop` calls!)
    r1 = subprocess.run(["/bin/bash", manage_agy, "watchdog"], cwd=root_dir, env=env, capture_output=True, text=True, timeout=15)
    assert r1.returncode == 0
    assert "deferring restart until idle" in r1.stdout
    assert "remote-control stop" not in (calls_log.read_text(encoding="utf-8") if calls_log.exists() else "")

    # 2. Cron update while chat is active -> queues pending restart file without stopping daemon
    r2 = subprocess.run(["/bin/bash", manage_agy, "cron-update"], cwd=root_dir, env=env, capture_output=True, text=True, timeout=15)
    assert r2.returncode == 0
    assert (tmp_path / "agy_update_restart_pending").exists()
    assert "remote-control stop" not in calls_log.read_text(encoding="utf-8")

    # 3. Once active chat is aged past 15m (>900s running bound & >120s idle), watchdog cycles the stuck daemon cleanly
    old_ts = time.time() - 1500.0
    os.utime(active_db, (old_ts, old_ts))
    r3 = subprocess.run(["/bin/bash", manage_agy, "watchdog"], cwd=root_dir, env=env, capture_output=True, text=True, timeout=20)
    assert r3.returncode == 0
    assert "remote-control stop" in calls_log.read_text(encoding="utf-8")
    assert not (tmp_path / "agy_update_restart_pending").exists()


def test_webchannel_reconnect_and_mid_turn_background_polls_do_not_mask_active_inference(store):
    cli, d1, _ = store
    (d1 / "pending-topic.db").write_text("y")
    _set_rows(["pending-topic"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids([]))
    _age_store(d1, 1500)

    log_dir = cli / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    daemon_log = log_dir / "cli-20261111_121811.log"
    now = time.time()

    # 1. Hourly WebChannel rotation + Read loop noise 4s after streamGenerateContent must NOT mask active inference
    daemon_log.write_text(
        "I1111 12:18:15.064759 172 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1111 13:18:14.575390 31543 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse\n"
        "W1111 13:18:18.763113 31664 remote_control_v2.go:2258] WebChannel: XMLHTTP Unknown SID (74885)\n"
        "2026/11/11 13:18:18 Read loop (req 2128): err=context canceled\n"
        "E1111 13:18:18.780000 98 client.go:98] Post \"https://play.googleapis.com/log\": net/http: request canceled\n"
        "I1111 13:18:18.850833 172 remote_control_v2.go:2056] Connection status: Connected\n",
        encoding="utf-8",
    )
    os.utime(daemon_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY
    assert asf.mode_active_check(120) == asf.UI_SYNC_BUSY

    # 2. Periodic 6m fetchAvailableModels + loadCodeAssist poll 5s into a deep-thinking streamGenerateContent keeps session busy
    daemon_log.write_text(
        "I1111 12:18:15.064759 172 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1111 12:23:10.000000 500 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse\n"
        "I1111 12:23:15.107213 695 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels\n"
        "I1111 12:23:15.160892 696 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist\n",
        encoding="utf-8",
    )
    os.utime(daemon_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # 3. Sonar prompt-prep lines and safety-monitor timeout lines also register as active turn markers
    daemon_log.write_text(
        "I1111 12:18:15.064759 172 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1111 13:02:50.909737 5698 latency_breakdown.go:168] SEND_USER_CASCADE_MESSAGE_LATENCY\n"
        "I1111 13:02:50.934735 5716 monitor_config_manager_external.go:53] externalMonitorConfigManager: initialized\n"
        "I1111 13:02:50.935221 5716 monitoring.go:409] [Sonar] Resolved active tags\n"
        "I1111 13:02:50.935297 5716 monitoring.go:409] [Sonar] Resolved active tags\n",
        encoding="utf-8",
    )
    os.utime(daemon_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_BUSY

    # 4. Once another 6m background poll arrives >= 120s after the last active marker, session is idle
    daemon_log.write_text(
        daemon_log.read_text(encoding="utf-8")
        + "I1111 13:08:50.935297 6000 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels\n"
        + "I1111 13:08:51.100000 6001 http_helpers.go:654] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist\n",
        encoding="utf-8",
    )
    os.utime(daemon_log, (now, now))
    assert asf.mode_ui_sync_check(120) == asf.UI_SYNC_READY











