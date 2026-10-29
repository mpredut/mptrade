"""Unit tests for agy_store_fix ui-sync-check (idle-gated daemon restart decision)."""
import os
import sqlite3
import time

import pytest

from orchestratorOS.admin import agy_store_fix as asf


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
    import json

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
    import json

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
    # 1. Known permanent topic thread updated in D1 and re-hardlinked in D2 -> 0 restarts
    (d1 / "topic-10m.db").write_text("new-turn-content")
    os.link(d1 / "topic-10m.db", d2 / "topic-10m.db")
    # 2. Brand-new interactive Web UI thread created directly by the running daemon in D1 (not yet in D2) -> 0 restarts
    (d1 / "webui-chat.db").write_text("user-chat")
    _set_rows(["topic-10m", "webui-chat"])
    (cli / "agyhub_summaries_proto.pb").write_bytes(_pb_with_cids(["topic-10m", "webui-chat"]))
    _age_store(d1, 1200)

    assert asf.pending_ui_sync() == []
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_NOTHING

    # 3. Background heartbeat lines (fetchAvailableModels / loadCodeAssist) do NOT block ui-sync when a new topic thread appears
    (d1 / "new-topic.db").write_text("first-turn")
    _set_rows(["topic-10m", "webui-chat", "new-topic"])
    with open(asf.TOPIC_REGISTRY, "w", encoding="utf-8") as f:
        f.write('{"auto_cryptotrade_executor": {"conversation_id": "new-topic"}}')
    log_dir = cli / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cli-20261026_100000.log"
    log_file.write_text(
        "I1026 10:00:01.000000 1 remote_control_v2.go:2056] Connection status: Connected\n"
        "I1026 10:05:00.000000 2 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist\n"
        "I1026 10:05:01.000000 3 http_helpers.go:349] URL: https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels\n"
    )
    assert len(asf.pending_ui_sync()) == 2
    assert asf.mode_ui_sync_check(900) == asf.UI_SYNC_READY




