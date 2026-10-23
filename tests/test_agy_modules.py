"""Unit tests for the orchestratorOS.agy modules (paths, protobuf, topics, UI mirror)."""
from datetime import datetime, timezone
import os
import sqlite3

import pytest

from orchestratorOS.admin.agy_store_fix import visible_conversations_dir
from orchestratorOS.agy import paths
from orchestratorOS.agy.protobuf import drop_entries, enc_fields, entry_cid, parse_fields, summary_ui_visible
from orchestratorOS.agy.topics import (
    TOPIC_SPECS,
    get_active_topic_cids,
    load_topic_registry,
    prepare_topic_session,
    record_topic_turn,
    topic_spec,
)
from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui


@pytest.fixture(autouse=True)
def _isolated_agy_stores(tmp_path, monkeypatch):
    """Never let topic mirroring touch the real Web UI store or topic registry."""
    ui_base = tmp_path / "ui_home" / ".gemini" / "antigravity-cli"
    (ui_base / "conversations").mkdir(parents=True)
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(ui_base))
    monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", str(tmp_path / "llm_topic_threads.json"))


class TestPaths:
    def test_resolvers_are_strict_and_never_fall_back_to_the_ui_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGY_BOT_HOME", "~/bot_home_x")
        monkeypatch.setenv("AGY_BOT_BIN", "~/.local/bin/agy-bot")
        home = os.path.expanduser("~")
        assert paths.resolve_bot_cli_base() == os.path.join(home, "bot_home_x", ".gemini", "antigravity-cli")
        assert paths.resolve_agy_bot_bin() == os.path.join(home, ".local", "bin", "agy-bot")
        assert paths.resolve_bot_cli_base(str(tmp_path / "x")) == str(tmp_path / "x")
        monkeypatch.delenv("AGY_BOT_HOME")
        with pytest.raises(Exception):
            paths.resolve_bot_cli_base()


class TestProtobuf:
    def test_summary_ui_visible_drops_cli_source_and_sets_title_keeping_other_annotations(self):
        raw = enc_fields([(1, 2, b"preview"), (15, 2, enc_fields([(1, 2, b"old"), (3, 0, 7)])), (20, 0, 17)])
        out = dict((t, v) for t, _, v in parse_fields(summary_ui_visible(raw, "New")))
        assert 20 not in out and out[1] == b"preview"
        assert parse_fields(out[15]) == [(1, 2, b"New"), (3, 0, 7)]

    def test_drop_entries_removes_summary_and_mtime_entries_of_given_cids(self):
        buf = b"".join(enc_fields([(tag, 2, enc_fields([(1, 2, cid.encode()), (2, 0, 1)]))])
                       for tag, cid in ((1, "a"), (1, "b"), (2, "a")))
        out, removed = drop_entries(buf, {"a"})
        assert removed == 2 and [entry_cid(v) for _, _, v in parse_fields(out)] == ["b"]


class TestTopics:
    def test_only_auto_trade_executor_trades_and_only_auto_technical_repair_self_heals(self):
        assert [k for k, v in TOPIC_SPECS.items() if v.can_trade] == ["auto_trade_executor"]
        assert [k for k, v in TOPIC_SPECS.items() if v.can_self_heal] == ["auto_technical_repair"]
        assert "[auto_trade_executor]" in topic_spec("auto_trade_executor").title
        assert "[auto_technical_repair]" in topic_spec("auto_technical_repair").title

    def test_topic_thread_is_permanent_and_compacts_in_place(self, tmp_path, monkeypatch):

        cli_base = str(tmp_path / "cli")
        os.makedirs(os.path.join(cli_base, "conversations"), exist_ok=True)
        db_path = os.path.join(cli_base, "conversation_summaries.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE conversation_summaries (
                conversation_id text PRIMARY KEY,
                title text,
                preview text,
                last_modified_time datetime,
                project_id text,
                step_count integer,
                status text
            )
        """)
        conn.commit()
        conn.close()
        monkeypatch.setenv("LLM_TOPIC_COMPACT_STEPS", "6")
        monkeypatch.setenv("LLM_TOPIC_HISTORY_MAX_ENTRIES", "3")

        def set_steps(cid, steps):
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT OR REPLACE INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cid, "x", "p", datetime.now(timezone.utc).isoformat(), "autonommptrade", steps,
                 "CASCADE_RUN_STATUS_IDLE"),
            )
            conn.commit()
            conn.close()

        # 1. No thread yet -> the first turn creates it
        assert tuple(prepare_topic_session("macro_sentiment", cli_base=cli_base)) == (None, "", False)
        cid1 = "cid-topic-111"
        with open(os.path.join(cli_base, "conversations", f"{cid1}.db"), "w") as f:
            f.write("db1")
        set_steps(cid1, 2)
        record_topic_turn("macro_sentiment", cid1, '{"market_bias": "BULLISH", "summary": "inflow"}', cli_base=cli_base)
        assert cid1 in get_active_topic_cids()

        # 2. Below the threshold -> resume, no compaction
        assert tuple(prepare_topic_session("macro_sentiment", cli_base=cli_base)) == (cid1, "", False)

        # 3. Threshold reached -> SAME thread, compact in place (never rotate or delete)
        set_steps(cid1, 6)
        session = prepare_topic_session("macro_sentiment", cli_base=cli_base)
        assert session == (cid1, "", True)
        set_steps(cid1, 10)
        record_topic_turn("macro_sentiment", cid1, '{"summary": "after compact"}', compacted=True, cli_base=cli_base)
        assert os.path.exists(os.path.join(cli_base, "conversations", f"{cid1}.db"))

        # 4. Counting restarts from the compaction point
        set_steps(cid1, 14)
        assert prepare_topic_session("macro_sentiment", cli_base=cli_base).compact is False
        set_steps(cid1, 16)
        assert prepare_topic_session("macro_sentiment", cli_base=cli_base).compact is True

        # 5. A different conversation id (e.g. a detached one-off answer) never replaces the permanent one
        record_topic_turn("macro_sentiment", "cid-other", '{"summary": "x"}', cli_base=cli_base)
        reg = load_topic_registry()["macro_sentiment"]
        assert reg["conversation_id"] == cid1
        assert reg["compacted_at_step"] == 10 and reg["compaction_count"] == 1
        assert reg["title"] == TOPIC_SPECS["macro_sentiment"].title

        # 6. Thread vanished from the store -> recreate once, replaying the rolling memory
        os.remove(os.path.join(cli_base, "conversations", f"{cid1}.db"))
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM conversation_summaries")
        conn.commit()
        conn.close()
        session = prepare_topic_session("macro_sentiment", cli_base=cli_base)
        assert session.conversation_id is None and "TOPIC MEMORY" in session.history_prefix

    def test_unknown_topic_fails_fast(self):
        with pytest.raises(KeyError):
            prepare_topic_session("not_a_topic", compact_steps=6)



def _make_summary_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE conversation_summaries (conversation_id text primary key, title text, source text, "
        "project_id text, step_count integer, raw_summary blob)"
    )
    for row in rows:
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()


def _make_conv_db(path, source=17):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trajectory_meta (trajectory_id text, cascade_id text, kind integer, source integer)")
    conn.execute("INSERT INTO trajectory_meta VALUES ('t', 'c', 4, ?)", (source,))
    conn.commit()
    conn.close()


class TestMirrorTopicThreadToUi:
    def test_mirror_marks_ui_visible_and_drops_stale_visible_copy(self, tmp_path):
        cid = "cid-mirror-1"
        bot_base = tmp_path / "bot" / ".gemini" / "antigravity-cli"
        (bot_base / "conversations").mkdir(parents=True)
        _make_conv_db(str(bot_base / "conversations" / f"{cid}.db"))
        raw = enc_fields([(1, 2, b"t"), (20, 0, 17)])
        _make_summary_db(str(bot_base / "conversation_summaries.db"),
                         [(cid, "t", "CORTEX_TRAJECTORY_SOURCE_CLI", "proj", 2, raw)])

        ui_base = paths.resolve_ui_cli_base()
        _make_summary_db(os.path.join(ui_base, "conversation_summaries.db"), [])
        visible = visible_conversations_dir(ui_base)
        os.makedirs(visible)
        stale = os.path.join(visible, f"{cid}.db")
        with open(stale, "w") as f:
            f.write("old")

        assert mirror_topic_thread_to_ui(cid, "Permanent Name", bot_cli_base=str(bot_base)) is True

        dst = os.path.join(ui_base, "conversations", f"{cid}.db")
        assert sqlite3.connect(dst).execute("SELECT source FROM trajectory_meta").fetchone()[0] == 1
        assert not os.path.exists(stale)
        title, source, stored = sqlite3.connect(os.path.join(ui_base, "conversation_summaries.db")).execute(
            "SELECT title, source, raw_summary FROM conversation_summaries WHERE conversation_id=?", (cid,)
        ).fetchone()
        assert (title, source) == ("Permanent Name", "")
        assert stored == summary_ui_visible(raw, "Permanent Name")
        with open(os.path.join(ui_base, "annotations", f"{cid}.pbtxt")) as f:
            assert 'title:"Permanent Name"' in f.read()
