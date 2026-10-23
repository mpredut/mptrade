"""Unit tests for Antigravity CLI automated thread pruner."""
from datetime import datetime, timezone, timedelta
import os
import sqlite3

import pytest

from orchestratorOS.admin.prune_llm_threads import (
    ensure_proto_synced,
    is_automated_thread,
    prune_old_cli_threads,
    sync_thread_title,
)


@pytest.fixture(autouse=True)
def _isolated_agy_stores(tmp_path, monkeypatch):
    """Never let topic mirroring touch the real Web UI store or topic registry."""
    ui_base = tmp_path / "ui_home" / ".gemini" / "antigravity-cli"
    (ui_base / "conversations").mkdir(parents=True)
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(ui_base))
    monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", str(tmp_path / "llm_topic_threads.json"))



class TestPruneCliThreads:
    """Tests for prune_old_cli_threads."""

    def test_is_automated_thread_classification(self):
        # Automated patterns
        assert is_automated_thread(
            "Macroeconomic Risk",
            "You are a senior macroeconomic and geopolitical risk officer for an algorithmic crypto fund"
        ) is True
        assert is_automated_thread(
            "BTC Trading Bot Risk",
            "You are a principal quantitative crypto risk officer."
        ) is True
        assert is_automated_thread(
            "Analiză Risc Tranzacție",
            "evaluated by gemini"
        ) is True

        # Non-automated manual user topics
        assert is_automated_thread("backtesting", "Let's review our strategies") is False
        assert is_automated_thread("topic greu de revizuit", "discutie despre portofoliu") is False
        assert is_automated_thread("Simple Echo Test", "echo 123") is False

    def test_prune_filters_by_age_and_project(self, tmp_path):
        cli_base = str(tmp_path / "cli")
        os.makedirs(os.path.join(cli_base, "conversations"), exist_ok=True)
        os.makedirs(os.path.join(cli_base, "brain"), exist_ok=True)

        db_path = os.path.join(cli_base, "conversation_summaries.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE conversation_summaries (
                conversation_id text PRIMARY KEY,
                title text,
                preview text,
                last_modified_time datetime,
                project_id text
            )
        """)

        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=5)).isoformat()
        ts_recent = (now - timedelta(hours=6)).isoformat()

        # 1. Old automated thread in default project -> SHOULD be pruned
        cid_old_auto = "old-auto-1111"
        # 2. Recent automated thread in default project -> SHOULD be kept (within 2 days)
        cid_recent_auto = "recent-auto-2222"
        # 3. Old manual user thread in default project -> SHOULD be kept (not automated)
        cid_old_manual = "old-manual-3333"
        # 4. Old automated thread in another project -> SHOULD be kept (different project)
        cid_other_project = "other-proj-4444"

        rows = [
            (cid_old_auto, "Macro Risk", "You are a senior macroeconomic and geopolitical risk officer", ts_old, "default-cli-project"),
            (cid_recent_auto, "Macro Risk", "You are a senior macroeconomic and geopolitical risk officer", ts_recent, "default-cli-project"),
            (cid_old_manual, "My manual prompt", "User discussing python code", ts_old, "default-cli-project"),
            (cid_other_project, "Macro Risk", "You are a senior macroeconomic and geopolitical risk officer", ts_old, "other-project"),
        ]

        for r in rows:
            conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", r)
            # Create dummy .db file and brain dir
            with open(os.path.join(cli_base, "conversations", f"{r[0]}.db"), "w") as f:
                f.write("dummy db")
            os.makedirs(os.path.join(cli_base, "brain", r[0]), exist_ok=True)
        conn.commit()
        conn.close()

        # Run pruner with default retention (1.0 day)
        pruned_count, bytes_freed = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            project_id="default-cli-project",
            dry_run=False,
        )

        assert pruned_count == 1
        assert bytes_freed > 0

        # Verify DB contents
        conn = sqlite3.connect(db_path)
        remaining = [r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()]
        conn.close()

        assert cid_old_auto not in remaining
        assert cid_recent_auto in remaining
        assert cid_old_manual in remaining
        assert cid_other_project in remaining

        # Verify files on disk
        assert not os.path.exists(os.path.join(cli_base, "conversations", f"{cid_old_auto}.db"))
        assert not os.path.exists(os.path.join(cli_base, "brain", cid_old_auto))
        assert os.path.exists(os.path.join(cli_base, "conversations", f"{cid_recent_auto}.db"))

    def test_exclude_cids_protection(self, tmp_path):
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
                project_id text
            )
        """)
        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=10)).isoformat()
        cid = "protected-active-cid"
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            cid, "Macro Risk", "You are a senior macroeconomic and geopolitical risk officer", ts_old, "default-cli-project"
        ))
        conn.commit()
        conn.close()

        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            exclude_cids={cid},
            dry_run=False,
        )
        assert pruned == 0

    def test_autonommptrade_project_pruning(self, tmp_path):
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
                project_id text
            )
        """)
        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=2)).isoformat()
        ts_recent = (now - timedelta(hours=3)).isoformat()

        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            "auto-old-1", "Query", "Automated reasoning", ts_old, "autonommptrade"
        ))
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            "auto-recent-2", "Query", "Automated reasoning", ts_recent, "autonommptrade"
        ))
        conn.commit()
        conn.close()

        # Both default projects (default-cli-project & autonommptrade) checked
        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            dry_run=False,
        )
        assert pruned == 1

        conn = sqlite3.connect(db_path)
        remaining = [r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()]
        conn.close()
        assert "auto-old-1" not in remaining
        assert "auto-recent-2" in remaining

    def test_user_manual_projects_and_threads_protected(self, tmp_path):
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
                project_id text
            )
        """)
        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=30)).isoformat()

        # Insert user threads in mptrade and default-cli-project
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            "user-mptrade-1", "backtesting", "my custom manual discussion", ts_old, "mptrade"
        ))
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            "user-default-2", "my ideas", "user discussion on prompt", ts_old, "default-cli-project"
        ))
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)", (
            "bot-auto-3", "Macro Risk", "You are a senior macroeconomic and geopolitical risk officer", ts_old, "autonommptrade"
        ))
        conn.commit()
        conn.close()

        # Default run: should ONLY prune bot-auto-3 in autonommptrade
        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            dry_run=False,
        )
        assert pruned == 1

        conn = sqlite3.connect(db_path)
        remaining = {r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()}
        conn.close()

        assert "user-mptrade-1" in remaining, "User mptrade thread must NEVER be pruned!"
        assert "user-default-2" in remaining, "User default-cli-project thread must NOT be pruned on default cycle!"
        assert "bot-auto-3" not in remaining

        # Attempt to prune protected project explicitly: must be refused
        pruned_protected, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=0.0,
            project_id="mptrade",
            dry_run=False,
        )
        assert pruned_protected == 0

    def test_max_threads_cap_and_wal_cleanup(self, tmp_path):
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
                project_id text
            )
        """)
        now = datetime.now(timezone.utc)
        for i in range(5):
            cid = f"auto-recent-{i}"
            ts = (now - timedelta(minutes=10 - i)).isoformat()
            conn.execute(
                "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?)",
                (cid, "ntfy-guard", "Automated reasoning", ts, "autonommptrade"),
            )
            for ext in (".db", ".db-shm", ".db-wal"):
                with open(os.path.join(cli_base, "conversations", f"{cid}{ext}"), "w") as f:
                    f.write("data")
        conn.commit()
        conn.close()

        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            max_threads=2,
            dry_run=False,
        )
        assert pruned == 3

        conn = sqlite3.connect(db_path)
        remaining = {r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()}
        conn.close()
        assert remaining == {"auto-recent-3", "auto-recent-4"}
        assert not os.path.exists(os.path.join(cli_base, "conversations", "auto-recent-0.db-wal"))
        assert not os.path.exists(os.path.join(cli_base, "conversations", "auto-recent-0.db-shm"))

    def test_multi_turn_step_count_protected_even_with_all(self, tmp_path):
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
                step_count integer
            )
        """)
        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=5)).isoformat()
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("multi-turn-user-1", "Implementing feature X", "Let us refactor the code", ts_old, "autonommptrade", 42),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("one-shot-auto-1", "Quick check", "Single turn prompt", ts_old, "autonommptrade", 4),
        )
        conn.commit()
        conn.close()

        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            all_threads_in_project=True,
            dry_run=False,
        )
        assert pruned == 1
        conn = sqlite3.connect(db_path)
        remaining = {r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()}
        conn.close()
        assert remaining == {"multi-turn-user-1"}

    def test_sync_thread_title_updates_annotations_and_proto(self, tmp_path):
        cli_base = str(tmp_path / "cli")
        os.makedirs(os.path.join(cli_base, "conversations"), exist_ok=True)
        os.makedirs(os.path.join(cli_base, "annotations"), exist_ok=True)
        cid = "11111111-2222-3333-4444-555555555555"
        with open(os.path.join(cli_base, "conversations", f"{cid}.db"), "wb") as f:
            f.write(b"sqlite")

        # Build a minimal TrajectorySummary with f1="My Preview" and f2=4
        raw_summary = b"\x0a\x0aMy Preview\x10\x04"
        db_path = os.path.join(cli_base, "conversation_summaries.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE conversation_summaries (
                conversation_id text PRIMARY KEY,
                title text,
                preview text,
                raw_summary blob
            )
        """)
        conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)", (cid, "", "My Preview", raw_summary))
        conn.commit()
        conn.close()

        cid_b = cid.encode("utf-8")
        map_entry = b"\x0a" + bytes([len(cid_b)]) + cid_b + b"\x12" + bytes([len(raw_summary)]) + raw_summary
        proto_record = b"\x0a" + bytes([len(map_entry)]) + map_entry
        for pb_name in ("jetbox_summaries_proto.pb", "agyhub_summaries_proto.pb"):
            with open(os.path.join(cli_base, pb_name), "wb") as f:
                f.write(proto_record)

        assert ensure_proto_synced(cli_base) == 0
        assert sync_thread_title(cid, "Custom Thread Title", cli_base) is True

        pbtxt_path = os.path.join(cli_base, "annotations", f"{cid}.pbtxt")
        assert os.path.exists(pbtxt_path)
        with open(pbtxt_path, "r", encoding="utf-8") as f:
            assert 'title:"Custom Thread Title"' in f.read()

        conn = sqlite3.connect(db_path)
        title_col, raw_col = conn.execute(
            "SELECT title, raw_summary FROM conversation_summaries WHERE conversation_id=?",
            (cid,),
        ).fetchone()
        conn.close()
        assert title_col == "Custom Thread Title"
        assert b"My Preview" in raw_col
        assert b"Custom Thread Title" in raw_col

    def test_agy_generated_titles_in_dedicated_project_are_pruned(self, tmp_path):
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
                step_count integer
            )
        """)
        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=3)).isoformat()
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("bot-synth-1", "Crypto Macro Risk Synthesis", "Crypto Macro Risk Synthesis", ts_old, "f5f9d01f-01e5-4fac-80f2-97364688afab", 2),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("bot-shield-2", "Macroeconomic Shock Shield Evaluation", "Macroeconomic Shock Shield Evaluation", ts_old, "f5f9d01f-01e5-4fac-80f2-97364688afab", 2),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("user-interactive-3", "Custom Architecture Discussion", "Let us discuss design", ts_old, "f5f9d01f-01e5-4fac-80f2-97364688afab", 18),
        )
        conn.commit()
        conn.close()

        pruned, _ = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=1.0,
            dry_run=False,
        )
        assert pruned == 2
        conn = sqlite3.connect(db_path)
        remaining = {r[0] for r in conn.execute("SELECT conversation_id FROM conversation_summaries").fetchall()}
        conn.close()
        assert remaining == {"user-interactive-3"}

    def test_resolve_bot_cli_base_and_agy_bot_bin_configurable(self, tmp_path, monkeypatch):
        from orchestratorOS.admin.prune_llm_threads import (
            expand_config_path,
            resolve_agy_bot_bin,
            resolve_bot_cli_base,
            resolve_project_uuid,
        )

        bot_home = tmp_path / "custom_bot_home"
        bot_cli = bot_home / ".gemini" / "antigravity-cli"
        bot_cli.mkdir(parents=True)
        proj_dir = bot_home / ".gemini" / "config" / "projects"
        proj_dir.mkdir(parents=True)
        (proj_dir / "custom.json").write_text(
            '{"id": "12345678-1234-1234-1234-123456789abc", "name": "autonommptrade"}',
            encoding="utf-8",
        )

        custom_bin = tmp_path / "agy-bot-custom"
        custom_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        custom_bin.chmod(0o755)

        monkeypatch.setenv("AGY_BOT_HOME", str(bot_home))
        monkeypatch.setenv("AGY_BOT_BIN", str(custom_bin))
        monkeypatch.delenv("MPTRADE_AGY_BIN", raising=False)
        monkeypatch.delenv("AGY_CLI_BASE", raising=False)

        assert resolve_bot_cli_base() == str(bot_cli)
        assert resolve_agy_bot_bin() == str(custom_bin)
        assert resolve_project_uuid("autonommptrade", cli_base=str(bot_cli)) == "12345678-1234-1234-1234-123456789abc"
        assert expand_config_path("$HOME/test_dir").endswith("/test_dir")
        assert "$HOME" not in expand_config_path("$HOME/test_dir")

    def test_topic_thread_is_permanent_and_compacts_in_place(self, tmp_path, monkeypatch):
        from orchestratorOS.admin.prune_llm_threads import (
            TOPIC_THREAD_TITLES,
            get_active_topic_cids,
            load_topic_registry,
            prepare_topic_session,
            record_topic_turn,
        )

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
        assert reg["title"] == TOPIC_THREAD_TITLES["macro_sentiment"]

        # 6. Thread vanished from the store -> recreate once, replaying the rolling memory
        os.remove(os.path.join(cli_base, "conversations", f"{cid1}.db"))
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM conversation_summaries")
        conn.commit()
        conn.close()
        session = prepare_topic_session("macro_sentiment", cli_base=cli_base)
        assert session.conversation_id is None and "TOPIC MEMORY" in session.history_prefix

    def test_unknown_topic_fails_fast(self):
        import pytest as _pytest
        from orchestratorOS.admin.prune_llm_threads import prepare_topic_session

        with _pytest.raises(KeyError):
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
        from orchestratorOS.admin.agy_store_fix import visible_conversations_dir
        from orchestratorOS.admin.prune_llm_threads import (
            _strip_proto_tag_20,
            encode_varint,
            mirror_topic_thread_to_ui,
            resolve_ui_cli_base,
        )

        cid = "cid-mirror-1"
        bot_base = tmp_path / "bot" / ".gemini" / "antigravity-cli"
        (bot_base / "conversations").mkdir(parents=True)
        _make_conv_db(str(bot_base / "conversations" / f"{cid}.db"))
        title_b = b"t"
        raw = b"\x0a" + encode_varint(len(title_b)) + title_b + b"\xa0\x01\x11"  # field 1 + field 20 = 17
        _make_summary_db(str(bot_base / "conversation_summaries.db"),
                         [(cid, "t", "CORTEX_TRAJECTORY_SOURCE_CLI", "proj", 2, raw)])

        ui_base = resolve_ui_cli_base()
        _make_summary_db(os.path.join(ui_base, "conversation_summaries.db"), [])
        visible = visible_conversations_dir(ui_base)
        os.makedirs(visible)
        stale = os.path.join(visible, f"{cid}.db")
        with open(stale, "w") as f:
            f.write("old")

        assert mirror_topic_thread_to_ui(cid, bot_cli_base=str(bot_base)) is True

        dst = os.path.join(ui_base, "conversations", f"{cid}.db")
        assert sqlite3.connect(dst).execute("SELECT source FROM trajectory_meta").fetchone()[0] == 1
        assert not os.path.exists(stale)
        row = sqlite3.connect(os.path.join(ui_base, "conversation_summaries.db")).execute(
            "SELECT source, raw_summary FROM conversation_summaries WHERE conversation_id=?", (cid,)
        ).fetchone()
        assert row[0] == ""
        assert row[1] == _strip_proto_tag_20(raw) == b"\x0a\x01t"
