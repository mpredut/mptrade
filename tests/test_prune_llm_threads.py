"""Unit tests for Antigravity CLI automated thread pruner."""
from datetime import datetime, timezone, timedelta
import os
import sqlite3

import pytest

from orchestratorOS.admin.prune_llm_threads import is_automated_thread, prune_old_cli_threads

AUTONOMOUS_PROJECT_UUID = "f5f9d01f-01e5-4fac-80f2-97364688afab"


@pytest.fixture(autouse=True)
def _isolated_agy_stores(tmp_path, monkeypatch):
    """Never let topic mirroring touch the real Web UI store or topic registry."""
    ui_base = tmp_path / "ui_home" / ".gemini" / "antigravity-cli"
    (ui_base / "conversations").mkdir(parents=True)
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(ui_base))
    monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", str(tmp_path / "llm_topic_threads.json"))
    projects = tmp_path / "ui_home" / ".gemini" / "config" / "projects"
    projects.mkdir(parents=True)
    (projects / "autonom.json").write_text(
        f'{{"id": "{AUTONOMOUS_PROJECT_UUID}", "name": "autonommptrade"}}', encoding="utf-8")
    (projects / "mptrade.json").write_text('{"id": "3ed4aa5d-mptrade", "name": "mptrade"}', encoding="utf-8")



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
            ("bot-synth-1", "Crypto Macro Risk Synthesis", "Crypto Macro Risk Synthesis", ts_old, AUTONOMOUS_PROJECT_UUID, 2),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("bot-shield-2", "Macroeconomic Shock Shield Evaluation", "Macroeconomic Shock Shield Evaluation", ts_old, AUTONOMOUS_PROJECT_UUID, 2),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            ("user-interactive-3", "Custom Architecture Discussion", "Let us discuss design", ts_old, AUTONOMOUS_PROJECT_UUID, 18),
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

    def test_parse_db_timestamp_handles_trailing_z_suffix(self):
        from orchestratorOS.agy.summary_store import parse_db_timestamp
        dt1 = parse_db_timestamp("2026-10-25T22:08:31Z")
        dt2 = parse_db_timestamp("2026-10-25T22:08:31.123Z")
        dt3 = parse_db_timestamp("2026-10-25T22:08:31.123456789Z")
        assert dt1 is not None and dt1.year == 2026
        assert dt2 is not None and dt2.microsecond == 123000
        assert dt3 is not None and dt3.microsecond == 123456

