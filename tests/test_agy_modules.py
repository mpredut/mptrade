"""Unit tests for the orchestratorOS.agy modules (paths, protobuf, topics, UI mirror)."""
from datetime import datetime, timezone
import os
import sqlite3

import pytest

from orchestratorOS.agy import paths
from orchestratorOS.agy.paths import visible_conversations_dir
from orchestratorOS.agy.protobuf import (
    drop_entries,
    enc_fields,
    entry_cid,
    list_entry_cids,
    parse_fields,
    summary_annotation_title,
    summary_ui_visible,
)
from orchestratorOS.agy.topics import (
    TOPIC_SPECS,
    condense_decision,
    extract_json_object,
    format_topic_turn_header,
    get_active_topic_cids,
    load_topic_registry,
    prepare_topic_session,
    record_topic_turn,
    topic_spec,
)
from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui, sync_topic_thread_from_ui


@pytest.fixture(autouse=True)
def _isolated_agy_stores(tmp_path, monkeypatch):
    """Never let topic mirroring touch the real Web UI store or topic registry."""
    ui_base = tmp_path / "ui_home" / ".gemini" / "antigravity-cli"
    (ui_base / "conversations").mkdir(parents=True, exist_ok=True)
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
        raw = enc_fields([(1, 2, b"preview"), (15, 2, enc_fields([(1, 2, b"old"), (3, 0, 7)])), (20, 0, 17), (28, 0, 2)])
        out = dict((t, v) for t, _, v in parse_fields(summary_ui_visible(raw, "New", reset_running=True)))
        assert 20 not in out and out[1] == b"preview" and out[28] == 4
        assert parse_fields(out[15]) == [(1, 2, b"New"), (3, 0, 7)]
        assert summary_annotation_title(summary_ui_visible(raw, "New")) == "New"

    def test_drop_entries_removes_summary_and_mtime_entries_of_given_cids(self):
        buf = b"".join(enc_fields([(tag, 2, enc_fields([(1, 2, cid.encode()), (2, 0, 1)]))])
                       for tag, cid in ((1, "a"), (1, "b"), (2, "a")))
        assert list_entry_cids(buf) == {"a", "b"}
        out, removed = drop_entries(buf, {"a"})
        assert removed == 2 and [entry_cid(v) for _, _, v in parse_fields(out)] == ["b"]

    def test_parse_fields_rejects_truncated_wire_payloads(self):
        # Wire type 2 declaring 10 bytes but only 3 bytes present
        with pytest.raises(ValueError, match="truncated"):
            parse_fields(b"\x0a\x0abc")
        # Wire type 1 (fixed64) with only 4 bytes present
        with pytest.raises(ValueError, match="truncated"):
            parse_fields(b"\x09abcd")
        # Wire type 5 (fixed32) with only 2 bytes present
        with pytest.raises(ValueError, match="truncated"):
            parse_fields(b"\x0dab")
        # entry_cid returns empty string cleanly on truncated entry
        assert entry_cid(b"\x0a\x0abc") == ""


class TestTopics:
    def test_only_auto_trade_executor_trades_and_only_auto_technical_repair_self_heals(self):
        assert [k for k, v in TOPIC_SPECS.items() if v.can_trade] == [
            "auto_cryptotrade_executor",
            "auto_sharetrade_executor",
        ]
        assert [k for k, v in TOPIC_SPECS.items() if v.can_self_heal] == ["auto_technical_repair"]
        assert "[auto_cryptotrade_executor]" in topic_spec("auto_cryptotrade_executor").title
        assert topic_spec("auto_trade_executor") == topic_spec("auto_cryptotrade_executor")
        assert "[auto_sharetrade_executor]" in topic_spec("auto_sharetrade_executor").title
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

    def test_gemini_client_compacts_after_prompt_preserving_json_decision(self, tmp_path):
        from unittest.mock import MagicMock, patch
        import subprocess
        from intelligence.sentiment import gemini_client as gc
        from orchestratorOS.agy.topics import TopicSession

        fake_bin = tmp_path / "agy-bot"
        fake_bin.write_text("#!/bin/sh\n")
        client = gc.GeminiClient(cli_path=str(fake_bin))

        cli_calls = []

        def _fake_run(cmd, **kwargs):
            prompt_arg = cmd[cmd.index("-p") + 1]
            cli_calls.append((cmd, prompt_arg))
            if prompt_arg.startswith("/compact "):
                resp = "Sistemul a compactat istoricul conversatiei si functioneaza autonom."
            else:
                resp = '{"summary": "Nominal 5m cycle", "market_trading_decision": {"decision": "HOLD", "symbol": "TAOUSDC", "confidence": 0.91}}'
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f'{{"response": {gc.json.dumps(resp)}, "conversation_id": "cid-perm-1"}}',
                stderr="",
            )

        mock_run = MagicMock(side_effect=_fake_run)
        with patch.object(gc.subprocess, "run", mock_run), \
             patch.object(gc, "prepare_topic_session", return_value=TopicSession("cid-perm-1", "", True)), \
             patch.object(gc, "record_topic_turn") as mock_record:
            parsed = client.query_json("Evaluate TAO & BTC", topic_key="auto_trade_executor", timeout_sec=60.0)

        assert parsed is not None
        assert parsed["market_trading_decision"]["decision"] == "HOLD"
        assert len(cli_calls) == 2
        assert not cli_calls[0][1].startswith("/compact ")
        assert cli_calls[1][1].startswith("/compact ")
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["compacted"] is True
        assert "Nominal 5m cycle" in mock_record.call_args.kwargs["response_text"]




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
        (bot_base / "conversations").mkdir(parents=True, exist_ok=True)
        _make_conv_db(str(bot_base / "conversations" / f"{cid}.db"))
        raw = enc_fields([(1, 2, b"t"), (20, 0, 17)])
        _make_summary_db(str(bot_base / "conversation_summaries.db"),
                         [(cid, "t", "CORTEX_TRAJECTORY_SOURCE_CLI", "proj", 2, raw)])

        ui_base = paths.resolve_ui_cli_base()
        _make_summary_db(os.path.join(ui_base, "conversation_summaries.db"), [])
        visible = visible_conversations_dir(ui_base)
        os.makedirs(visible, exist_ok=True)
        stale = os.path.join(visible, f"{cid}.db")
        with open(stale, "w") as f:
            f.write("old")

        assert mirror_topic_thread_to_ui(cid, "Permanent Name", bot_cli_base=str(bot_base)) is True

        dst = os.path.join(ui_base, "conversations", f"{cid}.db")
        assert sqlite3.connect(dst).execute("SELECT source FROM trajectory_meta").fetchone()[0] == 1
        assert os.path.exists(stale) and os.path.samefile(dst, stale)
        title, source, stored = sqlite3.connect(os.path.join(ui_base, "conversation_summaries.db")).execute(
            "SELECT title, source, raw_summary FROM conversation_summaries WHERE conversation_id=?", (cid,)
        ).fetchone()
        assert (title, source) == ("Permanent Name", "")
        assert stored == summary_ui_visible(raw, "Permanent Name")
        with open(os.path.join(ui_base, "annotations", f"{cid}.pbtxt")) as f:
            assert 'title:"Permanent Name"' in f.read()

    def test_bidirectional_sync_preserves_user_interventions_in_ui_and_hybrid_markdown_json(self, tmp_path):
        cid = "cid-interactive-topic"
        bot_base = tmp_path / "bot" / ".gemini" / "antigravity-cli"
        (bot_base / "conversations").mkdir(parents=True, exist_ok=True)
        _make_conv_db(str(bot_base / "conversations" / f"{cid}.db"), source=17)
        raw = enc_fields([(1, 2, b"t"), (20, 0, 17)])
        _make_summary_db(
            str(bot_base / "conversation_summaries.db"),
            [(cid, "t", "CORTEX_TRAJECTORY_SOURCE_CLI", "autonommptrade", 4, raw)],
        )

        ui_base = paths.resolve_ui_cli_base()
        _make_summary_db(os.path.join(ui_base, "conversation_summaries.db"), [])

        # Initial mirror from bot (step 4) -> UI
        assert mirror_topic_thread_to_ui(cid, "🎯 [auto_trade_executor]", bot_cli_base=str(bot_base)) is True

        # Simulate user intervening in the Web UI: UI conversation advances to step 6
        ui_db = os.path.join(ui_base, "conversations", f"{cid}.db")
        conn = sqlite3.connect(ui_db)
        conn.execute("UPDATE trajectory_meta SET trajectory_id = 'user_intervened'")
        conn.commit()
        conn.close()
        conn = sqlite3.connect(os.path.join(ui_base, "conversation_summaries.db"))
        conn.execute("UPDATE conversation_summaries SET step_count = 6 WHERE conversation_id = ?", (cid,))
        conn.commit()
        conn.close()

        # If mirror_topic_thread_to_ui is called while UI is ahead, it must NOT overwrite the UI's step 6;
        # instead it pulls the UI state back into bot_base.
        assert mirror_topic_thread_to_ui(cid, "🎯 [auto_trade_executor]", bot_cli_base=str(bot_base)) is False
        bot_db = str(bot_base / "conversations" / f"{cid}.db")
        assert sqlite3.connect(bot_db).execute("SELECT trajectory_id FROM trajectory_meta").fetchone()[0] == "user_intervened"
        bot_steps = sqlite3.connect(str(bot_base / "conversation_summaries.db")).execute(
            "SELECT step_count FROM conversation_summaries WHERE conversation_id = ?", (cid,)
        ).fetchone()[0]
        assert bot_steps == 6

        # Hybrid Markdown report + trailing JSON block condenses cleanly and keeps trade=... before long summary
        long_summary = "TAO si BTC in consolidare " + ("x" * 240)
        hybrid_reply = (
            "### Raport TAO & BTC {formula}\n"
            "- **TAOUSDC:** `$260.10` (`BULL +1`)\n"
            "```text\nsample log line\n```\n"
            "```json\n"
            f'{{"summary": "{long_summary}", "market_trading_decision": {{"decision": "HOLD", "symbol": "TAOUSDC", "confidence": 0.88}}}}\n'
            "```"
        )
        condensed = condense_decision(hybrid_reply)
        assert condensed.startswith("trade=HOLD:TAOUSDC@0.88; summary=TAO si BTC in consolidare")

        multi_reply = (
            "```json\n"
            '{"summary": "Protective de-risk", "market_trading_decisions": ['
            '{"decision": "SELL", "symbol": "BTCUSDC", "confidence": 0.91}, '
            '{"decision": "SELL", "symbol": "HYPEUSD", "confidence": 0.89}]}\n'
            "```"
        )
        assert condense_decision(multi_reply).startswith("trade=SELL:BTCUSDC@0.91,SELL:HYPEUSD@0.89; summary=Protective de-risk")
        assert extract_json_object(hybrid_reply)["market_trading_decision"]["symbol"] == "TAOUSDC"
        header = format_topic_turn_header("guard_high_stake", "ntfy-guard: BUY TAOUSDC €850")
        assert header.startswith(TOPIC_SPECS["guard_high_stake"].title)
        assert "ntfy-guard:" not in header

    def test_gemini_client_one_off_fallback_injects_rolling_topic_history(self, tmp_path, monkeypatch):
        import subprocess
        from unittest.mock import patch
        from intelligence.sentiment.gemini_client import GeminiClient
        from orchestratorOS.agy.topics import prepare_topic_session, record_topic_turn

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
        cid1 = "cid-auto-trade-111"
        conn.execute(
            "INSERT OR REPLACE INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid1, "x", "p", datetime.now(timezone.utc).isoformat(), "autonommptrade", 4, "CASCADE_RUN_STATUS_IDLE"),
        )
        conn.commit()
        conn.close()
        with open(os.path.join(cli_base, "conversations", f"{cid1}.db"), "w", encoding="utf-8") as f:
            f.write("db1")

        fake_bin = tmp_path / "agy-bot"
        fake_bin.write_text("#!/bin/sh\n")
        monkeypatch.setenv("AGY_BOT_HOME", str(tmp_path))
        monkeypatch.setenv("LLM_TOPIC_COMPACT_STEPS", "6")
        monkeypatch.setenv("LLM_TOPIC_HISTORY_MAX_ENTRIES", "5")

        record_topic_turn(
            "auto_trade_executor",
            cid1,
            '{"decision": "SELL", "symbol": "TAOUSDC", "confidence": 0.89}',
            cli_base=cli_base,
        )

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if "--conversation" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 15.0))
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"response": "{\\"decision\\": \\"HOLD\\"}", "conversation_id": "detached-999"}', stderr=""
            )

        with patch("subprocess.run", side_effect=fake_run), \
             patch("intelligence.sentiment.gemini_client.resolve_bot_cli_base", return_value=cli_base), \
             patch("intelligence.sentiment.gemini_client.prepare_topic_session", wraps=prepare_topic_session):
            client = GeminiClient(cli_path=str(fake_bin), project_id="mptrade")
            out = client.query_text("Evaluate current spot portfolio.", topic_key="auto_trade_executor", timeout_sec=30.0)

        assert out == '{"decision": "HOLD"}'
        assert len(calls) == 2
        fallback_cmd = calls[1][0]
        assert "--conversation" not in fallback_cmd
        fallback_prompt = fallback_cmd[-1]
        assert "[TOPIC MEMORY — PRIOR DECISIONS IN THIS TOPIC]:" in fallback_prompt
        assert "decision=SELL" in fallback_prompt




