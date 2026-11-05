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
    save_topic_registry,
    sync_all_topic_titles,
    topic_spec,
)
from orchestratorOS.agy.ui_mirror import mirror_topic_thread_to_ui


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

        # 6. Exceeding max_steps rotates the bloated thread, deletes the old .db, and replays rolling memory
        set_steps(cid1, 50)
        rotated = prepare_topic_session("macro_sentiment", cli_base=cli_base, max_steps=40)
        assert rotated.conversation_id is None and "TOPIC MEMORY" in rotated.history_prefix
        assert not os.path.exists(os.path.join(cli_base, "conversations", f"{cid1}.db"))
        assert load_topic_registry()["macro_sentiment"]["rotation_count"] == 1

        # 7. Thread vanished from the store -> recreate once, replaying the rolling memory
        cid2 = "cid-topic-222"
        with open(os.path.join(cli_base, "conversations", f"{cid2}.db"), "w") as f:
            f.write("db2")
        set_steps(cid2, 2)
        record_topic_turn("macro_sentiment", cid2, '{"summary": "rotated turn"}', cli_base=cli_base)
        os.remove(os.path.join(cli_base, "conversations", f"{cid2}.db"))
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
        _make_summary_db(
            os.path.join(ui_base, "conversation_summaries.db"),
            [("stale-dup-cid", "Permanent Name", "", "proj", 1, raw)],
        )
        visible = visible_conversations_dir(ui_base)
        os.makedirs(visible, exist_ok=True)
        stale = os.path.join(visible, f"{cid}.db")
        with open(stale, "w") as f:
            f.write("old")
        stale_dup_db = os.path.join(ui_base, "conversations", "stale-dup-cid.db")
        _make_conv_db(stale_dup_db)

        assert mirror_topic_thread_to_ui(cid, "Permanent Name", bot_cli_base=str(bot_base)) is True

        dst = os.path.join(ui_base, "conversations", f"{cid}.db")
        assert sqlite3.connect(dst).execute("SELECT source FROM trajectory_meta").fetchone()[0] == 1
        assert not os.path.exists(stale)
        assert not os.path.exists(stale_dup_db)
        rows = sqlite3.connect(os.path.join(ui_base, "conversation_summaries.db")).execute(
            "SELECT conversation_id, title, source, raw_summary FROM conversation_summaries"
        ).fetchall()
        assert len(rows) == 1
        row_cid, title, source, stored = rows[0]
        assert (row_cid, title, source) == (cid, "Permanent Name", "")
        assert stored == summary_ui_visible(raw, "Permanent Name")
        with open(os.path.join(ui_base, "annotations", f"{cid}.pbtxt")) as f:
            assert 'title:"Permanent Name"' in f.read()

        # Once hard-linked into the visible dir, sync_all_topic_titles must not unlink unchanged topic threads
        os.link(dst, stale)
        save_topic_registry({"macro_sentiment": {"conversation_id": cid, "title": "old"}})
        assert sync_all_topic_titles(cli_base=str(bot_base)) == 1
        assert os.path.exists(stale) and os.path.samefile(dst, stale)
        assert sync_all_topic_titles(cli_base=str(bot_base)) == 0
        assert os.path.exists(stale) and os.path.samefile(dst, stale)

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
        assert load_topic_registry()["auto_cryptotrade_executor"]["conversation_id"] == "detached-999"
        assert not os.path.exists(os.path.join(cli_base, "conversations", f"{cid1}.db"))

    def test_prune_old_cli_threads_handles_minimal_schema_and_orphan_db_files(self, tmp_path, monkeypatch):
        import time
        from orchestratorOS.admin.prune_llm_threads import prune_old_cli_threads

        cli_base = str(tmp_path / "bot_cli")
        conv_dir = os.path.join(cli_base, "conversations")
        os.makedirs(conv_dir, exist_ok=True)
        monkeypatch.setenv("AGY_PROJECT", "autonommptrade")
        monkeypatch.setenv("AGY_BOT_HOME", str(tmp_path))

        db_path = os.path.join(cli_base, "conversation_summaries.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE conversation_summaries (conversation_id text primary key, project_id text, step_count integer)"
        )
        conn.execute("INSERT INTO conversation_summaries VALUES ('cid-summary-old', 'autonommptrade', 2)")
        conn.commit()
        conn.close()

        summary_db_file = os.path.join(conv_dir, "cid-summary-old.db")
        orphan_db_file = os.path.join(conv_dir, "cid-orphan-no-row.db")
        protected_db_file = os.path.join(conv_dir, "cid-protected.db")
        for fpath in (summary_db_file, orphan_db_file, protected_db_file):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("sqlite-bytes")
            old_ts = time.time() - 3600.0
            os.utime(fpath, (old_ts, old_ts))

        count, freed = prune_old_cli_threads(
            cli_base=cli_base,
            retention_days=0.0,
            exclude_cids={"cid-protected"},
        )
        assert count == 2
        assert freed > 0
        assert not os.path.exists(summary_db_file)
        assert not os.path.exists(orphan_db_file)
        assert os.path.exists(protected_db_file)

    def test_bapi_submit_permit_normalizes_symbol_and_side_and_hl_none_price_uses_ioc(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from binance_api import bapi_placeorder as bp
        from providers.hyperliquid_provider import HyperliquidProvider
        from providers.strategy_executor import SubmissionRefused

        fake_status = SimpleNamespace(
            order_cache_version="v1",
            trade_cache_version="v1",
            order_age_sec=1.0,
            trade_age_sec=1.0,
        )
        monkeypatch.setattr(bp.cfg, "is_trade_enabled", lambda: True)
        monkeypatch.setattr(bp, "require_account_cache_for_submit", lambda: fake_status)
        fake_client = MagicMock()
        fake_client.get_symbol_info.return_value = {
            "baseAsset": "BTC",
            "quoteAsset": "USDC",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01", "maxPrice": "1000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
            ],
        }

        with pytest.raises(SubmissionRefused):
            bp.issue_account_cache_submit_permit(True, "BUY", 1.0, 100.0, api_client=fake_client)
        with pytest.raises(SubmissionRefused):
            bp.cancel_opposite_orders("BUY", True, 100.0)

        permit = bp.issue_account_cache_submit_permit(
            " btcusdc ", "bid", 1.5, 100.0, kind="entry", api_client=fake_client
        )
        fake_client.get_symbol_info.assert_called_once_with("BTCUSDC")
        bp._consume_account_cache_submit_permit(
            permit,
            symbol="BTCUSDC",
            side="BUY",
            qty=1.5,
            actual_price=100.0,
            requested_price=100.0,
            market=False,
            kind="entry",
        )

        hl = HyperliquidProvider(token="HYPE", live_enabled=True)
        monkeypatch.setattr(hl, "_pair", lambda: "@107")
        monkeypatch.setattr(hl, "get_current_price", lambda _s: 20.0)
        monkeypatch.setattr(hl, "round_price", lambda _s, p: round(float(p), 4))
        monkeypatch.setattr(hl, "round_quantity", lambda _s, q: round(float(q), 2))
        preflight_calls = []
        monkeypatch.setattr(
            hl,
            "preflight_order",
            lambda sym, side, qty, px, *, market=False, kind=None: preflight_calls.append((sym, side, qty, px, market, kind)),
        )
        fake_signer = MagicMock()
        fake_signer.sz_decimals.return_value = 2
        monkeypatch.setattr(hl, "_signer", lambda: fake_signer)
        invoke_calls = []

        def _fake_invoke(signer, pair, is_buy, qty, px, **kw):
            invoke_calls.append((pair, is_buy, qty, px, kw))
            return True, 12345, "ok"

        monkeypatch.setattr(hl, "_invoke_spot_order", _fake_invoke)
        oid = hl.submit_order("HYPEUSDC", "BUY", 1.25, price=None, market=False, kind="entry")
        assert oid == "12345"
        assert preflight_calls and preflight_calls[0][4] is True
        assert invoke_calls and invoke_calls[0][4]["is_market"] is True
        assert invoke_calls[0][4]["tif"] == "Ioc"

    def test_query_json_falls_back_to_next_model_when_candidate_returns_non_json_prose(self, tmp_path, monkeypatch):
        import subprocess
        from unittest.mock import patch
        from intelligence.sentiment.gemini_client import GeminiClient

        fake_bin = tmp_path / "agy-bot"
        fake_bin.write_text("#!/bin/sh\n")
        models_called = []

        def fake_run(cmd, **kwargs):
            model_idx = cmd.index("--model") + 1
            m_name = cmd[model_idx]
            models_called.append(m_name)
            if m_name == "gpt-oss-120b-medium":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"response": "**Clarification Needed** I see the latest summary, which option do you want?", "conversation_id": "cid-1"}',
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"response": "{\\"decision\\": \\"HOLD\\", \\"confidence\\": 0.91}", "conversation_id": "cid-1"}',
                stderr="",
            )

        with patch("subprocess.run", side_effect=fake_run):
            client = GeminiClient(
                model="gpt-oss-120b-medium",
                fallback_models=("claude-sonnet-5-5-medium",),
                cli_path=str(fake_bin),
                project_id="autonommptrade",
            )
            parsed = client.query_json("Return JSON decision only.")

        assert models_called == ["gpt-oss-120b-medium", "claude-sonnet-5-5-medium"]
        assert parsed == {"decision": "HOLD", "confidence": 0.91}

    def test_agy_store_fix_mode_post_links_files_mirrored_during_pre_post_gap(self, tmp_path, monkeypatch):
        from orchestratorOS.admin import agy_store_fix as asf

        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _make_conv_db(str(d2 / "cid-in-d2.db"))
        _make_conv_db(str(d1 / "cid-mirrored-gap.db"))

        monkeypatch.setattr(asf, "D1", str(d1))
        monkeypatch.setattr(asf, "D2", str(d2))
        monkeypatch.setattr(asf, "main_pid", lambda: "1234")
        monkeypatch.setattr(asf, "newest_daemon_log_ready", lambda since, pid="": True)
        monkeypatch.setattr(asf.time, "sleep", lambda _s: None)

        assert asf.mode_post(timeout=1) == 0
        assert os.path.samefile(str(d1 / "cid-in-d2.db"), str(d2 / "cid-in-d2.db"))
        assert os.path.samefile(str(d1 / "cid-mirrored-gap.db"), str(d2 / "cid-mirrored-gap.db"))

    def test_prune_ui_store_protects_interactive_and_record_topic_turn_adopts_when_existing_missing(
        self, tmp_path, monkeypatch
    ):
        from orchestratorOS.admin import agy_store_fix as asf
        from orchestratorOS.admin.prune_llm_threads import prune_old_cli_threads
        from orchestratorOS.agy.topics import load_topic_registry, record_topic_turn, save_topic_registry

        bot_base = str(tmp_path / "bot_cli")
        ui_base = str(tmp_path / "ui_cli")
        for b in (bot_base, ui_base):
            os.makedirs(os.path.join(b, "conversations"), exist_ok=True)
        monkeypatch.setenv("AGY_PROJECT", "autonommptrade")
        monkeypatch.setenv("AGY_BOT_HOME", str(tmp_path))

        # Bot store has one old automated thread
        bot_db = os.path.join(bot_base, "conversation_summaries.db")
        con = sqlite3.connect(bot_db)
        con.execute(
            "CREATE TABLE conversation_summaries ("
            "conversation_id text primary key, project_id text, step_count integer, "
            "title text, preview text, last_modified_time text)"
        )
        con.execute(
            "INSERT INTO conversation_summaries VALUES "
            "('bot-auto-old', 'autonommptrade', 2, '🤖 Crypto Trading', 'json', '2020-01-01T00:00:00Z')"
        )
        con.commit()
        con.close()
        _make_conv_db(os.path.join(bot_base, "conversations", "bot-auto-old.db"))

        # UI store has an old multi-turn human interactive conversation (>6 steps) and 2 recent automated threads
        ui_db = os.path.join(ui_base, "conversation_summaries.db")
        ucon = sqlite3.connect(ui_db)
        ucon.execute(
            "CREATE TABLE conversation_summaries ("
            "conversation_id text primary key, project_id text, step_count integer, "
            "title text, preview text, last_modified_time text)"
        )
        ucon.execute(
            "INSERT INTO conversation_summaries VALUES "
            "('ui-human-interactive', 'autonommptrade', 25, 'Debugging strategy', 'Let us inspect logs', '2020-01-01T00:00:00Z'),"
            "('ui-auto-1', 'autonommptrade', 2, '🤖 Crypto Trading', 'json', '2099-01-01T00:00:00Z'),"
            "('ui-auto-2', 'autonommptrade', 2, '🤖 Crypto Trading', 'json', '2099-01-02T00:00:00Z')"
        )
        ucon.commit()
        ucon.close()
        for ucid in ("ui-human-interactive", "ui-auto-1", "ui-auto-2"):
            _make_conv_db(os.path.join(ui_base, "conversations", f"{ucid}.db"))

        monkeypatch.setattr("orchestratorOS.admin.prune_llm_threads.resolve_bot_cli_base", lambda cb=None: bot_base)
        monkeypatch.setattr("orchestratorOS.admin.prune_llm_threads.resolve_ui_cli_base", lambda: ui_base)
        count, _ = prune_old_cli_threads(cli_base=None, retention_days=1.0, max_threads=1)
        assert count == 2  # bot-auto-old + ui-auto-1 (capped by max_threads=1)
        assert os.path.exists(os.path.join(ui_base, "conversations", "ui-human-interactive.db"))
        assert not os.path.exists(os.path.join(ui_base, "conversations", "ui-auto-1.db"))
        assert os.path.exists(os.path.join(ui_base, "conversations", "ui-auto-2.db"))

        # Dynamic LLM_TOPIC_REGISTRY_PATH resolution in agy_store_fix + dead-cid replacement in record_topic_turn
        reg_path = str(tmp_path / "custom_topic_registry.json")
        monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", reg_path)
        save_topic_registry(
            {
                "auto_cryptotrade_executor": {
                    "topic_key": "auto_cryptotrade_executor",
                    "conversation_id": "dead-missing-cid",
                    "turn_count": 3,
                    "rolling_history": [],
                }
            },
            reg_path,
        )
        assert asf.mirrored_topic_cids() == {"dead-missing-cid"}
        assert "dead-missing-cid" in asf.topic_canonical_titles()

        _make_conv_db(os.path.join(bot_base, "conversations", "fresh-adopted-cid.db"))
        con2 = sqlite3.connect(bot_db)
        con2.execute(
            "INSERT INTO conversation_summaries VALUES "
            "('fresh-adopted-cid', 'autonommptrade', 5, '🤖 Crypto Trading', 'json', '2026-01-01T00:00:00Z')"
        )
        con2.commit()
        con2.close()
        record_topic_turn(
            "auto_cryptotrade_executor",
            "fresh-adopted-cid",
            '{"decision": "HOLD"}',
            cli_base=bot_base,
            registry_path=reg_path,
            replace_conversation=False,
        )
        reg = load_topic_registry(reg_path)
        assert reg["auto_cryptotrade_executor"]["conversation_id"] == "fresh-adopted-cid"
        assert reg["auto_cryptotrade_executor"]["step_count"] == 5

    def test_execute_cli_isolates_trading_topics_from_repo_rules_and_includes_for_self_heal(
        self, tmp_path, monkeypatch
    ):
        import subprocess
        from intelligence.sentiment.gemini_client import GeminiClient, _ROOT_DIR

        fake_bin = tmp_path / "agy-bot"
        fake_bin.write_text("#!/bin/sh\n")
        bot_cli_base = tmp_path / "bot_cli"
        bot_cli_base.mkdir()

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            return subprocess.CompletedProcess(cmd, 0, stdout='{"response": "{}"}', stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        client = GeminiClient(cli_path=str(fake_bin), project_id="autonommptrade")

        # 1. Trading/macro/guard query (include_repo_rules=False): runs in bot_cli_base without --add-dir
        client._execute_cli(
            "cid-trade",
            "evaluate market",
            30.0,
            "gemini-4-argon-medium",
            str(bot_cli_base),
            {},
            is_mocked=False,
            include_repo_rules=False,
        )
        assert len(calls) == 1
        trade_cmd, trade_kw = calls[0]
        assert "--add-dir" not in trade_cmd
        assert trade_kw.get("cwd") == str(bot_cli_base)

        # 2. Code self-heal query (include_repo_rules=True): runs in _ROOT_DIR with --add-dir _ROOT_DIR
        client._execute_cli(
            "cid-repair",
            "audit system",
            60.0,
            "gemini-4-argon-medium",
            str(bot_cli_base),
            {},
            is_mocked=False,
            include_repo_rules=True,
        )
        assert len(calls) == 2
        repair_cmd, repair_kw = calls[1]
        assert "--add-dir" in repair_cmd
        assert repair_cmd[repair_cmd.index("--add-dir") + 1] == _ROOT_DIR
        assert repair_kw.get("cwd") == _ROOT_DIR

    def test_execute_cli_restores_newly_dirtied_tracked_files_on_timeout_and_preserves_preexisting_dirty(
        self, tmp_path, monkeypatch
    ):
        import subprocess
        import intelligence.sentiment.gemini_client as gc_mod
        from intelligence.sentiment.gemini_client import GeminiClient, _ORIG_SUBPROCESS_RUN

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _ORIG_SUBPROCESS_RUN(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
        _ORIG_SUBPROCESS_RUN(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True, capture_output=True)
        _ORIG_SUBPROCESS_RUN(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True, capture_output=True)

        clean_file = repo_dir / "manage_agy.sh"
        clean_file.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        pre_dirty_file = repo_dir / "already_dirty.py"
        pre_dirty_file.write_text("x = 1\n", encoding="utf-8")
        _ORIG_SUBPROCESS_RUN(["git", "-C", str(repo_dir), "add", "manage_agy.sh", "already_dirty.py"], check=True, capture_output=True)
        _ORIG_SUBPROCESS_RUN(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

        # Modify already_dirty.py BEFORE _execute_cli starts
        pre_dirty_file.write_text("x = 2\n", encoding="utf-8")

        fake_bin = tmp_path / "agy-bot"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        bot_cli_base = tmp_path / "bot_cli"
        bot_cli_base.mkdir()

        monkeypatch.setattr(gc_mod, "_ROOT_DIR", str(repo_dir))

        def rogue_timeout_run(cmd, **kwargs):
            # Simulate an LLM corrupting a clean tracked file during /compact before timing out
            clean_file.write_text("#!/usr/bin/env bash\nfi\nexit 1\nfi\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=120.0)

        monkeypatch.setattr(subprocess, "run", rogue_timeout_run)
        client = GeminiClient(cli_path=str(fake_bin), project_id="autonommptrade")

        with pytest.raises(subprocess.TimeoutExpired):
            client._execute_cli(
                "cid-repair",
                "/compact Summarize prior turns in place",
                120.0,
                "gpt-oss-120b-medium",
                str(bot_cli_base),
                {},
                is_mocked=False,
                include_repo_rules=True,
            )

        assert clean_file.read_text(encoding="utf-8") == "#!/usr/bin/env bash\nexit 0\n"
        assert pre_dirty_file.read_text(encoding="utf-8") == "x = 2\n"

    def test_record_topic_turn_pins_recent_actionable_trades_across_hold_streaks(self, tmp_path):
        from orchestratorOS.agy.topics import load_topic_registry, record_topic_turn

        reg_path = str(tmp_path / "reg.json")
        cli_base = str(tmp_path / "cli")
        os.makedirs(os.path.join(cli_base, "conversations"), exist_ok=True)
        record_topic_turn(
            "auto_cryptotrade_executor",
            "cid-pin",
            '{"summary": "Re-anchored limit orders", "market_trading_decisions": [{"decision": "SELL", "symbol": "TAOUSDC", "order_type": "LIMIT", "limit_price": 221.4, "confidence": 0.91}]}',
            cli_base=cli_base,
            max_history_entries=5,
            registry_path=reg_path,
        )
        for i in range(6):
            record_topic_turn(
                "auto_cryptotrade_executor",
                "cid-pin",
                f'{{"summary": "Hold cycle {i}", "market_trading_decisions": [{{"decision": "HOLD", "symbol": "BTCUSDC", "confidence": 0.9}}]}}',
                cli_base=cli_base,
                max_history_entries=5,
                registry_path=reg_path,
            )
        hist = load_topic_registry(reg_path)["auto_cryptotrade_executor"]["rolling_history"]
        assert len(hist) == 5
        assert any("SELL(LIMIT@221.4):TAOUSDC@0.91" in h["summary"] for h in hist)











