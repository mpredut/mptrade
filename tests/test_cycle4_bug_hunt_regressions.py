"""Regression tests for Bug-Hunting Cycle #4 fixes across venue bots, clients, and orchestratorOS."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

import botcore
from providers.quantity import format_fixed_decimal
from providers.ws_stream import mount_connect_retry


def test_cycle4_kraken_trailing_submits_marketable_limits_with_post_only_false():
    from kraken import trailing_stop as kts

    captured = {}

    class _DummyExec:
        def submit_order_with_intent(self, intent_id, pair, side, qty, price, **kwargs):
            captured.update(
                {
                    "intent_id": intent_id,
                    "pair": pair,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    **kwargs,
                }
            )
            return SimpleNamespace(accepted=True, order_id="O-KRK-1", state="accepted", reason="")

    tr = kts.KrakenTrailing.__new__(kts.KrakenTrailing)
    tr.executor = _DummyExec()
    tr.core = SimpleNamespace(min_notional=0.0)
    tr.log = lambda *_a, **_kw: None
    tr._pre_submit_order_guard = lambda **_kw: (True, "ok", 1.0)
    tr._submit_and_clear_if_refused = lambda intent, persist, fn: fn()

    intent = {"intent_id": "i-1", "client_order_id": "cid-1", "requested_qty": 2.0, "submitted_qty": 2.0}
    res = tr._submit_lifecycle_order(
        intent, lambda: None, "HYPEUSD", "BUY", 2.0, 40.2, kind="REBUY", market=False
    )
    assert res.accepted is True
    assert captured["post_only"] is False


def test_cycle4_kraken_xstock_env_preserves_symbol_label_and_sync_archive_cursor(tmp_path, monkeypatch):
    from kraken import kraken_xstock_watch as kxw
    import state_io as sio

    with botcore.temporary_environ(
        SYMBOL_LABEL="Kraken",
        XSTOCK_SYMBOL_LABEL="",
        XSTOCK_BOT_ENTRY="500",
        XSTOCK_BOT_DCA="",
        XSTOCK_BOT_TP_TRANCHES="",
    ):
        env = kxw._build_xstock_bot_env(12.5, symbol_label=None, ipo_mode=True)
        assert env["SYMBOL_LABEL"] == "Kraken"
        assert env["STRAT_ENTRY"] == "500"
        assert env.get("STRAT_DCA", "") != "" or "STRAT_DCA" not in env
        assert env["STRAT_TP_TRANCHES"] == ""

    # Archive sync seeds nanosecond cursor from prev_max_ts and persists cursor advances with 0 new rows.
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(cache_dir))
    disk_fn = kxw.Cache365PriceManager.disk_filename_for_symbol("SPCXxUSD")
    target = Path(kxw.cache_path(disk_fn))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    prev_max_ts = 1_700_000_000_000
    sio.atomic_write_json(
        kxw.meta_file_path(str(target)),
        {"max_ts": prev_max_ts, "counts": {"SPCXxUSD": 10}, "rest_sync_cursor": {"SPCXxUSD": prev_max_ts - 1000}},
    )

    seen_since = []

    class _DummyClient:
        def trades_page(self, pair, since):
            seen_since.append(since)
            # Return only already-seen boundary trade, advancing cursor.
            return [["12.5", "1.0", prev_max_ts / 1000.0]], "1700000000999000000"

    added = kxw.sync_trade_archive(_DummyClient(), "SPCXxUSD", max_pages=1)
    assert added == 0
    assert seen_since == [str(prev_max_ts * 1_000_000)]
    saved_meta = sio.read_json_dict(kxw.meta_file_path(str(target)), default={})
    assert saved_meta["kraken_trades_since_ns"] == "1700000000999000000"


def test_cycle4_kraken_client_parse_raises_on_empty_and_non_2xx():
    from kraken.kraken_client import KrakenClient, KrakenError

    with pytest.raises(KrakenError, match="empty response"):
        KrakenClient._parse(0, b"")
    with pytest.raises(KrakenError, match="HTTP 502"):
        KrakenClient._parse(502, b'{"message": "Bad Gateway"}')
    with pytest.raises(KrakenError, match="EService:Unavailable"):
        KrakenClient._parse(503, b'{"error": ["EService:Unavailable"]}')
    assert KrakenClient._parse(200, b'{"error": [], "result": {}}') == {}


def test_cycle4_t212_read_cached_rate_limit_and_auth_fail_closed(monkeypatch):
    t212_mod = importlib.import_module("212trading.t212_client")

    cli = t212_mod.T212Client(
        "k", "s", env="live", min_gap_sec=0.0, order_min_gap_sec=0.0, portfolio_ttl_sec=5.0
    )
    cli._pf_ttl = 5.0
    rate_limits = []
    fail_logs = []
    monkeypatch.setattr(cli, "notify_rate_limit", lambda sec: rate_limits.append(sec))
    monkeypatch.setattr(cli, "_log_read_fail", lambda what, st: fail_logs.append((what, st)))
    monkeypatch.setattr(cli, "_pace_shared_endpoint_locked", lambda _a: None)
    monkeypatch.setattr(cli, "_record_shared_cache_fail", lambda _a: None)

    def _fake_shared_cache(attr, ttl, allow_invalidated=False):
        if allow_invalidated:
            return True, [{"ticker": "NVDA_US_EQ"}], 20.0
        return False, None, 999.0

    monkeypatch.setattr(cli, "_read_shared_cache", _fake_shared_cache)

    # 429 must still trigger notify_rate_limit() and _log_read_fail() even with stale fallback.
    monkeypatch.setattr(t212_mod, "http_get", lambda *_a, **_kw: (429, b""))
    res_429 = cli._read_cached("_pf_cache", "/equity/portfolio", "portfolio", expect_type=list)
    assert res_429 == [{"ticker": "NVDA_US_EQ"}]
    assert len(rate_limits) == 1
    assert ("portfolio", 429) in fail_logs

    # 401/403 auth failure must NEVER serve stale cached data (shared or in-memory).
    cli._pf_cache = (0.0, [{"ticker": "NVDA_US_EQ"}])
    cli._fetch_fail.clear()
    monkeypatch.setattr(t212_mod, "http_get", lambda *_a, **_kw: (401, b'{"error":"unauthorized"}'))
    res_401 = cli._read_cached("_pf_cache", "/equity/portfolio", "portfolio", expect_type=list)
    assert res_401 is None
    assert ("portfolio", 401) in fail_logs


def test_cycle4_ipo_lazy_order_policy_allows_strategy_only_profile(monkeypatch):
    ipo_mod = importlib.import_module("212trading.ipo")
    fake_client = MagicMock()
    fake_client.list_instruments.return_value = [{"ticker": "RGNT_US_EQ", "name": "Regent", "currencyCode": "USD", "isin": "US1"}]
    monkeypatch.setattr(ipo_mod, "build_t212_client_from_env", lambda *_a, **_kw: fake_client)
    monkeypatch.setattr(sys, "argv", ["ipo.py", "--profile", "rgnt", "--find-ticker", "RGNT"])

    with botcore.temporary_environ(
        T212_ENV="live",
        STRAT_ENABLED="true",
        STRAT_EXECUTE="false",
        ORDER_VALIDITY=None,
        ORDER_EXECUTE=None,
    ):
        assert ipo_mod.main() == 0


def test_cycle4_hold_cancel_directives_obey_scope_and_readonly_xstocks():
    from orchestratorOS.lib.technical_repair_ssot import _is_decision_in_scope

    assert _is_decision_in_scope({"decision": "HOLD", "symbol": "SPCX_US_EQ", "venue": "t212"}, "crypto") is True
    assert (
        _is_decision_in_scope(
            {
                "decision": "HOLD",
                "symbol": "SPCX_US_EQ",
                "venue": "t212",
                "cancel_bot_open_orders": True,
                "cancel_order_ids": ["777"],
            },
            "crypto",
        )
        is False
    )
    assert (
        _is_decision_in_scope(
            {
                "decision": "HOLD",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "cancel_order_ids": ["12345"],
            },
            "equities",
        )
        is False
    )
    assert (
        _is_decision_in_scope(
            {
                "decision": "HOLD",
                "symbol": "ANTxUSD",
                "venue": "kraken",
                "cancel_bot_open_orders": True,
                "cancel_order_ids": ["888"],
            },
            "crypto",
        )
        is False
    )
    assert (
        _is_decision_in_scope(
            {
                "decision": "HOLD",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "cancel_bot_open_orders": True,
                "cancel_order_ids": ["999"],
            },
            "crypto",
        )
        is True
    )


def test_cycle4_os_notify_urgent_bypasses_buffer_and_keeps_pending_bulk():
    src = Path("/home/predut/mptrade/orchestratorOS/lib/os_notify.sh").read_text(encoding="utf-8")
    assert "urgent|5|max)" in src
    assert "should_buffer=0" in src
    assert ".pending_bulk_*" not in src.split("-mmin +60 -delete")[0].splitlines()[-1]
    res = subprocess.run(["bash", "-n", "/home/predut/mptrade/orchestratorOS/lib/os_notify.sh"], check=False)
    assert res.returncode == 0


def test_cycle4_vpn_watchdog_checks_creds_before_logout():
    src = Path("/home/predut/mptrade/orchestratorOS/livecheck/vpn_watchdog.sh").read_text(encoding="utf-8")
    body = src.split("rung_relogin() {", 1)[1].split("rung_reinstall()", 1)[0]
    assert body.index("PIA_DIP_TOKEN_FRANKFURT") < body.index("pia logout")
    res = subprocess.run(["bash", "-n", "/home/predut/mptrade/orchestratorOS/livecheck/vpn_watchdog.sh"], check=False)
    assert res.returncode == 0


def test_cycle4_orchestrator_dotenv_overrides_config_env():
    src = Path("/home/predut/mptrade/orchestratorTrade/orchestrator.py").read_text(encoding="utf-8")
    assert 'for env_fname in ("config.env", ".env"):' in src


def test_cycle4_fleet_health_monitor_pgrep_and_tail_guard(monkeypatch):
    from orchestratorOS.livecheck.fleet_health_monitor import FleetHealthMonitor

    captured_cmds = []

    def _fake_run(cmd, **_kw):
        captured_cmds.append(cmd)
        return SimpleNamespace(returncode=0, stdout="STOPPED\n[STEP 4] Done\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    mon = FleetHealthMonitor.__new__(FleetHealthMonitor)
    mon.dev_port = 22
    mon.dev_user = "predut"
    mon.dev_host = "192.168.0.138"
    st = mon.check_dev_backtest_progress()
    assert st["reachable"] is True
    assert st["running"] is False
    remote_cmd = captured_cmds[0][-1]
    assert "[r]un_overnight_master_suite.sh" in remote_cmd
    assert "|| true" in remote_cmd


def test_cycle4_prompt_sections_uses_disp_symbol_for_active_pos_syms(monkeypatch):
    from orchestratorOS.supervisor import prompt_sections as ps_mod

    monkeypatch.setattr(ps_mod, "observe_only_registry_symbols", lambda: {"ETHUSDC"})
    fake_view = {
        "tradable_positions": [
            {"disp_symbol": "ETHUSDC"},
        ],
        "has_open_orders": False,
        "clean_open_limit_orders": [],
        "venue_cycles": [],
        "kraken_xstocks": None,
        "crypto_bellwethers": None,
        "equity_bellwethers": None,
        "clean_last_cycle_execution_status": None,
        "runtime_states": [],
    }
    monkeypatch.setattr(ps_mod, "evaluate_portfolio_prompt_view", lambda *_a, **_kw: fake_view)
    monkeypatch.setattr(ps_mod, "_format_tradable_position_line", lambda pos, **_kw: f"  * {pos['disp_symbol']}")
    ps = {
        "recent_orders": {
            "ETHUSDC": {"side": "SELL", "price": 2000.0, "qty": 0.1, "ts_ms": 1_700_000_000_000}
        }
    }
    lines = ps_mod._format_portfolio_section(ps, asset_scope="crypto", now_ts=1_700_000_060.0)
    assert any("Last Action" in ln and "ETHUSDC" in ln for ln in lines)


def test_cycle4_reconciler_off_and_rth_closed_preserve_eval_artifact(tmp_path):
    from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler

    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cache_dir / "auto_sharetrade_executor_eval.json"
    prev_payload = {"ts": 1_700_000_000.0, "summary": "Genuine RTH eval", "completed_ts": 1_700_000_001.0}
    eval_path.write_text(json.dumps(prev_payload), encoding="utf-8")

    sup = AutonomousAIReconciler(
        cache_dir=str(cache_dir),
        mode="off",
        notify_enabled=False,
        gemini_auditor=MagicMock(),
        intent_gateway=MagicMock(),
    )
    rep_off = sup.run_cycle(topic_key="auto_sharetrade_executor")
    assert "mode is off" in rep_off.summary
    assert json.loads(eval_path.read_text(encoding="utf-8"))["summary"] == "Genuine RTH eval"

    sup.mode = "shadow"
    with patch("orchestratorOS.supervisor.autonomous_ai_reconciler.is_us_rth_open", return_value=False):
        rep_closed = sup.run_cycle(topic_key="auto_sharetrade_executor")
    assert "closed; cycle skipped" in rep_closed.summary
    assert json.loads(eval_path.read_text(encoding="utf-8"))["summary"] == "Genuine RTH eval"


def test_cycle4_binance_nav_raises_on_transient_ticker_error_but_ignores_1121(monkeypatch):
    from binance_api import bapi

    class _CodeErr(RuntimeError):
        def __init__(self, code, msg):
            super().__init__(msg)
            self.code = code

    monkeypatch.setattr(
        bapi,
        "get_account_assets_balances",
        lambda raise_errors=True: [
            {"asset": "USDC", "total": 500.0},
            {"asset": "BTC", "total": 0.1},
            {"asset": "LDDUST", "total": 1.0},
        ],
    )

    def _timeout_ticker(symbol):
        if symbol == "BTCUSDC":
            raise TimeoutError("read timed out")
        raise _CodeErr(-1121, "Invalid symbol")

    monkeypatch.setattr(bapi, "client", SimpleNamespace(get_symbol_ticker=_timeout_ticker))
    with pytest.raises(TimeoutError):
        bapi.get_total_assets_value_usdc(use_cache=False, raise_errors=True)

    # -1121 on unknown dust asset is ignored while BTC prices cleanly.
    def _ok_with_dust_1121(symbol):
        if symbol == "BTCUSDC":
            return {"symbol": "BTCUSDC", "price": "60000.0"}
        raise _CodeErr(-1121, "Invalid symbol")

    monkeypatch.setattr(bapi, "client", SimpleNamespace(get_symbol_ticker=_ok_with_dust_1121))
    val = bapi.get_total_assets_value_usdc(use_cache=False, raise_errors=True)
    assert val == pytest.approx(6500.0)


def test_cycle4_format_fixed_decimal_and_bapi_placeorder_no_scientific_notation(monkeypatch):
    from binance_api import bapi_placeorder as bp

    assert format_fixed_decimal(0.00009) == "0.00009"
    assert format_fixed_decimal(0.00000025) == "0.00000025"
    assert format_fixed_decimal(100.0) == "100"

    sent = {}

    class _FakeClient:
        _post_only_limit_installed = True

        def order_limit_buy(self, **kwargs):
            sent.update(kwargs)
            return {"orderId": 42}

    monkeypatch.setattr(bp.cfg, "is_trade_enabled", lambda: True)
    monkeypatch.setattr(bp, "require_account_cache_for_submit", lambda: None)

    res = bp._submit_binance_order(
        "BUY",
        "BTCUSDC",
        0.00009,
        price=0.00005,
        market=False,
        api_client=_FakeClient(),
        client_order_id="cid-fixed-1",
    )
    assert res == {"orderId": 42}
    assert sent["quantity"] == "0.00009"
    assert sent["price"] == "0.00005"
    assert "e-" not in sent["quantity"].lower()
    assert "e-" not in sent["price"].lower()


def test_cycle4_bapi_ws_event_stream_terminated_and_timestamp_offset(monkeypatch):
    from binance_api import bapi_ws, bapi_client

    monkeypatch.setattr(bapi_ws, "api_key_ws", "k-ws")
    monkeypatch.setattr(bapi_ws, "_sign_ed25519", lambda _k, s: f"sig:{s}")
    monkeypatch.setattr(bapi_client, "_client", SimpleNamespace(timestamp_offset=-2500))
    monkeypatch.setattr(bapi_ws.time, "time", lambda: 1_700_000_000.0)

    stream = bapi_ws.BinanceAccountStream(on_event=lambda _e: None)
    stream._signing_key = object()
    msg = json.loads(stream._signed_logon_msg("login"))
    assert msg["params"]["timestamp"] == 1_700_000_000_000 - 2500

    resyncs = []
    monkeypatch.setattr(bapi_client, "sync_time", lambda: resyncs.append(1))
    stream._maybe_resync_clock_on_1021({"status": 400, "error": {"code": -1021}})
    assert resyncs == [1]

    # eventStreamTerminated must raise instead of marking stream healthy.
    events_delivered = []
    stream2 = bapi_ws.BinanceAccountStream(on_event=lambda e: events_delivered.append(e))
    stream2._signing_key = object()
    frames = iter(
        [
            json.dumps({"id": "login", "status": 200}),
            json.dumps({"id": "sub", "status": 200}),
            json.dumps({"event": {"e": "eventStreamTerminated", "E": 1_700_000_001_000}}),
        ]
    )

    class _FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def send(self, _msg):
            return None

        async def recv(self):
            return next(frames)

    monkeypatch.setattr(bapi_ws.websockets, "connect", lambda *_a, **_kw: _FakeWS())
    with pytest.raises(RuntimeError, match="eventStreamTerminated|terminated"):
        asyncio.run(stream2._connect_and_run())
    assert events_delivered == []


def test_cycle4_mount_connect_retry_excludes_delete_when_read_only_allowed_methods():
    sess = requests.Session()
    ok = mount_connect_retry(
        sess,
        attempts=3,
        backoff=1.0,
        read=3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=True,
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
    )
    assert ok is True
    retry = sess.get_adapter("https://api.binance.com").max_retries
    assert "DELETE" not in retry.allowed_methods
    assert "POST" not in retry.allowed_methods
    assert "GET" in retry.allowed_methods


def test_cycle4_hl_strategy_does_not_cancel_or_leak_filled_close_order():
    hl_strat_mod = botcore.load_local_module(
        "/home/predut/mptrade/hyperliquid", "strategy", module_key="hl_strategy_cycle4_test"
    )
    Strategy = hl_strat_mod.Strategy

    cancels = []

    class _FakeHLClient:
        def cancel(self, coin, oid):
            cancels.append((coin, oid))
            return False

    strat = Strategy.__new__(Strategy)
    strat.coin = "HYPE"
    strat.sign = 1
    strat.dry_run = False
    strat.replay_mode = False
    strat.desktop = False
    strat.sz_dec = 2
    strat._qty_eps = 1e-4
    strat.p = SimpleNamespace(fee_pct=0.04, direction="long")
    strat.client = _FakeHLClient()
    strat._fmt_px = lambda px: f"{px:.4f}"
    strat._notify_fill = lambda *_a, **_kw: None
    strat._zero_position_exposure = lambda clear_spent=True: strat.s.update(
        {"qty": 0.0, "cost": 0.0, **({"spent": 0.0} if clear_spent else {})}
    )
    strat._reset_cycle_state = lambda _keep, orders=None: strat.s.update(
        {"cycle": strat.s.get("cycle", 1) + 1, "orders": list(orders or [])}
    )
    strat.s = {
        "cycle": 1,
        "qty": 10.0,
        "cost": 400.0,
        "spent": 400.0,
        "realized_gross": 0.0,
        "realized_net": 0.0,
        "fees_total": 0.0,
        "cycles_completed": 0,
        "wins": 0,
        "orders": [
            {"oid": "9001", "role": "close", "kind": "TP", "px": 45.0, "sz": 10.0},
            {"oid": "9002", "role": "open", "kind": "DCA", "px": 38.0, "sz": 5.0},
        ],
    }

    # Even if open_orders failed (active contains both 9001 and 9002), filled_close_oid=9001 must NOT be canceled or leaked.
    strat._apply_close(
        40.0,
        45.0,
        10.0,
        real_qty=0.0,
        real_avg=0.0,
        active_oids={"9001", "9002"},
        filled_close_oid="9001",
    )
    assert cancels == [("HYPE", "9002")]
    assert [o["oid"] for o in strat.s["orders"]] == ["9002"]
