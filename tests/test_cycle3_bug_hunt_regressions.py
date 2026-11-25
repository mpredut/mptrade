"""Regression tests for Cycle #3 bug hunt across caches, core infra, providers, execution, guards, and intelligence."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Group 1: Cache Producers, Account/Price Caches, Core Infra, State Stores
# ---------------------------------------------------------------------------


def test_cache_interface_generation_lock_order_matches_bounded_archive(tmp_path):
    """CacheManagerInterface persistence acquires _generation_lock outside self.lock (no AB-BA deadlock)."""
    from cacheprice import Cache365PriceManager

    fn = str(tmp_path / "cache_365_price_BTCUSDC.jsonl")
    mgr = Cache365PriceManager(60, ["BTCUSDC"], fn, data_sync_mode="producer")
    now_ms = int(time.time() * 1000)
    mgr.cache["BTCUSDC"] = [[now_ms - 2000, 60000.0], [now_ms - 1000, 60100.0]]

    events: list[str] = []
    orig_gen_lock = mgr._generation_lock
    orig_lock = mgr.lock

    class _TrackedLock:
        def __enter__(self):
            events.append("self.lock")
            return orig_lock.__enter__()

        def __exit__(self, *a):
            return orig_lock.__exit__(*a)

    class _TrackedGenCM:
        def __init__(self, cm):
            self._cm = cm

        def __enter__(self):
            events.append("gen_lock")
            return self._cm.__enter__()

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

    mgr.lock = _TrackedLock()
    mgr._generation_lock = lambda: _TrackedGenCM(orig_gen_lock())

    assert mgr._save_state_to_file_impl() is True
    assert events.index("gen_lock") < events.index("self.lock")

    events.clear()
    assert mgr.compact_jsonl() is True
    assert events.index("gen_lock") < events.index("self.lock")


def test_portfolio_snapshot_includes_kraken_locked_base_and_skips_missing_price_alerts(tmp_path, monkeypatch):
    """_live_kraken_held_qty sums free_base + locked_base_qty; missing prices must not wipe alert cooldowns."""
    from verify_tools import portfolio_snapshot as ps

    vb = {
        "pos:kraken:HYPEUSD": {"free_base": 2.0, "locked_base_qty": 8.0},
        "pos:kraken:SPCXxUSD": {"free_base": 0.0, "locked_base_qty": 12.5},
    }
    assert ps._live_kraken_held_qty(vb, "HYPEUSD", "HYPE") == pytest.approx(10.0)
    assert ps._live_kraken_held_qty(vb, "SPCXxUSD", "SPCXx") == pytest.approx(12.5)
    # Fallback to venue balance cache sums free + locked
    kr_bal = {"ANTx": {"free": 1.5, "locked": 3.5}}
    assert ps._live_kraken_held_qty({}, "ANTxUSD", "ANTx", kraken_bal_cache=kr_bal) == pytest.approx(5.0)

    # run_alerts must NOT pop existing alert cooldown when price_ok is False
    st_file = tmp_path / ".risk_alert_state.json"
    st_file.write_text(
        json.dumps({"Kraken HYPE": {"dd": -15.0, "ts": "2026-11-25T20:00:00+00:00"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ps, "ALERT_STATE", str(st_file))
    fired = ps.run_alerts([{"engine": "Kraken HYPE", "paper": False, "qty": 10.0, "dd_pct": 0.0, "price_ok": False}])
    assert fired == []
    persisted = json.loads(st_file.read_text(encoding="utf-8"))
    assert "Kraken HYPE" in persisted


def test_state_stores_resolve_flat_paths_and_preserve_empty_cached_dict(tmp_path):
    """Flat cachedb/<file> paths resolve to canonical subdomain paths during RMW without wiping sibling data."""
    from cachepaths import (
        LLM_QUOTA_STATUS_FILENAME,
        MICROSTRUCTURE_CYCLE_HISTORY_FILENAME,
        ORDER_RETRY_QUARANTINE_ALERTS_FILENAME,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
        VENUE_BALANCES_STATE_FILENAME,
        resolve_cachedb_filepath,
    )
    from state_stores.llm_quota_status import LlmQuotaStatusStore
    from state_stores.microstructure_cycle_history import MicrostructureCycleHistoryStore
    from state_stores.order_retry_quarantine_alerts import OrderRetryQuarantineAlertsStore
    from state_stores.portfolio_positions_state import PortfolioPositionsStateStore
    from state_stores.venue_balances_state import VenueBalancesStateStore

    cdir = tmp_path / "cachedb"
    cdir.mkdir(parents=True, exist_ok=True)
    for sub in ("portfolio", "orders", "bots", "llm", "state"):
        (tmp_path / "statedb" / sub).mkdir(parents=True, exist_ok=True)
    now_s = time.time()

    # 1. VenueBalancesStateStore: seed canonical statedb/portfolio file, upsert via flat cachedb/ path -> sibling preserved
    flat_vb = str(cdir / VENUE_BALANCES_STATE_FILENAME)
    resolved_vb = VenueBalancesStateStore.resolve_path(str(cdir))
    os.makedirs(os.path.dirname(resolved_vb), exist_ok=True)
    with open(resolved_vb, "w", encoding="utf-8") as fh:
        json.dump({"cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1000.0, "ts": now_s}}, fh)
    assert VenueBalancesStateStore.upsert_locked(
        flat_vb, {"cash:kraken:USD": {"venue": "kraken", "asset": "USD", "free": 500.0, "ts": now_s}}
    )
    vb_after = json.loads(open(resolved_vb, encoding="utf-8").read())
    assert "cash:binance:USDC" in vb_after and "cash:kraken:USD" in vb_after

    # 2. PortfolioPositionsStateStore: same protection
    flat_pp = str(cdir / PORTFOLIO_POSITIONS_STATE_FILENAME)
    resolved_pp = PortfolioPositionsStateStore.resolve_path(str(cdir))
    os.makedirs(os.path.dirname(resolved_pp), exist_ok=True)
    with open(resolved_pp, "w", encoding="utf-8") as fh:
        json.dump({"pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.1, "ts": now_s}}, fh)
    assert PortfolioPositionsStateStore.upsert_locked(
        flat_pp, {"pos:kraken:HYPEUSD": {"venue": "kraken", "symbol": "HYPEUSD", "free_base": 5.0, "ts": now_s}}
    )
    pp_after = json.loads(open(resolved_pp, encoding="utf-8").read())
    assert "pos:binance:BTCUSDC" in pp_after and "pos:kraken:HYPEUSD" in pp_after

    # 3. MicrostructureCycleHistoryStore, OrderRetryQuarantineAlertsStore, LlmQuotaStatusStore
    flat_mh = str(cdir / MICROSTRUCTURE_CYCLE_HISTORY_FILENAME)
    resolved_mh = MicrostructureCycleHistoryStore.resolve_path(str(cdir))
    os.makedirs(os.path.dirname(resolved_mh), exist_ok=True)
    with open(resolved_mh, "w", encoding="utf-8") as fh:
        json.dump({"BTCUSDC": [{"ts": now_s, "px": 60000.0}]}, fh)
    assert "BTCUSDC" in MicrostructureCycleHistoryStore.load_path(flat_mh)

    flat_qa = str(cdir / ORDER_RETRY_QUARANTINE_ALERTS_FILENAME)
    resolved_qa = OrderRetryQuarantineAlertsStore.resolve_path(str(cdir))
    os.makedirs(os.path.dirname(resolved_qa), exist_ok=True)
    with open(resolved_qa, "w", encoding="utf-8") as fh:
        json.dump({"k1": [["a"]], "k2": [["b"]]}, fh)
    assert OrderRetryQuarantineAlertsStore.clear_key_locked(flat_qa, "k1") is True
    assert "k1" not in json.loads(open(resolved_qa, encoding="utf-8").read())

    flat_lq = str(cdir / LLM_QUOTA_STATUS_FILENAME)
    resolved_lq = LlmQuotaStatusStore.resolve_path(str(cdir))
    os.makedirs(os.path.dirname(resolved_lq), exist_ok=True)
    with open(resolved_lq, "w", encoding="utf-8") as fh:
        json.dump({"available_credits": 12.5}, fh)
    assert LlmQuotaStatusStore.read_recovering_path(flat_lq)["available_credits"] == 12.5

    # 4. Empty canonical {} file with use_cache=True must NOT fall through to stale legacy cachedb/state/ file
    with open(resolved_vb, "w", encoding="utf-8") as fh:
        json.dump({}, fh)
    legacy_vb = cdir / "state" / VENUE_BALANCES_STATE_FILENAME
    legacy_vb.parent.mkdir(parents=True, exist_ok=True)
    legacy_vb.write_text(json.dumps({"cash:stale:USD": {"free": 9999.0}}), encoding="utf-8")
    assert VenueBalancesStateStore.read_dict(str(cdir), use_cache=True) == {}


def test_cache_order_filled_partial_overlapping_batch_preserves_earlier_fills(tmp_path):
    """Overlapping incremental batch [F2, F3] into existing [F1, F2] without _fills must preserve F1 qty."""
    from cacheaccount import CacheOrderFilledManager

    cache_file = str(tmp_path / "cache_order_filled_kraken.json")
    now_ms = 1_795_500_000_000
    # Seed legacy/prior row with F1 (1.0 @ 100) + F2 (1.0 @ 100) -> qty 2.0, no _fills dict
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "items": {
                    "HYPEUSD": [
                        {
                            "orderId": "ORD-1",
                            "symbol": "HYPEUSD",
                            "side": "BUY",
                            "price": 100.0,
                            "quantity": 2.0,
                            "timestamp": now_ms,
                            "_fillIds": ["F1", "F2"],
                        }
                    ]
                },
                "fetchtime": {"HYPEUSD": now_ms},
            },
            fh,
        )
    # Overlapping incremental batch contains F2 (1.0 @ 100) + unseen F3 (1.5 @ 110)
    CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
        cache_file,
        {
            "HYPEUSD": [
                {"id": "F2", "orderId": "ORD-1", "side": "BUY", "price": 100.0, "qty": 1.0, "timestamp": now_ms},
                {"id": "F3", "orderId": "ORD-1", "side": "BUY", "price": 110.0, "qty": 1.5, "timestamp": now_ms + 1000},
            ]
        },
        venue="kraken",
        now_ms=now_ms + 2000,
        is_snapshot=False,
    )
    out = json.loads(open(cache_file, encoding="utf-8").read())
    row = out["items"]["HYPEUSD"][0]
    # Total qty MUST be 2.0 (F1+F2) + 1.5 (unseen F3) = 3.5, NOT overwritten to 2.5!
    assert row["quantity"] == pytest.approx(3.5, abs=1e-6)
    assert set(row["_fillIds"]) == {"F1", "F2", "F3"}


def test_kraken_ws_loop_backs_off_on_periodic_rest_sync_failure(tmp_path, monkeypatch):
    """Failed periodic REST Ledgers sync must back off rather than firing every 15s keepalive."""
    import cacheaccount as ca

    calls = []
    now_box = [1_795_500_000_000]
    stop_ev = threading.Event()

    class _FakeWS:
        def __init__(self, url, on_open=None, on_message=None, on_error=None, on_close=None):
            self.on_open = on_open
            self.on_message = on_message

        def run_forever(self, **_kw):
            self.on_open(self)
            # Keepalive 1 at +310s -> triggers REST sync (which raises rate limit)
            now_box[0] += 310_000
            self.on_message(self, '{"event":"heartbeat"}')
            time.sleep(0.05)
            # Keepalive 2 only +16s later -> must be suppressed by 30s+ backoff!
            now_box[0] += 16_000
            self.on_message(self, '{"event":"heartbeat"}')
            time.sleep(0.05)
            stop_ev.set()

    def _fake_fetch(*_a, **_kw):
        calls.append(now_box[0])
        if len(calls) == 1:
            return {"HYPEUSD": []}  # seed succeeds
        raise RuntimeError("EAPI:Rate limit exceeded")

    fake_prov = SimpleNamespace(
        PRIVATE_WS_URL="wss://ws-auth.kraken.com/",
        fetch_ledger_fills_by_symbol=_fake_fetch,
        build_ws_own_trades_subscription=lambda _c: "{}",
        extract_ws_own_trades=lambda _d, pairs=None: None,
        classify_ws_control_event=lambda _d: ("keepalive", None),
    )
    fake_ws_mod = SimpleNamespace(WebSocketApp=_FakeWS)
    monkeypatch.setitem(sys.modules, "websocket", fake_ws_mod)
    monkeypatch.setattr(ca, "resolve_now_ms", lambda _v=None: now_box[0])
    monkeypatch.setattr(ca, "_record_venue_health_sync", lambda *_a, **_kw: None)
    monkeypatch.setattr("providers.kraken_provider.KrakenProvider", fake_prov)

    ca.run_kraken_trade_ws_loop(
        SimpleNamespace(),
        cache_file=str(tmp_path / "cache_trade_kraken.json"),
        pairs=["HYPEUSD"],
        poll_interval=30.0,
        backoff_init=5.0,
        backoff_max=60.0,
        lookback_sec=3600.0,
        stop_event=stop_ev,
    )
    # Exactly 2 calls: 1 initial seed + 1 failed periodic sync (2nd keepalive +16s later skipped due to backoff)
    assert len(calls) == 2


def test_alertnotifiers_get_notification_server_registers_canonical_module(monkeypatch):
    """_get_notification_server must register sys.modules['notify_engine.server'] cleanly."""
    from notify_engine import alertnotifiers as an

    prev_mod = sys.modules.pop("notify_engine.server", None)
    prev_cls = an._NOTIFICATION_SERVER_CLS
    try:
        an._NOTIFICATION_SERVER_CLS = None
        srv = an._get_notification_server()
        assert srv is not None
        assert "notify_engine.server" in sys.modules
        assert type(srv).__name__ == "NotificationServer"
    finally:
        an._NOTIFICATION_SERVER_CLS = prev_cls
        if prev_mod is not None:
            sys.modules["notify_engine.server"] = prev_mod


def test_new_coins_monitor_atomic_refresh_and_is_trade_enabled_throttle(tmp_path, monkeypatch):
    """NewCoinsMonitor refreshes atomically under lock; is_trade_enabled advances _CONFIG_LAST_CHECK on unchanged mtime."""
    from core import config as cfg
    from market_monitor.new_coins_discovery import NewCoinsMonitor

    fake_factory = SimpleNamespace(
        refresh_all=lambda: None,
        get_all_new_coins=lambda _d: {"cmc": [{"symbol": "NEWX", "name": "NewX", "slug": "newx"}]},
        get_available_sources=lambda: ["cmc"],
    )
    mon = NewCoinsMonitor(price_monitor=None, factory=fake_factory)
    assert "NEWX" in mon.all_symbols
    assert hasattr(mon, "_lock")

    # is_trade_enabled throttle check
    import botcore

    cfg_file = tmp_path / "config.env"
    cfg_file.write_text("TRADE_ENABLED=true\n", encoding="utf-8")
    txt_file = tmp_path / "config.txt"
    stat_calls = []
    orig_mtime = cfg._file_mtime

    def _counting_mtime(p):
        stat_calls.append(p)
        return orig_mtime(p)

    monkeypatch.setattr(cfg, "CONFIG_ENV_PATH", str(cfg_file))
    monkeypatch.setattr(cfg, "CONFIG_TXT_PATH", str(txt_file))
    monkeypatch.setattr(botcore, "CONFIG_ENV_PATH", str(cfg_file), raising=False)
    monkeypatch.setattr(botcore, "CONFIG_TXT_PATH", str(txt_file), raising=False)
    monkeypatch.setattr(cfg, "_file_mtime", _counting_mtime)
    monkeypatch.setattr(cfg, "_CONFIG_MTIME", (orig_mtime(str(cfg_file)), orig_mtime(str(txt_file))))
    monkeypatch.setattr(cfg, "_CONFIG_LAST_CHECK", 0.0)
    monkeypatch.setattr(cfg, "config_cache", {"trade_enabled": True})
    monkeypatch.setattr(botcore, "config_cache", {"trade_enabled": True}, raising=False)

    t_now = [1_795_500_100.0]
    monkeypatch.setattr(cfg._time, "time", lambda: t_now[0])
    assert cfg.is_trade_enabled() is True
    first_count = len(stat_calls)
    assert first_count == 2
    # 1 second later (within 5s throttle window): MUST NOT re-stat unchanged files
    t_now[0] += 1.0
    assert cfg.is_trade_enabled() is True
    assert len(stat_calls) == first_count


# ---------------------------------------------------------------------------
# Group 2: Venue Providers, Order Execution, Retry Queue, Valuation & Guards
# ---------------------------------------------------------------------------


def test_hyperliquid_submit_order_timeout_raises_provider_error_not_refused():
    """HyperliquidProvider.submit_order raises ProviderError on transport timeout, SubmissionRefused only on refusal."""
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.ohlc_utils import ProviderError
    from providers.strategy_executor import PairPrecision, SubmissionRefused

    prov = HyperliquidProvider(client=SimpleNamespace())
    prov.execution_enabled = lambda: True
    prov._pair = lambda: "HYPE/USDC"
    prov.validate_symbol = lambda _s: None
    prov._require_live_pair = lambda _s, _c: "HYPE/USDC"
    prov._resolve_order_price = lambda *_a, **_kw: 50.0
    prov._round_required_qty = lambda _s, q, _c: q
    prov.preflight_order = lambda *_a, **_kw: None

    # Timeout -> ProviderError (unknown outcome, preserves CLOID reconciliation)
    prov._execute_spot_order_live = lambda *_a, **_kw: (False, None, "ReadTimeout: timed out", None)
    with pytest.raises(ProviderError) as ei:
        prov.submit_order("HYPEUSDC", "BUY", 1.0, 50.0)
    assert not isinstance(ei.value, SubmissionRefused)

    # Definitive venue rejection -> SubmissionRefused
    prov._execute_spot_order_live = lambda *_a, **_kw: (False, None, "Order must have minimum value of $10", None)
    with pytest.raises(SubmissionRefused):
        prov.submit_order("HYPEUSDC", "BUY", 1.0, 50.0)


def test_kraken_balance_lookup_sums_spot_and_auto_earn_keys():
    """KrakenProvider sums spot + .F/.S keys even when spot key is 0.0."""
    from providers.kraken_provider import KrakenProvider

    bal_ex = {
        "ZUSD": {"balance": "0", "hold_trade": "0"},
        "ZUSD.F": {"balance": "808.83", "hold_trade": "800"},
        "USDC": {"balance": "0", "hold_trade": "0"},
        "USDC.F": {"balance": "2049.58", "hold_trade": "0"},
        "HYPE": {"balance": "0", "hold_trade": "0"},
        "HYPE.F": {"balance": "50", "hold_trade": "10"},
    }
    cli = SimpleNamespace(balance_ex=lambda: bal_ex, balance=lambda: {})
    kp = KrakenProvider(client=cli)

    usd_free, usd_hold = kp.free_and_locked_balance("USD")
    assert usd_free == pytest.approx(8.83 + 2049.58, abs=1e-4)
    assert usd_hold == pytest.approx(800.0, abs=1e-4)

    hype_free, hype_hold = kp.free_and_locked_balance("HYPE")
    assert hype_free == pytest.approx(40.0)
    assert hype_hold == pytest.approx(10.0)

    bd = kp.usd_cash_breakdown()
    assert bd["USD"] == pytest.approx(8.83, abs=0.01)
    assert bd["USDC"] == pytest.approx(2049.58, abs=0.01)

    all_b = kp.all_free_and_locked_balances(["USD", "HYPE"])
    assert all_b["USD"][0] == pytest.approx(8.83 + 2049.58, abs=1e-4)
    assert all_b["HYPE"] == pytest.approx((40.0, 10.0))


def test_portfolio_valuation_locked_base_falls_back_to_qty_times_price():
    """Locked base inventory without live_prices or locked_base_usd must still value as locked_base_qty * price."""
    from providers.portfolio_valuation import compute_venue_nav_breakdown

    now_s = time.time()
    out = compute_venue_nav_breakdown(
        venue_balances_state={
            "pos:kraken:SPCXxUSD": {
                "venue": "kraken",
                "symbol": "SPCXxUSD",
                "base": "SPCXx",
                "free_base": 10.0,
                "price": 120.0,
                "value_usd": 1200.0,
                "ts": now_s,
            }
        },
        venue_balance_caches={
            "kraken": {
                "fetchtime": now_s,
                "items": {
                    "SPCXx": {"free": 10.0, "locked": 5.0},
                },
            }
        },
        live_prices={},
    )
    # 10 free * 120 + 5 locked * 120 = 1800.0 (not 1200.0)
    assert out["invested_usd"] == pytest.approx(1800.0, abs=0.05)


def test_venue_fee_maintenance_none_cache_dir_and_bnb_burn_cooldown(tmp_path, monkeypatch):
    """run_venue_fee_maintenance(cache_dir=None) preserves other venues; BinanceFeeMaintainer keeps 24h burn cooldown."""
    from providers import venue_fee_maintenance as vfm
    from state_stores.cache_fee_schedule import CacheFeeScheduleStore
    from state_stores.venue_fee_maintenance_state import VenueFeeMaintenanceStateStore

    sched_file = tmp_path / "cache_fee_schedule.json"
    state_file = tmp_path / "venue_fee_maintenance_state.json"
    sched_file.write_text(
        json.dumps(
            {
                "ts": 1000.0,
                "venues": {
                    "kraken": {"fee_spec": {"tier": "K0", "maker_fee_bps": 25.0, "taker_fee_bps": 40.0}},
                },
            }
        ),
        encoding="utf-8",
    )
    state_file.write_text(json.dumps({"ts": 1000.0, "venues": {}}), encoding="utf-8")

    monkeypatch.setattr(CacheFeeScheduleStore, "resolve_schedule_path", classmethod(lambda cls, *_a, **_kw: str(sched_file)))
    monkeypatch.setattr(VenueFeeMaintenanceStateStore, "resolve_path", classmethod(lambda cls, *_a, **_kw: str(state_file)))

    class _DummyBinance(vfm.VenueFeeMaintainerBase):
        venue = "binance"

        def maintain(self, now_ts, prev_venue_state, cfg, *, dry_run=False):
            return {"ts": now_ts, "fee_spec": self.build_fee_spec("VIP0_BNB", 7.5, 7.5), "events": []}

    res = vfm.run_venue_fee_maintenance(
        venues=["binance"],
        force=True,
        cache_dir=None,
        maintainers={"binance": _DummyBinance()},
        now_ts=20000.0,
    )
    assert res["status"] == "ok"
    saved_sched = json.loads(sched_file.read_text(encoding="utf-8"))
    assert "kraken" in saved_sched["venues"] and "binance" in saved_sched["venues"]

    # BinanceFeeMaintainer 24h burn error cooldown must NOT be reset when suppressed
    cli = SimpleNamespace(
        get_bnb_burn_spot_margin=lambda: (_ for _ in ()).throw(RuntimeError("SAPI 403")),
        get_account=lambda: {"balances": [{"asset": "BNB", "free": "1.0", "locked": "0.0"}, {"asset": "USDC", "free": "500.0", "locked": "0.0"}]},
        get_orderbook_ticker=lambda **_kw: {"symbol": "BNBUSDC", "bidPrice": "600.0"},
        get_open_orders=lambda **_kw: [],
        get_trade_fee=lambda **_kw: [],
    )
    maint = vfm.BinanceFeeMaintainer(client=cli)
    cfg = {**vfm.load_fee_maintenance_config(), "kraken_idle_usdc_alert_cooldown_sec": 86400.0}
    t0 = 1_795_500_000.0
    out1 = maint.maintain(t0, {}, cfg, dry_run=False)
    assert out1["last_bnb_burn_error_ts"] == round(t0, 3)
    assert any("Spot BNB Burn" in e.get("title", "") for e in out1["events"])
    # 4h later (< 24h): alert suppressed AND last_bnb_burn_error_ts stays at t0 (not overwritten to t0+14400)
    out2 = maint.maintain(t0 + 14400.0, out1, cfg, dry_run=False)
    assert out2["last_bnb_burn_error_ts"] == round(t0, 3)
    assert not any("Spot BNB Burn" in e.get("title", "") for e in out2["events"])


def test_order_retry_market_ref_price_and_multi_deferral_ttl(tmp_path, monkeypatch):
    """MARKET retry falls back to ref_price when requested_price is None; multi-tick deferral preserves TTL."""
    import order_retry as oq
    import order_retry_worker as orw

    rec_mkt = {"side": "SELL", "requested_price": None, "ref_price": 100.0, "place_kwargs": {"smart": False}}
    assert oq.price_gate_ok(rec_mkt, 100.5) is True
    kw: dict = {}
    orw._prepare_retry_placement_kwargs(rec_mkt, kw, "binance", now=1000.0)
    assert kw["_retry_requested_price"] == 100.0

    # Multi-tick deferral after 1 failure must shift ttl_started_ts by the FULL deferral span
    q_file = tmp_path / "order_retry_queue.jsonl"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "RETRY_TTL_SEC", 1800.0)
    rid = oq.enqueue("BTCUSDC", "SELL", 0.01, {}, requested_price=60000.0, failure_reason="network_error", now=1000.0)
    assert isinstance(rid, str) and rid

    # Attempt 1 fails at t=1060 (60s active)
    c1 = oq.claim([rid], now=1060.0)[0]
    oq.complete_claim(c1, outcome="failure", failure_reason="network_error", now=1060.0)

    # Deferred repeatedly on trend_deferred from t=1400 .. t=3500 (2440s deferred)
    for t_def in (1400.0, 2000.0, 2600.0, 3200.0):
        cd = oq.claim([rid], now=t_def)[0]
        oq.complete_claim(cd, outcome="deferred", failure_reason="trend_deferred", now=t_def)

    # Non-deferred failure at t=3600 -> ttl_started_ts shifted by (3600 - 1060) = 2540s -> active age = 60s < 1800s!
    c2 = oq.claim([rid], now=3600.0)[0]
    oq.complete_claim(c2, outcome="failure", failure_reason="network_error", now=3600.0)
    items = oq.load_all()
    assert len(items) == 1
    assert not oq.is_expired(items[0], now=3600.0)


def test_order_guard_dynamic_mode_queries_provider_when_window_ref_none():
    """check_math_profit_reference in mode='dynamic' queries provider.get_orders even when window_ref is None."""
    import order_guard

    now_ms = int(time.time() * 1000)
    prov = SimpleNamespace(
        name="binance",
        get_orders=lambda sym, side, sec: [{"side": "SELL", "price": 100.0, "timestamp": now_ms - 3600_000}],
    )
    # Buying at 105.0 (-5% below last SELL @ 100.0) MUST be BLOCKED even when window_ref=None
    ok = order_guard.check_math_profit_reference(
        prov,
        "BTCUSDC",
        "BUY",
        105.0,
        profit_percentage=1.0,
        window_ref=None,
        reference_mode="dynamic",
    )
    assert ok is False


def test_t212_provider_fills_from_api_raises_provider_error_on_api_failure():
    """T212Provider._fills_from_api must fail closed (raise ProviderError) when API raises and no cache exists."""
    from providers.ohlc_utils import ProviderError
    from providers.t212_provider import T212Provider

    bad_cli = SimpleNamespace(
        refresh_order_history=lambda **_kw: (_ for _ in ()).throw(RuntimeError("HTTP 429")),
    )
    prov = T212Provider(client=bad_cli)
    with pytest.raises(ProviderError):
        prov._fills_from_api("SPCX_US_EQ", allow_cache_fallback=False)


def test_ai_survival_guard_xstock_zero_fee_and_lifo_prorated_notional_reload(tmp_path):
    """Fee resolver preserves mixed-case Kraken xStocks (0 bps); LIFO uses prorated remaining sell USD."""
    from providers.ai_survival_guard import build_persisted_round_trip_fee_resolver, match_filled_intent_rows_lifo
    from cachepaths import CACHE_FEE_SCHEDULE_FILENAME, resolve_cachedb_filepath

    cdir = tmp_path / "cachedb"
    sched_path = str(resolve_cachedb_filepath(str(cdir / CACHE_FEE_SCHEDULE_FILENAME)))
    os.makedirs(os.path.dirname(sched_path), exist_ok=True)
    with open(sched_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": time.time(),
                "venues": {
                    "kraken": {
                        "ts": time.time(),
                        "fee_spec": {"tier": "K0", "maker_fee_bps": 16.0, "taker_fee_bps": 26.0},
                    }
                },
            },
            fh,
        )
    resolver = build_persisted_round_trip_fee_resolver(str(cdir), now_ts=time.time())
    assert resolver is not None
    assert resolver("kraken", "NVDAxUSD", "LIMIT") == pytest.approx(0.0)

    # LIFO partial reload + net-new BUY exposure must surface in open_buys
    rows = [
        {"ts": 100.0, "inst_key": "binance:BTCUSDC", "side": "SELL", "qty": 10.0, "price": 100.0, "usd": 1000.0},
        {"ts": 200.0, "inst_key": "binance:BTCUSDC", "side": "BUY", "qty": 9.0, "price": 100.0, "usd": 900.0},
        {"ts": 300.0, "inst_key": "binance:BTCUSDC", "side": "BUY", "qty": 10.0, "price": 100.0, "usd": 1000.0},
    ]
    _, open_buys, open_sells = match_filled_intent_rows_lifo(rows)
    assert open_sells == []
    assert len(open_buys) == 1
    assert open_buys[0]["qty"] == pytest.approx(9.0)


def test_kraken_xstock_case_quantity_base_fee_and_attribution_price_merge():
    """Kraken wildcard/remote trades preserve xStock case; base-fee SELL PnL is exact; attribution updates placeholder px."""
    from providers.ai_attribution import _merge_audit_order_entry
    from providers.kraken_provider import KrakenProvider
    from providers.quantity import summarize_fills_position_and_cost
    from providers.venue_ws_protocols import KrakenWsLedgerMixin

    assert KrakenWsLedgerMixin.match_configured_pair("SPCXxUSD", pairs=["*"]) == "SPCXxUSD"
    assert KrakenWsLedgerMixin.match_configured_pair("NVDAxUSD", pairs=["*"]) == "NVDAxUSD"

    seen_pairs = []
    kp = KrakenProvider(client=SimpleNamespace(_private=lambda *_a, **_kw: {}))
    kp.fetch_ledger_fills_by_symbol = lambda _c, pairs=None, **_kw: seen_pairs.extend(pairs or ()) or {"SPCXxUSD": []}
    kp._remote_trade_rows_for_symbol("SPCXxUSD", "SPCXXUSD", 0.0, client=SimpleNamespace(_private=lambda *_a, **_kw: {}))
    assert seen_pairs == ["SPCXxUSD"]

    # Buy 10 @ 100, sell 5 @ 120 with 0.1 base fee -> proceeds 600 - cost 510 = 90.0 (not 102.0)
    summary = summarize_fills_position_and_cost(
        [
            {"side": "BUY", "qty": 10.0, "price": 100.0, "timestamp": 1000},
            {"side": "SELL", "qty": 5.0, "price": 120.0, "fee": 0.1, "fee_asset": "HYPE", "timestamp": 2000},
        ],
        base_asset="HYPE",
        quote_asset="USD",
    )
    assert summary["realized_net"] == pytest.approx(90.0, abs=1e-6)

    # Attribution fill price replaces placeholder submit price
    dst = {"is_fill": False, "price": 100.0, "filled_qty": 0.0, "cost_usd": 0.0, "fee_usd": 0.0, "ts": 100.0}
    src = {"is_fill": True, "price": 98.5, "filled_qty": 1.0, "cost_usd": 98.5, "fee_usd": 0.1, "ts": 105.0}
    _merge_audit_order_entry(dst, src)
    assert dst["is_fill"] is True
    assert dst["price"] == pytest.approx(98.5)


# ---------------------------------------------------------------------------
# Group 3: Intelligence Guards, Quota Router, Evaluators & Delta Gate
# ---------------------------------------------------------------------------


def test_funding_crowding_guard_normalizes_hyperliquid_1h_funding_to_8h():
    """Hyperliquid 1h funding (+200% APR = 0.000228/1h = 0.00182/8h) must trigger FundingCrowdingGuard."""
    from intelligence.external.collectors.derivatives_telemetry_collector import DerivativesTelemetry
    from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard

    guard = FundingCrowdingGuard(max_long_funding_rate=0.0005, crowding_policy="downscale", crowded_scale=0.5)
    # 1. With funding_annualized_pct populated (e.g. +200% APR on HYPEUSDC or HL fallback)
    hl_tel = DerivativesTelemetry(
        symbol="HYPEUSDC",
        funding_rate=0.000228,
        predicted_funding_rate=0.000228,
        open_interest=1000.0,
        open_interest_usd=60000.0,
        ts=time.time(),
        funding_annualized_pct=199.73,
    )
    dec_hl = guard._evaluate_external_snapshot("BUY", hl_tel)
    assert dec_hl.suggested_scale == pytest.approx(0.5)
    assert "long_crowding_extreme" in dec_hl.reason

    # 2. Without funding_annualized_pct (fallback via venue_for_symbol("HYPEUSDC") -> native L1 1h -> * 8)
    hl_tel_no_ann = DerivativesTelemetry(
        symbol="HYPEUSDC",
        funding_rate=0.000228,
        predicted_funding_rate=0.000228,
        open_interest=1000.0,
        open_interest_usd=60000.0,
        ts=time.time(),
        funding_annualized_pct=0.0,
    )
    dec_hl2 = guard._evaluate_external_snapshot("BUY", hl_tel_no_ann)
    assert dec_hl2.suggested_scale == pytest.approx(0.5)

    # 3. Standard Binance 8h funding at +0.02%/8h stays normal
    bn_tel = DerivativesTelemetry(
        symbol="BTCUSDC",
        funding_rate=0.0002,
        predicted_funding_rate=0.0002,
        open_interest=1000.0,
        open_interest_usd=60_000_000.0,
        ts=time.time(),
        funding_annualized_pct=21.9,
    )
    dec_bn = guard._evaluate_external_snapshot("BUY", bn_tel)
    assert dec_bn.allowed is True and dec_bn.suggested_scale == pytest.approx(1.0)


def test_clean_guard_symbol_key_and_pretrade_xstock_refresh(tmp_path, monkeypatch):
    """clean_guard_symbol_key separates SPCXxUSD vs SPCX_US_EQ; pretrade refresh finds mixed-case xStock files."""
    from cachepaths import CACHE_ORDERBOOK_DEPTH_PREFIX, resolve_cachedb_filepath
    from intelligence.internal.guards.guard_decision import clean_guard_symbol_key
    from intelligence.sentiment.collectors import pretrade_telemetry_collector as ptc

    assert clean_guard_symbol_key("SPCXxUSD") == "SPCXxUSD"
    assert clean_guard_symbol_key("SPCX_US_EQ") == "SPCX_US_EQ"
    assert clean_guard_symbol_key("ANTxUSD") == "ANTxUSD"
    assert clean_guard_symbol_key("NVDA_US_EQ") == "NVDA_US_EQ"
    assert clean_guard_symbol_key("hype/usdc") == "HYPEUSDC"

    cdir = tmp_path / "cachedb"
    ob_path = str(resolve_cachedb_filepath(str(cdir / f"{CACHE_ORDERBOOK_DEPTH_PREFIX}SPCXxUSD.json")))
    os.makedirs(os.path.dirname(ob_path), exist_ok=True)
    with open(ob_path, "w", encoding="utf-8") as fh:
        json.dump({"ts": time.time(), "symbol": "SPCXxUSD"}, fh)

    called_syms = []
    monkeypatch.setattr(
        ptc,
        "ensure_fresh_symbol_microstructure",
        lambda _cd, sym, **_kw: called_syms.append(sym) or True,
    )
    ok = ptc._ensure_fresh_pretrade_microstructure(
        str(cdir),
        "SPCXxUSD",
        ["SPCXxUSD"],
        True,
        True,
        ptc._default_read_json_cached,
        allow_network=None,
    )
    assert ok is True
    assert called_syms == ["SPCXxUSD"]


def test_llm_quota_router_filters_unrecognized_model_with_paid_credits(tmp_path, monkeypatch):
    """Unrecognized/dead models must be filtered out even when available_credits > 0."""
    from intelligence.sentiment.llm_quota_router import GeminiQuotaRouterMixin

    q_file = tmp_path / "llm_quota_status.json"
    q_file.write_text(
        json.dumps({"available_credits": 25.0, "buckets": {"gemini-4-dead": 1.0, "gemini-4-ok": 1.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(GeminiQuotaRouterMixin, "_get_quota_state_path", classmethod(lambda cls: str(q_file)))
    GeminiQuotaRouterMixin._record_model_quota_exhausted("gemini-4-dead", ttl_sec=3600.0, model_unrecognized=True)

    snap = json.loads(q_file.read_text(encoding="utf-8"))
    assert snap["available_credits"] == pytest.approx(25.0)
    filtered = GeminiQuotaRouterMixin.filter_models_with_confirmed_credits(
        ["gemini-4-dead", "gemini-4-ok"], snap
    )
    assert filtered == ["gemini-4-ok"]


def test_delta_gate_timeless_geo_shock_and_unmasked_trade_plan_invalidation():
    """Delta gate wakes once on timeless CRITICAL_SHOCK and preserves -xhigh on masked trade_plan_invalidated."""
    from orchestratorOS.supervisor.crypto_executor_delta_gate import (
        evaluate_crypto_executor_delta_gate,
        is_extreme_stakes_delta_gate_wakeup,
    )

    now_s = 1_795_500_000.0
    sensors_geo = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 60000.0}},
        portfolio_summary={},
        sentiment_advisor={},
        geopolitical_threat={"threat_level": "CRITICAL_SHOCK", "risk_score": 0.92},
        source_timestamps={},
    )
    prev_eval = {
        "ts": now_s - 120.0,
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 60000.0},
            "geo_ts": 0.0,
            "geo_verdict": "LOW",
        },
    }
    skip, reason, high_eff, meta = evaluate_crypto_executor_delta_gate(
        sensors_geo, prev_eval, now_ts=now_s, min_price_move_pct=1.0, max_idle_sec=1800.0
    )
    assert skip is False and reason == "geo_update" and high_eff is True
    # After acknowledging the same timeless CRITICAL_SHOCK verdict, quiet flat skips (no infinite loop)
    prev_ack = {"ts": now_s, "delta_gate_meta": meta}
    skip2, reason2, _, _ = evaluate_crypto_executor_delta_gate(
        sensors_geo, prev_ack, now_ts=now_s + 60.0, min_price_move_pct=1.0, max_idle_sec=1800.0
    )
    assert skip2 is True and reason2.startswith("quiet_flat")

    # Masked trade_plan_invalidated under focus_symbols still triggers extreme-stakes (-xhigh)
    sensors_tp = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 57000.0}},
        portfolio_summary={
            "tradable_positions": [{"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.1, "price": 57000.0, "value_usd": 5700.0}],
            "active_ai_trade_plans": [
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "invalidation_px": 58000.0,
                    "target_px": 64000.0,
                }
            ],
        },
        geopolitical_threat={},
        source_timestamps={},
    )
    prev_tp = {
        "ts": now_s - 60.0,
        "delta_gate_meta": {"eval_prices": {"BTCUSDC": 60000.0}, "plan_states": {"binance:BTCUSDC:BUY": "ACTIVE"}},
    }
    assert is_extreme_stakes_delta_gate_wakeup(
        "focus_symbols(BTCUSDC)", sensors=sensors_tp, prev_eval=prev_tp
    ) is True


def test_structural_7d_wave_evaluator_and_sibling_pred_funding_sync():
    """Structural7dWaveEvaluator calls compute_cross_btc_metrics cleanly; sibling sync copies pred_funding_rate."""
    from intelligence.sentiment.evaluators.microstructure_evaluator import reconcile_sibling_flow_and_liq_regimes
    from intelligence.sentiment.evaluators.price_series_evaluator import Structural7dWaveEvaluator

    base_ms = 1_795_500_000_000
    alt_s = [[base_ms + i * 300_000, 100.0 + i * 2.0] for i in range(6)]
    btc_s = [[base_ms + i * 300_000, 60000.0 + i * 200.0] for i in range(6)]
    res = Structural7dWaveEvaluator().evaluate(alt_s, symbol="ETHUSDC", btc_vals_7d=btc_s)
    assert isinstance(res, dict) and "alt_btc_dd_24h_high_pct" in res

    micro = {
        "BTCUSDC": {
            "oi_usd": 1_000_000_000.0,
            "funding_rate": 0.0001,
            "pred_funding_rate": 0.0002,
            "funding_annualized_pct": 10.95,
            "_deriv_ts": 1000.0,
        },
        "BTCUSDT": {
            "oi_usd": 1_010_000_000.0,
            "funding_rate": -0.0002,
            "pred_funding_rate": -0.0003,
            "funding_annualized_pct": -21.9,
            "_deriv_ts": 1000.0,
        },
    }
    reconcile_sibling_flow_and_liq_regimes(micro, {"BTCUSDC": {"price": 60000.0}, "BTCUSDT": {"price": 60000.0}})
    assert micro["BTCUSDT"]["funding_rate"] == micro["BTCUSDC"]["funding_rate"]
    assert micro["BTCUSDT"]["pred_funding_rate"] == micro["BTCUSDC"]["pred_funding_rate"]
