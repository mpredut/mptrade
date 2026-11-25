"""Regression tests for the shared Binance account-cache freshness marker."""

import account_cache_health as health
import ast
import json
import math
import os
import pytest
import unittest
from pathlib import Path
from tests.conftest import TempDirTestCase
from unittest.mock import patch


class BinanceAccountCacheHealthTest(TempDirTestCase):
    NOW_MS = 1_900_000_000_000

    WRITER_STARTED_MS = NOW_MS - 5_000
    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp, "health.json")

    def tearDown(self):
        health.disable_writer()
        super().tearDown()

    def _state(self, **overrides):
        state = {
            "schema_version": health.SCHEMA_VERSION,
            "writer_pid": 123,
            "writer_started_at_ms": self.WRITER_STARTED_MS,
            "published_at_ms": self.NOW_MS - 1_000,
            "generation": 2,
            "stopping": False,
            "order_cache_version": "order-v1",
            "trade_cache_version": "trade-v1",
            "order_sync_at_ms": self.NOW_MS - 1_000,
            "trade_sync_at_ms": self.NOW_MS - 2_000,
        }
        state.update(overrides)
        return state

    def _write(self, state):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
    def _enable_writer(self):
        with patch.object(
                health, "_pid_started_at_ms",
                return_value=self.WRITER_STARTED_MS):
            health.enable_writer(path=self.path, pid=123, now_ms=self.NOW_MS)


    def _inspect(self, **kwargs):
        return health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
            **kwargs,
        )

    def test_missing_and_malformed_markers_fail_closed(self):
        missing = self._inspect()
        self.assertFalse(missing.ready)
        self.assertEqual(missing.reason, "account_cache_health_missing")

        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("not-json")
        malformed = self._inspect()
        self.assertFalse(malformed.ready)
        self.assertEqual(malformed.reason, "account_cache_health_invalid")

    def test_schema_mismatch_fails_closed(self):
        self._write(self._state(schema_version=health.SCHEMA_VERSION + 1))
        status = self._inspect()
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_schema_mismatch")

    def test_initializing_marker_is_distinct_from_corrupt_state(self):
        self._write(self._state(
            generation=0,
            order_sync_at_ms=0,
            trade_sync_at_ms=0,
        ))
        status = self._inspect()
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_initializing")

    def test_dead_writer_fails_closed(self):
        self._write(self._state())
        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS,
            max_age_sec=10,
            pid_is_alive=lambda _pid: False,
            pid_started_at_ms=lambda _pid: None,
        )
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_writer_not_running")

    def test_alive_writer_without_start_identity_fails_closed(self):
        self._write(self._state())

        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: None,
        )

        self.assertFalse(status.ready)
        self.assertEqual(
            status.reason, "account_cache_writer_identity_unavailable")

    def test_enable_writer_refuses_unverifiable_linux_identity(self):
        with (
            patch.object(health, "_PROCESS_IDENTITY_REQUIRED", True),
            patch.object(health, "_pid_started_at_ms", return_value=None),
            self.assertRaises(health.AccountCacheNotReady) as raised,
        ):
            health.enable_writer(
                path=self.path, pid=123, now_ms=self.NOW_MS)

        self.assertEqual(
            raised.exception.reason,
            "account_cache_writer_identity_unavailable",
        )
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(health.record_persisted_version(
            "BinanceOrderFilled", "must-not-publish", now_ms=self.NOW_MS + 1))

    def test_dead_writer_takes_precedence_while_initializing(self):
        self._write(self._state(
            generation=0,
            order_sync_at_ms=0,
            trade_sync_at_ms=0,
        ))
        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS,
            max_age_sec=10,
            pid_is_alive=lambda _pid: False,
            pid_started_at_ms=lambda _pid: None,
        )
        self.assertEqual(status.reason, "account_cache_writer_not_running")

    def test_future_timestamp_fails_closed(self):
        self._write(self._state(order_sync_at_ms=self.NOW_MS + 6_000))
        status = self._inspect(future_tolerance_sec=5)
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_timestamp_in_future")

    def test_each_stale_cache_has_a_specific_reason(self):
        self._write(self._state(order_sync_at_ms=self.NOW_MS - 11_000))
        order_status = self._inspect()
        self.assertFalse(order_status.ready)
        self.assertEqual(order_status.reason, "order_cache_stale")

        # Stage-1 CacheOrderFilledManager SSoT: stale legacy trade cache is passive
        # observability and does not block CachePermit when filled-order cache is fresh.
        self._write(self._state(trade_sync_at_ms=self.NOW_MS - 11_000))
        trade_status = self._inspect()
        self.assertTrue(trade_status.ready)
        self.assertEqual(trade_status.reason, "")
        self.assertEqual(trade_status.order_age_sec, 1.0)
        self.assertEqual(trade_status.trade_age_sec, 11.0)

    def test_fresh_marker_reports_both_ages(self):
        self._write(self._state())
        status = self._inspect()
        self.assertTrue(status.ready)
        self.assertEqual(status.reason, "")
        self.assertEqual(status.order_age_sec, 1.0)
        self.assertEqual(status.trade_age_sec, 2.0)
        self.assertEqual(status.order_cache_version, "order-v1")
        self.assertEqual(status.trade_cache_version, "trade-v1")

    def test_writer_stays_initializing_until_both_caches_sync(self):
        self._enable_writer()
        initial = self._inspect()
        self.assertFalse(initial.ready)
        self.assertEqual(initial.reason, "account_cache_initializing")

        self.assertTrue(health.record_successful_sync(
            "BinanceOrderFilled", "order-v2", now_ms=self.NOW_MS + 1_000
        ))
        order_only = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS + 1_000,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
        )
        # Canonical BinanceOrderFilled alone satisfies readiness; legacy trade fields remain optional.
        self.assertTrue(order_only.ready)
        self.assertEqual(order_only.reason, "")
        self.assertEqual(order_only.order_age_sec, 0.0)
        self.assertIsNone(order_only.trade_age_sec)
        self.assertEqual(order_only.order_cache_version, "order-v2")
        self.assertEqual(order_only.trade_cache_version, "")

        with patch.object(
                health, "resolve_now_ms", side_effect=lambda v=None: (self.NOW_MS + 2_500) if v is None else v):
            self.assertTrue(health.record_successful_sync(
                "BinanceTrade", "trade-v2", now_ms=self.NOW_MS + 2_000
            ))
        with open(self.path, encoding="utf-8") as handle:
            state = json.load(handle)
        self.assertEqual(
            self.NOW_MS + 2_000, state["trade_sync_at_ms"])
        self.assertEqual(
            self.NOW_MS + 2_500, state["published_at_ms"])
        ready = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS + 2_000,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
        )
        self.assertTrue(ready.ready)
        self.assertEqual(ready.order_age_sec, 1.0)
        self.assertEqual(ready.trade_age_sec, 0.0)
        self.assertEqual(ready.order_cache_version, "order-v2")
        self.assertEqual(ready.trade_cache_version, "trade-v2")

        with open(self.path, "r", encoding="utf-8") as handle:
            serialized = handle.read()
        self.assertNotIn("\n", serialized)
        self.assertNotIn(": ", serialized)

    def test_version_only_publication_preserves_rest_sync_time(self):
        self._enable_writer()
        health.record_successful_sync(
            "BinanceOrderFilled", "order-v1", now_ms=self.NOW_MS + 1_000)
        health.record_successful_sync(
            "BinanceTrade", "trade-v1", now_ms=self.NOW_MS + 2_000)
        self.assertTrue(health.record_persisted_version(
            "BinanceOrderFilled", "order-v2", now_ms=self.NOW_MS + 3_000))

        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS + 3_000,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
        )
        self.assertTrue(status.ready)
        self.assertEqual(status.order_age_sec, 2.0)
        self.assertEqual(status.order_cache_version, "order-v2")

    def test_disable_writer_publishes_stopping_state(self):
        self._enable_writer()
        health.record_successful_sync(
            "BinanceOrderFilled", "order-v1", now_ms=self.NOW_MS + 1_000)
        health.record_successful_sync(
            "BinanceTrade", "trade-v1", now_ms=self.NOW_MS + 2_000)
        self.assertTrue(health.disable_writer(now_ms=self.NOW_MS + 3_000))

        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS + 3_000,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
        )
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_writer_stopping")

    def test_shutdown_write_failure_still_disables_future_publication(self):
        self._enable_writer()
        with patch.object(
                health, "_publish_locked",
                side_effect=OSError("simulated disk failure")):
            self.assertFalse(
                health.disable_writer(now_ms=self.NOW_MS + 1_000))
        self.assertFalse(health.record_persisted_version(
            "BinanceOrderFilled", "order-v2", now_ms=self.NOW_MS + 2_000))


    def test_writer_process_identity_change_fails_closed(self):
        self._write(self._state())
        status = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.NOW_MS - 60_000,
        )
        self.assertFalse(status.ready)
        self.assertEqual(
            status.reason, "account_cache_writer_identity_changed")

    def test_missing_cache_version_remains_initializing(self):
        self._write(self._state(order_cache_version=""))
        status = self._inspect()
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "account_cache_initializing")

    def test_unknown_cache_type_is_not_published(self):
        self._enable_writer()
        with open(self.path, "r", encoding="utf-8") as handle:
            before = handle.read()
        self.assertFalse(health.record_successful_sync(
            "CurrentPrice", "irrelevant", now_ms=self.NOW_MS + 1_000
        ))
        with open(self.path, "r", encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(after, before)

    def test_require_fresh_preserves_the_diagnostic_reason(self):
        with self.assertRaises(health.AccountCacheNotReady) as raised:
            health.require_fresh_account_cache(
                path=self.path,
                now_ms=self.NOW_MS,
                max_age_sec=10,
                pid_is_alive=lambda _pid: True,
                pid_started_at_ms=lambda _pid: None,
            )
        self.assertEqual(raised.exception.reason,
                         "account_cache_health_missing")

    def test_zombie_pid_is_treated_as_dead_writer(self):
        fake_stat = "12345 (cacheManager) Z 1 12345 12345 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 987654 0 0"
        with patch.object(health, "read_text_file", return_value=fake_stat), \
             patch.object(health.os, "kill", return_value=None):
            self.assertFalse(health._pid_is_alive(12345))
            self.assertIsNone(health._pid_started_at_ms(12345))


class BinanceCacheHealthConfigTest(unittest.TestCase):
    def test_positive_finite_values_are_accepted(self):
        self.assertEqual(health.require_pos_float(1, "TEST_VALUE"), 1.0)
        self.assertEqual(health.require_pos_float("0.5", "TEST_VALUE"), 0.5)

    def test_zero_negative_nan_and_infinity_are_rejected(self):
        for value in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, "must be finite and positive"):
                    health.require_pos_float(value, "TEST_VALUE")


# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_account_cache_health__clean_now_ms_ignores_bool_nan_iter38_s3():
    """[ITER38 #3] account_cache_health.resolve_now_ms ignores bool/NaN."""
    # 3. account_cache_health.resolve_now_ms ignores bool/NaN
    ts_clean = health.resolve_now_ms(True)
    assert ts_clean > 1_000_000_000_000
    assert health.resolve_now_ms(float("nan")) > 1_000_000_000_000

def test_account_cache_health_accountcachenotready_record_successful_sync_record_persisted_version_iter107_s1():
    """[ITER107 #1] account_cache_health: AccountCacheNotReady, record_successful_sync, record_persisted_version."""
    # 1. account_cache_health: AccountCacheNotReady, record_successful_sync, record_persisted_version
    err = health.AccountCacheNotReady(True)
    assert err.reason == "account_cache_not_fresh"
    assert health.record_successful_sync("BinanceOrderFilled", True) is False
    assert health.record_successful_sync(True, "v1") is False
    assert health.record_persisted_version("BinanceTrade", True) is False
    assert health.record_persisted_version(True, "v1") is False

def test_account_cache_health__clean_version_str_rejects_sentinel_strings_iter151_s2():
    """[ITER151 #2] account_cache_health.clean_scalar_str rejects sentinel strings."""
    # 2. account_cache_health.clean_scalar_str rejects sentinel strings
    for bad_ver in ("", "   ", "nan", "null", "none", "inf", "-inf", "+inf", None, 123):
        assert (health.clean_scalar_str(bad_ver) or "") == ""
    assert health.clean_scalar_str(" v123 ") == "v123"

def test_account_cache_health_require_int_accepts_integer_strings_floats_and_rejects_iter160_s4():
    """[ITER160 #4] account_cache_health: require_int accepts integer strings/floats and rejects bool/negative."""
    # 4. account_cache_health: require_int accepts integer strings/floats and rejects bool/negative
    assert health.require_int("123.0", "pid", min_value=1) == 123
    assert health.require_int("0.0", "gen", min_value=0) == 0
    with pytest.raises(ValueError):
        health.require_int("123.5", "pid", min_value=1)
    with pytest.raises(ValueError):
        health.require_int(True, "pid", min_value=1)


def test_persistence_and_cache_layer_dag_and_sibling_purity_ast():
    """Enforce strict acyclic DAG imports across Layer 1/2 Persistence and Layer 3 Cache modules."""
    from verify_tools.ssot_ast_audit import audit_persistence_and_cache_dag_ast

    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "binance_cache_health.py").exists()
    violations = audit_persistence_and_cache_dag_ast(repo_root)
    assert violations == [], f"Persistence/cache layer DAG violation(s): {violations}"


def test_persisted_sync_state_ignores_retired_symbols_and_requires_all_active_symbols():
    """Verify retired symbol cursors (e.g. BNBUSDC) do not block startup seeding while missing active symbols return 0."""
    from cacheaccount import CacheOrderFilledManager

    mgr = object.__new__(CacheOrderFilledManager)
    mgr.symbols = ["BTCUSDC", "TAOUSDC", "ARBUSDC"]
    mgr._persisted_data_version = "snapshot:live_v1"
    mgr._loaded_data_version = ""
    mgr._rest_sync_cursor_per_symbol = {
        "BTCUSDC": 1_795_173_456_830,
        "TAOUSDC": 1_795_173_457_096,
        "ARBUSDC": 1_795_173_457_360,
        "BNBUSDC": 1_795_009_281_722,  # retired 45h-old symbol in .meta
    }
    mgr.fetchtime_per_symbol = {}
    ver, sync_ms = mgr.get_persisted_sync_state()
    assert ver == "snapshot:live_v1"
    assert sync_ms == 1_795_173_456_830

    # Newly added active symbol without cursor must not claim fresh sync
    mgr.symbols = ["BTCUSDC", "TAOUSDC", "ARBUSDC", "SOLUSDC"]
    assert mgr.get_persisted_sync_state() == ("snapshot:live_v1", 0)


def test_multi_venue_account_cache_health_etapa3(tmp_path):
    """Etapa 3: cache_account_health tracks binance/kraken/hyperliquid/t212 across all streams with fail-closed checks."""
    marker = str(tmp_path / "cache_account_health.json")
    now_ms = 1_900_000_000_000
    started_ms = now_ms - 5_000

    # Writer disabled => record_venue_sync is a strict no-op and never touches disk.
    health.disable_writer()
    assert health.record_venue_sync("kraken", "balances", now_ms=now_ms) is False
    assert not os.path.exists(marker)

    with patch.object(health, "_pid_started_at_ms", return_value=started_ms):
        health.enable_writer(path=marker, pid=4321, now_ms=now_ms)
    try:
        # Unsupported venue / stream rejected without mutating state.
        assert health.record_venue_sync("unknown_exchange", "balances", now_ms=now_ms) is False
        assert health.record_venue_sync("kraken", "invalid_stream", now_ms=now_ms) is False

        # Binance canonical record_successful_sync mirrors into venues.binance.
        assert health.record_successful_sync("BinanceOrderFilled", "bn-ord-v1", now_ms=now_ms - 1_000)
        assert health.record_successful_sync("BinanceTrade", "bn-trd-v1", now_ms=now_ms - 2_000)
        assert health.record_venue_sync("binance", "open_orders", now_ms=now_ms - 500)
        assert health.record_venue_sync("binance", "balances", now_ms=now_ms - 1_500)

        # Record all 3 other venues (including alias normalization for trading212 / hl).
        for venue_arg, ts_offset in (("kraken", 1_000), ("hl", 2_000), ("trading212", 3_000)):
            assert health.record_venue_sync(venue_arg, "filled_orders", now_ms=now_ms - ts_offset)
            assert health.record_venue_sync(venue_arg, "open_orders", now_ms=now_ms - ts_offset)
            assert health.record_venue_sync(venue_arg, "balances", now_ms=now_ms - ts_offset)
            assert health.record_venue_sync(venue_arg, "trades", now_ms=now_ms - ts_offset)

        # Monotonic guard: older timestamp cannot regress a newer venue sync timestamp.
        assert health.record_venue_sync("kraken", "balances", now_ms=now_ms - 60_000)

        # Classic Binance inspect_health stays 100% backward compatible.
        bn_legacy = health.inspect_health(
            path=marker,
            now_ms=now_ms,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: started_ms,
        )
        assert bn_legacy.ready is True
        assert bn_legacy.order_age_sec == 1.0
        assert bn_legacy.trade_age_sec == 2.0
        assert bn_legacy.order_cache_version == "bn-ord-v1"

        all_status = health.inspect_all_venues_health(
            path=marker,
            now_ms=now_ms,
            max_age_sec=10,
            required_streams=("filled_orders", "open_orders", "balances"),
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: started_ms,
        )
        assert set(all_status.keys()) == {"binance", "kraken", "hyperliquid", "t212"}
        for v_key, st in all_status.items():
            assert st.ready is True, (v_key, st)
            assert st.reason == ""
        assert all_status["binance"].filled_orders_age_sec == 1.0
        assert all_status["binance"].open_orders_age_sec == 0.5
        assert all_status["binance"].balances_age_sec == 1.5
        assert all_status["binance"].filled_orders_version == "bn-ord-v1"
        assert all_status["kraken"].balances_age_sec == 1.0  # preserved newer ts despite -60s call
        assert all_status["hyperliquid"].open_orders_age_sec == 2.0
        assert all_status["t212"].filled_orders_age_sec == 3.0

        # Fail-closed on stale stream for a specific venue.
        stale_t212 = health.inspect_venue_health(
            "t212",
            path=marker,
            now_ms=now_ms + 10_000,
            max_age_sec=10,
            required_streams=("filled_orders", "open_orders"),
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: started_ms,
        )
        assert stale_t212.ready is False
        assert stale_t212.reason == "t212_filled_orders_stale"

        # Fail-closed on dead writer PID.
        dead_kr = health.inspect_venue_health(
            "kraken",
            path=marker,
            now_ms=now_ms,
            max_age_sec=10,
            pid_is_alive=lambda _pid: False,
            pid_started_at_ms=lambda _pid: None,
        )
        assert dead_kr.ready is False
        assert dead_kr.reason == "account_cache_writer_not_running"

        # Unknown venue fails closed cleanly.
        unk = health.inspect_venue_health("bitfinex", path=marker, now_ms=now_ms)
        assert unk.ready is False
        assert unk.reason == "account_cache_unknown_venue"
    finally:
        health.disable_writer(now_ms=now_ms + 1_000)

    # Stopping state fails closed for every venue.
    stopped = health.inspect_all_venues_health(
        path=marker,
        now_ms=now_ms + 1_000,
        max_age_sec=10,
        pid_is_alive=lambda _pid: True,
        pid_started_at_ms=lambda _pid: started_ms,
    )
    assert all((not st.ready and st.reason == "account_cache_writer_stopping") for st in stopped.values())

    # Legacy Binance-only marker (no "venues" dict) still resolves binance filled_orders & trades.
    legacy_path = str(tmp_path / "legacy_health.json")
    with open(legacy_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": health.SCHEMA_VERSION,
                "writer_pid": 4321,
                "writer_started_at_ms": started_ms,
                "published_at_ms": now_ms - 500,
                "generation": 2,
                "stopping": False,
                "order_sync_at_ms": now_ms - 1_000,
                "trade_sync_at_ms": now_ms - 2_000,
                "order_cache_version": "legacy-ord-v1",
                "trade_cache_version": "legacy-trd-v1",
            },
            fh,
        )
    bn_from_legacy = health.inspect_venue_health(
        "binance",
        path=legacy_path,
        now_ms=now_ms,
        max_age_sec=10,
        required_streams=("filled_orders",),
        pid_is_alive=lambda _pid: True,
        pid_started_at_ms=lambda _pid: started_ms,
    )
    assert bn_from_legacy.ready is True
    assert bn_from_legacy.filled_orders_age_sec == 1.0
    assert bn_from_legacy.trades_age_sec == 2.0
    assert bn_from_legacy.filled_orders_version == "legacy-ord-v1"
    kr_from_legacy = health.inspect_venue_health(
        "kraken",
        path=legacy_path,
        now_ms=now_ms,
        max_age_sec=10,
        required_streams=("open_orders",),
        pid_is_alive=lambda _pid: True,
        pid_started_at_ms=lambda _pid: started_ms,
    )
    assert kr_from_legacy.ready is False
    assert kr_from_legacy.reason == "kraken_open_orders_initializing"


def test_cacheaccount_bridges_publish_multi_venue_health_etapa3(tmp_path, monkeypatch):
    """Verify cacheaccount bridges (_VenueBalancesBridge & CacheOrderOpenManager) call record_venue_sync."""
    import cacheaccount as ca

    recorded: list[tuple[str, str, int | None]] = []
    monkeypatch.setattr(
        ca.account_cache_health,
        "record_venue_sync",
        lambda venue, stream, *, now_ms=None, data_version="": (
            recorded.append((venue, stream, now_ms)) or True
        ),
    )

    class _FakeProv:
        def all_free_and_locked_balances(self, assets):
            return {"USDC": (100.0, 0.0)}

    monkeypatch.setattr(
        ca.MarketDataProvider,
        "resolve_venue_provider_class",
        classmethod(lambda cls, v: _FakeProv),
    )
    monkeypatch.setattr(
        ca._VenueBalancesBridge,
        "_venue_requested_assets",
        staticmethod(lambda v: ["USDC"]),
    )
    monkeypatch.setattr(ca, "cache_path", lambda fname: str(tmp_path / fname))

    bridge = ca._VenueBalancesBridge()
    assert bridge.sync_venue_once("kraken", now_s=1_700_000_000.5) is True
    assert ("kraken", "balances", 1_700_000_000_500) in recorded

    mgr = ca.CacheOrderOpenManager(
        sync_ts=60,
        symbols=["BTCUSDC"],
        filename=str(tmp_path / "cache_order_open_binance.json"),
        read_only=False,
        venue="binance",
    )
    mgr._on_complete_remote_sync_cycle(1_700_000_001_234)
    mgr._publish_periodic_sync_health(sync_complete=True, persisted=True)
    assert ("binance", "open_orders", 1_700_000_001_234) in recorded


def test_etapa3b_multi_venue_health_snapshot_coherence_and_fleet_inspection(tmp_path, monkeypatch):
    """Etapa 3B: remote snapshot evaluation, check_cache_coherence RTH-aware venue checks, and inspect_prod_fleet."""
    from verify_tools import check_cache_coherence as ccc
    from verify_tools import inspect_prod_fleet

    now_s = 1_700_258_400.0  # Friday 22:00 UTC (outside US RTH)
    now_ms = round(now_s * 1000)
    started_ms = now_ms - 60_000

    # 1. Pure snapshot evaluator (no local PID checks): respects required_streams_by_venue & stopping
    snapshot = {
        "schema_version": health.SCHEMA_VERSION,
        "writer_pid": 999999,
        "writer_started_at_ms": started_ms,
        "published_at_ms": now_ms - 1_000,
        "generation": 5,
        "stopping": False,
        "order_sync_at_ms": now_ms - 5_000,
        "order_cache_version": "bn-v1",
        "venues": {
            "binance": {
                "filled_orders_sync_at_ms": now_ms - 5_000,
                "open_orders_sync_at_ms": now_ms - 4_000,
                "balances_sync_at_ms": now_ms - 6_000,
                "trades_sync_at_ms": now_ms - 8_000,
            },
            "kraken": {
                "filled_orders_sync_at_ms": now_ms - 5_000,
                "open_orders_sync_at_ms": now_ms - 4_000,
                "balances_sync_at_ms": now_ms - 6_000,
            },
            "hyperliquid": {
                "filled_orders_sync_at_ms": now_ms - 5_000,
                "open_orders_sync_at_ms": now_ms - 4_000,
                "balances_sync_at_ms": now_ms - 6_000,
            },
            "t212": {
                "filled_orders_sync_at_ms": now_ms - 10_000,
                "open_orders_sync_at_ms": now_ms - 10_000,
                "balances_sync_at_ms": now_ms - 7_200_000,  # 2h stale after US close
            },
        },
    }
    snap_res = health.inspect_all_venues_health_snapshot(
        snapshot,
        now_ms=now_ms,
        max_age_sec=600,
        required_streams_by_venue={
            "binance": ("filled_orders", "open_orders", "balances"),
            "kraken": ("filled_orders", "open_orders", "balances"),
            "hyperliquid": ("filled_orders", "open_orders", "balances"),
            "t212": ("filled_orders", "open_orders"),
        },
    )
    assert all(st.ready for st in snap_res.values()), snap_res
    assert snap_res["t212"].balances_age_sec == pytest.approx(7200.0)

    stopping_res = health.inspect_all_venues_health_snapshot(
        {**snapshot, "stopping": True},
        now_ms=now_ms,
        max_age_sec=600,
    )
    assert all((not st.ready and st.reason == "account_cache_writer_stopping") for st in stopping_res.values())

    # 2. check_cache_coherence validates account/cache_account_health.json across all 4 venues (RTH-aware for t212)
    coh_dir = tmp_path / "coh"
    (coh_dir / "account").mkdir(parents=True)
    (coh_dir / "cache_short_trend.json").write_text(
        json.dumps({"items": {"BTCUSDC": {"ts": now_s - 2.0, "current_price": 95000.0, "gradient_recent": 0.01}}}),
        encoding="utf-8",
    )
    (coh_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[(now_s - 1.0) * 1000.0, 95000.0]]}}),
        encoding="utf-8",
    )
    health_file = coh_dir / "account" / "cache_account_health.json"
    health_file.write_text(json.dumps(snapshot), encoding="utf-8")

    monkeypatch.setattr(ccc, "CACHEDB", str(coh_dir))
    monkeypatch.setattr(ccc, "expected_symbols", lambda: ["BTCUSDC"])
    monkeypatch.setattr(ccc.time, "time", lambda: now_s)
    monkeypatch.setattr(health, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(health, "_pid_started_at_ms", lambda _pid: started_ms)
    # Outside US RTH: stale t212 balances does not fail coherence while filled/open are fresh.
    assert ccc.main() == 0

    # Stale kraken balances fails coherence immediately.
    stale_kr = json.loads(json.dumps(snapshot))
    stale_kr["venues"]["kraken"]["balances_sync_at_ms"] = now_ms - 900_000
    health_file.write_text(json.dumps(stale_kr), encoding="utf-8")
    assert ccc.main() == 1

    # 3. inspect_prod_fleet exposes account_cache_health from remote SSH snapshot without local PID dependency
    monkeypatch.setattr(inspect_prod_fleet.time, "time", lambda: now_s)
    monkeypatch.setattr(
        inspect_prod_fleet,
        "ssh_cat",
        lambda p: snapshot if "cache_account_health.json" in str(p) else {},
    )
    fleet_res = inspect_prod_fleet.inspect_fleet(json_output=True)
    assert "account_cache_health" in fleet_res
    assert set(fleet_res["account_cache_health"].keys()) == {"binance", "kraken", "hyperliquid", "t212"}
    assert all(v["ready"] is True for v in fleet_res["account_cache_health"].values())
    assert fleet_res["account_cache_health"]["binance"]["filled_orders_age_sec"] == pytest.approx(5.0)
    assert fleet_res["account_cache_health"]["t212"]["balances_age_sec"] == pytest.approx(7200.0)



