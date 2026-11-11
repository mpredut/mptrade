"""Regression tests for the shared Binance account-cache freshness marker."""

import json
import os
import unittest
from unittest.mock import patch

import account_cache_health as health
from tests.conftest import TempDirTestCase


class BinanceAccountCacheHealthTest(TempDirTestCase):
    NOW_MS = 1_900_000_000_000

    WRITER_STARTED_MS = NOW_MS - 5_000
    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp.name, "health.json")

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
            "BinanceOrder", "must-not-publish", now_ms=self.NOW_MS + 1))

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

        self._write(self._state(trade_sync_at_ms=self.NOW_MS - 11_000))
        trade_status = self._inspect()
        self.assertFalse(trade_status.ready)
        self.assertEqual(trade_status.reason, "trade_cache_stale")

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
            "BinanceOrder", "order-v2", now_ms=self.NOW_MS + 1_000
        ))
        order_only = health.inspect_health(
            path=self.path,
            now_ms=self.NOW_MS + 1_000,
            max_age_sec=10,
            pid_is_alive=lambda _pid: True,
            pid_started_at_ms=lambda _pid: self.WRITER_STARTED_MS,
        )
        self.assertFalse(order_only.ready)
        self.assertEqual(order_only.reason, "account_cache_initializing")

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
            "BinanceOrder", "order-v1", now_ms=self.NOW_MS + 1_000)
        health.record_successful_sync(
            "BinanceTrade", "trade-v1", now_ms=self.NOW_MS + 2_000)
        self.assertTrue(health.record_persisted_version(
            "BinanceOrder", "order-v2", now_ms=self.NOW_MS + 3_000))

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
            "BinanceOrder", "order-v1", now_ms=self.NOW_MS + 1_000)
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
            "BinanceOrder", "order-v2", now_ms=self.NOW_MS + 2_000))


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
        with unittest.mock.patch.object(health, "read_text_file", return_value=fake_stat), \
             unittest.mock.patch.object(health.os, "kill", return_value=None):
            self.assertFalse(health._pid_is_alive(12345))
            self.assertIsNone(health._pid_started_at_ms(12345))


class BinanceCacheHealthConfigTest(unittest.TestCase):
    def test_positive_finite_values_are_accepted(self):
        self.assertEqual(health.require_pos_float(1, "TEST_VALUE"), 1.0)
        self.assertEqual(health.require_pos_float("0.5", "TEST_VALUE"), 0.5)

    def test_zero_negative_nan_and_infinity_are_rejected(self):
        import math
        for value in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, "must be finite and positive"):
                    health.require_pos_float(value, "TEST_VALUE")


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
    assert health.record_successful_sync("BinanceOrder", True) is False
    assert health.record_successful_sync(True, "v1") is False
    assert health.record_persisted_version("BinanceTrade", True) is False
    assert health.record_persisted_version(True, "v1") is False

def test_account_cache_health__clean_version_str_rejects_sentinel_strings_iter151_s2():
    """[ITER151 #2] account_cache_health._clean_version_str rejects sentinel strings."""
    # 2. account_cache_health._clean_version_str rejects sentinel strings
    for bad_ver in ("", "   ", "nan", "null", "none", "inf", "-inf", "+inf", None, 123):
        assert health._clean_version_str(bad_ver) == ""
    assert health._clean_version_str(" v123 ") == "v123"

def test_account_cache_health_require_int_accepts_integer_strings_floats_and_rejects_iter160_s4():
    """[ITER160 #4] account_cache_health: require_int accepts integer strings/floats and rejects bool/negative."""
    import pytest
    # 4. account_cache_health: require_int accepts integer strings/floats and rejects bool/negative
    assert health.require_int("123.0", "pid", min_value=1) == 123
    assert health.require_int("0.0", "gen", min_value=0) == 0
    with pytest.raises(ValueError):
        health.require_int("123.5", "pid", min_value=1)
    with pytest.raises(ValueError):
        health.require_int(True, "pid", min_value=1)


def test_persistence_and_cache_layer_dag_and_sibling_purity_ast():
    """Enforce strict acyclic DAG imports across Layer 1/2 Persistence and Layer 3 Cache modules."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    def _imported_roots(rel_path: str) -> set[str]:
        tree = ast.parse((repo_root / rel_path).read_text(encoding="utf-8"), filename=rel_path)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                ) or (
                    isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__")
                ):
                    roots.add(node.args[0].value.split(".")[0])
        return roots

    assert not (repo_root / "binance_cache_health.py").exists()

    # Layer 2 persistence & window modules must never import Layer 3+ cache/provider/strategy/intelligence modules.
    forbidden_layer2 = {
        "account_cache_health",
        "binance_cache_health",
        "cacheInterface",
        "cacheaccount",
        "cacheprice",
        "cachetrend",
        "cacheManager",
        "providers",
        "instrument",
        "order_guard",
        "strategies",
        "intelligence",
        "shadow_signals",
        "binance_api",
        "kraken",
        "hyperliquid",
        "212trading",
    }
    for l2_mod in ("cachepaths.py", "state_io.py", "cacheintegrity.py", "pricewindow.py"):
        l2_imports = _imported_roots(l2_mod)
        assert not (l2_imports & forbidden_layer2), f"{l2_mod} upward leak: {l2_imports & forbidden_layer2}"
    ach_imports = _imported_roots("account_cache_health.py")
    assert not (ach_imports & (forbidden_layer2 - {"account_cache_health"})), (
        f"account_cache_health.py upward leak: {ach_imports & (forbidden_layer2 - {'account_cache_health'})}"
    )
    ss_imports = _imported_roots("shadow_signals.py")
    assert "intelligence" not in ss_imports, f"shadow_signals.py upward leak into intelligence: {ss_imports}"

    # All 5 cache modules must never import higher layers or venue SDK packages.
    forbidden_cache_upward = {
        "binance_cache_health",
        "instrument",
        "order_guard",
        "guard_hierarchy",
        "order_lifecycle",
        "order_retry",
        "strategies",
        "trailing_core",
        "intelligence",
        "shadow_signals",
        "orchestratorOS",
        "orchestratorTrade",
        "binance_api",
        "kraken",
        "hyperliquid",
        "212trading",
    }
    cache_modules = (
        "cacheInterface.py",
        "cacheaccount.py",
        "cacheprice.py",
        "cachetrend.py",
        "cacheManager.py",
    )
    for mod_name in cache_modules:
        imports = _imported_roots(mod_name)
        assert not (imports & forbidden_cache_upward), f"{mod_name} upward leak: {imports & forbidden_cache_upward}"

    # Sibling domain cache modules must remain mutually independent.
    siblings = ("cacheaccount", "cacheprice", "cachetrend")
    for sib in siblings:
        imports = _imported_roots(f"{sib}.py")
        other_siblings = set(siblings) - {sib}
        assert not (imports & other_siblings), f"{sib}.py imports sibling cache domain: {imports & other_siblings}"
        assert "cacheManager" not in imports, f"{sib}.py must not import cacheManager"

