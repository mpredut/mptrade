"""Global safety and cleanup for the test suite."""

import os
import tempfile
import threading

import pytest


# Set at import time, before the test modules are collected. That way not even the calls
# made at import time, and subprocesses started by tests cannot send
# real notifications towards ntfy, email or the desktop.
os.environ["DISABLE_EXTERNAL_NOTIFICATIONS"] = "1"

# Some tests build the live engines with fake executors. Without a separate
# directory, the default audit ended up in logger/execution_audit and mixed
# TEST_US_EQ with the real fills used for calibration.
_execution_audit_tmp = tempfile.TemporaryDirectory(
    prefix="binance-tests-execution-audit-",
)
os.environ["EXECUTION_AUDIT_DIR"] = _execution_audit_tmp.name

_notification_state_tmp = tempfile.TemporaryDirectory(
    prefix="mptrade-tests-notification-state-",
)
os.environ["NOTIFICATION_STATE_FILE"] = os.path.join(
    _notification_state_tmp.name, "notification_delivery_state.json"
)


@pytest.fixture(autouse=True)
def _isolate_order_retry_queue(tmp_path, monkeypatch):
    """Keep every test away from the ignored operational retry outbox."""
    import order_retry

    monkeypatch.setattr(
        order_retry, "QUEUE_FILE", str(tmp_path / "order_retry_queue.jsonl"))
    monkeypatch.setattr(
        order_retry, "LOCK_FILE", str(tmp_path / "order_retry_queue.lock"))


@pytest.fixture(scope="session", autouse=True)
def _shutdown_runtime_threads_after_suite():
    yield
    import cacheManager as cm

    cm.CacheManagerInterface.shutdown_all_instances()
    cm.CachePriceShortTrendManager.shutdown_all_instances()
    cm.CacheFactory.shutdown_all()
    if cm._current_price_instance is not None:
        cm._current_price_instance.shutdown()
    if cm._short_trend_instance is not None:
        cm._short_trend_instance.shutdown()
    if cm._ws_bridge is not None:
        cm._ws_bridge.stop()

    from binance_api import bapi_client, bapi_ws
    assert bapi_client.stop_periodic_resync(), "BinanceTimeResync did not stop"
    bapi_ws.bapi_ws_manager.stop()

    forbidden = {
        "BinanceTimeResync", "CacheTradeManager", "CacheOrderManager",
        "CacheCurrentPriceManager", "NonBinanceTrendPoller",
        "InstantTrendFullEval", "InstantTrendFlush",
    }
    leaked = sorted(thread.name for thread in threading.enumerate()
                    if thread.name in forbidden)
    assert not leaked, f"runtime threads left behind after the suite: {leaked}"
    _execution_audit_tmp.cleanup()
