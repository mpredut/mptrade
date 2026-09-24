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


@pytest.fixture(autouse=True)
def _isolate_order_retry_queue(tmp_path, monkeypatch):
    """Keep every test away from the ignored operational retry outbox."""
    import order_retry

    monkeypatch.setattr(
        order_retry, "QUEUE_FILE", str(tmp_path / "order_retry_queue.jsonl"))
    monkeypatch.setattr(
        order_retry, "LOCK_FILE", str(tmp_path / "order_retry_queue.lock"))


@pytest.fixture(autouse=True)
def _isolate_order_outcomes_journal(tmp_path, monkeypatch):
    """Keep every test away from the live order journal (logger/order_outcomes_*.log).

    Several tests drive the real placement pipeline; without this their synthetic
    fills (CHARPIPEUSD, "BTCUSDC SELL 0.2 @ 100 accepted") landed in the journal that
    tradeall_observe.py reads for the production fleet.
    """
    import order_outcomes_log

    monkeypatch.setattr(
        order_outcomes_log, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path / "order_outcomes"))


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
