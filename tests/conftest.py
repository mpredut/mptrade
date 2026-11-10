"""Global safety and cleanup for the test suite."""

import os
import shutil
import tempfile
import threading

import pytest


# Set at import time, before the test modules are collected. That way not even the calls
# made at import time, and subprocesses started by tests cannot send
# real notifications towards ntfy, email or the desktop.
os.environ["DISABLE_EXTERNAL_NOTIFICATIONS"] = "1"
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
for _bt_env_var in ("BACKTEST_MODE", "REPLAY_MODE", "DISABLE_GEMINI_LLM"):
    os.environ.pop(_bt_env_var, None)

import botcore as _botcore_init  # noqa: E402
import log as _log_init  # noqa: E402

_botcore_init.configure_runtime_services(
    stale_cache_retry_delay_sec=0.25,
    enforce_rth_gate=False,
    allow_live_network_backfill=False,
    allow_unmocked_llm_queries=False,
    allow_shared_disk_cache=False,
)
_log_init.set_suppress_default_folder_writes(True)

# Track all tempfile.mkdtemp() directories created during the test session so
# tests that omit explicit shutil.rmtree() never leak directories into /tmp.
_REAL_MKDTEMP = tempfile.mkdtemp
_TRACKED_MKDTEMP_DIRS: list[str] = []


def _tracked_mkdtemp(*args, **kwargs) -> str:
    path = _REAL_MKDTEMP(*args, **kwargs)
    _TRACKED_MKDTEMP_DIRS.append(path)
    return path


tempfile.mkdtemp = _tracked_mkdtemp

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

import order_lifecycle as _order_lifecycle_init  # noqa: E402,F401
from intelligence.external.collectors import bybit_liquidations_collector as _liq_col_init  # noqa: E402,F401
os.environ["NOTIFICATION_QUEUE_FILE"] = os.path.join(
    _notification_state_tmp.name, "notify_queue.json"
)

_cachedb_tmp = tempfile.TemporaryDirectory(
    prefix="mptrade-tests-cachedb-",
)
os.environ["MPTRADE_CACHEDB_DIR"] = _cachedb_tmp.name


@pytest.fixture(autouse=True)
def _isolate_cachedb_dir(tmp_path, monkeypatch):
    """Keep every test isolated from live/rsynced cachedb state files and reset injected runtime services."""
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path / "cachedb"))
    _botcore_init.configure_runtime_services(
        clock_fn=None,
        sleep_fn=None,
        stale_cache_retry_delay_sec=0.25,
        enforce_rth_gate=False,
        allow_live_network_backfill=False,
        allow_unmocked_llm_queries=False,
        allow_shared_disk_cache=False,
    )
    _log_init.set_suppress_default_folder_writes(True)


@pytest.fixture(autouse=True)
def _sink_smtp(monkeypatch):
    """Never reach a real SMTP server, even in tests that re-enable notifications."""
    from notify_engine import mailer

    sent = []
    monkeypatch.setattr(mailer, "_smtp_send", lambda msg, cfg: sent.append(msg))
    return sent


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
    import order_lifecycle as order_outcomes_log
    from intelligence.external.collectors import bybit_liquidations_collector as _liq_col  # noqa: F401

    monkeypatch.setattr(
        order_outcomes_log, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path / "order_outcomes"))


@pytest.fixture(autouse=True)
def _isolate_monitortrades_state(tmp_path, monkeypatch):
    """Keep every test away from the shared monitortrades state file (cachedb/mt_state.json)."""
    import monitortrades as mt

    monkeypatch.setattr(
        mt, "_MT_STATE_FILE", str(tmp_path / "mt_state.json"))


@pytest.fixture(scope="session", autouse=True)
def _shutdown_runtime_threads_after_suite():
    yield
    import cacheManager as cm

    cm.CacheManagerInterface.shutdown_all_instances()
    cm.CacheShortTrendManager.shutdown_all_instances()
    cm.CacheFactory.shutdown_all()
    if cm._ws_bridge is not None:
        cm._ws_bridge.stop()

    from binance_api import bapi_client, bapi_ws
    assert bapi_client.stop_periodic_resync(), "BinanceTimeResync did not stop"
    bapi_ws.bapi_ws_manager.stop()

    forbidden = {
        "BinanceTimeResync", "CacheTradeManager",
        "CacheOrderManager",
        "CacheCurrentPriceManager", "NonBinanceTrendPoller",
        "InstantTrendFullEval", "InstantTrendFlush",
        "ShortTrendFullEval", "ShortTrendFlush",
    }
    leaked = sorted(thread.name for thread in threading.enumerate()
                    if thread.name in forbidden)
    assert not leaked, f"runtime threads left behind after the suite: {leaked}"
    for path in _TRACKED_MKDTEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    _TRACKED_MKDTEMP_DIRS.clear()
    _execution_audit_tmp.cleanup()
    _notification_state_tmp.cleanup()
    _cachedb_tmp.cleanup()

