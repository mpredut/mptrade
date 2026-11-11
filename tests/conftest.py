"""Global safety and cleanup for the test suite."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

import pytest
import cacheManager as cm


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

    monkeypatch.setattr(
        _order_lifecycle_init, "ORDER_OUTCOMES_LOG_DIR", str(tmp_path / "order_outcomes"))


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Shared temporary cachedb directory fixture."""
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def setup_isolated_agy_stores(tmp_path, monkeypatch, *, autonomous_project_uuid: str | None = None) -> None:
    """Isolate Antigravity Web UI conversation store and topic registry inside ``tmp_path``."""
    ui_base = tmp_path / "ui_home" / ".gemini" / "antigravity-cli"
    (ui_base / "conversations").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(ui_base))
    monkeypatch.setenv("LLM_TOPIC_REGISTRY_PATH", str(tmp_path / "llm_topic_threads.json"))
    if autonomous_project_uuid is not None:
        projects = tmp_path / "ui_home" / ".gemini" / "config" / "projects"
        projects.mkdir(parents=True, exist_ok=True)
        (projects / "autonom.json").write_text(
            f'{{"id": "{autonomous_project_uuid}", "name": "autonommptrade"}}', encoding="utf-8",
        )
        (projects / "mptrade.json").write_text('{"id": "3ed4aa5d-mptrade", "name": "mptrade"}', encoding="utf-8")


T212_TEST_BASE_ENV: dict[str, str] = {
    "STRAT_CURRENCY": "USD",
    "YAHOO_SYMBOL": "TEST",
    "STRATEGY_MODE": "avg_tp",
    "STRAT_ENTRY_PCT": "20",
    "STRAT_DCA_PCT": "10",
    "STRAT_ENTRY_DISCOUNT_PCT": "0.2",
    "STRAT_DCA_DROP_PCT": "2",
    "STRAT_TAKEPROFIT_PCT": "3",
    "STRAT_MAX_DCA_BUYS": "3",
    "STRAT_MAX_BUDGET": "500",
    "STRAT_FX_FEE_PCT": "0.15",
    "STRAT_STOP_LOSS_PCT": "20",
    "STRAT_CHECK_MINUTES": "5",
    "STRAT_ORDER_TTL_MIN": "10",
    "STRAT_REENTRY_DROP_PCT": "0",
    "STRAT_REENTRY_TOLERANCE_PCT": "0",
    "STRAT_LOSS_ALERT_STEP": "1",
    "STRAT_LADDER_MIN_FREE": "6",
    "STRAT_SL_REBUY_ENABLED": "false",
    "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
    "STRAT_DCA_TREND_GATE_PCT": "0",
    "STRAT_TRAIL_PCT": "0",
    "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
    "STRAT_REENTRY_HYBRID_ENABLED": "false",
    "STRAT_REENTRY_PEAK_RELATIVE": "false",
    "STRAT_REENTRY_PULLBACK_PCT": "1.5",
    "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
    "STRAT_REENTRY_PULLBACK_K": "1.0",
    "STRAT_REENTRY_PULLBACK_MIN": "0.8",
    "STRAT_REENTRY_PULLBACK_MAX": "4.0",
    "STRAT_REENTRY_TTL_HOURS": "0.0",
    "STRAT_SL_MHDSL_ENABLED": "false",
    "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
    "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
    "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
    "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
    "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
    "STRAT_TP_MDHTP_ENABLED": "false",
    "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
}


def make_t212_test_env(**overrides) -> dict[str, str]:
    """Build a standard Trading 212 test environment dictionary with stringified overrides."""
    cfg = dict(T212_TEST_BASE_ENV)
    cfg.update({k: str(v) for k, v in overrides.items()})
    return cfg


def make_spot_test_params(**overrides):
    """Build a standard ``strategies.spot_engine.StratParams`` test instance with overrides."""
    from strategies import spot_engine as strat

    values = dict(
        currency="USD", entry_amount=100.0, entry_discount_pct=0.2,
        dca_amount=50.0, dca_drop_pct=2.0, check_minutes=2.0,
        takeprofit_pct=3.0, max_budget=500.0, max_dca_buys=3,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=0.0,
        reentry_tolerance_pct=0.05, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[],
    )
    values.update(overrides)
    return strat.StratParams(**values)


def make_test_regime_context(symbol: str = "BTCUSDC", provider: str = "binance", *, regime: str = "bull"):
    """Build a fresh mock ``MarketRegimeContext`` for guard and instrument tests."""
    from market_regime import MarketRegimeContext, MarketRegimeDecision

    decision = MarketRegimeDecision(
        regime=regime, gradient=0.5, epsilon=0.1, strength=5.0,
        fresh=True, reason="signal", source="mock",
    )
    return MarketRegimeContext.from_decision(decision, symbol=symbol, provider=provider)


def make_cache24_test_manager(symbol: str, entries: list, tmp_dir: str, *, api_client=None):
    """Create a ``Cache24PriceManager`` pre-populated with ``entries = [[ts_ms, price], ...]``."""

    fname = os.path.join(tmp_dir, f"cache_24_{symbol}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump({"items": {symbol: entries}, "fetchtime": {}}, fh)
    return cm.Cache24PriceManager(
        sync_ts=9999,
        symbols=[symbol],
        filename=fname,
        api_client=api_client if api_client is not None else MagicMock(),
    )


def make_synthetic_price_entries(
    n: int = 100,
    start_price: float = 60000.0,
    delta: float = 10.0,
    interval_ms: int = 800,
    start_ts_ms: int | None = None,
) -> list[list]:
    """Generate ``n`` ``[ts_ms, price]`` entries with a constant interval and price delta."""
    if start_ts_ms is None:
        start_ts_ms = int(time.time() * 1000) - n * interval_ms
    return [[start_ts_ms + i * interval_ms, start_price + i * delta] for i in range(n)]


_VENUE_ISOLATED_MODULES = (
    "common",
    "notify",
    "hl_client",
    "ipo_common",
    "ipo_notify",
    "t212_client",
    "market_data",
    "price_analysis",
    "signals",
    "strategy",
    "replay",
    "backtest",
    "order_manager",
)


def load_venue_modules(venue_subdir: str, *module_names: str):
    """Import ``module_names`` from ``<repo_root>/<venue_subdir>`` while preserving ``sys.path`` and ``sys.modules``."""
    venue_dir = os.path.join(_botcore_init._REPO_ROOT, venue_subdir)
    orig_sys_path = list(sys.path)
    clear_keys = set(_VENUE_ISOLATED_MODULES) | set(module_names)
    saved_mods = {k: sys.modules.get(k) for k in clear_keys}
    try:
        for k in clear_keys:
            sys.modules.pop(k, None)
        sys.path.insert(0, venue_dir)
        loaded = tuple(importlib.import_module(name) for name in module_names)
        return loaded[0] if len(loaded) == 1 else loaded
    finally:
        sys.path[:] = orig_sys_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class SingleProviderApi:
    """Minimal ``api`` test stub exposing ``provider_by_name`` for ``Instrument`` and registry tests."""

    def __init__(self, provider):
        self.provider = provider

    def provider_by_name(self, name):
        if str(name or "").casefold() == str(self.provider.name).casefold():
            return self.provider
        return None


class DummySpotExecutionProvider:
    """Minimal spot execution provider stub for ``Instrument.place`` and ``MarketApi.place`` tests."""

    name = "kraken"

    def __init__(
        self,
        *,
        name: str | None = None,
        symbol: str | None = None,
        order_id: str = "ord-1",
        free_bal: float = 10000.0,
        wait_short_trend: bool = False,
    ):
        if name is not None:
            self.name = name
        self._symbol = symbol
        self._order_id = order_id
        self._free_bal = free_bal
        self._wait_short_trend = wait_short_trend

    def supports_symbol(self, sym):
        return self._symbol is None or sym == self._symbol

    def validate_symbol(self, sym):
        return None

    def guards_internally(self):
        return False

    def requires_cache_permit(self):
        return False

    def defers_smart_cancel(self):
        return False

    def defers_smart_cancellation(self):
        return False

    def supports_business_minimum(self):
        return False

    def execution_enabled(self):
        return True

    def preflight_order(self, *a, **k):
        return None

    def round_price(self, sym, px):
        return px

    def round_amount(self, sym, q):
        return round(float(q), 4)

    def format_price(self, _sym, p):
        return float(p)

    def format_quantity(self, _sym, q):
        return float(q)

    def validate_order(self, _sym, _side, _px, _q):
        return None

    def min_notional(self, _sym):
        return 5.0

    def get_orders(self, _sym, _side, _since):
        return []

    def should_wait_for_short_trend(self, _sym, _side):
        return self._wait_short_trend

    def free_balance(self, asset):
        return self._free_bal

    def get_current_price(self, sym):
        return 100.0

    def profit_guard_window_ref(self, *a, **k):
        return None

    def quantity_decision(self, symbol, side, price, qty, **kw):
        from providers.quantity import QuantityDecision

        rc = kw.get("regime_context")
        sc = getattr(rc, "suggested_scale", 1.0) if (rc is not None and side == "BUY") else 1.0
        final_q = float(qty or 1.0) * (sc if 0 < sc < 1.0 else 1.0)
        return QuantityDecision(
            requested_qty=float(qty or 1.0),
            balance_cap=100.0,
            policy_cap=100.0,
            fee_cap=100.0,
            final_qty=final_q,
            scale_applied=(0 < sc < 1.0),
            applied_scale=sc,
        )

    def place_order(self, symbol, side, price, qty, **kw):
        return {"orderId": self._order_id, "qty": qty}


def make_transient_complete_claim(order_retry_mod):
    """Return ``(attempts, transient_complete_fn)`` that raises ``OSError`` on the first call and delegates thereafter."""
    real_complete = order_retry_mod.complete_claim
    attempts = []

    def transient_complete(*args, **kwargs):
        attempts.append((args, kwargs))
        if len(attempts) == 1:
            raise OSError("transient fsync failure")
        return real_complete(*args, **kwargs)

    return attempts, transient_complete


class MockHttpResponse:
    """Reusable mock HTTP response (``urllib.request.urlopen`` or ``requests``) supporting ``bytes``, ``str``, or JSON payloads."""

    def __init__(self, payload: bytes | str | dict | list = b"{}", *, status: int = 200, headers: dict | None = None):
        self._raw_payload = payload
        if isinstance(payload, bytes):
            self._body = payload
        elif isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = dict(headers) if headers is not None else {}

    def read(self, _max_bytes: int | None = None) -> bytes:
        return self._body

    def json(self):
        if isinstance(self._raw_payload, (dict, list)):
            return self._raw_payload
        return json.loads(self._body.decode("utf-8"))

    def raise_for_status(self) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DummyTrailingAdapter:
    """Minimal ``TrailingCore`` adapter stub with configurable ``assets_list``, ``free``, ``px``, and ``trend_val``."""

    def __init__(
        self,
        *,
        assets_list=None,
        free=1.0,
        px=100.0,
        trend_val: float = 0.0,
    ):
        self._assets = list(assets_list) if assets_list is not None else [("BTC", "BTC", "BTCUSDC", 15.0)]
        self.free = free
        self.px = px
        self.trend_val = trend_val

    def assets(self):
        return list(self._assets)

    def begin_tick(self):
        return True

    def free_qty(self, _asset):
        return self.free

    def total_qty(self, _asset):
        return self.free

    def price(self, _pair):
        return self.px

    def trend(self, _pair):
        return self.trend_val

    def log_hold(self, *_a):
        pass


class _TempDirPath(str):
    """Temporary directory path string that also exposes ``.name`` and ``.cleanup()`` like ``TemporaryDirectory``."""

    @property
    def name(self) -> str:
        return str(self)

    def cleanup(self) -> None:
        shutil.rmtree(str(self), ignore_errors=True)


class TempDirTestCase(unittest.TestCase):
    """Shared ``unittest.TestCase`` base class that provisions ``self.tmp = _TempDirPath(tempfile.mkdtemp())`` in ``setUp``."""

    def setUp(self):
        self.tmp = _TempDirPath(tempfile.mkdtemp())
        self._temp_dir = self.tmp
        self._tmp = self.tmp
        self.temp = self.tmp
        self.tempdir = self.tmp

    def tearDown(self):
        self.tmp.cleanup()


class TrailingTestBase(unittest.TestCase):
    """Shared base class for trailing-stop unit tests with isolated temp state file and TRAIL_PCT restore."""

    TRAIL_MODULE = None
    TRAIL_DEFAULTS: dict[str, float] = {}

    def setUp(self):
        fd, self.sf = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.sf)
        if self.TRAIL_MODULE is not None:
            self._orig_trail_pct = dict(self.TRAIL_MODULE.TRAIL_PCT)
            self.TRAIL_MODULE.TRAIL_PCT.update(self.TRAIL_DEFAULTS)

    def tearDown(self):
        if self.TRAIL_MODULE is not None:
            self.TRAIL_MODULE.TRAIL_PCT.clear()
            self.TRAIL_MODULE.TRAIL_PCT.update(self._orig_trail_pct)
        for p in (self.sf, self.sf + ".tmp"):
            if os.path.exists(p):
                os.remove(p)


@pytest.fixture(autouse=True)
def restore_strategy_step():
    """Ensure spot_engine.Strategy.step is restored if a test monkeypatches it directly."""
    from strategies import spot_engine

    orig = spot_engine.Strategy.step
    yield
    spot_engine.Strategy.step = orig


@pytest.fixture(autouse=True)
def _isolate_monitortrades_state(tmp_path, monkeypatch):
    """Keep every test away from the shared monitortrades state file (cachedb/mt_state.json)."""
    import monitortrades as mt

    monkeypatch.setattr(
        mt, "_MT_STATE_FILE", str(tmp_path / "mt_state.json"))


@pytest.fixture(scope="session", autouse=True)
def _shutdown_runtime_threads_after_suite():
    yield

    cm.CacheManagerInterface.shutdown_all_instances()
    cm.CacheShortTrendManager.shutdown_all_instances()
    cm.CacheFactory.shutdown_all()

    from binance_api import bapi_client, bapi_ws, bapi_ws_bridge
    if bapi_ws_bridge._ws_bridge is not None:
        bapi_ws_bridge._ws_bridge.stop()
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

