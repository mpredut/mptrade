"""Regression tests for border and edge cases across providers and SSOT cache managers."""

import argparse
import ast
from datetime import datetime, timezone
from decimal import Decimal
import functools
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import account_cache_health as health
import active_intents
import assetguardian as ag
from binance_api import bapi_placeorder, bapi_trades, bapi_ws_bridge, bapi_ws_normalize as ws_norm
import botcore
from botcore import (
    clean_scale,
    coerce_dict_maps,
    extract_timestamp_ms,
    extract_timestamp_s,
    first_epoch_sec,
    first_pos_float,
    is_market_order,
    load_cli_env_stack,
    parse_cli_float,
    parse_cli_int,
    parse_nonneg_cli_float,
    parse_nonneg_cli_int,
    parse_pos_cli_float,
    parse_pos_cli_int,
    parse_ratio_or_pct,
    resolve_cli_env_path,
    temporary_environ,
)
from cacheaccount import CacheAssetValueManager, CacheOrderManager, CacheTradeManager
import cacheintegrity
import cachepaths
from cachepaths import (
    filter_candidates_by_asset_class,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    iter_venue_cycle_state_files,
)
from cacheprice import (
    Cache24PriceManager,
    Cache365PriceManager,
    CacheCurrentPriceManager,
    resolve_cached_symbol_price,
    resolve_ssot_price_from_cache,
)
import cachetrend
from cachetrend import CacheShortTrendManager
import external_order_guard as eog
import guard_hierarchy as gh
from guard_hierarchy import preflight_intelligence_guard_entry
from instrument import Instrument, _is_aborted
from instrument_registry import has_workspace_instrument_sources
from intelligence.external.collectors.base_collector import _DiskBackedSymbolCollector
from intelligence.external.collectors.bybit_liquidations_collector import BybitLiquidationCollector
from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.guard_decision import BrakeAction, GuardDecision, unpack_guard_decision
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
from intelligence.internal.state import survival as surv_mod
from intelligence.macro import calendar_ssot
from intelligence.macro.calendar_ssot import headline_effective_ts, parse_headline_age_minutes
import intelligence.macro.daemon_ssot as dssot
from intelligence.macro.daemon_ssot import (
    ExternalTelemetryMixin,
    add_external_telemetry_cli_args,
    add_macro_analyzer_cli_args,
    run_daemon_cli_once_or_start,
)
import intelligence.macro.geopolitical_analyzer as geo_an_mod
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
import intelligence.macro.geopolitical_guard as geo_gd_mod
import intelligence.macro.lexicon_ssot as lex_ssot
from intelligence.macro.lexicon_ssot import _tracked_equity_base_tickers
from intelligence.macro.news_feed_collector import NewsFeedCollector
import intelligence.macro_analyzer as macro_an_mod
import intelligence.macro_sentiment_ssot as mss
from intelligence.macro_sentiment_ssot import FearGreedSnapshot
from intelligence.sentiment import llm_quota_router as lqr_mod
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
import intelligence.sentiment.collectors.macro_telemetry_collector as mtc_mod
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    MacroAndNewsSensorsCollector,
    MacroBenchmarksSnapshotCollector,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _load_open_order_timestamps,
    _resolve_ai_intents_and_known_order_ids,
)
from intelligence.sentiment.evaluators import (
    microstructure_evaluator as me_mod,
    price_series_evaluator as pse_mod,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import evaluate_microstructure_cycle_deltas
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    clip_open_lots_to_held_qty,
    parse_venue_cycle_state,
    reconcile_fresh_lots_vwap,
    select_fresh_open_lots,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    OpenLimitOrdersTcaEvaluator,
    VenueFeeAndLimitsEvaluator,
)
from intelligence.sentiment.guards.high_stake_guard import resolve_llm_guard_cache_ttl
from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
from kraken import (
    kraken_xstock_watch as xstock_watch,
    market_data as kraken_md_mod,
    shadow_live as kraken_shadow,
)
from kraken.kraken_client import KrakenClient
from kraken.kraken_common import load_isolated_kraken_cfg
from kraken.trailing_stop import KrakenTrailing
import llm_order_guard
from market_monitor.new_coins_discovery import NewCoinsFactory
from market_monitor.pricechecker import PriceAlert, PriceChecker
from market_monitor.pricefetcher import StockYahooPricePlatform
from market_regime import MarketRegimeHorizon, MarketRegimeService
import monitortrades as mt
from notify_engine import alertnotifiers as alert_mod, server as notify_server_mod
from offline.backtests import evaluation as bt_eval
from offline.backtests.comparison import add_path_cli_args, add_report_output_cli_args
from offline.backtests.evaluation import build_execution_model_from_args, parse_intervals
from offline.research import backtest_ranges, short_trend_estimator_benchmark as st_bench
from offline.research.monitortrades_backtest import sweep_mt_models as smt
from offline.research.monitortrades_backtest.run_replay_backtest import step_adaptive_trend_window
from offline.research.monitortrades_backtest.sweep_mt_models import (
    _accepts_two_positional_args,
    compute_window_bounds,
)
from offline.runners import apply_proposals
from orchestratorOS.admin import prune_llm_threads as plt_mod
from orchestratorOS.agy import paths as agy_paths, ui_mirror as agy_ui_mirror
from orchestratorOS.lib import technical_repair_ssot as tr_ssot
from orchestratorOS.livecheck import os_resources_watchdog as wd
from orchestratorOS.supervisor import prompt_sections as ps_mod
import order_guard
from order_lifecycle import TrackedOrderLifecycle
import order_retry
import priceAnalysis
import pricewindow as pw
from providers import base as pbase_mod, binance_provider as bp_mod, order_id_context as oic
from providers.base import MarketDataProvider
from providers.binance_provider import BinanceProvider
from providers.execution_audit import (
    _filter_intents_by_scope,
    _is_open_resting_detail,
    reconcile_intents_from_execution_audit,
    summarize_recent_ai_intents,
)
from providers.hyperliquid_provider import HyperliquidProvider, _call_hl_info_read
from providers.intent_state import _resolve_clamped_intent_notional
from providers.kraken_provider import KrakenProvider
from providers.ohlc_utils import (
    build_price_history_points,
    drop_incomplete_last_bar,
    resolve_ohlc_time_window_ms,
    validate_dataset,
    validate_ohlc_candle_bar,
)
from providers.order_ohlc_utils import (
    POST_ONLY_CROSS_TOKENS,
    compute_post_only_maker_price,
    derive_venue_128bit_client_order_id,
    filter_and_normalize_open_orders,
    format_venue_intent_client_order_id,
    is_post_only_cross_refusal,
    iter_expired_open_order_ids,
    parse_nonneg_order_field,
)
from providers.price_poller import (
    NonBinanceTrendPoller,
    fetch_price_with_deadline,
    start_nonbinance_trend_poller,
)
from providers.quantity import (
    BinanceFilterError,
    clamp_notional_to_wallet_balance,
    decimal_places,
    dust_protected_partial_sell_qty,
    dust_protected_sell_balance_cap,
    venue_adjust_partial_sell_quantity,
    venue_fee_cap_quantity,
)
from providers.replay_provider import ReplayMarketDataProvider
from providers.strategy_executor import OrderReconciliationCapabilities, ProviderError, SubmissionRefused
from providers.t212_provider import T212Provider
import qlimit_order_guard as qog
import rtrade
from rtrade_pair_store import RTradePairStore
import state_io
from state_io import (
    atomic_write_text,
    decode_embedded_json_dict,
    extract_json_object,
    previous_generation_path,
    read_jsonl_dicts,
    read_tail_complete_lines_text,
    read_text_file,
)
from strategies import (
    mt_position_accounting as mt_pa_mod,
    rtrade_pair as rtp_mod,
    spot_engine as spot_eng_mod,
    state_store as strat_ss_mod,
)
from strategies.base_spot_mixin import BaseSpotMixin, parse_tp_ladder_env
from strategies.replay_execution import ExecutionModel, select_eligible_intrabar_orders
from strategies.replay_metrics import (
    calculate_performance_metrics,
    print_ascii_table_header,
    print_single_backtest_report,
    run_cli_single_or_sweep_backtest,
    run_tp_drop_sl_sweep,
    validate_common_backtest_cli_bounds,
    write_markdown_report,
)
from strategies.spot_replay import (
    iter_end_of_replay_market_sell_orders,
    params_require_trend_interval,
    post_exit_peak_step_scope,
    ratchet_post_exit_peak_sell_price,
    required_replay_intervals,
    split_warmup_ohlc,
    summarize_grid_replay_metrics,
)
from tests.conftest import (
    DummyCachedHttpJsonCollector,
    DummyGeminiClient,
    MockHttpResponse,
    load_venue_modules,
    make_test_margins,
)
import tradeall
from trailing_core import TrailingAdapterBase
import utils
from verify_tools import check_cache_coherence, monitor_night, ownership_inventory, portfolio_snapshot
import visuals.tradeall_observe as tobs


class _StubKrakenPairClient:
    def __init__(self, info: dict):
        self._info = info

    def cached_pair_info(self, _cands):
        return self._info


class _TestMarketProvider(MarketDataProvider):
    _USES_SSOT_MARKET_CACHE = False
    _REMOTE_TRADES_CLIENT_ATTR = "fetch_fills"

    def __init__(self, prov_name: str = "probe", *, price_fn=None, fills_fn=None, cli=None, client=None):
        eff_cli = client if client is not None else cli
        self._prov_name = prov_name
        self._price_fn = price_fn
        self._fills_fn = fills_fn or (
            (lambda sym, publish_cache=False: eff_cli.fetch_fills(sym))
            if eff_cli is not None and callable(getattr(eff_cli, "fetch_fills", None))
            else None
        )
        if eff_cli is not None:
            self._cli = eff_cli

    @property
    def name(self) -> str:
        return self._prov_name

    def supports_symbol(self, symbol: str) -> bool:
        return True

    def _fetch_current_price(self, symbol: str):
        return self._price_fn(symbol) if self._price_fn is not None else None

    def _fills_from_api(self, symbol: str, *, publish_cache: bool = False):
        return self._fills_fn(symbol, publish_cache=publish_cache) if self._fills_fn is not None else []


class _CustomDexProvider(MarketDataProvider):
    _LIVE_ORDERS_ENV = "CUSTOM_DEX_LIVE_ORDERS"
    _CLIENT_ID_128BIT_SPEC = (
        "CustomDex",
        b"assertguardian:customdex:cloid:v1\0",
        re.compile(r"^0x[0-9a-fA-F]{32}$"),
        "0x",
    )
    _INTENT_CLIENT_ID_FORMAT = "CDX_{token}"
    _BUY_SLIPPAGE_FEE_BUFFER = 0.97
    _SELL_DUST_SPEC = (3, 5.0)

    def __init__(self, live_enabled=None):
        self._live_enabled = live_enabled

    @property
    def name(self) -> str:
        return "CustomDex"

    def supports_symbol(self, symbol: str) -> bool:
        return True

    def free_balance(self, asset: str):
        return 1.0 if asset == "ABC" else 100.0


def test_freshness_parameters_fail_closed_on_nonpositive_bool_and_nan(tmp_path: Path):
    """Verify freshness thresholds reject 0, negative, bool, NaN, and inf across SSOT caches."""
    now_s = time.time()
    now_ms = int(now_s * 1000)

    price_file = tmp_path / "cache_current_price.json"
    price_file.write_text(
        json.dumps({"items": {"BTCUSDC": 91000.0}, "fetchtime": {"BTCUSDC": now_ms}}),
        encoding="utf-8",
    )
    assert CacheCurrentPriceManager.read_ssot_current_price_candidates(
        ("BTCUSDC",), str(price_file), max_age_sec=None, use_memory_manager=False
    ) == pytest.approx(91000.0)
    for bad_age in (0, 0.0, -5.0, False, True, float("nan"), float("inf")):
        assert (
            CacheCurrentPriceManager.read_ssot_current_price_candidates(
                ("BTCUSDC",), str(price_file), max_age_sec=bad_age, use_memory_manager=False
            )
            is None
        )
        assert (
            resolve_ssot_price_from_cache(
                "BTCUSDC",
                cache_dir=str(tmp_path),
                max_age_sec=bad_age,
                now_ts=now_s,
            )
            is None
        )

    c24_file = tmp_path / "cache_24_price_BTCUSDC.json"
    c24_file.write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        [now_ms - 1_500_000, 90000.0],
                        [now_ms - 10_000, 91000.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    hist = Cache24PriceManager.read_ssot_price_history(
        ("BTCUSDC",), 0.5, max_latest_age_sec=None, custom_dir=str(tmp_path)
    )
    assert isinstance(hist, list) and len(hist) == 2
    for bad_age in (0, -1.0, False, True, float("nan"), float("inf")):
        assert (
            Cache24PriceManager.read_ssot_price_history(
                ("BTCUSDC",), 0.5, max_latest_age_sec=bad_age, custom_dir=str(tmp_path)
            )
            is None
        )

    trend_file = tmp_path / "cache_short_trend.json"
    trend_file.write_text(
        json.dumps(
            {
                "items": {"BTCUSDC": {"direction": "UP", "slope": 0.25, "ts": now_s}},
                "fetchtime": {"BTCUSDC": now_ms},
            }
        ),
        encoding="utf-8",
    )
    snap = CacheShortTrendManager.read_ssot_short_trend_file(
        str(trend_file), ("BTCUSDC",), max_age_sec=None, now_s=now_s
    )
    assert snap is not None and snap["direction"] == "UP"
    for bad_age in (0, -2.0, False, True, float("nan"), float("inf")):
        assert (
            CacheShortTrendManager.read_ssot_short_trend_file(
                str(trend_file), ("BTCUSDC",), max_age_sec=bad_age, now_s=now_s
            )
            is None
        )

    fills_file = tmp_path / "cache_trade_kraken.json"
    fills_file.write_text(
        json.dumps(
            {
                "items": {"XBTUSD": [{"orderId": "o1", "price": 91000.0, "qty": 0.1, "time": now_ms}]},
                "fetchtime": {"XBTUSD": now_ms},
            }
        ),
        encoding="utf-8",
    )
    assert (
        CacheTradeManager.read_venue_fills_snapshot(
            "XBTUSD", str(fills_file), max_stale_sec=None, candidate_keys=("XBTUSD",)
        )
        is not None
    )
    for bad_stale in (0, -1.0, False, True, float("nan"), float("inf")):
        assert (
            CacheTradeManager.read_venue_fills_snapshot(
                "XBTUSD", str(fills_file), max_stale_sec=bad_stale, candidate_keys=("XBTUSD",)
            )
            is None
        )


def test_previous_generation_recovery_and_trailing_malformed_entry_scan(tmp_path: Path):
    """Verify .previous recovery when primary files are missing and reverse scan over malformed 24h tail."""
    now_s = time.time()
    now_ms = int(now_s * 1000)

    # 1. CacheCurrentPriceManager recovers from .previous when primary file is absent
    price_file = tmp_path / "cache_current_price.json"
    Path(previous_generation_path(str(price_file))).write_text(
        json.dumps({"items": {"ETHUSDC": 3250.5}, "fetchtime": {"ETHUSDC": now_ms}}),
        encoding="utf-8",
    )
    assert CacheCurrentPriceManager.read_ssot_current_price_candidates(
        ("ETHUSDC",), str(price_file), max_age_sec=30.0, use_memory_manager=False
    ) == pytest.approx(3250.5)

    # 2. resolve_ssot_price_from_cache scans backwards past trailing malformed 24h entry and recovers .previous
    c24_sol = tmp_path / "cache_24_price_SOLUSDC.json"
    Path(previous_generation_path(str(c24_sol))).write_text(
        json.dumps(
            {
                "items": {
                    "SOLUSDC": [
                        [now_ms - 5_000, 185.25],
                        [now_ms - 1_000, "corrupt_px"],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert resolve_ssot_price_from_cache(
        "SOLUSDC",
        cache_dir=str(tmp_path),
        max_age_sec=30.0,
        now_ts=now_s,
    ) == pytest.approx(185.25)

    # 3. resolve_cached_symbol_price rejects bool symbol and supports dict/zero-ts 24h entries via .previous
    assert math.isnan(resolve_cached_symbol_price(True, cache_dir=str(tmp_path)))  # type: ignore[arg-type]
    assert math.isnan(resolve_cached_symbol_price(False, cache_dir=str(tmp_path)))  # type: ignore[arg-type]

    c24_nvda = tmp_path / "cache_24_price_NVDA_US_EQ.json"
    Path(previous_generation_path(str(c24_nvda))).write_text(
        json.dumps({"items": {"NVDA": [{"price": 142.75}, [0, 143.5]]}}),
        encoding="utf-8",
    )
    assert resolve_cached_symbol_price(
        "NVDA_US_EQ", venue="t212", base="NVDA", quote="USD", cache_dir=str(tmp_path)
    ) == pytest.approx(143.5)

    # 4. CacheTradeManager.invalidate_venue_fills_snapshot recovers from .previous when primary is missing
    fills_file = tmp_path / "cache_trade_kraken.json"
    Path(previous_generation_path(str(fills_file))).write_text(
        json.dumps({"items": {"XBTUSD": []}, "fetchtime": {"XBTUSD": now_ms}}),
        encoding="utf-8",
    )
    CacheTradeManager.invalidate_venue_fills_snapshot(
        "XBTUSD", str(fills_file), candidate_keys=("XBTUSD",)
    )
    recovered = json.loads(fills_file.read_text(encoding="utf-8"))
    assert "XBTUSD" not in recovered["fetchtime"]


def test_merge_venue_fills_handles_mixed_string_int_timestamps_and_bool_now_ms():
    """Verify CacheTradeManager.merge_venue_fills_into_dict sorts mixed string/int timestamps safely."""
    cache_obj = {
        "items": {
            "NVDA_US_EQ": [
                {"orderId": "ord_2", "timestamp": "1700000002000", "price": 121.0, "qty": 1.0}
            ]
        },
        "fetchtime": {},
    }
    changed = CacheTradeManager.merge_venue_fills_into_dict(
        cache_obj,
        {
            "NVDA_US_EQ": [
                {"id": "ord_1", "timestamp": 1700000001000, "price": 120.0, "qty": 1.0},
                {"orderId": "ord_3", "time": "1700000003000", "price": 122.0, "qty": 1.0},
            ]
        },
        now_ms=True,  # type: ignore[arg-type]
    )
    assert changed is True
    rows = cache_obj["items"]["NVDA_US_EQ"]
    assert [r.get("orderId") or r.get("id") for r in rows] == ["ord_1", "ord_2", "ord_3"]
    assert cache_obj["fetchtime"]["NVDA_US_EQ"] > 1_000_000_000_000


def test_provider_base_border_cases_live_quote_falsy_client_and_fills_typeerror(monkeypatch):
    """Verify MarketDataProvider live quote sanitization, falsy mock client detection, and _invoke_fills_from_api."""
    monkeypatch.setattr(botcore._RUNTIME_SERVICES, "allow_shared_disk_cache", True)
    # A falsy injected mock client (empty dict) must still disable shared disk SSOT cache!
    kr_with_falsy_mock = KrakenProvider(client={})
    assert bool(kr_with_falsy_mock._cli) is False
    assert kr_with_falsy_mock._allow_ssot_price_cache() is False

    bad_quotes = iter([0.0, -5.0, float("nan"), float("inf"), True, False, 42.5])
    probe = _TestMarketProvider(price_fn=lambda _sym: next(bad_quotes))
    for _ in range(6):
        assert probe.get_current_price("BTCUSDC") is None
    assert probe.get_current_price("BTCUSDC") == pytest.approx(42.5)

    # Internal TypeError inside _fills_from_api(symbol, publish_cache=...) must NOT trigger a duplicate API call
    calls = []

    def _buggy_fills(symbol: str, *, publish_cache: bool = False):
        calls.append((symbol, publish_cache))
        raise TypeError("internal bug inside _fills_from_api")

    buggy = _TestMarketProvider(fills_fn=_buggy_fills)
    with pytest.raises(TypeError, match="internal bug"):
        buggy.fetch_remote_trades("BTCUSDC", 0)
    assert len(calls) == 1

    # Valid fills provider filters non-dict entries and rejects invalid start_time_ms
    fp = _TestMarketProvider(
        fills_fn=lambda _sym, *, publish_cache=False: [
            "not_a_dict",
            {"orderId": "1", "time": 1_000, "price": 10.0, "qty": 1.0},
            {"orderId": "2", "time": 5_000, "price": 11.0, "qty": 1.0},
        ]
    )
    assert fp.fetch_remote_trades("BTCUSDC", 0) == [
        {"orderId": "1", "time": 1_000, "price": 10.0, "qty": 1.0},
        {"orderId": "2", "time": 5_000, "price": 11.0, "qty": 1.0},
    ]
    assert fp.fetch_remote_trades("BTCUSDC", 3_000) == [
        {"orderId": "2", "time": 5_000, "price": 11.0, "qty": 1.0}
    ]
    for bad_start in (-1, True, False, float("nan"), float("inf")):
        assert fp.fetch_remote_trades("BTCUSDC", bad_start) == []  # type: ignore[arg-type]


def test_fee_and_limits_resolution_symmetry_and_no_double_trailing_rest_call(tmp_path: Path):
    """Verify resolve_cached_fee_and_limits across venues/stubs and single REST call on KrakenTrailing failure."""
    # 1. Custom MarketDataProvider subclass named 'kraken' resolves Kraken fee/limits via base class
    stub = _TestMarketProvider(
        "kraken",
        cli=_StubKrakenPairClient(
            {
                "ordermin": "0.0001",
                "pair_decimals": 1,
                "lot_decimals": 8,
                "fees_maker": [[0, 0.16]],
                "fees": [[0, 0.26]],
            }
        ),
    )
    meta = stub.resolve_cached_fee_and_limits("XBTUSD", price=80000.0)
    assert meta["maker_fee_bps"] == pytest.approx(16.0)
    assert meta["tick_size"] == pytest.approx(0.1)
    assert MarketDataProvider.resolve_provider_cached_fee_and_limits("kraken", "XBTUSD", stub, price=80000.0) == meta

    # 2. SimpleNamespace hook without price kwarg works cleanly
    ns_prov = SimpleNamespace(resolve_cached_fee_and_limits=lambda sym: {"symbol": sym, "maker_fee_bps": 5.0})
    assert MarketDataProvider.resolve_provider_cached_fee_and_limits(
        "custom", "BTCUSDC", ns_prov, price=90000.0
    ) == {"symbol": "BTCUSDC", "maker_fee_bps": 5.0}

    # 3. HyperliquidProvider inherits resolve_cached_fee_and_limits with dynamic price tick_size
    hl = HyperliquidProvider(
        token="HYPE", client=SimpleNamespace(_sz_decimals_cache={"HYPE": 2}), live_enabled=False
    )
    hl_meta = hl.resolve_cached_fee_and_limits("HYPE", price=42.35)
    assert hl_meta["step_size"] == pytest.approx(0.01)
    assert hl_meta["tick_size"] > 0

    # 4. KrakenTrailing.price() calls client.last_price at most once when quote is 0.0 or None
    price_calls = []
    fake_k_cli = SimpleNamespace(
        last_price=lambda pair: price_calls.append(pair) or 0.0,
        balance_ex=lambda: {},
    )
    kt = KrakenTrailing(
        client=fake_k_cli,  # type: ignore[arg-type]
        enabled=False,
        state_file=str(tmp_path / "kt_state.json"),
    )
    assert kt.price("XBTUSD") is None
    assert price_calls == ["XBTUSD"]


def test_t212_empty_fills_publish_stampede_prevention_and_open_orders_normalization(tmp_path: Path):
    """Verify T212Provider publishes fetchtime on empty fills and normalizes open_orders via build_open_order_entry."""
    fills_cache = tmp_path / "cache_trade_t212.json"

    refresh_calls = []

    fake_t212_cli = SimpleNamespace(
        refresh_order_history=lambda max_age_sec=0.0: refresh_calls.append(max_age_sec) or [],
        get_portfolio=lambda: [],
        list_active_orders=lambda: [
            {"ticker": "NVDA_US_EQ", "side": "BUY", "quantity": 2.0, "limitPrice": 130.0},  # missing id -> skipped
            {
                "id": 98765,
                "ticker": "NVDA_US_EQ",
                "side": "BUY",
                "type": "LIMIT",
                "status": "WORKING",
                "quantity": 2.5,
                "filledQuantity": 0.5,
                "limitPrice": 131.25,
                "dateCreated": "2026-11-10T18:00:00Z",
            },
        ],
    )

    prov = T212Provider(client=fake_t212_cli, live_enabled=False)
    prov._ssot_fills_cache_path = str(fills_cache)

    # Authoritative refresh with publish_cache=True records fetchtime even when symbol has 0 fills
    assert prov.get_orders("NVDA_US_EQ", None, 3600.0, authoritative=True, publish_cache=True) == []
    assert len(refresh_calls) == 1
    saved = json.loads(fills_cache.read_text(encoding="utf-8"))
    assert saved["items"]["NVDA_US_EQ"] == []
    assert saved["fetchtime"]["NVDA_US_EQ"] > 0

    # Subsequent cached read hits SSOT cache without calling refresh_order_history again
    assert prov.get_orders("NVDA_US_EQ", None, 3600.0) == []
    assert len(refresh_calls) == 1

    # open_orders skips missing orderId and populates standard keys via build_open_order_entry
    open_ords = prov.open_orders("NVDA_US_EQ")
    assert len(open_ords) == 1
    entry = open_ords[0]
    assert entry["orderId"] == "98765"
    assert entry["id"] == "98765"
    assert "clientOrderId" in entry
    assert entry["time"] > 0
    assert entry["origQty"] == pytest.approx(2.5)
    assert entry["executedQty"] == pytest.approx(0.5)

    # fetch_remote_trades rejects invalid start_time_ms
    for bad_start in (-5, True, False, float("nan")):
        assert prov.fetch_remote_trades("NVDA_US_EQ", bad_start) == []  # type: ignore[arg-type]


def test_kraken_fetch_remote_trades_case_insensitivity_and_60s_floor_filtering():
    """Verify KrakenProvider.fetch_remote_trades handles lowercase symbols, >= start_time_ms within 60s, and invalid bounds."""
    now_s = time.time()
    old_ts = now_s - 45.0
    new_ts = now_s - 5.0

    cli = SimpleNamespace(
        _private=lambda method, params=None, fresh=False: {
            "ledger": {
                "L1_B": {"refid": "REF_OLD", "type": "trade", "asset": "XXBT", "amount": "0.01", "fee": "0", "time": old_ts},
                "L1_Q": {"refid": "REF_OLD", "type": "trade", "asset": "ZUSD", "amount": "-900.0", "fee": "0.9", "time": old_ts},
                "L2_B": {"refid": "REF_NEW", "type": "trade", "asset": "XXBT", "amount": "0.02", "fee": "0", "time": new_ts},
                "L2_Q": {"refid": "REF_NEW", "type": "trade", "asset": "ZUSD", "amount": "-1820.0", "fee": "1.8", "time": new_ts},
            }
        }
    )
    kp = KrakenProvider(client=cli)

    # Cutoff at now - 20s must exclude REF_OLD (at now - 45s) even though Ledgers uses a 60s lookback floor,
    # and lowercase/padded ' xbtusd ' must still match configured pair 'XBTUSD'.
    cutoff_ms = int((now_s - 20.0) * 1000)
    rows = kp.fetch_remote_trades(" xbtusd ", cutoff_ms, client=cli)
    assert [r["orderId"] for r in rows] == ["REF_NEW"]
    assert rows[0]["symbol"] == "XBTUSD"

    for bad_start in (-1, True, False, float("nan"), float("inf")):
        assert kp.fetch_remote_trades("XBTUSD", bad_start, client=cli) == []  # type: ignore[arg-type]
    for bad_lb in (0, -10.0, False, True, float("nan"), float("inf")):
        assert KrakenProvider.fetch_ledger_fills_by_symbol(cli, pairs=["XBTUSD"], lookback_sec=bad_lb) == {}  # type: ignore[arg-type]


def test_equity_asset_class_ssot_helpers_and_intent_scope_filtering():
    """Verify has_equity_suffix, is_equity_venue, filter_candidates_by_asset_class, and _filter_intents_by_scope."""

    for bad_sym in (None, True, False, 123, "", "   ", "BTCUSDC"):
        assert has_equity_suffix(bad_sym) is False
    for eq_sym in ("NVDA_US_EQ", " nvda_us_eq ", "AAPL_EQ", "aapl_eq"):
        assert has_equity_suffix(eq_sym) is True

    assert is_equity_venue("t212") is True
    assert is_equity_venue("Trading212") is True
    assert is_equity_venue("212") is True
    for non_eq_v in (None, True, False, "", "binance", "kraken", "hyperliquid"):
        assert is_equity_venue(non_eq_v) is False

    assert is_equity_venue_or_suffix("AAPL", "t212") is True
    assert is_equity_venue_or_suffix("nvda_us_eq", None) is True
    assert is_equity_venue_or_suffix("BTCUSDC", "binance") is False

    mixed = ["nvda_us_eq", "NVDA_US_EQ", "NVDA", "NVDAUSDC", None, True]
    assert filter_candidates_by_asset_class(mixed, is_equity=True) == [
        "nvda_us_eq",
        "NVDA_US_EQ",
        "NVDA",
    ]
    assert filter_candidates_by_asset_class(mixed, is_equity=False) == [
        "NVDA",
        "NVDAUSDC",
    ]
    assert filter_candidates_by_asset_class(None, is_equity=True) == []
    assert filter_candidates_by_asset_class("NVDA_US_EQ", is_equity=True) == []

    # _filter_intents_by_scope must never crash on None/bool/numeric symbols and must match lowercase _us_eq
    intents = [
        {"id": "1", "symbol": None},
        {"id": "2", "symbol": True},
        {"id": "3", "symbol": 999},
        {"id": "4", "symbol": " nvda_us_eq "},
        {"id": "5", "venue": "Trading212", "symbol": "AAPL"},
        {"id": "6", "venue": "binance", "symbol": "BTCUSDC"},
    ]
    eq_only = _filter_intents_by_scope(intents, "equities")
    assert [i["id"] for i in eq_only] == ["4", "5"]
    crypto_only = _filter_intents_by_scope(intents, "crypto")
    assert [i["id"] for i in crypto_only] == ["1", "2", "3", "6"]


def test_dust_protected_sell_sizing_and_ohlc_client_id_border_cases():
    """Verify venue-agnostic dust-protected SELL sizing, client ID venue alias resolution, and OHLC window overflow."""

    # Parameterized dust-protected cap and partial sell adjuster work across arbitrary decimals & thresholds
    assert dust_protected_sell_balance_cap(
        1.005, 100.0, qty_decimals=2, min_remainder_notional=6.0
    ) == pytest.approx(0.94)
    assert dust_protected_sell_balance_cap(
        1.0005, 100.0, qty_decimals=3, min_remainder_notional=1.0
    ) == pytest.approx(0.99)
    qty_adj, mode = dust_protected_partial_sell_qty(
        0.97, 1.0, 100.0, qty_decimals=2, min_remainder_notional=6.0, min_notional_usd=10.0
    )
    assert qty_adj == pytest.approx(0.94)
    assert mode == "leave_min_6_remainder"

    # Declarative T212 specs via venue_fee_cap_quantity / venue_adjust_partial_sell_quantity
    assert venue_fee_cap_quantity("Trading212", "SELL", 100.0, 1.005) == pytest.approx(0.94)
    v_qty, v_mode = venue_adjust_partial_sell_quantity(
        "212", "SELL", 0.08, 0.10, 100.0, min_notional_usd=10.0
    )
    assert v_qty == pytest.approx(0.10)
    assert v_mode == "avoid_sub_6_dust"

    # Venue aliases and invalid bool/blank tokens in client_order_id helpers
    assert derive_venue_128bit_client_order_id("hl", "intent_1").startswith("0x")
    assert len(derive_venue_128bit_client_order_id("krk", "intent_1")) == 32
    with pytest.raises(ProviderError):
        derive_venue_128bit_client_order_id(True, "intent_1")  # type: ignore[arg-type]

    assert format_venue_intent_client_order_id("bin", "a" * 32) == "SD_" + ("a" * 32)
    assert format_venue_intent_client_order_id("hl", "b" * 32) == "0x" + ("b" * 32)
    assert format_venue_intent_client_order_id("krk", "c" * 32) == "c" * 32
    for bad_tok in (None, True, False, "", "   "):
        assert format_venue_intent_client_order_id("binance", bad_tok) is None  # type: ignore[arg-type]

    # Huge finite lookback_days / lookback_hours that overflow to inf when scaled to ms must raise ValueError, not OverflowError
    with pytest.raises(ValueError, match="lookback_days"):
        resolve_ohlc_time_window_ms(lookback_days=1e308)
    with pytest.raises(ValueError, match="lookback_hours"):
        resolve_ohlc_time_window_ms(lookback_hours=1e308)


def test_account_cache_and_legacy_jsonl_border_cases(tmp_path: Path):
    """Verify CacheAccountManagerBase cursor/fetchtime sanitization and legacy JSONL strict ID / tail handling."""

    assert not hasattr(cachepaths, "_is_eq_suffix")

    snap_file = tmp_path / "cache_trade_kraken.json"
    snap_file.write_text(
        json.dumps(
            {
                "items": {"XBTUSD": []},
                "fetchtime": {
                    "XBTUSD": 1_700_000_005_000.0,
                    "BAD_NAN": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    mgr = CacheTradeManager(
        venue="kraken",
        filename=str(snap_file),
        data_sync_mode="disk_reader",
    )
    mgr._persisted_data_version = "v_test"
    mgr._rest_sync_cursor_per_symbol = {
        "BOOL_SYM": True,
        "INF_SYM": float("inf"),
        "NAN_SYM": float("nan"),
        "VALID_SYM": 1_700_000_001_234,
    }
    ver, sync_ms = mgr.get_persisted_sync_state()
    assert ver == "v_test"
    assert sync_ms == 1_700_000_001_234

    # Injecting non-finite fetchtime values into the JSON snapshot must not poison _persisted_max_ts
    snap_file.write_text(
        '{"items": {}, "fetchtime": {"XBTUSD": 1700000005000, "INF": 1e309}}',
        encoding="utf-8",
    )
    assert mgr._persisted_max_ts() == pytest.approx(1_700_000_005_000.0)

    # read_legacy_trade_jsonl_strict recovers trailing partial JSON line when counts match,
    # and rejects bool/float IDs strictly
    valid_row = {
        "symbol": "BTCUSDC",
        "id": 101,
        "orderId": 202,
        "price": "50000.0",
        "qty": "0.01",
        "quoteQty": "500.0",
        "commission": "0.1",
        "commissionAsset": "USDC",
        "time": 1_700_000_000_000,
        "isBuyer": True,
        "isMaker": False,
    }
    legacy_jsonl = tmp_path / "legacy_trade.jsonl"
    legacy_jsonl.write_text(
        json.dumps({"s": "BTCUSDC", "i": valid_row}) + "\n" + '{"s": "BTCUSDC", "i": {',
        encoding="utf-8",
    )
    recovered = cacheintegrity.read_legacy_trade_jsonl_strict(
        str(legacy_jsonl),
        {"counts": {"BTCUSDC": 1}},
        is_valid_item_fn=CacheTradeManager._is_valid_item,
        reader_reason="test_legacy",
    )
    assert len(recovered["BTCUSDC"]) == 1

    bad_id_row = dict(valid_row, id=True)
    legacy_jsonl.write_text(
        json.dumps({"s": "BTCUSDC", "i": bad_id_row}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(health.AccountCacheNotReady):
        cacheintegrity.read_legacy_trade_jsonl_strict(
            str(legacy_jsonl),
            {"counts": {"BTCUSDC": 1}},
            is_valid_item_fn=lambda _r: True,
            reader_reason="test_legacy_bool_id",
        )


def test_open_closed_provider_class_specs_execution_symmetry_and_border_cases(monkeypatch):
    """Verify declarative provider class specs, cross-venue live/dry-run symmetry, and border hardening."""

    # 1. Open-Closed custom MarketDataProvider subclass works without touching any central dict
    monkeypatch.setenv("CUSTOM_DEX_LIVE_ORDERS", "false")
    custom_p = _CustomDexProvider()
    assert custom_p.execution_enabled() is False
    assert _CustomDexProvider.strategy_dry_run(False, True) is True
    monkeypatch.setenv("CUSTOM_DEX_LIVE_ORDERS", "true")
    assert custom_p.execution_enabled() is True
    assert _CustomDexProvider.strategy_dry_run(False, True) is False
    assert _CustomDexProvider(live_enabled=False).execution_enabled() is False

    cloid = derive_venue_128bit_client_order_id(_CustomDexProvider, "intent-42")
    assert cloid.startswith("0x") and len(cloid) == 34
    assert derive_venue_128bit_client_order_id(custom_p, "intent-42") == cloid
    assert format_venue_intent_client_order_id(custom_p, "deadbeef" * 4) == "CDX_" + ("deadbeef" * 4)
    assert venue_fee_cap_quantity(custom_p, "BUY", 10.0, 100.0) == pytest.approx(97.0)
    adj_q, adj_mode = venue_adjust_partial_sell_quantity(
        custom_p, "SELL", 0.8, 1.0, 10.0, min_notional_usd=5.0
    )
    assert adj_mode == "leave_min_5_remainder"
    assert adj_q == pytest.approx(0.5)

    ok_clamp, clamped_usd, clamp_reason = clamp_notional_to_wallet_balance(
        SimpleNamespace(symbol="ABCUSD", base="ABC", quote="USD", provider_name="CustomDex", _provider=custom_p),
        "SELL",
        10.0,
        8.0,
        min_notional_usd=5.0,
    )
    assert ok_clamp is True
    assert clamped_usd == pytest.approx(5.0)
    assert "$5" in clamp_reason

    # 2. Cross-venue live_enabled override & strategy_dry_run symmetry across all 4 concrete providers
    for prov_cls in (BinanceProvider, KrakenProvider, HyperliquidProvider, T212Provider):
        assert prov_cls(live_enabled=True).execution_enabled() is True
        assert prov_cls(live_enabled=False).execution_enabled() is False
        assert prov_cls(live_enabled="false").execution_enabled() is False  # type: ignore[arg-type]
        assert prov_cls(live_enabled=float("nan")).execution_enabled() is False  # type: ignore[arg-type]
        assert prov_cls.strategy_dry_run(True, True) is True
        assert prov_cls.strategy_dry_run(False, False) is True

    # 3. KrakenProvider.extract_cached_fee_and_limits does not mistake bound _client method for client object
    fresh_kr = KrakenProvider()
    assert fresh_kr._cli is None
    assert callable(getattr(fresh_kr, "_client", None))
    assert KrakenProvider.extract_cached_fee_and_limits(fresh_kr, "XBTUSD") == {}

    bad_kr = KrakenProvider(
        client=_StubKrakenPairClient(
            {
                "ordermin": "0.01",
                "pair_decimals": True,
                "lot_decimals": "2.5",
                "fees_maker": [[0, 0.16]],
                "fees": [[0, 0.26]],
            }
        )
    )
    meta_bad = KrakenProvider.extract_cached_fee_and_limits(bad_kr, "XBTUSD", price=80000.0)
    assert "tick_size" not in meta_bad
    assert "step_size" not in meta_bad
    assert meta_bad["min_qty"] == pytest.approx(0.01)
    assert KrakenProvider.extract_cached_fee_and_limits(bad_kr, True) == {}  # type: ignore[arg-type]

    bad_hl = SimpleNamespace(_meta_cache={"HYPE": {"szDecimals": "2.5"}}, _client=None)
    assert HyperliquidProvider.extract_cached_fee_and_limits(bad_hl, "HYPEUSDC", price=30.0) == {}
    assert HyperliquidProvider.extract_cached_fee_and_limits(bad_hl, True, price=30.0) == {}  # type: ignore[arg-type]

    # 4. _call_hl_info_read safely coerces bool/zero/NaN attempts and negative/NaN backoff_s
    calls = []

    def _flaky_hl():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("(429, 'rate limit')")
        return ["ok"]

    assert _call_hl_info_read(_flaky_hl, attempts=2, backoff_s=float("nan")) == ["ok"]
    assert len(calls) == 2

    single_calls = []
    with pytest.raises(RuntimeError):
        _call_hl_info_read(
            lambda: (single_calls.append(1), (_ for _ in ()).throw(RuntimeError("(503, 'busy')"))),
            attempts=True,  # type: ignore[arg-type]
            backoff_s=-1.0,
        )
    assert len(single_calls) == 1

    # 5. BinanceProvider.supports_symbol enforces instrument_registry venue ownership and non-empty base
    bp = BinanceProvider()
    assert bp.supports_symbol("BTCUSDC") is True
    assert bp.supports_symbol("HYPEUSDC") is False
    assert bp.supports_symbol("HYPE/USDC") is False
    assert bp.supports_symbol("USDC") is False
    assert bp.supports_symbol("/USDC") is False
    assert bp.supports_symbol(True) is False  # type: ignore[arg-type]
    assert bp.supports_symbol("") is False
    monkeypatch.setattr(bp_mod, "venue_for_symbol", lambda s: "hyperliquid" if "PURR" in s else "binance")
    assert bp.supports_symbol("PURRUSDC") is False


def test_provider_client_id_ratio_grace_and_history_border_cases(monkeypatch):
    """Verify parse_ratio_or_pct, derive_client_order_id, cancel grace, and extreme lookback border cases."""

    # 1. parse_ratio_or_pct & clean_scale strict border validation
    assert parse_ratio_or_pct("2.5%") == pytest.approx(0.025)
    assert parse_ratio_or_pct(2.5) == pytest.approx(0.025)
    assert parse_ratio_or_pct(0.03) == pytest.approx(0.03)
    assert parse_ratio_or_pct(0.0) == pytest.approx(0.0)
    for bad_val in (True, False, float("nan"), float("inf"), -0.01, "-5%", "", None):
        with pytest.raises(ValueError):
            parse_ratio_or_pct(bad_val)
    with pytest.raises(ValueError):
        parse_ratio_or_pct(0.02, pct_cutoff=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_ratio_or_pct(0.02, pct_cutoff=-0.1)
    assert clean_scale("85%", 0.0, allow_percent=True) == pytest.approx(0.85)
    with pytest.raises(TypeError):
        clean_scale("85%", 0.0, allow_percentage=True)  # type: ignore[call-arg]

    # 2. Polymorphic MarketDataProvider.derive_client_order_id
    kr_cid = KrakenProvider.derive_client_order_id("AI_intent-border-1")
    hl_cid = HyperliquidProvider.derive_client_order_id("AI_intent-border-1")
    assert len(kr_cid) == 32 and not kr_cid.startswith("0x")
    assert len(hl_cid) == 34 and hl_cid.startswith("0x")
    with pytest.raises(ProviderError):
        BinanceProvider.derive_client_order_id("AI_intent-border-1")

    # 3. MarketDataProvider.resolve_unfilled_order_cancel_grace_sec rejects bool/NaN/negative
    monkeypatch.setattr(mss, "resolve_derived_telemetry_intervals", lambda: {"unfilled_order_cancel_grace_sec": 45.0})
    assert MarketDataProvider.resolve_unfilled_order_cancel_grace_sec() == pytest.approx(45.0)
    for bad_grace in (True, float("nan"), float("inf"), -5.0, "bad"):
        monkeypatch.setattr(
            mss, "resolve_derived_telemetry_intervals", lambda v=bad_grace: {"unfilled_order_cancel_grace_sec": v}
        )
        with pytest.raises(ValueError):
            MarketDataProvider.resolve_unfilled_order_cancel_grace_sec()

    # 4. HyperliquidProvider token normalization & _fetch_price_history border cases
    assert HyperliquidProvider(token="HYPE/USDC")._token == "HYPE"
    assert HyperliquidProvider(token="PURR-USDC")._token == "PURR"
    captured_hl: list[tuple[str, str, int, int]] = []
    fake_hl_client = SimpleNamespace(
        resolve_spot_pair=lambda _tok: "@107",
        info=SimpleNamespace(
            candles_snapshot=lambda pair, iv, s, e: (
                captured_hl.append((pair, iv, s, e)) or [{"t": 1700000000000, "c": "25.5"}]
            )
        ),
    )
    hp = HyperliquidProvider(token="HYPE/USDC", client=fake_hl_client)
    for bad_lb in (0.0, -1.0, True, float("nan"), float("inf")):
        assert hp._fetch_price_history("HYPEUSDC", bad_lb) is None  # type: ignore[arg-type]
    assert captured_hl == []
    pts_hl = hp._fetch_price_history("HYPEUSDC", 1e15)
    assert pts_hl == [{"timestamp": 1700000000000, "price": 25.5}]
    assert captured_hl[-1][2] == 0 and captured_hl[-1][3] > 0
    _, start_clamped, end_clamped, _ = resolve_ohlc_time_window_ms(lookback_hours=1e15)
    assert start_clamped == 0 and end_clamped > 0

    # 5. KrakenProvider._fetch_price_history, build_price_history_points & fetch_ledger_fills_by_symbol overflow guards
    captured_kr: list[dict] = []
    captured_ledgers: list[dict] = []

    fake_kr_cli = SimpleNamespace(
        _public=lambda method, params: (
            captured_kr.append(dict(params))
            or {"HYPEUSD": [[1700000000, "20", "21", "19", "20.5"], [1e308, "20", "21", "19", "21.0"]], "last": 1}
        ),
        _private=lambda method, params=None, fresh=False: (
            captured_ledgers.append(dict(params or {})) or {"ledger": {}}
        ),
    )

    kp = KrakenProvider(client=fake_kr_cli)
    pts_kr = kp._fetch_price_history("HYPEUSD", 1e308)
    assert pts_kr == [{"timestamp": 1700000000000, "price": 20.5}]
    assert captured_kr[-1]["interval"] == 21600
    assert build_price_history_points(
        [[1e308, 10.0], [1700000000, 12.5]],
        lambda r: r[0],
        lambda r: r[1],
        ts_in_seconds=True,
    ) == [{"timestamp": 1700000000000, "price": 12.5}]
    assert KrakenProvider.fetch_ledger_fills_by_symbol(fake_kr_cli, pairs=("HYPEUSD",), lookback_sec=1e20) == {
        "HYPEUSD": []
    }
    assert captured_ledgers[-1]["start"] == 0


def test_embedded_json_ssot_fetch_public_bytes_and_hl_client_border_cases(monkeypatch, tmp_path):
    """Verify state_io embedded JSON helpers, botcore.fetch_public_bytes, HLClient._coin_candidates/candles, and iter_venue_cycle_state_files."""

    # 1. extract_json_object & decode_embedded_json_dict
    assert extract_json_object("```json\n{\"ok\": true, \"n\": 3}\n```") == {"ok": True, "n": 3}
    assert extract_json_object("prefix text {\"a\": {\"b\": 1}} trailing") == {"a": {"b": 1}}
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object("{\"bad\": NaN}") is None
    assert extract_json_object(True) is None  # type: ignore[arg-type]

    assert decode_embedded_json_dict("{\"title\": \"Alert\"}") == {"title": "Alert"}
    assert decode_embedded_json_dict("{\"title\": \"Alert\"} extra words") == {"title": "Alert"}
    assert decode_embedded_json_dict("```json\n{\"title\": \"Fenced\"}\n```") == {"title": "Fenced"}
    assert decode_embedded_json_dict("[1, 2]") is None
    with pytest.raises(ValueError):
        decode_embedded_json_dict("{\"x\": Infinity}")
    with pytest.raises(ValueError):
        decode_embedded_json_dict(None)  # type: ignore[arg-type]

    # 2. botcore.fetch_public_bytes
    for bad_timeout in (0, -1.0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            botcore.fetch_public_bytes("https://example.com/feed.xml", timeout=bad_timeout)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="HTTP"):
        botcore.fetch_public_bytes("file:///etc/passwd", timeout=5.0)

    monkeypatch.setattr(botcore.urllib.request, "urlopen", lambda req, timeout=10.0: MockHttpResponse(b"<rss><channel/></rss>"))
    assert botcore.fetch_public_bytes("https://example.com/feed.xml", timeout=5.0) == b"<rss><channel/></rss>"

    # 3. HLClient._coin_candidates & candles overflow safety
    hl_mod = load_venue_modules("hyperliquid", "hl_client")
    HLClient = hl_mod.HLClient
    assert HLClient._coin_candidates("HYPE/USDC") == ("HYPE/USDC", "HYPE")
    assert HLClient._coin_candidates("HYPE-USDC") == ("HYPE-USDC", "HYPE")
    assert HLClient._coin_candidates("HYPEUSDC") == ("HYPEUSDC", "HYPE")
    assert HLClient._coin_candidates("HYPE") == ("HYPE",)
    for bad_coin in (None, True, "", "   "):
        assert HLClient._coin_candidates(bad_coin) == ()

    dummy_cli = object.__new__(HLClient)
    seen_windows: list[tuple[str, str, int, int]] = []
    dummy_cli.info = SimpleNamespace(
        candles_snapshot=lambda coin, iv, start, end: (
            seen_windows.append((coin, iv, start, end)) or [{"t": 1700000000000, "c": "25.0"}]
        )
    )
    for bad_lh in (0, -5, True, float("nan"), float("inf"), 1e308):
        assert dummy_cli.candles("HYPE", "1h", lookback_hours=bad_lh) == []  # type: ignore[arg-type]
    assert seen_windows == []
    assert dummy_cli.candles("HYPE", "1h", lookback_hours=1e15) == [{"t": 1700000000000, "c": "25.0"}]
    assert seen_windows[-1][2] == 0 and seen_windows[-1][3] > 0

    # 4. iter_venue_cycle_state_files prioritizes _LONG/_SHORT state files across all venues
    kr_dir = tmp_path / "kraken"
    kr_dir.mkdir()
    (kr_dir / ".state_HYPEUSD.json").write_text("{}", encoding="utf-8")
    (kr_dir / ".state_HYPEUSD_LONG.json").write_text("{}", encoding="utf-8")
    (kr_dir / ".state_ADAUSD_SHORT.json").write_text("{}", encoding="utf-8")
    ordered = [os.path.basename(p) for p in iter_venue_cycle_state_files("kraken", workspace_dir=str(tmp_path))]
    assert ordered == [
        ".state_ADAUSD_SHORT.json",
        ".state_HYPEUSD_LONG.json",
        ".state_HYPEUSD.json",
    ]


def test_guard_regime_lifecycle_retry_and_cache_path_border_cases(monkeypatch, tmp_path):
    """Verify cache_path SSOT, http_request body, guard config validation, geo ts normalization, MarketRegimeService, OrderLifecycle, and order_retry border cases."""

    # 1. cache_path SSOT: single canonical definition in cachepaths, re-exported identically in botcore and utils
    assert botcore.cache_path is cachepaths.cache_path
    assert utils.cache_path is cachepaths.cache_path
    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", str(tmp_path))
    assert cachepaths.cache_path("custom_rel/file.json") == "custom_rel/file.json"
    assert cachepaths.cache_path("cache_current_price.json").startswith(str(tmp_path))
    monkeypatch.setattr(utils, "cache_path", lambda name: f"/override/{name}")
    assert cachepaths.cache_path("foo.json") == "/override/foo.json"
    monkeypatch.setattr(utils, "cache_path", cachepaths.cache_path)

    # 2. botcore.http_request raw body mutual exclusion & delivery
    with pytest.raises(ValueError, match="mutually exclusive"):
        botcore.http_request("POST", "https://example.com/hook", payload={"a": 1}, body=b"raw")
    with pytest.raises(ValueError, match="mutually exclusive"):
        botcore.http_request("POST", "https://example.com/hook", form={"a": 1}, body=b"raw")

    captured_req = {}

    def _fake_urlopen(req, timeout=10.0):
        captured_req["data"] = req.data
        captured_req["method"] = req.get_method()
        return MockHttpResponse(b"{}")

    monkeypatch.setattr(botcore.urllib.request, "urlopen", _fake_urlopen)
    st_code, _ = botcore.http_request("POST", "https://example.com/hook", body=b"hello", timeout=5.0)
    assert st_code == 200 and captured_req == {"data": b"hello", "method": "POST"}

    # 3. order_guard.get_window_cfg & daily_limit_guard reject corrupt/bool/fractional values
    for bad_w in (True, -1.0, float("nan"), float("inf"), "bad"):
        monkeypatch.setattr(order_guard, "_venue_cfg", lambda _n, _s, v=bad_w: v)
        with pytest.raises(ValueError, match="window_h"):
            order_guard.get_window_cfg("binance")
    dummy_prov = SimpleNamespace(name="binance", get_orders=lambda *_a: [])
    for bad_lim in (True, -1, 1.5, float("nan"), "bad"):
        monkeypatch.setattr(order_guard, "_venue_cfg", lambda _n, _s, v=bad_lim: v)
        with pytest.raises(ValueError, match="max_daily_trades"):
            order_guard.daily_limit_guard(dummy_prov, "BTCUSDC", "BUY")

    # 4. llm_order_guard.check_geopolitical_order_guards handles ISO/ms/bool/corrupt timestamps cleanly
    seen_absorbed: list[bool] = []
    now_ts = 1700003600.0
    iso_sa_ts = datetime.fromtimestamp(now_ts - 600.0, timezone.utc).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(
        geo_an_mod,
        "GeopoliticalThreatAnalyzer",
        lambda **_kw: SimpleNamespace(
            _cached_assessment=SimpleNamespace(threat_level="ELEVATED", ts=int((now_ts - 2400.0) * 1000)),
            _effective_cache_dir=lambda: str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        geo_gd_mod,
        "GeopoliticalShockGuard",
        lambda: SimpleNamespace(
            check=lambda _sym, _side, _ass, market_absorbed=False: (
                seen_absorbed.append(market_absorbed)
                or SimpleNamespace(allowed=True, reason="ok", brake_action="NONE", suggested_scale=1.0)
            )
        ),
    )
    monkeypatch.setattr(
        mss,
        "read_first_cached_dict",
        lambda *_a, **_kw: (
            {
                "ts": iso_sa_ts,
                "market_bias": "BULLISH",
                "recommended_action": "ACCUMULATE",
            },
            "sentiment_eval.json",
        ),
    )
    monkeypatch.setattr(mss, "load_cached_macro_benchmarks", lambda *_a, **_kw: {"ts": "corrupt"})
    ok, _, scale = llm_order_guard.check_geopolitical_order_guards(
        "BTCUSDC",
        "BUY",
        margins={"intelligence_guards_enabled": True, "geopolitical_guard_mode": "enforce"},
        now=now_ts,
    )
    assert ok is True and scale == 1.0 and seen_absorbed == [True]

    seen_absorbed.clear()
    monkeypatch.setattr(
        mss,
        "read_first_cached_dict",
        lambda *_a, **_kw: ({"ts": True, "market_bias": "BULLISH", "recommended_action": "ACCUMULATE"}, "s.json"),
    )
    llm_order_guard.check_geopolitical_order_guards(
        "BTCUSDC",
        "BUY",
        margins={"intelligence_guards_enabled": True, "geopolitical_guard_mode": "enforce"},
        now=now_ts,
    )
    assert seen_absorbed == [False]

    # 5. MarketRegimeService._age_limit, _evidence & evaluate_closes border cases
    svc = MarketRegimeService()
    age_cfg = {1: 120.0, "short": 600.0}
    for bad_iv in (float("nan"), 1.5, True, "bad", -1):
        assert svc._age_limit(age_cfg, MarketRegimeHorizon.SHORT, bad_iv) == 600.0
    assert svc._age_limit(age_cfg, MarketRegimeHorizon.SHORT, 1) == 120.0
    ev = svc._evidence(
        SimpleNamespace(name="kraken"),
        "HYPEUSD",
        svc.evaluator.unknown("test"),
        role="asset_short",
        horizon=MarketRegimeHorizon.SHORT,
        evaluated_at=1700000000.0,
        interval_min=True,  # type: ignore[arg-type]
    )
    assert ev.interval_min is None
    assert svc.evaluate_closes([10.0, 0.0, 12.0]).reason == "non_finite_ohlc"
    assert svc.evaluate_closes([10.0, float("nan"), 12.0]).reason == "non_finite_ohlc"

    # 6. TrackedOrderLifecycle.new_intent clean ValueError on corrupt/bool/<=0 requested_qty & requested_price
    lc = TrackedOrderLifecycle(SimpleNamespace(), provider_name="binance")
    for bad_q in (None, "bad", True, 0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="requested_qty must be positive"):
            lc.new_intent(
                intent_id="i1", client_order_id="c1", symbol="BTCUSDC", side="BUY", requested_qty=bad_q  # type: ignore[arg-type]
            )
    for bad_px in ("bad", True, 0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="requested_price must be positive"):
            lc.new_intent(
                intent_id="i1",
                client_order_id="c1",
                symbol="BTCUSDC",
                side="BUY",
                requested_qty=1.0,
                requested_price=bad_px,  # type: ignore[arg-type]
            )

    # 7. order_retry._process_start_identity & _client_order_id border cases
    for bad_pid in (True, False, 0, -5, 1.5, "bad", None):
        with pytest.raises(ValueError, match="positive integer"):
            order_retry._process_start_identity(bad_pid)
    for bad_rev in (None, True, "bad", -3, 2.7):
        assert order_retry._client_order_id("abcdef1234567890", bad_rev).endswith("_0")


def test_venue_fee_precision_ssot_and_runtime_border_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify polymorphic venue fee/precision SSOT and runtime border-case hardening."""

    # 1. Polymorphic venue fee & step/tick precision SSOT on MarketDataProvider
    assert MarketDataProvider.resolve_venue_default_fee_spec("binance") == ("VIP0", 10.0, 10.0, 5.0)
    assert MarketDataProvider.resolve_venue_default_fee_spec("kraken") == ("PRO_TIER0", 16.0, 26.0, 1.0)
    assert MarketDataProvider.resolve_venue_default_fee_spec("hyperliquid") == ("SPOT_TIER0", 4.0, 7.0, 10.0)
    assert MarketDataProvider.resolve_venue_default_fee_spec("t212") == ("INVEST_RTH", 15.0, 15.0, 1.0)
    assert MarketDataProvider.resolve_venue_default_fee_spec("unknown_venue") == ("STANDARD", 15.0, 15.0, 5.0)

    tier, maker, taker, min_notional, step_sz, tick_sz, prefer_inst = (
        MarketDataProvider.resolve_venue_default_fee_and_precision("hyperliquid", 2500.0)
    )
    assert (tier, maker, taker, min_notional, prefer_inst) == ("SPOT_TIER0", 4.0, 7.0, 10.0, True)
    assert step_sz > 0 and tick_sz > 0

    _, _, _, _, t212_step, t212_tick, t212_prefer = (
        MarketDataProvider.resolve_venue_default_fee_and_precision("t212", 150.0)
    )
    assert (t212_step, t212_tick, t212_prefer) == (0.001, 0.01, False)

    # VenueFeeAndLimitsEvaluator prefers instrument tick_size on hyperliquid vs live tick_size on kraken
    monkeypatch.setattr(
        MarketDataProvider,
        "resolve_instrument_cached_fee_and_limits",
        classmethod(
            lambda _cls, _inst, v_norm, _sym, *, price=0.0: {
                "tick_size": 0.001 if v_norm == "hyperliquid" else 0.01,
                "step_size": 0.001,
                "maker_fee_bps": None,
                "taker_fee_bps": "bad",
            }
        ),
    )
    ev = VenueFeeAndLimitsEvaluator()
    hl_out = ev.evaluate(
        "hyperliquid",
        "ETHUSDC",
        SimpleNamespace(params={"tick_size": 0.1, "step_size": 0.01}),
        price=2500.0,
    )
    kr_out = ev.evaluate(
        "kraken",
        "SOLUSDC",
        SimpleNamespace(params={"tick_size": 0.05, "step_size": 0.01}),
        price=150.0,
    )
    assert hl_out["tick_size"] == pytest.approx(0.1)
    assert hl_out["maker_fee_bps"] == pytest.approx(4.0)
    assert kr_out["tick_size"] == pytest.approx(0.01)

    # 2. account_cache_health._pid_is_alive TOCTOU vanish & _pid_started_at_ms invalid SC_CLK_TCK / btime
    monkeypatch.setattr(health.os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(health.os.path, "isdir", lambda p: True if p == "/proc" else False)

    def _raise_fnf(_path: str, **_kw):
        raise FileNotFoundError("vanished after os.kill")

    monkeypatch.setattr(health, "read_text_file", _raise_fnf)
    assert health._pid_is_alive(42424) is False

    fake_stat = "42424 (cacheManager) S 1 42424 42424 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 987654 0 0"
    monkeypatch.setattr(health, "read_text_file", lambda _p, **_kw: fake_stat)
    monkeypatch.setattr(health, "read_text_lines", lambda _p, **_kw: ["btime ", "btime -5", "btime 1700000000"])
    monkeypatch.setattr(health.os, "sysconf", lambda _name: 0)
    assert health._pid_started_at_ms(42424) is None
    monkeypatch.setattr(health.os, "sysconf", lambda _name: 100)
    assert health._pid_started_at_ms(42424) == int((1700000000 + 987654 / 100) * 1000)

    # 3. RTradePairStore.mutate tolerates corrupt/ISO updated_ts on evicted terminal records
    store = RTradePairStore(terminal_retention=1)
    now_ts = 1700000100.0

    def _seed(pairs: dict) -> bool:
        pairs["old_iso"] = {
            "symbol": "BTCUSDC",
            "pair_id": "old_iso",
            "start_side": "BUY",
            "qty": 0.01,
            "phase": "closed",
            "terminal": True,
            "intents": {},
            "state": None,
            "created_ts": now_ts - 50.0,
            "updated_ts": "2020-01-01T00:00:00Z",
        }
        pairs["newest"] = {
            "symbol": "BTCUSDC",
            "pair_id": "newest",
            "start_side": "BUY",
            "qty": 0.01,
            "phase": "closed",
            "terminal": True,
            "intents": {},
            "state": None,
            "created_ts": now_ts,
            "updated_ts": now_ts,
        }
        return True

    monkeypatch.setattr(store, "_read", lambda: {"version": 1, "pairs": {}})
    store.mutate(_seed)
    assert set(store._mem_data["pairs"]) == {"newest"}

    # 4. rtrade._LivePairVenue._ticket_from_pending rejects missing/zero/NaN qty cleanly
    venue = object.__new__(rtrade._LivePairVenue)
    venue.current_price = lambda: 50000.0
    for bad_qty in (None, 0.0, -1.0, float("nan"), "bad"):
        with pytest.raises(rtrade._IntentRecoveryPending, match="no valid quantity"):
            venue._ticket_from_pending(
                {
                    "order_id": "oid-1",
                    "side": "BUY",
                    "kind": "limit",
                    "submitted_price": 50000.0,
                    "submitted_qty": bad_qty,
                    "requested_qty": None,
                },
                "pair-1",
            )

    # 5. ParabolicSurgeGuard survives None/corrupt disarmed_peak_ts & ts in state dict
    pg = ParabolicSurgeGuard()
    pg._states["BTCUSDC"] = {
        "peak": 90000.0,
        "surge_active": True,
        "surge_start_px": 85000.0,
        "ts": None,
        "disarmed_peak_ts": None,
    }
    dec = pg.check(
        "BTCUSDC",
        "BUY",
        current_price=88000.0,
        price_history=[(1700000000.0, 87000.0), (1700000060.0, 88000.0)],
        now=1700000060.0,
    )
    assert dec.allowed is True


def test_portfolio_lot_reconciliation_ssot_and_cache_coercion_border_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify shared open-lot pruning/VWAP SSOT and timestamp/margin coercion border cases."""

    # 1. Shared select_fresh_open_lots + reconcile_fresh_lots_vwap SSOT & corrupt ts_ms handling
    now_ms = 1_750_000_000_000
    day_ms = 86_400_000
    raw_lots = [
        {"qty": 1.0, "entry_px": 50.0, "ts_ms": now_ms - 60 * day_ms},
        {"qty": 0.5, "entry_px": 80.0, "ts_ms": float("nan")},
        {"qty": 0.5, "price": 90.0, "ts_ms": now_ms - 20 * day_ms},
        {"qty": 0.5, "entry_px": 100.0, "ts_ms": now_ms - 2 * day_ms},
    ]
    fresh, newest_ts = select_fresh_open_lots(raw_lots, held_qty=1.5)
    assert newest_ts == now_ms - 2 * day_ms
    assert len(fresh) == 3
    assert [lt["entry_px"] for lt in fresh] == [80.0, 90.0, 100.0]

    ro_state = {"active_cycle_rem_qty": 2.5, "active_avg_buy_px": 72.0, "tactical_avg_buy_px": 90.0}
    vwap, fq = reconcile_fresh_lots_vwap(ro_state, raw_lots, fresh, 1.5)
    assert fq == pytest.approx(1.5)
    assert vwap == pytest.approx(90.0)
    assert ro_state["active_avg_buy_px"] == pytest.approx(90.0)
    assert "tactical_avg_buy_px" not in ro_state

    # 2. Cache365PriceManager._merge_max_nonneg_int_map handles corrupt/float/bool values safely
    dst_ft = {"BTCUSDC": "1700000000100", "ETHUSDC": True, "SOLUSDC": None}
    Cache365PriceManager._merge_max_nonneg_int_map(
        dst_ft,
        {"BTCUSDC": 1700000000250.0, "ETHUSDC": float("nan"), "SOLUSDC": -50, "ADAUSDC": "42"},
    )
    assert dst_ft == {
        "BTCUSDC": 1700000000250,
        "ETHUSDC": 0,
        "SOLUSDC": 0,
        "ADAUSDC": 42,
    }

    # 3. guard_hierarchy._load_margins survives NaN / corrupt mtime or last_check on order_guard
    monkeypatch.setattr(order_guard, "_MARGINS_FILE_MTIME", float("nan"), raising=False)
    monkeypatch.setattr(order_guard, "_MARGINS_LAST_CHECK", "corrupt_ts", raising=False)
    margins = gh._load_margins()
    assert isinstance(margins, dict)

    # 4. assetguardian._price_row normalizes ms tuples and rejects bool / non-positive entries
    assert ag._price_row([1_700_000_000_500, "42000.5"]) == pytest.approx((1_700_000_000.5, 42000.5))
    assert ag._price_row([True, 42000.5]) is None
    assert ag._price_row([1_700_000_000, 0.0]) is None


def test_mt_conf_fail_fast_ohlc_and_cycle_history_border_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fail-fast config/OHLC bounds, commonpath cycle-history gating, and microstructure delta hardening."""

    # 1. monitortrades._load_mt_conf rejects NaN, Inf, and negative config values fail-fast
    repo_conf_text = Path(mt.__file__).with_name("monitortrades.conf").read_text(encoding="utf-8")
    try:
        for bad_override in (
            "hard_tp_pct = inf",
            "hard_tp_pct = nan",
            "hard_tp_cooldown_h = -1",
            "sl_trend_defer_max_s = nan",
            "sl_bounce_ratchet_k = -0.1",
            "sl_disaster_vol_multiplier = inf",
            "tp_ratchet_k = nan",
            "tp_stagnation_max_s = -5",
            "reentry_post_stop_cooldown_s = -1",
        ):
            bad_conf = tmp_path / "bad_monitortrades.conf"
            bad_conf.write_text(f"{repo_conf_text}\n{bad_override}\n", encoding="utf-8")
            with pytest.raises(ValueError):
                mt._load_mt_conf(str(bad_conf))
    finally:
        mt._load_mt_conf()

    # 2. validate_dataset & drop_incomplete_last_bar reject NaN/Inf/bool/fractional interval & timestamps
    valid_row = {"timestamp": 3600, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0}
    for bad_iv in (0, -60, True, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="interval_minutes"):
            validate_dataset([valid_row], interval_minutes=bad_iv)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="interval_minutes"):
            drop_incomplete_last_bar([valid_row], interval_minutes=bad_iv, now_timestamp=7200)  # type: ignore[arg-type]
    for bad_now in (None, True, 7200.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="now_timestamp"):
            drop_incomplete_last_bar([valid_row], interval_minutes=60, now_timestamp=bad_now)  # type: ignore[arg-type]
    for bad_bound in (True, 10.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="regular session bounds"):
            drop_incomplete_last_bar(
                [valid_row],
                interval_minutes=1440,
                now_timestamp=4000,
                regular_session_start=bad_bound,  # type: ignore[arg-type]
                regular_session_end=8000,
            )

    # 3. load/save_microstructure_cycle_history uses commonpath (does not confuse cachedb_alt prefix with repo cachedb)
    monkeypatch.setattr(mss, "is_live_network_backfill_enabled", lambda: False)
    alt_dir = tmp_path / "cachedb_isolated"
    alt_dir.mkdir()
    assert mss.save_microstructure_cycle_history(str(alt_dir), {"BTCUSDC": [{"ts": 1700000000.0, "oi_usd": 1000.0}]}) is True
    loaded_hist = mss.load_microstructure_cycle_history(str(alt_dir))
    assert "BTCUSDC" in loaded_hist
    assert mss.save_microstructure_cycle_history(str(mss._REPO_CACHEDB_DIR), {"BTCUSDC": []}) is False
    assert mss.load_microstructure_cycle_history(str(mss._REPO_CACHEDB_DIR)) == {}

    # 4. evaluate_microstructure_cycle_deltas safely evaluates rolling window and ignores malformed history entries
    micro = {"BTCUSDC": {"ob_imbalance": 0.25, "cvd_15m_usd": 5000.0, "oi_usd": 1_200_000.0}}
    hist = {
        "BTCUSDC": [
            "corrupt_entry",
            {"ts": "bad_ts", "ob_imbalance": 0.1},
            {"ts": 1700000000.0, "ob_imbalance": 0.10, "cvd_15m_usd": 2000.0, "oi_usd": 1_000_000.0},
        ]
    }
    out_micro, out_hist, dirty = evaluate_microstructure_cycle_deltas(micro, hist, now_ts=1700000400.0)
    assert dirty is True
    cd = out_micro["BTCUSDC"]["cycle_delta"]
    assert cd["ob_imbalance_delta"] == pytest.approx(0.15)
    assert cd["cvd_15m_delta_usd"] == pytest.approx(3000.0)
    assert cd["oi_delta_pct"] == pytest.approx(20.0)
    assert len(out_hist["BTCUSDC"]) == 2

    # 5. _tracked_equity_base_tickers returns deduplicated uppercase equity bases
    eq_bases = _tracked_equity_base_tickers()
    assert isinstance(eq_bases, list)
    assert len(eq_bases) == len(set(eq_bases))


def test_auto_ws_ssot_tp_ladder_checkpoint_and_estimator_border_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SSOT auto-WS flag, empty-comma TP ladder rejection, EET checkpoint bounds, and estimator validation."""

    # 1. botcore.is_auto_start_websockets_enabled handles all falsy tokens ("n", "f", "0", "off") & env overrides
    monkeypatch.setenv("BINANCE_AUTO_START_WEBSOCKETS", "n")
    assert botcore.is_auto_start_websockets_enabled() is False
    assert botcore.is_auto_start_websockets_enabled(env={"BINANCE_AUTO_START_WEBSOCKETS": "1"}) is True
    monkeypatch.setenv("BINANCE_AUTO_START_WEBSOCKETS", "yes")
    assert botcore.is_auto_start_websockets_enabled() is True

    # 2. parse_tp_ladder_env rejects comma-only STRAT_TP_LADDER=" , " instead of silently returning []
    with pytest.raises(ValueError, match="Invalid STRAT_TP_LADDER"):
        parse_tp_ladder_env({"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER": " , , "}, lambda *_a, **_kw: [])
    assert parse_tp_ladder_env(
        {"STRATEGY_MODE": "avg_tp", "STRAT_TP_LADDER": "3:40, 6:60"},
        lambda *_a, **_kw: [],
    ) == [(3.0, 0.4), (6.0, 0.6)]

    # 3. calendar_ssot.did_cross_session_checkpoint ignores out-of-range hour/minute entries without crashing
    monkeypatch.setattr(
        mss,
        "_cfg_str",
        lambda sec, key, *a, **kw: "25:00:BAD_HOUR, 12:99:BAD_MIN, 09:00:EU_MORNING_BUY_OPEN"
        if (sec, key) == ("sessions", "checkpoints_eet")
        else "Europe/Bucharest",
    )
    # Cross 09:00 EET on a weekday (2026-01-12 06:00 UTC -> 08:00 UTC = 08:00 EET -> 10:00 EET)
    t_before = 1768197600.0
    t_after = 1768204800.0
    assert calendar_ssot.did_cross_session_checkpoint(t_before, t_after, min_cooldown_sec=60.0) == "EU_MORNING_BUY_OPEN"

    # 4. pricewindow._normalize_short_trend_estimator rejects unsupported names and normalizes empty to None
    assert pw._normalize_short_trend_estimator("  ") is None
    assert pw._normalize_short_trend_estimator("STEP_WOLS") == "step_wols"
    with pytest.raises(ValueError, match="Unsupported short trend estimator"):
        pw._normalize_short_trend_estimator("invalid_estimator")

    # 5. os_resources_watchdog._parse_proc_stat rejects non-integer / negative tick fields cleanly
    bad_stat = "123 (python3) S 1 123 123 0 -1 4194304 0 0 0 0 -5 10 0 0 20 0 1 0 1000 0 100"
    with pytest.raises(ValueError, match="utime"):
        wd._parse_proc_stat(bad_stat)


def test_hoisted_patch_seams_symbol_csv_and_observe_cli_border_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Verify hoisted module-level patch seams, SSOT CSV/symbol parsing, and tradeall_observe CLI bounds."""

    # 1. microstructure_evaluator.enrich_cross_venue_trends honors survival.estimate_T monkeypatches
    called_est: list[str] = []
    monkeypatch.setattr(
        surv_mod,
        "estimate_T",
        lambda sym: called_est.append(sym) or {"T": 10.0, "median_d": 4.0, "p90_d": 9.0},
    )
    price_map = {"BTCUSDC": {"price": 70000.0}}
    t_map: dict[str, Any] = {}
    me_mod.enrich_cross_venue_trends(
        price_map,
        {},
        {},
        {"BTCUSDC": {"direction": "UP", "duration_seconds": 86400.0}},
        {},
        t_map,
        None,
        allow_network=True,
    )
    assert called_est == ["BTCUSDC"]
    assert price_map["BTCUSDC"]["weibull_median_d"] == 4.0
    assert price_map["BTCUSDC"]["long_trend_est_rem_hours"] > 0.0

    # 2. order_guard.resolve_caller_module honors providers.order_id_context.get_process_name monkeypatches
    monkeypatch.setattr(oic, "get_process_name", lambda: "TA")
    assert order_guard.resolve_caller_module() == "tradeall"
    monkeypatch.setattr(oic, "get_process_name", lambda: "ORW")
    assert order_guard.resolve_caller_module() is None

    # 3. botcore.parse_symbol_list, StockYahooPricePlatform, and NewCoinsFactory deduplicate comma/newline lists
    assert botcore.parse_symbol_list(" nvda , tsla\nNVDA , ") == ["NVDA", "TSLA"]
    yahoo_pf = StockYahooPricePlatform(tickers=" nvda , aapl\nNVDA ")
    assert yahoo_pf._symbols == {"NVDA", "AAPL"}
    factory = NewCoinsFactory(enabled_sources=" CoinGecko , DEXSCREENER\ncoingecko ", cmc_api_key="")
    assert list(factory._all_sources.keys()) == ["coingecko", "dexscreener"]

    # 4. tradeall_observe.main rejects non-positive / NaN numeric CLI flags (--frame-hours 0 must not enter live loop)
    for bad_argv in (
        ["tradeall_observe.py", "--backtest-dir", str(tmp_path), "--frame-hours", "0"],
        ["tradeall_observe.py", "--backtest-dir", str(tmp_path), "--window-hours", "-2"],
        ["tradeall_observe.py", "--interval", "0"],
        ["tradeall_observe.py", "--live-minutes", "nan"],
    ):
        monkeypatch.setattr(sys, "argv", bad_argv)
        with pytest.raises((ValueError, SystemExit)):
            tobs.main()


def test_botcore_reexport_sync_t212_check_market_and_shadow_env_border_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Verify botcore reexport sync, state_io patch seams, T212 check_market NaN/ms guards, and shadow env paths."""

    # 1. Instrument._apply_outbox_lineage_prefix honors providers.order_id_context.get_process_name monkeypatches
    monkeypatch.setattr(oic, "get_process_name", lambda: "TA")
    target_kw: dict[str, Any] = {}
    Instrument._apply_outbox_lineage_prefix(target_kw, {})
    assert target_kw == {"prefix": "TA_OR"}

    # 2. botcore.touch_heartbeat_file honors state_io.atomic_write_json monkeypatch & CSV hb_log
    captured_hb: list[tuple[str, Any]] = []
    hb_file = tmp_path / "unit.heartbeat"
    hb_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(state_io, "atomic_write_json", lambda p, payload: captured_hb.append((str(p), payload)))
    botcore.touch_heartbeat_file(hb_file, force=True, now=100.0, status={"ok": True})
    assert len(captured_hb) == 1 and captured_hb[0][0] == str(hb_file)

    hb_paths, _ = botcore.resolve_heartbeat_paths(
        str(tmp_path),
        str(tmp_path),
        "demo",
        "demo.py",
        "logs/a.log, , logs/b.log",
        60.0,
    )
    assert hb_paths == [str(tmp_path / "logs/a.log"), str(tmp_path / "logs/b.log")]

    # 3. botcore._sync_reexport_modules preserves cross-module function identity across dict rollback
    saved_dict = dict(botcore.__dict__)
    try:
        importlib.reload(botcore)
        assert bt_eval.load_layered_env_config is botcore.load_layered_env_config
    finally:
        botcore.__dict__.update(saved_dict)
        botcore._sync_reexport_modules()
    assert bt_eval.load_layered_env_config is botcore.load_layered_env_config

    # 4. 212trading/market_data.py::check_market guards NaN/None bar volumes, closes, and ms timestamps
    t212_md = botcore.load_local_module(
        Path(__file__).resolve().parents[1] / "212trading",
        "market_data",
        module_key="t212_md_border_test",
        register=False,
    )
    now_s = time.time()
    monkeypatch.setattr(
        t212_md,
        "_chart",
        lambda _sym: (
            {
                "regularMarketPrice": 25.5,
                "regularMarketVolume": 0,
                "marketState": "REGULAR",
                "regularMarketTime": int(now_s * 1000),
            },
            [
                (int((now_s - 300) * 1000), None, float("nan")),
                (int((now_s - 180) * 1000), 25.0, 12.0),
                (int((now_s - 60) * 1000), float("nan"), -5.0),
            ],
        ),
    )
    info = t212_md.check_market("SPCX")
    assert info is not None
    assert info["price"] == 25.5
    assert info["volume"] == 12.0 and not math.isnan(info["volume"])
    assert info["launched"] is True and info["trading"] is True

    # 5. kraken/shadow_live._resolve_runtime_paths ignores empty ENV_FILE & monitor_night splits CSV hb_log
    monkeypatch.setenv("ENV_FILE", "   ")
    resolved_env, resolved_cfg = kraken_shadow._resolve_runtime_paths()
    assert resolved_env == kraken_shadow.DEFAULT_ENV
    assert resolved_cfg.endswith("kraken/config.env")

    monkeypatch.setattr(monitor_night, "ROOT", "/srv/mptrade")
    monkeypatch.setattr(
        monitor_night,
        "run",
        lambda _c, _cmd: "bot.py | $ROOT/kraken | python3 bot.py | kbot | cachedb/system/heartbeat/k.heartbeat, logs/kbot.log | 120 | bot\n",
    )
    procs = monitor_night.load_procs(object())
    assert len(procs) == 1 and procs[0].log == "logs/kbot.log"


def test_ws_normalize_headline_age_apply_proposals_and_csv_border_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Verify WS trade ID float-string guards, headline age SSOT, branch validation, and CSV trimming."""

    # 1. bapi_ws_normalize rejects positive non-integer float string trade IDs ("12.5") while preserving 64-bit & alphanumeric IDs
    base_trade = {"s": "BTCUSDC", "x": "TRADE", "S": "BUY", "i": "42", "L": "0", "Y": "5000.0", "l": "0.1"}
    assert ws_norm.parse_trade_execution_report({**base_trade, "t": "12.5"}) is None
    assert ws_norm.parse_trade_execution_report({**base_trade, "t": "-5.5"}) is None
    ok_big = ws_norm.parse_trade_execution_report({**base_trade, "t": "9007199254740993.0"})
    assert ok_big is not None and ok_big[1]["id"] == "9007199254740993"
    assert float(ok_big[1]["price"]) == pytest.approx(50000.0)
    ok_alpha = ws_norm.parse_trade_execution_report({**base_trade, "t": "TRD-77"})
    assert ok_alpha is not None and ok_alpha[1]["id"] == "TRD-77"
    ok_ord = ws_norm.parse_order_execution_report({**base_trade, "X": "FILLED", "t": "9007199254740993.0"})
    assert ok_ord is not None and ok_ord[1]["_fillId"] == "9007199254740993"

    # 2. parse_headline_age_minutes, headline_effective_ts, and _headline_age_suffix guard bool/NaN/out-of-range age_min
    for bad_age in (True, False, float("nan"), float("inf"), -1.0, 5000.0, "bad"):
        assert parse_headline_age_minutes({"age_min": bad_age}) is None
        assert headline_effective_ts({"age_min": bad_age}, 10_000.0) == 0.0
        assert GeopoliticalThreatAnalyzer._headline_age_suffix({"age_min": bad_age}, 10_000.0) == ""
    assert parse_headline_age_minutes({"age_min": "15.4"}) == pytest.approx(15.4)
    assert headline_effective_ts({"age_min": "15"}, 10_000.0) == pytest.approx(9_100.0)
    assert GeopoliticalThreatAnalyzer._headline_age_suffix({"age_min": "15.4"}, 10_000.0) == ", age=15m"

    # 3. apply_proposals._read_proposals rejects whitespace, dot-dot, flag-like, and non-string branch names
    for bad_branch in ("", "  ", "-evil", "a..b", "main evil", "a\nb", True):
        with pytest.raises(RuntimeError, match="Invalid BACKTEST_PROPOSALS_BRANCH"):
            apply_proposals._read_proposals(bad_branch)  # type: ignore[arg-type]

    # 4. short_trend_estimator_benchmark.build_candidates trims whitespace around BENCH_ONLY CSV entries
    monkeypatch.setenv("BENCH_ONLY", " SW p1 , SW p2 ")
    px_arr = np.linspace(100.0, 102.0, 400)
    cand_names = [name for name, _sig in st_bench.build_candidates(px_arr)]
    assert cand_names == ["LEGACY (sf+gr)/2", "SW p1", "SW p2"]

    cfg_file = tmp_path / "sample.conf"
    cfg_file.write_text("[SEC]\n# BACKTEST: 1.0, , 2.5 , 4.0 # inline\nmt.gain = 2.5\n", encoding="utf-8")
    assert backtest_ranges.scan_backtest_ranges(str(cfg_file)) == {"SEC.mt.gain": ["1.0", "2.5", "4.0"]}


def test_expired_orders_sync_mode_trailing_status_and_runtime_flag_border_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify Wave 8 border cases across iter_expired_open_order_ids, SyncModeMixin, equity context bars, trailing logs, and RuntimeServices."""

    # 1. iter_expired_open_order_ids rejects non-positive/bool/NaN expire_sec and normalizes ms current_time_sec
    orders = {
        "ord_old": {"time": 1_700_000_000_000},
        "ord_fresh": {"time": 1_700_000_090_000},
    }
    for bad_exp in (0, 0.0, -5.0, False, True, float("nan"), float("inf"), None):
        assert iter_expired_open_order_ids(orders, bad_exp, current_time_sec=1_700_000_100.0) == []  # type: ignore[arg-type]
    for bad_now in (0, -10.0, False, True, float("nan"), None):
        assert iter_expired_open_order_ids(orders, 30.0, current_time_sec=bad_now) == []  # type: ignore[arg-type]
    assert iter_expired_open_order_ids(orders, 30.0, current_time_sec=1_700_000_100_000) == ["ord_old"]

    # 2. SyncModeMixin string boolean flags ("false"/"0") and non-timestamp TTL durations (>= 1e11)
    dummy_llm = DummyGeminiClient({"market_bias": "NEUTRAL"})
    adv = SentimentAdvisor(
        gemini_client=dummy_llm,  # type: ignore[arg-type]
        cache_ttl_sec=2e11,
        cache_file=str(tmp_path / "sentiment_eval.json"),
        start_sync="false",  # type: ignore[arg-type]
    )
    assert adv.start_sync is False
    assert adv.cache_ttl_sec == pytest.approx(2e11)
    assert adv.review(now_ts=1_700_000_000.0) is None
    assert dummy_llm.call_count == 0

    adv_default = SentimentAdvisor(
        gemini_client=dummy_llm,  # type: ignore[arg-type]
        cache_file=str(tmp_path / "sentiment_eval2.json"),
    )
    assert adv_default.review(now_ts=1_700_000_000.0, allow_llm="0") is None  # type: ignore[arg-type]
    assert dummy_llm.call_count == 0

    http_col = DummyCachedHttpJsonCollector(
        2e11, 60.0, "https://example.invalid/feed", lambda: "", start_sync="false"
    )  # type: ignore[arg-type]
    assert http_col.start_sync is False
    assert http_col.cache_ttl_sec == pytest.approx(2e11)
    assert http_col.fetch(now_ts=1_700_000_000.0) is None

    news_col = NewsFeedCollector(history_file=str(tmp_path / "news_collect.json"), cache_ttl_sec=2e11, start_sync="0")  # type: ignore[arg-type]
    assert news_col.start_sync is False
    assert news_col.cache_ttl_sec == pytest.approx(2e11)

    bybit_col = BybitLiquidationCollector(symbols=["BTCUSDT"], start_sync="false")  # type: ignore[arg-type]
    assert bybit_col.start_sync is False
    assert resolve_llm_guard_cache_ttl(2e11, False, None, 600.0) == pytest.approx(2e11)

    # 3. _resolve_equity_context_bars_and_price filters malformed bars and NaN/None close prices
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path), start_sync=False)
    monkeypatch.setattr(
        ob_col,
        "_fetch_equity_market_context",
        lambda *_a, **_kw: {"last_price": 0.0, "bars": [[1, 2, 3], (1.0, 2.0, 3.0, 1.5, float("nan"), 10.0)]},
    )
    assert ob_col._resolve_equity_context_bars_and_price("AAPL_US_EQ", 1_700_000_000.0) is None
    res_no_px = ob_col._resolve_equity_context_bars_and_price(
        "AAPL_US_EQ", 1_700_000_000.0, require_positive_price=False
    )
    assert res_no_px is not None and res_no_px[2] == 0.0 and len(res_no_px[1]) == 1

    # 4. TrailingAdapterBase._extract_order_result_status and KrakenTrailing log helpers handle None intent/price
    assert TrailingAdapterBase._extract_order_result_status(None) == ("PENDING", "orderId=None")
    assert TrailingAdapterBase._extract_order_result_status(SimpleNamespace(outcome="refused", intent=None)) == (
        "REFUSED",
        "None",
    )
    logged: list[str] = []
    kt = object.__new__(KrakenTrailing)
    kt.log = logged.append  # type: ignore[assignment]
    kt._log_sell_result(
        SimpleNamespace(outcome="refused", intent=None),
        asset="HYPE",
        pair="HYPEUSD",
        qty=1.5,
        price=None,
        order_price=None,
        peak=None,
        trail=15.0,
        use_market=False,
    )
    kt._log_rebuy_result(
        SimpleNamespace(outcome="pending", intent=None),
        asset="HYPE",
        pair="HYPEUSD",
        qty=1.5,
        price_ref=0.0,
        order_price=None,
    )
    assert len(logged) == 2 and "@ ~0.0000" in logged[0] and "@ ~0.0000" in logged[1]

    # 5. RuntimeServices & configure_runtime_services coerce string booleans and non-finite retry delays
    rs = botcore.RuntimeServices(
        stale_cache_retry_delay_sec=float("nan"),
        enforce_rth_gate="false",  # type: ignore[arg-type]
        allow_unmocked_llm_queries="0",  # type: ignore[arg-type]
    )
    assert rs.stale_cache_retry_delay_sec == pytest.approx(1.5)
    assert rs.enforce_rth_gate is False
    assert rs.allow_unmocked_llm_queries is False
    saved_unmocked = botcore._RUNTIME_SERVICES.allow_unmocked_llm_queries
    try:
        botcore.configure_runtime_services(allow_unmocked_llm_queries="false")
        assert botcore._RUNTIME_SERVICES.allow_unmocked_llm_queries is False
    finally:
        botcore.configure_runtime_services(allow_unmocked_llm_queries=saved_unmocked)


def test_stop_loss_avg_archive_seed_replay_summary_fg_and_guard_preflight_border_cases(
    tmp_path, monkeypatch
):
    """Regression suite for stop-loss zero-price guard, 365d seed bounds, replay summary, F&G clamping, and guard/collector SSOT."""

    # 1. BaseSpotMixin._resolve_stop_loss_avg_and_loss_pct never triggers +100% SL on 0/None/bool/NaN price
    dummy = object.__new__(BaseSpotMixin)
    dummy.p = SimpleNamespace(stop_loss_pct=10.0)
    dummy._avg = lambda: 100.0  # type: ignore[method-assign]
    for bad_px in (0.0, -5.0, None, False, True, float("nan"), float("inf")):
        assert dummy._resolve_stop_loss_avg_and_loss_pct(bad_px) is None  # type: ignore[arg-type]
    avg_long, loss_long = dummy._resolve_stop_loss_avg_and_loss_pct(92.0, sign=1) or (0.0, 0.0)
    assert avg_long == pytest.approx(100.0) and loss_long == pytest.approx(8.0)
    avg_short, loss_short = dummy._resolve_stop_loss_avg_and_loss_pct(108.0, sign=-1) or (0.0, 0.0)
    assert avg_short == pytest.approx(100.0) and loss_short == pytest.approx(8.0)
    dummy.p = SimpleNamespace(stop_loss_pct=True)
    assert dummy._resolve_stop_loss_avg_and_loss_pct(92.0) is None

    # 2. Cache365PriceManager.seed_initial_archive_if_missing filters malformed points and normalizes timestamps
    seed_target = str(tmp_path / "cache_365_price_BTCUSDC.jsonl")
    assert Cache365PriceManager.seed_initial_archive_if_missing(seed_target, "BTCUSDC", 123) is False  # type: ignore[arg-type]
    assert (
        Cache365PriceManager.seed_initial_archive_if_missing(
            seed_target,
            "BTCUSDC",
            [
                [None, 100.0],
                [True, 100.0],
                [float("nan"), 100.0],
                [1700000000000, -1.0],
                [1700000000000, float("nan")],
                [1700000000.0, 42123.5],
            ],
        )
        is True
    )
    seeded_rows = read_jsonl_dicts(seed_target)
    assert seeded_rows == [{"s": "BTCUSDC", "i": [1700000000000, 42123.5]}]

    # 3. summarize_grid_replay_metrics handles None metrics, bool bh_digits, string include_fills, and negative p1
    grid_sum = summarize_grid_replay_metrics(
        None,  # type: ignore[arg-type]
        budget=1000.0,
        p0=100.0,
        p1=112.3456,
        bh_digits=True,  # type: ignore[arg-type]
        include_fills="false",  # type: ignore[arg-type]
    )
    assert "fills" not in grid_sum and grid_sum["bh"] == pytest.approx(12.346)
    grid_neg_p1 = summarize_grid_replay_metrics(
        {"total": 25.0, "fills": 4},
        budget=1000.0,
        p0=100.0,
        p1=-10.0,
        include_fills="true",  # type: ignore[arg-type]
    )
    assert grid_neg_p1["bh"] == 0.0 and grid_neg_p1["fills"] == 4

    # 4. FearGreedSnapshot.extract_value_and_label & FearGreedCollector.parse_payload reject -1 sentinel & clamp >100
    assert FearGreedSnapshot.extract_value_and_label({"value": -1, "classification": "Error"}) is None
    assert FearGreedSnapshot.extract_value_and_label(
        {"value": 105.4, "classification": "Euphoria"}, preserve_float="false"  # type: ignore[arg-type]
    ) == (100, "Euphoria")
    assert FearGreedCollector.parse_payload({"data": [{"value": "-1"}]}) is None
    fg_clamped = FearGreedCollector.parse_payload({"data": [{"value": "104", "value_classification": "Extreme Greed"}]})
    assert fg_clamped is not None and fg_clamped.value == 100 and fg_clamped.is_extreme_greed is True

    # 5. MarketDataProvider._resolve_live_order_flag, parse_intervals, _accepts_two_positional_args, guard/collector SSOT
    dummy_prov = SimpleNamespace(execution_enabled=lambda: True)
    assert MarketDataProvider._resolve_live_order_flag(dummy_prov, None) is True  # type: ignore[arg-type]
    assert MarketDataProvider._resolve_live_order_flag(dummy_prov, {"live_enabled": "false"}) is False  # type: ignore[arg-type]
    assert parse_intervals(" 5 , 15 , 60 ") == [5, 15, 60]

    bm = SimpleNamespace()
    one_arg = MethodType(lambda self, sig: sig, bm)
    two_args = MethodType(lambda self, sig, sym: (sig, sym), bm)
    assert _accepts_two_positional_args(one_arg) is False
    assert _accepts_two_positional_args(two_args) is True
    assert _accepts_two_positional_args(functools.partial(two_args, "prebound")) is False

    early_res, norm_side, _, _ = preflight_intelligence_guard_entry(
        "SELL",
        guard_code="GLLM3",
        guard_bypass_reason="ok",
        allowed_sides=("BUY",),
        unsupported_side_reason="not_buy_side",
    )
    assert early_res == (True, "not_buy_side", 1.0) and norm_side == "SELL"
    monkeypatch.setattr(eog, "_load_margins", lambda: {"intelligence_guards_mode": "off"})
    assert eog.check_external_order_guards("BTCUSDC", "BUY", 85000.0) == (
        True,
        "intelligence_guards_off",
        1.0,
    )

    read_calls: list[str] = []
    intents_out, oids_out = _resolve_ai_intents_and_known_order_ids(
        str(tmp_path),
        known_ai_order_ids={"ai-1"},
        read_json_fn=lambda p: (read_calls.append(p), {})[1],
    )
    assert intents_out is None and oids_out == {"ai-1"} and read_calls == []
    intents_empty, oids_empty = _resolve_ai_intents_and_known_order_ids(
        str(tmp_path),
        ai_intents=[],
        read_json_fn=lambda p: (read_calls.append(p), {})[1],
        need_intents=True,
    )
    assert intents_empty == [] and oids_empty == set() and read_calls == []


def test_temporary_environ_daemon_fetch_equity_oi_remote_trades_and_guard_preflight_border_cases(
    tmp_path, monkeypatch
):
    """Regression suite for temporary_environ, daemon_ssot single-call fetch, equity OI bounds, remote trade LSP, and strategy/guard preflight."""

    # 1. temporary_environ sets/unsets and restores exact environment even on exceptions
    monkeypatch.setenv("_TEST_TEMP_ENV_KEEP", "orig")
    monkeypatch.delenv("_TEST_TEMP_ENV_NEW", raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_environ({"_TEST_TEMP_ENV_KEEP": None}, _TEST_TEMP_ENV_NEW="42"):
            assert "_TEST_TEMP_ENV_KEEP" not in os.environ
            assert os.environ["_TEST_TEMP_ENV_NEW"] == "42"
            raise RuntimeError("boom")
    assert os.environ["_TEST_TEMP_ENV_KEEP"] == "orig"
    assert "_TEST_TEMP_ENV_NEW" not in os.environ

    # 2. portfolio_snapshot._policy_float falls back to _ROOT_CONFIG when env var is empty/whitespace
    monkeypatch.setitem(portfolio_snapshot._ROOT_CONFIG, "_TEST_POLICY_KEY", "12.5")
    monkeypatch.setenv("_TEST_POLICY_KEY", "   ")
    assert portfolio_snapshot._policy_float("_TEST_POLICY_KEY") == pytest.approx(12.5)

    # 3. ExternalTelemetryMixin._call_collector_fetch never double-invokes on internal TypeError & sanitizes flags
    calls: list[tuple] = []

    def _fetch_with_now(*, force_refresh: bool = False, now_ts=None):
        calls.append((force_refresh, now_ts))
        raise TypeError("internal collector bug")

    with pytest.raises(TypeError, match="internal collector bug"):
        ExternalTelemetryMixin._call_collector_fetch(
            SimpleNamespace(fetch=_fetch_with_now), force_refresh="false", now_ts=1700000000.0  # type: ignore[arg-type]
        )
    assert calls == [(False, 1700000000.0)]

    def _fetch_legacy(*, force_refresh: bool = False):
        calls.append((force_refresh, None))
        return {"ok": True}

    calls.clear()
    assert ExternalTelemetryMixin._call_collector_fetch(
        SimpleNamespace(fetch=_fetch_legacy), force_refresh="true", now_ts=1700000000.0  # type: ignore[arg-type]
    ) == {"ok": True}
    assert calls == [(True, None)]

    daemon_actions: list[tuple] = []
    fake_daemon = SimpleNamespace(
        run_cycle=lambda force=True: daemon_actions.append(("once", force)),
        start=lambda: daemon_actions.append(("start", None)),
    )
    run_daemon_cli_once_or_start(fake_daemon, once="false", force="false")  # type: ignore[arg-type]
    run_daemon_cli_once_or_start(fake_daemon, once="true", force="false")  # type: ignore[arg-type]
    assert daemon_actions == [("start", None), ("once", False)]

    # 4. _DiskBackedSymbolCollector._extract_equity_options_oi handles None ctx and NaN/negative/bool mark_px
    assert _DiskBackedSymbolCollector._extract_equity_options_oi(None, float("nan")) == (  # type: ignore[arg-type]
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert _DiskBackedSymbolCollector._extract_equity_options_oi(
        {"call_oi": 10, "put_oi": 5}, -20.0
    ) == (10.0, 5.0, 15.0, 1500.0, 0.0)
    assert _DiskBackedSymbolCollector._extract_equity_options_oi(
        {"call_oi": 10, "put_oi": 5}, True  # type: ignore[arg-type]
    ) == (10.0, 5.0, 15.0, 1500.0, 0.0)

    # 5. MarketDataProvider.fetch_remote_trades_with_client LSP + ReplayMarketDataProvider malformed order timestamp
    fake_cli = SimpleNamespace(
        fetch_fills=lambda sym: [
            {"symbol": sym, "time": 1700000005000, "price": 100.0, "qty": 1.0},
            {"symbol": sym, "time": 1699999990000, "price": 99.0, "qty": 1.0},
        ]
    )
    assert _TestMarketProvider.fetch_remote_trades_with_client("BTCUSDC", True, client=fake_cli) == [  # type: ignore[arg-type]
        {"symbol": "BTCUSDC", "time": 1700000005000, "price": 100.0, "qty": 1.0},
        {"symbol": "BTCUSDC", "time": 1699999990000, "price": 99.0, "qty": 1.0},
    ]

    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0), (1010.0, 50100.0)]})
    rp.advance("BTCUSDC", 1)
    rp._orders["BTCUSDC"] = [{"orderId": "1", "side": "BUY", "timestamp": None}]
    assert rp.has_orders_since("BTCUSDC", 5.0) is False
    assert rp.get_orders("BTCUSDC", "BUY", 5.0) == []

    # 6. step_adaptive_trend_window & compute_window_bounds preserve monotonic prev_ts on out-of-order ticks
    prev_t, w_sz, c_len, sr = step_adaptive_trend_window(100.0, 120.0, 200, 5, 1.0, 222.0)
    assert prev_t == 120.0 and c_len == 6 and sr == 1.0
    a_bounds, sr_bounds = compute_window_bounds(np.array([100.0, 102.0, 101.0, 104.0]), win_sec=200.0)
    assert sr_bounds.tolist() == pytest.approx([pw.DEFAULT_SAMPLE_RATE_SEC, 2.0, 2.0, 2.0])
    assert a_bounds.tolist() == [0, 0, 0, 0]

    # 7. order_guard._preflight_intelligence_guards_with_qlimit2 & BaseSpotMixin._init_common_spot_runtime_state flags
    monkeypatch.setattr(order_guard, "_load_margins", lambda: {"intelligence_guards_mode": "off"})
    early_og, side_og, _, _, p_only, w_only = order_guard._preflight_intelligence_guards_with_qlimit2(
        "INVALID_SIDE", {"parabolic_surge"}
    )
    assert early_og == (True, "intelligence_guards_off", 1.0) and side_og == "" and p_only is True and w_only is False

    spot_dummy = object.__new__(BaseSpotMixin)
    spot_dummy._sanitize_state_dict = lambda d: dict(d)  # type: ignore[attr-defined]
    spot_dummy._init_common_spot_runtime_state(
        SimpleNamespace(currency="USD", sl_mhdsl_enabled=False, tp_mdhtp_enabled=False),
        state_file=str(tmp_path / "st.json"),
        initial_state={"cycle": 1},
        dry_run="false",  # type: ignore[arg-type]
        desktop="false",  # type: ignore[arg-type]
        replay_mode="false",  # type: ignore[arg-type]
        observe_only="true",  # type: ignore[arg-type]
    )
    assert (
        spot_dummy.dry_run is False
        and spot_dummy.desktop is False
        and spot_dummy.replay_mode is False
        and spot_dummy.observe_only is True
    )


def test_coerce_dict_maps_cli_env_path_replay_metrics_peak_scope_and_hl_status_border_cases(
    tmp_path, monkeypatch
):
    """Regression suite for coerce_dict_maps SSOT, resolve_cli_env_path, replay_metrics, post_exit_peak_step_scope, and HL status extraction."""
    hl_client_mod = load_venue_modules("hyperliquid", "hl_client")
    _first_exchange_status = hl_client_mod._first_exchange_status
    _parse_order_response = hl_client_mod._parse_order_response

    # 1. coerce_dict_maps & active_intents.format_summary handle None/non-dict payloads safely
    assert coerce_dict_maps({"a": 1}, None, [1, 2], True) == ({"a": 1}, {}, {}, {})
    assert mss.coerce_dict_maps({"a": 1}, None, [1, 2], True) == ({"a": 1}, {}, {}, {})
    assert mss.coerce_dict_maps.__qualname__ == coerce_dict_maps.__qualname__
    summary_txt = active_intents.format_summary(None)  # type: ignore[arg-type]
    assert "Total active intents:  0" in summary_txt

    # 2. resolve_cli_env_path & load_cli_env_stack fall back cleanly when --env-file is blank/whitespace
    fake_script = tmp_path / "bot_runner.py"
    monkeypatch.delenv("ENV_FILE", raising=False)
    expected_default = str(tmp_path / ".env")
    assert resolve_cli_env_path(fake_script, ["bot", "--env-file", "   "]) == expected_default
    loaded_paths: list[str] = []
    assert (
        load_cli_env_stack(
            fake_script,
            ["bot", "--env-file=   "],
            load_fn=loaded_paths.append,
        )
        == expected_default
    )
    assert loaded_paths == [expected_default]

    # 3. run_cli_single_or_sweep_backtest & print_single_backtest_report sanitize bool/NaN bh, mode, min_bars, include_funding
    emitted: list[str] = []
    rc_single = run_cli_single_or_sweep_backtest(
        " SINGLE ",
        {"tp": 2.0, "drop": 2.0},
        lambda _p: {"total": 15.0, "net": 15.0, "cycles": 2, "wins": 1, "realized": 15.0, "final_upnl": 0.0, "fees": 0.5, "maxdd": 3.0, "open_qty": 0.0},
        budget=100.0,
        bh=True,  # type: ignore[arg-type]
        ohlc=[(1, 1, 1, 1)] * 5,
        min_bars="3",  # type: ignore[arg-type]
        single_header="HDR",
        single_params_line="PARAMS",
        sweep_header="SWEEP",
        include_funding="false",  # type: ignore[arg-type]
        emit=emitted.append,
    )
    assert rc_single == 0
    joined_out = "\n".join(emitted)
    assert "funding $" not in joined_out and "buy&hold: +0.00%" in joined_out

    emitted.clear()
    print_single_backtest_report(
        "HDR2",
        "P2",
        {"total": 5.0, "funding": 2.0},
        budget=100.0,
        bh=float("nan"),
        include_funding="false",  # type: ignore[arg-type]
        emit=emitted.append,
    )
    assert all("funding $" not in ln for ln in emitted)

    # 4. post_exit_peak_step_scope parses string flags ("false") and tolerates non-dict flags
    orig_fn = lambda: "orig"
    peak_fn = lambda: "peak"
    dummy_cls = SimpleNamespace(step=orig_fn)
    flags_box = {"peak": True}
    with post_exit_peak_step_scope(dummy_cls, flags_box, "false", peak_fn, orig_fn):
        assert flags_box["peak"] is False and dummy_cls.step is orig_fn
    assert flags_box["peak"] is True and dummy_cls.step is orig_fn
    with post_exit_peak_step_scope(dummy_cls, None, "true", peak_fn, orig_fn):  # type: ignore[arg-type]
        assert dummy_cls.step is peak_fn
    assert dummy_cls.step is orig_fn

    # 5. Hyperliquid _first_exchange_status / _parse_order_response & T212Provider._signed_qty
    assert _first_exchange_status({"status": "ok", "response": {"data": {"statuses": ["success"]}}}) == "success"
    assert _first_exchange_status({"status": "ok", "response": None}) is None
    assert _parse_order_response({"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 77}}]}}}) == (
        True,
        77,
        "resting",
    )
    with pytest.raises(ProviderError, match="invalid T212 side"):
        T212Provider._signed_qty(True, 1.0)  # type: ignore[arg-type]
    assert T212Provider._signed_qty(" sell ", 1.5) == pytest.approx(-1.5)


def test_clean_nonempty_str_t212_fills_guard_inf_scale_rounding_and_daemon_cli_border_cases(monkeypatch, capsys):
    """Verify Wave 12 border cases: clean_nonempty_str, T212 fills/history guards, +inf parsing, IEEE-754 scale_pct rounding, daemon CLI helpers, and sweep footer sanitization."""

    # 1. clean_nonempty_str preserves case while rejecting bool/non-str/blank; re-exported on macro_sentiment_ssot
    assert botcore.clean_nonempty_str("  abc_1 ") == "abc_1"
    assert mss.clean_nonempty_str("  XyZ ") == "XyZ"
    for bad in ("", "   ", True, False, 123, None, ["a"]):
        assert botcore.clean_nonempty_str(bad) is None
        assert botcore.clean_upper_token(bad) is None

    # 2. T212Provider._fills_from_api rejects non-str symbols without AttributeError; _remote_trade_rows_for_symbol guards malformed maps
    t212 = T212Provider(client=SimpleNamespace(refresh_order_history=lambda max_age_sec=0.0: [{"id": 1}]))
    assert t212._fills_from_api(123) == []  # type: ignore[arg-type]
    monkeypatch.setattr(t212, "normalize_history_items", lambda _items: None)
    assert t212._remote_trade_rows_for_symbol("NVDA_US_EQ", "NVDA_US_EQ", 0.0) == []
    monkeypatch.setattr(
        t212,
        "normalize_history_items",
        lambda _items: {123: [{"orderId": "bad"}], "OTHER": "not-a-list", "NVDA_US_EQ": [{"orderId": "ok1"}]},
    )
    assert t212._remote_trade_rows_for_symbol("NVDA_US_EQ", "NVDA_US_EQ", 0.0) == [{"orderId": "ok1"}]

    prov = _TestMarketProvider()
    prov._VALIDATE_REMOTE_TRADES_SYMBOL = "false"  # type: ignore[assignment]
    validated_calls: list[str] = []
    monkeypatch.setattr(prov, "validate_symbol", lambda s: validated_calls.append(s))
    monkeypatch.setattr(prov, "_fills_from_api", lambda _s, **_kw: [{"id": 1, "timestamp": 1700000000000}])
    assert prov.fetch_remote_trades("BTCUSDC", 0) == [{"id": 1, "timestamp": 1700000000000}]
    assert validated_calls == []

    # 3. _clean_required_qty_or_inf handles math.inf and "+inf" strings while rejecting -inf/bool; shadow notify rounds 0.29*100 -> 29%
    assert math.isinf(order_guard._clean_required_qty_or_inf(math.inf))
    assert math.isinf(order_guard._clean_required_qty_or_inf(" +INF "))
    assert order_guard._clean_required_qty_or_inf("-inf") == 0.0
    assert order_guard._clean_required_qty_or_inf(True) == 0.0

    notified_msgs: list[tuple[str, str]] = []
    eog._MICRO_SHADOW_NOTIFY_COOLDOWN.clear()
    monkeypatch.setattr(
        eog._alertnotifiers,
        "notify",
        lambda *, title="", body="", **_kw: notified_msgs.append((title, body)),
    )
    monkeypatch.setattr(eog, "get_orderbook_collector", lambda: SimpleNamespace(fetch=lambda *_a, **_k: {"ts": 1_700_000_000.0}))
    monkeypatch.setattr(
        eog.owg,
        "OrderbookWallGuard",
        lambda **_kw: SimpleNamespace(
            check=lambda *_a, **_k: GuardDecision(
                allowed=True,
                reason="wall_near",
                guard_name="OrderbookWallGuard",
                brake_action=BrakeAction.DOWNSCALE_QTY,
                suggested_scale=0.29,
            )
        ),
    )
    margins_shadow = make_test_margins(
        disable_external_pillars=True,
        external_guard_mode="shadow",
        orderbook_wall_guard_mode="shadow",
        shadow_notify=True,
    )
    assert eog.check_external_order_guards("BTCUSDC", "BUY", 60000.0, margins=margins_shadow, now=1_700_000_000.0) == (True, "ok", 1.0)
    assert len(notified_msgs) == 1
    assert "SCALE 29%" in notified_msgs[0][0] and "(Scale: 29%)" in notified_msgs[0][1]

    # 4. run_daemon_cli_once_or_start supports print_json_on_once, exit_on_once, and single_instance_name
    capsys.readouterr()
    locks_acquired: list[str] = []
    started: list[bool] = []
    monkeypatch.setattr(dssot, "single_instance", lambda name: locks_acquired.append(name))
    daemon_stub = SimpleNamespace(
        run_cycle=lambda *, force=True: {"ok": force, "count": 2},
        start=lambda: started.append(True),
    )
    with pytest.raises(SystemExit) as exc_info:
        dssot.run_daemon_cli_once_or_start(
            daemon_stub,
            once="true",  # type: ignore[arg-type]
            force="true",  # type: ignore[arg-type]
            print_json_on_once="true",  # type: ignore[arg-type]
            exit_on_once="true",  # type: ignore[arg-type]
        )
    assert exc_info.value.code == 0
    assert '"count": 2' in capsys.readouterr().out

    dssot.run_daemon_cli_once_or_start(
        daemon_stub,
        once="false",  # type: ignore[arg-type]
        single_instance_name="test_intel_daemon",
    )
    assert locks_acquired == ["test_intel_daemon"] and started == [True]

    # 5. print_mt_buy_hold_footer sanitizes None/NaN entries & re-export seams remain intact
    capsys.readouterr()
    smt.print_mt_buy_hold_footer(
        ["BTCUSDC", "TAOUSDC"],
        {"BTCUSDC": 12.5, "TAOUSDC": None},  # type: ignore[dict-item]
        "HEADER",
        col_width="22",  # type: ignore[arg-type]
    )
    footer_out = capsys.readouterr().out
    assert "Buy & Hold Benchmark" in footer_out and "+12.50" in footer_out
    assert hasattr(bapi_trades.apiorders, "sym") and bapi_trades.api._normalize_side(" b ") == "BUY"
    assert callable(bt_eval.load_layered_env_config)


def test_wave13_sleep_interval_max_age_ladders_warmup_and_guard_border_cases(tmp_path):
    """Verify Wave 13 border cases: sleep_monitoring_interval exact chunk math, resolve_positive_max_age, provider ladders, reconciliation horizon validation, BaseSpotMixin string flags, split_warmup_ohlc divisor<=0, PriceAlert bool timestamp, and guard templates."""

    # 1. sleep_monitoring_interval sleeps exact total duration (never 60*60=3600s) and honors string/bool _running
    slept_60: list[float] = []
    runner_on = SimpleNamespace(_running=True)
    botcore.sleep_monitoring_interval(runner_on, 60.0, max_sleep_step=61.0, sleep_fn=slept_60.append)
    assert slept_60 == [pytest.approx(60.0)]

    slept_frac: list[float] = []
    botcore.sleep_monitoring_interval(runner_on, 1.5, max_sleep_step=1.0, sleep_fn=slept_frac.append)
    assert slept_frac == [pytest.approx(1.0), pytest.approx(0.5)]

    slept_sub: list[float] = []
    botcore.sleep_monitoring_interval(runner_on, 2.0, max_sleep_step=0.5, sleep_fn=slept_sub.append)
    assert len(slept_sub) == 4 and sum(slept_sub) == pytest.approx(2.0)

    slept_str_off: list[float] = []
    botcore.sleep_monitoring_interval(
        SimpleNamespace(_running="false"), 10.0, max_sleep_step=1.0, sleep_fn=slept_str_off.append
    )
    assert slept_str_off == []

    slept_interrupt: list[float] = []
    runner_flip = SimpleNamespace(_running=True)

    def _flip_after_two(dt: float) -> None:
        slept_interrupt.append(dt)
        if len(slept_interrupt) >= 2:
            runner_flip._running = False

    botcore.sleep_monitoring_interval(runner_flip, 10.0, max_sleep_step=1.0, sleep_fn=_flip_after_two)
    assert slept_interrupt == [pytest.approx(1.0), pytest.approx(1.0)]

    # 2. resolve_positive_max_age sanitizes both explicit max_age_sec and default
    assert botcore.resolve_positive_max_age(None, 90.0) == pytest.approx(90.0)
    assert botcore.resolve_positive_max_age(None, 0.0) is None
    assert botcore.resolve_positive_max_age(None, math.nan) is None
    assert botcore.resolve_positive_max_age("15.5", 90.0) == pytest.approx(15.5)
    assert botcore.resolve_positive_max_age(0, 90.0) is None
    assert botcore.resolve_positive_max_age(True, 90.0) is None

    # 3. MarketDataProvider ladder helpers & OrderReconciliationCapabilities horizon validation
    step_sz, tick_sz = MarketDataProvider.default_step_and_tick(math.nan)
    assert step_sz > 0.0 and tick_sz > 0.0
    assert isinstance(MarketDataProvider.tca_price_decimals(-5.0), int)

    caps = OrderReconciliationCapabilities(True, True, True, True, "120.5")  # type: ignore[arg-type]
    assert caps.not_found_reliable_for_seconds == pytest.approx(120.5)
    with pytest.raises(TypeError, match="positive number"):
        OrderReconciliationCapabilities(True, True, True, True, "not-a-number")  # type: ignore[arg-type]
    for bad_h in (0, -5.0, math.nan, math.inf, "inf", "nan"):
        with pytest.raises(ValueError, match="finite and > 0"):
            OrderReconciliationCapabilities(True, True, True, True, bad_h)  # type: ignore[arg-type]

    # 4. BaseSpotMixin string-boolean flags in _persist_pending_state_slot & _resolve_run_mode_and_banner
    saved_cnt: list[int] = []
    spot_stub = SimpleNamespace(
        s={},
        p=SimpleNamespace(enable_takeprofit="false"),
        observe_only="false",
        dry_run="false",
        _state_write_failed="true",
        _save=lambda: saved_cnt.append(1),
    )
    BaseSpotMixin._persist_pending_state_slot(
        spot_stub, "pending_buy", {"oid": "1"}, require_write="false"  # type: ignore[arg-type]
    )
    assert spot_stub.s["pending_buy"] == {"oid": "1"} and len(saved_cnt) == 1
    with pytest.raises( ValueError, match="could not be persisted"):
        BaseSpotMixin._persist_pending_state_slot(
            spot_stub, "pending_buy", None, require_write="true", error_cls=ValueError  # type: ignore[arg-type]
        )
    assert BaseSpotMixin._resolve_run_mode_and_banner(spot_stub, real_banner="[REAL]") == ("dca_only", "[REAL]")

    # 5. split_warmup_ohlc falls back to divisor=5 when divisor <= 0 and handles tuple/None inputs
    bars_10 = tuple(range(10))
    wn_zero, warm_z, run_z = split_warmup_ohlc(bars_10, max_warmup=30, divisor=0)  # type: ignore[arg-type]
    wn_neg, warm_n, run_n = split_warmup_ohlc(bars_10, max_warmup=30, divisor=-3)  # type: ignore[arg-type]
    assert (wn_zero, len(warm_z), len(run_z)) == (2, 2, 8)
    assert (wn_neg, len(warm_n), len(run_n)) == (2, 2, 8)
    assert split_warmup_ohlc(None) == (0, [], [])  # type: ignore[arg-type]

    # 6. PriceAlert rejects boolean timestamps, read_tail_complete_lines_text tolerates invalid keep_bytes, and reload(botcore) preserves RTH gate
    alert_bool = PriceAlert("BTC", "up", 105.0, 100.0, 5.0, 4.0)
    alert_bool.timestamp = True  # type: ignore[assignment]
    assert alert_bool.to_dict()["timestamp_readable"] == "True"
    assert "at True)" in str(alert_bool)

    log_f = tmp_path / "sample_tail.jsonl"
    atomic_write_text(str(log_f), '{"a":1}\n{"a":2}\n')
    assert read_tail_complete_lines_text(str(log_f), "bad") == '{"a":2}\n'  # type: ignore[arg-type]

    saved_botcore_dict = dict(botcore.__dict__)
    try:
        pre_rth_fn = botcore.is_us_rth_open
        botcore.configure_runtime_services(enforce_rth_gate=False)
        importlib.reload(botcore)
        assert botcore._RUNTIME_SERVICES.enforce_rth_gate is False
        assert botcore.is_rth_gate_enabled(None, rth_fn=pre_rth_fn) is False
        assert botcore.is_rth_gate_enabled(None, rth_fn=lambda: False) is True
        assert botcore.is_rth_gate_enabled("false") is False  # type: ignore[arg-type]
    finally:
        botcore.__dict__.update(saved_botcore_dict)
        botcore._sync_reexport_modules()

    # 7. Internal & External guard template methods accept unsupported sides and resolved snapshots cleanly
    assert ParabolicSurgeGuard().check("BTCUSDC", "HOLD", 60000.0).reason == "unsupported_side_skipped"
    assert WeibullExhaustionGuard().check("BTCUSDC", "HOLD", 3600.0).reason == "unsupported_side"
    for ext_g in (
        LiquidationCascadeGuard(),
        FundingCrowdingGuard(),
        OrderbookWallGuard(),
        WhaleDivergenceGuard(),
    ):
        dec_hold = ext_g.check("BTCUSDC", "HOLD", SimpleNamespace())  # type: ignore[arg-type]
        assert dec_hold.allowed is True and dec_hold.reason == "unsupported_side_allowed"
        dec_none = ext_g.check("BTCUSDC", "BUY", None)
        assert dec_none.allowed is True


def test_wave14_markdown_report_default_writer_cli_specs_and_provider_border_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify write_markdown_report default persistence, CLI flag spec normalization, and venue provider border cases."""
    # 1. write_markdown_report writes via atomic_write_text by default when write_fn is omitted
    md_default = tmp_path / "default_report.md"
    emitted: list[str] = []
    write_markdown_report(md_default, ["# Title", 42], joiner="\n", emit=emitted.append)  # type: ignore[list-item]
    assert read_text_file(str(md_default)) == "# Title\n42"
    assert emitted == [f"Report written to {md_default}"]

    custom_calls: list[tuple[object, str]] = []
    md_custom = tmp_path / "custom_report.md"
    write_markdown_report(
        md_custom,
        ["lineA\n", "lineB\n"],
        write_fn=lambda p, t: custom_calls.append((p, t)),
        emit=lambda _m: None,
    )
    assert custom_calls == [(md_custom, "lineA\nlineB\n")]

    # 2. print_ascii_table_header sanitizes string/non-positive widths and empty bottom_char
    hdr_lines: list[str] = []
    print_ascii_table_header(
        [("Sym", 0), ("Ret", "6")],  # type: ignore[list-item]
        width=-10,
        bottom_char="",
        emit=hdr_lines.append,
    )
    assert len(hdr_lines) == 3
    assert len(hdr_lines[2]) == 105 and hdr_lines[2].startswith("=")

    # 3. add_store_true_cli_args supports 1-tuples, lists, blank specs, and multi-alias flag tuples
    ap = argparse.ArgumentParser()
    assert (
        botcore.add_store_true_cli_args(
            ap,
            "  ",
            "--plain",
            ("--solo",),
            ["--list-flag", "List help"],
            (("--verbose", "-v"), "Verbose help"),
            (("", "   "), "Ignored blank aliases"),
        )
        is ap
    )
    parsed = ap.parse_args(["--plain", "--solo", "--list-flag", "-v"])
    assert (parsed.plain, parsed.solo, parsed.list_flag, parsed.verbose) == (True, True, True, True)

    ap_bot = argparse.ArgumentParser()
    assert (
        botcore.add_paper_skip_wait_cli_args(
            ap_bot,
            include_desktop="true",  # type: ignore[arg-type]
            include_skip_wait="false",  # type: ignore[arg-type]
        )
        is ap_bot
    )
    parsed_bot = ap_bot.parse_args(["--desktop", "--paper"])
    assert (parsed_bot.desktop, parsed_bot.paper, hasattr(parsed_bot, "skip_wait")) == (True, True, False)

    ap_cmp = argparse.ArgumentParser()
    assert add_path_cli_args(ap_cmp, "  ", "--baseline") is ap_cmp
    assert add_report_output_cli_args(ap_cmp) is ap_cmp
    parsed_cmp = ap_cmp.parse_args(["--baseline", "a.json", "--output", "b.json", "--markdown", "c.md"])
    assert (parsed_cmp.baseline, parsed_cmp.output, parsed_cmp.markdown) == (
        Path("a.json"),
        Path("b.json"),
        Path("c.md"),
    )

    ap_ext = argparse.ArgumentParser()
    assert add_external_telemetry_cli_args(ap_ext, skip_flag="--skip-liq", skip_help="Skip liq") is ap_ext
    parsed_ext = ap_ext.parse_args(["-v", "--once", "--skip-liq"])
    assert (parsed_ext.verbose, parsed_ext.once, parsed_ext.skip_liq) == (True, True, True)

    ap_macro = argparse.ArgumentParser()
    assert add_macro_analyzer_cli_args(ap_macro) is ap_macro
    parsed_macro = ap_macro.parse_args(["-v", "--force"])
    assert (parsed_macro.verbose, parsed_macro.force, parsed_macro.once) == (True, True, False)

    # 4. Concrete venue provider border cases (HL precision, T212 5% reserve retry, Kraken historical filter)
    hl_stub = SimpleNamespace(_meta_cache={"HYPE": {"szDecimals": 2}})
    hl_sub = HyperliquidProvider.extract_cached_fee_and_limits(hl_stub, "HYPEUSDC", price=0.045)
    hl_big = HyperliquidProvider.extract_cached_fee_and_limits(hl_stub, "HYPEUSDC", price=95000.0)
    assert hl_sub["step_size"] == pytest.approx(0.01)
    assert hl_sub["tick_size"] == pytest.approx(1e-6)
    assert hl_big["tick_size"] == pytest.approx(1.0)

    t212_calls: list[float] = []

    def _fake_market_order(_sym: str, qty_arg: float, extended_hours: bool = False):
        t212_calls.append(qty_arg)
        if len(t212_calls) == 1:
            return 400, {"type": "/api-errors/insufficient-free-for-stocks-buy"}
        return 200, {"id": 777}

    t212_prov = T212Provider.__new__(T212Provider)
    monkeypatch.setattr(t212_prov, "_client", lambda: SimpleNamespace(place_market_order=_fake_market_order))
    st_code, resp_data = t212_prov._send_order("NVDA_US_EQ", "BUY", 10.0, None, market=True)
    assert (st_code, resp_data) == (200, {"id": 777})
    assert len(t212_calls) == 2 and t212_calls[0] == pytest.approx(10.0) and t212_calls[1] == pytest.approx(9.45)

    krk_prov = KrakenProvider.__new__(KrakenProvider)
    monkeypatch.setattr(
        krk_prov,
        "ohlc_series",
        lambda _sym, _iv: SimpleNamespace(
            timestamps=[1700000000, -1, 1700003600, "1700007200"],
            closes=[50000.0, 51000.0, 0.0, "50500.5"],
        ),
    )
    assert krk_prov.fetch_historical_ohlc_points("BTCUSDC", days=1) == [
        (1700000000.0, 50000.0),
        (1700007200.0, 50500.5),
    ]
    monkeypatch.setattr(
        krk_prov,
        "ohlc_series",
        lambda _sym, _iv: SimpleNamespace(timestamps=[1700000000], closes=[50000.0, 50100.0]),
    )
    assert krk_prov.fetch_historical_ohlc_points("BTCUSDC", days=1) == []


def test_wave15_bind_attr_call_monkeypatch_seams_isolated_cfg_loader_and_nan_ts_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wave 15: dynamic bind_attr_call monkeypatch seams, strict kwarg resolvers, factory isolation, and NaN/empty-items guards."""
    # 1. KrakenClient, HLClient, TradingBot, and T212Client honor instance monkeypatches on bound methods
    krk = KrakenClient(api_key="", api_secret="")
    priv_calls: list[tuple[str, Any]] = []
    pub_calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        krk,
        "_private",
        lambda method, data=None: priv_calls.append((method, data)) or {
            "open": {"O1": {"status": "open"}},
            "closed": {"C1": {"status": "closed"}},
            "ZUSD": {"vol": "100.0"},
        },
    )
    monkeypatch.setattr(
        krk,
        "_public",
        lambda endpoint, params=None: pub_calls.append((endpoint, params)) or {"XBTUSDC": {"c": ["95000.0"]}},
    )
    assert krk.balance_ex()["ZUSD"] == {"vol": "100.0"}
    assert krk.open_orders() == {"O1": {"status": "open"}}
    assert krk.closed_orders() == {"C1": {"status": "closed"}}
    assert krk.cancel_order("OABCDE-12345-FGHIJK") is not None
    assert krk.query_orders("OABCDE-12345-FGHIJK") is not None
    assert krk.pair_info("XBTUSDC") == {"c": ["95000.0"]}
    assert krk.ticker("XBTUSDC") == {"c": ["95000.0"]}
    assert [m for m, _ in priv_calls] == ["BalanceEx", "OpenOrders", "ClosedOrders", "CancelOrder", "QueryOrders"]
    assert [e for e, _ in pub_calls] == ["AssetPairs", "Ticker"]
    monkeypatch.setattr(krk, "_private", lambda _method, _data=None: ["unexpected-list"])
    assert krk.open_orders() == {}
    assert krk.closed_orders() == {}

    hl_client_mod, hl_common_mod = load_venue_modules("hyperliquid", "hl_client", "common")
    hl_cli = hl_client_mod.HLClient.__new__(hl_client_mod.HLClient)
    monkeypatch.setattr(hl_cli, "_mid_for_key", lambda key, label="": 42.5 if (key, label) == ("HYPE", "mid") else 43.0)
    assert hl_cli.mid("HYPE") == pytest.approx(42.5)
    assert hl_cli.spot_mid("@107") == pytest.approx(43.0)
    assert math.isnan(hl_client_mod._round_px(float("nan"), 2))
    with pytest.raises(ValueError):
        hl_client_mod._round_px(25.0, True)

    t212_client_mod = load_venue_modules("212trading", "t212_client")
    t212_cli = t212_client_mod.T212Client.__new__(t212_client_mod.T212Client)
    monkeypatch.setattr(t212_cli, "_read_cached_dict_list", lambda attr, path, label: [{"attr": attr, "path": path, "label": label}])
    assert t212_cli.get_portfolio() == [{"attr": "_pf_cache", "path": "/equity/portfolio", "label": "portfolio"}]
    assert t212_cli.list_active_orders() == [{"attr": "_ord_cache", "path": "/equity/orders", "label": "active orders"}]

    bot_stub = rtrade.TradingBot.__new__(rtrade.TradingBot)
    monkeypatch.setattr(bot_stub, "_repetitive_side_loop", lambda side, cp, fp: (side, cp, fp))
    assert bot_stub.repetitive_buy(100.0, 105.0) == ("BUY", 100.0, 105.0)
    assert bot_stub.repetitive_sell(100.0, 95.0) == ("SELL", 100.0, 95.0)

    # 2. Strict keyword validation on bind_isolated_venue_cfg_loader and agy path resolvers
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_isolated_kraken_cfg(cfg_path=str(tmp_path))
    with pytest.raises(TypeError, match="multiple values"):
        load_isolated_kraken_cfg(str(tmp_path), kraken_dir=str(tmp_path))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        hl_common_mod.load_isolated_hl_cfg(wrong_kw=str(tmp_path))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        agy_paths.resolve_bot_cli_base(wrong_kw=str(tmp_path))
    with pytest.raises(TypeError, match="multiple values"):
        agy_paths.resolve_bot_cli_base(str(tmp_path), cli_base=str(tmp_path))
    assert agy_paths.resolve_bot_cli_base(cli_base=str(tmp_path)) == str(tmp_path)

    lock_file = tmp_path / "ui_store.lock"
    monkeypatch.setattr(agy_ui_mirror, "resolve_store_lock_path", lambda: str(lock_file))
    with agy_ui_mirror._store_flock():
        assert lock_file.exists()

    # 3. LazyPluginRegistryFactory isolates _registry and _instances per subclass
    sub_a = type("SubFactoryA", (botcore.LazyPluginRegistryFactory,), {})
    sub_b = type("SubFactoryB", (botcore.LazyPluginRegistryFactory,), {})
    sub_a.register("alpha", SimpleNamespace)
    assert "alpha" in sub_a._registry and "alpha" not in sub_b._registry
    assert sub_a._instances is not sub_b._instances

    # 4. Adverse trend normalization, bool rejection in decimal_places, NaN ts guard, and crypto-only macro summary
    assert order_guard.is_adverse_trend("buy", "  BEAR ") is True
    assert order_guard.is_adverse_trend("sell", " BULL ") is True
    with pytest.raises(BinanceFilterError):
        decimal_places(True)

    coord = tradeall.TrendCoordinator.__new__(tradeall.TrendCoordinator)
    coord.symbol_aliases = {}
    coord.trend_states = {}
    coord.trend_states_big = {}
    coord._new_trend_state = lambda: SimpleNamespace()
    coord._resolve_symbol = lambda sym: sym
    coord.current_price_mgr = SimpleNamespace(
        STALE_THRESHOLD_MS=5000,
        get_price=lambda _sym: [float("nan"), 95000.0],
    )
    assert coord.evaluate("BTCUSDC") is None

    summary_crypto_only = mss.format_macro_benchmarks_summary_line(
        {"items": {}, "coinbase_premium_bps": 12.5, "btc_dvol": 55.0}
    )
    assert "CB_Premium=+12.5bps" in summary_crypto_only and "BTC_DVOL=55.0%" in summary_crypto_only


def test_wave16_bind_attr_call_duck_stubs_timed_filelock_feed_url_seams_and_sweep_null_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wave 16: bind_attr_call duck-stub descriptor, timed FileLock, feed URL monkeypatch seams, and sweep null guards."""
    # 1. bind_attr_call works on unbound duck-typed SimpleNamespace stubs AND bound/overridden instances
    assert MarketDataProvider.requires_cache_permit(SimpleNamespace(name="binance")) is True
    assert MarketDataProvider.requires_cache_permit(SimpleNamespace(name="kraken")) is False
    assert MarketDataProvider.defers_smart_cancellation(SimpleNamespace(name="binance")) is True
    assert MarketDataProvider.supports_business_minimum(SimpleNamespace(name="binance")) is True
    assert MarketDataProvider.supports_order_validity(SimpleNamespace(name="t212")) is True
    assert MarketDataProvider.defers_preflight_quantity(SimpleNamespace(name="kraken")) is True
    assert (
        MarketDataProvider.requires_cache_permit(
            SimpleNamespace(_provider_bool_attr=lambda attr: attr == "_REQUIRES_CACHE_PERMIT")
        )
        is True
    )
    assert KrakenClient.open_orders(SimpleNamespace(_private=lambda m: {"open": {"ODUCK": {"m": m}}})) == {
        "ODUCK": {"m": "OpenOrders"}
    }

    # 2. Timed botcore.FileLock contention & input guards + state_io append/binary-read helpers
    with pytest.raises(TypeError):
        botcore.FileLock(tmp_path / "bad_bool.lock", timeout=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        botcore.FileLock(tmp_path / "bad_nan.lock", timeout=float("nan"))

    contended_lock_path = tmp_path / "contended.lock"
    with botcore.FileLock(contended_lock_path) as outer_lock:
        assert outer_lock.acquired is True
        with pytest.raises(TimeoutError):
            with botcore.FileLock(contended_lock_path, timeout=0.02, poll_interval=0.005, raise_on_timeout=True):
                pass
        with botcore.FileLock(
            contended_lock_path, timeout=0.02, poll_interval=0.005, raise_on_timeout=False
        ) as soft_lock:
            assert soft_lock.acquired is False

    with botcore.FileLock(contended_lock_path, timeout=0.2, poll_interval=0.005) as after_lock:
        assert after_lock.acquired is True

    nested_log = tmp_path / "logs" / "nested" / "append.log"
    with state_io.open_append_text_file(nested_log) as fh1:
        fh1.write("alpha\n")
    with state_io.open_append_text_file(nested_log) as fh2:
        fh2.write("beta\n")
    with state_io.open_binary_read_file(nested_log) as bfh:
        assert bfh.read() == b"alpha\nbeta\n"

    # 3. Feed URL getters honor monkeypatches on mss._cfg_str and forward positional config_path accurately
    seen_feed_calls: list[tuple[str, str, str, str | None]] = []
    monkeypatch.setattr(
        mss,
        "_cfg_str",
        lambda sec, key, fallback="", config_path=None: seen_feed_calls.append((sec, key, fallback, config_path))
        or f"https://example.test/{key}",
    )
    assert mss.get_fear_greed_endpoint_url("/tmp/custom_macro.conf") == "https://example.test/fear_greed_endpoint_url"
    assert (
        mss.get_market_breadth_endpoint_url(config_path="/tmp/custom_macro2.conf")
        == "https://example.test/market_breadth_endpoint_url"
    )
    assert seen_feed_calls == [
        ("feeds", "fear_greed_endpoint_url", "", "/tmp/custom_macro.conf"),
        ("feeds", "market_breadth_endpoint_url", "", "/tmp/custom_macro2.conf"),
    ]

    # 4. run_tp_drop_sl_sweep tolerates None/non-Mapping eval_fn returns and priceAnalysis._bounded_plot_indices rejects bool
    rows_no_sl = run_tp_drop_sl_sweep(
        {"entry": 10.0},
        lambda _p: None,
        budget=100.0,
        tp_values=(1.5,),
        drop_values=(2.0,),
        sl_values=None,
        emit=lambda _s: None,
    )
    assert rows_no_sl == [(0.0, 1.5, 2.0, 0, 0.0, 0.0)]
    rows_with_sl = run_tp_drop_sl_sweep(
        {"entry": 10.0},
        lambda _p: None,
        budget=100.0,
        tp_values=(1.5,),
        drop_values=(2.0,),
        sl_values=(8.0,),
        emit=lambda _s: None,
    )
    assert rows_with_sl == [(0.0, 1.5, 2.0, 8.0, 0, 0.0)]
    assert len(priceAnalysis._bounded_plot_indices(10, True)) == 10


def test_wave17_kalman_reset_string_price_short_trend_nan_bool_and_repo_zero_raw_casts() -> None:
    """Wave 17: Kalman string/Decimal reset, short-trend NaN/bool guards, strict tier counts, VP/swing NaN defaults, and repo-wide zero raw open/float/int invariant."""
    # 1. KalmanTrend._reset_state & update coerce numeric strings/Decimals cleanly while rejecting bool/NaN
    kt = pw.KalmanTrend()
    kt._reset_state("95000.0", "0.04", "1000.0")  # type: ignore[arg-type]
    assert kt._x0 == pytest.approx(95000.0)
    assert kt._p00 == pytest.approx(0.4)
    assert kt._p11 == pytest.approx((95000.0 * 1e-3) ** 2)
    assert kt.last_ts == pytest.approx(1000.0)

    kt._reset_state(Decimal("96000.5"), Decimal("0.01"), Decimal("1005.0"))  # type: ignore[arg-type]
    assert kt._x0 == pytest.approx(96000.5)
    assert kt._p11 == pytest.approx((96000.5 * 1e-3) ** 2)

    for bad_px in (True, False, math.nan, math.inf, "not-a-price"):
        with pytest.raises(ValueError):
            kt._reset_state(bad_px, 0.01, 1000.0)  # type: ignore[arg-type]

    upd = kt.update(1010.0, "96100.0", "5.0")  # type: ignore[arg-type]
    assert isinstance(upd["vel"], float) and kt.last_ts == pytest.approx(1010.0)

    # 2. eval_short_trend_upward rejects bool/NaN payload values and sanitizes upward_epsilon_k
    assert pw.eval_short_trend_upward({"growth_coefficient": True, "epsilon": 0.1}) is False
    assert pw.eval_short_trend_upward({"growth_coefficient": math.nan, "slope_small": 5.0, "epsilon": 0.1}) is False
    assert (
        pw.eval_short_trend_upward(
            {"growth_coefficient": 0.5, "epsilon": 1.0}, upward_epsilon_k=math.nan
        )
        is True
    )
    assert (
        pw.eval_short_trend_upward(
            {"growth_coefficient": 0.1, "epsilon": 1.0}, upward_epsilon_k=0.0
        )
        is True
    )
    assert (
        pw.eval_short_trend_upward(
            {"growth_coefficient": 0.1, "epsilon": 1.0}, upward_epsilon_k=True  # type: ignore[arg-type]
        )
        is False
    )

    # 3. AssetGuardian tier/campaign helpers reject bool and fractional integer inputs
    for bad_cnt in (True, False, 0, -2, 2.9, "2.5"):
        with pytest.raises(ValueError):
            ag._progressive_tier_allocations(bad_cnt, 0.2)
    with pytest.raises(ValueError):
        ag._build_dynamic_buy_tiers(thresholds=[True, 5.0])
    with pytest.raises(ValueError):
        ag._ag_client_order_id("buy", "BTCUSDC", {"peak_ts": 100}, 4.0, True)

    # 4. Volume profile, swing waves, cache sample-rate span, fill-ID normalization, and survival day keys
    pts_vp = [[1700000000000 + i * 300000, 100.0 + (i % 12), 10.0 + (i % 5)] for i in range(48)]
    vp_ref = pse_mod.compute_volume_profile_metrics(pts_vp, num_bins=24, value_area_pct=0.70)
    vp_bad_args = pse_mod.compute_volume_profile_metrics(
        pts_vp, num_bins=True, value_area_pct=math.nan  # type: ignore[arg-type]
    )
    assert vp_ref and vp_bad_args == vp_ref

    sw_series = [100.0, 104.5, 100.2, 106.0, 101.5]
    assert pse_mod.compute_swing_wave_metrics(sw_series, reversal_pct=math.nan) == pse_mod.compute_swing_wave_metrics(
        sw_series, reversal_pct=3.0
    )

    assert CacheOrderManager._normalize_fill_id("00042") == "42"
    assert CacheOrderManager._normalize_fill_id("0000") == "0"
    assert CacheOrderManager._normalize_fill_id(True) is None

    c_cur = CacheCurrentPriceManager.__new__(CacheCurrentPriceManager)
    c_cur.lock = threading.RLock()
    c_cur._update_timestamps = {"BTCUSDC": pw.deque([100.0, 112.0])}
    assert c_cur.get_sustained_sample_rate("BTCUSDC", min_span_sec=5.0) == pytest.approx(12.0)
    assert c_cur.get_sustained_sample_rate("BTCUSDC", min_span_sec=math.nan) is None
    assert c_cur.get_sustained_sample_rate("BTCUSDC", min_span_sec=True) is None  # type: ignore[arg-type]

    st_mgr = CacheShortTrendManager.__new__(CacheShortTrendManager)
    st_mgr.window_seconds = [222.0, 9000.0]
    st_mgr.PRICE_CHANGE_THRESHOLD_SMALL = 0.2
    st_mgr.PRICE_CHANGE_THRESHOLD_BIG = 1.0
    thr_fn = st_mgr._build_threshold_fn({222.0: 0.15})
    assert thr_fn("BTCUSDC", 222.0) == pytest.approx(0.15)
    assert thr_fn("BTCUSDC", True) == pytest.approx(1.0)
    assert thr_fn("BTCUSDC", None) == pytest.approx(1.0)

    # 5. unpack_daily_limit_result 2-tuple contract & open-order/lot border guards
    assert order_guard.unpack_daily_limit_result((True, None)) == (True, None)
    assert order_guard.unpack_daily_limit_result((False, "daily_limit_exceeded")) == (False, "daily_limit_exceeded")
    assert order_guard.unpack_daily_limit_result((False, None)) == (False, "daily_limit")
    assert order_guard.unpack_daily_limit_result(()) == (False, "daily_limit")
    assert order_guard.unpack_daily_limit_result("false") == (False, "daily_limit")
    assert order_guard.unpack_daily_limit_result((False, 12.5, 10, "weight_cap")) == (False, "weight_cap")
    assert filter_and_normalize_open_orders([{"orderId": "1", "price": "100", "origQty": "1"}], "") == {}
    assert filter_and_normalize_open_orders([{"orderId": "1", "price": "100", "origQty": "1"}], True) == {}  # type: ignore[arg-type]
    sample_lots = [{"qty": 1.0, "price": 100.0}, {"qty": 1.0, "price": 102.0}]
    assert clip_open_lots_to_held_qty(sample_lots, math.nan) == sample_lots
    assert clip_open_lots_to_held_qty(sample_lots, 1.0, total_qty=math.nan) == [{"qty": 1.0, "price": 102.0}]

    # 6. Repo-wide invariant: zero raw open(), float(), or int() calls in any non-root Python module
    repo_root = Path(botcore.__file__).resolve().parent
    skip_dirs = {".git", "myenv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "cachedb", "node_modules"}
    root_primitives = {
        "botcore.py",
        "cachepaths.py",
        "state_io.py",
        "log.py",
        "orchestratorOS/agy/protobuf.py",
    }
    offenders: dict[str, list[tuple[int, str]]] = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(repo_root).as_posix()
            if rel.startswith("tests/") or rel in root_primitives:
                continue
            mod_tree = ast.parse(read_text_file(str(fpath)), filename=rel)
            hits: list[tuple[int, str]] = []
            for node in ast.walk(mod_tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in {"open", "float", "int"}:
                    hits.append((node.lineno, node.func.id))
                elif isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "json"
                        and node.func.attr in {"load", "loads", "dump", "dumps"}
                    ):
                        hits.append((node.lineno, f"json.{node.func.attr}"))
                    elif node.func.attr in {"read_text", "write_text", "read_bytes", "write_bytes"}:
                        hits.append((node.lineno, f".{node.func.attr}"))
                    elif node.func.attr == "add_argument":
                        for kw in node.keywords:
                            if kw.arg == "type" and isinstance(kw.value, ast.Name) and kw.value.id in {"float", "int"}:
                                hits.append((node.lineno, f"add_argument(type={kw.value.id})"))
            if hits:
                offenders[rel] = hits
    assert offenders == {}


def test_wave18_sentiment_eval_loader_execution_gate_market_flag_and_duplicate_def_invariants(tmp_path):
    """Verify Wave 18 border cases: load_cached_sentiment_advisor_eval reads sentiment_eval.json, Instrument execution gate fails closed on string false, T212 pending market flag, ParabolicSurgeGuard, alertnotifiers budget warning, window metric NaN/bool sanitization, and zero duplicate definitions repo-wide."""

    # 1. load_cached_sentiment_advisor_eval reads canonical sentiment_eval.json and normalizes fields
    sent_path = tmp_path / "sentiment_eval.json"
    atomic_write_text(
        str(sent_path),
        json.dumps(
            {
                "market_bias": "bull",
                "risk_level": "elevated",
                "recommended_action": "buy",
                "confidence": "72%",
                "ts": 1_700_000_123_000,
            }
        ),
    )
    loaded_sent = mss.load_cached_sentiment_advisor_eval(str(tmp_path))
    assert loaded_sent is not None
    assert loaded_sent["market_bias"] == "BULLISH"
    assert loaded_sent["risk_level"] == "HIGH"
    assert loaded_sent["recommended_action"] == "ACCUMULATE"
    assert loaded_sent["confidence"] == pytest.approx(0.72)
    assert loaded_sent["ts"] == pytest.approx(1_700_000_123.0)

    # 2. Instrument._check_abort_and_execution_gate fails closed when execution_enabled() returns "false" or "0"
    for disabled_ret in ("false", "0", False, None):
        ctx_dis = SimpleNamespace(abort_if=None, execution_enabled=lambda v=disabled_ret: v, reason=None)
        with pytest.raises(SubmissionRefused, match="execution_disabled"):
            Instrument._check_abort_and_execution_gate(ctx_dis)
    ctx_ok = SimpleNamespace(abort_if=None, execution_enabled=lambda: "true", reason=None)
    assert Instrument._check_abort_and_execution_gate(ctx_ok) is True

    # 3. T212 Strategy _adopt_pending_submit & _submit_pending_order honor "market": "false" as LIMIT
    t212_strat_mod = load_venue_modules("212trading", "strategy")
    captured_submits: list[tuple[Any, Any]] = []
    t212_stub = t212_strat_mod.Strategy.__new__(t212_strat_mod.Strategy)
    t212_stub.ticker = "NVDA_US_EQ"
    t212_stub.s = {"orders": [], "pending_submit": None}
    t212_stub._now = lambda: 1_700_000_000.0
    t212_stub._persist_pending_submit = lambda val: t212_stub.s.__setitem__("pending_submit", val)
    t212_stub.executor = SimpleNamespace(
        submit_outcome_with_intent=lambda _iid, _sym, _side, _qty, lim, *, market=False, **_kw: (
            captured_submits.append((lim, market)) or SimpleNamespace(state="accepted", order_id="ord-77", reason=None)
        )
    )
    pending_lim = {
        "intent_id": "i-77",
        "side": "BUY",
        "qty": 2.0,
        "limit": 125.5,
        "market": "false",
        "kind": "ENTRY",
    }
    t212_stub._submit_pending_order(pending_lim)
    assert captured_submits == [(125.5, False)]
    assert len(t212_stub.s["orders"]) == 1 and t212_stub.s["orders"][0]["market"] is False

    # 4. ParabolicSurgeGuard string-false state, _claim_budget_warning, NewsFeedCollector _seen_first_ts, and window metric NaN/bool sanitization
    pg = ParabolicSurgeGuard(surge_threshold_pct=5.0, pullback_required_pct=1.5, window_seconds=3600.0)
    pg._states["BTCUSDC"] = {
        "peak": 100.0,
        "surge_active": "false",
        "surge_start_px": 100.0,
        "ts": 1_700_000_000.0,
        "disarmed_peak_ts": 0.0,
    }
    pg_dec = pg.check("BTCUSDC", "BUY", 101.0, [(1_700_000_000.0, 100.0), (1_700_000_060.0, 101.0)])
    assert pg_dec.allowed is True and pg._states["BTCUSDC"]["surge_active"] is False

    ch_state: dict[str, Any] = {"budget_warning_sent": "false"}
    assert alert_mod._claim_budget_warning(ch_state) is True
    assert ch_state["budget_warning_sent"] is True
    assert alert_mod._claim_budget_warning(ch_state) is False

    nfc = NewsFeedCollector(history_file=str(tmp_path / "news_feed_collect.json"), start_sync=False)
    nfc._seen_first_ts = {"bad_bool": True, "nan_ts": math.nan, "good_str": "1700000100.0"}
    nfc._aggregate(SimpleNamespace(headlines=[]), 1_700_000_200.0)
    assert nfc._seen_first_ts == {"good_str": pytest.approx(1_700_000_100.0)}

    assert pse_mod._resolve_window_metric_with_fallbacks(None, 0.01) is None
    assert pse_mod._resolve_window_metric_with_fallbacks(math.nan, None) == 0.0
    assert pse_mod._resolve_window_metric_with_fallbacks(True, None) == 0.0
    assert pse_mod._resolve_window_metric_with_fallbacks(math.nan, 0.002) == pytest.approx(0.002)

    # 5. Repo-wide invariant: zero duplicate def/async def/class names within any module or class body
    repo_root = Path(botcore.__file__).resolve().parent
    skip_dirs = {".git", "myenv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "cachedb", "node_modules"}
    dup_defs: list[tuple[str, int, int, str]] = []

    def _scan_scope(body: list[ast.stmt], scope_prefix: str, rel_path: str) -> None:
        seen_names: dict[str, int] = {}
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decs = [ast.unparse(d) for d in stmt.decorator_list]
                if any("overload" in d or d.endswith((".setter", ".deleter")) for d in decs):
                    continue
                qualified = f"{scope_prefix}.{stmt.name}" if scope_prefix else stmt.name
                if stmt.name in seen_names:
                    dup_defs.append((rel_path, seen_names[stmt.name], stmt.lineno, qualified))
                seen_names[stmt.name] = stmt.lineno
                _scan_scope(stmt.body, qualified, rel_path)

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(repo_root).as_posix()
            mod_tree = ast.parse(read_text_file(str(fpath)), filename=rel)
            _scan_scope(mod_tree.body, "", rel)
    assert dup_defs == []


def test_wave19_is_market_order_state_flag_replay_ownership_and_ntfy_border_cases(tmp_path):
    """Verify Wave 19 border cases: is_market_order SSOT predicate, BaseSpotMixin._state_flag on string/dict slots, T212 pending_submit market limit=None sanitization, calculate_performance_metrics None error message, ntfy blocked string flag, ownership_inventory trade_enabled fallback, and prompt_sections inactive assetguardian filtering."""

    # 1. botcore.is_market_order & select_eligible_intrabar_orders reject string "false"/"0" market flags
    assert is_market_order({"market": "false"}) is False
    assert is_market_order({"market": "0"}) is False
    assert is_market_order({"market": "true"}) is True
    assert is_market_order({"ordertype": "market"}) is True
    assert is_market_order({"descr": {"ordertype": "MARKET"}}) is True
    assert is_market_order({"market": "false", "ordertype": "market"}) is False
    assert is_market_order("not-a-mapping") is False

    elig, ambig = select_eligible_intrabar_orders(
        [{"side": "SELL", "limit": 200.0, "market": "false"}],
        ExecutionModel(),
        high=150.0,
        low=140.0,
        limit_key="limit",
    )
    assert elig == [] and ambig is False

    # 2. BaseSpotMixin._state_flag, spot_engine._has_pending_market_exit, and T212 Strategy state/pending_submit border cases
    mixin_obj = BaseSpotMixin()
    mixin_obj.s = {
        "sl_pending": "false",
        "sl_alerted": "1",
        "sl_pending_exit": {"reason": "mhdsl_exit"},
        "empty_exit": {},
    }
    assert mixin_obj._state_flag("sl_pending") is False
    assert mixin_obj._state_flag("sl_alerted") is True
    assert mixin_obj._state_flag("sl_pending_exit") is True
    assert mixin_obj._state_flag("empty_exit") is False
    assert mixin_obj._state_flag("missing_key", True) is True

    spot_stub = spot_eng_mod.Strategy.__new__(spot_eng_mod.Strategy)
    spot_stub.s = {"orders": [{"side": "sell", "market": "false"}]}
    assert spot_stub._has_pending_market_exit() is False
    spot_stub.s["orders"].append({"side": "sell", "market": "true"})
    assert spot_stub._has_pending_market_exit() is True

    t212_strat_mod = load_venue_modules("212trading", "strategy")
    sanitized_mkt = t212_strat_mod._sanitize_t212_state(
        {
            "pending_submit": {
                "side": "SELL",
                "qty": 1.5,
                "limit": None,
                "market": "true",
                "intent_id": "mkt-1",
            }
        }
    )
    assert sanitized_mkt["pending_submit"] is not None
    assert sanitized_mkt["pending_submit"]["market"] is True
    assert sanitized_mkt["pending_submit"]["limit"] == 0.0

    sanitized_lim_bad = t212_strat_mod._sanitize_t212_state(
        {
            "pending_submit": {
                "side": "SELL",
                "qty": 1.5,
                "limit": None,
                "market": "false",
                "intent_id": "lim-1",
            }
        }
    )
    assert sanitized_lim_bad["pending_submit"] is None

    t212_dust = t212_strat_mod.Strategy.__new__(t212_strat_mod.Strategy)
    t212_dust.p = SimpleNamespace(tp_ladder=[(3.0, 0.5), (6.0, 0.5)])
    t212_dust.s = {"tp_sold_levels": [3.0], "sl_pending": "false", "sl_pending_exit": "false"}
    assert t212_dust._can_close_dust_cycle() is False
    t212_dust.s["sl_pending_exit"] = {"reason": "mhdsl"}
    assert t212_dust._can_close_dust_cycle() is True

    # 3. calculate_performance_metrics distinguishes None (non-finite) from bool
    with pytest.raises(ValueError, match="equity_curve must contain finite numbers"):
        calculate_performance_metrics([100.0, None], initial_capital=100.0)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="equity_curve cannot contain booleans"):
        calculate_performance_metrics([100.0, True], initial_capital=100.0)

    # 4. alertnotifiers ntfy blocked string flag, ownership_inventory trade_enabled fallback, and prompt_sections inactive assetguardian filtering
    assert alert_mod._is_ntfy_channel_blocked("ntfy", {"blocked": "false", "blocked_until": 9e9}, 1_700_000_000.0) is False
    assert alert_mod._is_ntfy_channel_blocked("ntfy", {"blocked": "true", "blocked_until": 9e9}, 1_700_000_000.0) is True

    inv_root = tmp_path / "inv_ws"
    (inv_root / "orchestratorOS").mkdir(parents=True)
    atomic_write_text(str(inv_root / "orchestratorOS" / "procs.manifest"), "bot|monitortrades.py|python3 monitortrades.py\n")
    atomic_write_text(str(inv_root / "config.txt"), "trade_enabled=\n")
    atomic_write_text(str(inv_root / "config.env"), "TRADE_ENABLED=false\n")
    atomic_write_text(
        str(inv_root / "instruments.conf"),
        "[BTCUSDC]\nprovider = binance\nsymbol = BTCUSDC\nenabled = true\nrole.mt = true\n",
    )
    owners = ownership_inventory.build_inventory(inv_root, commands=[])
    mt_owners = [o for o in owners if o.owner_id == "monitortrades:BTCUSDC"]
    assert len(mt_owners) == 1 and mt_owners[0].live_enabled is False

    pf_lines = ps_mod._format_portfolio_section(
        {
            "assetguardian_state": {
                "BTCUSDC": {
                    "dip_tier_1": {
                        "active": "false",
                        "pending": "false",
                        "open_orders": [],
                    }
                }
            }
        },
        include_macro_benchmarks=False,
    )
    assert all("Active AssetGuardian State" not in ln for ln in pf_lines)


def test_wave20_llm_guard_audit_hl_fills_macro_news_xstock_and_spot_flags(tmp_path, monkeypatch):
    """Verify Wave 20 border cases: GLLM1 cache_hit/allowed string flags, execution_audit reconciled_terminal string flag, Hyperliquid extract_ws_user_fills isSnapshot string flag, MacroAndNewsSensorsCollector is_high_severity string flag, prompt_sections macro_calendar string flags, kraken_xstock_watch alerted flags, and spot_engine trend_mode/cancel_requested/_replay_had_fill string flags."""

    # 1. llm_order_guard.evaluate_high_stake_order_guard: "cache_hit": "false" does not suppress notify, and cached "false" allowed still blocks
    hs_notifies: list[dict[str, Any]] = []
    monkeypatch.setattr(llm_order_guard._high_stake_guard, "is_llm_guard_exempt", lambda: False)
    monkeypatch.setattr(
        llm_order_guard._high_stake_guard,
        "HighStakeGuard",
        lambda **_kw: SimpleNamespace(
            check=lambda *_a, **_k: GuardDecision.veto(
                "GLLM1",
                "llm_veto_test",
                cache_hit="false",
                evaluated_notional_eur=1000.0,
            )
        ),
    )
    monkeypatch.setattr(
        llm_order_guard._alertnotifiers,
        "notify",
        lambda **kw: hs_notifies.append(kw),
    )
    llm_order_guard._LLM_SHADOW_NOTIFY_COOLDOWN.clear()
    margins_hs = make_test_margins(llm_guard_mode="enforce", high_stake_min_notional_usd=700.0)
    hs_ctx = SimpleNamespace()
    hs_ok, hs_reason, hs_scale = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=100.0,
        qty=10.0,
        computed_notional=1000.0,
        regime_context=hs_ctx,
        valid_context=True,
        cache_key=("BUY", ()),
        margins=margins_hs,
        bypassed=set(),
        now=1_700_000_000.0,
        notify_on_eval=True,
    )
    assert (hs_ok, hs_reason, hs_scale) == (False, "llm_veto_test", 0.0)
    assert len(hs_notifies) == 1

    cached_ctx = SimpleNamespace(
        _intelligence_decisions_by_key={
            ("BUY", ()): {
                "gemini_notional": 1000.0,
                "gemini_decision": ("false", "cached_string_veto", 1.0),
            }
        }
    )
    c_ok, c_reason, c_scale = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=100.0,
        qty=10.0,
        computed_notional=1000.0,
        regime_context=cached_ctx,
        valid_context=True,
        cache_key=("BUY", ()),
        margins=margins_hs,
        bypassed=set(),
        now=1_700_000_010.0,
        notify_on_eval=False,
    )
    assert (c_ok, c_reason, c_scale) == (False, "cached_string_veto", 0.0)

    # 2. reconcile_intents_from_execution_audit: "reconciled_terminal": "false" still reconciles live order to FILLED
    audit_dir = tmp_path / "audit_w20"
    audit_dir.mkdir(parents=True)
    day_str = datetime.fromtimestamp(1_700_000_000.0, timezone.utc).strftime("%Y-%m-%d")
    atomic_write_text(
        str(audit_dir / f"execution_audit_{day_str}.jsonl"),
        json.dumps(
            {
                "event": "order_terminal",
                "intent_id": "int-w20",
                "order_id": "oid-w20",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "venue_status": "FILLED",
                "lifecycle_status": "filled",
                "filled_qty": 2.0,
                "cost": 200.0,
                "avg_fill_price": 100.0,
                "ts": 1_700_000_005.0,
            }
        )
        + "\n",
    )
    intents_w20 = [
        {
            "intent_id": "int-w20",
            "status": "EXECUTED",
            "timestamp": 1_700_000_000.0,
            "execution_payload": {
                "orderId": "oid-w20",
                "status": "NEW",
                "reconciled_terminal": "false",
                "_ts_backfill_checked": "false",
            },
        }
    ]
    assert reconcile_intents_from_execution_audit(intents_w20, str(audit_dir)) is True
    assert intents_w20[0]["execution_payload"]["status"] == "FILLED"
    assert intents_w20[0]["execution_payload"]["reconciled_terminal"] is True

    # 3. HyperliquidProvider.extract_ws_user_fills: "isSnapshot": "false" is incremental (does not zero untouched symbols)
    hl_parsed = HyperliquidProvider.extract_ws_user_fills(
        {"isSnapshot": "false", "fills": [{"coin": "@107", "px": "25.0", "sz": "1.0"}]},
        ["HYPEUSDC", "PURRUSDC"],
        candidate_mids_fn=lambda sym: ("@107",) if sym == "HYPEUSDC" else ("@1",),
    )
    assert hl_parsed is not None
    is_snap, fills_by_sym = hl_parsed
    assert is_snap is False
    assert list(fills_by_sym.keys()) == ["HYPEUSDC"]

    # 4. MacroAndNewsSensorsCollector.collect: "is_high_severity": "false" stays False
    _macro_s, breaking = MacroAndNewsSensorsCollector().collect(
        cachedb_dir=str(tmp_path),
        now_ts=1_700_000_000.0,
        read_json_fn=lambda p: (
            {
                "items": [
                    {
                        "title": "Routine crypto market recap",
                        "source": "CoinDesk",
                        "first_seen_ts": 1_700_000_000.0 - 300.0,
                        "is_high_severity": "false",
                        "matched_keywords": [],
                    }
                ]
            }
            if str(p).endswith("news_feed_collect.json")
            else {}
        ),
        allow_network=False,
        persist=False,
    )
    assert len(breaking) == 1 and breaking[0]["is_high_severity"] is False

    # 5. prompt_sections._format_macro_and_news_section: macro_calendar string "false" flags
    mc_out = "\n".join(
        ps_mod._format_macro_and_news_section(
            SimpleNamespace(
                macro_calendar={
                    "session_label": "US_PRE",
                    "next_high_impact_event": "US Regular Open",
                    "next_event_is_high_impact": "false",
                    "active_release_window": "false",
                    "next_scheduled_event": "CPI",
                    "fomc_today": "false",
                }
            )
        )
    )
    assert 'next_routine_session_window="US Regular Open"' in mc_out
    assert 'next_scheduled="CPI"' not in mc_out
    assert "fomc_today=False" in mc_out

    # 6. kraken_xstock_watch.check_levels: "alerted_tp": "false" still triggers alert
    xw_alerts: list[dict[str, Any]] = []
    monkeypatch.setattr(xstock_watch, "notify", lambda **kw: xw_alerts.append(kw))
    st_xw = {
        "pair": "AAPLXUSD",
        "allocated": {"qty": 10.0},
        "alerted_tp": "false",
        "alerted_tp2": "false",
        "alerted_sl": "false",
    }
    with temporary_environ({"XSTOCK_TP2_ALERT_PCT": "25.0"}):
        xstock_watch.check_levels(
            SimpleNamespace(last_price=lambda _p: 115.0),
            st_xw,
            100.0,
            10.0,
            10.0,
            "",
            False,
        )
    assert st_xw["alerted_tp"] is True
    assert len(xw_alerts) == 1

    # 7. spot_engine._is_trend_up and _dispatch_take_profit string boolean flags
    spot_stub = spot_eng_mod.Strategy.__new__(spot_eng_mod.Strategy)
    spot_stub.p = SimpleNamespace(
        enable_takeprofit=True,
        takeprofit_pct=3.0,
        tp_dynamic_flat=False,
        tp_trend_hold=False,
        tp_tranches=[],
    )
    spot_stub.price_dec = 2
    spot_stub.vol_dec = 4
    spot_stub.ordermin = 0.01
    spot_stub._dust_safe_qty = lambda q: q
    spot_stub._is_trend_direction = lambda *_a, **_k: False
    spot_stub.s = {
        "trend_mode": "false",
        "trail_peak": None,
        "trail_stop": None,
        "tp_sold_levels": [],
        "orders": [
            {
                "side": "sell",
                "price": 103.0,
                "vol": 1.0,
                "cancel_requested": "false",
                "_replay_had_fill": "false",
            }
        ],
    }
    assert spot_stub._is_trend_up(None) is False
    spot_stub.s["trend_mode"] = "true"
    assert spot_stub._is_trend_up(None) is True

    canceled_tp: list[dict[str, Any]] = []
    placed_tp: list[tuple[Any, Any]] = []
    spot_stub._cancel_order = lambda o: canceled_tp.append(o)
    spot_stub._place = lambda *a, **k: placed_tp.append((a, k)) or True
    assert spot_stub._dispatch_take_profit(100.0, 100.0, 1.0) is False
    assert canceled_tp == [] and placed_tp == []
    spot_stub.s["orders"][0]["cancel_requested"] = "true"
    assert spot_stub._dispatch_take_profit(100.0, 100.0, 1.0) is False
    assert len(canceled_tp) == 1


def test_wave21_callable_gates_sre_sibling_structure_audit_zero_qty_and_replay_ratchet():
    """Verify Wave 21 border cases: Instrument._is_execution_gate_enabled fail-closed on NaN/Inf/containers/string-false, _is_aborted no spurious abort on string-false/NaN, prompt_sections SRE fleet/error/VPN flags and sibling_structure_presence guard, execution_audit _is_open_resting_detail and zero-string filled_qty dedup, ratchet_post_exit_peak_sell_price sl_rebuy string/dict handling, and PriceChecker has_data string-false."""

    # 1. Instrument._is_execution_gate_enabled (fail-closed) and _is_aborted (no spurious abort)
    assert Instrument._is_execution_gate_enabled(None) is True
    assert Instrument._is_execution_gate_enabled(lambda: True) is True
    assert Instrument._is_execution_gate_enabled(lambda: "true") is True
    assert Instrument._is_execution_gate_enabled(lambda: "false") is False
    assert Instrument._is_execution_gate_enabled(lambda: "0") is False
    assert Instrument._is_execution_gate_enabled(lambda: float("nan")) is False
    assert Instrument._is_execution_gate_enabled(lambda: float("inf")) is False
    assert Instrument._is_execution_gate_enabled(lambda: [0]) is False
    assert Instrument._is_execution_gate_enabled(lambda: {"live": False}) is False

    assert _is_aborted(None) is False
    assert _is_aborted(lambda: "false") is False
    assert _is_aborted(lambda: "0") is False
    assert _is_aborted(lambda: float("nan")) is False
    assert _is_aborted(lambda: "true") is True
    assert _is_aborted(lambda: True) is True

    # 2. prompt_sections._format_sre_and_fleet_section & _format_market_prices_section (sibling_structure_presence)
    sre_out = "\n".join(
        ps_mod._format_sre_and_fleet_section(
            SimpleNamespace(
                active_heartbeats={"bot.heartbeat": 1.0},
                state_files=[],
                fleet_processes=[{"name": "bot", "is_running": "false", "heartbeat_fresh": "false"}],
                system_and_vpn={"vpn_tunnel_up": "false"},
            ),
            SimpleNamespace(
                total_scanned_files=1,
                error_count=1,
                warning_count=0,
                persistent_errors=[
                    {"unresolved_at_tail": "false", "count": 2, "file": "bot.log", "sample": "err"}
                ],
                critical_events=[],
                guard_interventions=[],
            ),
            6.0,
        )
    )
    assert "0/1 running | STOPPED: ['bot']" in sre_out
    assert "[RECURRING x2 in bot.log]" in sre_out
    assert "DOWN / NOT PRESENT" in sre_out

    mkt_out = "\n".join(
        ps_mod._format_market_prices_section(
            {
                "BTCUSDC": {
                    "price": 100.0,
                    "chg_6h_pct": -2.0,
                    "low_15m": 95.0,
                    "low_24h": 94.0,
                },
                "BTCUSDT": {
                    "price": 100.0,
                    "recent_1h_low_px": 99.0,
                    "prior_6h_low_px": 98.0,
                    "bottom_structure_6h": "HIGHER_LOW",
                },
            }
        )
    )
    assert "HIGHER_LOW" not in mkt_out

    # 3. execution_audit._is_open_resting_detail & summarize_recent_ai_intents zero-string filled_qty dedup
    assert _is_open_resting_detail({"is_open_in_venue": "false", "exec_status": "OPEN_RESTING"}) is False
    assert _is_open_resting_detail({"is_open_in_venue": "true", "exec_status": "OPEN_RESTING"}) is True

    summarized = summarize_recent_ai_intents(
        [
            {
                "intent_id": "old-cancel",
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "status": "EXECUTED",
                "timestamp": 1_700_000_000.0,
                "execution_payload": {
                    "orderId": "oid-old",
                    "status": "CANCELED",
                    "executedQty": "0.00000000",
                },
            },
            {
                "intent_id": "new-fill",
                "decision": "BUY",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "status": "EXECUTED",
                "timestamp": 1_700_000_060.0,
                "execution_payload": {
                    "orderId": "oid-new",
                    "status": "FILLED",
                    "executedQty": "1.0",
                    "price": "100.0",
                    "cummulativeQuoteQty": "100.0",
                },
            },
        ],
        "crypto",
        1_700_000_120.0,
        lambda o, **_kw: 100.0 if o else None,
    )
    assert len(summarized) == 1 and summarized[0]["order_id"] == "oid-new"

    # 4. ratchet_post_exit_peak_sell_price sl_rebuy string vs dict & PriceChecker has_data string-false
    st_r = {"qty": 0.0, "last_sell_price": 100.0, "sl_rebuy": "false"}
    ratchet_post_exit_peak_sell_price(st_r, 108.0)
    assert st_r["last_sell_price"] == 108.0

    st_r_dict = {"qty": 0.0, "last_sell_price": 100.0, "sl_rebuy": {"low": 95.0}}
    ratchet_post_exit_peak_sell_price(st_r_dict, 108.0)
    assert st_r_dict["last_sell_price"] == 100.0

    pc = PriceChecker.__new__(PriceChecker)
    pc._calculate_24h_stats = lambda _s: {"has_data": "false", "error": "empty"}
    assert pc.check_symbol("BTCUSDC") == []


def test_wave22_first_pos_float_cli_bounds_replay_end_exit_and_string_zero_fallbacks():
    # 1. SSOT botcore.first_pos_float skips None, 0, "0", "0.00000000", NaN, Inf, bool, and negative numbers
    assert first_pos_float(None, 0, "0", "0.00000000", float("nan"), float("inf"), True, -5.0, "125.5") == 125.5
    assert first_pos_float("0", "0.0", float("nan"), default=3.5) == 3.5
    assert first_pos_float("0", None) is None

    # 2. validate_common_backtest_cli_bounds rejects NaN, Inf, bool, and fractional maxdca (no fail-open)
    valid_kw = dict(
        budget=1000.0,
        entry=100.0,
        dca=50.0,
        maxdca=5,
        fee=0.25,
        sl=10.0,
        disc=0.2,
        tp=2.0,
        drop=2.0,
    )
    assert validate_common_backtest_cli_bounds(**valid_kw) is None
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "budget": float("nan")}) == "budget must be > 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "budget": float("inf")}) == "budget must be > 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "entry": float("nan")}) == "entry and dca must be > 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "dca": True}) == "entry and dca must be > 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "maxdca": True}) == "maxdca must be >= 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "maxdca": 1.5}) == "maxdca must be >= 0"
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "fee": float("nan")}) == "fee/sl/disc must be >= 0"
    assert (
        validate_common_backtest_cli_bounds(
            **valid_kw,
            pct_label="fee/sl/disc/trail",
            extra_nonneg=(float("nan"),),
        )
        == "fee/sl/disc/trail must be >= 0"
    )
    assert (
        validate_common_backtest_cli_bounds(
            **valid_kw,
            pct_label="fee/sl/disc/trail",
            extra_pct_lt_100=(float("inf"),),
        )
        == "fee/sl/disc/trail must be < 100"
    )
    assert validate_common_backtest_cli_bounds(**{**valid_kw, "tp": float("nan")}) == "tp and drop must be > 0"

    # 3. iter_end_of_replay_market_sell_orders guards empty OHLC, NaN/string-zero open_qty, and non-dict orders
    with pytest.raises(ValueError, match="ohlc cannot be empty"):
        iter_end_of_replay_market_sell_orders([], [], 1.0)
    orders_sample = [
        None,
        {"side": " BUY ", "market": True, "vol": 1.0},
        {"side": " SELL ", "market": "false", "vol": 1.0},
        {"side": " sell ", "market": "true", "vol": 1.0},
    ]
    lc_nan, elig_nan = iter_end_of_replay_market_sell_orders([(100.0, 105.0, 99.0, 102.5)], orders_sample, float("nan"))
    assert lc_nan == 102.5 and elig_nan == []
    lc_ok, elig_ok = iter_end_of_replay_market_sell_orders([(100.0, 105.0, 99.0, 102.5)], orders_sample, 0.5)
    assert lc_ok == 102.5 and len(elig_ok) == 1 and elig_ok[0]["vol"] == 1.0

    # 4. summarize_recent_ai_intents resolves fill_px and ref_px when venue history has string-zero "0.00000000"
    summarized = summarize_recent_ai_intents(
        [
            {
                "intent_id": "mkt-zero-str",
                "decision": "BUY",
                "order_type": "MARKET",
                "symbol": "BTCUSDC",
                "venue": "binance",
                "status": "EXECUTED",
                "timestamp": 1_700_000_100.0,
                "requested_price": "64500.0",
                "limit_price": "0.00000000",
                "reference_price": "0.00000000",
                "history": [{"action": "EXECUTED", "price": "0.00000000"}],
                "execution_payload": {
                    "orderId": "oid-mkt-1",
                    "status": "FILLED",
                    "executedQty": "0.01",
                    "price": "0.00000000",
                },
            },
            {
                "intent_id": "mkt-ref-zero-str",
                "decision": "BUY",
                "order_type": "MARKET",
                "symbol": "ETHUSDC",
                "venue": "binance",
                "status": "EXECUTED",
                "timestamp": 1_700_000_110.0,
                "requested_price": "2000.0",
                "limit_price": "0.00000000",
                "reference_price": "0.00000000",
                "execution_payload": {
                    "orderId": "oid-mkt-2",
                    "status": "FILLED",
                    "executedQty": "1.0",
                    "cummulativeQuoteQty": "2004.0",
                    "price": "0.00000000",
                },
            },
        ],
        "crypto",
        1_700_000_120.0,
        lambda _o, **_kw: None,
    )
    by_oid = {row["order_id"]: row for row in summarized}
    assert by_oid["oid-mkt-1"]["fill_px"] == 64500.0
    assert by_oid["oid-mkt-2"]["fill_px"] == 2004.0
    assert by_oid["oid-mkt-2"]["slippage_bps"] == 20.0

    # 5. T212Provider.normalize_history_items & rtrade._LivePairVenue._ticket_from_pending string-zero fallbacks
    grouped_t212 = T212Provider.normalize_history_items(
        [
            {
                "order": {
                    "id": 777,
                    "ticker": "NVDA_US_EQ",
                    "status": "FILLED",
                    "orderedQuantity": 2.0,
                    "filledQuantity": "0.00",
                    "fillPrice": "0.00",
                    "limitPrice": "132.25",
                    "dateExecuted": "2026-01-15T15:00:00Z",
                },
                "fill": None,
            }
        ]
    )
    assert "NVDA_US_EQ" in grouped_t212
    assert grouped_t212["NVDA_US_EQ"][0]["price"] == 132.25
    assert grouped_t212["NVDA_US_EQ"][0]["qty"] == 2.0

    venue_rt = rtrade._LivePairVenue.__new__(rtrade._LivePairVenue)
    venue_rt.current_price = lambda: 0.0
    ticket = venue_rt._ticket_from_pending(
        {
            "order_id": "rt-oid-1",
            "side": "BUY",
            "kind": "limit_0",
            "submitted_price": "0.00000000",
            "requested_price": "61250.0",
            "submitted_qty": "0.00000000",
            "requested_qty": "0.02",
        },
        "pair-1",
    )
    assert ticket is not None and ticket.price == 61250.0 and ticket.qty == 0.02

    # 6. T212 Strategy._close_cycle & portfolio_evaluator.parse_venue_cycle_state string-zero fallbacks
    t212_strat_mod = load_venue_modules("212trading", "strategy")
    t212_strat_cls = t212_strat_mod.Strategy
    st_inst = t212_strat_cls.__new__(t212_strat_cls)
    st_inst.ticker = "NVDA_US_EQ"
    st_inst.yahoo_sym = "NVDA"
    st_inst.ccy = "USD"
    st_inst.desktop = False
    st_inst.dry_run = True
    st_inst.exit_manager = None
    st_inst.p = SimpleNamespace(sl_rebuy_enabled=True, sl_rebuy_bounce_pct=1.5, tp_ladder=[])
    st_inst.s = {
        "cycle": 1,
        "qty": 0.0,
        "spent_cash": 0.0,
        "dca_buys": 0,
        "entry_price": None,
        "last_buy_price": None,
        "last_sell_price": "0.00",
        "orders": [],
        "tp_sold_levels": [],
        "sl_pending": True,
        "realized_pnl_usd": 0.0,
        "realized_net_usd": 0.0,
        "fees_usd": 0.0,
    }
    st_inst._now = lambda: 1_700_000_000.0
    st_inst._save = lambda: None
    exit_anchor = first_pos_float(st_inst.s.get("last_sell_price"), 140.5) or 140.5
    st_inst._close_cycle(exit_anchor, exit_kind="SL")
    assert st_inst.s["sl_rebuy"] == {"low": 140.5, "sell_price": 140.5}
    assert st_inst.s["last_sell_price"] == 140.5

    cyc_parsed = parse_venue_cycle_state(
        {
            "qty": 0.5,
            "cost_usd": 0.0,
            "avg_cost": "0.00000000",
            "avg_cost_px": "62000.0",
            "pos_peak": "0.0",
            "trail_peak": "65000.0",
            "orders": [{"side": "SELL", "px": "0.0", "price": "66000.0", "sz": "0.0", "qty": "0.5", "kind": "TP"}],
        },
        "BTCUSDC",
        {"BTCUSDC": {"price": 63000.0}},
    )
    assert cyc_parsed["avg_cost_px"] == 62000.0
    assert cyc_parsed["pos_peak"] == 65000.0
    assert cyc_parsed["resting_orders"][0]["px"] == 66000.0
    assert cyc_parsed["resting_orders"][0]["sz"] == 0.5

    tca_orders = OpenLimitOrdersTcaEvaluator().evaluate(
        [{"symbol": "BTCUSDC", "venue": "binance", "side": "BUY", "price": "0.0", "limit_price": "60000.0"}],
        {"BTCUSDC": {"price": 61000.0, "vol_1h_pct": 1.2}},
        {},
    )
    assert "fill_prob_1h_pct" in tca_orders[0]


def test_wave23_first_epoch_sec_timestamp_zero_shadow_quota_expiry_and_wf_cli_bounds(tmp_path, monkeypatch):
    # 1. SSOT botcore.first_epoch_sec skips None/0/'0'/NaN/Inf/bool and normalizes ms + ISO-8601
    assert first_epoch_sec(None, 0, "0", "0.0", float("nan"), float("inf"), True, -10.0, 1_700_000_250_000) == 1_700_000_250.0
    assert (
        first_epoch_sec("0", "2026-01-15T15:00:00Z", allow_iso=True)
        == datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc).timestamp()
    )
    assert first_epoch_sec("0", None, default=9.5) == 9.5

    # 2. extract_timestamp_ms & extract_timestamp_s do not let unset 0/'0' shadow valid positive timestamps
    assert extract_timestamp_ms({"time": 0, "updateTime": 1_700_000_123_000}) == 1_700_000_123_000
    assert extract_timestamp_ms({"timestamp": "0", "transactTime": "1700000456000"}) == 1_700_000_456_000
    assert extract_timestamp_ms({"time": 0}) == 0
    iso_exec_s = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc).timestamp()
    assert extract_timestamp_s({"timestamp": "0", "dateExecuted": "2026-01-15T15:00:00Z"}) == pytest.approx(iso_exec_s)
    assert extract_timestamp_s({"time": "0"}) == 0.0

    # 3. CacheAssetValueManager, heartbeat resolver, and open-order timestamp loader skip string-zero "0"
    av_summary = CacheAssetValueManager.extract_asset_value_summary(
        {"t": "0", "i": {"total_value_usdc": 1200.0, "t": 1_700_000_000.0}},
        now_ts=1_700_000_300.0,
    )
    assert av_summary["total_value_usdc_age_min"] == 5.0

    hb_entry = tr_ssot._resolve_heartbeat_entry(
        str(tmp_path / "missing.heartbeat"),
        {"ts": "0", "timestamp": 1_700_000_240.0, "status": "ok"},
        now_ts=1_700_000_300.0,
        max_age=120.0,
        allow_mtime_fallback=False,
    )
    assert hb_entry is not None and hb_entry["age_seconds"] == 60.0

    oo_ts_map = _load_open_order_timestamps(
        None,
        {"open_orders:binance": [{"orderId": "oid-z1", "time": "0", "timestamp": 1_700_000_500_000}]},
    )
    assert oo_ts_map["oid-z1"] == 1_700_000_500.0

    # 4. GeminiQuotaRouterMixin credits & OAuth token expiry (string-zero nested expiry + millisecond expiry)
    assert (
        lqr_mod.GeminiQuotaRouterMixin._extract_available_credits(
            {"paidTier": {"id": "pro", "availableCredits": [{"remainingCredits": 0, "credits": 500}]}}
        )
        == 0.0
    )
    assert (
        lqr_mod.GeminiQuotaRouterMixin._extract_available_credits(
            {"paidTier": {"id": "pro", "availableCredits": [{"creditAmount": "0.0", "amount": "18.5"}]}}
        )
        == 18.5
    )

    bot_cli = tmp_path / "bot_cli"
    ui_cli = tmp_path / "ui_cli"
    bot_cli.mkdir()
    ui_cli.mkdir()
    now_now = time.time()
    # bot_cli token has nested "expiry_timestamp": "0" trying to mask an expired millisecond timestamp
    atomic_write_text(
        str(bot_cli / "antigravity-oauth-token"),
        json.dumps(
            {
                "token": {"access_token": "tok-expired-masked", "expiry_timestamp": "0"},
                "expiry_timestamp": int((now_now - 3600.0) * 1000),
            }
        ),
    )
    atomic_write_text(
        str(ui_cli / "antigravity-oauth-token"),
        json.dumps(
            {
                "token": {"access_token": "tok-fresh-ms", "expiry_timestamp": int((now_now + 3600.0) * 1000)},
            }
        ),
    )
    seen_auth: list[str] = []
    monkeypatch.setattr(lqr_mod, "_QUOTA_MEM_CACHE", {}, raising=False)
    monkeypatch.setattr(lqr_mod.GeminiQuotaRouterMixin, "_QUOTA_MEM_CACHE", {})
    monkeypatch.setattr(
        lqr_mod.GeminiQuotaRouterMixin,
        "_get_quota_state_path",
        staticmethod(lambda: str(tmp_path / "quota_state.json")),
    )
    monkeypatch.setattr(lqr_mod, "resolve_bot_cli_base", lambda: str(bot_cli))
    monkeypatch.setattr(lqr_mod, "resolve_ui_cli_base", lambda: str(ui_cli))
    monkeypatch.setattr(
        lqr_mod,
        "fetch_json_post",
        lambda url, **_kw: (
            seen_auth.append(_kw.get("headers", {}).get("Authorization", ""))
            or (
                {"buckets": [{"modelId": "gemini-2.5-flash", "remainingFraction": 0.8}]}
                if "retrieveUserQuota" in url
                else {"paidTier": {"id": "t1", "availableCredits": []}}
            )
        ),
    )
    quota_snap = lqr_mod.GeminiQuotaRouterMixin.probe_live_model_quota(
        bot_cli_base=str(bot_cli), force_refresh=True
    )
    assert quota_snap is not None
    assert seen_auth and all(hdr == "Bearer tok-fresh-ms" for hdr in seen_auth)

    # 5. build_execution_model_from_args & kraken/shadow_live CLI NaN/Inf/fractional/fee>=100 bounds
    ap_wf = argparse.ArgumentParser()
    base_ns = dict(
        fee=0.25,
        train=100,
        validation=50,
        test=50,
        step=25,
        warmup=10,
        spread_bps=0.0,
        market_slippage_bps=0.0,
        partial_fill_ratio=1.0,
        intrabar_policy="worst_case",
    )
    assert isinstance(build_execution_model_from_args(ap_wf, argparse.Namespace(**base_ns)), ExecutionModel)
    for bad_wf_override in (
        {"train": float("nan")},
        {"step": float("inf")},
        {"validation": 12.5},
        {"warmup": float("nan")},
        {"warmup": 2.5},
    ):
        with pytest.raises(SystemExit):
            build_execution_model_from_args(ap_wf, argparse.Namespace(**{**base_ns, **bad_wf_override}))

    for bad_shadow_argv in (
        ["--fee", "100"],
        ["--fee", "nan"],
        ["--loop", "-1"],
        ["--loop", "nan"],
    ):
        with pytest.raises(SystemExit):
            kraken_shadow.main(bad_shadow_argv)


def test_wave24_cli_parsers_patch_seam_defaults_rtrade_recovery_and_prune_overflow(tmp_path, monkeypatch):
    """Verify Wave 24 border cases: Level-1 SSOT CLI parsers, import-time defaults for resolve_patched_attr across 5 modules, rtrade _canonical_intent & recover_intent terminal state preservation, MacroAnalyzer & llm_order_guard & check_cache_coherence zero-string fallbacks, and prune_llm_threads retention_days overflow clamp."""

    # 1. Level-1 SSOT CLI parsers & daemon CLI argument builders reject NaN/Inf/bool/non-positive
    assert parse_cli_float("-1.5") == pytest.approx(-1.5)
    assert parse_pos_cli_float("0.25") == pytest.approx(0.25)
    assert parse_nonneg_cli_float("0") == pytest.approx(0.0)
    assert parse_cli_int("-7") == -7
    assert parse_pos_cli_int("12") == 12
    assert parse_pos_cli_int("2.0") == 2
    assert parse_nonneg_cli_int("0") == 0
    for parser_fn, bad_inputs in (
        (parse_cli_float, ("nan", "inf", "-inf", True, "abc")),
        (parse_pos_cli_float, ("0", "-0.1", "nan", "inf", False)),
        (parse_nonneg_cli_float, ("-0.01", "nan", "inf", True)),
        (parse_cli_int, ("3.5", "nan", "inf", True)),
        (parse_pos_cli_int, ("0", "-1", "2.5", "nan", False)),
        (parse_nonneg_cli_int, ("-1", "1.5", "nan", True)),
    ):
        for raw_bad in bad_inputs:
            with pytest.raises(argparse.ArgumentTypeError):
                parser_fn(raw_bad)

    ext_ap = add_external_telemetry_cli_args(argparse.ArgumentParser())
    for bad_ext_argv in (["--ob-interval", "nan"], ["--ob-interval", "0"], ["--whale-interval", "-5"]):
        with pytest.raises(SystemExit):
            ext_ap.parse_args(bad_ext_argv)
    macro_ap = add_macro_analyzer_cli_args(argparse.ArgumentParser())
    for bad_macro_argv in (["--macro-llm-interval", "inf"], ["--advisor-interval", "0"]):
        with pytest.raises(SystemExit):
            macro_ap.parse_args(bad_macro_argv)

    # 2. resolve_patched_attr import-time _ORIG_* defaults honor local module monkeypatches across all 5 modules
    seen_mtc_urls: list[str] = []
    monkeypatch.setattr(
        mtc_mod,
        "fetch_public_json",
        lambda url, **_kw: seen_mtc_urls.append(str(url)) or {},
    )
    MacroBenchmarksSnapshotCollector().collect(
        cachedb_dir=str(tmp_path), allow_network=True, persist=False
    )
    assert len(seen_mtc_urls) >= 1

    assert lex_ssot._cfg_str is mss._ORIG_MACRO_CFG_ATTRS["_cfg_str"]
    monkeypatch.setattr(
        mss,
        "_cfg_str",
        lambda sec, key, fallback="", config_path=None: "3/4" if key == "r" else ("42" if key == "i" else "https://mock.feed/fg"),
    )
    assert mss._cfg_ratio("s", "r", 0.1) == pytest.approx(0.75)
    assert mss._cfg_int("s", "i", 1) == 42
    assert mss.get_fear_greed_endpoint_url() == "https://mock.feed/fg"

    monkeypatch.setattr(
        order_guard,
        "get_weight_for_cash_permission_at_quant_time",
        lambda sym, side: 0.42 if sym == "BTCUSDC" else None,
    )
    assert order_guard.resolve_trade_weight("BTCUSDC", "BUY") == pytest.approx(0.42)

    assert "monitortrades" in sys.modules
    monkeypatch.setattr(mt_pa_mod, "round_provider_qty", lambda _prov, _sym, q: round(q, 4))
    monkeypatch.setattr(mt_pa_mod, "check_order_filter_refusal", lambda *_a, **_kw: "MOCK_FILTER_REFUSED")
    ok_scaled, q_scaled = mt_pa_mod._validate_scaled_buy_qty(
        SimpleNamespace(), "BTCUSDC", 60000.0, 1.0, 0.5, 0.01, "Re-entry"
    )
    assert ok_scaled is False and q_scaled == pytest.approx(0.5)

    seen_ohlc_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pbase_mod,
        "load_local_module",
        lambda mdir, mname: (
            seen_ohlc_calls.append((str(mdir), str(mname)))
            or SimpleNamespace(fetch_ohlc_candles=lambda sym, interval=60, drop_incomplete=True, **kw: [{"close": 25.5}])
        ),
    )
    assert KrakenProvider().fetch_ohlc_candles("HYPEEUR", interval=60) == [{"close": 25.5}]
    assert len(seen_ohlc_calls) == 1

    # 3. rtrade._LivePairVenue._canonical_intent & recover_intent terminal state preservation
    persisted_snapshots: list[dict] = []
    clock_box = [1_700_010_000.0]
    venue = rtrade._LivePairVenue.__new__(rtrade._LivePairVenue)
    venue.symbol = "TAOUSDC"
    venue._recovery_clock = lambda: clock_box[0]
    venue._intent_id = lambda pid, sd, kd: f"{pid}:{kd}:{sd}"
    venue._persist_callback = lambda *_a, **_kw: (lambda snap: persisted_snapshots.append(dict(snap)))

    pair_rec = {"pair_id": "p1", "symbol": "TAOUSDC", "start_side": "BUY", "qty": 1.5, "created_ts": 1_700_000_111.0}
    canon = venue._canonical_intent(
        pair_rec,
        {"client_order_id": "cid-1", "side": "BUY", "kind": "limit", "created_at": "0"},
    )
    assert canon["created_at"] == pytest.approx(1_700_000_111.0)

    # 3a. absence_confirmed_no_reuse must NOT be overwritten to horizon_exhausted when horizon elapses
    absent_stored = {
        "client_order_id": "cid-abs",
        "side": "BUY",
        "kind": "limit",
        "recovery_state": "absence_confirmed_no_reuse",
        "recovery_started_at": 1_700_000_000.0,
    }
    with pytest.raises(rtrade._IntentRecoveryPending, match="requires manual reconciliation"):
        venue.recover_intent(pair_rec, absent_stored)
    assert persisted_snapshots[-1]["recovery_state"] == "absence_confirmed_no_reuse"
    assert "recovery_horizon_exhausted_at" not in persisted_snapshots[-1]

    # 3b. First horizon hit sets recovery_horizon_exhausted_at=t1; subsequent tick t2 preserves t1 and raises manual reconciliation
    exhaust_stored = {
        "client_order_id": "cid-exh",
        "side": "SELL",
        "kind": "limit",
        "recovery_started_at": 1_700_000_000.0,
    }
    with pytest.raises(rtrade._IntentRecoveryPending, match="recovery horizon is exhausted"):
        venue.recover_intent(pair_rec, exhaust_stored)
    first_persisted = dict(persisted_snapshots[-1])
    assert first_persisted["recovery_state"] == "horizon_exhausted"
    assert first_persisted["recovery_horizon_exhausted_at"] == pytest.approx(1_700_010_000.0)

    clock_box[0] = 1_700_020_500.0
    with pytest.raises(rtrade._IntentRecoveryPending, match="requires manual reconciliation"):
        venue.recover_intent(pair_rec, first_persisted)
    assert persisted_snapshots[-1]["recovery_state"] == "horizon_exhausted"
    assert persisted_snapshots[-1]["recovery_horizon_exhausted_at"] == pytest.approx(1_700_010_000.0)

    # 4. MacroAnalyzer, llm_order_guard, check_cache_coherence, & prune_llm_threads border cases
    monkeypatch.setattr(
        macro_an_mod._geo_analyzer,
        "GeopoliticalThreatAnalyzer",
        lambda **_kw: SimpleNamespace(_last_eval_ts=1_700_000_200_000),
    )
    monkeypatch.setattr(
        macro_an_mod._sentiment_advisor,
        "SentimentAdvisor",
        lambda **_kw: SimpleNamespace(_cached_assessment={"ts": "0"}, _last_eval_ts=1_700_000_500_000),
    )
    ma_inst = macro_an_mod.MacroAnalyzer(cache_dir=str(tmp_path))
    assert ma_inst._last_macro_llm_ts == pytest.approx(1_700_000_200.0)
    assert ma_inst._last_advisor_ts == pytest.approx(1_700_000_500.0)

    seen_hs_dur: list[Any] = []
    allowed_flag_box: list[Any] = [True]
    monkeypatch.setattr(
        llm_order_guard._high_stake_guard,
        "HighStakeGuard",
        lambda **_kw: SimpleNamespace(
            check=lambda _sym, _side, _px, _qty, **ckw: (
                seen_hs_dur.append(ckw.get("dur_sec"))
                or SimpleNamespace(allowed=allowed_flag_box[0], reason="veto_str_false", suggested_scale=1.0, metadata={})
            )
        ),
    )
    monkeypatch.setattr(llm_order_guard._high_stake_guard, "is_llm_guard_exempt", lambda: False)
    hs_margins = make_test_margins(
        intelligence_mode="enforce",
        llm_guard_mode="enforce",
        high_stake_min_notional_usd=500.0,
        high_stake_min_notional_buy_usd=500.0,
    )
    ok_hs, _, _ = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=60000.0,
        qty=0.02,
        computed_notional=1200.0,
        regime_context={"trend_duration_seconds": "0", "resolved_duration_sec": 3600.0},
        valid_context=False,
        cache_key=("BTCUSDC", "BUY"),
        margins=hs_margins,
        bypassed=set(),
        now=1_700_000_000.0,
        notify_on_eval=False,
    )
    assert ok_hs is True
    assert seen_hs_dur == [pytest.approx(3600.0)]

    allowed_flag_box[0] = "false"
    ok_hs_false, reason_hs_false, scale_hs_false = llm_order_guard.evaluate_high_stake_order_guard(
        symbol="BTCUSDC",
        side="BUY",
        price=60000.0,
        qty=0.02,
        computed_notional=1200.0,
        regime_context={"trend_duration_seconds": "0", "resolved_duration_sec": 3600.0},
        valid_context=False,
        cache_key=("BTCUSDC", "BUY"),
        margins=hs_margins,
        bypassed=set(),
        now=1_700_000_000.0,
        notify_on_eval=False,
    )
    assert ok_hs_false is False and reason_hs_false == "veto_str_false" and scale_hs_false == 0.0

    coh_dir = tmp_path / "coh_cachedb"
    coh_dir.mkdir()
    now_coh = time.time()
    atomic_write_text(
        str(coh_dir / "cache_short_trend.json"),
        json.dumps({"items": {"BTCUSDC": {"trend": 1, "price": 62000.0}}, "fetchtime": {"BTCUSDC": now_coh * 1000.0}}),
    )
    atomic_write_text(
        str(coh_dir / "cache_current_price.json"),
        json.dumps({"items": {"BTCUSDC": {"price": 62000.0, "ts": "0"}}, "fetchtime": {"BTCUSDC": now_coh}}),
    )
    monkeypatch.setattr(check_cache_coherence, "CACHEDB", str(coh_dir))
    monkeypatch.setattr(check_cache_coherence, "expected_symbols", lambda: ["BTCUSDC"])
    assert check_cache_coherence.main() == 0

    prune_store = tmp_path / "agy_store"
    prune_store.mkdir()
    db_file = Path(plt_mod.summaries_db(str(prune_store)))
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with plt_mod.sqlite3.connect(str(db_file)) as sq_conn:
        sq_conn.execute(
            "CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY, project_id TEXT, title TEXT, preview TEXT, step_count INTEGER, last_modified_time TEXT)"
        )
        sq_conn.execute(
            "INSERT INTO conversation_summaries VALUES ('cid-auto-1', 'default-cli-project', 'morning reconciliation', '', 2, '2026-01-01T00:00:00Z')"
        )
        sq_conn.commit()
    monkeypatch.setattr(plt_mod, "sync_all_topic_titles", lambda *_a, **_kw: None)
    monkeypatch.setattr(plt_mod, "get_active_topic_cids", lambda *_a, **_kw: set())
    assert plt_mod.prune_old_cli_threads(
        cli_base=str(prune_store),
        retention_days=1e9,
        max_threads=2.7,
        dry_run=True,
    ) == (0, 0)


def test_wave25_guard_decision_unpack_macro_zero_age_and_patch_seam_border_cases(tmp_path, monkeypatch):
    gd = GuardDecision(
        allowed="false",
        brake_action="downscale_qty",
        guard_name="G",
        reason="str_false_ctor",
        suggested_scale=50,
    )
    assert gd.allowed is False
    assert gd.brake_action == BrakeAction.DOWNSCALE_QTY
    assert gd.suggested_scale == pytest.approx(0.5)

    assert unpack_guard_decision(SimpleNamespace(allowed="false", reason="str_false")) == (
        False,
        BrakeAction.HARD_VETO,
        0.0,
        "str_false",
    )
    assert unpack_guard_decision(
        {"allowed": "true", "suggested_scale": "0", "reason": "zero_scale"}
    ) == (False, BrakeAction.HARD_VETO, 0.0, "zero_scale")
    u_ok, u_brake, u_scale, u_reason = unpack_guard_decision(
        SimpleNamespace(allowed="true", suggested_scale=40, reason="pct_40")
    )
    assert u_ok is True and u_brake == BrakeAction.DOWNSCALE_QTY and u_scale == pytest.approx(0.4) and u_reason == "pct_40"

    geo_margins = make_test_margins(
        intelligence_guards_mode="enforce",
        geopolitical_guard_mode="enforce",
        shadow_notify=False,
    )
    monkeypatch.setattr(
        geo_an_mod,
        "GeopoliticalThreatAnalyzer",
        lambda **_kw: SimpleNamespace(
            _cached_assessment=SimpleNamespace(
                ts=1_700_000_000.0,
                threat_level="CRITICAL_SHOCK",
                risk_score=0.9,
                summary="shock",
            )
        ),
    )
    monkeypatch.setattr(
        geo_gd_mod,
        "GeopoliticalShockGuard",
        lambda: SimpleNamespace(
            check=lambda *_a, **_kw: SimpleNamespace(allowed="false", reason="shock_str_false")
        ),
    )
    assert llm_order_guard.check_geopolitical_order_guards(
        "BTCUSDC", "BUY", margins=geo_margins, now=1_700_000_010.0, bypass_guards=set()
    ) == (False, "shock_str_false", 0.0)

    ext_margins = make_test_margins(
        intelligence_guards_mode="enforce",
        external_guard_mode="enforce",
        whale_guard_mode="off",
        orderbook_wall_guard_mode="enforce",
        funding_guard_mode="off",
        shadow_notify=False,
    )
    monkeypatch.setattr(eog, "get_orderbook_collector", lambda: SimpleNamespace())
    monkeypatch.setattr(eog, "fetch_guard_snapshot", lambda *_a, **_kw: {"bid_ask_ratio": 0.2})
    monkeypatch.setattr(
        eog.owg,
        "OrderbookWallGuard",
        lambda **_kw: SimpleNamespace(
            check=lambda *_a, **_k: {"allowed": "false", "reason": "wall_dict"}
        ),
    )
    assert eog.check_external_order_guards(
        "BTCUSDC", "BUY", 60000.0, margins=ext_margins, now=1_700_000_000.0, bypass_guards=set()
    ) == (False, "wall_dict", 0.0)

    ql2_margins = make_test_margins(
        intelligence_guards_mode="enforce",
        internal_guard_mode="enforce",
    )
    monkeypatch.setattr(
        qog,
        "WeibullExhaustionGuard",
        lambda **_kw: SimpleNamespace(
            check=lambda *_a, **_k: SimpleNamespace(
                allowed="false", brake_action=BrakeAction.NONE, reason="exhaust_str_false"
            )
        ),
    )
    assert qog.check_qlimit2_order_guards(
        "BTCUSDC",
        "BUY",
        60000.0,
        price_history=[],
        trend_duration_seconds=3600.0,
        margins=ql2_margins,
        now=1_700_000_000.0,
        bypass_guards={"parabolic_guard"},
    ) == (False, "exhaust_str_false", 0.0)

    ps_dec_box = [SimpleNamespace(allowed=True, suggested_scale=0.0, reason="zero_ps_scale")]
    monkeypatch.setattr(mt, "_is_simulated_env_or_inst", lambda _inst: False)
    monkeypatch.setattr(mt._high_stake_guard, "is_llm_guard_exempt", lambda: False)
    monkeypatch.setattr(
        mt._post_stop_guard,
        "get_post_stop_reentry_guard",
        lambda: SimpleNamespace(mode="enforce", check=lambda **_kw: ps_dec_box[0]),
    )
    re_snap = SimpleNamespace(
        post_sell_trough=58000.0,
        current_time_s=1_700_000_100.0,
        bounce_ratio=lambda trough: (62500.0 - trough) / trough,
    )
    ok_ps0, bud_ps0, _, _ = mt._apply_post_stop_ai_guard(
        "BTCUSDC", 62500.0, 60000.0, 1_700_000_000.0, re_snap, 1000.0, None, 0.01, inst=None
    )
    assert ok_ps0 is False and bud_ps0 == pytest.approx(1000.0)

    ps_dec_box[0] = SimpleNamespace(allowed="true", suggested_scale=40, reason="pct_40_ps")
    ok_ps40, bud_ps40, _, _ = mt._apply_post_stop_ai_guard(
        "BTCUSDC", 62500.0, 60000.0, 1_700_000_000.0, re_snap, 1000.0, None, 0.01, inst=None
    )
    assert ok_ps40 is True and bud_ps40 == pytest.approx(400.0)

    macro_dir = tmp_path / "macro_zero_age"
    macro_dir.mkdir()
    now_macro = 1_700_000_000.0
    atomic_write_text(
        str(macro_dir / "geopolitical_eval.json"),
        json.dumps(
            {
                "ts": now_macro - 30.0,
                "threat_level": "NORMAL",
                "risk_score": 0.1,
                "recommended_brake": "NONE",
                "summary": "Geopolitical threat de-escalated to NORMAL: no fresh high-severity macro headlines in the last 90 minutes.",
                "headlines_digest": "",
            }
        ),
    )
    atomic_write_text(
        str(macro_dir / "news_feed_collect.json"),
        json.dumps(
            {
                "items": [
                    {
                        "title": "Emergency central bank rate cut shock hits bond markets",
                        "source": "Reuters",
                        "first_seen_ts": now_macro,
                        "is_high_severity": "true",
                        "matched_keywords": [],
                    },
                    {
                        "title": "Routine treasury auction settles in line with consensus",
                        "source": "Reuters",
                        "first_seen_ts": now_macro,
                        "is_high_severity": "false",
                        "matched_keywords": [],
                    },
                ]
            }
        ),
    )
    sensors_out, breaking_out = MacroAndNewsSensorsCollector().collect(
        cachedb_dir=str(macro_dir),
        now_ts=now_macro,
        allow_network=False,
        persist=False,
    )
    by_title = {item["title"]: item for item in breaking_out}
    assert by_title["Emergency central bank rate cut shock hits bond markets"]["is_high_severity"] is True
    assert by_title["Emergency central bank rate cut shock hits bond markets"]["age_min"] == 0.0
    assert by_title["Routine treasury auction settles in line with consensus"]["is_high_severity"] is False
    assert "monitoring 1 fresh high-severity macro headline(s)" in sensors_out["geopolitical_threat"]["summary"]

    assert WhaleDivergenceGuard(severe_cascade_oi_pct=5.0).severe_cascade_oi_pct == pytest.approx(-5.0)
    hl_prov = HyperliquidProvider.__new__(HyperliquidProvider)
    hl_prov.get_current_price = lambda _sym: 10.0
    hl_prov.round_price = lambda _sym, px: round(px, 4)
    hl_px, hl_tif = hl_prov._resolve_spot_order_price_and_tif(
        "HYPE/USDC",
        True,
        10.0,
        is_market=True,
        slippage=0.0,
    )
    assert hl_px == pytest.approx(10.0) and hl_tif == "Ioc"
    assert bapi_ws_bridge._bridge_is_usable(SimpleNamespace(is_running=lambda: False)) is False
    assert bapi_ws_bridge._bridge_is_usable(SimpleNamespace(is_running="false")) is False
    assert bapi_ws_bridge._bridge_is_usable(SimpleNamespace(is_running=lambda: True)) is True

    live_venue = rtrade._LivePairVenue.__new__(rtrade._LivePairVenue)
    live_venue.symbol = "BTCUSDC"
    live_venue.pair_store = None
    live_venue._recovery_clock = lambda: 1_700_100_000.0
    with pytest.raises(rtrade._IntentRecoveryPending, match="requires manual reconciliation"):
        live_venue.recover_intent(
            {"pair_id": "p1", "symbol": "BTCUSDC", "start_side": "BUY", "qty": 0.01},
            {
                "client_order_id": "cid1",
                "side": "BUY",
                "kind": "limit",
                "requested_qty": 0.01,
                "requested_price": 60000.0,
                "recovery_state": "horizon_exhausted",
                "recovery_started_at": 1_700_000_000.0,
            },
        )
    coord_blocked = rtp_mod.PairCoordinator.__new__(rtp_mod.PairCoordinator)
    coord_blocked.venue = SimpleNamespace(recovery_blocked="false")
    assert coord_blocked._is_venue_recovery_pending(None, False) is False

    sentinel_mgr = SimpleNamespace(tag="patched_short_trend")
    importlib.import_module("cacheManager")
    monkeypatch.setattr(MarketDataProvider, "_ssot_short_trend_getter", None)
    monkeypatch.setattr(cachetrend, "get_short_trend_manager", lambda: sentinel_mgr)
    assert MarketDataProvider.get_ssot_short_trend_manager() is sentinel_mgr

    t212_strat_mod = load_venue_modules("212trading", "strategy")
    t212_strat = t212_strat_mod.Strategy.__new__(t212_strat_mod.Strategy)
    t212_strat.p = SimpleNamespace(tp_ladder=[(5.0, 0.5)])
    t212_strat.s = {"tp_sold_levels": None}
    assert t212_strat._can_close_dust_cycle() is False
    t212_strat.s = {"tp_sold_levels": ["5.0", {"bad": 1}]}
    assert t212_strat._can_close_dust_cycle() is True
    t212_strat.fx_to_usd = 1.0
    t212_strat.ccy = "USD"
    t212_strat.dry_run = True
    assert t212_strat._place_buy(100.0, 1e-308, "DCA") is False


def test_wave26_post_only_ssot_tick_fanout_and_border_cases(monkeypatch, tmp_path):
    """Wave 26: post-only SSOT maker step-back, AF_UNIX symbol variant fanout, RMW locks, and border flags."""
    assert "eorder:post only" in POST_ONLY_CROSS_TOKENS
    assert "badalopx" in POST_ONLY_CROSS_TOKENS
    assert is_post_only_cross_refusal("EOrder:Post only order") is True
    assert is_post_only_cross_refusal("Order failed: BadAloPx") is True
    assert MarketDataProvider._is_post_only_cross_refusal("EOrder:Post only order") is True
    assert compute_post_only_maker_price(
        "BUY", 100.0, current_price=99.50, tick_size=0.01, round_price_fn=lambda p: round(p, 2)
    ) == pytest.approx(99.45)
    assert compute_post_only_maker_price(
        "SELL", 100.0, current_price=100.50, tick_size=0.01, round_price_fn=lambda p: round(p, 2)
    ) == pytest.approx(100.55)
    assert compute_post_only_maker_price(
        "BUY", 0.004, tick_size=0.0, round_price_fn=lambda p: round(p, 2)
    ) == pytest.approx(0.004)

    monkeypatch.setattr(MarketDataProvider, "read_passive_ssot_current_price", classmethod(lambda cls, _sym: 99.50))
    dummy_bapi_cli = SimpleNamespace(
        get_symbol_info=lambda _sym: {
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
            ]
        }
    )
    stepped_buy = bapi_placeorder._step_binance_post_only_price("BUY", 100.0, "TAOUSDC", dummy_bapi_cli)
    assert stepped_buy == pytest.approx(99.45)

    hlc_mod = load_venue_modules("hyperliquid", "hl_client")
    hl_calls = []
    hl_inst = hlc_mod.HLClient.__new__(hlc_mod.HLClient)
    hl_inst._meta_cache = {"HYPE": {"name": "HYPE", "szDecimals": 2, "maxLeverage": 5}}
    hl_inst.exchange = SimpleNamespace(
        order=lambda coin, is_buy, sz, px, otype, reduce_only=False: (
            hl_calls.append((coin, is_buy, sz, px, otype, reduce_only)),
            (
                {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "BadAloPx"}]}}}
                if len(hl_calls) == 1
                else {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 4242}}]}}}
            ),
        )[1]
    )
    monkeypatch.setattr(MarketDataProvider, "read_passive_ssot_current_price", classmethod(lambda cls, _sym: 24.50))
    ok_hl, oid_hl, _ = hl_inst.place_limit("HYPE", True, 1.0, 25.0, post_only=True)
    assert ok_hl is True and oid_hl == 4242 and len(hl_calls) == 2
    assert hl_calls[1][3] < 24.50

    ws_empty = tmp_path / "ws_empty"
    ws_empty.mkdir()
    assert has_workspace_instrument_sources(ws_empty) is False
    assert has_workspace_instrument_sources(None) is False
    (ws_empty / "212trading").mkdir()
    assert has_workspace_instrument_sources(ws_empty, asset_scope="crypto") is False
    assert has_workspace_instrument_sources(ws_empty, asset_scope="EQUITIES") is True
    assert has_workspace_instrument_sources(ws_empty, asset_scope="all") is True

    kal_path = tmp_path / "kalman_state_test.json"
    tracker = cachetrend.KalmanStateTracker(state_path=str(kal_path), state_min_interval=0.0, read_only=False)
    tracker._state["BTCUSDC"] = {"x": 60000.0, "v": 1.0, "P": [[1.0, 0.0], [0.0, 1.0]], "ts": 1700000000.0, "count": 5}
    with state_io.generation_file_lock(str(kal_path)):
        tracker._write_state(1700000010.0)
    assert kal_path.exists()

    with tempfile.TemporaryDirectory() as sock_dir:
        sock_file = os.path.join(sock_dir, "ticks.sock")
        prod_cpm = CacheCurrentPriceManager(
            1,
            symbols=["HYPE/USDC"],
            filename=str(tmp_path / "prod_cp.json"),
            data_sync_mode="disk_reader",
        )
        cons_cpm = CacheCurrentPriceManager(
            1,
            symbols=["HYPEUSDC"],
            filename=str(tmp_path / "cons_cp.json"),
            data_sync_mode="disk_reader",
        )
        try:
            now_ms = math.trunc(time.time() * 1000)
            prod_cpm.on_items_update("HYPE/USDC", [[now_ms, 38.75]])
            bound = prod_cpm.start_tick_fanout_server(sock_file)
            if bound is not None:
                cons_cpm.start_local_tick_stream(sock_file)
                deadline = time.monotonic() + 1.5
                obs = None
                while time.monotonic() < deadline:
                    obs = cons_cpm.get_cached_price_observation("HYPEUSDC", 5.0)
                    if obs is not None:
                        break
                    time.sleep(0.02)
                assert obs is not None and obs[1] == pytest.approx(38.75)
        finally:
            cons_cpm.shutdown(timeout=1.0)
            prod_cpm.shutdown(timeout=1.0)

    assert _resolve_clamped_intent_notional({"notional_usd": 50.0}, 10.0, 0.0) == 0.0
    assert _resolve_clamped_intent_notional({"notional_usd": 50.0}, 25.0, 15.0) == 15.0

    liq_col = BybitLiquidationCollector(symbols=["BTCUSDT"])
    join_timeouts = []
    liq_col._thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout=None: join_timeouts.append(timeout))
    liq_col.stop(timeout=0.0)
    assert join_timeouts == [0.0]

    dummy_prov = _TestMarketProvider("ssot_flag_probe")
    dummy_prov._ssot_cache_enabled = "false"
    assert dummy_prov._allow_ssot_price_cache() is False
    dummy_prov._ssot_cache_enabled = None
    dummy_prov._force_ssot_cache = "false"
    dummy_prov._cli = SimpleNamespace()
    assert dummy_prov._allow_ssot_price_cache() is False

    str_false_params = SimpleNamespace(
        trend_overlay="false",
        dca_trend_brake="0",
        reentry_hybrid_enabled="false",
        surge_guard="true",
        surge_window_hours=None,
        surge_move_pct="nan",
        surge_dynamic="false",
        tp_dynamic_flat="false",
        tp_trail_adaptive="false",
        dca_vol_scale_k="0",
        trend_interval=60,
        tp_trail_vol_interval=15,
        dca_vol_interval=15,
    )
    assert params_require_trend_interval(str_false_params) is False
    assert required_replay_intervals(str_false_params) == set()

    quota_state_file = tmp_path / "llm_quota_state.json"
    monkeypatch.setattr(
        lqr_mod.GeminiQuotaRouterMixin,
        "_get_quota_state_path",
        classmethod(lambda cls: str(quota_state_file)),
    )
    monkeypatch.setattr(
        lqr_mod.GeminiQuotaRouterMixin,
        "_maybe_notify_low_quota",
        classmethod(lambda cls, _snap: None),
    )
    lqr_mod.GeminiQuotaRouterMixin._record_model_quota_exhausted("gpt-oss-120b-medium", ttl_sec=-10.0)
    assert quota_state_file.exists()
    assert Path(state_io.generation_lock_path(str(quota_state_file))).exists()
    saved_quota = state_io.read_json_dict(str(quota_state_file), default={})
    assert saved_quota["runtime_exhausted"]["gpt-oss-120b-medium"] > time.time() + 60.0


def test_wave27_price_poller_deadlines_canonical_imports_and_border_cases(tmp_path, monkeypatch):
    """Verify Wave 27 border cases across providers/price_poller, parse_nonneg_order_field, price_series_evaluator, T212 strategy/listing_watcher, notify_engine.server, tradeall, bapi_trades, and TrendSignificanceGuard."""
    # 1. start_nonbinance_trend_poller rejects bare string / bool symbols and sanitizes string/NaN deadlines + string-false ws_health_fn
    pushed_prices = []
    stub_cpm = SimpleNamespace(
        market_api=SimpleNamespace(get_current_price=lambda symbol=None, provider_name=None: 44.25),
        bind_providers=lambda _b: None,
        is_real_ws_healthy=lambda _s: "false",
        push_price=lambda sym, px: pushed_prices.append((sym, px)),
        _delegated_poller_symbols=set(),
    )
    for bad_syms in ("XBTUSD", b"XBTUSD", True, 123):
        with pytest.raises(ValueError, match="iterable"):
            start_nonbinance_trend_poller(stub_cpm, bad_syms, provider_names={"XBTUSD": "kraken"}, start_ws=False)
    with pytest.raises(ValueError, match="explicit provider bindings"):
        start_nonbinance_trend_poller(stub_cpm, [True], provider_names={"TRUE": "kraken"}, start_ws=False)

    poller = start_nonbinance_trend_poller(
        stub_cpm,
        ["XBTUSD"],
        interval_sec="0.05",
        fetch_deadline_sec=float("nan"),
        provider_names={"XBTUSD": "kraken"},
        start_ws=False,
    )
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not pushed_prices:
            time.sleep(0.02)
        assert pushed_prices and pushed_prices[0] == ("XBTUSD", pytest.approx(44.25))
    finally:
        poller.stop(timeout=0.5)

    # 2. NonBinanceTrendPoller.stop(timeout=0.0) passes 0.0 to attached WS streams instead of forcing 0.05s
    ws_stop_timeouts = []
    dummy_ws = SimpleNamespace(
        request_stop=lambda: None,
        stop=lambda timeout=None: ws_stop_timeouts.append(timeout),
    )
    dummy_pool = SimpleNamespace(shutdown=lambda **kw: None, _threads=())
    dummy_thread = SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False)
    nb_poller = NonBinanceTrendPoller(dummy_thread, dummy_pool, threading.Event(), ws_streams=[dummy_ws])
    assert nb_poller.stop(timeout=0.0) is True
    assert ws_stop_timeouts == [0.0]

    # 3. fetch_price_with_deadline sanitizes invalid/negative deadline_sec instead of raising ValueError
    captured_fut_timeouts = []
    fake_fut = SimpleNamespace(
        result=lambda timeout=None: (captured_fut_timeouts.append(timeout) or 19.5),
        cancel=lambda: None,
    )
    fake_pool = SimpleNamespace(submit=lambda fn: (fn() and fake_fut))
    px_res = fetch_price_with_deadline(
        SimpleNamespace(get_current_price=lambda **kw: 19.5),
        "XBTUSD",
        fake_pool,
        deadline_sec=-5.0,
        provider_name="kraken",
    )
    assert px_res == pytest.approx(19.5)
    assert captured_fut_timeouts and captured_fut_timeouts[0] > 0.0

    # 4. parse_nonneg_order_field handles whitespace-only strings and string-boolean allow_empty_str
    assert parse_nonneg_order_field("  \t ", allow_empty_str=True) == 0.0
    assert parse_nonneg_order_field("   ", allow_empty_str="false") is None
    assert parse_nonneg_order_field("", allow_empty_str="true") == 0.0

    # 5. price_series_evaluator preserves epoch-0 points when cutoff_ms <= 300000 and clamps invalid num_bins
    ohlc_dir = tmp_path / "ohlc24"
    ohlc_dir.mkdir()
    ohlc_file = ohlc_dir / "cache_ohlc_24h_BTCUSDC.json"
    pts_zero = [[i * 3 * 3600 * 1000, 100.0 + i, 10.0] for i in range(9)]
    state_io.save_json_state(str(ohlc_file), {"points": pts_zero})
    loaded_pts = pse_mod._load_stitched_24h_ohlc_points(
        None,
        "BTCUSDC",
        cutoff_24h=0.0,
        require_positive_volume=True,
        cache_dir=str(ohlc_dir),
    )
    assert len(loaded_pts) == 9 and loaded_pts[0][0] == 0

    vp_samples = [[100.0 + (i % 10), 1.0] for i in range(30)]
    vp_res = pse_mod.compute_volume_profile_metrics(
        [[1700000000000 + i * 60000, s[0], s[1]] for i, s in enumerate(vp_samples)],
        num_bins=0,
    )
    assert vp_res and vp_res.get("poc_px") is not None

    # 6. T212 Strategy._reconcile_real handles _unconfirmed_zero_polls=None and listing_watcher._wait clamps negative sleep
    t212_strat_mod = load_venue_modules("212trading", "strategy")
    t212_lw_mod = load_venue_modules("212trading", "listing_watcher")
    strat_obj = t212_strat_mod.Strategy.__new__(t212_strat_mod.Strategy)
    strat_obj.ticker = "NVDA_US_EQ"
    strat_obj.yahoo_sym = "NVDA"
    strat_obj._unconfirmed_zero_polls = None
    strat_obj._portfolio_position = lambda: (0.0, 0.0)
    strat_obj._recover_pending_submit = lambda _rq, _ao: None
    strat_obj._exact_execution_price = lambda *_a, **_kw: None
    strat_obj._avg = lambda: 120.0
    strat_obj._now = lambda: 1_700_000_000.0
    strat_obj.s = {"qty": 2.0, "orders": [], "pending_submit": None}
    strat_obj.client = SimpleNamespace(
        list_active_orders=lambda: [],
        latest_filled_order_for_ticker=lambda *_a, **_kw: None,
    )
    assert strat_obj._reconcile_real(125.0) is False
    assert strat_obj._unconfirmed_zero_polls == 1

    waited_vals = []
    t212_lw_mod._wait(SimpleNamespace(wait=lambda s: waited_vals.append(s)), -3.5)
    assert waited_vals == [0.0]

    # 7. notify_engine.server._send_ntfy handles status_code=None / string "500" without TypeError
    srv_stub = notify_server_mod.NotificationServer.__new__(notify_server_mod.NotificationServer)
    srv_stub.ntfy_token = ""
    srv_stub._in_gated_dispatch = True
    enqueued = []
    srv_stub._enqueue_payload = lambda kind, payload: enqueued.append((kind, payload))
    monkeypatch.setattr(notify_server_mod, "_external_notifications_disabled", lambda: False)
    monkeypatch.setattr(notify_server_mod, "is_fake_alert_text", lambda *_a: False)
    monkeypatch.setattr(notify_server_mod, "is_email_mirrored", lambda _t: False)
    monkeypatch.setattr(notify_server_mod, "format_ntfy_url", lambda _t: "https://ntfy.example/topic")
    monkeypatch.setattr(
        notify_server_mod.requests,
        "post",
        lambda *_a, **_kw: SimpleNamespace(status_code=None, headers={}),
    )
    assert srv_stub._send_ntfy("ProdCheck", "BodyMsg", "default", "ops_topic", skip_email=True) is True
    monkeypatch.setattr(
        notify_server_mod.requests,
        "post",
        lambda *_a, **_kw: SimpleNamespace(status_code="500", headers={}),
    )
    assert srv_stub._send_ntfy("ProdCheck", "BodyMsg", "default", "ops_topic", skip_email=True) is False
    assert len(enqueued) == 1 and enqueued[0][0] == "ntfy_webhook"

    # 8. tradeall._new_trend_state handles min_interval=None, bapi_trades clamps limit=0->1000, and TrendSignificanceGuard clamps max_p_value=0->0.05
    ta_stub = tradeall.TrendCoordinator.__new__(tradeall.TrendCoordinator)
    ta_stub.min_interval = None
    ts_state = ta_stub._new_trend_state()
    assert ts_state._min_confirm_interval == 0.0
    ta_stub.min_interval = 0.01
    assert ta_stub._new_trend_state()._min_confirm_interval == pytest.approx(tradeall.TIME_SLEEP_GET_PRICE)

    seen_limits = []
    monkeypatch.setattr(bapi_trades.apiorders, "validate_order_query_args", lambda ot, _s: ot)
    monkeypatch.setattr(
        bapi_trades.apiorders,
        "paginate_my_trades",
        lambda _cli, _sym, _st, lim: (seen_limits.append(lim) or []),
    )
    assert bapi_trades.get_my_trades_24("ALL", "BTCUSDC", days_ago=-2, limit=0) == []
    assert seen_limits == [1000]

    tsg = TrendSignificanceGuard(max_p_value=0.0)
    assert tsg.max_p_value == pytest.approx(0.05)


def test_wave28_spot_params_bool_ssot_ohlc_drop_incomplete_and_border_cases(tmp_path, monkeypatch):
    """Verify Wave 28 SSOT bool normalization on SpotParamsMixin/HL StratParams, cross-venue drop_incomplete='false' symmetry, BaseSpotMixin._qty_for/replay_mode, AssetGuardian pending dict guard, KalmanStateTracker._write_state, and negative ob_mid_price fallbacks."""
    # 1. SpotParamsMixin.__post_init__ normalizes string booleans across spot_engine and T212 StratParams
    t212_strat_mod = load_venue_modules("212trading", "strategy")
    t212_p = t212_strat_mod.StratParams.from_dict(
        {
            "enable_takeprofit": "false",
            "sl_rebuy_enabled": "0",
            "reentry_hybrid_enabled": "false",
            "reentry_pullback_adaptive": "false",
            "sl_mhdsl_enabled": "0",
            "sl_structural_break_enabled": "false",
            "tp_mdhtp_enabled": "false",
        }
    )
    assert t212_p.enable_takeprofit is False
    assert t212_p.sl_rebuy_enabled is False
    assert t212_p.reentry_hybrid_enabled is False
    assert t212_p.reentry_pullback_adaptive is False
    assert t212_p.sl_mhdsl_enabled is False
    assert t212_p.sl_structural_break_enabled is False
    assert t212_p.tp_mdhtp_enabled is False

    spot_p = spot_eng_mod.StratParams.from_dict(
        {
            "currency": "USDC",
            "entry_amount": 50.0,
            "entry_discount_pct": 0.2,
            "dca_amount": 25.0,
            "dca_drop_pct": 2.0,
            "check_minutes": 15.0,
            "takeprofit_pct": 3.0,
            "max_budget": 200.0,
            "max_dca_buys": 4,
            "enable_takeprofit": "false",
            "order_ttl_min": 30.0,
            "stop_loss_pct": 5.0,
            "trend_overlay": "false",
            "surge_guard": "0",
        }
    )
    assert spot_p.enable_takeprofit is False
    assert spot_p.trend_overlay is False
    assert spot_p.surge_guard is False

    # 2. Hyperliquid StratParams normalizes signal_gate="false" and rejects invalid signal_gate strings
    hl_strat_mod = load_venue_modules("hyperliquid", "strategy")
    hl_p = hl_strat_mod.StratParams(signal_gate="false", enable_takeprofit="false")
    hl_p._validate_numeric_bounds("long")
    assert hl_p.signal_gate is False
    assert hl_p.enable_takeprofit is False
    with pytest.raises(ValueError, match="signal_gate"):
        hl_bad = hl_strat_mod.StratParams(signal_gate="maybe")
        hl_bad._validate_numeric_bounds("long")

    # 3. state_store.compute_adaptive_reentry_pullback_pct honors string "false" on SimpleNamespace
    ns_p = SimpleNamespace(
        reentry_pullback_adaptive="false",
        reentry_pullback_pct=1.75,
        reentry_pullback_min=0.8,
        reentry_pullback_max=4.0,
        reentry_pullback_k=1.5,
    )
    assert strat_ss_mod.compute_adaptive_reentry_pullback_pct(ns_p, 2.0) == (1.75, "fixed")

    # 4. BaseSpotMixin._qty_for handles vol_dec=None / string sz_dec and _record_state_file_mtime honors replay_mode="false"
    mixin_q = BaseSpotMixin()
    mixin_q.vol_dec = None
    mixin_q.sz_dec = "4"
    assert mixin_q._qty_for(10.0, 8.0) == pytest.approx(1.25)
    sf_path = tmp_path / "spot_mtime.json"
    atomic_write_text(str(sf_path), "{}")
    mixin_q.replay_mode = "false"
    mixin_q.state_file = str(sf_path)
    mixin_q._record_state_file_mtime()
    assert getattr(mixin_q, "_state_file_mtime", 0.0) > 0.0

    # 5. Cross-venue OHLC drop_incomplete="false" symmetry across ohlc_utils, Kraken, Hyperliquid, and T212/Yahoo
    bar_keep = validate_ohlc_candle_bar(
        1700000000000,
        100.0,
        105.0,
        99.0,
        102.0,
        close_ts=1700003600000,
        now_ms=1700001000000,
        drop_incomplete="false",
        ts_in_ms=True,
    )
    assert bar_keep is not None and bar_keep["timestamp"] == 1700000000

    monkeypatch.setattr(
        kraken_md_mod,
        "fetch_public_json",
        lambda *_a, **_kw: {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    [1700000000, "100", "105", "99", "101", "101", "10", 5],
                    [1700003600, "101", "106", "100", "104", "103", "12", 6],
                ],
                "last": 1700000000,
            },
        },
    )
    assert len(kraken_md_mod.fetch_ohlc_candles("XBTUSD", 60, drop_incomplete="false")) == 2

    hl_md_mod = load_venue_modules("hyperliquid", "market_data")
    hl_rows = [
        {"t": 1700000000000, "T": 1700003599999, "o": "40", "h": "42", "l": "39", "c": "41"},
        {"t": 1700003600000, "T": 1700007199999, "o": "41", "h": "43", "l": "40", "c": "42"},
    ]
    assert len(hl_md_mod.normalize_closed_candles(hl_rows, now_ms=1700004000000, drop_incomplete="false")) == 2

    t212_md_mod = load_venue_modules("212trading", "market_data")
    now_sec = math.trunc(time.time())
    bar_start = (now_sec // 300) * 300
    monkeypatch.setattr(
        t212_md_mod,
        "fetch_public_json",
        lambda *_a, **_kw: {
            "chart": {
                "result": [
                    {
                        "meta": {"currentTradingPeriod": {"regular": {"start": bar_start - 3600, "end": bar_start + 3600}}},
                        "timestamp": [bar_start - 300, bar_start],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 101.0],
                                    "high": [102.0, 103.0],
                                    "low": [99.0, 100.5],
                                    "close": [101.0, 102.5],
                                }
                            ]
                        },
                    }
                ]
            }
        },
    )
    assert len(t212_md_mod.fetch_yahoo_candles("NVDA", "1d", "5m", drop_incomplete="false")) == 2

    # 6. T212 _dca_gate_until=None, AssetGuardian _has_pending_campaign, KalmanStateTracker._last_written_count=None, and negative ob_mid_price fallback
    t212_s = t212_strat_mod.Strategy.__new__(t212_strat_mod.Strategy)
    t212_s.ticker = "NVDA_US_EQ"
    t212_s.yahoo_sym = "NVDA"
    t212_s.exit_manager = None
    t212_s._dca_gate_until = None
    t212_s._now = lambda: 1_700_000_000.0
    t212_s._trend_slope_provider = lambda _sym: -2.0
    t212_s._has_open = lambda _side: False
    t212_s.p = SimpleNamespace(
        max_dca_buys=3,
        dca_amount=25.0,
        max_budget=200.0,
        reentry_tolerance_pct=0.0,
        dca_trend_gate_pct=0.5,
        dca_drop_pct=2.0,
    )
    t212_s.s = {
        "dca_buys": 0,
        "spent_cash": 50.0,
        "last_buy_price": 100.0,
        "sl_pending": False,
        "sl_pending_exit": None,
        "buy_backoff_until": 0.0,
    }
    t212_s._dispatch_dca_buy(95.0, 1.0, in_backoff=False)
    assert t212_s._dca_gate_until == pytest.approx(1_700_000_300.0)

    assert ag._has_pending_campaign({"pending": "false"}) is False
    assert ag._has_pending_campaign({"pending": {}}) is False
    assert ag._has_pending_campaign({"pending": {"client_order_id": "ag-1"}}) is True

    kst = cachetrend.KalmanStateTracker(state_path=str(tmp_path / "kalman_test.json"), read_only=False)
    kst._last_written_count = None
    kst._state = {"BTCUSDC": {"price": 65000.0, "slope": 0.01}}
    kst._write_state(time.time())
    assert (tmp_path / "kalman_test.json").exists()

    sec3b = ps_mod._format_microstructure_section(
        {"BTCUSDC": {"ob_mid_price": -5.0, "best_bid": 100.0, "best_ask": 102.0, "spread_bps": 200.0}},
        {"BTCUSDC": {"price": 0.0}},
    )
    assert "ob_mid=101.0" in "\n".join(sec3b)











