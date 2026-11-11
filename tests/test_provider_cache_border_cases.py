"""Regression tests for border and edge cases across providers and SSOT cache managers."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import botcore
from cacheaccount import CacheTradeManager
from cacheprice import (
    Cache24PriceManager,
    CacheCurrentPriceManager,
    resolve_cached_symbol_price,
    resolve_ssot_price_from_cache,
)
from cachetrend import CacheShortTrendManager
from providers.base import MarketDataProvider
from providers.hyperliquid_provider import HyperliquidProvider
from providers.kraken_provider import KrakenProvider
from providers.t212_provider import T212Provider
from state_io import previous_generation_path


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

    class _ZeroLenMockClient(dict):
        pass

    monkeypatch.setattr(botcore._RUNTIME_SERVICES, "allow_shared_disk_cache", True)
    # A falsy injected mock client (empty dict subclass) must still disable shared disk SSOT cache!
    kr_with_falsy_mock = KrakenProvider(client=_ZeroLenMockClient())
    assert bool(kr_with_falsy_mock._cli) is False
    assert kr_with_falsy_mock._allow_ssot_price_cache() is False

    bad_quotes = iter([0.0, -5.0, float("nan"), float("inf"), True, False, 42.5])

    class _ProbeProvider(MarketDataProvider):
        _USES_SSOT_MARKET_CACHE = False

        @property
        def name(self) -> str:
            return "probe"

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def _fetch_current_price(self, symbol: str):
            return next(bad_quotes)

    probe = _ProbeProvider()
    for _ in range(6):
        assert probe.get_current_price("BTCUSDC") is None
    assert probe.get_current_price("BTCUSDC") == pytest.approx(42.5)

    # Internal TypeError inside _fills_from_api(symbol, publish_cache=...) must NOT trigger a duplicate API call
    calls = []

    class _BuggyFillsProvider(_ProbeProvider):
        def _fills_from_api(self, symbol: str, *, publish_cache: bool = False):
            calls.append((symbol, publish_cache))
            raise TypeError("internal bug inside _fills_from_api")

    buggy = _BuggyFillsProvider()
    with pytest.raises(TypeError, match="internal bug"):
        buggy.fetch_remote_trades("BTCUSDC", 0)
    assert len(calls) == 1

    # Valid fills provider filters non-dict entries and rejects invalid start_time_ms
    class _FillsProvider(_ProbeProvider):
        def _fills_from_api(self, symbol: str, *, publish_cache: bool = False):
            return [
                "not_a_dict",
                {"orderId": "1", "time": 1_000, "price": 10.0, "qty": 1.0},
                {"orderId": "2", "time": 5_000, "price": 11.0, "qty": 1.0},
            ]

    fp = _FillsProvider()
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
    from kraken.trailing_stop import KrakenTrailing

    # 1. Custom MarketDataProvider subclass named 'kraken' resolves Kraken fee/limits via base class
    class _FakeKrakenPairClient:
        def cached_pair_info(self, _cands):
            return {
                "ordermin": "0.0001",
                "pair_decimals": 1,
                "lot_decimals": 8,
                "fees_maker": [[0, 0.16]],
                "fees": [[0, 0.26]],
            }

    class _KrakenSubStub(MarketDataProvider):
        name = "kraken"

        def __init__(self):
            self._cli = _FakeKrakenPairClient()

        def supports_symbol(self, symbol: str) -> bool:
            return True

    stub = _KrakenSubStub()
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
    class _FakeHLClient:
        _sz_decimals_cache = {"HYPE": 2}

    hl = HyperliquidProvider(token="HYPE", client=_FakeHLClient(), live_enabled=False)
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

    class _FakeT212Cli:
        def refresh_order_history(self, max_age_sec: float = 0.0):
            refresh_calls.append(max_age_sec)
            return []

        def get_portfolio(self):
            return []

        def list_active_orders(self):
            return [
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
            ]

    prov = T212Provider(client=_FakeT212Cli(), live_enabled=False)
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

    class _FakeKrakenLedgerCli:
        def _private(self, method: str, params=None, fresh: bool = False):
            assert method == "Ledgers"
            return {
                "ledger": {
                    "L1_B": {"refid": "REF_OLD", "type": "trade", "asset": "XXBT", "amount": "0.01", "fee": "0", "time": old_ts},
                    "L1_Q": {"refid": "REF_OLD", "type": "trade", "asset": "ZUSD", "amount": "-900.0", "fee": "0.9", "time": old_ts},
                    "L2_B": {"refid": "REF_NEW", "type": "trade", "asset": "XXBT", "amount": "0.02", "fee": "0", "time": new_ts},
                    "L2_Q": {"refid": "REF_NEW", "type": "trade", "asset": "ZUSD", "amount": "-1820.0", "fee": "1.8", "time": new_ts},
                }
            }

    cli = _FakeKrakenLedgerCli()
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
    from cachepaths import (
        filter_candidates_by_asset_class,
        has_equity_suffix,
        is_equity_venue,
        is_equity_venue_or_suffix,
    )
    from providers.execution_audit import _filter_intents_by_scope

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
    from providers.order_ohlc_utils import (
        ProviderError,
        derive_venue_128bit_client_order_id,
        format_venue_intent_client_order_id,
        resolve_ohlc_time_window_ms,
    )
    from providers.quantity import (
        dust_protected_partial_sell_qty,
        dust_protected_sell_balance_cap,
        venue_adjust_partial_sell_quantity,
        venue_fee_cap_quantity,
    )

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
    import account_cache_health
    import cacheintegrity
    import cachepaths

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
    with pytest.raises(account_cache_health.AccountCacheNotReady):
        cacheintegrity.read_legacy_trade_jsonl_strict(
            str(legacy_jsonl),
            {"counts": {"BTCUSDC": 1}},
            is_valid_item_fn=lambda _r: True,
            reader_reason="test_legacy_bool_id",
        )


def test_open_closed_provider_class_specs_execution_symmetry_and_border_cases(monkeypatch):
    """Verify declarative provider class specs, cross-venue live/dry-run symmetry, and border hardening."""
    import re
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import _call_hl_info_read
    from providers.order_ohlc_utils import (
        derive_venue_128bit_client_order_id,
        format_venue_intent_client_order_id,
    )
    from providers.quantity import (
        clamp_notional_to_wallet_balance,
        venue_adjust_partial_sell_quantity,
        venue_fee_cap_quantity,
    )

    # 1. Open-Closed custom MarketDataProvider subclass works without touching any central dict
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

    class _BadDecimalsKrakenClient:
        def cached_pair_info(self, _cands):
            return {
                "ordermin": "0.01",
                "pair_decimals": True,
                "lot_decimals": "2.5",
                "fees_maker": [[0, 0.16]],
                "fees": [[0, 0.26]],
            }

    bad_kr = KrakenProvider(client=_BadDecimalsKrakenClient())
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
    from providers import binance_provider as bp_mod

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
    import intelligence.macro_sentiment_ssot as mss
    from botcore import clean_scale, parse_ratio_or_pct
    from providers.binance_provider import BinanceProvider
    from providers.order_ohlc_utils import build_price_history_points, resolve_ohlc_time_window_ms
    from providers.strategy_executor import ProviderError

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

    class _FakeKrakenClient:
        def _public(self, method, params):
            assert method == "OHLC"
            captured_kr.append(dict(params))
            return {"HYPEUSD": [[1700000000, "20", "21", "19", "20.5"], [1e308, "20", "21", "19", "21.0"]], "last": 1}

        def _private(self, method, params=None, fresh=False):
            assert method == "Ledgers"
            captured_ledgers.append(dict(params or {}))
            return {"ledger": {}}

    kp = KrakenProvider(client=_FakeKrakenClient())
    pts_kr = kp._fetch_price_history("HYPEUSD", 1e308)
    assert pts_kr == [{"timestamp": 1700000000000, "price": 20.5}]
    assert captured_kr[-1]["interval"] == 21600
    assert build_price_history_points(
        [[1e308, 10.0], [1700000000, 12.5]],
        lambda r: r[0],
        lambda r: r[1],
        ts_in_seconds=True,
    ) == [{"timestamp": 1700000000000, "price": 12.5}]
    assert KrakenProvider.fetch_ledger_fills_by_symbol(_FakeKrakenClient(), pairs=("HYPEUSD",), lookback_sec=1e20) == {
        "HYPEUSD": []
    }
    assert captured_ledgers[-1]["start"] == 0


def test_embedded_json_ssot_fetch_public_bytes_and_hl_client_border_cases(monkeypatch, tmp_path):
    """Verify state_io embedded JSON helpers, botcore.fetch_public_bytes, HLClient._coin_candidates/candles, and iter_venue_cycle_state_files."""
    import os
    from cachepaths import iter_venue_cycle_state_files
    from state_io import decode_embedded_json_dict, extract_json_object

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

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _max_bytes=None):
            return b"<rss><channel/></rss>"

    monkeypatch.setattr(botcore.urllib.request, "urlopen", lambda req, timeout=10.0: _FakeResp())
    assert botcore.fetch_public_bytes("https://example.com/feed.xml", timeout=5.0) == b"<rss><channel/></rss>"

    # 3. HLClient._coin_candidates & candles overflow safety
    hl_mod = botcore.load_local_module(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid"),
        "hl_client",
    )
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
    import cachepaths
    import intelligence.macro.geopolitical_analyzer as geo_an_mod
    import intelligence.macro.geopolitical_guard as geo_gd_mod
    import intelligence.macro_sentiment_ssot as mss
    import llm_order_guard
    import order_guard
    import order_retry
    import utils
    from market_regime import MarketRegimeHorizon, MarketRegimeService
    from order_lifecycle import TrackedOrderLifecycle

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

    class _FakeHttpResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def _fake_urlopen(req, timeout=10.0):
        captured_req["data"] = req.data
        captured_req["method"] = req.get_method()
        return _FakeHttpResp()

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
    from datetime import datetime, timezone

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
    import account_cache_health as health
    import rtrade
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.sentiment.evaluators.tca_risk_evaluator import VenueFeeAndLimitsEvaluator
    from rtrade_pair_store import RTradePairStore

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





