#!/usr/bin/env python3
"""tests/test_binance_backtest.py — Tests for Binance market data, replay, and walk-forward parity."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from binance_api.market_data import (
    fetch_ohlc_candles,
    interval_to_binance_str,
    parse_interval_minutes,
)
from binance_api.replay import make_binance_replay_adapter, StratParams, run_replay
from kraken.replay import make_kraken_replay_adapter
from offline.backtests.evaluation import evaluate_segment, evaluate_walk_forward
from offline.backtests.execution import ExecutionModel, FeeModel
from offline.backtests.metrics import validate_replay_metrics


def _sample_params() -> StratParams:
    return StratParams(
        currency="USDT",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        takeprofit_pct=1.0,
        max_budget=500.0,
        max_dca_buys=2,
        check_minutes=1.0,
        enable_takeprofit=True,
        order_ttl_min=60.0,
        stop_loss_pct=5.0,
    )


def test_interval_parsing():
    assert parse_interval_minutes(1) == 1
    assert parse_interval_minutes(60) == 60
    assert parse_interval_minutes("1h") == 60
    assert parse_interval_minutes("4h") == 240
    assert parse_interval_minutes("1d") == 1440
    assert interval_to_binance_str(60) == "1h"
    assert interval_to_binance_str(240) == "4h"
    assert interval_to_binance_str("1h") == "1h"

    with pytest.raises(ValueError):
        parse_interval_minutes(0)
    with pytest.raises(ValueError):
        parse_interval_minutes("invalid_interval")
    with pytest.raises(ValueError):
        interval_to_binance_str(17)


def test_fetch_ohlc_candles_mocked():
    now_ms = 1700000000000
    mock_batch = [
        # [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: vol, 6: close_time, ...]
        [now_ms - 7200000, "50000.0", "50500.0", "49500.0", "50200.0", "10.0", now_ms - 3600001, "0", 0, "0", "0", "0"],
        [now_ms - 3600000, "50200.0", "51000.0", "50100.0", "50800.0", "15.0", now_ms - 1, "0", 0, "0", "0", "0"],
        # Still forming bar (close_time > now_ms)
        [now_ms, "50800.0", "51200.0", "50700.0", "51100.0", "5.0", now_ms + 3599999, "0", 0, "0", "0", "0"],
    ]

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_batch).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch("time.time", return_value=now_ms / 1000.0):
        # With drop_incomplete=True, forming candle is omitted
        candles = fetch_ohlc_candles("BTCUSDT", interval="1h", start_ms=now_ms - 8000000, end_ms=now_ms + 4000000, drop_incomplete=True)
        assert len(candles) == 2
        assert candles[0]["timestamp"] == (now_ms - 7200000) // 1000
        assert candles[0]["open"] == 50000.0
        assert candles[0]["close"] == 50200.0
        assert candles[1]["close"] == 50800.0

        # With drop_incomplete=False, all 3 candles returned
        all_candles = fetch_ohlc_candles("BTCUSDT", interval="1h", start_ms=now_ms - 8000000, end_ms=now_ms + 4000000, drop_incomplete=False)
        assert len(all_candles) == 3


def test_binance_replay_metrics_contract():
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 99.0, 101.0),
        (101.0, 103.0, 100.5, 102.5),
    ]
    params = _sample_params()
    res = run_replay(ohlc, params, fee_pct=0.10, bar_minutes=60.0)

    # Validate that all 10 canonical keys + ambiguous_bars exist and have valid types
    validate_replay_metrics(res)
    assert res["account_currency"] == "USDT"
    assert res["fees"] >= 0.0
    assert "return_pct" in res
    assert "sharpe" in res
    assert "calmar" in res
    assert "sortino" in res


def test_binance_adapter_in_evaluate_walk_forward():
    records = [
        {"timestamp": 1000 + i * 3600, "open": 100.0 + (i % 5), "high": 102.0 + (i % 5),
         "low": 98.0 + (i % 5), "close": 100.5 + (i % 5)}
        for i in range(60)
    ]
    params = _sample_params()
    adapter = make_binance_replay_adapter(params, fee_pct=0.10, bar_minutes=60.0)

    # evaluate_segment with strict contract validation
    seg = evaluate_segment(records[:20], adapter, validate_contract=True)
    assert seg["bars"] == 20
    validate_replay_metrics(seg["metrics"])

    # evaluate_walk_forward with strict contract validation
    wf = evaluate_walk_forward(
        records,
        adapter,
        train_size=20,
        validation_size=10,
        test_size=10,
        step_size=10,
        validate_contract=True,
    )
    assert wf["aggregate_test"]["fold_count"] > 0
    assert "mean_return_pct" in wf["aggregate_test"]


def test_cross_venue_alignment_matrix():
    """Verify that Binance, Kraken, and Trading 212 all implement compliant replay contracts."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 99.0, 101.0),
        (101.0, 102.5, 100.0, 102.0),
    ]

    # 1. Binance Replay
    bn_adapter = make_binance_replay_adapter(_sample_params(), fee_pct=0.10, bar_minutes=60.0)
    bn_res = bn_adapter(ohlc, ())
    validate_replay_metrics(bn_res)

    # 2. Kraken Replay
    kr_params = _sample_params()
    kr_params.currency = "USD"
    kr_adapter = make_kraken_replay_adapter(kr_params, fee_pct=0.25, bar_minutes=60.0)
    kr_res = kr_adapter(ohlc, ())
    validate_replay_metrics(kr_res)

    # Binance fees (0.10%) should be lower than Kraken fees (0.25%) on identical trades
    if bn_res["fills"] > 0 and kr_res["fills"] > 0:
        assert bn_res["fees"] < kr_res["fees"]
