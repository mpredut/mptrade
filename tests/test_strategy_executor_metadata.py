import pytest

from providers.strategy_executor import PairPrecision, ProviderError, candle_interval


def test_pair_precision_rejects_invalid_venue_metadata():
    invalid_cases = [
        {"price_decimals": -1, "volume_decimals": 2, "order_min": 0.01},
        {"price_decimals": 2, "volume_decimals": -1, "order_min": 0.01},
        {"price_decimals": 1.5, "volume_decimals": 2, "order_min": 0.01},
        {"price_decimals": 2, "volume_decimals": 2, "order_min": -0.01},
        {"price_decimals": 2, "volume_decimals": 2, "order_min": float("nan")},
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            PairPrecision(**kwargs)


def test_pair_precision_normalizes_valid_metadata_once():
    assert PairPrecision(2.0, 6, "0.01", " HYPE ") == PairPrecision(
        2, 6, 0.01, "HYPE"
    )


def test_candle_interval_maps_only_supported_horizons():
    cases = [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"), (1440, "1d")]
    for minutes, expected in cases:
        assert candle_interval(minutes) == expected


def test_candle_interval_rejects_unknown_horizons():
    unknown_cases = [None, "bad", 0, 30, 60.5]
    for minutes in unknown_cases:
        with pytest.raises(ProviderError):
            candle_interval(minutes)
