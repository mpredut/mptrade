import unittest

from market_regime import (
    ClosedPriceSeries,
    MarketRegimeEvidence,
    MarketRegimeEvaluator,
    MarketRegimeService,
)


class MarketRegimeEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = MarketRegimeEvaluator(2.0)

    def test_direction_and_strength_are_provider_neutral(self):
        bull = self.evaluator.evaluate({
            "gradient_recent": 0.5, "epsilon": 0.1,
            "n_samples": 40, "window_seconds": 900,
        })
        bear = self.evaluator.evaluate({"gradient_recent": -0.5, "epsilon": 0.1})
        flat = self.evaluator.evaluate({"gradient_recent": 0.2, "epsilon": 0.1})

        self.assertEqual((bull.regime, bull.strength), ("bull", 5.0))
        self.assertEqual(bear.regime, "bear")
        self.assertEqual(flat.regime, "sideways")
        self.assertEqual((bull.n_samples, bull.window_seconds), (40, 900.0))
        self.assertEqual(bull.fitted_move_pct, 1950.0)

    def test_legacy_gradient_alias_is_normalized_by_the_common_evaluator(self):
        decision = self.evaluator.evaluate({
            "growth_coefficient": 0.5,
            "epsilon": 0.1,
        })

        self.assertEqual(decision.regime, "bull")
        self.assertEqual(decision.gradient, 0.5)
        self.assertTrue(decision.fresh)

        # When both growth_coefficient and gradient_recent are present, SSOT growth_coefficient takes precedence
        ssot_decision = self.evaluator.evaluate({
            "growth_coefficient": 0.6,
            "gradient_recent": -0.3,
            "epsilon": 0.1,
        })
        self.assertEqual(ssot_decision.regime, "bull")
        self.assertEqual(ssot_decision.gradient, 0.6)

    def test_exposure_adversity_is_symmetric(self):
        bull = self.evaluator.evaluate({"gradient_recent": 0.5, "epsilon": 0.1})
        bear = self.evaluator.evaluate({"gradient_recent": -0.5, "epsilon": 0.1})
        self.assertTrue(bear.adverse_to("LONG"))
        self.assertTrue(bull.adverse_to("SOLD"))
        self.assertFalse(bull.adverse_to("LONG"))
        self.assertFalse(bear.adverse_to("SOLD"))

    def test_invalid_and_incomplete_signals_are_explicit_unknown(self):
        with self.subTest(msg="unavailable_and_invalid_signals"):
            self.assertEqual(self.evaluator.evaluate(None).regime, "unknown")
            invalid = self.evaluator.evaluate({"gradient_recent": "bad", "epsilon": 1})
            self.assertEqual((invalid.regime, invalid.fresh, invalid.reason),
                             ("unknown", False, "invalid_signal"))

        with self.subTest(msg="incomplete_snapshot"):
            decision = self.evaluator.evaluate({"gradient_recent": 0.5, "ts": 1000})
            self.assertEqual(
                (decision.regime, decision.fresh, decision.reason),
                ("unknown", False, "missing_signal_fields"),
            )

        with self.subTest(msg="invalid_snapshot_metadata"):
            invalid_samples = self.evaluator.evaluate({
                "gradient_recent": 0.5,
                "epsilon": 0.1,
                "n_samples": float("nan"),
            })
            invalid_window = self.evaluator.evaluate({
                "gradient_recent": 0.5,
                "epsilon": 0.1,
                "window_seconds": -1,
            })
            self.assertEqual(invalid_samples.reason, "invalid_signal_metadata")
            self.assertEqual(invalid_window.reason, "invalid_window_metadata")
            self.assertFalse(invalid_samples.fresh)
            self.assertFalse(invalid_window.fresh)

    def test_snapshot_fallback_and_staleness_behavior(self):
        class Provider:
            name = "Binance"

            def __init__(self):
                self.calls = []

            def ohlc_closes(self, _symbol, interval):
                self.calls.append(interval)
                return [100, 101, 102, 103, 104, 105]

        with self.subTest(msg="incomplete_snapshot_is_retained_before_ohlc_fallback"):
            provider = Provider()
            resolution = MarketRegimeService().resolve_with_evidence(
                provider,
                "TAOUSDC",
                snapshot={"gradient_recent": 0.5, "ts": 1000},
                now=1000,
            )

            self.assertEqual(provider.calls, [1])
            self.assertEqual(len(resolution.evidence), 2)
            self.assertEqual(
                resolution.evidence[0].decision.reason,
                "missing_signal_fields",
            )
            self.assertEqual(resolution.primary.decision.source, "ohlc:1m")

        with self.subTest(msg="stale_snapshot_falls_back_and_remains_observable"):
            provider = Provider()
            resolution = MarketRegimeService().resolve_with_evidence(
                provider,
                "TAOUSDC",
                snapshot={"gradient_recent": 0.5, "epsilon": 0.1, "ts": 900},
                snapshot_max_age_seconds=30,
                now=1000,
            )

            self.assertEqual(provider.calls, [1])
            self.assertEqual(resolution.evidence[0].temporal_state, "stale")
            self.assertFalse(resolution.evidence[0].usable)
            self.assertEqual(resolution.selected_index, 1)
            self.assertEqual(resolution.primary.decision.source, "ohlc:1m")

        with self.subTest(msg="stale_snapshot_without_fallback_has_no_selected_source"):
            resolution = MarketRegimeService().resolve_with_evidence(
                object(),
                "TAOUSDC",
                snapshot={"gradient_recent": 0.5, "epsilon": 0.1, "ts": 900},
                snapshot_max_age_seconds=30,
                allow_fallback=False,
                now=1000,
            )

            self.assertIsNone(resolution.selected_index)
            self.assertIsNone(resolution.primary)
            self.assertFalse(resolution.decision.fresh)
            self.assertEqual(resolution.decision.reason, "stale_source")

    def test_snapshot_observed_at_falls_back_to_valid_ts(self):
        resolution = MarketRegimeService().resolve_with_evidence(
            object(),
            "TAOUSDC",
            snapshot={
                "gradient_recent": 0.5,
                "epsilon": 0.1,
                "observed_at": None,
                "ts": 990,
            },
            snapshot_max_age_seconds=30,
            allow_fallback=False,
            now=1000,
        )

        self.assertEqual(resolution.primary.observed_at, 990)
        self.assertEqual(resolution.primary.temporal_state, "fresh")

    def test_snapshot_at_exact_age_limit_remains_fresh(self):
        resolution = MarketRegimeService().resolve_with_evidence(
            object(),
            "TAOUSDC",
            snapshot={"gradient_recent": 0.5, "epsilon": 0.1, "ts": 970},
            snapshot_max_age_seconds=30,
            allow_fallback=False,
            now=1000,
        )

        self.assertEqual(resolution.primary.temporal_state, "fresh")
        self.assertTrue(resolution.primary.usable)

    def test_interval_specific_age_limits_select_fresh_fallback(self):
        class Provider:
            name = "Kraken"

            def ohlc_series(self, _symbol, interval):
                observed_at = 900 if interval == 1 else 980
                return ClosedPriceSeries(
                    (100, 101, 102, 103, 104, 105),
                    interval,
                    observed_at,
                )

        resolution = MarketRegimeService().resolve_with_evidence(
            Provider(),
            "HYPEUSD",
            horizon="short",
            ohlc_max_age_seconds={1: 30, 5: 60},
            now=1000,
        )

        self.assertEqual(resolution.selected_index, 1)
        self.assertEqual(resolution.primary.interval_min, 5)
        self.assertEqual(
            [item.max_age_seconds for item in resolution.evidence],
            [30.0, 60.0],
        )
        self.assertEqual(
            [item.temporal_state for item in resolution.evidence],
            ["stale", "fresh"],
        )

    def test_gap_outside_consumed_window_does_not_invalidate_series(self):
        class Provider:
            name = "Kraken"

            def ohlc_series(self, _symbol, interval):
                timestamps = [100 + index * 60 for index in range(20)]
                timestamps[2] += 30
                return ClosedPriceSeries(
                    tuple(range(100, 120)),
                    interval,
                    timestamps[-1],
                    tuple(timestamps),
                )

        resolution = MarketRegimeService().resolve_with_evidence(
            Provider(),
            "HYPEUSD",
            allow_fallback=False,
            ohlc_max_age_seconds={1: 30},
            now=1250,
        )

        self.assertIsNotNone(resolution.primary)
        self.assertTrue(resolution.primary.continuous_candles)

    def test_alternate_collection_is_explicit_and_does_not_change_selection(self):
        class Provider:
            name = "Kraken"

            def __init__(self):
                self.calls = []

            def ohlc_closes(self, _symbol, interval):
                self.calls.append(interval)
                return [100, 101, 102, 103, 104, 105]

        primary_provider = Provider()
        primary = MarketRegimeService().resolve_with_evidence(
            primary_provider, "HYPEUSD", horizon="short", now=1000)
        alternate_provider = Provider()
        with_alternates = MarketRegimeService().resolve_with_evidence(
            alternate_provider,
            "HYPEUSD",
            horizon="short",
            include_alternates=True,
            now=1000,
        )

        self.assertEqual(primary_provider.calls, [1])
        self.assertEqual(alternate_provider.calls, [1, 5])
        self.assertEqual(primary.decision, with_alternates.decision)
        self.assertEqual(with_alternates.selected_index, 0)
        self.assertEqual(len(with_alternates.evidence), 2)
        self.assertEqual(
            len({item.correlation_key for item in with_alternates.evidence}),
            1,
        )

    def test_timestamped_closed_series_propagates_verified_freshness(self):
        class Provider:
            name = "Hyperliquid"

            def ohlc_series(self, _symbol, interval):
                return ClosedPriceSeries(
                    (100, 101, 102, 103, 104, 105),
                    interval,
                    980,
                    (680, 740, 800, 860, 920, 980),
                )

        resolution = MarketRegimeService().resolve_with_evidence(
            Provider(),
            "HYPE",
            now=1000,
            ohlc_max_age_seconds=30,
        )

        self.assertEqual(resolution.primary.observed_at, 980)
        self.assertEqual(resolution.primary.temporal_state, "fresh")
        self.assertTrue(resolution.primary.closed_candles)

    def test_candle_gap_is_retained_but_not_selected(self):
        class Provider:
            name = "Gap"

            def ohlc_series(self, _symbol, interval):
                return ClosedPriceSeries(
                    (100, 101, 102),
                    interval,
                    980,
                    (800, 860, 980),
                )

        resolution = MarketRegimeService().resolve_with_evidence(
            Provider(),
            "ABCUSD",
            allow_fallback=False,
            ohlc_max_age_seconds={1: 60},
            now=1000,
        )

        self.assertIsNone(resolution.primary)
        self.assertFalse(resolution.decision.fresh)
        self.assertIn("candle_gap", resolution.decision.reason)
        self.assertFalse(resolution.evidence[0].continuous_candles)

    def test_unknown_provider_result_uses_short_backoff_then_recovers(self):
        class Provider:
            name = "Recovering"

            def __init__(self):
                self.calls = 0

            def ohlc_closes(self, _symbol, _interval):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary")
                return [100, 101, 102, 103, 104, 105]

        provider = Provider()
        current = [0.0]
        service = MarketRegimeService(
            cache_ttl_sec=60,
            negative_cache_ttl_sec=2,
            clock=lambda: current[0],
        )
        first = service._evaluate_provider_source(provider, "ABCUSD")[0]
        backed_off = service._evaluate_provider_source(provider, "ABCUSD")[0]
        current[0] = 2.1
        recovered = service._evaluate_provider_source(provider, "ABCUSD")[0]

        self.assertEqual(first.regime, "unknown")
        self.assertEqual(backed_off.regime, "unknown")
        self.assertEqual(recovered.regime, "bull")
        self.assertEqual(provider.calls, 2)

    def test_common_service_derives_regime_from_provider_ohlc(self):
        class Provider:
            def __init__(self, name):
                self.name = name

            def ohlc_closes(self, _symbol, _interval):
                return [100, 101, 102, 103, 104, 105]

        service = MarketRegimeService(2.0, cache_ttl_sec=30)
        decisions = [
            service._evaluate_provider_source(
                Provider(name), "HYPEUSD", interval_min=1, window_seconds=360,
            )[0]
            for name in ("Kraken", "Hyperliquid")
        ]
        self.assertEqual(decisions[0], decisions[1])
        decision = decisions[0]
        self.assertEqual(decision.regime, "bull")
        self.assertEqual(decision.n_samples, 6)

    def test_closes_horizon_uses_the_canonical_window_and_annotation(self):
        decision = MarketRegimeService().evaluate_closes_for_horizon(
            list(range(100, 142)),
            horizon="long",
            interval_min=240,
        )
        self.assertEqual(decision.regime, "bull")
        self.assertEqual(decision.window_seconds, 7 * 86400.0)
        self.assertEqual((decision.horizon, decision.source), ("long", "closes:240m"))
        self.assertEqual(
            MarketRegimeService.horizon_sample_capacity("long", 240), 42,
        )

    def test_common_service_is_bounded_and_unknown_on_provider_error(self):
        class Provider:
            name = "Hyperliquid"
            def ohlc_closes(self, _symbol, _interval):
                raise RuntimeError("offline")

        service = MarketRegimeService(cache_max=1)
        decision = service._evaluate_provider_source(Provider(), "HYPE", interval_min=1)[0]
        self.assertEqual(decision.regime, "unknown")
        self.assertEqual(decision.reason, "source_error:RuntimeError")

    def test_ohlc_fallback_behavior_for_horizons(self):
        with self.subTest(msg="short_falls_back_from_snapshot_to_same_provider_ohlc"):
            class Provider1:
                name = "Binance"
                def ohlc_closes(self, _symbol, interval):
                    return [100, 101, 102, 103] if interval == 1 else []

            decision = MarketRegimeService().resolve(
                Provider1(), "TAOUSDC", horizon="short", snapshot={})
            self.assertEqual((decision.regime, decision.horizon, decision.source),
                             ("bull", "short", "ohlc:1m"))
            self.assertTrue(decision.fallback_used)

        with self.subTest(msg="long_uses_daily_fallback_when_four_hour_source_fails"):
            class Provider2:
                name = "Kraken"
                def ohlc_closes(self, _symbol, interval):
                    if interval == 240:
                        raise RuntimeError("4h unavailable")
                    return list(range(100, 130))

            decision = MarketRegimeService().resolve(
                Provider2(), "HYPEUSD", horizon="long")
            self.assertEqual((decision.regime, decision.horizon, decision.source),
                             ("bull", "long", "ohlc:1440m"))
            self.assertTrue(decision.fallback_used)

    def test_cross_venue_symbol_and_power_normalization(self):
        import pricewindow as pw
        from market_regime import MarketRegimeContext
        bull = self.evaluator.evaluate({"gradient_recent": 0.6, "epsilon": 0.1})
        ctx = MarketRegimeContext.from_decision(
            bull, evaluated_at=1000.0, symbol="BTC-USDC", provider="BinanceSpot"
        )
        self.assertTrue(ctx.is_valid_for(now=1005.0, symbol="BTC_USDC", provider="binance"))
        self.assertTrue(ctx.is_valid_for(now=1005.0, symbol="BTC/USDC", provider="Binance"))
        self.assertEqual(pw.step_wols_power_for_symbol("XBTUSD"), 3.0)
        self.assertEqual(pw.step_wols_power_for_symbol("XXBTZUSD"), 3.0)
        self.assertEqual(pw.fast_trend_window_for_symbol("TAO_USDC"), 90.0)


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_marketregimeservice__evaluate_provider_source_clock_skew_iter27_s3():
    """[ITER27 #3] MarketRegimeService._evaluate_provider_source clock skew."""
    # 3. MarketRegimeService._evaluate_provider_source clock skew
    svc = MarketRegimeService(clock=lambda: 1000.0, cache_ttl_sec=60.0)

    calls = []
    class DummyRegimeProv:
        name = "kraken"
        def ohlc_series(self, sym, interval_min):
            calls.append(sym)
            return ClosedPriceSeries(
                closes=tuple(100.0 + i for i in range(30)),
                interval_min=interval_min,
                last_closed_at=1000.0,
                timestamps=tuple(1000.0 - (29 - i) * 60.0 for i in range(30)),
            )

    prov = DummyRegimeProv()
    svc._evaluate_provider_source(prov, "TAOUSD", interval_min=1, window_seconds=900.0)
    assert len(calls) == 1
    # Inject future timestamp into cache (clock skew: cached_at = 2000.0 > clock = 1000.0)
    cache_key = ("kraken", "TAOUSD", 1, 900.0)
    svc._cache[cache_key] = (2000.0,) + svc._cache[cache_key][1:]
    svc._evaluate_provider_source(prov, "TAOUSD", interval_min=1, window_seconds=900.0)
    assert len(calls) == 2

def test_market_regime_marketregimedecision_marketregimecontext_marketregimeevidence_marketregimeevaluator_marketregimeservice_iter47_s2():
    """[ITER47 #2] market_regime: MarketRegimeDecision, MarketRegimeContext, MarketRegimeEvidence, MarketRegimeEvaluator, MarketRegimeService."""
    import pytest
    import market_regime as mr
    # 2. market_regime: MarketRegimeDecision, MarketRegimeContext, MarketRegimeEvidence, MarketRegimeEvaluator, MarketRegimeService
    dec_bool = mr.MarketRegimeDecision(
        regime="bull", gradient=True, epsilon=0.01, strength=float("nan"), fresh=True, reason="ok", n_samples=10
    )
    assert dec_bool.fitted_move_pct is None

    ctx = mr.MarketRegimeContext(
        decision=dec_bool, resolved_trend="bull", evaluated_at=1000.0, trend_duration_seconds=60.0
    )
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=float("nan")) is False
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=True) is False
    assert ctx.is_valid_for(now=1010.0, max_horizon_seconds=float("nan")) is False
    assert ctx.is_valid_for(now=1010.0, max_horizon_seconds=True) is False
    assert ctx.is_valid_for(now=1010.0, max_age_seconds=30.0, max_horizon_seconds=120.0) is True

    ev = mr.MarketRegimeEvidence(
        role="asset_short",
        family="price_trend",
        classifier="test",
        provider="binance",
        symbol="BTCUSDC",
        decision=dec_bool,
        evaluated_at=1010.0,
        observed_at=1000.0,
        max_age_seconds=float("nan"),
    )
    assert ev.temporal_state == "stale"
    assert ev.usable is False

    with pytest.raises(ValueError):
        mr.MarketRegimeEvaluator(strength_threshold=True)
    with pytest.raises(ValueError):
        mr.MarketRegimeService(cache_ttl_sec=True)

    svc = mr.MarketRegimeService()
    assert svc._snapshot_observed_at({"ts": True}) is None
    assert svc.evaluate_closes([100.0, 101.0, 102.0], interval_min=0).reason == "invalid_window_metadata"
    assert svc.evaluate_closes([100.0, 101.0, 102.0], window_seconds=float("nan")).reason == "invalid_window_metadata"

def test_marketregimecontext_from_decision_accepts_provider_objects_with_name_and_iter88_s1():
    """[ITER88 #1] MarketRegimeContext.from_decision accepts provider objects with .name and rejects booleans & MarketRegimeEvidence.temporal_state handles non-numeric max_age_seconds without ValueError & MarketRegimeService rejects boolean now / interval_min and does not silently replace interval_min=0 with default."""
    import pytest
    from market_regime import MarketRegimeContext, MarketRegimeDecision, MarketRegimeEvidence, MarketRegimeService
    # 1. MarketRegimeContext.from_decision accepts provider objects with .name and rejects booleans
    dec = MarketRegimeDecision(
        regime="bull", gradient=0.01, epsilon=0.001, strength=10.0, fresh=True, reason="directional_signal"
    )

    class _ProvObj:
        name = "Binance"

    ctx = MarketRegimeContext.from_decision(
        dec,
        evaluated_at=1000.0,
        symbol=" btcusdc ",
        provider=_ProvObj(),
        trend_duration_seconds=True,
        benchmark_symbol=True,
    )
    assert ctx.symbol == "BTCUSDC"
    assert ctx.provider == "binance"
    assert ctx.trend_duration_seconds == 0.0
    assert ctx.benchmark_symbol is None


    # 2. MarketRegimeEvidence.temporal_state handles non-numeric max_age_seconds without ValueError
    ev = MarketRegimeEvidence(
        role="asset_short",
        family="price_trend",
        classifier="linear_noise_v1",
        provider="Binance",
        symbol="BTCUSDC",
        decision=dec,
        evaluated_at=1000.0,
        observed_at=990.0,
        max_age_seconds="bad_age",
    )
    assert ev.temporal_state == "stale"


    # 3. MarketRegimeService rejects boolean now / interval_min and does not silently replace interval_min=0 with default
    svc = MarketRegimeService()
    with pytest.raises(ValueError, match="now must be a positive"):
        svc.resolve_with_evidence(_ProvObj(), "BTCUSDC", now=True)
    with pytest.raises(ValueError, match="regime interval"):
        svc.resolve_with_evidence(_ProvObj(), "BTCUSDC", interval_min=True)
    with pytest.raises(ValueError, match="regime interval"):
        svc.resolve_with_evidence(_ProvObj(), "BTCUSDC", interval_min=0)

def test_marketregimehorizon_parse_adverse_to_and_marketregimeservice_interval_closes_validation_iter98_s1():
    """[ITER98 #1] MarketRegimeHorizon.parse, adverse_to, and MarketRegimeService interval/closes validation."""
    import pytest
    from market_regime import MarketRegimeEvaluator, MarketRegimeHorizon, MarketRegimeService
    # 1. MarketRegimeHorizon.parse, adverse_to, and MarketRegimeService interval/closes validation
    assert MarketRegimeHorizon.parse(" short ") == MarketRegimeHorizon.SHORT
    with pytest.raises(ValueError, match="unsupported market-regime horizon"):
        MarketRegimeHorizon.parse(True)

    ev = MarketRegimeEvaluator(2.0)
    bull_dec = ev.evaluate({"gradient_recent": 0.5, "epsilon": 0.1})
    bear_dec = ev.evaluate({"gradient_recent": -0.5, "epsilon": 0.1})
    assert bear_dec.adverse_to(" long ") is True
    assert bear_dec.adverse_to(" buy ") is True
    assert bull_dec.adverse_to(" sold ") is True
    assert bull_dec.adverse_to(True) is False

    svc = MarketRegimeService()
    with pytest.raises(ValueError, match="the regime interval must be positive"):
        svc.horizon_sample_capacity("short", interval_min=1.5)
    with pytest.raises(ValueError, match="the regime interval must be positive"):
        svc.evaluate_closes_for_horizon([100, 101, 102], horizon="short", interval_min=True)
    with pytest.raises(ValueError, match="the regime interval must be positive"):
        svc.evaluate_closes_for_horizon([100, 101, 102], horizon="short", interval_min=1.5)
    assert svc.evaluate_closes([100, 101, 102], interval_min=1.5).reason == "invalid_window_metadata"
    assert svc.evaluate_closes("12345").reason == "invalid_ohlc"

def test_market_regime_marketregimecontext_is_valid_for_compose__closed_series_iter107_s2():
    """[ITER107 #2] market_regime: MarketRegimeContext.is_valid_for, compose, _closed_series."""
    import pytest
    from types import SimpleNamespace
    import market_regime as mr
    # 2. market_regime: MarketRegimeContext.is_valid_for, compose, _closed_series
    dec = mr.MarketRegimeEvaluator().unknown()
    ctx = mr.MarketRegimeContext(
        decision=dec,
        resolved_trend="unknown",
        evaluated_at=1000.0,
        symbol=True,
        provider=True,
    )
    assert ctx.is_valid_for(now=1000.0, require_identity=True) is False
    ctx_ok = mr.MarketRegimeContext(
        decision=dec,
        resolved_trend="unknown",
        evaluated_at=1000.0,
        symbol="BTCUSDC",
        provider="binance",
    )
    assert ctx_ok.is_valid_for(symbol=True, now=1000.0) is False
    assert ctx_ok.is_valid_for(provider=True, now=1000.0) is False

    svc = mr.MarketRegimeService()

    # _closed_series rejects boolean interval_min even when interval_min == 1 (since int(True) == 1)
    bad_provider = SimpleNamespace(
        name="binance",
        ohlc_series=lambda _s, _i: mr.ClosedPriceSeries((100.0, 101.0, 102.0), True),
    )
    with pytest.raises(ValueError):
        svc._closed_series(bad_provider, "BTCUSDC", 1)

def test_marketregimeservice__age_limit_ignores_boolean_interval_min_and_resolve_with_evidence_validates_iter121_s2():
    """[ITER121 #2] MarketRegimeService._age_limit ignores boolean interval_min and resolve_with_evidence validates window_seconds."""
    import pytest
    from market_regime import MarketRegimeHorizon, MarketRegimeService
    # 2. MarketRegimeService._age_limit ignores boolean interval_min and resolve_with_evidence validates window_seconds
    assert MarketRegimeService._age_limit({1: 180.0, "default": 300.0}, MarketRegimeHorizon.SHORT, interval_min=True) == 300.0
    svc = MarketRegimeService()
    with pytest.raises(ValueError, match="the regime interval and window must be positive"):
        svc.resolve_with_evidence(object(), "BTCUSDC", window_seconds="invalid")
    with pytest.raises(ValueError, match="the regime interval and window must be positive"):
        svc.resolve_with_evidence(object(), "BTCUSDC", window_seconds=-10.0)

def test_market_regime_evaluate_closes_with_interval_min_evaluate_with_n_samples_and_iter161_s3():
    """[ITER161 #3] market_regime: evaluate_closes with interval_min='5.0', evaluate with n_samples='15.0', and is_valid_for with require_identity='false'."""
    import market_regime as mr
    # 3. market_regime: evaluate_closes with interval_min="5.0", evaluate with n_samples="15.0", and is_valid_for with require_identity="false"
    svc = mr.MarketRegimeService(cache_max="64.0")
    assert svc.cache_max == 64
    dec = svc.evaluate_closes([100.0 + i for i in range(10)], interval_min="5.0", window_seconds="1800.0")
    assert dec.fresh is True and dec.regime == "bull"

    snap_dec = svc.evaluator.evaluate({"gradient_recent": 0.01, "epsilon": 0.001, "n_samples": "15.0", "window_seconds": "900.0"})
    assert snap_dec.fresh is True and snap_dec.n_samples == 15

    ctx = mr.MarketRegimeContext.from_decision(snap_dec, evaluated_at=1000.0)
    assert ctx.is_valid_for(now=1005.0, require_identity="false") is True

def test_marketregimedecision_adverse_to_supports_bid_ask_a_and_marketregimecontext_iter179_s3():
    """[ITER179 #3] MarketRegimeDecision.adverse_to supports BID/ASK/A and MarketRegimeContext.from_decision supports dict."""
    import pytest
    from market_regime import MarketRegimeDecision, MarketRegimeContext
    # 3. MarketRegimeDecision.adverse_to supports BID/ASK/A and MarketRegimeContext.from_decision supports dict
    bear_dec = MarketRegimeDecision(regime="bear", gradient="-0.01", epsilon=0.001, strength=10.0, fresh=True, reason="ok", n_samples="11")
    bull_dec = MarketRegimeDecision(regime="bull", gradient=0.01, epsilon=0.001, strength=10.0, fresh=True, reason="ok", n_samples=11)
    assert bear_dec.adverse_to("BID") is True
    assert bull_dec.adverse_to("ASK") is True
    assert bull_dec.adverse_to("A") is True
    assert bear_dec.fitted_move_pct == pytest.approx(-10.0)
    ctx_from_dict = MarketRegimeContext.from_decision({"regime": "sideways", "fresh": True, "strength": 1.2}, evaluated_at=1000.0)
    assert ctx_from_dict.resolved_trend == "flat"
    assert ctx_from_dict.fresh is True
