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

    def test_composite_context_and_use_case_behavior(self):
        service = MarketRegimeService()
        bull = self.evaluator.evaluate({"gradient_recent": 0.6, "epsilon": 0.1})
        bear = self.evaluator.evaluate({"gradient_recent": -0.6, "epsilon": 0.1})
        sideways = self.evaluator.evaluate(
            {"gradient_recent": 0.0, "epsilon": 0.1})
        unknown = self.evaluator.unknown()

        with self.subTest(msg="benchmark_context_cannot_create_or_reverse_asset_direction"):
            context_only = service.compose(
                sideways, unknown, (("BTC", bull, bull),))
            self.assertTrue(context_only.actionable)
            self.assertEqual(context_only.regime, "sideways")
            self.assertEqual(context_only.score, 0.0)

            unavailable_asset = service.compose(
                unknown, unknown, (("BTC", bull, bull),))
            self.assertFalse(unavailable_asset.actionable)
            self.assertEqual(unavailable_asset.regime, "unknown")

            hostile_context = service.compose(
                bull,
                bull,
                (("BTC", bear, bear),),
                weights={
                    "asset_short": 0.1,
                    "asset_long": 0.1,
                    "benchmark_short": 0.4,
                    "benchmark_long": 0.4,
                },
            )
            self.assertEqual(hostile_context.regime, "sideways")
            self.assertNotEqual(hostile_context.regime, "bear")
            self.assertTrue(hostile_context.conflict)

            weak_bull = self.evaluator.evaluate({
                "gradient_recent": 0.21,
                "epsilon": 0.1,
            })
            aligned_context = service.compose(
                unknown,
                weak_bull,
                (("BTC", bull, bull),),
            )
            self.assertEqual(aligned_context.regime, "bull")
            self.assertGreater(aligned_context.score, 0.15)

        with self.subTest(msg="composite_uses_benchmark_as_context_not_asset_replacement"):
            decision = service.compose(bull, bull, (("BTC", bear, bear),))
            self.assertTrue(decision.actionable)
            self.assertTrue(decision.conflict)
            self.assertEqual(decision.regime, "bull")

            benchmark_only = service.compose(
                unknown, unknown, (("BTC", bull, bull),))
            self.assertFalse(benchmark_only.actionable)
            self.assertEqual(benchmark_only.regime, "unknown")
            self.assertLessEqual(benchmark_only.confidence, 0.2)

        with self.subTest(msg="composite_rejects_unsafe_weights"):
            with self.assertRaises(ValueError):
                service.compose(
                    bull, bull, weights={
                        "asset_short": 1, "asset_long": 1,
                        "benchmark_short": 0, "benchmark_long": 0,
                    })

        with self.subTest(msg="composite_profiles_detect_pullback_and_change_conviction"):
            execution = service.compose(bear, bull, use_case="execution")
            risk = service.compose(bear, bull, use_case="risk")
            self.assertEqual(execution.regime, "bear")
            self.assertEqual(risk.regime, "bull")
            self.assertEqual(risk.pattern, "bullish_pullback")
            self.assertLess(risk.conviction, risk.confidence)

        with self.subTest(msg="composite_rejects_unknown_use_case"):
            with self.assertRaises(ValueError):
                service.compose(bull, bull, use_case="magic")

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
        self.assertTrue(resolution.primary.time_verified)
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
        first = service.evaluate_provider(provider, "ABCUSD")
        backed_off = service.evaluate_provider(provider, "ABCUSD")
        current[0] = 2.1
        recovered = service.evaluate_provider(provider, "ABCUSD")

        self.assertEqual(first.regime, "unknown")
        self.assertEqual(backed_off.regime, "unknown")
        self.assertEqual(recovered.regime, "bull")
        self.assertEqual(provider.calls, 2)

    def test_compose_evidence_ignores_non_directional_families(self):
        bull = self.evaluator.evaluate({
            "gradient_recent": 0.6,
            "epsilon": 0.1,
        })
        bear = self.evaluator.evaluate({
            "gradient_recent": -0.6,
            "epsilon": 0.1,
        })

        def item(role, decision, family="price_trend"):
            return MarketRegimeEvidence(
                role=role,
                family=family,
                classifier="test",
                provider="Kraken",
                symbol="HYPEUSD",
                decision=decision,
                evaluated_at=1000,
                observed_at=990,
                max_age_seconds=30,
                correlation_key=f"HYPEUSD:{role}:{family}",
            )

        decision = MarketRegimeService().compose_evidence(
            (
                item("asset_short", bull),
                item("asset_short", bear, family="persistence"),
                item("asset_long", bull),
            ),
            asset_symbol="HYPEUSD",
        )

        self.assertEqual(decision.regime, "bull")
        self.assertEqual(len(decision.components), 2)

    def test_common_service_derives_regime_from_provider_ohlc(self):
        class Provider:
            def __init__(self, name):
                self.name = name

            def ohlc_closes(self, _symbol, _interval):
                return [100, 101, 102, 103, 104, 105]

        service = MarketRegimeService(2.0, cache_ttl_sec=30)
        decisions = [
            service.evaluate_provider(
                Provider(name), "HYPEUSD", interval_min=1, window_seconds=360,
            )
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
        decision = service.evaluate_provider(Provider(), "HYPE", interval_min=1)
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




if __name__ == "__main__":
    unittest.main()
