import unittest
from unittest.mock import Mock

from providers.strategy_executor import SubmissionRefused
from strategies.rtrade_pair import (
    OrderSnapshot,
    OrderTicket,
    PairCoordinator,
    PairPolicy,
    anchored_exit_price,
    quote_prices,
)


class FakeVenue:
    def __init__(self, current=100.0, fail_side=None):
        self.current = current
        self.fail_side = fail_side
        self.orders = []
        self.statuses = {}
        self.canceled = []
        self.market_calls = []
        self.fill_on_cancel = {}
        self.cancel_fail = set()
        self.cancel_raise = set()
        self.status_raise = set()
        self.raise_side = None
        self.failure_reasons = {}
        self.allow_market = True
        self.preflight_calls = []
        self.preflight_error = None
        self.preflight_permit = object()
        self.limit_permits = []
        self.limit_revisions = []
        self.market_permits = []

    def current_price(self):
        return self.current

    def preflight_order(self, side, qty, price=None, *,
                        market=False, kind=None):
        self.preflight_calls.append((
            side.upper(), qty, price, market, kind))
        if self.preflight_error is not None:
            raise self.preflight_error
        return self.preflight_permit

    def place_limit(self, side, price, qty, pair_id, *, cache_permit=None,
                    revision=0):
        self.limit_permits.append(cache_permit)
        self.limit_revisions.append(revision)
        if side.upper() == self.raise_side:
            raise RuntimeError(f"{side} placement transport failure")
        if side.upper() == self.fail_side:
            return None
        ticket = OrderTicket(
            order_id=f"L{len(self.orders) + 1}", side=side.upper(),
            price=price, qty=qty, revision=revision)
        self.orders.append((ticket, pair_id))
        self.statuses[ticket.order_id] = OrderSnapshot()
        return ticket

    def last_place_failure_reason(self, side):
        return self.failure_reasons.get(side.upper())

    def place_market_exit(self, side, qty, reason, pair_id=None, *,
                          cache_permit=None, revision=0):
        ticket = OrderTicket(
            order_id=f"M{len(self.orders) + 1}", side=side.upper(),
            price=self.current, qty=qty, revision=revision)
        self.orders.append((ticket, pair_id or "market"))
        self.market_calls.append((side.upper(), qty, reason))
        self.market_permits.append(cache_permit)
        self.market_revisions = getattr(self, "market_revisions", [])
        self.market_revisions.append(revision)
        self.statuses[ticket.order_id] = OrderSnapshot(
            status="closed", filled_qty=qty, cost=qty * self.current, fee=0.1)
        return ticket

    def market_exit_allowed(self, exposure_side, loss_fraction, reason):
        return self.allow_market

    def order_status(self, order_id):
        if order_id in self.status_raise:
            raise RuntimeError(f"status unavailable for {order_id}")
        return self.statuses[order_id]

    def cancel(self, order_id):
        self.canceled.append(order_id)
        if order_id in self.cancel_raise:
            raise RuntimeError(f"cancel response lost for {order_id}")
        if order_id in self.cancel_fail:
            return False
        if order_id in self.fill_on_cancel:
            qty, price = self.fill_on_cancel.pop(order_id)
            self.statuses[order_id] = OrderSnapshot(
                status="closed", filled_qty=qty, cost=qty * price, fee=0.1)
            return True
        old = self.statuses[order_id]
        self.statuses[order_id] = OrderSnapshot(
            status="canceled", filled_qty=old.filled_qty,
            cost=old.cost, fee=old.fee)
        return True

    def fill(self, order_id, qty, price, *, status="closed", fee=0.1):
        self.statuses[order_id] = OrderSnapshot(
            status=status, filled_qty=qty, cost=qty * price, fee=fee)


def _coordinator(venue, *, start_side="BUY", **policy_overrides):
    policy = PairPolicy(adjustment_fraction=0.0064, **policy_overrides)
    return PairCoordinator(
        venue, qty=1.0, policy=policy, start_side=start_side,
        clock=lambda: 0.0, sleeper=lambda _seconds: None,
        pair_id_factory=lambda: "pair-1")


class PricePolicyTest(unittest.TestCase):
    def test_price_and_coordinator_policy_validation(self):
        with self.subTest("quotes_share_one_midpoint"):
            self.assertEqual(quote_prices(100.0, 0.0064), (99.36, 100.64))

        with self.subTest("long_never_follows_market_below_cost_plus_edge"):
            target = anchored_exit_price("SELL", 100.0, 90.0, 0.0064, 0.0115)
            self.assertEqual(target, 101.1634)
            self.assertGreater(target, 100.0)

        with self.subTest("sold_never_chases_market_above_profitable_buyback"):
            target = anchored_exit_price("BUY", 100.0, 110.0, 0.0064, 0.0115)
            self.assertEqual(target, 98.85)
            self.assertLess(target, 100.0)

        with self.subTest("nonfinite_policy_and_quantity_fail_closed"):
            with self.assertRaises(ValueError):
                PairPolicy(adjustment_fraction=float("nan"))
            with self.assertRaises(ValueError):
                PairCoordinator(FakeVenue(), qty=float("inf"),
                                policy=PairPolicy(adjustment_fraction=0.0064))


class PairCoordinatorTest(unittest.TestCase):
    def test_leg_placement_startup_failures_and_cancel_races(self):
        with self.subTest("places_both_legs_under_same_pair_id"):
            venue = FakeVenue()
            outcome = _coordinator(venue).start(mid=100.0)
            self.assertEqual(outcome.phase, "quoting")
            self.assertEqual(
                [(t.side, t.price, pair) for t, pair in venue.orders],
                [("BUY", 99.36, "pair-1"), ("SELL", 100.64, "pair-1")])

        with self.subTest("second_leg_cancels_first_and_fails_closed"):
            venue = FakeVenue(fail_side="SELL")
            outcome = _coordinator(venue).start(mid=100.0)
            self.assertTrue(outcome.terminal)
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertEqual(venue.canceled, ["L1"])

        with self.subTest("second_leg_unconfirmed_cancel_remains_recoverable"):
            venue = FakeVenue(fail_side="SELL")
            venue.cancel_fail.add("L1")
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertTrue(coordinator.tickets[0].active)
            self.assertEqual(coordinator.export_state()["phase"], "startup_recovery")

        with self.subTest("second_leg_cancel_exception_remains_recoverable"):
            venue = FakeVenue(fail_side="SELL")
            venue.cancel_raise.add("L1")
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertTrue(coordinator.tickets[0].active)

        with self.subTest("exception_keeps_the_first_leg_managed"):
            venue = FakeVenue()
            venue.raise_side = "SELL"
            venue.cancel_fail.add("L1")
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertEqual([ticket.order_id for ticket in coordinator.tickets], ["L1"])

        with self.subTest("second_leg_cancel_partial_fill"):
            venue = FakeVenue(fail_side="SELL")
            venue.fill_on_cancel["L1"] = (0.4, 99.36)
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 0.4)

        with self.subTest("second_leg_cancel_full_fill"):
            venue = FakeVenue(fail_side="SELL")
            venue.fill_on_cancel["L1"] = (1.0, 99.36)
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 1.0)

        with self.subTest("sell_first_places_both_in_reverse_order"):
            venue = FakeVenue()
            outcome = _coordinator(venue, start_side="SELL").start(mid=100.0)
            self.assertEqual(outcome.phase, "quoting")
            self.assertEqual(
                [(t.side, t.price, pair) for t, pair in venue.orders],
                [("SELL", 100.64, "pair-1"), ("BUY", 99.36, "pair-1")])

        with self.subTest("sell_first_failure_does_not_attempt_buy"):
            venue = FakeVenue(fail_side="SELL")
            outcome = _coordinator(venue, start_side="SELL").start(mid=100.0)
            self.assertTrue(outcome.terminal)
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertEqual(venue.orders, [])

        with self.subTest("venue_specific_insufficient_funds_reason_is_preserved"):
            venue = FakeVenue(fail_side="BUY")
            venue.failure_reasons["BUY"] = "buy_insufficient_funds:USDC"
            outcome = _coordinator(venue).start(mid=100.0)
            self.assertEqual(outcome.reason, "buy_insufficient_funds:USDC")

    def test_startup_recovery_for_ambiguous_legs(self):
        with self.subTest("first_leg_response_loss_stays_managed"):
            venue = FakeVenue(fail_side="BUY")
            venue.failure_reasons["BUY"] = "buy_recovery_pending"
            venue.recovery_blocked = True
            blocked = [True]
            recovered = OrderTicket(
                order_id="late-buy", side="BUY", price=99.36, qty=1.0)
            recovered_snapshot = OrderSnapshot(
                status="closed", filled_qty=1.0,
                cost=99.36, fee=0.01)
            recovery = [
                [],
                [(recovered, recovered_snapshot, "limit")],
            ]
            venue.recover_pair_intents = lambda _pair_id: recovery.pop(0)
            venue.pair_recovery_blocked = lambda _pair_id: blocked[0]
            original_place = venue.place_limit
            venue.place_limit = Mock(wraps=original_place)

            coordinator = _coordinator(venue)
            started = coordinator.start(mid=100.0)
            self.assertEqual(started.phase, "startup_recovery")
            self.assertFalse(started.terminal)

            still_blocked = coordinator.step(now=1.0)
            self.assertEqual(still_blocked.phase, "startup_recovery")
            self.assertFalse(still_blocked.terminal)
            blocked[0] = False
            venue.statuses["late-buy"] = recovered_snapshot
            venue.current = None
            recovered_outcome = coordinator.step(now=2.0)

            self.assertEqual(recovered_outcome.phase, "exposed")
            self.assertEqual(recovered_outcome.net_qty, 1.0)
            self.assertEqual(venue.place_limit.call_count, 1)

        with self.subTest("ambiguous_second_leg_adopted_without_resubmit"):
            venue = FakeVenue(fail_side="SELL")
            venue.failure_reasons["SELL"] = "sell_recovery_pending"
            venue.recovery_blocked = True
            blocked = [True]
            late_sell = OrderTicket(
                order_id="late-sell", side="SELL", price=100.64, qty=1.0)
            late_snapshot = OrderSnapshot(
                status="closed", filled_qty=1.0,
                cost=100.64, fee=0.01)
            recovery = [
                [],
                [(late_sell, late_snapshot, "limit")],
            ]
            venue.recover_pair_intents = lambda _pair_id: recovery.pop(0)
            venue.pair_recovery_blocked = lambda _pair_id: blocked[0]
            original_place = venue.place_limit
            venue.place_limit = Mock(wraps=original_place)

            coordinator = _coordinator(venue)
            started = coordinator.start(mid=100.0)
            self.assertEqual(started.phase, "startup_recovery")
            self.assertEqual(venue.canceled, ["L1"])

            coordinator.step(now=1.0)
            blocked[0] = False
            venue.statuses["late-sell"] = late_snapshot
            venue.current = None
            outcome = coordinator.step(now=2.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(outcome.net_qty, -1.0)
            self.assertEqual(venue.place_limit.call_count, 2)
            self.assertIn(
                "late-sell",
                [ticket.order_id for ticket in coordinator.tickets])

    def test_ttl_cancel_behaviors(self):
        with self.subTest(msg="no_fill_until_ttl_cancels_both"):
            venue = FakeVenue()
            coordinator = _coordinator(venue, quote_ttl_sec=32)
            coordinator.start(mid=100.0)

            outcome = coordinator.step(now=33.0)

            self.assertEqual(outcome.phase, "expired")
            self.assertTrue(outcome.terminal)
            self.assertCountEqual(venue.canceled, ["L1", "L2"])
            self.assertEqual(coordinator.tickets, [])
            self.assertEqual(coordinator.snapshots, {})

        with self.subTest(msg="ttl_does_not_terminalize_while_cancel_is_unconfirmed"):
            venue = FakeVenue()
            coordinator = _coordinator(venue, quote_ttl_sec=32)
            coordinator.start(mid=100.0)
            venue.cancel_fail.add("L1")

            pending = coordinator.step(now=33.0)

            self.assertFalse(pending.terminal)
            self.assertEqual(pending.reason, "quote_cancel_pending")
            self.assertTrue(any(ticket.order_id == "L1" and ticket.active
                                for ticket in coordinator.tickets))

        with self.subTest(msg="balanced_pair_waits_for_cancel_confirmation"):
            venue = FakeVenue()
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.5, 99.36, status="open")
            venue.fill("L2", 0.5, 100.64, status="open")
            venue.cancel_fail.add("L1")

            pending = coordinator.step(now=5.0)
            self.assertFalse(pending.terminal)
            self.assertEqual(pending.phase, "closing")
            self.assertEqual(pending.reason, "balanced_cancel_pending")

            venue.cancel_fail.clear()
            complete = coordinator.step(now=6.0)
            self.assertTrue(complete.terminal)
            self.assertEqual(complete.phase, "complete")

        with self.subTest(msg="fill_during_ttl_cancel_is_reconciled_as_exposure"):
            venue = FakeVenue(current=99.36)
            coordinator = _coordinator(venue, quote_ttl_sec=32)
            coordinator.start(mid=100.0)
            venue.fill_on_cancel["L1"] = (1.0, 99.36)

            outcome = coordinator.step(now=33.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertFalse(outcome.terminal)
            self.assertAlmostEqual(outcome.net_qty, 1.0)
            replacement = venue.orders[-1][0]
            self.assertEqual(replacement.side, "SELL")
            self.assertGreater(replacement.price, 99.36)

    def test_fill_latency_and_stale_preflight_behaviors(self):
        with self.subTest("both_fills_complete_cycle_and_measure_fast_latency"):
            venue = FakeVenue()
            coordinator = _coordinator(venue, quote_ttl_sec=32, fast_fill_ratio=0.25)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.fill("L2", 1.0, 100.64)

            outcome = coordinator.step(now=5.0)

            self.assertEqual(outcome.phase, "complete")
            self.assertTrue(outcome.shock)
            self.assertEqual(outcome.first_fill_side, "BOTH")
            self.assertEqual(outcome.fill_latency_sec, 5.0)
            self.assertAlmostEqual(outcome.gross_pnl, 1.28)

        with self.subTest("slow_single_fill_keeps_existing_profitable_opposite_leg"):
            venue = FakeVenue(current=99.36)
            coordinator = _coordinator(venue, quote_ttl_sec=32, fast_fill_ratio=0.25)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            outcome = coordinator.step(now=20.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertFalse(outcome.shock)
            self.assertEqual(outcome.first_fill_side, "BUY")
            self.assertAlmostEqual(outcome.net_qty, 1.0)
            self.assertEqual(venue.canceled, [])
            self.assertEqual(len(venue.orders), 2, "the initial ask is already a profitable exit")

        with self.subTest("stale_preflight_preserves_active_anchored_exit"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="closed")
            venue.preflight_error = SubmissionRefused("cache_stale")

            with self.assertRaisesRegex(SubmissionRefused, "cache_stale"):
                coordinator.step(now=12.0)

            self.assertEqual(venue.canceled, [])
            self.assertEqual(len(venue.orders), 2)
            self.assertTrue(venue.orders[1][0].active)
            side, qty, price, market, kind = venue.preflight_calls[0]
            self.assertEqual((side, qty, market, kind),
                             ("SELL", 0.4, False, "rtrade_pair_quote"))
            self.assertGreater(price, 99.36)

        with self.subTest("permit_is_forwarded_when_cache_turns_stale_during_cancel"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="closed")
            original_cancel = venue.cancel

            def cancel_then_stale(order_id):
                canceled = original_cancel(order_id)
                if order_id == "L2":
                    venue.preflight_error = SubmissionRefused(
                        "account_cache_not_fresh")
                return canceled

            venue.cancel = cancel_then_stale
            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(venue.canceled, ["L2"])
            self.assertEqual(len(venue.preflight_calls), 1)
            self.assertIs(
                venue.limit_permits[-1],
                venue.preflight_permit,
            )
            replacement, pair_id = venue.orders[-1]
            self.assertEqual((replacement.side, pair_id), ("SELL", "pair-1"))

    def test_anchored_exit_revisions_and_fill_accounting(self):
        with self.subTest("uses_net_remainder_after_fill_during_cancel"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="closed")
            venue.fill_on_cancel["L2"] = (0.15, 100.64)

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 0.25)
            replacement, _pair_id = venue.orders[-1]
            self.assertEqual(replacement.side, "SELL")
            self.assertAlmostEqual(replacement.qty, 0.25)
            self.assertEqual(venue.preflight_calls[0][1], 0.4)
            self.assertIs(venue.limit_permits[-1], venue.preflight_permit)

        with self.subTest("replacement_advances_and_persists_limit_revision"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0, pair_id="durable-reprice")
            venue.fill("L1", 0.4, 99.36, status="closed")

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(venue.canceled, ["L2"])
            self.assertEqual(venue.limit_revisions, [0, 0, 1])
            state = coordinator.export_state()
            self.assertEqual(state["limit_revisions"], {"BUY": 0, "SELL": 1})

            restored = PairCoordinator.from_state(
                venue, coordinator.policy, state,
                clock=lambda: 12.0, sleeper=lambda _seconds: None)
            self.assertEqual(restored.limit_revisions, {"BUY": 0, "SELL": 1})

        with self.subTest("zero_fill_expired_exit_advances_revision_once"):
            venue = FakeVenue(current=100.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.fill("L2", 0.0, 100.64, status="expired", fee=0.0)

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            replacement = coordinator.tickets[-1]
            self.assertEqual(replacement.side, "SELL")
            self.assertEqual(replacement.qty, 1.0)
            self.assertEqual(replacement.revision, 1)
            self.assertEqual(coordinator.limit_revisions["SELL"], 1)
            self.assertEqual(venue.limit_revisions, [0, 0, 1])
            self.assertEqual(
                len({ticket.order_id for ticket in coordinator.tickets}),
                len(coordinator.tickets))

        with self.subTest("partial_expired_exit_counts_fill_once_and_replaces_remainder"):
            venue = FakeVenue(current=100.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.fill("L2", 0.25, 100.64, status="expired")

            outcome = coordinator.step(now=12.0)

            replacement = coordinator.tickets[-1]
            self.assertEqual(replacement.side, "SELL")
            self.assertAlmostEqual(replacement.qty, 0.75)
            self.assertEqual(replacement.revision, 1)
            self.assertAlmostEqual(outcome.sell_qty, 0.25)
            self.assertAlmostEqual(outcome.net_qty, 0.75)
            self.assertEqual(
                len({ticket.order_id for ticket in coordinator.tickets}),
                len(coordinator.tickets))

        with self.subTest("skips_replacement_when_cancel_race_flattens_exposure"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="closed")
            venue.fill_on_cancel["L2"] = (0.4, 100.64)

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "complete")
            self.assertTrue(outcome.terminal)
            self.assertAlmostEqual(outcome.net_qty, 0.0)
            self.assertEqual(len(venue.orders), 2)

    def test_entry_cancel_and_exit_blocking_lifecycle(self):
        with self.subTest("partial_fill_cancels_entry_remainder_and_resizes_exit"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="open")

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 0.4)
            self.assertCountEqual(venue.canceled, ["L1", "L2"])
            replacement, pair_id = venue.orders[-1]
            self.assertEqual(replacement.side, "SELL")
            self.assertAlmostEqual(replacement.qty, 0.4)
            self.assertEqual(pair_id, "pair-1")
            self.assertGreater(replacement.price, 99.36)

        with self.subTest("refresh_restores_active_when_cancel_ack_is_not_terminal"):
            venue = FakeVenue()
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            ticket = coordinator.tickets[0]
            ticket.active = False
            venue.statuses[ticket.order_id] = OrderSnapshot(status="open")

            coordinator._refresh()

            self.assertTrue(ticket.active)

        with self.subTest("entry_cancel_false_blocks_any_exit_submit"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="open")
            venue.cancel_fail.add("L1")

            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(outcome.reason, "entry_cancel_failed")
            self.assertEqual(len(venue.orders), 2)
            self.assertTrue(coordinator.tickets[0].active)

        with self.subTest("entry_cancel_ack_with_open_status_blocks_any_exit_submit"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="open")

            def acknowledge_without_terminal(order_id):
                venue.canceled.append(order_id)
                return True

            venue.cancel = acknowledge_without_terminal
            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(outcome.reason, "entry_cancel_ambiguous")
            self.assertEqual(len(venue.orders), 2)
            self.assertTrue(coordinator.tickets[0].active)

        with self.subTest("balanced_during_entry_cancel_closes_other_active_ticket_first"):
            venue = FakeVenue(current=99.0)
            coordinator = _coordinator(venue)
            coordinator.start(mid=100.0)
            venue.fill("L1", 0.4, 99.36, status="open")
            normal_cancel = venue.cancel

            def cancel_with_opposite_fill(order_id):
                if order_id == "L1":
                    venue.canceled.append(order_id)
                    venue.statuses["L1"] = OrderSnapshot(
                        status="canceled", filled_qty=0.4,
                        cost=0.4 * 99.36, fee=0.1)
                    venue.statuses["L2"] = OrderSnapshot(
                        status="open", filled_qty=0.4,
                        cost=0.4 * 100.64, fee=0.1)
                    return True
                return normal_cancel(order_id)

            venue.cancel = cancel_with_opposite_fill
            outcome = coordinator.step(now=12.0)

            self.assertEqual(outcome.phase, "complete")
            self.assertEqual(venue.canceled, ["L1", "L2"])
            self.assertEqual(len(venue.orders), 2)
            self.assertFalse(any(ticket.active for ticket in coordinator.tickets))

    def test_checkpoint_roundtrip_restores_owned_tickets(self):
        venue = FakeVenue(current=100.0)
        coordinator = _coordinator(venue)
        coordinator.start(mid=100.0, pair_id="durable-pair")
        state = coordinator.export_state()

        restored = PairCoordinator.from_state(venue, coordinator.policy, state)

        self.assertEqual(restored.pair_id, "durable-pair")
        self.assertEqual(restored.phase, "quoting")
        self.assertEqual([t.order_id for t in restored.tickets], ["L1", "L2"])

        for bad_patch in (
            {"reason": True},
            {"hard_stop_reason": ["invalid"]},
            {"first_fill_side": True},
            {"first_fill_side": "SIDEWAYS"},
        ):
            with self.subTest(bad_patch=bad_patch):
                bad_state = dict(state, **bad_patch)
                with self.assertRaises(ValueError):
                    PairCoordinator.from_state(venue, coordinator.policy, bad_state)

    def test_hard_stop_thresholds_and_shock_policies(self):
        with self.subTest("stale_preflight_preserves_active_exit"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.preflight_error = SubmissionRefused("cache_stale")

            with self.assertRaisesRegex(SubmissionRefused, "cache_stale"):
                coordinator.step(now=5.0)

            self.assertEqual(venue.canceled, [])
            self.assertEqual(venue.market_calls, [])
            self.assertEqual(len(venue.orders), 2)
            self.assertTrue(venue.orders[1][0].active)
            self.assertEqual(
                venue.preflight_calls,
                [("SELL", 1.0, None, True,
                  "rtrade:fast_fill_hard_stop:pair-1")],
            )

        with self.subTest("fast_buy_fill_hard_stop_after_threshold"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            submitted = coordinator.step(now=5.0)
            finished = coordinator.step(now=6.0)

            self.assertEqual(submitted.phase, "stopping")
            self.assertEqual(venue.market_calls, [("SELL", 1.0, "fast_fill_hard_stop")])
            self.assertEqual(finished.phase, "hard_stop")

        with self.subTest("hard_stop_uses_net_remainder_after_partial_fill"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.fill_on_cancel["L2"] = (0.25, 100.64)

            outcome = coordinator.step(now=5.0)

            self.assertEqual(outcome.phase, "stopping")
            self.assertEqual(
                venue.market_calls,
                [("SELL", 0.75, "fast_fill_hard_stop")])
            self.assertEqual(venue.preflight_calls[0][1], 1.0)
            self.assertIs(venue.market_permits[-1], venue.preflight_permit)

        with self.subTest("hard_stop_skips_market_when_race_flattens_exposure"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.fill_on_cancel["L2"] = (1.0, 100.64)

            outcome = coordinator.step(now=5.0)

            self.assertEqual(outcome.phase, "hard_stop")
            self.assertTrue(outcome.terminal)
            self.assertAlmostEqual(outcome.net_qty, 0.0)
            self.assertEqual(venue.market_calls, [])

        with self.subTest("slow_buy_fill_does_not_use_tighter_shock_threshold"):
            venue = FakeVenue(current=96.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            outcome = coordinator.step(now=20.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertFalse(outcome.shock)
            self.assertEqual(venue.market_calls, [])

        with self.subTest("slow_buy_fill_still_has_wider_inventory_hard_stop"):
            venue = FakeVenue(current=90.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04, hard_stop_fraction=0.08)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            submitted = coordinator.step(now=20.0)
            finished = coordinator.step(now=21.0)

            self.assertFalse(submitted.shock)
            self.assertEqual(
                venue.market_calls, [("SELL", 1.0, "inventory_hard_stop")])
            self.assertEqual(finished.phase, "hard_stop")
            self.assertEqual(finished.reason, "inventory_hard_stop")

        with self.subTest("dynamic_policy_keeps_anchored_limit_when_not_justified"):
            venue = FakeVenue(current=90.0)
            venue.allow_market = False
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04, hard_stop_fraction=0.08)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            outcome = coordinator.step(now=20.0)

            self.assertEqual(outcome.phase, "exposed")
            self.assertEqual(outcome.reason, "market_exit_waiting_for_trend")
            self.assertEqual(venue.market_calls, [])
            self.assertTrue(any(ticket.side == "SELL" and ticket.active
                                for ticket, _pair in venue.orders))

        with self.subTest("fast_sell_fill_has_symmetric_market_buy_hard_stop"):
            venue = FakeVenue(current=105.0)
            coordinator = _coordinator(
                venue, start_side="SELL", quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 100.64)

            submitted = coordinator.step(now=5.0)
            finished = coordinator.step(now=6.0)

            self.assertEqual(submitted.phase, "stopping")
            self.assertEqual(venue.market_calls, [("BUY", 1.0, "fast_fill_hard_stop")])
            self.assertEqual(finished.phase, "hard_stop")
            self.assertAlmostEqual(finished.net_qty, 0.0)

        with self.subTest("failed_hard_stop_does_not_reopen_limit_and_recovers_on_bounce"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            venue.place_market_exit = lambda *_a, **_kw: None

            failed_stop = coordinator.step(now=5.0)
            self.assertEqual(failed_stop.phase, "exposed")
            self.assertEqual(failed_stop.reason, "hard_stop_place_failed")
            self.assertEqual(len(venue.orders), 2)

            venue.current = 98.0
            bounced = coordinator.step(now=6.0)
            self.assertEqual(bounced.phase, "exposed")
            self.assertIsNone(bounced.reason)
            self.assertEqual(len(venue.orders), 3)
            self.assertEqual(venue.orders[-1][0].side, "SELL")

    def test_hard_stop_revisions_and_recovery(self):
        with self.subTest("ambiguous_hard_stop_recovered_without_second_market_submit"):
            venue = FakeVenue(current=100.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=20.0,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            buy_id = coordinator.tickets[0].order_id
            venue.fill(buy_id, 1.0, 99.36)
            venue.current = 90.0
            submit_calls = []

            def ambiguous_market(*args, **kwargs):
                submit_calls.append((args, kwargs))
                venue.recovery_blocked = True
                venue.failure_reasons["SELL"] = "sell_recovery_pending"
                return None

            venue.place_market_exit = ambiguous_market
            blocked = [True]
            recovered_stop = OrderTicket(
                order_id="late-stop", side="SELL", price=90.0, qty=1.0)
            recovered_snapshot = OrderSnapshot(
                status="closed", filled_qty=1.0,
                cost=90.0, fee=0.01)
            recovery = [
                [],
                [(recovered_stop, recovered_snapshot, "hard_stop")],
            ]
            venue.recover_pair_intents = lambda _pair_id: recovery.pop(0)
            venue.pair_recovery_blocked = lambda _pair_id: blocked[0]

            pending = coordinator.step(now=1.0)
            self.assertEqual(pending.phase, "hard_stop_recovery")
            self.assertEqual(len(submit_calls), 1)
            coordinator.step(now=2.0)
            blocked[0] = False
            venue.statuses["late-stop"] = recovered_snapshot
            finished = coordinator.step(now=3.0)

            self.assertEqual(finished.phase, "hard_stop")
            self.assertEqual(finished.net_qty, 0.0)
            self.assertEqual(len(submit_calls), 1)
            self.assertTrue(finished.terminal)
            self.assertAlmostEqual(finished.net_qty, 0.0)

        with self.subTest("terminal_partial_hard_stop_submits_one_remainder_revision"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            coordinator.step(now=5.0)
            first_stop = coordinator.stop_ticket
            venue.fill(
                first_stop.order_id, 0.4, 95.0, status="canceled")

            retried = coordinator.step(now=6.0)

            self.assertEqual(retried.phase, "stopping")
            self.assertEqual(venue.market_calls[-1],
                             ("SELL", 0.6, "fast_fill_hard_stop"))
            self.assertEqual(venue.market_revisions, [0, 1])
            self.assertNotEqual(coordinator.stop_ticket.order_id, first_stop.order_id)

        with self.subTest("revision_one_response_loss_recovers_without_resubmit"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            coordinator.step(now=5.0)
            first_stop = coordinator.stop_ticket
            venue.fill(
                first_stop.order_id, 0.4, 95.0, status="canceled")
            ambiguous_calls = []

            def ambiguous_revision(*args, **kwargs):
                ambiguous_calls.append((args, kwargs))
                venue.recovery_blocked = True
                venue.failure_reasons["SELL"] = "sell_recovery_pending"
                return None

            venue.place_market_exit = ambiguous_revision
            blocked = [True]
            recovered_stop = OrderTicket(
                order_id="late-stop-r1", side="SELL", price=95.0, qty=0.6)
            recovered_snapshot = OrderSnapshot(
                status="closed", filled_qty=0.6,
                cost=57.0, fee=0.01)
            recovery = [
                [],
                [(recovered_stop, recovered_snapshot, "hard_stop_1")],
            ]
            venue.recover_pair_intents = lambda _pair_id: recovery.pop(0)
            venue.pair_recovery_blocked = lambda _pair_id: blocked[0]

            pending = coordinator.step(now=6.0)
            self.assertEqual(pending.phase, "hard_stop_recovery")
            self.assertEqual(pending.reason, "hard_stop_recovery_pending")
            self.assertEqual(len(ambiguous_calls), 1)
            coordinator.step(now=7.0)
            blocked[0] = False
            venue.statuses["late-stop-r1"] = recovered_snapshot
            finished = coordinator.step(now=8.0)

            self.assertEqual(finished.phase, "hard_stop")
            self.assertEqual(finished.net_qty, 0.0)
            self.assertEqual(len(ambiguous_calls), 1)
            self.assertEqual(coordinator.hard_stop_revision, 1)

        with self.subTest("revision_one_pre_submit_failure_is_retried_on_later_tick"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            coordinator.step(now=5.0)
            first_stop = coordinator.stop_ticket
            venue.fill(
                first_stop.order_id, 0.4, 95.0, status="canceled")
            venue.preflight_error = SubmissionRefused("cache_stale")

            deferred = coordinator.step(now=6.0)
            self.assertEqual(deferred.phase, "hard_stop_recovery")
            self.assertEqual(deferred.reason, "hard_stop_retry_failed")
            self.assertEqual(venue.market_revisions, [0])

            venue.preflight_error = None
            retried = coordinator.step(now=7.0)

            self.assertEqual(retried.phase, "stopping")
            self.assertEqual(venue.market_revisions, [0, 1])
            self.assertEqual(venue.market_calls[-1],
                             ("SELL", 0.6, "fast_fill_hard_stop"))

        with self.subTest("terminal_zero_fill_hard_stop_submits_one_full_revision"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)
            coordinator.step(now=5.0)
            first_stop = coordinator.stop_ticket
            venue.fill(
                first_stop.order_id, 0.0, 95.0, status="expired", fee=0.0)

            retried = coordinator.step(now=6.0)

            self.assertEqual(retried.phase, "stopping")
            self.assertEqual(venue.market_calls[-1],
                             ("SELL", 1.0, "fast_fill_hard_stop"))
            self.assertEqual(venue.market_revisions, [0, 1])

        with self.subTest("second_leg_exception_with_first_cancel_failure_preserves_recovery_pending"):
            venue = FakeVenue(current=100.0)
            orig_place = venue.place_limit
            calls = []

            def place_second_raises(side, price, qty, pair_id, *, cache_permit=None, revision=0):
                calls.append(side)
                if len(calls) == 2:
                    raise RuntimeError("socket timeout during second leg submit")
                return orig_place(side, price, qty, pair_id, cache_permit=cache_permit, revision=revision)

            venue.place_limit = place_second_raises
            venue.cancel = lambda _order_id: False
            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertTrue(outcome.reason.endswith("_recovery_pending"))

        with self.subTest("initial_hard_stop_exception_transitions_to_hard_stop_recovery_pending"):
            venue = FakeVenue(current=95.0)
            coordinator = _coordinator(
                venue, quote_ttl_sec=32, fast_fill_ratio=0.25,
                shock_hard_stop_fraction=0.04)
            coordinator.start(mid=100.0)
            venue.fill("L1", 1.0, 99.36)

            def raise_on_market(*_a, **_kw):
                raise RuntimeError("socket timeout on initial hard stop")

            venue.place_market_exit = raise_on_market
            outcome = coordinator.step(now=5.0)
            self.assertEqual(outcome.phase, "hard_stop_recovery")
            self.assertEqual(outcome.reason, "hard_stop_recovery_pending")


import json as _json
import os as _os
import tempfile as _tempfile
from pathlib import Path as _Path
from unittest.mock import patch as _patch
import rtrade as _rtrade
from rtrade_pair_store import RTradePairStore, rtrade_client_order_id

_RTRADE_ROOT = _Path(__file__).resolve().parents[1]


class RTradeHeartbeatTest(unittest.TestCase):
    def setUp(self):
        self._last = _rtrade._rtrade_heartbeat_last

    def tearDown(self):
        _rtrade._rtrade_heartbeat_last = self._last

    def test_heartbeat_is_written_and_throttled_by_loop_progress(self):
        with _tempfile.TemporaryDirectory() as tmp:
            path = _os.path.join(tmp, "rtrade.heartbeat")
            with (
                _patch.object(_rtrade, "_RTRADE_HEARTBEAT_PATH", path),
                _patch.object(_rtrade, "_RTRADE_HEARTBEAT_INTERVAL_SEC", 30.0),
                _patch.object(_rtrade.time, "monotonic", side_effect=(100.0, 110.0, 131.0)),
            ):
                _rtrade._rtrade_heartbeat_last = float("-inf")
                _rtrade._touch_rtrade_heartbeat()
                self.assertTrue(_os.path.exists(path))
                self.assertEqual(_rtrade._rtrade_heartbeat_last, 100.0)

                _rtrade._touch_rtrade_heartbeat()
                self.assertEqual(_rtrade._rtrade_heartbeat_last, 100.0)

                _rtrade._touch_rtrade_heartbeat()
                self.assertEqual(_rtrade._rtrade_heartbeat_last, 131.0)

    def test_process_manifest_uses_dedicated_heartbeat(self):
        line = next(
            row for row in (_RTRADE_ROOT / "procs.conf").read_text(encoding="utf-8").splitlines()
            if row.startswith("rtrade.py|")
        )
        parts = line.split("|")
        self.assertEqual(parts[0], "rtrade.py")
        self.assertEqual(parts[1], "$ROOT")
        self.assertEqual(parts[3], "rtrade")
        hb_targets = [f.strip() for f in parts[4].split(",")]
        self.assertIn("cachedb/rtrade.heartbeat", hb_targets)
        self.assertEqual(parts[5], "180")
        self.assertEqual(parts[6], "fleet")

    def test_rtrade_imports_sys_and_defines_recovery_timeout(self):
        self.assertTrue(hasattr(_rtrade, "sys"), "rtrade must import sys for sys.exit")
        self.assertTrue(callable(_rtrade.sys.exit))
        self.assertTrue(hasattr(_rtrade, "_RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC"))
        self.assertLess(
            _rtrade._RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC, 180.0,
            "Startup recovery timeout must be strictly less than orchestrator 180s threshold"
        )

    def test_venue_pair_recovery_blocked_management(self):
        venue = _rtrade._LivePairVenue.__new__(_rtrade._LivePairVenue)
        venue._recovery_blocked_pairs = set()
        venue.recovery_blocked = False
        venue._set_pair_recovery_blocked("pair-1", True)
        self.assertTrue(venue.recovery_blocked)
        self.assertTrue(venue.pair_recovery_blocked("pair-1"))
        venue._set_pair_recovery_blocked("pair-1", False)
        self.assertFalse(venue.recovery_blocked)
        self.assertFalse(venue.pair_recovery_blocked("pair-1"))

    def test_coordinator_loop_exits_on_venue_recovery_blocked_timeout(self):
        from unittest.mock import MagicMock
        bot = _rtrade.TradingBot.__new__(_rtrade.TradingBot)
        bot.symbol = "TAOUSDC"
        bot.DEFAULT_ADJUSTMENT_PERCENT = 0.005
        mock_venue = MagicMock()
        mock_venue.recovery_blocked = True
        mock_venue.executor.open_orders.return_value = []
        mock_pair_store = MagicMock()
        mock_pair_store.active.return_value = []

        start_time = 1000.0
        timeout = _rtrade._RTRADE_STARTUP_RECOVERY_TIMEOUT_SEC
        monotonic_times = [start_time, start_time + 1.0, start_time + timeout + 5.0]
        with (
            _patch.object(_rtrade.time, "monotonic", side_effect=monotonic_times),
            _patch.object(_rtrade.time, "sleep", return_value=None),
            _patch.object(_rtrade.sys, "exit") as mock_exit,
            _patch.object(_rtrade, "_LivePairVenue", return_value=mock_venue),
            _patch.object(_rtrade, "RTradePairStore", return_value=mock_pair_store),
        ):
            mock_exit.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                bot._run_coordinator_forever()
            mock_exit.assert_called_once_with(1)

    def test_startup_recovery_terminal_record_clears_venue_blockage(self):
        from unittest.mock import MagicMock
        bot = _rtrade.TradingBot.__new__(_rtrade.TradingBot)
        bot.symbol = "TAOUSDC"
        bot.DEFAULT_ADJUSTMENT_PERCENT = 0.005
        mock_venue = MagicMock()
        mock_venue.recovery_blocked = False
        mock_pair_store = MagicMock()
        record = {
            "pair_id": "test_pair_1",
            "qty": 1.0,
            "start_side": "SELL",
            "state": {"phase": "quoting", "tickets": []},
            "intents": {"limit:SELL": {"recovery_state": "absence_confirmed_no_reuse"}},
        }
        mock_pair_store.active.return_value = [record]
        mock_venue.merge_checkpoint_intents.return_value = record["state"]
        mock_venue.executor.open_orders.return_value = []

        with (
            _patch.object(_rtrade, "_LivePairVenue", return_value=mock_venue),
            _patch.object(_rtrade, "RTradePairStore", return_value=mock_pair_store),
            _patch.object(_rtrade.time, "sleep", side_effect=StopIteration),
        ):
            try:
                bot._run_coordinator_forever()
            except StopIteration:
                pass
            mock_pair_store.checkpoint.assert_any_call("test_pair_1", record["state"], terminal=True)
            mock_venue._set_pair_recovery_blocked.assert_called_with("test_pair_1", False)

    def test_all_procs_conf_entries_have_dedicated_heartbeat_and_stale_threshold(self):
        rows = [
            row.strip()
            for row in (_RTRADE_ROOT / "procs.conf").read_text(encoding="utf-8").splitlines()
            if row.strip() and not row.strip().startswith("#")
        ]
        self.assertGreaterEqual(len(rows), 17)
        for row in rows:
            parts = row.split("|")
            self.assertEqual(len(parts), 7, f"Malformed procs.conf row: {row}")
            label = parts[3].strip()
            hb_targets = [f.strip() for f in parts[4].split(",") if f.strip()]
            stale_s = float(parts[5].strip() or 0)
            self.assertTrue(
                any(t.endswith(".heartbeat") for t in hb_targets),
                f"Process {label} must have a dedicated .heartbeat target in procs.conf: {row}",
            )
            self.assertGreater(stale_s, 0.0, f"Process {label} must have a positive hb_stale_s in procs.conf")

    def test_touch_module_heartbeat_and_harvest_fleet_heartbeats(self):
        import botcore
        from orchestratorOS.lib.technical_repair_ssot import harvest_heartbeats_and_state_files

        with _tempfile.TemporaryDirectory() as tmp:
            (_Path(tmp) / "system" / "heartbeat").mkdir(parents=True, exist_ok=True)
            with _patch.dict(_os.environ, {"MPTRADE_CACHEDB_DIR": tmp}, clear=False):
                ts1 = botcore.touch_module_heartbeat(
                    "cacheManager", interval_sec=30.0, force=True, now=500.0,
                    status={"ts": 1700000500.0}, symbols=["BTCUSDC", "TAOUSDC"],
                )
                self.assertEqual(ts1, 500.0)
                hb_file = _Path(tmp) / "system" / "heartbeat" / "cacheManager.heartbeat"
                self.assertTrue(hb_file.exists())
                data = _json.loads(hb_file.read_text(encoding="utf-8"))
                self.assertEqual(data["status"], "alive")
                self.assertEqual(data["symbols"], ["BTCUSDC", "TAOUSDC"])

                ts2 = botcore.touch_module_heartbeat("cacheManager", interval_sec=30.0, now=510.0)
                self.assertEqual(ts2, 500.0)

                ts3 = botcore.touch_module_heartbeat(
                    "cacheManager", interval_sec=30.0, now=535.0, symbols=["BTCUSDC"],
                )
                self.assertEqual(ts3, 535.0)
                data_after = _json.loads(hb_file.read_text(encoding="utf-8"))
                self.assertEqual(data_after["symbols"], ["BTCUSDC", "TAOUSDC"])
                self.assertEqual(data_after["status"], "alive")

                mtime = _os.path.getmtime(str(hb_file))
                active_hb, _ = harvest_heartbeats_and_state_files(tmp, tmp, mtime + 1500.0)
                self.assertIn("cacheManager.heartbeat", active_hb)
                self.assertEqual(active_hb["cacheManager.heartbeat"]["status"], "alive")

                log_file = _Path(tmp) / "Kraken-trailing.log"
                log_file.write_text("quiet stdout\n", encoding="utf-8")
                _os.utime(str(log_file), (mtime - 3600.0, mtime - 3600.0))
                fresh, age_s, _, stale_path = botcore.evaluate_heartbeat_files_freshness(
                    [str(hb_file), str(log_file)], 900.0, mtime + 30.0,
                )
                self.assertTrue(fresh, f"Expected fresh=True when .heartbeat is fresh even if .log is quiet, got {stale_path}")
                self.assertAlmostEqual(age_s, 30.0, delta=1.0)

                stale_fresh, _, _, stale_reason = botcore.evaluate_heartbeat_files_freshness(
                    [str(hb_file), str(log_file)], 900.0, mtime + 950.0,
                )
                self.assertFalse(stale_fresh)
                self.assertIn("cacheManager.heartbeat", stale_reason)


class RTradePairStoreTest(unittest.TestCase):
    def test_the_client_order_id_is_stable_and_rtrade_specific(self):
        first = rtrade_client_order_id("pair-1", "buy")
        self.assertEqual(first, rtrade_client_order_id("pair-1", "BUY"))
        self.assertTrue(first.startswith("RT_"))
        self.assertEqual(len(first), 35)
        self.assertNotEqual(first, rtrade_client_order_id("pair-1", "SELL"))
        self.assertNotEqual(first, rtrade_client_order_id("pair-1", "BUY", "hard_stop"))

    def setUp(self):
        self.tmp = _tempfile.TemporaryDirectory()
        self.store = RTradePairStore(_os.path.join(self.tmp.name, "pairs.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_existing_invalid_store_is_preserved(self, payload):
        original = _json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with open(self.store.path, "wb") as handle:
            handle.write(original)

        with self.assertRaises(ValueError):
            self.store.active("TAOUSDC")
        self.assertEqual(open(self.store.path, "rb").read(), original)

        with self.assertRaises(ValueError):
            self.store.begin("TAOUSDC", "new-pair", "BUY", 1.0)
        self.assertEqual(open(self.store.path, "rb").read(), original)

    def test_existing_invalid_roots_fail_closed_without_rewrite(self):
        for payload in ([], {}, {"version": 1, "pairs": []}, {"pairs": []}, {"version": 2, "pairs": {}}):
            with self.subTest(payload=payload):
                self._assert_existing_invalid_store_is_preserved(payload)

    def test_existing_malformed_active_pair_fails_closed_without_rewrite(self):
        self._assert_existing_invalid_store_is_preserved({
            "version": 1,
            "pairs": {
                "pair-1": {
                    "symbol": "TAOUSDC", "pair_id": "another-pair", "start_side": "BUY",
                    "qty": 1.0, "phase": "reserved", "terminal": False,
                    "intents": {}, "state": None, "created_ts": 100.0, "updated_ts": 100.0,
                },
            },
        })

    def test_existing_malformed_intent_identity_fails_closed_without_rewrite(self):
        self._assert_existing_invalid_store_is_preserved({
            "version": 1,
            "pairs": {
                "pair-1": {
                    "symbol": "TAOUSDC", "pair_id": "pair-1", "start_side": "BUY",
                    "qty": 1.0, "phase": "reserved", "terminal": False,
                    "intents": {
                        "limit:BUY": {
                            "side": "SELL", "kind": "limit", "qty": 1.0, "price": 100.0,
                            "client_order_id": "RT_invalid", "order_id": None,
                        },
                    },
                    "state": None, "created_ts": 100.0, "updated_ts": 100.0,
                },
            },
        })

    def test_valid_legacy_v1_record_loads_and_gains_only_requested_checkpoint(self):
        legacy = {
            "version": 1,
            "pairs": {
                "pair-v1": {
                    "symbol": "TAOUSDC", "pair_id": "pair-v1", "start_side": "BUY",
                    "qty": 1.0, "phase": "reserved", "terminal": False,
                    "intents": {
                        "limit:BUY": {
                            "side": "BUY", "kind": "limit", "qty": 1.0, "price": 99.0,
                            "client_order_id": "RT_legacy", "order_id": None,
                        },
                    },
                    "state": None, "created_ts": 100.0, "updated_ts": 100.0,
                },
            },
        }
        with open(self.store.path, "w", encoding="utf-8", newline="\n") as handle:
            _json.dump(legacy, handle, separators=(",", ":"))

        loaded = self.store.active("TAOUSDC")
        self.assertEqual(loaded[0]["intents"]["limit:BUY"]["qty"], 1.0)

        state = {"pair_id": "pair-v1", "qty": 1.0, "start_side": "BUY", "phase": "quoting", "tickets": []}
        self.store.checkpoint("pair-v1", state, terminal=False)
        updated = self.store.active("TAOUSDC")[0]
        self.assertEqual(updated["state"], state)
        self.assertNotIn("limit_revisions", updated["state"])

    def test_intent_is_durable_before_acceptance_and_checkpoint(self):
        self.store.begin("TAOUSDC", "pair-1", "BUY", 2.0)
        self.store.intent("pair-1", "BUY", 100.0, 2.0, "SD_client", kind="limit")
        pending = self.store.active("TAOUSDC")[0]
        self.assertIsNone(pending["intents"]["limit:BUY"]["order_id"])

        self.store.accepted("pair-1", "BUY", "77", kind="limit")
        state = {"pair_id": "pair-1", "qty": 2.0, "start_side": "BUY", "phase": "quoting", "tickets": []}
        self.store.checkpoint("pair-1", state, terminal=False)
        adopted = self.store.active("TAOUSDC")[0]
        self.assertEqual(adopted["intents"]["limit:BUY"]["order_id"], "77")
        self.assertEqual(adopted["state"]["phase"], "quoting")

    def test_terminal_pair_is_not_recovered(self):
        self.store.begin("TAOUSDC", "pair-2", "SELL", 1.0)
        self.store.checkpoint("pair-2", {"pair_id": "pair-2", "phase": "complete", "tickets": []}, terminal=True)
        self.assertEqual(self.store.active("TAOUSDC"), [])

    def test_intent_can_atomically_create_pair_without_separate_begin(self):
        self.store.intent("pair-3", "BUY", 100.0, 1.0, "SD_atomic", kind="limit", symbol="TAOUSDC", start_side="BUY")
        rec = self.store.active("TAOUSDC")[0]
        self.assertEqual(rec["pair_id"], "pair-3")
        self.assertEqual(rec["intents"]["limit:BUY"]["client_order_id"], "SD_atomic")

    def test_canonical_tracked_intent_replaces_and_removes_exact_leg(self):
        pending = {
            "intent_id": "rtrade:pair-4:limit:buy", "client_order_id": "RT_canonical",
            "symbol": "TAOUSDC", "side": "BUY", "kind": "limit",
            "requested_qty": 1.25, "requested_price": 99.5,
            "attempt": 1, "created_at": 100.0, "lookup_misses": 0,
        }
        self.store.persist_intent("pair-4", "BUY", "limit", pending, symbol="TAOUSDC", start_side="BUY")
        rec = self.store.active("TAOUSDC")[0]
        stored = rec["intents"]["limit:BUY"]
        self.assertEqual(stored["requested_qty"], 1.25)
        self.assertEqual(stored["requested_price"], 99.5)
        self.assertEqual(stored["qty"], 1.25)
        self.assertEqual(stored["price"], 99.5)

        accepted = dict(pending, order_id="77", submitted_qty=1.2)
        self.store.persist_intent("pair-4", "BUY", "limit", accepted, symbol="TAOUSDC")
        rec = self.store.active("TAOUSDC")[0]
        self.assertEqual(rec["intents"]["limit:BUY"]["order_id"], "77")

        self.store.persist_intent("pair-4", "BUY", "limit", None, symbol="TAOUSDC")
        rec = self.store.active("TAOUSDC")[0]
        self.assertNotIn("limit:BUY", rec["intents"])

    def test_checkpoint_many_updates_rounds_in_one_transaction(self):
        for pair_id in ("p1", "p2"):
            self.store.begin("TAOUSDC", pair_id, "BUY", 1.0)
        self.store.checkpoint_many([
            ("p1", {"pair_id": "p1", "qty": 1.0, "start_side": "BUY", "phase": "quoting", "tickets": []}, False),
            ("p2", {"phase": "complete"}, True),
        ])
        active = self.store.active("TAOUSDC")
        self.assertEqual([rec["pair_id"] for rec in active], ["p1"])

    def test_terminal_history_is_bounded(self):
        store = RTradePairStore(_os.path.join(self.tmp.name, "bounded.json"), terminal_retention=2)
        for i in range(4):
            pair_id = f"t{i}"
            store.begin("TAOUSDC", pair_id, "BUY", 1.0)
            store.checkpoint(pair_id, {"phase": "complete"}, terminal=True)
        with open(store.path, encoding="utf-8") as handle:
            pairs = _json.load(handle)["pairs"]
        self.assertEqual(len(pairs), 2)

    def test_default_in_memory_store_creates_no_disk_file(self):
        store = RTradePairStore()
        self.assertIsNone(store.path)
        self.assertIsNone(store.lock_path)
        store.begin("TAOUSDC", "mem-1", "BUY", 1.0)
        self.assertEqual(len(store.active("TAOUSDC")), 1)
        store.checkpoint("mem-1", {"phase": "complete"}, terminal=True)
        self.assertEqual(store.active("TAOUSDC"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
