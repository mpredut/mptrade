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
    def test_quotes_share_one_midpoint(self):
        self.assertEqual(quote_prices(100.0, 0.0064), (99.36, 100.64))

    def test_anchored_exit_prices(self):
        with self.subTest(msg="long_never_follows_market_below_cost_plus_edge"):
            target = anchored_exit_price("SELL", 100.0, 90.0, 0.0064, 0.0115)
            self.assertEqual(target, 101.1634)
            self.assertGreater(target, 100.0)

        with self.subTest(msg="sold_never_chases_market_above_profitable_buyback"):
            target = anchored_exit_price("BUY", 100.0, 110.0, 0.0064, 0.0115)
            self.assertEqual(target, 98.85)
            self.assertLess(target, 100.0)

    def test_nonfinite_policy_and_quantity_fail_closed(self):
        with self.assertRaises(ValueError):
            PairPolicy(adjustment_fraction=float("nan"))
        with self.assertRaises(ValueError):
            PairCoordinator(FakeVenue(), qty=float("inf"),
                            policy=PairPolicy(adjustment_fraction=0.0064))


class PairCoordinatorTest(unittest.TestCase):
    def test_places_both_legs_under_same_pair_id(self):
        venue = FakeVenue()
        outcome = _coordinator(venue).start(mid=100.0)

        self.assertEqual(outcome.phase, "quoting")
        self.assertEqual(
            [(t.side, t.price, pair) for t, pair in venue.orders],
            [("BUY", 99.36, "pair-1"), ("SELL", 100.64, "pair-1")])

    def test_second_leg_failure_behaviors(self):
        with self.subTest(msg="cancels_first_and_fails_closed"):
            venue = FakeVenue(fail_side="SELL")
            outcome = _coordinator(venue).start(mid=100.0)

            self.assertTrue(outcome.terminal)
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertEqual(venue.canceled, ["L1"])

        with self.subTest(msg="with_unconfirmed_cancel_remains_recoverable"):
            venue = FakeVenue(fail_side="SELL")
            venue.cancel_fail.add("L1")

            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)

            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertTrue(coordinator.tickets[0].active)
            self.assertEqual(coordinator.export_state()["phase"], "startup_recovery")

        with self.subTest(msg="with_cancel_exception_remains_recoverable"):
            venue = FakeVenue(fail_side="SELL")
            venue.cancel_raise.add("L1")

            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)

            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertTrue(coordinator.tickets[0].active)

        with self.subTest(msg="exception_keeps_the_first_leg_managed"):
            venue = FakeVenue()
            venue.raise_side = "SELL"
            venue.cancel_fail.add("L1")

            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)

            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "startup_recovery")
            self.assertEqual([ticket.order_id for ticket in coordinator.tickets], ["L1"])

    def test_first_leg_response_loss_stays_managed_until_later_recovery(self):
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

    def test_ambiguous_second_leg_is_adopted_without_another_submit(self):
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

    def test_second_leg_failure_fill_during_cancel_behaviors(self):
        with self.subTest(msg="partial_fill"):
            venue = FakeVenue(fail_side="SELL")
            venue.fill_on_cancel["L1"] = (0.4, 99.36)

            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)

            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 0.4)

        with self.subTest(msg="full_fill"):
            venue = FakeVenue(fail_side="SELL")
            venue.fill_on_cancel["L1"] = (1.0, 99.36)

            coordinator = _coordinator(venue)
            outcome = coordinator.start(mid=100.0)

            self.assertFalse(outcome.terminal)
            self.assertEqual(outcome.phase, "exposed")
            self.assertAlmostEqual(outcome.net_qty, 1.0)

    def test_sell_first_behaviors(self):
        with self.subTest(msg="places_both_legs_in_reverse_order"):
            venue = FakeVenue()
            outcome = _coordinator(venue, start_side="SELL").start(mid=100.0)

            self.assertEqual(outcome.phase, "quoting")
            self.assertEqual(
                [(t.side, t.price, pair) for t, pair in venue.orders],
                [("SELL", 100.64, "pair-1"), ("BUY", 99.36, "pair-1")])

        with self.subTest(msg="failure_does_not_attempt_buy"):
            venue = FakeVenue(fail_side="SELL")
            outcome = _coordinator(venue, start_side="SELL").start(mid=100.0)

            self.assertTrue(outcome.terminal)
            self.assertEqual(outcome.reason, "sell_place_failed")
            self.assertEqual(venue.orders, [])

    def test_venue_specific_insufficient_funds_reason_is_preserved(self):
        venue = FakeVenue(fail_side="BUY")
        venue.failure_reasons["BUY"] = "buy_insufficient_funds:USDC"

        outcome = _coordinator(venue).start(mid=100.0)

        self.assertEqual(outcome.reason, "buy_insufficient_funds:USDC")

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

    def test_both_fills_complete_cycle_and_measure_fast_latency(self):
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

    def test_slow_single_fill_keeps_existing_profitable_opposite_leg(self):
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

    def test_stale_preflight_preserves_active_anchored_exit(self):
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

    def test_permit_is_forwarded_when_cache_turns_stale_during_cancel(self):
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

    def test_anchored_exit_uses_net_remainder_after_fill_during_cancel(self):
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

    def test_anchored_replacement_advances_and_persists_limit_revision(self):
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

    def test_zero_fill_expired_exit_advances_revision_once(self):
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

    def test_partial_expired_exit_counts_fill_once_and_replaces_remainder(self):
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

    def test_anchored_exit_skips_replacement_when_cancel_race_flattens_exposure(self):
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

    def test_partial_fill_cancels_entry_remainder_and_resizes_exit(self):
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

    def test_refresh_restores_active_when_cancel_ack_is_not_terminal(self):
        venue = FakeVenue()
        coordinator = _coordinator(venue)
        coordinator.start(mid=100.0)
        ticket = coordinator.tickets[0]
        ticket.active = False
        venue.statuses[ticket.order_id] = OrderSnapshot(status="open")

        coordinator._refresh()

        self.assertTrue(ticket.active)

    def test_entry_cancel_false_blocks_any_exit_submit(self):
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

    def test_entry_cancel_ack_with_open_status_blocks_any_exit_submit(self):
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

    def test_balanced_during_entry_cancel_closes_other_active_ticket_first(self):
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

    def test_stale_preflight_preserves_active_exit_before_hard_stop(self):
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

    def test_fast_buy_fill_hard_stop_exits_market_only_after_threshold(self):
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

    def test_ambiguous_hard_stop_is_recovered_without_second_market_submit(self):
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

    def test_hard_stop_fill_during_cancel_behaviors(self):
        with self.subTest(msg="uses_net_remainder_after_partial_fill"):
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

        with self.subTest(msg="skips_market_when_race_flattens_exposure"):
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

    def test_terminal_partial_hard_stop_submits_one_remainder_revision(self):
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

    def test_revision_one_response_loss_recovers_without_resubmit(self):
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

    def test_revision_one_pre_submit_failure_is_retried_on_later_tick(self):
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

    def test_terminal_zero_fill_hard_stop_submits_one_full_revision(self):
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

    def test_slow_buy_fill_does_not_use_tighter_shock_threshold(self):
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

    def test_slow_buy_fill_still_has_wider_inventory_hard_stop(self):
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

    def test_dynamic_policy_keeps_anchored_limit_when_market_not_justified(self):
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

    def test_fast_sell_fill_has_symmetric_market_buy_hard_stop(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
