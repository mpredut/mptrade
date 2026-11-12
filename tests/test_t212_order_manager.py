"""Non-blocking durability and response-loss recovery for T212 one-shot orders."""

import json
from types import SimpleNamespace
import unittest

import pytest

import account_cache_health as bch
from binance_api import bapi_ws_bridge as bapi_ws_health
from cacheaccount import CacheOrderManager
from tests.conftest import load_venue_modules

order_manager = load_venue_modules("212trading", "order_manager")


TICKER = "NVDA_US_EQ"


class FakeClient:
    def __init__(self, submit_results=None):
        self.portfolio = []
        self.active = []
        self.statuses = {}
        self.submit_results = list(submit_results or [(201, {"id": "order-1", "status": "NEW"})])
        self.place_calls = []
        self.status_calls = []

    def get_portfolio(self):
        return self.portfolio

    def list_active_orders(self):
        return self.active

    def get_order_status(self, order_id):
        self.status_calls.append(str(order_id))
        return self.statuses.get(str(order_id))

    def place_limit_order(self, ticker, qty, limit, validity):
        self.place_calls.append((ticker, qty, limit, validity))
        result = self.submit_results.pop(0)
        if callable(result):
            result = result()
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def marker_env(tmp_path, monkeypatch):
    marker = tmp_path / "intent.json"
    test_marker = tmp_path / "intent.test.json"
    monkeypatch.setattr(
        order_manager,
        "_marker_path",
        lambda _ticker, test=False: str(test_marker if test else marker),
    )
    monkeypatch.setattr(order_manager, "get_price_usd", lambda _symbol: None)
    monkeypatch.setattr(order_manager, "notify", lambda **_kwargs: None)
    return marker


def _place(client, **overrides):
    args = {
        "client": client,
        "ticker": TICKER,
        "quantity": 1.25,
        "limit_price": 100.0,
        "validity": "DAY",
        "dry_run": False,
        "retry_delay": 0,
    }
    args.update(overrides)
    return order_manager.place_order_with_retry(**args)



class TestT212OrderManager(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup_env(self, marker_env, monkeypatch):
        self.marker_env = marker_env
        self.monkeypatch = monkeypatch

    def _reset_env(self):
        if self.marker_env.exists():
            self.marker_env.unlink()

    def test_submit_and_recovery(self):
        with self.subTest(msg="intent_is_fsynced_before_single_submit"):
            self._reset_env()
            observed = {}

            def inspect_pre_submit_marker():
                observed.update(json.loads(self.marker_env.read_text(encoding="utf-8")))
                return 201, {"id": "accepted-7", "status": "NEW"}

            client = FakeClient([inspect_pre_submit_marker])
            self.monkeypatch.setattr(order_manager.time, "sleep", pytest.fail)

            assert _place(client) is True

            assert len(client.place_calls) == 1
            assert observed["lifecycle"] == "submit_pending"
            assert observed["attempts"] == 1
            assert observed["intent_id"].startswith("t212-one-shot-NVDA_US_EQ-")
            final = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert final["lifecycle"] == "accepted"
            assert final["order_id"] == "accepted-7"

        with self.subTest(msg="response_loss_recovers_nested_active_order_without_resubmit"):
            self._reset_env()
            client = FakeClient([TimeoutError("response lost")])
            assert _place(client) is False

            client.active = [{
                "id": "recovered-1",
                "instrument": {"ticker": TICKER},
                "quantity": 1.25,
                "limitPrice": 100.0,
            }]
            assert _place(client) is True

            assert len(client.place_calls) == 1
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "accepted"
            assert record["order_id"] == "recovered-1"

        with self.subTest(msg="response_loss_recovers_portfolio_delta_without_resubmit"):
            self._reset_env()
            client = FakeClient([TimeoutError("response lost")])
            client.portfolio = [{"ticker": TICKER, "quantity": 2.0}]
            assert _place(client) is False

            client.portfolio = [{"ticker": TICKER, "quantity": 3.25}]
            assert _place(client) is True

            assert len(client.place_calls) == 1
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "filled"
            assert record["portfolio_fill_observed"] is True
            assert record["filled_qty"] == pytest.approx(1.25)

        with self.subTest(msg="ambiguous_submit_retries_only_after_two_complete_absence_snapshots"):
            self._reset_env()
            client = FakeClient([
                TimeoutError("response lost"),
                (201, {"id": "retry-2", "status": "NEW"}),
            ])

            assert _place(client) is False      # ambiguous submit is not reported accepted
            assert _place(client) is False      # first complete absence snapshot
            assert len(client.place_calls) == 1
            assert _place(client) is True       # second absence snapshot permits one retry

            assert len(client.place_calls) == 2
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["attempts"] == 2
            assert record["order_id"] == "retry-2"

    def test_rejections_and_failures(self):
        with self.subTest(msg="http_rejection_stays_pending_but_is_not_reported_as_success"):
            self._reset_env()
            client = FakeClient([(503, {"error": "temporarily unavailable"})])

            assert _place(client) is False

            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "submit_pending"
            assert "HTTP 503" in record["submit_error"]
            assert len(client.place_calls) == 1

        with self.subTest(msg="unavailable_portfolio_never_counts_as_proven_absence"):
            self._reset_env()
            client = FakeClient([TimeoutError("response lost")])
            assert _place(client) is False
            client.portfolio = None

            assert _place(client) is False
            assert _place(client) is False

            assert len(client.place_calls) == 1
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lookup_misses"] == 0

        with self.subTest(msg="canceled_order_is_terminal_and_never_retried"):
            self._reset_env()
            client = FakeClient([(201, {"id": "cancel-me", "status": "NEW"})])
            assert _place(client) is True
            client.statuses["cancel-me"] = {"id": "cancel-me", "status": "CANCELED"}

            assert _place(client) is False
            assert _place(client) is False

            assert len(client.place_calls) == 1
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "canceled"

        with self.subTest(msg="unknown_venue_status_fails_closed_without_resubmit"):
            self._reset_env()
            client = FakeClient([(201, {"id": "odd-1", "status": "NEW"})])
            assert _place(client) is True
            client.statuses["odd-1"] = {"id": "odd-1", "status": "MYSTERY_STATE"}

            assert _place(client) is True

            assert len(client.place_calls) == 1
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "status_unknown"

        with self.subTest(msg="corrupt_marker_blocks_submit_instead_of_becoming_absent"):
            self._reset_env()
            self.marker_env.write_text("{broken", encoding="utf-8")
            client = FakeClient()

            assert _place(client) is False
            assert client.place_calls == []

        with self.subTest(msg="accepted_order_absent_from_history_recovers_fill_from_portfolio"):
            self._reset_env()
            client = FakeClient([(201, {"id": "vanished-fill", "status": "NEW"})])
            client.portfolio = [{"ticker": TICKER, "quantity": 1.0}]
            assert _place(client) is True

            client.statuses["vanished-fill"] = False
            client.portfolio = [{"ticker": TICKER, "quantity": 2.25}]
            assert _place(client) is True
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "filled"
            assert record["filled_qty"] == pytest.approx(1.25)

        with self.subTest(msg="accepted_order_definitively_absent_without_fill_becomes_terminal_failed"):
            self._reset_env()
            client = FakeClient([(201, {"id": "vanished-cancel", "status": "NEW"})])
            client.portfolio = [{"ticker": TICKER, "quantity": 1.0}]
            assert _place(client) is True

            client.statuses["vanished-cancel"] = False
            assert _place(client) is True   # first absence snapshot stays waiting
            assert _place(client) is False  # second absence snapshot marks terminal_failed
            record = json.loads(self.marker_env.read_text(encoding="utf-8"))
            assert record["lifecycle"] == "canceled"
            assert record["venue_status"] == "ABSENT"

    def test_preflight_and_namespace(self):
        with self.subTest(msg="preflight_refuses_existing_matching_order_without_claiming_it"):
            self._reset_env()
            client = FakeClient()
            client.active = [{
                "id": "external-order",
                "ticker": TICKER,
                "quantity": 1.25,
                "limit": 100.0,
            }]

            assert _place(client) is False
            assert client.place_calls == []
            assert not self.marker_env.exists()

        with self.subTest(msg="marker_namespace_separates_profiles_and_symbols"):
            self._reset_env()
            
            # This test requires the unmocked _marker_path
            self.monkeypatch.undo() # this undoes ALL mocks from the monkeypatch fixture for this subtest!
            
            self.monkeypatch.setenv("IPO_PROFILE", "account one")
            first = order_manager._marker_path("NVDA_US_EQ")
            self.monkeypatch.setenv("IPO_PROFILE", "account two")
            second = order_manager._marker_path("NVDA_US_EQ")
            third = order_manager._marker_path("SPCX_US_EQ")

            assert first != second
            assert second != third
            assert "account_one" in first
            assert "account_two" in second

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_bapi_ws_health_normalizes_whitespace_padded_manager_names_and_class_iter131_s3():
    """[ITER131 #3] bapi_ws_health normalizes whitespace-padded manager names and class objects."""
    # 3. bapi_ws_health normalizes whitespace-padded manager names and class objects
    bapi_ws_health.mark_ws_available(True)
    bapi_ws_health.mark_ws_event_received()
    bapi_ws_health.mark_ws_unhealthy()
    assert bapi_ws_health.should_poll_for_manager(" CacheOrderManager ", ws_only_mode=True) is True
    bapi_ws_health.mark_ws_event_received()
    bapi_ws_health.clear_ws_rest_resync_for_manager(" CacheOrderManager ")
    assert bapi_ws_health.should_poll_for_manager(" CacheOrderManager ", ws_only_mode=True) is False


def test_account_cache_health_require_int_accepts_integer_floats_and_normalize_class_name_handles_iter133_s5():
    """[ITER133 #5] account_cache_health.require_int accepts integer floats and normalize_class_name handles class objects."""
    # 5. account_cache_health.require_int accepts integer floats and normalize_class_name handles class objects
    assert bch.require_int(100.0, "writer_pid", min_value=1) == 100
    assert bch.require_int(0.0, "generation", min_value=0) == 0
    assert bch.normalize_class_name(CacheOrderManager) == "CacheOrderManager"


def test_212trading_order_manager_waiting_with_sentinel_order_id_returns_false_iter153_s1(tmp_path, monkeypatch):
    """[ITER153 #1] 212trading/order_manager: waiting with sentinel order_id='0' returns False, not True."""
    om = load_venue_modules("212trading", "order_manager")
    # 1. 212trading/order_manager: waiting with sentinel order_id="0" returns False, not True
    monkeypatch.setattr(om, "_marker_path", lambda ticker, test=False: str(tmp_path / f"marker_{ticker}.json"))
    monkeypatch.setattr(om, "get_price_usd", lambda sym: 100.0)
    marker_file = tmp_path / "marker_NVDA_US_EQ.json"
    marker_file.write_text(json.dumps({
        "version": 2,
        "ticker": "NVDA_US_EQ",
        "qty": 1.0,
        "limit_price": 100.0,
        "lifecycle": "submit_pending",
        "order_id": "0",
        "lookup_misses": 0,
        "next_retry_at": 9999999999.0,
    }), encoding="utf-8")
    dummy_client = SimpleNamespace(
        list_active_orders=lambda: [],
        get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": 0.0}],
        get_order_status=lambda oid: (_ for _ in ()).throw(AssertionError(f"should not poll sentinel oid {oid}")),
    )
    res_wait = om.place_order_with_retry(
        dummy_client, "NVDA_US_EQ", 1.0, 100.0, "DAY", dry_run=False, write_marker=True
    )
    assert res_wait is False
