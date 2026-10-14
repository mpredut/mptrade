import time
import pytest
from unittest.mock import patch, MagicMock

import monitororder
from binance_api import order_id_context as rc
from providers.strategy_executor import OrderStatus


@pytest.fixture(autouse=True)
def _reset_monitororder():
    monitororder.reset_monitororder_state_for_test()
    yield
    monitororder.reset_monitororder_state_for_test()


class TestMonitororderTTLResolution:
    def test_module_default_ttl(self):
        # rtrade default: 300s
        assert monitororder.resolve_effective_order_ttl("1", "RT_rtrade-TAOUSDC_1_abc", owner="rtrade") == 300.0
        # tradeall default: 7200s
        assert monitororder.resolve_effective_order_ttl("2", "TA_OR_abc_0", owner="tradeall") == 7200.0
        # manual default: 86400s
        assert monitororder.resolve_effective_order_ttl("3", "MN_abc_0", owner="manual") == 86400.0
        # fallback default: 14400s
        assert monitororder.resolve_effective_order_ttl("4", "UNKNOWN_0", owner="unknown") == 14400.0

    def test_custom_ttl_from_map(self):
        custom_map = {
            "100": 600.0,
            "TA_OR_custom_0": 1800.0,
        }
        # By order ID
        assert monitororder.resolve_effective_order_ttl("100", "TA_OR_abc_0", owner="tradeall", custom_ttl_map=custom_map) == 600.0
        # By client order ID
        assert monitororder.resolve_effective_order_ttl("101", "TA_OR_custom_0", owner="tradeall", custom_ttl_map=custom_map) == 1800.0

    def test_hard_cap_3_days_enforced_on_custom_ttl(self):
        # 10 days custom TTL must be clamped to MAX_GLOBAL_ORDER_TTL_SEC (259200s / 3 days)
        ten_days = 10 * 86400.0
        custom_map = {"999": ten_days}
        effective = monitororder.resolve_effective_order_ttl("999", "MN_abc_0", owner="manual", custom_ttl_map=custom_map)
        assert effective == monitororder.MAX_GLOBAL_ORDER_TTL_SEC
        assert effective == 259200.0


class TestMonitororderTTLExpiration:
    def test_expired_order_is_canceled(self):
        now = time.time()
        # Order placed 400 seconds ago (exceeds rtrade 300s TTL)
        open_orders = {
            "711457409": {
                "price": 277.4,
                "quantity": 1.8,
                "timestamp": now - 400.0,
                "clientOrderId": "RT_rtrade-TAOUSDC_1_abc",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=278.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
            patch.object(monitororder, "_notify_order_expired") as mock_notify,
        ):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_cancel.assert_called_once_with("TAOUSDC", "711457409")
            mock_notify.assert_called_once()

    def test_non_expired_order_is_not_canceled(self):
        now = time.time()
        # Order placed 100 seconds ago (below rtrade 300s TTL)
        open_orders = {
            "711457410": {
                "price": 277.4,
                "quantity": 1.8,
                "timestamp": now - 100.0,
                "clientOrderId": "RT_rtrade-TAOUSDC_1_abc",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=278.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
        ):
            monitororder.monitor_open_orders_by_type("TAOUSDC", "BUY")
            mock_cancel.assert_not_called()

    def test_hard_cap_cancels_even_if_custom_ttl_larger(self):
        now = time.time()
        # Order placed 3.5 days ago (exceeds 3 days hard cap)
        open_orders = {
            "10544265098": {
                "price": 83000.0,
                "quantity": 0.01,
                "timestamp": now - (3.5 * 86400.0),
                "clientOrderId": "MN_OR_manual_0",
            }
        }
        with (
            patch.object(monitororder.api, "get_open_orders", return_value=open_orders),
            patch.object(monitororder.api, "get_current_price", return_value=83500.0),
            patch.object(monitororder.mkt, "cancel_order") as mock_cancel,
            patch.object(monitororder, "_notify_order_expired") as mock_notify,
        ):
            monitororder.monitor_open_orders_by_type("BTCUSDC", "BUY")
            mock_cancel.assert_called_once_with("BTCUSDC", "10544265098")
            mock_notify.assert_called_once()


class TestOrderRetryTTL:
    def test_enqueue_persists_ttl_sec(self, tmp_path):
        import order_retry as oq
        with patch.object(oq, "QUEUE_FILE", str(tmp_path / "queue.jsonl")), \
             patch.object(oq, "LOCK_FILE", str(tmp_path / "queue.lock")):
            rec_id = oq.enqueue(
                "BTCUSDC", "BUY", 0.01,
                requested_price=80000.0, ref_price=80000.0,
                ttl_sec=1200.0
            )
            records = oq.load_all()
            found = [r for r in records if r.get("id") == rec_id]
            assert len(found) == 1
            assert found[0].get("ttl_sec") == 1200.0
            assert found[0].get("expires_at_ts") is not None

    def test_get_accepted_order_ttl_map(self, tmp_path):
        import order_retry as oq
        with patch.object(oq, "QUEUE_FILE", str(tmp_path / "queue.jsonl")), \
             patch.object(oq, "LOCK_FILE", str(tmp_path / "queue.lock")):
            oq.enqueue(
                "BTCUSDC", "BUY", 0.01,
                place_kwargs={"client_order_id": "TA_OR_test_1", "ttl_sec": 750.0},
                requested_price=80000.0, ref_price=80000.0,
            )
            ttl_map = oq.get_accepted_order_ttl_map()
            assert ttl_map.get("TA_OR_test_1") == 750.0
