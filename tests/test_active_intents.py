import json
import os

from active_intents import build_active_intent_index, format_summary


def _write(path, payload, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("\n".join(json.dumps(row) for row in payload) + "\n"
            if lines else json.dumps(payload))
    path.write_text(text, encoding="utf-8")


def test_normalizes_multiple_owners_without_writing(tmp_path):
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    t212 = tmp_path / "212trading/.state_NVDA_US_EQ.json"
    rtrade = tmp_path / "cachedb/rtrade_pairs.json"
    _write(outbox, [{
        "intent_id": "out-1", "provider_name": "Binance", "symbol": "TAOUSDC",
        "side": "BUY", "qty": 2, "lifecycle": "submit_pending",
    }], lines=True)
    _write(t212, {
        "pending_submit": {"intent_id": "t-1", "side": "SELL", "qty": 1,
                           "limit": 150, "submission_outcome": "unknown"},
    })
    _write(rtrade, {"pairs": {"pair-1": {
        "symbol": "BTCUSDC", "terminal": False,
        "intents": {"limit:BUY": {
            "intent_id": "r-1", "side": "BUY", "requested_qty": 0.1,
            "client_order_id": "CID-R", "order_id": "77",
        }},
    }}})
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (outbox, t212, rtrade)}

    result = build_active_intent_index(str(tmp_path))

    assert result["read_only"] is True
    assert {row["intent_id"] for row in result["intents"]} == {"out-1", "t-1", "r-1"}
    assert next(row for row in result["intents"] if row["intent_id"] == "t-1")["status"] == "unknown"
    assert next(row for row in result["intents"] if row["intent_id"] == "r-1")["symbol"] == "BTCUSDC"
    for path, snapshot in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot


def test_excludes_terminal_intents_and_reports_corrupt_sources(tmp_path):
    _write(tmp_path / "cachedb/rtrade_pairs.json", {"pairs": {
        "done": {"terminal": True, "intents": {"x": {
            "intent_id": "done", "side": "SELL", "qty": 1,
        }}},
    }})
    corrupt = tmp_path / "cachedb/assetguardian_state.json"
    corrupt.write_text("{broken", encoding="utf-8")

    result = build_active_intent_index(str(tmp_path))

    assert result["intents"] == []
    assert result["errors"][0]["source"] == "cachedb/assetguardian_state.json"


def test_summary_calculates_pending_notional_and_net_exposures(tmp_path):
    outbox = tmp_path / "cachedb/order_retry_queue.jsonl"
    _write(outbox, [
        {"intent_id": "buy-1", "symbol": "BTCUSDC", "side": "BUY", "qty": 0.5, "requested_price": 60000.0, "lifecycle": "pending"},
        {"intent_id": "sell-1", "symbol": "ETHUSDC", "side": "SELL", "qty": 2.0, "executed_qty": 0.5, "requested_price": 3000.0, "lifecycle": "pending"},
    ], lines=True)

    kraken = tmp_path / "kraken/.state_BTCUSDC.json"
    _write(kraken, {
        "qty": 1.5, "cost": 90000.0, "entry_price": 60000.0, "entry_ts": 1700000000.0,
    })

    result = build_active_intent_index(str(tmp_path))

    summary = result["summary"]
    notional = summary["pending_notional"]
    assert notional["total_buy"] == 30000.0
    assert notional["total_sell"] == 4500.0
    assert notional["by_symbol"]["BTCUSDC"]["buy"] == 30000.0
    assert notional["by_symbol"]["ETHUSDC"]["sell"] == 4500.0

    exposures = summary["net_exposures"]
    assert exposures["total_cost"] == 90000.0
    assert len(exposures["positions"]) == 1
    assert exposures["positions"][0]["symbol"] == "BTCUSDC"
    assert exposures["positions"][0]["qty"] == 1.5

    formatted = format_summary(result)
    assert "Pending BUY notional:  $30000.00" in formatted
    assert "Pending SELL notional: $4500.00" in formatted
    assert "Total cost basis:      $90000.00" in formatted
    assert "BTCUSDC: 1.5 ($90000.00) [kraken]" in formatted


def test_filters_backup_files_and_recognizes_id_fallback(tmp_path):
    # Backup file should be ignored
    bak = tmp_path / "hyperliquid/.state_HYPE.pre_fee_reconcile.json"
    _write(bak, {"qty": 999.0, "cost": 999.0})

    # Legitimate state with order having only id (e.g. T212 format)
    t212 = tmp_path / "212trading/.state_TEST_US_EQ.json"
    _write(t212, {
        "qty": 10.0,
        "cost_usd": 500.0,
        "entry_price": 50.0,
        "orders": [
            {"id": "t212-order-99", "side": "SELL", "qty": 10.0, "limit": 60.0}
        ]
    })

    result = build_active_intent_index(str(tmp_path))
    assert not any(exp["symbol"] == "HYPE.pre_fee_reconcile" for exp in result["exposures"])
    assert any(it["intent_id"] == "t212-order-99" and it["requested_price"] == 60.0 for it in result["intents"])
    assert any(exp["symbol"] == "TEST_US_EQ" and exp["qty"] == 10.0 for exp in result["exposures"])

