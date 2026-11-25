"""Tests for Kraken transfer credentials, KrakenClient Funding v1 endpoints, and rebalance_kraken_to_hl."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import credentials
from kraken.kraken_client import KrakenClient, _WRITE_METHODS
from verify_tools import rebalance_kraken_to_hl as reb

_WALLET = "0xBf48fd0c624c1ab06FDc87F919C0A9108bD90233"
_OTHER_WALLET = "0x1111111111111111111111111111111111111111"
_ARB_METHOD_ID = "5fe7fe84-0626-4828-8d7f-c0a26b435003"
_ARB_NETWORK_ID = "bc7562cf-1e51-4308-b52b-d062aaa6aa3b"
_OTHER_NETWORK_ID = "b81dfccc-4c32-4272-b4cd-26e1ac5ee3a5"
_ADDR_ID = "ABDCNZB-O5YPH-VHIPLQ"


def _make_cfg(**overrides) -> reb.RebalanceConfig:
    base = dict(
        withdraw_asset="USDC",
        withdraw_method="Arbitrum One Network - Native",
        arb_rpc_url="https://arb1.arbitrum.io/rpc",
        hl_info_url="https://api.hyperliquid.xyz/info",
        arb_usdc_contract="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        min_net_usdc=6.0,
        min_arb_eth_gas=0.0001,
        poll_interval_sec=0.01,
        poll_timeout_sec=1.0,
        target_address=_WALLET,
        configured_withdraw_key=None,
    )
    base.update(overrides)
    return reb.RebalanceConfig(**base)


def test_kraken_transfer_credential_profile_is_isolated_from_bot():
    pair = credentials.kraken_credentials(
        "transfer",
        values={
            "KRAKEN_API_KEY_TRANSFER": "tk",
            "KRAKEN_API_SECRET_TRANSFER": "ts",
            "KRAKEN_API_KEY_BOT": "bk",
            "KRAKEN_API_SECRET_BOT": "bs",
        },
    )
    assert pair.key == "tk"
    assert pair.secret == "ts"
    assert pair.profile == "Kraken TRANSFER"

    with pytest.raises(credentials.CredentialProfileMissingError):
        credentials.kraken_credentials(
            "transfer",
            values={"KRAKEN_API_KEY_BOT": "bk", "KRAKEN_API_SECRET_BOT": "bs"},
        )

    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.kraken_credentials("transfer", values={"KRAKEN_API_KEY_TRANSFER": "only_key"})


def test_kraken_client_funding_v1_non_retry_and_list_parsing():
    assert "Withdraw" in _WRITE_METHODS
    assert "WithdrawCancel" in _WRITE_METHODS

    cli = KrakenClient("key", "c2VjcmV0")
    list_body = b'{"error": [], "result": [{"method": "Arbitrum One Network - Native", "asset": "USDC"}]}'
    assert cli._parse(200, list_body, allow_list=False) == {}
    assert cli._parse(200, list_body, allow_list=True) == [
        {"method": "Arbitrum One Network - Native", "asset": "USDC"}
    ]

    addr_body = (
        b'{"addresses":[{"address_id":"ABDCNZB-O5YPH-VHIPLQ",'
        b'"scope":{"network_id":"bc7562cf-1e51-4308-b52b-d062aaa6aa3b"},'
        b'"address_details":{"crypto":{"address":"0xbf48fd0c624c1ab06fdc87f919c0a9108bd90233"}},'
        b'"name":"hyperliquid","verified":true}]}'
    )
    with patch("kraken.kraken_client.http_request", return_value=(200, addr_body)) as mock_req:
        addrs = cli.funding_addresses()
        assert len(addrs) == 1
        assert addrs[0]["name"] == "hyperliquid"
        assert mock_req.call_count == 1
        assert "API-Nonce" in mock_req.call_args.kwargs["headers"]
        assert "API-Sign" in mock_req.call_args.kwargs["headers"]

    # Funding v1 POST withdrawal must NEVER auto-retry on network/5xx error (max_attempts == 1)
    # and must serialize the official nested scope/amount/fee JSON payload.
    with patch("kraken.kraken_client.http_request", return_value=(0, b"")) as mock_req:
        with pytest.raises(Exception):
            cli.funding_withdraw("USDC", _ARB_METHOD_ID, _ADDR_ID, 11.0, fee_included=True, withdrawal_fee_token="tok")
        assert mock_req.call_count == 1
        sent_body = mock_req.call_args.kwargs["body"].decode("utf-8")
        assert '"scope":{"method_id":"' + _ARB_METHOD_ID + '"}' in sent_body
        assert '"address_id":"' + _ADDR_ID + '"' in sent_body
        assert '"asset":{"class":"currency","name":"USDC"}' in sent_body
        assert '"quoted_fee":{"token":"tok"}' in sent_body
        assert '"fee_included":true' in sent_body


def test_resolve_verified_withdraw_address_matches_arb_network_and_rejects_mismatches():
    cfg = _make_cfg()
    method_row = {
        "asset": "USDC",
        "method": "Arbitrum One Network - Native",
        "method_id": _ARB_METHOD_ID,
        "network_id": _ARB_NETWORK_ID,
    }

    # Rejects wrong network_id (e.g. HyperEVM address saved under "HL")
    with pytest.raises(ValueError, match="does not match"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "address_id": "AB44EMJ-AADVH-NYVBPJ",
                    "name": "HL",
                    "scope": {"network_id": _OTHER_NETWORK_ID},
                    "address_details": {"crypto": {"address": _WALLET}},
                    "verified": True,
                }
            ],
            cfg,
            method_row=method_row,
        )

    # Rejects unverified address
    with pytest.raises(ValueError, match="NOT email-verified"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "address_id": _ADDR_ID,
                    "name": "hyperliquid",
                    "scope": {"network_id": _ARB_NETWORK_ID},
                    "address_details": {"crypto": {"address": _WALLET}},
                    "verified": False,
                }
            ],
            cfg,
            method_row=method_row,
        )

    # Rejects wrong destination wallet
    with pytest.raises(ValueError, match="expected Hyperliquid wallet"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "address_id": _ADDR_ID,
                    "name": "other",
                    "scope": {"network_id": _ARB_NETWORK_ID},
                    "address_details": {"crypto": {"address": _OTHER_WALLET}},
                    "verified": True,
                }
            ],
            cfg,
            method_row=method_row,
        )

    # Auto-selects the exact Arbitrum One address ("hyperliquid") while ignoring other-network "HL"
    chosen = reb.resolve_verified_withdraw_address(
        [
            {
                "address_id": "AB44EMJ-AADVH-NYVBPJ",
                "name": "HL",
                "scope": {"network_id": _OTHER_NETWORK_ID},
                "address_details": {"crypto": {"address": _WALLET.lower()}},
                "verified": True,
            },
            {
                "address_id": _ADDR_ID,
                "name": "hyperliquid",
                "scope": {"network_id": _ARB_NETWORK_ID},
                "address_details": {"crypto": {"address": _WALLET.lower()}},
                "verified": True,
            },
        ],
        cfg,
        method_row=method_row,
    )
    assert chosen["name"] == "hyperliquid"
    assert chosen["address_id"] == _ADDR_ID


_VALID_TX_HASH_1 = "0x" + "12" * 32
_VALID_TX_HASH_42 = "0x" + "42" * 32


def _usdc_receipt_resp(cfg: reb.RebalanceConfig, net_usdc: float = 10.0) -> dict:
    to_topic = "0x" + "0" * 24 + _WALLET[2:].lower()
    usdc_hex = "0x" + format(round(net_usdc * 1_000_000), "064x")
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "status": "0x1",
            "blockNumber": "0x10",
            "logs": [
                {
                    "address": cfg.arb_usdc_contract,
                    "topics": [reb._ERC20_TRANSFER_TOPIC, "0x" + "0" * 64, to_topic],
                    "data": usdc_hex,
                }
            ],
        },
    }


def _mock_client(*, zusd: float = 6400.0, usdc: float = 20.0, fee: float = 1.0) -> MagicMock:
    cli = MagicMock(spec=KrakenClient)
    cli.balance.return_value = {"ZUSD": str(zusd), "USDC": str(usdc)}
    cli.withdraw_methods.return_value = [
        {
            "asset": "USDC",
            "method": "Arbitrum One Network - Native",
            "method_id": _ARB_METHOD_ID,
            "network_id": _ARB_NETWORK_ID,
            "network": "Arbitrum One (USDC)",
            "minimum": "4.00",
            "fee": {"asset": "USDC", "fee": str(fee)},
            "limits": [
                {"limit_type": "amount", "limits": {"86400": {"remaining": "500000.0"}}}
            ],
        }
    ]
    cli.funding_addresses.return_value = [
        {
            "address_id": _ADDR_ID,
            "name": "hyperliquid",
            "scope": {"network_id": _ARB_NETWORK_ID},
            "address_details": {"crypto": {"address": _WALLET.lower()}},
            "verified": True,
        },
        {
            "address_id": "AB44EMJ-AADVH-NYVBPJ",
            "name": "HL",
            "scope": {"network_id": _OTHER_NETWORK_ID},
            "address_details": {"crypto": {"address": _WALLET.lower()}},
            "verified": True,
        },
    ]

    def _quote(method_id, amount, *, fee_included=True):
        net = amount - fee if fee_included else amount
        gross = amount if fee_included else amount + fee
        return {
            "fee": {"asset": {"name": "USDC"}, "amount": f"{fee:.8f}"},
            "gross_amount": {"asset": {"name": "USDC"}, "amount": f"{gross:.8f}"},
            "net_amount": {"asset": {"name": "USDC"}, "amount": f"{net:.8f}"},
            "withdrawal_fee_token": "FEE-TOK-123",
        }

    cli.funding_fee_quote.side_effect = _quote
    cli.funding_withdraw.return_value = {"withdrawal_id": "WD-TEST-123"}
    cli.funding_withdrawals.return_value = [
        {"withdrawal_id": "WD-TEST-123", "status": "success", "onchain_transaction": _VALID_TX_HASH_1}
    ]
    return cli


def test_plan_and_run_dry_run_and_min_net_guard():
    cfg = _make_cfg()
    cli = _mock_client(zusd=6400.0, usdc=20.0, fee=1.0)

    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.0012, "usdc": 0.0}),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        res = reb.plan_and_run_transfer(cli, cfg, amount_usdc=11.0, execute=False)
        assert res["mode"] == "dry_run"
        assert res["plan"]["net_arbitrum_usdc"] == pytest.approx(10.0)
        assert res["plan"]["withdraw_name"] == "hyperliquid"
        assert res["plan"]["address_id"] == _ADDR_ID
        assert res["plan"]["method_id"] == _ARB_METHOD_ID
        cli.funding_withdraw.assert_not_called()
        cli.add_order.assert_not_called()

        with pytest.raises(ValueError, match="KRAKEN_HL_MIN_NET_USDC"):
            reb.plan_and_run_transfer(cli, cfg, amount_usdc=5.5, execute=False)


def test_execute_refuses_shortfall_without_convert_flag_and_executes_when_funded():
    cfg = _make_cfg()
    cli_short = _mock_client(zusd=6400.0, usdc=1.5, fee=1.0)

    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.0012, "usdc": 0.0}),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        with pytest.raises(ValueError, match="Kraken UI Convert"):
            reb.plan_and_run_transfer(cli_short, cfg, amount_usdc=11.0, execute=True, convert_usd=False)
        cli_short.funding_withdraw.assert_not_called()

    cli_funded = _mock_client(zusd=6400.0, usdc=25.0, fee=1.0)
    arb_seq = [{"eth": 0.0012, "usdc": 0.0}, {"eth": 0.0012, "usdc": 10.0}]
    with (
        patch.object(reb, "fetch_arbitrum_balances", side_effect=arb_seq),
        patch.object(reb, "fetch_json_post", return_value=_usdc_receipt_resp(cfg, 10.0)),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
        patch.object(reb.alert, "notify") as mock_notify,
    ):
        res = reb.plan_and_run_transfer(
            cli_funded, cfg, amount_usdc=11.0, execute=True, wait_arb=True, send_notify=True
        )
        assert res["mode"] == "executed"
        assert res["withdrawal_id"] == "WD-TEST-123"
        assert res["arbitrum_arrival"]["arrived"] is True
        assert res["arbitrum_arrival"]["arb_usdc_delta"] == pytest.approx(10.0)
        cli_funded.funding_withdraw.assert_called_once_with(
            "USDC", _ARB_METHOD_ID, _ADDR_ID, 11.0, fee_included=True, withdrawal_fee_token="FEE-TOK-123", max_fee=1.25
        )
        mock_notify.assert_called_once()
        assert mock_notify.call_args.kwargs == {"source": "rebalance_kraken_to_hl", "symbol": "HYPEUSDC"}


def test_wait_for_arbitrum_arrival_requires_tx_confirmation_not_independent_balance_jump():
    """Audit #4: Pending Kraken withdrawal with no txid + independent +10 USDC balance jump MUST NOT
    set arrived=True or trigger arrival notification; confirmation requires the withdrawal's transaction."""
    cfg = _make_cfg(poll_interval_sec=0.005, poll_timeout_sec=0.02)
    cli = _mock_client(zusd=6400.0, usdc=25.0, fee=1.0)
    cli.funding_withdrawals.return_value = [
        {"withdrawal_id": "WD-PEND-1", "status": "pending", "onchain_transaction": None}
    ]
    cli.funding_withdraw.return_value = {"withdrawal_id": "WD-PEND-1"}

    # Balance independently jumps from 100 -> 110 USDC while Kraken withdrawal is still pending without txid
    arb_seq = [{"eth": 0.0012, "usdc": 100.0}] + [{"eth": 0.0012, "usdc": 110.0}] * 10
    with (
        patch.object(reb, "fetch_arbitrum_balances", side_effect=arb_seq),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
        patch.object(reb.alert, "notify") as mock_notify,
    ):
        res = reb.plan_and_run_transfer(
            cli, cfg, amount_usdc=11.0, execute=True, wait_arb=True, send_notify=True
        )
    assert res["withdrawal_id"] == "WD-PEND-1"
    assert res["status"] == "withdrawal_accepted_monitoring_incomplete"
    assert res["arbitrum_arrival"]["arrived"] is False
    assert res["arbitrum_arrival"]["stages"] == {
        "withdrawal_accepted": True,
        "transaction_confirmed": False,
        "balance_observed": True,
    }
    assert res["arbitrum_arrival"]["arb_usdc_delta"] == pytest.approx(10.0)
    mock_notify.assert_not_called()

    # With a 66-char EVM tx hash, arrival confirms ONLY once eth_getTransactionReceipt confirms the USDC Transfer log
    tx_hash = "0x" + "ab" * 32
    cli.funding_withdrawals.return_value = [
        {"withdrawal_id": "WD-EVM-1", "status": "success", "onchain_transaction": tx_hash}
    ]
    to_topic = "0x" + "0" * 24 + _WALLET[2:].lower()
    usdc_10_hex = "0x" + format(10_000_000, "064x")
    receipt_seq = [
        {"jsonrpc": "2.0", "id": 3, "result": None},  # poll 1: tx not mined yet
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "status": "0x1",
                "blockNumber": "0x10",
                "logs": [
                    {
                        "address": cfg.arb_usdc_contract,
                        "topics": [reb._ERC20_TRANSFER_TOPIC, "0x" + "0" * 64, to_topic],
                        "data": usdc_10_hex,
                    }
                ],
            },
        },
    ]
    cfg_long = _make_cfg(poll_interval_sec=0.005, poll_timeout_sec=0.5)
    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.0012, "usdc": 110.0}),
        patch.object(reb, "fetch_json_post", side_effect=receipt_seq),
    ):
        arr = reb.wait_for_arbitrum_arrival(cli, cfg_long, "WD-EVM-1", 100.0, 10.0)
    assert arr["arrived"] is True
    assert arr["arb_txid"] == tx_hash
    assert arr["tx_usdc_amount"] == pytest.approx(10.0)
    assert arr["stages"] == {
        "withdrawal_accepted": True,
        "transaction_confirmed": True,
        "balance_observed": True,
    }

    # Receipt Transfer log with non-zero upper bytes in topics[2] (malformed ABI topic) MUST NOT confirm
    bad_topic_receipt = _usdc_receipt_resp(cfg, 10.0)
    bad_topic_receipt["result"]["logs"][0]["topics"][2] = "0x" + "ff" * 12 + _WALLET[2:].lower()
    with patch.object(reb, "fetch_json_post", return_value=bad_topic_receipt):
        chk_bad_topic = reb.verify_arbitrum_usdc_transfer_tx(cfg, tx_hash, target_address=_WALLET, min_usdc=9.99)
    assert chk_bad_topic["confirmed"] is False and chk_bad_topic["received_usdc"] == 0.0


def test_rpc_error_after_withdrawal_preserves_id_and_cli_resume_avoids_duplicate_withdraw(capsys):
    """P1 Rebalance Resumption: RPC failure after Kraken accepts withdrawal MUST return withdrawal_id
    in monitoring_incomplete state; resuming with --withdrawal-id MUST NEVER call funding_withdraw twice."""
    import json

    cfg = _make_cfg(poll_interval_sec=0.005, poll_timeout_sec=0.2)
    cli = _mock_client(zusd=6400.0, usdc=25.0, fee=1.0)
    cli.funding_withdraw.return_value = {"withdrawal_id": "WD-RESUME-42"}
    cli.funding_withdrawals.return_value = [
        {"withdrawal_id": "WD-RESUME-42", "status": "success", "onchain_transaction": _VALID_TX_HASH_42, "net_amount": {"amount": "10.0"}}
    ]

    # 1. Execute: snapshot RPC succeeds, withdrawal accepted, then first monitoring RPC raises
    arb_calls = [
        {"eth": 0.0012, "usdc": 100.0},
        RuntimeError("Arbitrum RPC timeout"),
    ]

    def _arb_side_effect(*_a, **_kw):
        nxt = arb_calls.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    fake_creds = MagicMock(key="tk", secret="ts")
    with (
        patch.object(reb, "load_rebalance_config", return_value=cfg),
        patch.object(reb, "kraken_credentials", return_value=fake_creds),
        patch.object(reb.KrakenProvider, "build_client", return_value=cli),
        patch.object(reb, "fetch_arbitrum_balances", side_effect=_arb_side_effect),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        rc1 = reb.main(["--amount", "11", "--execute", "--json"])
    cap1 = capsys.readouterr()
    assert rc1 == 2
    payload1 = json.loads(cap1.out)
    assert payload1["ok"] is False
    assert payload1["withdrawal_id"] == "WD-RESUME-42"
    assert payload1["status"] == "withdrawal_accepted_monitoring_incomplete"
    assert payload1["stages"]["withdrawal_accepted"] is True
    assert payload1["stages"]["transaction_confirmed"] is False
    assert cli.funding_withdraw.call_count == 1

    # 2. Resume monitoring by --withdrawal-id: verifies arrival WITHOUT submitting another withdrawal
    with (
        patch.object(reb, "load_rebalance_config", return_value=cfg),
        patch.object(reb, "kraken_credentials", return_value=fake_creds),
        patch.object(reb.KrakenProvider, "build_client", return_value=cli),
        patch.object(reb, "fetch_arbitrum_balances", side_effect=[{"eth": 0.0012, "usdc": 100.0}, {"eth": 0.0012, "usdc": 110.0}]),
        patch.object(reb, "fetch_json_post", return_value=_usdc_receipt_resp(cfg, 10.0)),
        patch.object(reb.alert, "notify") as mock_notify,
    ):
        rc2 = reb.main(["--withdrawal-id", "WD-RESUME-42", "--notify", "--json"])
    cap2 = capsys.readouterr()
    assert rc2 == 0
    payload2 = json.loads(cap2.out)
    assert payload2["ok"] is True
    assert payload2["mode"] == "resumed"
    assert payload2["withdrawal_id"] == "WD-RESUME-42"
    assert payload2["status"] == "arrived"
    assert payload2["arbitrum_arrival"]["arrived"] is True
    assert cli.funding_withdraw.call_count == 1  # NEVER called a second time!
    mock_notify.assert_called_once()
