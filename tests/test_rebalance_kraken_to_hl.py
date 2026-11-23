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
        {"withdrawal_id": "WD-TEST-123", "status": "success", "onchain_transaction": "0xarbtx123"}
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
